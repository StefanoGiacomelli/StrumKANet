"""Descriptor cache for the recording-disjoint re-evaluation (``main_recording_disjoint.py``).

Every excerpt is analysed ONCE with the stored pipeline (``PopMusicDataset`` + ``rhythm_features`` with its unchanged defaults),
and the fixed-size descriptors are stored as float32 ``.npy`` arrays in dataset order, so that training on a CPU does not pay
the on-the-fly feature extraction of the 2025 runs (about 4 minutes per epoch). The cached arrays are exactly the tensors that
``PopMusicDataset.__getitem__`` returns, so models see the same inputs as in the stored protocol.
"""
from __future__ import annotations

import json
import os
import time
from functools import partial
from typing import Dict, List, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from data import PopMusicDataset, custom_collate_fn
from globals import FEATURES_MAP

# Descriptors with a fixed shape per excerpt (cached densely). Variable-length ones (beats, ticks, bpm_estimates), the raw
# waveform and the tempogram (384 x 938 floats per excerpt, about 8 GB for the corpus) are not cached.
FIXED_SIZE_FEATURES = ("rp", "rh", "tssd", "trh", "onset_strength", "tempogram_ratio")


def _identity(batch):
    return batch


def cache_file(cache_dir: str, feature: str) -> str:
    return os.path.join(cache_dir, f"{feature}.npy")


def index_dataset(csv_path: str, audio_folder: str, segment_duration: float = 10.0, label_type: str = "pattern") -> PopMusicDataset:
    """Dataset without preprocessing: provides excerpt ids, recording ids and labels (cheap: only audio metadata is read)."""
    return PopMusicDataset(csv_path=csv_path, audio_folder=audio_folder, segment_duration=segment_duration,
                           features=None, label_type=label_type, preprocessor=None)


def missing_audio(csv_path: str, audio_folder: str) -> List[str]:
    """Identifiers of the 'downloaded' rows of the CSV whose WAV file is not in ``audio_folder``.

    ``PopMusicDataset`` only prints a warning and skips such rows, which would silently shrink the corpus (for example when an
    archive tool mis-decodes accented file names such as ``Modà-Come_un_pittore.wav``: the Zenodo archive stores UTF-8 names without
    the zip UTF-8 flag and Python's ``zipfile`` reads them as CP437).
    """
    import csv
    missing = []
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["downloaded"].lower() != "true":
                continue
            file_id = f"{row['performer'].strip().replace(' ', '_')}-{row['title'].strip().replace(' ', '_')}"
            if not os.path.isfile(os.path.join(audio_folder, f"{file_id}.wav")):
                missing.append(file_id)
    return missing


GROUPINGS = ("recording", "performer", "excerpt")


def grouping_keys(dataset: PopMusicDataset, grouping: str) -> List[str]:
    """Group of every excerpt, in dataset order, for the K-fold partitions of ``main_recording_disjoint.py``.

    * ``recording``: the source recording; CSV rows that share a YouTube URL are one recording (the released CSV repeats three
      sources: two rows are exact duplicates of one file, and one video appears under two titles), so identical audio never falls
      into two folds.
    * ``performer``: all recordings of one performer (case and spacing ignored), a stricter control.
    * ``excerpt``: one group per distinct excerpt (excerpts of one recording are spread over the folds), a looser control.
    """
    if grouping not in GROUPINGS:
        raise ValueError(f"grouping must be one of {GROUPINGS}, got {grouping!r}")
    first_with_url: Dict[str, str] = {}
    recording_key = []
    for file_id, row, _, _ in dataset.entries:
        url = (row.get("yt_url") or "").strip()
        recording_key.append(first_with_url.setdefault(url, file_id) if url else file_id)
    keys = []
    for entry_idx, segment_idx in dataset.segment_map:
        if grouping == "recording":
            keys.append(recording_key[entry_idx])
        elif grouping == "performer":
            keys.append(" ".join(dataset.entries[entry_idx][1]["performer"].casefold().split()))
        else:
            keys.append(f"{recording_key[entry_idx]}|{segment_idx}")
    return keys


def build_cache(csv_path: str, audio_folder: str, features: Sequence[str], cache_dir: str, segment_duration: float = 10.0,
                label_type: str = "pattern", num_workers: int = 0, force: bool = False) -> List[str]:
    """Compute and store the missing descriptors; return the list of descriptors that were (re)built."""
    from data_processing import rhythm_features        # heavy imports (librosa, optional rp_extract/essentia) only when building

    for feature in features:
        if feature not in FIXED_SIZE_FEATURES:
            raise ValueError(f"descriptor {feature!r} cannot be cached (fixed-size descriptors: {FIXED_SIZE_FEATURES})")
    os.makedirs(cache_dir, exist_ok=True)
    index = index_dataset(csv_path, audio_folder, segment_duration, label_type)
    ids = [index.excerpt_id(i) for i in range(len(index))]
    index_path = os.path.join(cache_dir, "index.json")
    if os.path.isfile(index_path):
        with open(index_path) as handle:
            stored = json.load(handle)
        if stored["ids"] != ids or stored["segment_duration"] != segment_duration:
            raise ValueError(f"{index_path} belongs to another dataset or segment duration; use another --cache-dir")
    else:
        with open(index_path, "w") as handle:
            json.dump({"segment_duration": segment_duration, "label_type": label_type, "ids": ids}, handle)
    missing = [f for f in features if force or not os.path.isfile(cache_file(cache_dir, f))]
    if not missing:
        return []
    extractors = sorted({FEATURES_MAP[f] for f in missing})
    analysed = PopMusicDataset(csv_path=csv_path, audio_folder=audio_folder, segment_duration=segment_duration, features=list(missing),
                               label_type=label_type, preprocessor=partial(rhythm_features, extractors=extractors))
    loader = DataLoader(analysed, batch_size=8, shuffle=False, num_workers=num_workers, collate_fn=_identity)
    arrays: Dict[str, List[np.ndarray]] = {f: [] for f in missing}
    started = time.time()
    for step, batch in enumerate(loader, 1):
        for item in batch:
            for feature in missing:
                arrays[feature].append(item["input"][feature].numpy())
        if step % 100 == 0:
            print(f"[cache] {len(arrays[missing[0]])}/{len(analysed)} excerpts analysed ({time.time() - started:.0f} s)", flush=True)
    for feature in missing:
        temporary = cache_file(cache_dir, feature) + ".partial.npy"             # written completely before it gets its final name
        np.save(temporary, np.stack(arrays[feature]).astype(np.float32))
        os.replace(temporary, cache_file(cache_dir, feature))
    return list(missing)


def load_index(cache_dir: str) -> Dict[str, object]:
    with open(os.path.join(cache_dir, "index.json")) as handle:
        return json.load(handle)


class CachedFeatureDataset(Dataset):
    """One cached descriptor, returned in the same structure as ``PopMusicDataset`` (``{'id', 'input': {feature: tensor}, 'label'}``)."""

    def __init__(self, cache_dir: str, feature: str, ids: Sequence[str], labels: Sequence):
        self.feature = feature
        self.array = np.load(cache_file(cache_dir, feature), mmap_mode="r")
        self.ids, self.labels = list(ids), list(labels)
        if len(self.array) != len(self.ids):
            raise ValueError(f"cache of {feature!r} has {len(self.array)} excerpts, the dataset has {len(self.ids)}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> Dict[str, object]:
        return {"id": self.ids[i], "input": {self.feature: torch.from_numpy(np.array(self.array[i]))}, "label": self.labels[i]}


class PartitionDataModule(pl.LightningDataModule):
    """DataModule over explicit index lists (the partition is computed outside and never re-drawn)."""

    def __init__(self, dataset: Dataset, train: Sequence[int], val: Sequence[int], test: Sequence[int], batch_size: int, num_workers: int = 0):
        super().__init__()
        self.train_set, self.val_set, self.test_set = (Subset(dataset, list(p)) for p in (train, val, test))
        self.batch_size, self.num_workers = batch_size, num_workers

    def train_dataloader(self):
        drop_last = len(self.train_set) % self.batch_size == 1          # BatchNorm cannot be trained on a final batch of one excerpt
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=custom_collate_fn,
                          drop_last=drop_last)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=custom_collate_fn)
