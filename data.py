import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl
from typing import Callable, List, Optional, Dict, Tuple
from globals import FEATURES_MAP


class PopMusicDataset(Dataset):
    """
    Dataset for loading and segmenting pop music audio files with optional preprocessing and feature extraction.

    Each audio file is divided into segments of fixed duration (default 10 seconds), excluding the first and last segments.
    Labels are assigned based on BPM or rhythmic pattern depending on the `label_type` parameter.
    Additional features can be extracted via a user-defined `preprocessor` function.

    Args:
        csv_path (str): Path to the metadata CSV file.
        audio_folder (str): Path to the directory containing audio files.
        segment_duration (float): Duration (in seconds) of each segment.
        features (Optional[List[str]]): List of feature names to extract from the preprocessed output.
        bpm_source (str): Column name in CSV to use for BPM-based labels.
        label_type (str): Type of label to use, either 'bpm' or 'pattern'.
        preprocessor (Optional[Callable]): A function that extracts features from raw audio.

    Raises:
        ValueError: If audio file sample rate is not 48kHz.
        KeyError: If feature keys or groups are missing or mismatched.
        TypeError: If a feature has an unsupported data type.
    """
    def __init__(self,
                 csv_path: str,
                 audio_folder: str,
                 segment_duration: float = 10.0,
                 features: Optional[List[str]] = None,
                 bpm_source: str = "bpm_marco",
                 label_type: str = "bpm",  # or "pattern"
                 preprocessor: Optional[Callable] = None):
        
        self.audio_folder = audio_folder
        self.segment_duration = segment_duration
        self.sr = 48000
        self.features = features
        self.bpm_source = bpm_source
        self.label_type = label_type
        self.preprocessor = preprocessor
        self.entries = []
        self.feature_source_map = FEATURES_MAP

        if self.preprocessor:
            if self.features is None:
                self.features = list(FEATURES_MAP.keys())  # use all features if not specified
            elif not isinstance(self.features, list):
                raise ValueError("`features` must be a list or None.")
        else:
            # No preprocessor --> no features (coherence)
            self.features = None
        
        # Read the CSV file
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')   
            for row in reader:
                # Check if audio file exist
                if row["downloaded"].lower() != "true":
                    continue
                # Format audio filename
                performer = row["performer"].strip().replace(" ", "_")
                title = row["title"].strip().replace(" ", "_")
                file_id = f"{performer}-{title}"
                path = os.path.join(self.audio_folder, f"{file_id}.wav")
                if not os.path.isfile(path):
                    print(f"Warning: Audio file {path} does not exist. Skipping entry.")
                    continue
                label = float(row[self.bpm_source]) if label_type == "bpm" else int(row["label"])
                
                self.entries.append((file_id, row, path, label))

        self.segment_len = int(segment_duration * self.sr)

        # Read audio files metadata and create segments
        self.segment_map = []
        for i, (file_id, row, path, label) in enumerate(self.entries):
            info = torchaudio.info(path)
            if info.sample_rate != self.sr:
                raise ValueError(f"Expected sample rate {self.sr}, but got {info.sample_rate} in {path}")
            tot_points = info.num_frames
            total_segments = tot_points // self.segment_len
            if total_segments > 2:
                for segment_idx in range(1, total_segments - 1):
                    self.segment_map.append((i, segment_idx))

    def __len__(self):
        """
        Returns the number of segments in the dataset.
        """
        return len(self.segment_map)

    def __getitem__(self, idx):
        """
        Returns a single data sample consisting of a waveform segment (optionally pre-processed) and its corresponding label.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'id' (str): A unique ID for the segment.
                - 'input' (Union[Tensor, Dict[str, Tensor]]): The waveform or preprocessed feature(s).
                - 'label' (float or int): The ground-truth label (BPM or rhythmic pattern).
        """
        # Segment retrieval
        entry_idx, segment_idx = self.segment_map[idx]
        file_id, _, path, label = self.entries[entry_idx]
        waveform, _ = torchaudio.load(path)
        start = segment_idx * self.segment_len
        end = start + self.segment_len
        segment = waveform[:, start:end]
        
        # Feature extraction
        result = self.preprocessor(segment) if self.preprocessor else segment

        def standardize_input_shape(x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 1:
                return x.unsqueeze(0)               # (Feats,) -> (1, Feats)
            elif x.ndim == 2:
                if x.shape[0] == 1:
                    return x                        # (1, Samples)
                else:
                    return x.unsqueeze(0)           # (Feats_dim1, Feats_dim2) -> (1, Feats_dim1, Feats_dim2)
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")

        # Feature selection
        if not isinstance(result, dict):
            # Raw waveform or flat tensor returned
            x = result
        else:
            # Expecting a dict with grouped features
            if self.features is None:
                raise ValueError("Feature list is None, but preprocessor returned a dict. Cannot index.")
            x = {}
            for k in self.features:
                group_name = self.feature_source_map.get(k, None)
                if group_name is None:
                    raise KeyError(f"Feature '{k}' is not mapped to any known group. Check feature_source_map.")
                group = result.get(group_name)
                if not isinstance(group, dict):
                    raise KeyError(f"Expected a 'dict' for feature group '{group_name}', but got: {type(group)}")
                inner_key = "bpm" if k in ["bpm_librosa", "bpm_essentia"] else k
                if inner_key not in group:
                    raise KeyError(f"Feature '{inner_key}' not found in group '{group_name}'.")
                feat = group[inner_key]

                # Apply 2D reshaping for rp_extract group
                if group_name == "rp_extract":
                    if k == "rp":
                        feat = np.reshape(feat, (24, 60), order='F')
                    elif k == "tssd":
                        feat = np.reshape(feat, (24, 7, 7), order='F')
                        feat_transposed = feat.transpose(2, 1, 0)                           # (Segments, Stats, Bark)
                        feat = np.concatenate([seg.T for seg in feat_transposed], axis=1)   # (Bark: 24, 7seg × 7metrics = 49)
                    elif k == "trh":
                        feat = np.reshape(feat, (60, 7), order='F')

                if isinstance(feat, np.ndarray):
                    feat_tensor = torch.from_numpy(feat).float()
                elif torch.is_tensor(feat):
                    feat_tensor = feat.float()
                else:
                    raise TypeError(f"Unsupported feature type for key {k}: {type(feat)}")
                x[k] = standardize_input_shape(feat_tensor)

        return {"id": f"{file_id}_seg{segment_idx}", "input": x, "label": label}


def pad_to_shape_centered(tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Pads a tensor with zeros to match the target shape. Padding is centered, split evenly on both sides.
    Supports tensors of arbitrary dimension.
    """
    current_shape = tensor.shape
    pad = []

    # Reverse order to apply padding from last to first dimension
    for curr, target in zip(reversed(current_shape), reversed(target_shape)):
        total_pad = max(target - curr, 0)
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        pad.extend([pad_left, pad_right])

    return F.pad(tensor, pad, "constant", 0)


def custom_collate_fn(batch):
    """
    Collate function to be used in DataLoader. It:
    - Pads all tensors in the batch to the same shape, centered.
    - Supports both raw tensors and dictionary-based feature inputs.
    - Assumes only one feature per sample is used if input is a dictionary.
    """
    ids = [item["id"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])

    # Handle feature dictionary input
    if isinstance(batch[0]["input"], dict):
        keys = list(batch[0]["input"].keys())
        if len(keys) != 1:
            raise ValueError("Only one feature key is supported per sample.")
        key = keys[0]

        tensors = [item["input"][key] for item in batch]
    else:
        # Handle raw tensor input (e.g., waveform)
        tensors = [item["input"] for item in batch]

    # Compute target shape
    max_shape = tuple(max(t.shape[dim] for t in tensors) for dim in range(tensors[0].ndim))

    # Pad and stack tensors
    padded = [pad_to_shape_centered(t, max_shape) for t in tensors]
    inputs = torch.stack(padded)

    return {"id": ids, "input": inputs, "label": labels}


class PopMusicDataModule(pl.LightningDataModule):
    def __init__(self,
                 csv_path: str,
                 audio_folder: str,
                 segment_duration: float = 10.0,
                 features: Optional[List[str]] = None,
                 bpm_source: str = "bpm_marco",
                 label_type: str = "bpm",
                 preprocessor: Optional[Callable] = None,
                 split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                 batch_size: int = 16,
                 num_workers: int = 4,
                 shuffle: bool = True):
        super().__init__()
        self.csv_path = csv_path
        self.audio_folder = audio_folder
        self.segment_duration = segment_duration
        self.features = features
        self.bpm_source = bpm_source
        self.label_type = label_type
        self.preprocessor = preprocessor
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def setup(self, stage: Optional[str] = None):
        dataset = PopMusicDataset(csv_path=self.csv_path,
                                  audio_folder=self.audio_folder,
                                  segment_duration=self.segment_duration,
                                  features=self.features,
                                  bpm_source=self.bpm_source,
                                  label_type=self.label_type,
                                  preprocessor=self.preprocessor)

        total = len(dataset)
        train_len = int(self.split_ratio[0] * total)
        val_len = int(self.split_ratio[1] * total)
        test_len = total - train_len - val_len
        self.train_set, self.val_set, self.test_set = random_split(dataset, [train_len, val_len, test_len])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, collate_fn=custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=custom_collate_fn, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=custom_collate_fn, persistent_workers=True)


# Test commands ------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from functools import partial
    from globals import CSV_PATH, AUDIO_FOLDER, BATCH_SIZE, NUM_WORKERS, SPLIT_RATIO, SHUFFLE
    from data_processing import rhythm_features

    # === Parameters ===
    SEGMENT_DURATION = 10.0          # in seconds
    FEATURES = None                 # or ['tempogram']
    BPM_SOURCE = "bpm_marco"        # or "bpm_tunebat", "bpm_songBPM"
    LABEL_TYPE = "pattern"          # or "bpm" or "pattern"

    preprocessor = None             # or partial(rhythm_features, extractors=['librosa'])

    # === Init DataModule ===
    data_module = PopMusicDataModule(csv_path=CSV_PATH,
                                     audio_folder=AUDIO_FOLDER,
                                     segment_duration=SEGMENT_DURATION,
                                     features=FEATURES,
                                     bpm_source=BPM_SOURCE,
                                     label_type=LABEL_TYPE,
                                     preprocessor=preprocessor,
                                     split_ratio=SPLIT_RATIO,
                                     batch_size=BATCH_SIZE,
                                     num_workers=NUM_WORKERS,
                                     shuffle=SHUFFLE)

    # === Setup (prepares splits) ===
    data_module.setup()

    # === Dataloaders ===
    dataloaders = {"train": data_module.train_dataloader(),
                   "val": data_module.val_dataloader(),
                   "test": data_module.test_dataloader()}

    # === Summary ===
    print("== DATALOADERs SUMMARY ==")
    for phase in ["train", "val", "test"]:
        loader = dataloaders[phase]
        num_batches = len(loader)
        print(f"{phase.capitalize()} set: {len(loader.dataset)} samples, {num_batches} batches (batch_size={BATCH_SIZE})")

    # === Iterate through Dataloader Batches ===
    for phase in ["train", "val", "test"]:
        print(f"\n== {phase.upper()} DATASET ==")
        loader = dataloaders[phase]

        for batch_idx, batch in enumerate(loader):
            print(f"Batch {batch_idx + 1}")
            print("IDs:", batch["id"])
            print("Inputs:")
            if isinstance(batch["input"], dict):
                for k, v in batch["input"].items():
                    print(f" - {k}: {v.shape}")
            else:
                print(" -", batch["input"].shape)
            print("Labels:", batch["label"].shape)
