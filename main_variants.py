"""Training-recipe and descriptor variants under the recording-disjoint protocol (new, optional; ``main_recording_disjoint.py`` is unchanged).

``main_recording_disjoint.py`` re-evaluates the stored configurations. This script keeps its protocol (same cached descriptors, same
class-stratified folds of whole recordings, same recording-disjoint validation part, early stopping with patience 20 and at most 50
epochs, batch size 32, Adam 5e-4, evaluation of the held-out fold with the best-validation checkpoint, training seed 42 + fold) and changes
ONE ingredient per variant, so that every variant can be compared, recording by recording, with the stored predictions of the same
configuration (paired cluster bootstrap, ``analysis/variants_report.py``). Only training recordings (and the recording-disjoint validation
part) are used for training, class weights, standardization, early stopping and normalization, and no statistic is computed over several
excerpts of the held-out fold. Results and reading: docs/PROTOCOL.md section 13, ``results/recording_disjoint_2026/variants/``.

    python main_variants.py --build-cache --build-views                      # all fixed-size descriptors + the extra RP caches (needs dataset and rp_extract)
    python main_variants.py --variants base aug views --configs "ENC_chs16_ks_[5, 5]_att_None_smooth_False_rp" --heads mlp --workers 8 --threads 1
    python main_variants.py --variants base bal --configs VEC_tros --heads kan mlp linear --workers 4 --threads 1
    python main_variants.py --classical lr_rp lr_tr lr_osscale lr_tros       # logistic regressions on cached descriptors (seconds to minutes)
    python main_variants.py --classical lr_tros --grouping performer --out results/variants_control_performer      # controls: --grouping, --permute-labels, --seed

Inputs. A stored configuration (``experiments/configs/pattern``, any descriptor: the tempogram-ratio networks too) or ``VEC_<set>``: one of the
heads (``--heads kan mlp linear``) on a standardized vector of cached descriptors (mean and standard deviation of the training excerpts of the fold):
``VEC_tr`` (26 values: mean and standard deviation over time of the 13 tempogram-ratio bands), ``VEC_os`` (32 values: scale-transform proxy of the
onset-strength autocorrelation, see ``onset_scale_features``), ``VEC_tros`` (both, 58 values), ``VEC_trosrh`` (plus the 60 rhythm-histogram bins).

Variants (``+`` combines them: ``bal+aug``; the RP-specific ones need an RP configuration):

====================  ==========================================================================================================
``base``              the recipe of ``main_recording_disjoint.py`` (control of this script; reproduces its predictions for a given thread count)
``bal``               class-weighted cross-entropy (weights proportional to 1 / class count of the training part of the fold)
``ls``                label smoothing 0.1
``aug``               label-preserving descriptor augmentation on RP: tempo jitter (modulation axis scaled by 0.92 to 1.08 about 0 Hz),
                      Bark-axis shift (-2 to +2 bands), gain jitter (0.8 to 1.25)
``mixup``             mixup of descriptors and labels (Beta(0.4, 0.4))
``mask``              SpecAugment-style masks (up to 2 Bark bands and 4 modulation bins)
``hier``              adds a binary "ternary against the rest" cross-entropy on the logit ``z_2 - logsumexp(z_0, z_1)``: the evidence for the ternary class is
                      learned from the pooled binary and quaternary recordings
``valloss``           early stopping and checkpoint on the validation cross-entropy instead of the validation accuracy
``bandnorm``          RP with every Bark band divided by its own modulation-axis norm (removes the per-band level, i.e. part of the timbre)
``l2``                RP divided by its overall L2 norm
``logmod``            RP resampled on a logarithmic modulation-frequency axis (60 bins from 0.5 to 10 Hz): a change of tempo becomes a shift
``logaug``            ``logmod`` with a tempo shift of -2 to +2 bins, a Bark shift of -2 to +2 bands and a gain jitter as augmentation
``nofsw``             RP computed without the fluctuation-strength weighting of ``rp_extract`` (which attenuates the modulation frequencies above 4 Hz, where
                      subdivisions of the pulse lie), ``--build-views`` writes ``rp_nofsw.npy``
``pool``              encoder output pooled to 12 x 30 (native resolution / 2) instead of 64 x 64: a head with 16 times fewer inputs
``views``             three RP views per excerpt (the stored RP analyses only the first 5.46 s of the 10-s excerpt; the other two are computed on the
                      segments that start 2.47 s and 4.94 s into the excerpt, ``--build-views``): a random view per training sample, the
                      logits of the three views averaged at validation and test (time-shift augmentation and test-time averaging)
``ens3``              three training seeds per fold, probabilities averaged (``ens5``: five)
====================  ==========================================================================================================

Classical models (``--classical``): standardization and logistic regression (C = 1) of the stored baseline, fitted on the training and validation
recordings of every fold, on one or several cached descriptors (``CLASSICAL`` below); ``--permute-labels SEED`` shuffles the labels among the
recordings (the folds are kept): the accuracy must fall to the chance level.

Every number produced here is a NEW run, to be reported next to (never in place of) the stored results.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import Dataset

import main_recording_disjoint as base
from evaluation import recording_of, softmax
from feature_cache import PartitionDataModule, cache_file
from globals import BATCH_SIZE, LR, SEED
from model import Model, init_modules

# ----------------------------------------------------------------------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------------------------------------------------------------------
VARIANTS: Dict[str, Dict[str, object]] = {
    "base": {},
    "bal": {"class_weight": "balanced"},
    "ls": {"label_smoothing": 0.1},
    "aug": {"augment": "geo"},
    "mixup": {"mixup": 0.4},
    "mask": {"augment": "mask"},
    "hier": {"hier_weight": 1.0},
    "valloss": {"monitor": "val_loss"},
    "bandnorm": {"transform": "bandnorm"},
    "l2": {"transform": "l2"},
    "logmod": {"transform": "logmod"},
    "logaug": {"transform": "logmod", "augment": "geo_log"},
    "nofsw": {"descriptor": "rp_nofsw"},
    "pool": {"out_size": [12, 30]},
    "views": {"views": 3},
    "ens3": {"seeds": 3},
    "ens5": {"seeds": 5},
}


def parse_variant(name: str) -> Dict[str, object]:
    spec: Dict[str, object] = {}
    for part in name.split("+"):
        if part not in VARIANTS:
            raise ValueError(f"unknown variant {part!r}; known: {sorted(VARIANTS)}")
        for key, value in VARIANTS[part].items():
            if key in spec and spec[key] != value:
                raise ValueError(f"variant {name!r} sets {key!r} twice")
            spec[key] = value
    return spec


# ----------------------------------------------------------------------------------------------------------------------------------
# Descriptor transforms (per excerpt only: no statistic is shared between excerpts, so no information crosses the folds)
# ----------------------------------------------------------------------------------------------------------------------------------
def transform_rp(array: np.ndarray, name: Optional[str]) -> np.ndarray:
    x = np.asarray(array, dtype=np.float32)
    if name is None:
        return x
    if name == "bandnorm":                                          # (N, 1, 24 Bark bands, 60 modulation bins)
        norm = np.sqrt((x ** 2).sum(axis=-1, keepdims=True))
        floor = 0.1 * norm.mean(axis=2, keepdims=True)              # 10 % of the mean band norm of the same excerpt
        return x / (norm + floor + 1e-12)
    if name == "l2":
        return x / (np.sqrt((x ** 2).sum(axis=(2, 3), keepdims=True)) + 1e-12)
    if name == "logmod":                                            # linear bins k <-> (k + 1) * 10 / 60 Hz  ->  60 log-spaced bins between 0.5 and 10 Hz
        f_in = (np.arange(60) + 1.0) * (10.0 / 60.0)
        f_out = 0.5 * 20.0 ** (np.arange(60) / 59.0)
        low = np.clip(np.searchsorted(f_in, f_out) - 1, 0, 58)
        weight = np.clip((f_out - f_in[low]) / (f_in[low + 1] - f_in[low]), 0.0, 1.0).astype(np.float32)
        return x[..., low] * (1.0 - weight) + x[..., low + 1] * weight
    raise ValueError(f"unknown transform {name!r}")


class ArrayDataset(Dataset):
    """In-memory descriptor array in the structure of ``PopMusicDataset`` (``{'id', 'input': {feature: tensor}, 'label'}``)."""

    def __init__(self, feature: str, array: np.ndarray, ids: Sequence[str], labels: Sequence[int]):
        self.feature, self.array, self.ids, self.labels = feature, np.ascontiguousarray(array), list(ids), list(labels)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> Dict[str, object]:
        return {"id": self.ids[i], "input": {self.feature: torch.from_numpy(self.array[i].copy())}, "label": self.labels[i]}


RP_SEGMENT = 2 ** 18                               # samples analysed by rp_extract per segment (5.46 s at the nominal 44.1 kHz)
VIEW_STARTS = (0, 108928, 217856)                  # view 0 is the stored RP (first segment); 217,856 + 262,144 = 480,000 = the end of the excerpt
VIEW_FILES = ("rp", "rp_view1", "rp_view2")


class ViewsDataset(Dataset):
    """Several cached views of one descriptor per excerpt: the input has shape [views, 1, 24, 60]."""

    def __init__(self, feature: str, arrays: Sequence[np.ndarray], ids: Sequence[str], labels: Sequence[int]):
        self.feature, self.arrays, self.ids, self.labels = feature, [np.ascontiguousarray(a) for a in arrays], list(ids), list(labels)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> Dict[str, object]:
        return {"id": self.ids[i], "input": {self.feature: torch.from_numpy(np.stack([a[i] for a in self.arrays]))}, "label": self.labels[i]}


def rp_view(segment, start: int = 0):
    """Preprocessor of the view cache: the stored RP analysis (``rhythm_features`` defaults) of the samples [start, start + 2^18) of the excerpt."""
    from data_processing import rhythm_features
    return rhythm_features(segment[..., start:start + RP_SEGMENT], extractors=["rp_extract"])


def build_view_caches(csv_path: str, audio_folder: str, cache_dir: str, num_workers: int) -> List[str]:
    """Cache the RP of the second and third view of every excerpt (``rp_view1.npy``, ``rp_view2.npy``) with the stored pipeline."""
    from functools import partial
    from torch.utils.data import DataLoader
    from data import PopMusicDataset
    from feature_cache import _identity
    built = []
    for name, start in zip(VIEW_FILES[1:], VIEW_STARTS[1:]):
        target = os.path.join(cache_dir, f"{name}.npy")
        if os.path.isfile(target):
            continue
        analysed = PopMusicDataset(csv_path=csv_path, audio_folder=audio_folder, segment_duration=10.0, features=["rp"], label_type="pattern",
                                   preprocessor=partial(rp_view, start=start))
        loader = DataLoader(analysed, batch_size=8, shuffle=False, num_workers=num_workers, collate_fn=_identity)
        rows, began = [], time.time()
        for step, batch in enumerate(loader, 1):
            rows.extend(item["input"]["rp"].numpy() for item in batch)
            if step % 100 == 0:
                print(f"[views] {name}: {len(rows)}/{len(analysed)} excerpts analysed ({time.time() - began:.0f} s)", flush=True)
        temporary = target + ".partial.npy"
        np.save(temporary, np.stack(rows).astype(np.float32))
        os.replace(temporary, target)
        built.append(name)
    return built


def rp_without_fluctuation_weighting(segment):
    """Preprocessor of the ``rp_nofsw`` cache: the stored RP analysis of the excerpt (same call as ``rhythm_features``) without the fluctuation-strength weighting."""
    from data_processing import rp_extract
    x = segment.detach().cpu().numpy() if isinstance(segment, torch.Tensor) else np.asarray(segment)
    x = x.mean(axis=0) if x.ndim > 1 else x
    return {"rp_extract": rp_extract(wavedata=x, samplerate=44100, extract_rp=True, extract_rh=False, extract_rh2=False, extract_mvd=False, extract_trh=False, extract_tssd=False,
                                     spectral_masking=True, transform_db=True, transform_phon=True, transform_sone=True, fluctuation_strength_weighting=False,
                                     skip_leadin_fadeout=1, step_width=1)}


def build_extra_rp_cache(csv_path: str, audio_folder: str, cache_dir: str, num_workers: int) -> List[str]:
    """Cache ``rp_nofsw.npy`` (RP without the fluctuation-strength weighting) with the stored pipeline otherwise unchanged."""
    from torch.utils.data import DataLoader
    from data import PopMusicDataset
    from feature_cache import _identity
    target = os.path.join(cache_dir, "rp_nofsw.npy")
    if os.path.isfile(target):
        return []
    analysed = PopMusicDataset(csv_path=csv_path, audio_folder=audio_folder, segment_duration=10.0, features=["rp"], label_type="pattern",
                               preprocessor=rp_without_fluctuation_weighting)
    loader = DataLoader(analysed, batch_size=8, shuffle=False, num_workers=num_workers, collate_fn=_identity)
    rows, began = [], time.time()
    for step, batch in enumerate(loader, 1):
        rows.extend(item["input"]["rp"].numpy() for item in batch)
        if step % 100 == 0:
            print(f"[views] rp_nofsw: {len(rows)}/{len(analysed)} excerpts analysed ({time.time() - began:.0f} s)", flush=True)
    temporary = target + ".partial.npy"
    np.save(temporary, np.stack(rows).astype(np.float32))
    os.replace(temporary, target)
    return ["rp_nofsw"]


# ----------------------------------------------------------------------------------------------------------------------------------
# Augmentations (training batches only; x has shape [B, 1, 24, 60]: Bark bands x modulation-frequency bins)
# ----------------------------------------------------------------------------------------------------------------------------------
def augment_geo(x: torch.Tensor, tempo=(0.92, 1.08), shift=2, gain=(0.8, 1.25)) -> torch.Tensor:
    batch, _, height, width = x.shape
    scale = torch.empty(batch).uniform_(*tempo)                      # a peak at modulation frequency f moves to scale * f (tempo jitter), zeros outside the axis
    origin = -1.0 - 1.0 / width                                       # normalized coordinate of 0 Hz (bin k is centred on (k + 1) * 10 / 60 Hz): scale about it
    theta = torch.zeros(batch, 2, 3)
    theta[:, 0, 0] = 1.0 / scale
    theta[:, 0, 2] = origin * (1.0 - 1.0 / scale)
    theta[:, 1, 1] = 1.0
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    offsets = torch.randint(-shift, shift + 1, (batch,))             # Bark shift with edge replication
    padded = F.pad(x, (0, 0, shift, shift), mode="replicate")
    x = torch.stack([padded[i, :, shift - int(o): shift - int(o) + height, :] for i, o in enumerate(offsets)])
    factor = torch.exp(torch.empty(batch).uniform_(float(np.log(gain[0])), float(np.log(gain[1]))))
    return x * factor.view(batch, 1, 1, 1)


def augment_geo_log(x: torch.Tensor, shift=2, gain=(0.8, 1.25)) -> torch.Tensor:
    """Augmentation for the log-modulation RP: a tempo change is a shift along the modulation axis (edge replication), plus the Bark shift and the gain jitter."""
    batch, _, height, width = x.shape
    padded = F.pad(x, (shift, shift, shift, shift), mode="replicate")
    tempo = torch.randint(-shift, shift + 1, (batch,))
    bark = torch.randint(-shift, shift + 1, (batch,))
    x = torch.stack([padded[i, :, shift - int(b): shift - int(b) + height, shift - int(t): shift - int(t) + width] for i, (t, b) in enumerate(zip(tempo, bark))])
    factor = torch.exp(torch.empty(batch).uniform_(float(np.log(gain[0])), float(np.log(gain[1]))))
    return x * factor.view(batch, 1, 1, 1)


def augment_mask(x: torch.Tensor, bands=2, bins=4) -> torch.Tensor:
    x = x.clone()
    for i in range(x.shape[0]):
        for limit, axis in ((bands, 2), (bins, 3)):
            width = int(torch.randint(0, limit + 1, (1,)))
            if width:
                start = int(torch.randint(0, x.shape[axis] - width + 1, (1,)))
                if axis == 2:
                    x[i, :, start:start + width, :] = 0.0
                else:
                    x[i, :, :, start:start + width] = 0.0
    return x


class VariantModel(Model):
    """The stored Lightning module with a different training loss and/or training-time augmentation (validation and test are unchanged)."""

    def __init__(self, *args, spec: Dict[str, object], class_weight: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec = spec
        self.register_buffer("class_weight", class_weight if class_weight is not None else torch.ones(3), persistent=False)
        self.weighted = class_weight is not None
        self.label_smoothing = float(spec.get("label_smoothing", 0.0))
        if getattr(self, "use_smoothing", False):
            raise ValueError("the variants are defined for the plain cross-entropy configurations (use_smoothing: false)")

    def forward(self, x):
        if x.dim() == 5:                                              # [batch, views, 1, H, W]: random view while training, mean logits of all views otherwise
            if self.training:
                return super().forward(x[torch.arange(x.shape[0]), torch.randint(0, x.shape[1], (x.shape[0],))])
            return torch.stack([super(VariantModel, self).forward(x[:, v]) for v in range(x.shape[1])]).mean(dim=0)
        return super().forward(x)

    def _loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(logits, target, weight=self.class_weight if self.weighted else None, label_smoothing=self.label_smoothing)
        hier = float(self.spec.get("hier_weight", 0.0))
        if hier > 0.0:
            ternary_logit = logits[:, 2] - torch.logsumexp(logits[:, :2], dim=1)
            loss = loss + hier * F.binary_cross_entropy_with_logits(ternary_logit, (target == 2).float())
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch["input"], batch["label"]
        if x.dim() == 5:                                              # views: pick one view per sample before the augmentations
            x = x[torch.arange(x.shape[0]), torch.randint(0, x.shape[1], (x.shape[0],))]
        augment = self.spec.get("augment")
        if augment == "geo":
            x = augment_geo(x)
        elif augment == "geo_log":
            x = augment_geo_log(x)
        elif augment == "mask":
            x = augment_mask(x)
        alpha = float(self.spec.get("mixup", 0.0))
        if alpha > 0.0:
            lam = float(torch.distributions.Beta(alpha, alpha).sample())
            order = torch.randperm(x.shape[0])
            logits = self(lam * x + (1.0 - lam) * x[order])
            loss = lam * self._loss(logits, y) + (1.0 - lam) * self._loss(logits, y[order])
        else:
            loss = self._loss(self(x), y)
        self.train_losses.append(loss.detach())
        self.log("train_loss_step", loss, prog_bar=True)
        return loss


VECTOR_SETS = {"tr": ("tr_stats",), "os": ("os_scale",), "tros": ("tr_stats", "os_scale"), "trosrh": ("tr_stats", "os_scale", "rh")}


def load_config(experiment_id: str) -> dict:
    """A stored configuration (YAML) or, for ``VEC_<set>``, a head on a standardized vector of cached descriptors (tempogram-ratio statistics, onset scale features)."""
    if experiment_id.startswith("VEC_"):
        if experiment_id[4:] not in VECTOR_SETS:
            raise ValueError(f"unknown vector set {experiment_id!r}; known: {sorted('VEC_' + k for k in VECTOR_SETS)}")
        return {"task": "pattern", "data": {"features": ["vector"], "preprocessor": True, "label_type": "pattern"},
                "loss": {"name": "categorical_cross_entropy", "kwargs": {"tolerance": 0.3, "use_smoothing": False}},
                "encoder": {"type": "Vector", "params": {}, "attention": {"type": None, "params": None}}, "kan": {"type": "PatternKAN"}}
    return base.load_configuration(experiment_id)


class VectorEncoder(torch.nn.Module):
    """Standardization of a descriptor vector with the mean and standard deviation of the training excerpts of the fold (no held-out information)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        return (x.flatten(1) - self.mean) / self.std


def vector_modules(config: dict, raw: np.ndarray):
    from model import PatternFC, PatternKAN
    encoder = VectorEncoder(raw.mean(axis=0), raw.std(axis=0) + 1e-6)
    kind = config["kan"]["type"]
    if kind == "PatternKAN":
        return encoder, PatternKAN(in_feats=raw.shape[1], out_probs=3)
    if kind == "PatternFC":
        return encoder, PatternFC(in_feats=raw.shape[1], out_probs=3, **(config["kan"].get("params") or {}))
    raise ValueError(kind)


# ----------------------------------------------------------------------------------------------------------------------------------
# One run (one variant, configuration, head and fold; ``seeds`` training runs averaged)
# ----------------------------------------------------------------------------------------------------------------------------------
def train_variant(config: dict, head: str, dataset, split: Dict[str, List[int]], labels: np.ndarray, fold: int, seed: int, spec: Dict[str, object],
                  checkpoint_dir: str, run_name: str, accelerator: str, max_epochs: int, patience: int, batch_size: int) -> Dict[str, object]:
    config = base.with_head(config, head)
    if "out_size" in spec:
        config["encoder"]["params"]["out_size"] = list(spec["out_size"])
    monitor = str(spec.get("monitor", "val_acc"))
    mode = "max" if monitor == "val_acc" else "min"
    members, member_accuracy, fit_seconds, started = [], [], 0.0, time.time()
    stop_epochs, best_epochs, val_best = [], [], []
    outputs = None
    for member in range(int(spec.get("seeds", 1))):
        seed_everything(seed + fold + 1000 * member, workers=True)
        data = PartitionDataModule(dataset, split["train"], split["val"], split["test"], batch_size, 0)
        batch = next(iter(data.train_dataloader()))
        if config["encoder"]["type"] == "Vector":
            encoder, head_module = vector_modules(config, dataset.array[np.array(split["train"])])
        else:
            encoder, head_module = init_modules(config, batch["input"][:, 0] if batch["input"].dim() == 5 else batch["input"])
        weight = None
        if spec.get("class_weight") == "balanced":
            counts = np.bincount(labels[np.array(split["train"])], minlength=3).astype(np.float64)
            weight = torch.tensor(counts.sum() / (3.0 * counts), dtype=torch.float32)
        model = VariantModel(encoder_module=encoder, kan_module=head_module, downstream_task="pattern", lr=LR, config=config, spec=spec, class_weight=weight)
        directory = os.path.join(checkpoint_dir, f"fold{fold}_member{member}")        # one directory per run: workers must not share it
        checkpoint = ModelCheckpoint(dirpath=directory, filename=f"{base.safe_name(run_name)}-fold{fold}" + "-{epoch:02d}", monitor=monitor, mode=mode, save_top_k=1,
                                     save_weights_only=False)
        history = base.History()
        trainer = Trainer(accelerator=accelerator, max_epochs=max_epochs, callbacks=[checkpoint, EarlyStopping(monitor=monitor, patience=patience, mode=mode), history],
                          logger=False, enable_checkpointing=True, enable_progress_bar=False, enable_model_summary=False, log_every_n_steps=10)
        t0 = time.time()
        trainer.fit(model, datamodule=data)
        fit_seconds += time.time() - t0
        trainer.test(model, datamodule=data, ckpt_path="best", verbose=False)
        outputs = model.test_outputs_npz
        members.append(softmax(np.asarray(outputs["predictions"], dtype=float)))
        member_accuracy.append(float(np.mean(np.asarray(outputs["targets"]) == members[-1].argmax(axis=1))))
        val_acc = [row["val_acc"] for row in history.rows]
        val_loss = [row["val_loss"] for row in history.rows]
        stop_epochs.append(int(trainer.current_epoch))
        val_best.append(float(max(val_acc)))
        best_epochs.append(int(np.argmax(val_acc)) if monitor == "val_acc" else int(np.argmin(val_loss)))
        shutil.rmtree(directory, ignore_errors=True)
    probabilities = np.mean(members, axis=0)
    records = [{"id": i, "recording": recording_of(i), "label": int(t), "prediction": int(p.argmax()), "probs": [float(v) for v in p], "fold": fold}
               for i, t, p in zip(outputs["ids"], outputs["targets"], probabilities)]
    return {"records": records, "stop_epoch": float(np.mean(stop_epochs)), "best_epoch": float(np.mean(best_epochs)), "fit_seconds": fit_seconds,
            "member_accuracy": member_accuracy, "n_members": len(members), "val_acc_best": float(np.mean(val_best)), "wall_seconds": time.time() - started,
            "n_train": len(split["train"]), "n_val": len(split["val"]), "n_test": len(split["test"])}


# ----------------------------------------------------------------------------------------------------------------------------------
# Jobs
# ----------------------------------------------------------------------------------------------------------------------------------
def run_file(runs_dir: str, experiment_id: str, head: str, variant: str, fold: int) -> str:
    return os.path.join(runs_dir, f"{base.safe_name(f'{experiment_id}|head={head}|variant={variant}')}_fold{fold}.json")


def train_missing(ctx, configs, jobs, seed: int, checkpoint_dir: str, accelerator: str, max_epochs: int, patience: int, batch_size: int, claim_dir: Optional[str],
                  worker: str, cache_dir: str) -> int:
    labels = np.array(ctx.labels)
    arrays: Dict[tuple, ArrayDataset] = {}
    trained = 0
    for experiment_id, head, variant, fold in jobs:
        path = run_file(ctx.runs_dir, experiment_id, head, variant, fold)
        if os.path.isfile(path):
            continue
        if claim_dir is not None and not base.claim_job(claim_dir, os.path.basename(path)[:-5], worker):
            continue
        spec = parse_variant(variant)
        feature = configs[experiment_id]["data"]["features"][0]
        key = (feature, spec.get("transform"), int(spec.get("views", 1)), spec.get("descriptor"), experiment_id if feature == "vector" else None)
        if key not in arrays:
            if feature == "vector":
                parts = classical_features(cache_dir, [], extra=VECTOR_SETS[experiment_id[4:]])
                arrays[key] = ArrayDataset(feature, np.concatenate([parts[p] for p in VECTOR_SETS[experiment_id[4:]]], axis=1), ctx.ids, ctx.labels)
            elif key[2] > 1:
                if spec.get("descriptor"):
                    raise ValueError("the views are cached for the stored RP only")
                views = [transform_rp(np.load(os.path.join(cache_dir, f"{name}.npy")), spec.get("transform")) for name in VIEW_FILES[:key[2]]]
                arrays[key] = ViewsDataset(feature, views, ctx.ids, ctx.labels)
            else:
                raw = np.load(cache_file(cache_dir, str(spec.get("descriptor", feature))), mmap_mode="r")
                arrays[key] = ArrayDataset(feature, transform_rp(np.asarray(raw), spec.get("transform")), ctx.ids, ctx.labels)
        name = f"{experiment_id}|head={head}|variant={variant}"
        started = base.utc_now()
        result = train_variant(configs[experiment_id], head, arrays[key], ctx.splits[fold], labels, fold, seed, spec,
                               os.path.join(checkpoint_dir, f"seed{seed}", base.safe_name(name)), name, accelerator, max_epochs, patience, batch_size)
        result.update({"worker": worker, "device": accelerator, "threads": torch.get_num_threads(), "torch": torch.__version__, "started_utc": started,
                       "finished_utc": base.utc_now(), "grouping": ctx.grouping, "spec": spec})
        base.write_json_atomic(path, result)
        trained += 1
        accuracy = float(np.mean([r["label"] == r["prediction"] for r in result["records"]]))
        print(f"[{worker}] [{name}] fold {fold + 1}/{len(ctx.splits)}: held-out accuracy {accuracy:.4f} (members {['%.3f' % a for a in result['member_accuracy']]}, "
              f"best epoch {result['best_epoch']:.1f}, stop {result['stop_epoch']:.1f}, {result['fit_seconds']:.0f} s)", flush=True)
    return trained


def spawn_workers(args, plan_path: str, all_files: List[str], out_dir: str) -> None:
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    claim_dir = os.path.join(out_dir, "claims")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[variable] = str(args.threads)
    for attempt in range(args.retries + 1):
        missing = [p for p in all_files if not os.path.isfile(p)]
        if not missing:
            return
        shutil.rmtree(claim_dir, ignore_errors=True)
        os.makedirs(claim_dir)
        count = min(args.workers, len(missing))
        print(f"[parent] attempt {attempt + 1}: {len(missing)} of {len(all_files)} runs to train with {count} worker process(es) x {args.threads} thread(s)", flush=True)
        children = []
        for index in range(count):
            command = [sys.executable, "-B", os.path.abspath(__file__), "--plan", plan_path, "--worker-index", str(index), "--claim-dir", claim_dir, "--csv", args.csv,
                       "--audio", args.audio, "--cache-dir", args.cache_dir, "--out", out_dir, "--checkpoints", args.checkpoints, "--folds", str(args.folds),
                       "--seed", str(args.seed), "--accelerator", args.accelerator, "--max-epochs", str(args.max_epochs), "--patience", str(args.patience),
                       "--batch-size", str(args.batch_size), "--threads", str(args.threads), "--grouping", args.grouping, "--segment-duration", str(args.segment_duration),
                       "--no-nnpack"]
            log = open(os.path.join(logs_dir, f"worker_{index}_attempt{attempt + 1}.log"), "a")
            children.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment), log))
        last_report = time.time()
        while any(child.poll() is None for child, _ in children):
            time.sleep(5)
            if time.time() - last_report >= 300:
                print(f"[parent] {base.utc_now()}: {sum(os.path.isfile(p) for p in all_files)} of {len(all_files)} runs finished", flush=True)
                last_report = time.time()
        for _, log in children:
            log.close()
        print(f"[parent] workers exited with codes {[child.returncode for child, _ in children]}", flush=True)
    still = [os.path.basename(p) for p in all_files if not os.path.isfile(p)]
    if still:
        raise RuntimeError(f"{len(still)} runs could not be trained (see {logs_dir}); first: {still[:3]}")


# ----------------------------------------------------------------------------------------------------------------------------------
# Output files in the schema of main_recording_disjoint.py (so that analysis/recording_disjoint_report.py helpers can read them)
# ----------------------------------------------------------------------------------------------------------------------------------
def collect_runs(ctx, plan: Dict[str, object]):
    """Records, fold rows and run metadata of every finished run of ``plan`` (one entry per method, folds concatenated in order)."""
    methods: Dict[str, List[Dict[str, object]]] = {}
    fold_rows: List[List[object]] = []
    metadata: List[Dict[str, object]] = []
    for experiment_id, head, variant in plan["methods"]:
        name = f"{experiment_id}|head={head}|variant={variant}"
        for fold in range(len(ctx.splits)):
            with open(run_file(ctx.runs_dir, experiment_id, head, variant, fold)) as handle:
                result = json.load(handle)
            methods.setdefault(name, []).extend(result["records"])
            accuracy = float(np.mean([r["label"] == r["prediction"] for r in result["records"]]))
            fold_rows.append([name, fold, len(result["records"]), len({r["recording"] for r in result["records"]}), accuracy, "", result["best_epoch"],
                              result["stop_epoch"], result["val_acc_best"], result["fit_seconds"]])
            metadata.append({"run": name, "fold": fold, **{k: result.get(k) for k in ("device", "threads", "worker", "torch", "started_utc", "finished_utc", "fit_seconds",
                                                                                      "stop_epoch", "best_epoch", "n_members", "member_accuracy")}})
    return methods, fold_rows, metadata


def write_outputs(ctx, methods, fold_rows, metadata, variants: Dict[str, object], settings: Dict[str, object], out_dir: str, bootstrap: int, seed: int, command: str,
                  started_utc: str, started: float) -> None:
    """Merge the methods into ``summary.csv`` / ``folds.csv`` / ``predictions/`` / ``confusion_matrices.json`` / ``PROVENANCE.json`` (earlier waves stay)."""
    summaries = [base.summarize(name, records, bootstrap, seed) for name, records in methods.items()]
    columns = ["method", "n_excerpts", "n_recordings", "accuracy", "accuracy_ci_cluster_low", "accuracy_ci_cluster_high", "accuracy_wilson_low", "accuracy_wilson_high",
               "balanced_accuracy", "macro_f1", "recall_0", "recall_1", "recall_2", "recording_accuracy", "recording_ci_low", "recording_ci_high"]
    path = os.path.join(out_dir, "summary.csv")
    existing = {}
    if os.path.isfile(path):
        with open(path, newline="") as handle:
            existing = {row["method"]: [row[c] for c in columns] for row in csv.DictReader(handle)}
    existing.update({s["method"]: [s[c] for c in columns] for s in summaries})
    base.write_csv(path, columns, list(existing.values()))
    folds_path = os.path.join(out_dir, "folds.csv")
    folds_existing: Dict[tuple, List[object]] = {}
    if os.path.isfile(folds_path):
        with open(folds_path, newline="") as handle:
            for row in csv.reader(handle):
                if row and row[0] != "method":
                    folds_existing[(row[0], row[1])] = row
    for row in fold_rows:
        folds_existing[(row[0], str(row[1]))] = [f"{v:.6g}" if isinstance(v, float) else v for v in row]
    base.write_csv(folds_path, ["method", "fold", "n_test_excerpts", "n_test_recordings", "test_accuracy", "test_loss", "best_epoch", "stop_epoch", "val_acc_best", "fit_seconds"],
                   list(folds_existing.values()))
    matrices_path = os.path.join(out_dir, "confusion_matrices.json")
    matrices = {}
    if os.path.isfile(matrices_path):
        with open(matrices_path) as handle:
            matrices = json.load(handle)
    matrices.update({s["method"]: s["confusion_matrix"] for s in summaries})
    with open(matrices_path, "w") as handle:
        json.dump(matrices, handle, indent=1)
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    for name, records in methods.items():
        target = os.path.join(out_dir, "predictions", f"{base.safe_name(name)}.csv.gz")
        with gzip.open(target, "wt", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["id", "recording", "label", "prediction", "prob_0", "prob_1", "prob_2", "fold"])
            for r in records:
                writer.writerow([r["id"], r["recording"], r["label"], r["prediction"], *[f"{v:.6g}" for v in r["probs"]], r["fold"]])
    provenance_path = os.path.join(out_dir, "PROVENANCE.json")
    previous = {}
    if os.path.isfile(provenance_path):
        with open(provenance_path) as handle:
            previous = json.load(handle)
    runs = {(m["run"], m["fold"]): m for m in previous.get("runs", [])}
    runs.update({(m["run"], m["fold"]): m for m in metadata})
    known = dict(previous.get("variants", {}))
    known.update(variants)
    commands = list(previous.get("commands", [])) + [command]
    with open(ctx.partition_path) as handle:
        partition = json.load(handle)
    here = os.path.dirname(os.path.abspath(__file__))
    provenance = {"generated_utc": base.utc_now(), "first_started_utc": previous.get("first_started_utc", started_utc), "commands": commands,
                  "NEW_RUN": "training-recipe and descriptor variants under the recording-disjoint protocol; report next to, never in place of, the stored results",
                  "parameters": {"folds": len(ctx.splits), "seed": seed, "grouping": ctx.grouping, "bootstrap": bootstrap, **{k: v for k, v in settings.items() if k != "notes"}},
                  "variants": known,
                  "partition_summary": {"excerpts": len(ctx.ids), "recordings": len(set(ctx.groups)), "test_groups_per_fold": [len(p["test"]) for p in partition["parts"]]},
                  "environment": {"torch": torch.__version__, "numpy": np.__version__, "hardware": base.hardware_description()},
                  "code_sha256": {name: base.sha256_of(os.path.join(here, name)) for name in base.SOURCE_FILES + ("main_variants.py",) if os.path.isfile(os.path.join(here, name))},
                  "rp_extract": base.rp_extract_description() or previous.get("rp_extract"), "notes": {**previous.get("notes", {}), **settings.get("notes", {})},
                  "runs": list(runs.values()), "wall_clock_seconds_last_invocation": time.time() - started}
    with open(provenance_path, "w") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")


# ----------------------------------------------------------------------------------------------------------------------------------
# Classical models on cached descriptors (logistic regression on standardized features: the stored baseline, with other inputs)
# ----------------------------------------------------------------------------------------------------------------------------------
def onset_scale_features(onset: np.ndarray, fps: float = 44100 / 512, n_lag: int = 64, n_coeff: int = 32, lag_min: float = 0.25, lag_max: float = 4.0) -> np.ndarray:
    """Tempo-invariant rhythm descriptor of the onset-strength curve: magnitude of the Fourier transform, along the log-lag axis, of its autocorrelation
    (a scale-transform proxy: a change of tempo shifts the autocorrelation along log-lag, which leaves the magnitude unchanged)."""
    x = onset - onset.mean(axis=1, keepdims=True)
    length = x.shape[1]
    spectrum = np.fft.rfft(x, n=2 * length, axis=1)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, :length]
    acf = acf / (acf[:, :1] + 1e-12)
    lags = np.geomspace(lag_min * fps, lag_max * fps, n_lag)
    low = np.floor(lags).astype(int)
    weight = lags - low
    curve = acf[:, low] * (1.0 - weight) + acf[:, low + 1] * weight
    return np.abs(np.fft.rfft(curve * np.hanning(n_lag), axis=1))[:, :n_coeff].astype(np.float32)


CLASSICAL = {                      # name -> (descriptors concatenated after flattening, class_weight)
    "lr_rp": (("rp",), None),
    "lr_rp_bal": (("rp",), "balanced"),
    "lr_rp_bandnorm": (("rp_bandnorm",), None),
    "lr_rp_nofsw": (("rp_nofsw",), None),
    "lr_rh": (("rh",), None),
    "lr_trh": (("trh",), None),
    "lr_tssd": (("tssd",), None),
    "lr_tr": (("tr_stats",), None),
    "lr_tr_bal": (("tr_stats",), "balanced"),
    "lr_tr_mean": (("tr_mean",), None),
    "lr_tr_std": (("tr_std",), None),
    "lr_tr_q": (("tr_quantiles",), None),
    "lr_tros": (("tr_stats", "os_scale"), None),
    "lr_tros_bal": (("tr_stats", "os_scale"), "balanced"),
    "lr_tros_rh": (("tr_stats", "os_scale", "rh"), None),
    "lr_osscale": (("os_scale",), None),
    "lr_rp_rh_trh_tssd": (("rp", "rh", "trh", "tssd"), None),
    "lr_all": (("rp", "rh", "trh", "tssd", "tr_stats", "os_scale"), None),
    "lr_compact": (("rh", "trh", "tr_stats", "os_scale"), None),
}


def classical_features(cache_dir: str, names: Sequence[str], extra: Sequence[str] = ()) -> Dict[str, np.ndarray]:
    wanted = {d for name in names for d in CLASSICAL[name][0]} | set(extra)
    out: Dict[str, np.ndarray] = {}
    for key in sorted(wanted):
        if key == "rp_bandnorm":
            out[key] = transform_rp(np.load(cache_file(cache_dir, "rp")), "bandnorm").reshape(-1, 24 * 60)
        elif key in ("tr_stats", "tr_mean", "tr_std", "tr_quantiles"):
            ratio = np.load(cache_file(cache_dir, "tempogram_ratio"))[:, 0]                   # (N, 13 tempo ratios, 938 frames)
            parts = {"tr_mean": [ratio.mean(axis=-1)], "tr_std": [ratio.std(axis=-1)], "tr_stats": [ratio.mean(axis=-1), ratio.std(axis=-1)],
                     "tr_quantiles": [np.quantile(ratio, q, axis=-1) for q in (0.1, 0.5, 0.9)]}[key]
            out[key] = np.concatenate(parts, axis=-1).astype(np.float32)
        elif key == "os_scale":
            out[key] = onset_scale_features(np.load(cache_file(cache_dir, "onset_strength"))[:, 0, :])
        else:
            array = np.load(cache_file(cache_dir, key))
            out[key] = array.reshape(len(array), -1)
    return out


def classical_records(name: str, features: Dict[str, np.ndarray], ctx, permute_seed: Optional[int] = None) -> Dict[str, object]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    parts, class_weight = CLASSICAL[name]
    matrix = np.concatenate([features[p] for p in parts], axis=1)
    labels = np.array(ctx.labels)
    if permute_seed is not None:                                                  # control: labels are shuffled among the recordings
        recordings = sorted(set(ctx.groups))
        shuffled = np.random.default_rng(permute_seed).permutation(len(recordings))
        label_of = {r: int(labels[ctx.groups.index(recordings[j])]) for r, j in zip(recordings, shuffled)}
        labels = np.array([label_of[g] for g in ctx.groups])
    records, fold_rows = [], []
    for fold, split in enumerate(ctx.splits):
        train, test = np.array(split["train"] + split["val"]), np.array(split["test"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=1.0, class_weight=class_weight)).fit(matrix[train], labels[train])
        probabilities = np.zeros((len(test), 3))
        probabilities[:, classifier.classes_] = classifier.predict_proba(matrix[test])
        fold_records = [{"id": ctx.ids[i], "recording": recording_of(ctx.ids[i]), "label": int(labels[i]), "prediction": int(p.argmax()), "probs": [float(v) for v in p], "fold": fold}
                        for i, p in zip(test, probabilities)]
        records.extend(fold_records)
        fold_rows.append([f"classical:{name}|descriptor={'+'.join(parts)}", fold, len(fold_records), len({r["recording"] for r in fold_records}),
                          float(np.mean([r["label"] == r["prediction"] for r in fold_records])), "", "", "", "", ""])
    return {"name": f"classical:{name}|descriptor={'+'.join(parts)}", "records": records, "fold_rows": fold_rows, "dimension": int(matrix.shape[1])}


# ----------------------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=None, help="variant names (see the table above); '+' combines")
    parser.add_argument("--classical", nargs="+", default=None, choices=sorted(CLASSICAL), help="logistic regressions on cached descriptors (no network training)")
    parser.add_argument("--configs", nargs="+", help="experiment ids (file stems of experiments/configs/pattern)")
    parser.add_argument("--heads", nargs="+", choices=sorted(base.HEAD_OVERRIDES), default=["kan"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED, help="seed of the fold partition (and base seed of the training runs), as in main_recording_disjoint.py")
    parser.add_argument("--grouping", choices=base.GROUPINGS, default="recording",
                        help="recording (default); performer / excerpt: the controls of main_recording_disjoint.py (stricter / looser folds)")
    parser.add_argument("--permute-labels", type=int, default=None, metavar="SEED",
                        help="control for --classical: permute the labels among the recordings with this seed (the folds are kept); the accuracy must fall to the chance level")
    parser.add_argument("--segment-duration", type=float, default=10.0, help="excerpt length in seconds (10: the released protocol; other values are for the tests)")
    parser.add_argument("--csv", default="./dataset/dataset.csv")
    parser.add_argument("--audio", default="./dataset/file_audio/")
    parser.add_argument("--cache-dir", default="./dataset/feature_cache")
    parser.add_argument("--out", default="./results/variants")
    parser.add_argument("--checkpoints", default=os.path.join("./experiments", "checkpoints", "variants"))
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="cpu")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=1, help="training processes working on different runs at the same time")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads per training process")
    parser.add_argument("--no-nnpack", action="store_true", help="switch off the NNPACK convolution path of the CPU build of PyTorch")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--build-cache", action="store_true", help="compute the missing cached descriptors (RP, RH, TSSD, TRH, onset strength, tempogram ratio) first")
    parser.add_argument("--build-views", action="store_true", help="compute the RP of the second and third view of every excerpt and the RP without fluctuation-strength weighting (needed by the 'views' and 'nofsw' variants)")
    parser.add_argument("--cache-workers", type=int, default=8, help="processes of --build-cache")
    parser.add_argument("--note", action="append", default=[], metavar="KEY=VALUE", help="free-text provenance entries stored in PROVENANCE.json (repeatable)")
    parser.add_argument("--plan", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--claim-dir", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


ALL_FEATURES = ("rp", "rh", "tssd", "trh", "onset_strength", "tempogram_ratio")
DEFAULT_CONFIG = "ENC_chs16_ks_[5, 5]_att_None_smooth_False_rp"          # only used to build the partition / context in --classical mode


def main(argv=None) -> int:
    args = parse_args(argv)
    started, started_utc = time.time(), base.utc_now()
    base.configure_cpu(args.threads, args.no_nnpack)
    command = "python main_variants.py " + " ".join(sys.argv[1:] if argv is None else argv)
    if args.build_cache:                                                           # all fixed-size descriptors in one pass (RP, RH, TSSD, TRH, onset strength, tempogram ratio)
        from feature_cache import build_cache
        built = build_cache(args.csv, args.audio, list(ALL_FEATURES), args.cache_dir, args.segment_duration, "pattern", args.cache_workers, False)
        print(f"[cache] built now: {built}", flush=True)
    if args.build_views:
        print(f"[views] built now: {build_view_caches(args.csv, args.audio, args.cache_dir, args.cache_workers) + build_extra_rp_cache(args.csv, args.audio, args.cache_dir, args.cache_workers)}", flush=True)
    if (args.build_cache or args.build_views) and not (args.variants or args.classical or args.plan):
        return 0
    if args.plan:
        with open(args.plan) as handle:
            plan = json.load(handle)
    elif args.variants:
        if not args.configs:
            raise SystemExit("--configs is required with --variants")
        plan = {"methods": [[c, h, v] for v in args.variants for c in args.configs for h in args.heads], "max_epochs": args.max_epochs, "patience": args.patience,
                "batch_size": args.batch_size, "accelerator": args.accelerator, "threads": args.threads}
    else:
        plan = {"methods": []}
    for _, _, variant in plan["methods"]:
        parse_variant(variant)
    configs = {c: load_config(c) for c in sorted({m[0] for m in plan["methods"]})} or {DEFAULT_CONFIG: load_config(DEFAULT_CONFIG)}
    os.makedirs(args.out, exist_ok=True)
    cached = {c: cfg for c, cfg in configs.items() if cfg["encoder"]["type"] != "Vector"} or {DEFAULT_CONFIG: load_config(DEFAULT_CONFIG)}   # vector inputs are built from the caches
    ctx = base.prepare(cached, args.csv, args.audio, args.cache_dir, args.out, args.folds, args.seed, args.segment_duration, 0, False, args.grouping)
    settings = {"max_epochs": args.max_epochs, "patience": args.patience, "batch_size": args.batch_size, "accelerator": base.resolve_accelerator(args.accelerator),
                "workers": args.workers, "threads_per_worker": args.threads, "no_nnpack": bool(args.no_nnpack),
                "notes": dict(item.split("=", 1) for item in args.note if "=" in item)}
    if args.classical:
        features = classical_features(args.cache_dir, args.classical)
        methods, fold_rows, variants = {}, [], {}
        for name in args.classical:
            result = classical_records(name, features, ctx, args.permute_labels)
            methods[result["name"]] = result["records"]
            fold_rows += result["fold_rows"]
            variants[result["name"]] = {"model": "StandardScaler + LogisticRegression(C=1, max_iter=300)", "descriptors": list(CLASSICAL[name][0]),
                                        "class_weight": CLASSICAL[name][1], "input_dimension": result["dimension"]}
            print(f"[{result['name']}] accuracy {np.mean([r['label'] == r['prediction'] for r in result['records']]):.4f} (dimension {result['dimension']})", flush=True)
        write_outputs(ctx, methods, fold_rows, [], variants, settings, args.out, args.bootstrap, args.seed, command, started_utc, started)
    if not plan["methods"]:
        print(f"[done] results in {args.out}")
        return 0
    jobs = [(c, h, v, fold) for c, h, v in plan["methods"] for fold in range(args.folds)]
    accelerator = base.resolve_accelerator(args.accelerator)
    train_args = (args.seed, args.checkpoints, accelerator, args.max_epochs, args.patience, args.batch_size)
    if args.worker_index is not None:
        logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
        warnings.filterwarnings("ignore", message=".*does not have many workers.*")
        train_missing(ctx, configs, jobs, *train_args, claim_dir=args.claim_dir, worker=f"worker{args.worker_index}", cache_dir=args.cache_dir)
        return 0
    all_files = [run_file(ctx.runs_dir, c, h, v, fold) for c, h, v, fold in jobs]
    if args.workers > 1:
        plan_path = os.path.join(args.out, f"plan_{int(started)}.json")
        with open(plan_path, "w") as handle:
            json.dump(plan, handle)
        spawn_workers(args, plan_path, all_files, args.out)
    else:
        logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
        train_missing(ctx, configs, jobs, *train_args, claim_dir=None, worker="main", cache_dir=args.cache_dir)
    methods, fold_rows, metadata = collect_runs(ctx, plan)
    write_outputs(ctx, methods, fold_rows, metadata, {v: parse_variant(v) for v in sorted({m[2] for m in plan["methods"]})}, settings, args.out, args.bootstrap, args.seed,
                  command, started_utc, started)
    print(f"[done] results in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
