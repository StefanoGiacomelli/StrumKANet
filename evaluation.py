"""Evaluation helpers (numpy / standard library) and the protocol options of the entry points.

The stored 2025 protocol is the default of every option (``ProtocolOptions()``): unseeded excerpt-level split drawn on every
``setup()`` call, test on the final-epoch weights, no saved predictions, ``accelerator='auto'``. See docs/PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from globals import SEED


# ----------------------------------------------------------------------------------------------------------------------
# Protocol options (default = legacy)
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class ProtocolOptions:
    split_mode: str = "legacy"                 # legacy | seeded | grouped
    split_seed: int = SEED
    partition_file: Optional[str] = None
    eval_ckpt: str = "last"                    # last (legacy: final-epoch weights) | best (best-validation checkpoint)
    save_predictions: bool = False
    accelerator: str = "auto"
    tag: Optional[str] = None

    def is_legacy(self) -> bool:
        return self.split_mode == "legacy" and self.eval_ckpt == "last"

    def namespace(self) -> str:
        """Sub-folder for logs, checkpoints and result files; empty for the legacy protocol, so stored files are never mixed with new ones."""
        if self.tag:
            return self.tag
        return "" if self.is_legacy() else f"{self.split_mode}_seed{self.split_seed}_{self.eval_ckpt}"

    def warnings(self) -> List[str]:
        out = []
        if self.eval_ckpt == "best" and self.split_mode == "legacy":
            out.append("--eval-ckpt best with --split legacy evaluates on a partition that is re-drawn at test time; use --split seeded or grouped")
        if self.split_mode == "grouped" and self.partition_file is None:
            out.append("--split grouped without --partition-file: the partition is not stored for audit")
        return out


def add_protocol_arguments(parser: argparse.ArgumentParser, stage: int) -> None:
    group = parser.add_argument_group("protocol options (all default to the stored 2025 protocol)")
    group.add_argument("--split", dest="split_mode", choices=("legacy", "seeded", "grouped"), default="legacy",
                       help="legacy: unseeded excerpt-level split re-drawn on every setup() call (default); seeded: excerpt-level split drawn once; "
                            "grouped: recording-disjoint, class-stratified split drawn once")
    group.add_argument("--split-seed", type=int, default=SEED, help="seed of the seeded/grouped split (default: globals.SEED)")
    group.add_argument("--partition-file", default=None, help="JSON file where the computed partition is stored (and checked if it exists)")
    if stage == 2:
        group.add_argument("--eval-ckpt", choices=("last", "best"), default="last",
                           help="last: final-epoch weights (default, as stored); best: best-validation checkpoint (trainer.test ckpt_path='best')")
    group.add_argument("--save-predictions", action="store_true", help="write test predictions, confusion matrix and metrics")
    group.add_argument("--accelerator", choices=("auto", "cpu", "gpu", "mps"), default="auto",
                       help="Lightning accelerator (default auto; the model cannot run on Apple MPS, use cpu there)")
    group.add_argument("--tag", default=None, help="output namespace; derived automatically for any non-legacy protocol")


def options_from_args(args: argparse.Namespace) -> ProtocolOptions:
    return ProtocolOptions(split_mode=args.split_mode, split_seed=args.split_seed, partition_file=args.partition_file,
                           eval_ckpt=getattr(args, "eval_ckpt", "last"), save_predictions=args.save_predictions,
                           accelerator=args.accelerator, tag=args.tag)


# ----------------------------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------------------------
def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int = 3) -> np.ndarray:
    """Rows: true class, columns: predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def class_report(cm: np.ndarray) -> Dict[str, object]:
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    diag = np.diag(cm)
    recall = np.divide(diag, support, out=np.zeros(len(diag)), where=support > 0)
    precision = np.divide(diag, predicted, out=np.zeros(len(diag)), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(diag)), where=(precision + recall) > 0)
    return {"accuracy": float(diag.sum() / max(cm.sum(), 1)), "balanced_accuracy": float(recall.mean()), "macro_f1": float(f1.mean()),
            "per_class": [{"class": c, "support": int(support[c]), "precision": float(precision[c]), "recall": float(recall[c]), "f1": float(f1[c])}
                          for c in range(len(diag))]}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def cluster_bootstrap_ci(groups: Sequence[str], values: Sequence[float], resamples: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """Percentile 95 % interval of the mean of ``values`` when whole recordings (groups) are resampled with replacement."""
    by_group: Dict[str, List[float]] = {}
    for g, v in zip(groups, values):
        by_group.setdefault(g, []).append(float(v))
    keys = sorted(by_group)
    sums = np.array([sum(by_group[k]) for k in keys])
    counts = np.array([len(by_group[k]) for k in keys])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(resamples, len(keys)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def recording_of(excerpt_id: str) -> str:
    """Recording identifier of an excerpt id of the form ``<performer>-<title>_seg<k>``."""
    return excerpt_id.rsplit("_seg", 1)[0]


def recording_level(groups: Sequence[str], labels: Sequence[int], probabilities: np.ndarray) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """One prediction per recording: argmax of the mean excerpt probability."""
    order = sorted(set(groups))
    labels_arr, probs_arr = np.asarray(labels), np.asarray(probabilities)
    y_true, y_pred = [], []
    for g in order:
        mask = np.array([x == g for x in groups])
        y_true.append(int(labels_arr[mask][0]))
        y_pred.append(int(probs_arr[mask].mean(axis=0).argmax()))
    return order, np.array(y_true), np.array(y_pred)


# ----------------------------------------------------------------------------------------------------------------------
# Saving test outputs of a Lightning model (``model.test_outputs_npz`` is filled by ``Model.on_test_epoch_end``)
# ----------------------------------------------------------------------------------------------------------------------
def save_test_outputs(outputs: Dict[str, object], directory: str, run_name: str, n_classes: int = 3) -> Dict[str, object]:
    """Write ``<run_name>_predictions.csv``, ``_confusion_matrix.csv`` and ``_metrics.json``; return the metrics."""
    os.makedirs(directory, exist_ok=True)
    logits = np.asarray(outputs["predictions"], dtype=float)
    targets = np.asarray(outputs["targets"]).astype(int)
    ids = list(outputs.get("ids") or [""] * len(targets))
    probabilities = softmax(logits)
    predicted = logits.argmax(axis=1)
    with open(os.path.join(directory, f"{run_name}_predictions.csv"), "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "recording", "label", "prediction"] + [f"logit_{c}" for c in range(n_classes)] + [f"prob_{c}" for c in range(n_classes)])
        for i, excerpt in enumerate(ids):
            writer.writerow([excerpt, recording_of(excerpt) if excerpt else "", targets[i], predicted[i]]
                            + [f"{v:.6g}" for v in logits[i]] + [f"{v:.6g}" for v in probabilities[i]])
    cm = confusion_matrix(targets, predicted, n_classes)
    with open(os.path.join(directory, f"{run_name}_confusion_matrix.csv"), "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["true\\predicted"] + list(range(n_classes)))
        for c in range(n_classes):
            writer.writerow([c] + [int(v) for v in cm[c]])
    metrics = class_report(cm)
    metrics.update(n_excerpts=int(len(targets)), n_recordings=len({recording_of(i) for i in ids if i}), confusion_matrix=cm.tolist())
    with open(os.path.join(directory, f"{run_name}_metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=1)
        handle.write("\n")
    return metrics
