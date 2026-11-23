"""Recording-disjoint re-evaluation of the stored StrumKANet configurations (new, optional; not part of the 2025 campaign).

One command, once the dataset and ``rp_extract`` are in place::

    python main_recording_disjoint.py                  # nine stored configurations (test accuracy > 0.90), 5 folds, WavKAN head
    python main_recording_disjoint.py --benchmark      # no dataset needed: measures seconds per epoch here and prints the runtime estimate

Protocol: descriptors are cached once (``feature_cache.py``); recordings are split into K stratified, recording-disjoint folds
(``splits.py``); inside the training recordings of each fold a recording-disjoint validation part (12.5 %) drives early stopping
(patience 20, at most 50 epochs, as in Stage 2); the held-out fold is evaluated with the best-validation checkpoint. Every
recording is held out exactly once, so the pooled out-of-fold predictions cover the whole corpus. Reported: excerpt accuracy and
recording-level accuracy with recording-level (cluster) bootstrap intervals, per-class metrics, confusion matrices, and baselines
on the same folds (majority class, logistic regression on the flattened descriptor; optionally ``--heads kan mlp linear``).

Running time. Every (configuration, head, fold) run is independent and is stored as soon as it finishes (``<out>/runs``), so an
interrupted command resumes where it stopped. ``--workers N --threads T`` trains N runs at the same time in separate processes
with T CPU threads each (the WavKAN head is memory-bandwidth bound, so several small processes use a CPU better than one large
one); ``--no-nnpack`` switches off the NNPACK convolution path of the CPU build of PyTorch, which was found to be about ten times
slower than the default path for the one-dimensional onset-strength encoder on Apple-silicon builds (see docs/PROTOCOL.md).
``--grouping performer`` (all recordings of a performer held out together; stricter) and ``--grouping excerpt`` (excerpt-level folds;
looser) are controls: the same pipeline with other grouping units, to place the recording-disjoint result between a looser and a
stricter protocol.

Every number produced here is a NEW run, to be reported next to (never in place of) the stored 2025 results.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from evaluation import class_report, cluster_bootstrap_ci, confusion_matrix, recording_level, recording_of, softmax, wilson_interval
from feature_cache import (GROUPINGS, CachedFeatureDataset, PartitionDataModule, build_cache, cache_file, grouping_keys, index_dataset, load_index,
                           missing_audio)
from globals import AUDIO_FOLDER, BATCH_SIZE, CSV_PATH, LR, MAIN_FOLDER, SEED
from model import Model, init_modules
from splits import check_partition, grouped_kfold, grouped_partition, mixed_group_kfold, mixed_group_partition

CONFIG_DIR = os.path.join(MAIN_FOLDER, "configs", "pattern")
STORED_TRAINING_RESULTS = os.path.join(MAIN_FOLDER, "results", "training_results.csv")
HEAD_OVERRIDES = {"kan": None,                                                                    # the stored WavKAN head
                  "mlp": {"type": "PatternFC", "params": {"hidden_dims": [64, 32]}},             # repository MLP head (PatternFC defaults)
                  "linear": {"type": "PatternFC", "params": {"hidden_dims": []}}}                # single linear layer
INPUT_SHAPES = {"rp": (24, 60), "rh": (60,), "tssd": (24, 49), "trh": (60, 7), "onset_strength": (938,), "tempogram_ratio": (13, 938)}
HEAD_COST = {"kan": 1.0, "mlp": 0.35, "linear": 0.25}                                             # relative training cost, used only to order jobs


# ---------------------------------------------------------------------------------------------------------------- helpers
def resolve_accelerator(name: str) -> str:
    """'auto' never selects Apple MPS (AdaptiveAvgPool with non-divisible sizes fails there): GPU if CUDA is available, else CPU."""
    if name == "auto":
        return "gpu" if torch.cuda.is_available() else "cpu"
    return name


def configure_cpu(threads: int = 0, no_nnpack: bool = False) -> None:
    """Process-level CPU settings: intra-op threads (0 keeps the PyTorch default) and, optionally, the NNPACK convolution path."""
    if threads > 0:
        torch.set_num_threads(threads)
    if no_nnpack and hasattr(torch.backends, "nnpack"):
        torch.backends.nnpack.set_flags(False)


def stored_configuration_ids(min_test_accuracy: float) -> List[str]:
    with open(STORED_TRAINING_RESULTS, newline="") as handle:
        return [r["experiment_id"] for r in csv.DictReader(handle) if float(r["test_metric"]) > min_test_accuracy]


def load_configuration(experiment_id: str) -> dict:
    with open(os.path.join(CONFIG_DIR, f"{experiment_id}.yaml")) as handle:
        return yaml.safe_load(handle)


def with_head(config: dict, head: str) -> dict:
    config = copy.deepcopy(config)
    if HEAD_OVERRIDES[head] is not None:
        config["kan"] = dict(HEAD_OVERRIDES[head])
    return config


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", text).strip("_")


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json_atomic(path: str, payload: object) -> None:
    """Write a JSON file so that an interrupted process never leaves a truncated file behind."""
    temporary = f"{path}.tmp{os.getpid()}"
    with open(temporary, "w") as handle:
        json.dump(payload, handle)
    os.replace(temporary, path)


def claim_job(claim_dir: str, key: str, owner: str) -> bool:
    """Atomically claim a job (``O_EXCL`` file creation): True for exactly one of several concurrent processes."""
    try:
        descriptor = os.open(os.path.join(claim_dir, f"{key}.claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w") as handle:
        handle.write(f"{owner} {utc_now()}\n")
    return True


class History(Callback):
    """Per-epoch validation metrics (the Lightning module logs them in its own hook, so read them at validation end)."""

    def __init__(self):
        self.rows: List[Dict[str, float]] = []

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            metrics = trainer.callback_metrics
            self.rows.append({"epoch": trainer.current_epoch, "val_acc": float(metrics["val_acc"]), "val_loss": float(metrics["val_loss"])})


# ---------------------------------------------------------------------------------------------------------- model training
def train_and_test(config: dict, head: str, dataset, split: Dict[str, List[int]], fold: int, seed: int, checkpoint_dir: str, run_name: str,
                   accelerator: str, max_epochs: int, patience: int, batch_size: int, num_workers: int) -> Dict[str, object]:
    config = with_head(config, head)
    seed_everything(seed + fold, workers=True)
    data = PartitionDataModule(dataset, split["train"], split["val"], split["test"], batch_size, num_workers)
    batch = next(iter(data.train_dataloader()))
    encoder, head_module = init_modules(config, batch["input"])
    model = Model(encoder_module=encoder, kan_module=head_module, downstream_task="pattern", lr=LR, config=config)
    checkpoint = ModelCheckpoint(dirpath=checkpoint_dir, filename=f"{safe_name(run_name)}-fold{fold}" + "-{epoch:02d}-{val_acc:.4f}",
                                 monitor="val_acc", mode="max", save_top_k=1, save_weights_only=False)
    history = History()
    trainer = Trainer(accelerator=accelerator, max_epochs=max_epochs, callbacks=[checkpoint, EarlyStopping(monitor="val_acc", patience=patience, mode="max"), history],
                      logger=False, enable_checkpointing=True, enable_progress_bar=False, enable_model_summary=False, log_every_n_steps=10)
    started = time.time()
    trainer.fit(model, datamodule=data)
    fit_seconds = time.time() - started
    trainer.test(model, datamodule=data, ckpt_path="best", verbose=False)           # best-validation checkpoint, fixed partition
    outputs = model.test_outputs_npz
    probabilities = softmax(np.asarray(outputs["predictions"], dtype=float))
    val_acc = [row["val_acc"] for row in history.rows]
    records = [{"id": i, "recording": recording_of(i), "label": int(t), "prediction": int(p.argmax()), "probs": [float(v) for v in p], "fold": fold}
               for i, t, p in zip(outputs["ids"], outputs["targets"], probabilities)]
    return {"records": records, "stop_epoch": int(trainer.current_epoch), "best_epoch": int(np.argmax(val_acc)), "val_acc_best": float(max(val_acc)),
            "test_loss": float(model.test_metrics_summary["test_loss"]), "fit_seconds": fit_seconds, "val_history": history.rows,
            "n_train": len(split["train"]), "n_val": len(split["val"]), "n_test": len(split["test"])}


# --------------------------------------------------------------------------------------------------------------- baselines
def baseline_predictions(kind: str, features: np.ndarray, labels: np.ndarray, split: Dict[str, List[int]], fold: int) -> List[Dict[str, object]]:
    train = np.array(split["train"] + split["val"])           # baselines use all training-fold recordings (no early stopping needed)
    test = np.array(split["test"])
    y_train = labels[train]
    prior = np.bincount(y_train, minlength=3) / len(y_train)
    if kind == "majority":
        probabilities = np.tile(prior, (len(test), 1))
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        flat = features.reshape(len(features), -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=1.0)).fit(flat[train], y_train)
        probabilities = np.zeros((len(test), 3))
        probabilities[:, classifier.classes_] = classifier.predict_proba(flat[test])
    return [{"index": int(i), "label": int(labels[i]), "prediction": int(p.argmax()), "probs": [float(v) for v in p], "fold": fold}
            for i, p in zip(test, probabilities)]


# ----------------------------------------------------------------------------------------------------------------- summary
def summarize(name: str, records: List[Dict[str, object]], bootstrap: int, seed: int) -> Dict[str, object]:
    y_true = np.array([r["label"] for r in records])
    y_pred = np.array([r["prediction"] for r in records])
    groups = [r["recording"] for r in records]
    probabilities = np.array([r["probs"] for r in records])
    correct = (y_true == y_pred).astype(float)
    cm = confusion_matrix(y_true, y_pred)
    report = class_report(cm)
    low, high = cluster_bootstrap_ci(groups, correct, bootstrap, seed)
    wilson = wilson_interval(int(correct.sum()), len(correct))
    recordings, rec_true, rec_pred = recording_level(groups, y_true, probabilities)
    rec_correct = (rec_true == rec_pred).astype(float)
    rec_low, rec_high = cluster_bootstrap_ci(list(recordings), rec_correct, bootstrap, seed)
    return {"method": name, "n_excerpts": len(records), "n_recordings": len(recordings), "accuracy": report["accuracy"],
            "accuracy_ci_cluster_low": low, "accuracy_ci_cluster_high": high, "accuracy_wilson_low": wilson[0], "accuracy_wilson_high": wilson[1],
            "balanced_accuracy": report["balanced_accuracy"], "macro_f1": report["macro_f1"],
            **{f"recall_{c['class']}": c["recall"] for c in report["per_class"]},
            "recording_accuracy": float(rec_correct.mean()), "recording_ci_low": rec_low, "recording_ci_high": rec_high,
            "confusion_matrix": cm.tolist()}


def write_csv(path: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])


# ------------------------------------------------------------------------------------------------------------ provenance
def hardware_description() -> Dict[str, object]:
    chip = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=10).stdout.strip() or chip
        except (OSError, subprocess.SubprocessError):
            pass
    return {"cpu": chip, "logical_cpus": os.cpu_count(), "load_average_1_5_15_min": [round(v, 2) for v in os.getloadavg()] if hasattr(os, "getloadavg") else None}


def rp_extract_description() -> Optional[Dict[str, object]]:
    """Location, SHA-256 and (if a git checkout) commit of the ``rp_extract`` module that was imported, or None when it is not installed."""
    module = sys.modules.get("rp_extract.rp_extract")
    path = getattr(module, "__file__", None)
    if not path:
        return None
    info: Dict[str, object] = {"file": path, "sha256": sha256_of(path)}
    try:
        info["git_commit"] = subprocess.run(["git", "-C", os.path.dirname(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        info["git_commit"] = None
    return info


SOURCE_FILES = ("main_recording_disjoint.py", "feature_cache.py", "splits.py", "evaluation.py", "data.py", "data_processing.py", "model.py", "WavKAN.py", "globals.py")


def source_hashes() -> Dict[str, str]:
    here = os.path.dirname(os.path.abspath(__file__))
    return {name: sha256_of(os.path.join(here, name)) for name in SOURCE_FILES if os.path.isfile(os.path.join(here, name))}


# ------------------------------------------------------------------------------------------------------- preparation / jobs
@dataclass
class Context:
    features: List[str]
    ids: List[str]
    groups: List[str]
    split_keys: List[str]
    labels: List[int]
    splits: List[Dict[str, List[int]]]
    datasets: Dict[str, CachedFeatureDataset]
    runs_dir: str
    partition_path: str
    built: List[str]
    grouping: str


def prepare(configs: Dict[str, dict], csv_path: str, audio_folder: str, cache_dir: str, out_dir: str, folds: int, seed: int, segment_duration: float,
            cache_workers: int, rebuild_cache: bool, grouping: str) -> Context:
    """Descriptor cache, K-fold partition (recording-level, or excerpt-level for the control) and the cached datasets."""
    if grouping not in GROUPINGS:
        raise ValueError(f"grouping must be one of {GROUPINGS}, got {grouping!r}")
    absent = missing_audio(csv_path, audio_folder)
    if absent:
        raise FileNotFoundError(f"{len(absent)} rows of {csv_path} have no WAV file in {audio_folder} (e.g. {absent[:3]}); the corpus would silently shrink. "
                                "Accented file names mis-decoded by an archive tool are the usual cause (see docs/PROTOCOL.md).")
    features = sorted({config["data"]["features"][0] for config in configs.values()})
    built = build_cache(csv_path, audio_folder, features, cache_dir, segment_duration, "pattern", cache_workers, rebuild_cache)
    index = load_index(cache_dir)
    reference = index_dataset(csv_path, audio_folder, segment_duration, "pattern")
    groups, labels = reference.partition_keys()
    ids = index["ids"]
    if ids != [reference.excerpt_id(i) for i in range(len(reference))]:
        raise ValueError("the descriptor cache does not match the dataset; rebuild it with --rebuild-cache")
    split_keys = grouping_keys(reference, grouping)

    # --- stratified K-fold, plus a validation part (same grouping) inside every training set -------------------------------------------
    # recording / excerpt groups hold one class each (greedy size-balanced assignment, splits.py); performer groups can hold several classes
    # (37 of the 142 performers of the released corpus do), which the mixed-class variant of the same greedy rule handles.
    kfold, partition = (mixed_group_kfold, mixed_group_partition) if grouping == "performer" else (grouped_kfold, grouped_partition)
    fold_parts = kfold(split_keys, labels, folds, seed)
    check_partition(fold_parts, split_keys, len(ids))
    splits = []
    for fold in range(folds):
        rest = sorted(i for other, part in enumerate(fold_parts) if other != fold for i in part)
        inner = partition([split_keys[i] for i in rest], [labels[i] for i in rest], (0.875, 0.125), seed + 1000 + fold)
        check_partition([inner[0], inner[1]], [split_keys[i] for i in rest], len(rest))
        splits.append({"train": [rest[j] for j in inner[0]], "val": [rest[j] for j in inner[1]], "test": list(fold_parts[fold])})
        if grouping != "excerpt":
            sets = [{groups[i] for i in splits[-1][name]} for name in ("train", "val", "test")]
            assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "recordings shared between parts"
    os.makedirs(out_dir, exist_ok=True)
    partition_path = os.path.join(out_dir, f"partition_k{folds}_seed{seed}.json")
    payload = {"folds": folds, "seed": seed, "grouping": grouping, "validation_share_of_training_groups": 0.125,
               "parts": [{name: sorted({split_keys[i] for i in split[name]}) for name in ("train", "val", "test")} for split in splits]}
    if os.path.isfile(partition_path):                                              # a re-start must see the same partition
        with open(partition_path) as handle:
            if json.load(handle) != json.loads(json.dumps(payload)):
                raise ValueError(f"{partition_path} differs from the partition computed now; remove it (and <out>/runs) or use another --out")
    else:
        with open(partition_path, "w") as handle:
            json.dump(payload, handle, indent=1)
    datasets = {feature: CachedFeatureDataset(cache_dir, feature, ids, labels) for feature in features}
    runs_dir = os.path.join(out_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    return Context(features, ids, groups, split_keys, labels, splits, datasets, runs_dir, partition_path, built, grouping)


def run_file_of(runs_dir: str, experiment_id: str, head: str, fold: int) -> str:
    return os.path.join(runs_dir, f"{safe_name(f'{experiment_id}|head={head}')}_fold{fold}.json")


def job_list(configs: Dict[str, dict], heads: Sequence[str], folds: int, by_cost: bool = False) -> List[Tuple[str, str, int]]:
    jobs = [(experiment_id, head, fold) for experiment_id in configs for head in heads for fold in range(folds)]
    if by_cost:                                                                     # longest first (better balance between workers)
        jobs.sort(key=lambda job: -configs[job[0]]["encoder"]["params"].get("out_chs", 16) * HEAD_COST[job[1]])
    return jobs


def train_missing(ctx: Context, configs: Dict[str, dict], jobs: Sequence[Tuple[str, str, int]], seed: int, checkpoint_dir: str, accelerator: str, max_epochs: int,
                  patience: int, batch_size: int, num_workers: int, claim_dir: Optional[str] = None, worker: str = "main") -> int:
    """Train every run of ``jobs`` that has no result file yet (and, in multi-process mode, that this process can claim); return how many were trained."""
    trained = 0
    for experiment_id, head, fold in jobs:
        run_file = run_file_of(ctx.runs_dir, experiment_id, head, fold)
        if os.path.isfile(run_file):
            continue
        if claim_dir is not None and not claim_job(claim_dir, os.path.basename(run_file)[:-5], worker):
            continue
        name = f"{experiment_id}|head={head}"
        started = utc_now()
        result = train_and_test(configs[experiment_id], head, ctx.datasets[configs[experiment_id]["data"]["features"][0]], ctx.splits[fold], fold, seed,
                                checkpoint_dir, name, accelerator, max_epochs, patience, batch_size, num_workers)
        result.update({"worker": worker, "device": accelerator, "threads": torch.get_num_threads(), "torch": torch.__version__,
                       "started_utc": started, "finished_utc": utc_now(), "grouping": ctx.grouping})
        write_json_atomic(run_file, result)
        trained += 1
        print(f"[{worker}] [{name}] fold {fold + 1}/{len(ctx.splits)}: held-out accuracy "
              f"{np.mean([r['label'] == r['prediction'] for r in result['records']]):.4f} (best epoch {result['best_epoch']}, stop {result['stop_epoch']}, "
              f"{result['fit_seconds']:.0f} s)", flush=True)
    return trained


def spawn_workers(parameters: Dict[str, object], configs_path: str, all_run_files: List[str], workers: int, threads: int, out_dir: str, retries: int) -> None:
    """Train the missing runs in ``workers`` child processes (each re-executes this script in worker mode); retry runs lost to a crashed worker."""
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    claim_dir = os.path.join(out_dir, "claims")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    if threads > 0:
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
            environment[variable] = str(threads)
    for attempt in range(retries + 1):
        missing = [path for path in all_run_files if not os.path.isfile(path)]
        if not missing:
            return
        shutil.rmtree(claim_dir, ignore_errors=True)                                # no worker is alive here: every old claim is stale
        os.makedirs(claim_dir)
        print(f"[parent] attempt {attempt + 1}: {len(missing)} of {len(all_run_files)} runs to train with {min(workers, len(missing))} worker process(es) "
              f"x {threads or 'default'} thread(s)", flush=True)
        children = []
        for index in range(min(workers, len(missing))):
            command = [sys.executable, "-B", os.path.abspath(__file__), "--configs-json", configs_path, "--worker-index", str(index), "--claim-dir", claim_dir,
                       "--workers", "1", "--csv", str(parameters["csv_path"]), "--audio", str(parameters["audio_folder"]), "--cache-dir", str(parameters["cache_dir"]),
                       "--out", out_dir, "--checkpoints", str(parameters["checkpoint_dir"]), "--folds", str(parameters["folds"]), "--seed", str(parameters["seed"]),
                       "--heads", *parameters["heads"], "--segment-duration", str(parameters["segment_duration"]), "--accelerator", str(parameters["accelerator"]),
                       "--max-epochs", str(parameters["max_epochs"]), "--patience", str(parameters["patience"]), "--batch-size", str(parameters["batch_size"]),
                       "--num-workers", str(parameters["num_workers"]), "--grouping", str(parameters["grouping"]), "--threads", str(threads)]
            if parameters["no_nnpack"]:
                command.append("--no-nnpack")
            log = open(os.path.join(logs_dir, f"worker_{index}_attempt{attempt + 1}.log"), "a")
            children.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment), log))
        last_report = time.time()
        while any(child.poll() is None for child, _ in children):
            time.sleep(5)
            if time.time() - last_report >= 600:
                print(f"[parent] {utc_now()}: {sum(os.path.isfile(p) for p in all_run_files)} of {len(all_run_files)} runs finished", flush=True)
                last_report = time.time()
        codes = [child.returncode for child, _ in children]
        for _, log in children:
            log.close()
        print(f"[parent] workers exited with codes {codes}", flush=True)
    still_missing = [os.path.basename(p) for p in all_run_files if not os.path.isfile(p)]
    if still_missing:
        raise RuntimeError(f"{len(still_missing)} runs could not be trained after {retries + 1} attempts (see {logs_dir}); first: {still_missing[:3]}")


# ------------------------------------------------------------------------------------------------------------------ driver
def run_recording_disjoint(configs: Dict[str, dict], csv_path: str, audio_folder: str, cache_dir: str, out_dir: str, checkpoint_dir: str,
                           folds: int = 5, seed: int = SEED, heads: Sequence[str] = ("kan",), baselines: bool = True, segment_duration: float = 10.0,
                           accelerator: str = "auto", max_epochs: int = 50, patience: int = 20, batch_size: int = BATCH_SIZE, num_workers: int = 0,
                           bootstrap: int = 2000, rebuild_cache: bool = False, resume: bool = True, command: str = "", workers: int = 1, threads: int = 0,
                           no_nnpack: bool = False, grouping: str = "recording", cache_workers: Optional[int] = None, worker_index: Optional[int] = None,
                           claim_dir: Optional[str] = None, retries: int = 2, notes: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    started = time.time()
    started_utc = utc_now()
    load_at_start = hardware_description()["load_average_1_5_15_min"]
    accelerator = resolve_accelerator(accelerator)
    configure_cpu(threads, no_nnpack)
    ctx = prepare(configs, csv_path, audio_folder, cache_dir, out_dir, folds, seed, segment_duration,
                  num_workers if cache_workers is None else cache_workers, rebuild_cache, grouping)
    train_args = (seed, checkpoint_dir, accelerator, max_epochs, patience, batch_size, num_workers)
    if worker_index is not None:                                                    # child process: train the runs it can claim, then stop
        logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
        warnings.filterwarnings("ignore", message=".*does not have many workers.*")
        trained = train_missing(ctx, configs, job_list(configs, heads, folds, by_cost=True), *train_args, claim_dir=claim_dir, worker=f"worker{worker_index}")
        return {"trained": trained}
    if not resume:
        shutil.rmtree(ctx.runs_dir, ignore_errors=True)
        os.makedirs(ctx.runs_dir)
    jobs = job_list(configs, heads, folds)
    if workers > 1:
        configs_path = os.path.join(out_dir, "configs.json")
        with open(configs_path, "w") as handle:
            json.dump(configs, handle, indent=1)
        parameters = {"csv_path": csv_path, "audio_folder": audio_folder, "cache_dir": cache_dir, "checkpoint_dir": checkpoint_dir, "folds": folds, "seed": seed,
                      "heads": list(heads), "segment_duration": segment_duration, "accelerator": accelerator, "max_epochs": max_epochs, "patience": patience,
                      "batch_size": batch_size, "num_workers": num_workers, "grouping": grouping, "no_nnpack": no_nnpack}
        spawn_workers(parameters, configs_path, [run_file_of(ctx.runs_dir, *job) for job in jobs], workers, threads, out_dir, retries)
    else:
        train_missing(ctx, configs, jobs, *train_args)
    labels_array = np.array(ctx.labels)
    methods: Dict[str, List[Dict[str, object]]] = {}
    fold_rows: List[List[object]] = []
    run_metadata: List[Dict[str, object]] = []

    def remember(name: str, fold: int, records, extra: Dict[str, object]) -> None:
        methods.setdefault(name, []).extend(records)
        accuracy = float(np.mean([r["label"] == r["prediction"] for r in records]))
        fold_rows.append([name, fold, len(records), len({r["recording"] for r in records}), accuracy, extra.get("test_loss", ""), extra.get("best_epoch", ""),
                          extra.get("stop_epoch", ""), extra.get("val_acc_best", ""), extra.get("fit_seconds", "")])

    for experiment_id, config in configs.items():
        for head in heads:
            name = f"{experiment_id}|head={head}"
            for fold in range(folds):
                with open(run_file_of(ctx.runs_dir, experiment_id, head, fold)) as handle:
                    result = json.load(handle)
                remember(name, fold, result["records"], result)
                run_metadata.append({"run": name, "fold": fold, **{k: result.get(k) for k in ("device", "threads", "worker", "torch", "started_utc", "finished_utc",
                                                                                              "fit_seconds", "stop_epoch", "best_epoch")}})
    if baselines:
        for feature in ctx.features:
            array = np.load(cache_file(cache_dir, feature), mmap_mode="r")
            for kind in ("majority", "logreg"):
                name = f"baseline:{kind}|descriptor={feature}"
                for fold, split in enumerate(ctx.splits):
                    records = baseline_predictions(kind, np.asarray(array), labels_array, split, fold)
                    for r in records:
                        r["id"] = ctx.ids[r.pop("index")]
                        r["recording"] = recording_of(r["id"])
                    remember(name, fold, records, {})
                print(f"[{name}] done", flush=True)

    summaries = [summarize(name, records, bootstrap, seed) for name, records in methods.items()]
    columns = ["method", "n_excerpts", "n_recordings", "accuracy", "accuracy_ci_cluster_low", "accuracy_ci_cluster_high", "accuracy_wilson_low",
               "accuracy_wilson_high", "balanced_accuracy", "macro_f1", "recall_0", "recall_1", "recall_2", "recording_accuracy", "recording_ci_low", "recording_ci_high"]
    write_csv(os.path.join(out_dir, "summary.csv"), columns, [[s[c] for c in columns] for s in summaries])
    write_csv(os.path.join(out_dir, "folds.csv"), ["method", "fold", "n_test_excerpts", "n_test_recordings", "test_accuracy", "test_loss", "best_epoch",
                                                  "stop_epoch", "val_acc_best", "fit_seconds"], fold_rows)
    with open(os.path.join(out_dir, "confusion_matrices.json"), "w") as handle:
        json.dump({s["method"]: s["confusion_matrix"] for s in summaries}, handle, indent=1)
    for name, records in methods.items():
        write_csv(os.path.join(out_dir, "predictions", f"{safe_name(name)}.csv"), ["id", "recording", "label", "prediction", "prob_0", "prob_1", "prob_2", "fold"],
                  [[r["id"], r["recording"], r["label"], r["prediction"], *r["probs"], r["fold"]] for r in records])
    outputs = sorted(os.path.join(dp, f) for dp, _, files in os.walk(out_dir) for f in files
                     if not {"runs", "logs", "claims"} & set(os.path.relpath(dp, out_dir).split(os.sep)))
    with open(ctx.partition_path) as handle:
        partition = json.load(handle)
    provenance = {"generated_utc": utc_now(), "started_utc": started_utc, "command": command,
                  "NEW_RUN": {"recording": "recording-disjoint re-evaluation; report next to, never in place of, the stored 2025 results",
                              "performer": "CONTROL: performer-disjoint folds (stricter than recording-disjoint; all recordings of a performer are held out together)",
                              "excerpt": "CONTROL: excerpt-level folds (excerpts of one recording in training and test); not a recording-disjoint estimate"}[grouping],
                  "parameters": {"folds": folds, "seed": seed, "grouping": grouping, "heads": list(heads), "baselines": baselines, "segment_duration": segment_duration,
                                 "max_epochs": max_epochs, "patience": patience, "batch_size": batch_size, "accelerator": accelerator, "bootstrap": bootstrap,
                                 "workers": workers, "threads_per_worker": threads or torch.get_num_threads(), "no_nnpack": no_nnpack},
                  "configurations": list(configs), "descriptors_cached_now": ctx.built,
                  "partition_summary": {"excerpts": len(ctx.ids), "recordings": len(set(ctx.groups)), "classes": {int(c): int((labels_array == c).sum()) for c in sorted(set(ctx.labels))},
                                        "test_excerpts_per_fold": [len(s["test"]) for s in ctx.splits], "val_excerpts_per_fold": [len(s["val"]) for s in ctx.splits],
                                        "train_excerpts_per_fold": [len(s["train"]) for s in ctx.splits], "test_groups_per_fold": [len(p["test"]) for p in partition["parts"]]},
                  "inputs": {"dataset_csv": {"path": csv_path, "sha256": sha256_of(csv_path)}, "cache_index": {"path": os.path.join(cache_dir, "index.json"),
                            "sha256": sha256_of(os.path.join(cache_dir, "index.json"))},
                            "descriptor_caches": {f: sha256_of(cache_file(cache_dir, f)) for f in ctx.features}},
                  "environment": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
                                  "hardware": hardware_description(), "load_average_at_start": load_at_start},
                  "code_sha256": source_hashes(), "rp_extract": rp_extract_description(), "notes": notes or {},
                  "runs": run_metadata, "wall_clock_seconds": time.time() - started,
                  "outputs": {os.path.relpath(p, out_dir): sha256_of(p) for p in outputs if not p.endswith("PROVENANCE.json")}}
    with open(os.path.join(out_dir, "PROVENANCE.json"), "w") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"summaries": summaries, "partition": ctx.partition_path, "out_dir": out_dir}


# -------------------------------------------------------------------------------------------------------- runtime benchmark
def benchmark(config_ids: Sequence[str], folds: int, heads: Sequence[str], steps: int, batch_size: int, pool: int = 5605,
              stored_stop_epochs: Optional[Sequence[int]] = None, repeats: int = 3) -> Dict[str, object]:
    """Dataset-free timing of training steps on random tensors of the real descriptor shapes (CPU), extrapolated to the full run."""
    torch.manual_seed(0)
    train_excerpts = int(pool * (1 - 1 / folds) * 0.875)
    steps_per_epoch = math.ceil(train_excerpts / batch_size)
    typical_epochs = float(np.mean(stored_stop_epochs)) if stored_stop_epochs else 34.0
    rows, total_typical, total_max = [], 0.0, 0.0
    for experiment_id in config_ids:
        config = load_configuration(experiment_id)
        shape = INPUT_SHAPES[config["data"]["features"][0]]
        for head in heads:
            cfg = with_head(config, head)
            encoder, head_module = init_modules(cfg, torch.randn(2, 1, *shape))
            model = Model(encoder_module=encoder, kan_module=head_module, downstream_task="pattern", lr=LR, config=cfg)
            optimizer = model.configure_optimizers()
            x, y = torch.randn(batch_size, 1, *shape), torch.randint(0, 3, (batch_size,))
            model.train()
            timings = []
            for repeat in range(repeats + 1):                                  # first block = warm-up, not timed
                started = time.perf_counter()
                for _ in range(steps):
                    optimizer.zero_grad()
                    model.loss_fn(model(x), y).backward()
                    optimizer.step()
                if repeat:
                    timings.append((time.perf_counter() - started) / steps)
            seconds_per_step = min(timings)                                    # least-disturbed block (other processes inflate timings)
            epoch_seconds = seconds_per_step * steps_per_epoch * 1.08          # + validation pass (about 8 % of a training epoch)
            typical, maximum = folds * typical_epochs * epoch_seconds, folds * 50 * epoch_seconds
            total_typical, total_max = total_typical + typical, total_max + maximum
            rows.append({"configuration": experiment_id, "head": head, "seconds_per_step": seconds_per_step, "seconds_per_epoch": epoch_seconds,
                         "hours_typical": typical / 3600, "hours_max": maximum / 3600})
    return {"rows": rows, "steps_per_epoch": steps_per_epoch, "train_excerpts_per_fold": train_excerpts, "typical_epochs": typical_epochs,
            "hours_typical": total_typical / 3600, "hours_max": total_max / 3600, "threads": torch.get_num_threads(), "torch": torch.__version__}


# ------------------------------------------------------------------------------------------------------------------- CLI
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", help="experiment ids (file stems of experiments/configs/pattern); default: stored runs above --min-test-accuracy")
    parser.add_argument("--min-test-accuracy", type=float, default=0.90, help="select the stored Stage-2 runs whose legacy test accuracy exceeds this (default 0.90: nine runs)")
    parser.add_argument("--all-stored", action="store_true", help="use all 14 stored Stage-2 configurations")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED, help="seed of the fold partition (and base seed of the training runs)")
    parser.add_argument("--grouping", choices=GROUPINGS, default="recording",
                        help="recording (default): folds hold out whole recordings; performer: CONTROL, folds hold out all recordings of a performer (stricter); "
                             "excerpt: CONTROL with excerpt-level folds (siblings of a held-out excerpt are in training; looser)")
    parser.add_argument("--heads", nargs="+", choices=sorted(HEAD_OVERRIDES), default=["kan"], help="kan = stored WavKAN head; mlp / linear = ablation heads on the same encoder")
    parser.add_argument("--no-baselines", action="store_true", help="skip the majority-class and logistic-regression baselines")
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--audio", default=AUDIO_FOLDER)
    parser.add_argument("--segment-duration", type=float, default=10.0)
    parser.add_argument("--cache-dir", default="./dataset/feature_cache")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--out", default="./results/recording_disjoint")
    parser.add_argument("--checkpoints", default=os.path.join(MAIN_FOLDER, "checkpoints", "recording_disjoint"))
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto", help="auto = CUDA if available, else CPU (never MPS)")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0, help="data-loader workers while training (0: the cached descriptors are read in the main process)")
    parser.add_argument("--cache-workers", type=int, default=None, help="processes used to compute the descriptor cache (default: --num-workers)")
    parser.add_argument("--workers", type=int, default=1, help="training processes working on different runs at the same time (default 1: sequential)")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads per training process (0: PyTorch default); with --workers N use about cores / N")
    parser.add_argument("--no-nnpack", action="store_true", help="switch off the NNPACK convolution path of the CPU build of PyTorch (much faster for the 1D encoders on Apple silicon)")
    parser.add_argument("--retries", type=int, default=2, help="with --workers: how many times to restart workers if runs are missing after they exit")
    parser.add_argument("--bootstrap", type=int, default=2000, help="recording-level bootstrap resamples")
    parser.add_argument("--no-resume", action="store_true", help="retrain even if a finished run file exists in <out>/runs")
    parser.add_argument("--note", action="append", default=[], metavar="KEY=VALUE", help="free-text provenance entries stored in PROVENANCE.json (repeatable)")
    parser.add_argument("--benchmark", action="store_true", help="dataset-free timing on random tensors and runtime estimate; trains nothing on data")
    parser.add_argument("--benchmark-steps", type=int, default=20, help="training steps per timed block (3 blocks, the fastest is used)")
    parser.add_argument("--configs-json", default=None, help=argparse.SUPPRESS)           # internal: configurations written by the parent of --workers
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)  # internal: child process of --workers
    parser.add_argument("--claim-dir", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.configs_json:
        with open(args.configs_json) as handle:
            configs = json.load(handle)
        config_ids = list(configs)
    else:
        if args.configs:
            config_ids = args.configs
        elif args.all_stored:
            config_ids = stored_configuration_ids(-1.0)
        else:
            config_ids = stored_configuration_ids(args.min_test_accuracy)
        configs = None
    configure_cpu(args.threads, args.no_nnpack)
    if args.benchmark:
        with open(STORED_TRAINING_RESULTS, newline="") as handle:
            stops = [int(r["stopped_epoch"]) for r in csv.DictReader(handle)]
        result = benchmark(config_ids, args.folds, args.heads, args.benchmark_steps, args.batch_size, stored_stop_epochs=stops)
        for row in result["rows"]:
            print(f"{row['configuration']:62s} head={row['head']:6s} {row['seconds_per_epoch']:6.1f} s/epoch  typical {row['hours_typical']:.2f} h  max {row['hours_max']:.2f} h")
        print(f"[benchmark] {len(config_ids)} configurations x {args.folds} folds x {len(args.heads)} head(s), {result['steps_per_epoch']} steps/epoch "
              f"({result['train_excerpts_per_fold']} training excerpts), {result['threads']} CPU threads of {os.cpu_count()}, torch {result['torch']}, "
              f"load average at the end {os.getloadavg()[0]:.1f} (timings are inflated when other jobs share the machine)")
        print(f"[benchmark] estimated training time (one process, sequential): typical ({result['typical_epochs']:.1f} epochs per run, the mean stored stop epoch) "
              f"{result['hours_typical']:.1f} h; upper bound (50 epochs per run) {result['hours_max']:.1f} h; plus descriptor cache and baselines")
        return 0
    if configs is None:
        configs = {i: load_configuration(i) for i in config_ids}
    command = "python main_recording_disjoint.py " + " ".join(sys.argv[1:] if argv is None else argv)
    notes = dict(item.split("=", 1) for item in args.note if "=" in item)
    result = run_recording_disjoint(configs, args.csv, args.audio, args.cache_dir, args.out, args.checkpoints, args.folds, args.seed, args.heads, not args.no_baselines,
                                    args.segment_duration, args.accelerator, args.max_epochs, args.patience, args.batch_size, args.num_workers, args.bootstrap,
                                    args.rebuild_cache, not args.no_resume, command, args.workers, args.threads, args.no_nnpack, args.grouping, args.cache_workers,
                                    args.worker_index, args.claim_dir, args.retries, notes)
    if args.worker_index is not None:
        return 0
    for summary in result["summaries"]:
        print(f"{summary['method']:80s} accuracy {summary['accuracy']:.4f} [{summary['accuracy_ci_cluster_low']:.3f}, {summary['accuracy_ci_cluster_high']:.3f}]  "
              f"recording-level {summary['recording_accuracy']:.3f}  macro-F1 {summary['macro_f1']:.3f}")
    print(f"[done] results in {result['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
