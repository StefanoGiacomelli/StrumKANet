"""Tables for a recording-disjoint re-evaluation (outputs of ``main_recording_disjoint.py``): intervals, paired comparisons, stored values.

Usage (from any directory)::

    python analysis/recording_disjoint_report.py --run results/recording_disjoint_2026/primary_seed42 \
        --control-excerpt results/recording_disjoint_2026/control_excerpt_seed42 \
        --control-performer results/recording_disjoint_2026/control_performer_seed42 \
        --repeat results/recording_disjoint_2026/recording_seed43 --repeat results/recording_disjoint_2026/recording_seed44 \
        [--out results/recording_disjoint_2026/report]

``--run`` is the recording-disjoint evaluation; the two controls (same pipeline, other grouping unit) and the repeats (other partition seeds) are
optional. For every run directory the script reads ``summary.csv``, ``folds.csv`` and ``predictions/*.csv`` (or ``*.csv.gz``) and writes, into
``--out`` (default ``<run>/report``):

* ``methods.csv`` (and ``methods_control_excerpt.csv`` / ``methods_control_performer.csv``): one row per method: excerpt accuracy and
  recording-level accuracy with recording-level (cluster) bootstrap intervals, balanced accuracy, macro-F1, per-class recall, fold-to-fold
  mean / sd / min / max of the held-out accuracy and, for the stored configurations, the 2025 test accuracy and best validation accuracy
  (``results/stage2_table.csv``);
* ``families.csv``: mean held-out accuracy over the configurations of one head and descriptor, per protocol (selection-free: no configuration is
  picked by its held-out accuracy), with a cluster-bootstrap interval;
* ``ladder.csv``: for every configuration with the WavKAN head and for the baselines, the accuracy under the stored in-corpus test, the stored
  validation, the excerpt-level folds, the recording-disjoint folds and the performer-disjoint folds, with the paired differences between protocols;
* ``paired.csv``: accuracy differences between methods that cover the same excerpts, with paired cluster-bootstrap 95 % intervals (WavKAN head
  against the logistic-regression and majority baselines, against the linear and MLP heads, per configuration and as a mean over the RP
  configurations; recording-disjoint minus excerpt-level and performer-disjoint folds for the same model);
* ``repeats.csv``: with ``--repeat``: the accuracy of every common method for every partition seed, and its mean;
* ``confusion_long.csv``: pooled confusion counts and row shares of every method and protocol;
* ``report.json``: the headline numbers used in the documentation, with the hashes of the inputs.

``--write-curves`` additionally writes ``<run>/validation_curves.csv`` (per-epoch validation accuracy and loss of every run) when ``<run>/runs`` exists.
Everything is deterministic (seeded bootstrap). Nothing under ``experiments/`` is modified.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.common import DERIVED_RESULTS, sha256_file, write_json    # noqa: E402

BOOTSTRAP = 4000
PROTOCOLS = ("recording", "excerpt", "performer")


def open_text(path: Path):
    return gzip.open(path, "rt", newline="") if str(path).endswith(".gz") else open(path, newline="")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open_text(path) as handle:
        return list(csv.DictReader(handle))


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", name).strip("_")


class RunDirectory:
    """Outputs of one ``main_recording_disjoint.py`` invocation."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.summary = {row["method"]: row for row in read_csv(self.path / "summary.csv")}
        self.folds: Dict[str, List[Dict[str, str]]] = {}
        for row in read_csv(self.path / "folds.csv"):
            self.folds.setdefault(row["method"], []).append(row)
        with open(self.path / "PROVENANCE.json") as handle:
            self.provenance = json.load(handle)
        with open(self.path / "confusion_matrices.json") as handle:
            self.confusion = json.load(handle)
        self._predictions: Dict[str, List[Tuple[str, int, int]]] = {}

    @property
    def methods(self) -> List[str]:
        return list(self.summary)

    @property
    def grouping(self) -> str:
        return self.provenance["parameters"].get("grouping", "recording")

    def predictions(self, method: str) -> List[Tuple[str, int, int]]:
        """(recording, label, prediction) of every held-out excerpt, as a list: the two repeated rows of the corpus give the same excerpt id twice."""
        if method not in self._predictions:
            stem = self.path / "predictions" / safe(method)
            path = Path(f"{stem}.csv") if Path(f"{stem}.csv").is_file() else Path(f"{stem}.csv.gz")
            self._predictions[method] = [(r["recording"], int(r["label"]), int(r["prediction"])) for r in read_csv(path)]
        return self._predictions[method]

    def correct_by_recording(self, method: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Recordings (sorted), correct predictions per recording, excerpts per recording."""
        by_recording: Dict[str, List[int]] = {}
        for recording, label, prediction in self.predictions(method):
            by_recording.setdefault(recording, []).append(int(label == prediction))
        keys = sorted(by_recording)
        return keys, np.array([sum(by_recording[k]) for k in keys], dtype=float), np.array([len(by_recording[k]) for k in keys], dtype=float)


def split_method(method: str) -> Tuple[str, str]:
    """('<experiment id>', 'head=kan') or ('baseline:logreg', 'descriptor=rp')."""
    left, right = method.rsplit("|", 1)
    return left, right


def descriptor_of(experiment: str) -> str:
    return "onset_strength" if experiment.endswith("onset_strength") else experiment.rsplit("_", 1)[-1]


# ------------------------------------------------------------------------------------------------------------ bootstrap
def _stack(run: RunDirectory, methods: Sequence[str]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    keys, counts, rows = None, None, []
    for method in methods:
        method_keys, correct, count = run.correct_by_recording(method)
        if keys is None:
            keys, counts = method_keys, count
        elif method_keys != keys or not np.array_equal(count, counts):
            raise ValueError(f"{method} does not cover the same excerpts")
        rows.append(correct)
    return keys, np.array(rows), counts


def mean_accuracy_draws(corrects: np.ndarray, counts: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Mean over methods of the accuracy of every bootstrap resample (``corrects``: methods x recordings)."""
    total = counts[draws].sum(axis=1)
    return np.mean([row[draws].sum(axis=1) / total for row in corrects], axis=0)


def family_difference(a: RunDirectory, methods_a: Sequence[str], b: RunDirectory, methods_b: Sequence[str], resamples: int = BOOTSTRAP, seed: int = 7) -> Dict[str, float]:
    """mean_accuracy(A methods) - mean_accuracy(B methods) with a paired cluster bootstrap (the same resampled recordings for both sides).

    One method on each side gives the plain paired difference of two models; several methods give the difference of the means over configurations."""
    keys_a, corrects_a, counts_a = _stack(a, methods_a)
    keys_b, corrects_b, counts_b = _stack(b, methods_b)
    if keys_a != keys_b or not np.array_equal(counts_a, counts_b):
        raise ValueError("the two sides do not cover the same excerpts")
    draws = np.random.default_rng(seed).integers(0, len(keys_a), size=(resamples, len(keys_a)))
    differences = mean_accuracy_draws(corrects_a, counts_a, draws) - mean_accuracy_draws(corrects_b, counts_b, draws)
    observed = float(np.mean(corrects_a.sum(axis=1) / counts_a.sum()) - np.mean(corrects_b.sum(axis=1) / counts_b.sum()))
    low, high = float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5))
    return {"difference": observed, "ci_low": low, "ci_high": high, "ci_excludes_zero": bool(low > 0 or high < 0),
            "share_of_resamples_not_positive": float((differences <= 0).mean())}


def family_interval(run: RunDirectory, methods: Sequence[str], resamples: int = BOOTSTRAP, seed: int = 11) -> Dict[str, object]:
    """Mean held-out accuracy over several methods with a cluster-bootstrap interval: the same resampled recordings are used for every
    method, so the interval reflects the recordings, not the choice of configuration."""
    keys, corrects, counts = _stack(run, methods)
    draws = np.random.default_rng(seed).integers(0, len(keys), size=(resamples, len(keys)))
    estimates = mean_accuracy_draws(corrects, counts, draws)
    per_method = corrects.sum(axis=1) / counts.sum()
    return {"mean_accuracy": float(per_method.mean()), "ci_low": float(np.percentile(estimates, 2.5)), "ci_high": float(np.percentile(estimates, 97.5)),
            "min_accuracy": float(per_method.min()), "max_accuracy": float(per_method.max()), "n_methods": len(methods)}


def families(run: RunDirectory) -> Dict[Tuple[str, str], List[str]]:
    groups: Dict[Tuple[str, str], List[str]] = {}
    for method in run.methods:
        if method.startswith("baseline:"):
            continue
        experiment, head = split_method(method)
        groups.setdefault((head, descriptor_of(experiment)), []).append(method)
    return groups


# --------------------------------------------------------------------------------------------------------------- tables
def stored_values(root: Path) -> Dict[str, Dict[str, str]]:
    path = root / "results" / "stage2_table.csv"
    return {row["experiment_id"]: row for row in read_csv(path)} if path.is_file() else {}


def methods_table(run: RunDirectory, stored: Dict[str, Dict[str, str]]) -> List[Dict[str, object]]:
    rows = []
    for method, summary in run.summary.items():
        fold_accuracy = [float(r["test_accuracy"]) for r in run.folds.get(method, [])]
        experiment, extra = split_method(method)
        published = stored.get(experiment, {})
        rows.append({"method": method, "kind": "baseline" if experiment.startswith("baseline:") else extra,
                     "accuracy": float(summary["accuracy"]), "accuracy_ci_low": float(summary["accuracy_ci_cluster_low"]), "accuracy_ci_high": float(summary["accuracy_ci_cluster_high"]),
                     "recording_accuracy": float(summary["recording_accuracy"]), "recording_ci_low": float(summary["recording_ci_low"]), "recording_ci_high": float(summary["recording_ci_high"]),
                     "balanced_accuracy": float(summary["balanced_accuracy"]), "macro_f1": float(summary["macro_f1"]),
                     "recall_0": float(summary["recall_0"]), "recall_1": float(summary["recall_1"]), "recall_2": float(summary["recall_2"]),
                     "fold_mean": statistics.mean(fold_accuracy) if fold_accuracy else "", "fold_sd": statistics.stdev(fold_accuracy) if len(fold_accuracy) > 1 else "",
                     "fold_min": min(fold_accuracy) if fold_accuracy else "", "fold_max": max(fold_accuracy) if fold_accuracy else "",
                     "stored_test_accuracy": float(published["test_accuracy"]) if published else "", "stored_val_acc_best": float(published["val_acc_best"]) if published else "",
                     "stored_val_acc_last": float(published["val_acc_last"]) if published else ""})
    return rows


def families_table(runs: Dict[str, RunDirectory]) -> List[Dict[str, object]]:
    rows = []
    for protocol, run in runs.items():
        for (head, descriptor), methods in sorted(families(run).items()):
            rows.append({"protocol": protocol, "head": head, "descriptor": descriptor, **family_interval(run, methods)})
    return rows


def paired_table(runs: Dict[str, RunDirectory]) -> List[Dict[str, object]]:
    primary = runs["recording"]
    rows: List[Dict[str, object]] = []

    def add(contrast: str, a: RunDirectory, methods_a: Sequence[str], b: RunDirectory, methods_b: Sequence[str]) -> None:
        try:
            result = family_difference(a, methods_a, b, methods_b)
        except (KeyError, ValueError, FileNotFoundError):
            return
        rows.append({"contrast": contrast, "method_a": methods_a[0] if len(methods_a) == 1 else f"mean of {len(methods_a)} configurations",
                     "method_b": methods_b[0] if len(methods_b) == 1 else f"mean of {len(methods_b)} configurations", **result})

    models = [m for m in primary.methods if not m.startswith("baseline:")]
    experiments = sorted({split_method(m)[0] for m in models})
    for experiment in experiments:                                                      # per configuration
        descriptor = descriptor_of(experiment)
        kan = f"{experiment}|head=kan"
        if kan in primary.summary:
            add("WavKAN head - logistic regression (same descriptor)", primary, [kan], primary, [f"baseline:logreg|descriptor={descriptor}"])
            add("WavKAN head - majority class", primary, [kan], primary, [f"baseline:majority|descriptor={descriptor}"])
            for other in ("linear", "mlp"):
                if f"{experiment}|head={other}" in primary.summary:
                    add(f"WavKAN head - {other} head (same encoder)", primary, [kan], primary, [f"{experiment}|head={other}"])
        if f"{experiment}|head=mlp" in primary.summary and f"{experiment}|head=linear" in primary.summary:
            add("MLP head - linear head (same encoder)", primary, [f"{experiment}|head=mlp"], primary, [f"{experiment}|head=linear"])
    groups = families(primary)                                                          # mean over the configurations of one descriptor
    for descriptor in ("rp",):
        kan = groups.get(("head=kan", descriptor), [])
        if kan:
            add(f"WavKAN head - logistic regression, mean over the {len(kan)} {descriptor.upper()} configurations", primary, kan, primary, [f"baseline:logreg|descriptor={descriptor}"])
            add(f"WavKAN head - majority class, mean over the {len(kan)} {descriptor.upper()} configurations", primary, kan, primary, [f"baseline:majority|descriptor={descriptor}"])
            for other in ("mlp", "linear"):
                others = groups.get((f"head={other}", descriptor), [])
                if len(others) == len(kan):
                    add(f"WavKAN head - {other} head, mean over the {len(kan)} {descriptor.upper()} configurations", primary, kan, primary, others)
            mlp, linear = groups.get(("head=mlp", descriptor), []), groups.get(("head=linear", descriptor), [])
            if mlp and len(mlp) == len(linear):
                add(f"MLP head - linear head, mean over the {len(mlp)} {descriptor.upper()} configurations", primary, mlp, primary, linear)
            for other, other_run in ((p, runs[p]) for p in ("excerpt", "performer") if p in runs):
                present = [m for m in kan if m in other_run.summary]
                if present:
                    label = {"excerpt": "excerpt-level folds", "performer": "performer-disjoint folds"}[other]
                    add(f"recording-disjoint - {label}, WavKAN head, mean over the {len(present)} {descriptor.upper()} configurations", primary, present, other_run, present)
    kan_os = groups.get(("head=kan", "onset_strength"), [])
    for method in kan_os:
        add("WavKAN head, onset strength - majority class", primary, [method], primary, ["baseline:majority|descriptor=onset_strength"])
        add("WavKAN head, onset strength - logistic regression", primary, [method], primary, ["baseline:logreg|descriptor=onset_strength"])
    for protocol, other_run in ((p, runs[p]) for p in ("excerpt", "performer") if p in runs):          # per model and baseline across protocols
        label = {"excerpt": "excerpt-level folds", "performer": "performer-disjoint folds"}[protocol]
        for method in primary.methods:
            if method in other_run.summary and (method.startswith("baseline:") or method.endswith("|head=kan")):
                add(f"recording-disjoint - {label} (same {'baseline' if method.startswith('baseline:') else 'model'})", primary, [method], other_run, [method])
    return rows


def ladder_table(runs: Dict[str, RunDirectory], stored: Dict[str, Dict[str, str]]) -> List[Dict[str, object]]:
    primary = runs["recording"]
    names = [m for m in primary.methods if m.endswith("|head=kan") or m.startswith("baseline:")]
    rows = []
    for method in sorted(names, key=lambda m: (m.startswith("baseline:"), -float(stored.get(split_method(m)[0], {}).get("test_accuracy", 0)))):
        experiment, extra = split_method(method)
        row: Dict[str, object] = {"method": method, "stored_test_accuracy_2025": float(stored[experiment]["test_accuracy"]) if experiment in stored else "",
                                  "stored_val_acc_best_2025": float(stored[experiment]["val_acc_best"]) if experiment in stored else ""}
        for protocol in PROTOCOLS:
            run_p = runs.get(protocol)
            summary = run_p.summary.get(method) if run_p else None
            row[f"{protocol}_accuracy"] = float(summary["accuracy"]) if summary else ""
            row[f"{protocol}_ci_low"] = float(summary["accuracy_ci_cluster_low"]) if summary else ""
            row[f"{protocol}_ci_high"] = float(summary["accuracy_ci_cluster_high"]) if summary else ""
            row[f"{protocol}_recording_level_accuracy"] = float(summary["recording_accuracy"]) if summary else ""
        rows.append(row)
    return rows


def repeats_table(runs: Sequence[RunDirectory]) -> List[Dict[str, object]]:
    rows = []
    common = [m for m in runs[0].methods if all(m in r.summary for r in runs)]
    for method in common:
        accuracies = [float(r.summary[method]["accuracy"]) for r in runs]
        recordings = [float(r.summary[method]["recording_accuracy"]) for r in runs]
        row: Dict[str, object] = {"method": method}
        for run, accuracy in zip(runs, accuracies):
            row[f"accuracy_seed{run.provenance['parameters']['seed']}"] = accuracy
        row.update({"accuracy_mean": statistics.mean(accuracies), "accuracy_sd": statistics.stdev(accuracies) if len(accuracies) > 1 else "",
                    "recording_accuracy_mean": statistics.mean(recordings)})
        rows.append(row)
    return rows


def confusion_table(runs: Dict[str, RunDirectory]) -> List[Dict[str, object]]:
    rows = []
    for protocol, run in runs.items():
        for method, matrix in run.confusion.items():
            for true_class, counts in enumerate(matrix):
                total = sum(counts)
                for predicted_class, count in enumerate(counts):
                    rows.append({"protocol": protocol, "method": method, "true_class": true_class, "predicted_class": predicted_class, "count": count,
                                 "share_of_true_class": count / total if total else 0.0})
    return rows


def write_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})


def write_validation_curves(run_dir: Path) -> Optional[Path]:
    """Per-epoch validation accuracy and loss of every run (from ``<run>/runs/*.json``, which are not shipped with the repository)."""
    files = sorted((run_dir / "runs").glob("*.json"))
    if not files:
        return None
    rows = []
    for path in files:
        with open(path) as handle:
            result = json.load(handle)
        fold = result["records"][0]["fold"]
        name = path.name[:-len(f"_fold{fold}.json")]
        for entry in result.get("val_history", []):
            rows.append({"run": name, "fold": fold, "epoch": entry["epoch"], "val_acc": entry["val_acc"], "val_loss": entry["val_loss"],
                         "is_best_epoch": int(entry["epoch"] == result["best_epoch"])})
    out = run_dir / "validation_curves.csv"
    write_rows(out, rows)
    return out


# ------------------------------------------------------------------------------------------------------------------ driver
def run(run_dir: Path, control_excerpt: Optional[Path], control_performer: Optional[Path], repeat_dirs: Sequence[Path], out_dir: Path, root: Path,
        write_curves: bool = False) -> Dict[str, object]:
    runs: Dict[str, RunDirectory] = {"recording": RunDirectory(run_dir)}
    if control_excerpt:
        runs["excerpt"] = RunDirectory(control_excerpt)
    if control_performer:
        runs["performer"] = RunDirectory(control_performer)
    for protocol, directory in runs.items():
        expected = protocol
        if directory.grouping != expected:
            raise ValueError(f"{directory.path} was run with --grouping {directory.grouping}, expected {expected}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stored = stored_values(root)
    outputs: List[str] = []

    def write(name: str, rows: Sequence[Dict[str, object]]) -> None:
        write_rows(out_dir / name, rows)
        outputs.append(name)

    methods = methods_table(runs["recording"], stored)
    write("methods.csv", methods)
    for protocol in ("excerpt", "performer"):
        if protocol in runs:
            write(f"methods_control_{protocol}.csv", methods_table(runs[protocol], stored))
    write("families.csv", families_table(runs))
    write("ladder.csv", ladder_table(runs, stored))
    write("paired.csv", paired_table(runs))
    if repeat_dirs:
        write("repeats.csv", repeats_table([runs["recording"]] + [RunDirectory(p) for p in repeat_dirs]))
    write("confusion_long.csv", confusion_table(runs))
    report: Dict[str, object] = {"run": run_dir.name, "protocols": {p: r.path.name for p, r in runs.items()}, "n_methods": len(methods)}
    for protocol, run_p in runs.items():
        rows = methods_table(run_p, stored)
        models = [m for m in rows if m["kind"] != "baseline"]
        summary: Dict[str, object] = {}
        for head in ("head=kan", "head=mlp", "head=linear"):
            selected = [m for m in models if m["kind"] == head]
            if selected:
                best = max(selected, key=lambda m: m["accuracy"])
                summary[head] = {"n_configurations": len(selected), "best": best["method"], "best_accuracy": best["accuracy"], "best_ci": [best["accuracy_ci_low"], best["accuracy_ci_high"]],
                                 "median_accuracy": statistics.median(m["accuracy"] for m in selected), "min_accuracy": min(m["accuracy"] for m in selected),
                                 "max_accuracy": max(m["accuracy"] for m in selected)}
        summary["baselines"] = {m["method"]: {"accuracy": m["accuracy"], "ci": [m["accuracy_ci_low"], m["accuracy_ci_high"]], "recording_accuracy": m["recording_accuracy"]}
                                for m in rows if m["kind"] == "baseline"}
        report[protocol] = summary
    rp_kan = families(runs["recording"]).get(("head=kan", "rp"), [])
    if rp_kan:                                                                          # mean confusion (row shares) of the RP WavKAN configurations
        matrices = np.array([runs["recording"].confusion[m] for m in rp_kan], dtype=float)
        shares = (matrices / matrices.sum(axis=2, keepdims=True)).mean(axis=0)
        report["rp_wavkan_mean_confusion_row_shares_recording_disjoint"] = [[round(float(v), 4) for v in row] for row in shares]
    report["inputs_sha256"] = {f"{p.name}/{f.name}": sha256_file(f) for p in [r.path for r in runs.values()] + list(repeat_dirs)
                               for f in sorted(p.glob("*.csv")) + [p / "PROVENANCE.json"]}
    if write_curves:
        for directory in [r.path for r in runs.values()] + list(repeat_dirs):
            curves = write_validation_curves(directory)
            if curves:
                outputs.append(f"(wrote {curves})")
    write_json(out_dir / "report.json", report)
    outputs.append("report.json")
    return {"outputs": outputs, "report": report}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True, help="output directory of main_recording_disjoint.py (recording-disjoint folds)")
    parser.add_argument("--control-excerpt", default=None, help="output directory of the excerpt-level control (--grouping excerpt)")
    parser.add_argument("--control-performer", default=None, help="output directory of the performer-disjoint control (--grouping performer)")
    parser.add_argument("--repeat", action="append", default=[], help="further recording-disjoint runs with other partition seeds (repeatable)")
    parser.add_argument("--out", default=None, help="output directory (default: <run>/report)")
    parser.add_argument("--write-curves", action="store_true", help="write <run>/validation_curves.csv from <run>/runs (when that folder exists)")
    args = parser.parse_args(argv)
    run_dir = Path(args.run).resolve()
    result = run(run_dir, Path(args.control_excerpt).resolve() if args.control_excerpt else None, Path(args.control_performer).resolve() if args.control_performer else None,
                 [Path(p).resolve() for p in args.repeat], Path(args.out).resolve() if args.out else run_dir / "report", DERIVED_RESULTS.parent, args.write_curves)
    print("wrote " + ", ".join(result["outputs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
