"""Tables for the training-recipe and descriptor variants of ``main_variants.py``: accuracy with cluster-bootstrap intervals, paired differences.

Usage (from any directory)::

    python analysis/variants_report.py --variants results/recording_disjoint_2026/variants/seed42 \
        --stored results/recording_disjoint_2026/primary_seed42 [--out DIR]      # default DIR: <variants>/report

``--variants`` is a run directory written by ``main_variants.py`` (``summary.csv``, ``folds.csv``, ``predictions/``, ``PROVENANCE.json``);
``--stored`` is the run directory of ``main_recording_disjoint.py`` that holds the stored predictions of the same configurations and folds
(the WavKAN, MLP and linear heads and the logistic-regression baselines). Every variant is compared, recording by recording, with the
stored predictions of the SAME configuration (paired cluster bootstrap: the same resampled recordings on both sides, 4,000 resamples, seeded):

* ``variants.csv``: one row per method: accuracy with its recording-level interval, balanced accuracy, macro-F1, ternary recall, recording-level
  accuracy, and the paired difference to the stored WavKAN head and to the stored head of the same kind (and to the stored logistic regression on RP
  for the classical models), with the 95 % interval and whether it excludes zero;
* ``variants_pooled.csv``: the same for a variant and a head pooled over the configurations that were run (mean over configurations: the same
  resampled recordings are used for every configuration), against the mean of the stored WavKAN heads of the same configurations;
* ``variants.md``: the pooled table in Markdown.

With ``--replication`` the script takes the run directories of the partition seeds 42, 43 and 44 (``--variants``) and the matching stored runs (``--stored``) and writes
``replication.csv`` and ``replication.md`` (``--out``, default ``<first variants directory>/../replication``): for a fixed list of methods the accuracy and the paired
difference to the stored reference on every partition, and the mean of the three differences with a paired cluster-bootstrap interval (the partitions cover the same recordings,
so one resample of recordings serves the three of them).

The decision rule of the study is fixed in ``docs/PROTOCOL.md`` section 13: a variant is documented as an improvement only if the interval of its paired difference to the
stored RP + WavKAN excludes zero, the sign is the same on every configuration, the gain replicates on the other partition seeds, and no held-out information was used.
Everything is deterministic. Nothing under ``experiments/`` is modified.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.recording_disjoint_report import BOOTSTRAP, RunDirectory, family_difference, family_interval    # noqa: E402


def read_csv(path) -> List[Dict[str, str]]:
    from analysis.recording_disjoint_report import read_csv as _read
    return _read(Path(path))


def read_summary_rows(path) -> List[Dict[str, str]]:
    return read_csv(path)


def read_summary(run: RunDirectory) -> Dict[str, Dict[str, float]]:
    return {m: {k: float(v) for k, v in row.items() if k != "method"} for m, row in run.summary.items()}


def parse_method(method: str):
    """('<experiment id>', 'kan', 'bal+aug') for '<experiment id>|head=kan|variant=bal+aug'; classical: (None, None, 'classical:lr_rp|descriptor=rp')."""
    parts = method.split("|")
    if method.startswith("classical:"):
        return None, None, method
    experiment = parts[0]
    head = parts[1].split("=", 1)[1]
    variant = parts[2].split("=", 1)[1]
    return experiment, head, variant


HEAD_NAMES = {"kan": "WavKAN", "mlp": "MLP", "linear": "linear"}
DESCRIPTOR_NAMES = {"rp": "RP", "tempogram_ratio": "tempogram ratio", "onset_strength": "onset strength", "rh": "RH", "tssd": "TSSD", "trh": "TRH"}
VECTOR_NAMES = {"tr": "tempogram-ratio statistics (26)", "os": "onset scale features (32)", "tros": "tempogram-ratio statistics + onset scale features (58)",
                "trosrh": "tempogram-ratio statistics + onset scale features + RH (118)"}
CLASSICAL_INPUT_NAMES = {"rp": "RP", "rp_bandnorm": "RP, band-normalized", "rp_nofsw": "RP without fluctuation weighting", "rh": "RH", "trh": "TRH", "tssd": "TSSD",
                         "tr_stats": "tempogram-ratio statistics (26)", "tr_mean": "tempogram-ratio means (13)", "tr_std": "tempogram-ratio deviations (13)",
                         "tr_quantiles": "tempogram-ratio quantiles (39)", "os_scale": "onset scale features (32)"}


def pretty(method: str) -> str:
    """Readable label of a method (no vertical bars, which break Markdown tables)."""
    experiment, head, variant = parse_method(method)
    if experiment is None:
        name, _, inputs = variant[len("classical:"):].partition("|descriptor=")
        flag = ", class-balanced" if name.endswith("_bal") else ""
        return "logistic regression on " + " + ".join(CLASSICAL_INPUT_NAMES.get(i, i) for i in inputs.split("+")) + flag
    if experiment.startswith("VEC_"):
        return f"{HEAD_NAMES[head]} head on {VECTOR_NAMES.get(experiment[4:], experiment)}, {variant}"
    kind = descriptor(method)
    channels, kernel = experiment.split("_")[1][3:], experiment.split("_ks_")[1].split("_att_")[0].replace("[", "").replace("]", "").replace(", ", "x")
    return f"{DESCRIPTOR_NAMES.get(kind, kind)} network, {channels} ch, {kernel}, {HEAD_NAMES[head]} head, {variant}"


def paired(variants: RunDirectory, methods_a: Sequence[str], stored: RunDirectory, methods_b: Sequence[str]) -> Dict[str, object]:
    try:
        result = family_difference(variants, list(methods_a), stored, list(methods_b))
    except (KeyError, ValueError, FileNotFoundError) as error:
        return {"difference": "", "ci_low": "", "ci_high": "", "excludes_zero": "", "note": str(error)}
    return {"difference": result["difference"], "ci_low": result["ci_low"], "ci_high": result["ci_high"], "excludes_zero": result["ci_excludes_zero"], "note": ""}


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def stored_rp_methods(stored: RunDirectory, head: str) -> List[str]:
    return sorted(m for m in stored.methods if m.endswith(f"|head={head}") and not m.startswith("baseline:") and descriptor(m) == "rp")


def add_pair(row: Dict[str, object], label: str, variants: RunDirectory, methods: Sequence[str], stored: RunDirectory, reference: Sequence[str],
             summary_s: Dict[str, Dict[str, float]]) -> None:
    if not reference or not all(m in stored.summary for m in reference):
        return
    result = paired(variants, methods, stored, reference)
    row.update({f"{label}_{k}": v for k, v in result.items() if k != "note"})
    row[f"{label}_stored_accuracy"] = mean([summary_s[m]["accuracy"] for m in reference])


def build_rows(variants: RunDirectory, stored: RunDirectory):
    """One row per method (paired differences to the stored predictions) and one pooled row per recipe variant and head over the stored RP configurations run."""
    summary_v, summary_s = read_summary(variants), read_summary(stored)
    rows: List[Dict[str, object]] = []
    pooled_groups: Dict[tuple, List[str]] = {}
    for method in variants.methods:
        experiment, head, variant = parse_method(method)
        row: Dict[str, object] = {"method": method, "label": pretty(method), "variant": variant, "head": head or "", "configuration": experiment or "",
                                  "accuracy": summary_v[method]["accuracy"], "ci_low": summary_v[method]["accuracy_ci_cluster_low"],
                                  "ci_high": summary_v[method]["accuracy_ci_cluster_high"], "balanced_accuracy": summary_v[method]["balanced_accuracy"],
                                  "macro_f1": summary_v[method]["macro_f1"], "recall_ternary": summary_v[method]["recall_2"],
                                  "recording_accuracy": summary_v[method]["recording_accuracy"]}
        add_pair(row, "vs_stored_rp_wavkan_mean", variants, [method], stored, stored_rp_methods(stored, "kan"), summary_s)        # the headline baseline (mean of the RP configurations)
        if experiment is not None:
            if f"{experiment}|head=kan" in stored.summary:                                                                         # a stored RP configuration
                pooled_groups.setdefault((variant, head), []).append(method)
                add_pair(row, "vs_stored_kan", variants, [method], stored, [f"{experiment}|head=kan"], summary_s)
                add_pair(row, "vs_stored_same_head", variants, [method], stored, [f"{experiment}|head={head}"], summary_s)
            else:                                                                                                                  # another input (tempogram ratio, vector): same head, mean of the RP configurations
                add_pair(row, "vs_stored_rp_same_head_mean", variants, [method], stored, stored_rp_methods(stored, head), summary_s)
        else:
            add_pair(row, "vs_stored_logreg", variants, [method], stored, ["baseline:logreg|descriptor=rp"], summary_s)
        rows.append(row)
    pooled: List[Dict[str, object]] = []
    for (variant, head), methods in sorted(pooled_groups.items()):
        experiments = [parse_method(m)[0] for m in methods]
        kan = [f"{e}|head=kan" for e in experiments]
        same = [f"{e}|head={head}" for e in experiments]
        interval = family_interval(variants, methods)
        row = {"variant": variant, "head": head, "n_configurations": len(methods), "accuracy": interval["mean_accuracy"], "ci_low": interval["ci_low"],
               "ci_high": interval["ci_high"], "balanced_accuracy": mean([summary_v[m]["balanced_accuracy"] for m in methods]),
               "macro_f1": mean([summary_v[m]["macro_f1"] for m in methods]), "recall_ternary": mean([summary_v[m]["recall_2"] for m in methods]),
               "recording_accuracy": mean([summary_v[m]["recording_accuracy"] for m in methods])}
        add_pair(row, "vs_stored_kan", variants, methods, stored, kan, summary_s)
        if kan and all(m in stored.summary for m in kan):
            signs = [rows_by_method(rows, m).get("vs_stored_kan_difference") for m in methods]
            row["same_sign_on_every_configuration"] = bool(all(isinstance(x, float) and x > 0 for x in signs) or all(isinstance(x, float) and x < 0 for x in signs))
        add_pair(row, "vs_stored_same_head", variants, methods, stored, same, summary_s)
        row["configurations"] = ";".join(experiments)
        pooled.append(row)
    return rows, pooled


def rows_by_method(rows: List[Dict[str, object]], method: str) -> Dict[str, object]:
    return next(r for r in rows if r["method"] == method)


def descriptor(method: str) -> str:
    experiment = method.split("|")[0]
    return "onset_strength" if experiment.endswith("onset_strength") else experiment.rsplit("_", 1)[-1]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})


def fmt(value, digits: int = 3, sign: bool = False) -> str:
    if value == "" or value is None:
        return "-"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def interval(row: Dict[str, object], prefix: str) -> str:
    if row.get(f"{prefix}_difference", "") == "":
        return "-"
    star = "*" if row.get(f"{prefix}_excludes_zero") in (True, "True") else ""
    return f"{fmt(row[f'{prefix}_difference'], sign=True)} [{fmt(row[f'{prefix}_ci_low'], sign=True)}, {fmt(row[f'{prefix}_ci_high'], sign=True)}]{star}"


def write_markdown(path: Path, pooled: List[Dict[str, object]], rows: List[Dict[str, object]]) -> None:
    columns = "| {} | Accuracy [95 % interval] | Balanced acc. | Ternary recall | Recording-level acc. | {} | {} |"
    lines = [columns.format("Recipe variant / head (stored RP configurations, pooled)", "vs stored WavKAN (same configurations)", "vs stored same head (same configurations)"),
             "|---|---|---|---|---|---|---|"]
    for row in pooled:
        lines.append(f"| {row['variant']} / {HEAD_NAMES[row['head']]} head (n={row['n_configurations']}) | {fmt(row['accuracy'])} [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | "
                     f"{fmt(row['balanced_accuracy'])} | {fmt(row['recall_ternary'])} | {fmt(row['recording_accuracy'])} | {interval(row, 'vs_stored_kan')} | {interval(row, 'vs_stored_same_head')} |")
    others = [r for r in rows if "vs_stored_kan_difference" not in r]
    if others:
        lines += ["", columns.format("Other inputs, per method", "vs stored RP + WavKAN (mean of the RP configurations)", "vs stored RP, same head (mean), or logistic regression on RP"),
                  "|---|---|---|---|---|---|---|"]
        for row in others:
            second = interval(row, "vs_stored_rp_same_head_mean") if row["configuration"] else interval(row, "vs_stored_logreg")
            lines.append(f"| {pretty(row['method'])} | {fmt(row['accuracy'])} [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {fmt(row['balanced_accuracy'])} | {fmt(row['recall_ternary'])} | "
                         f"{fmt(row['recording_accuracy'])} | {interval(row, 'vs_stored_rp_wavkan_mean')} | {second} |")
    lines += ["", "`*`: the 95 % paired cluster-bootstrap interval of the difference excludes zero."]
    path.write_text("\n".join(lines) + "\n")


REPLICATION_ROWS = [      # (method, label, reference): "mean" = stored RP + WavKAN, mean of the RP configurations of the partition; "same" = stored WavKAN head of the same network
    ("classical:lr_rp|descriptor=rp", "logistic regression on RP (stored baseline)", "mean"),
    ("classical:lr_tr|descriptor=tr_stats", "logistic regression on tempogram-ratio statistics (26)", "mean"),
    ("classical:lr_osscale|descriptor=os_scale", "logistic regression on onset scale features (32)", "mean"),
    ("classical:lr_tros|descriptor=tr_stats+os_scale", "logistic regression on both (58)", "mean"),
    ("VEC_tros|head=mlp|variant=base", "MLP head on both (58)", "mean"),
    ("VEC_tros|head=kan|variant=base", "WavKAN head on both (58)", "mean"),
    ("VEC_tros|head=mlp|variant=bal", "MLP head on both (58), class-weighted loss", "mean"),
    ("ENC_chs16_ks_[5, 5]_att_None_smooth_False_tempogram_ratio|head=kan|variant=base", "tempogram-ratio network, 16 ch, 5x5, WavKAN head (stored configuration)", "mean"),
    ("ENC_chs16_ks_[5, 5]_att_None_smooth_False_rp|head=mlp|variant=aug", "RP network, 16 ch, 5x5, MLP head, augmentation", "same"),
    ("ENC_chs16_ks_[5, 5]_att_None_smooth_False_rp|head=mlp|variant=views", "RP network, 16 ch, 5x5, MLP head, three views", "same"),
    ("ENC_chs16_ks_[5, 5]_att_None_smooth_False_rp|head=kan|variant=aug", "RP network, 16 ch, 5x5, WavKAN head, augmentation", "same"),
]


def replication(variant_dirs: Sequence[str], stored_dirs: Sequence[str], out: Path, resamples: int = BOOTSTRAP, seed: int = 7) -> List[Dict[str, object]]:
    """Per-partition and pooled paired differences of the replicated methods: the pooled difference is the mean over the partitions of the per-partition difference,
    computed on the same resampled recordings (every partition covers the same recordings and excerpts)."""
    import numpy as np
    from analysis.recording_disjoint_report import _stack
    runs, stores = [RunDirectory(Path(d)) for d in variant_dirs], [RunDirectory(Path(d)) for d in stored_dirs]
    rows: List[Dict[str, object]] = []
    draws = None
    for method, label, kind in REPLICATION_ROWS:
        row: Dict[str, object] = {"method": method, "label": label, "reference": "stored RP + WavKAN, mean of the RP configurations of the partition" if kind == "mean" else "stored WavKAN head of the same network"}
        parts = []
        for index, (run, store) in enumerate(zip(runs, stores)):
            tag = f"seed{[42, 43, 44][index]}" if len(runs) == 3 else f"part{index}"
            if method not in run.summary:
                continue
            refs = stored_rp_methods(store, "kan") if kind == "mean" else [method.split("|")[0] + "|head=kan"]
            if not refs or not all(r in store.summary for r in refs):
                continue
            keys, corrects, counts = _stack(run, [method])
            keys_ref, corrects_ref, counts_ref = _stack(store, refs)
            if keys != keys_ref or not np.array_equal(counts, counts_ref):
                raise ValueError(f"{method}: the stored predictions do not cover the same excerpts")
            parts.append((corrects[0], corrects_ref, counts, keys))
            result = family_difference(run, [method], store, refs)
            row.update({f"{tag}_accuracy": run.summary[method]["accuracy"], f"{tag}_ci_low": run.summary[method]["accuracy_ci_cluster_low"], f"{tag}_ci_high": run.summary[method]["accuracy_ci_cluster_high"],
                        f"{tag}_difference": result["difference"], f"{tag}_difference_ci_low": result["ci_low"], f"{tag}_difference_ci_high": result["ci_high"],
                        f"{tag}_stored_accuracy": float(np.mean(corrects_ref.sum(axis=1) / counts.sum()))})
        if len(parts) == len(runs) and len(parts) > 1:
            keys = parts[0][3]
            if any(p[3] != keys or not np.array_equal(p[2], parts[0][2]) for p in parts):
                raise ValueError("the partitions do not cover the same recordings")
            if draws is None:
                draws = np.random.default_rng(seed).integers(0, len(keys), size=(resamples, len(keys)))
            total = parts[0][2][draws].sum(axis=1)
            estimates = np.mean([p[0][draws].sum(axis=1) / total - np.mean([r[draws].sum(axis=1) / total for r in p[1]], axis=0) for p in parts], axis=0)
            observed = float(np.mean([p[0].sum() / p[2].sum() - np.mean(p[1].sum(axis=1) / p[2].sum()) for p in parts]))
            low, high = float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))
            row.update({"pooled_difference": observed, "pooled_ci_low": low, "pooled_ci_high": high, "pooled_excludes_zero": bool(low > 0 or high < 0)})
        rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "replication.csv", rows)
    lines = ["| Method | Reference of the difference | Seed 42: accuracy [95 % interval]; difference | Seed 43: accuracy [95 % interval]; difference | Seed 44: accuracy [95 % interval]; difference | Mean difference over the three partitions |", "|---|---|---|---|---|---|"]
    for row in rows:
        cells = []
        for tag in ("seed42", "seed43", "seed44"):
            if f"{tag}_accuracy" not in row:
                cells.append("not run")
                continue
            star = "*" if (float(row[f"{tag}_difference_ci_low"]) > 0 or float(row[f"{tag}_difference_ci_high"]) < 0) else ""
            cells.append(f"{fmt(float(row[f'{tag}_accuracy']))} [{fmt(float(row[f'{tag}_ci_low']))}, {fmt(float(row[f'{tag}_ci_high']))}]; "
                         f"{fmt(float(row[f'{tag}_difference']), sign=True)} [{fmt(float(row[f'{tag}_difference_ci_low']), sign=True)}, {fmt(float(row[f'{tag}_difference_ci_high']), sign=True)}]{star}")
        pooled = "-"
        if "pooled_difference" in row:
            pooled = f"{fmt(row['pooled_difference'], sign=True)} [{fmt(row['pooled_ci_low'], sign=True)}, {fmt(row['pooled_ci_high'], sign=True)}]{'*' if row['pooled_excludes_zero'] else ''}"
        lines.append(f"| {row['label']} | {row['reference']} | " + " | ".join(cells) + f" | {pooled} |")
    first = next(r for r in rows if r["method"] == REPLICATION_ROWS[0][0])
    reference = [f"seed {tag[4:]}: {first[f'{tag}_stored_accuracy']:.3f}" for tag in ("seed42", "seed43", "seed44") if f"{tag}_stored_accuracy" in first]
    lines += ["", "Stored RP + WavKAN reference of the first kind (mean of the RP configurations that the stored run covers): " + ", ".join(reference) + "."]
    (out / "replication.md").write_text("\n".join(lines) + "\n")
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--variants", required=True, nargs="+", help="run directory written by main_variants.py (with --replication: the directories of the partition seeds 42, 43, 44, in this order)")
    parser.add_argument("--stored", required=True, nargs="+", help="run directory of main_recording_disjoint.py with the stored predictions (same partition seed); with --replication: one per partition")
    parser.add_argument("--out", default=None, help="output directory (default: <variants>/report, or <first variants directory>/../replication with --replication)")
    parser.add_argument("--replication", action="store_true", help="per-partition and pooled paired differences of the replicated methods (partition seeds 42, 43, 44)")
    args = parser.parse_args(argv)
    if args.replication:
        out = Path(args.out) if args.out else Path(args.variants[0]).parent / "replication"
        rows = replication(args.variants, args.stored, out)
        print(f"[done] {len(rows)} methods -> {out}")
        return 0
    if len(args.variants) != 1 or len(args.stored) != 1:
        parser.error("exactly one --variants and one --stored directory are expected without --replication")
    variants, stored = RunDirectory(Path(args.variants[0])), RunDirectory(Path(args.stored[0]))
    out = Path(args.out) if args.out else Path(args.variants[0]) / "report"
    out.mkdir(parents=True, exist_ok=True)
    rows, pooled = build_rows(variants, stored)
    write_csv(out / "variants.csv", rows)
    write_csv(out / "variants_pooled.csv", pooled)
    write_markdown(out / "variants.md", pooled, rows)
    print(f"[done] {len(rows)} methods, {len(pooled)} pooled variants -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
