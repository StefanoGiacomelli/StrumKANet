"""Stage 1 (468 one-epoch runs): regenerate the reported numbers from the stored CSV and YAML files.

Inputs : experiments/results/hp_search_results.csv, experiments/results/training_results.csv,
         experiments/configs/pattern/*.yaml
Outputs: results/stage1_runs.csv, stage1_top10_accuracy.csv, stage1_top10_loss.csv, selection_check.csv,
         stage1_medians_by_smoothing.csv, stage1_descriptor_medians.csv, stage1_grid_composition.csv,
         stage1_paired_analysis.csv, stage1_summary.json
Nothing in experiments/ is modified. The Stage-1 "test" accuracy is measured on a partition that is re-drawn at
evaluation time after a single epoch (see docs/PROTOCOL.md); the paired analysis uses all 468 runs (no selection).
"""
from __future__ import annotations

import collections
import statistics
import sys
from pathlib import Path
from typing import Dict, List

if __package__ in (None, ""):                       # allow ``python analysis/stage1_grid.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.common import (CONFIG_DIR, DERIVED_RESULTS, DESCRIPTORS_1D, DESCRIPTORS_2D, LN3, STORED_RESULTS,
                             bootstrap_median_ci, file_record, directory_record, parse_experiment_id,
                             read_csv_dicts, sign_test_p, write_csv, write_json)

TOP_K = 10                                         # main_training.TOP_K
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 2000, 1
# Values reported for the stored campaign (IS2 2025 paper, protocol counts), used only for the PASS/FAIL report (never for the computation).
PUBLISHED = {"median_acc_smooth_false_pct": 42.42, "median_acc_smooth_true_pct": 42.78,
             "median_loss_smooth_false": 1.0759, "median_loss_smooth_true": 1.0790,
             "n_runs": 468, "n_union": 14, "n_overlap": 6}


def load_runs() -> List[Dict[str, object]]:
    runs = []
    for row in read_csv_dicts(STORED_RESULTS / "hp_search_results.csv"):
        record = parse_experiment_id(row["experiment_id"])
        record.update(test_loss=float(row["test_loss"]), test_accuracy=float(row["test_metric"]),
                      test_status=row["test_status"])
        runs.append(record)
    return runs


def paired(runs, factor: str, base, other, keys) -> Dict[str, object]:
    """Pair runs that differ only in ``factor`` (base level versus other level)."""
    index = {tuple(r[k] for k in keys) + (r[factor],): r for r in runs}
    cells = {tuple(r[k] for k in keys) for r in runs}
    d_acc, d_loss, better, worse, ties = [], [], 0, 0, 0
    for cell in sorted(cells, key=str):
        a, b = index.get(cell + (base,)), index.get(cell + (other,))
        if a is None or b is None:
            continue
        diff = b["test_accuracy"] - a["test_accuracy"]
        d_acc.append(diff)
        d_loss.append(b["test_loss"] - a["test_loss"])
        better += diff > 0
        worse += diff < 0
        ties += diff == 0
    low, high = bootstrap_median_ci(d_acc, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    return {"n_pairs": len(d_acc), "other_better": better, "other_worse": worse, "ties": ties,
            "median_delta_accuracy": statistics.median(d_acc), "ci95_low": low, "ci95_high": high,
            "median_delta_loss": statistics.median(d_loss), "sign_test_p": sign_test_p(better, worse)}


def run(out_dir: Path = DERIVED_RESULTS) -> Dict[str, object]:
    import pandas as pd
    import yaml

    csv_path = STORED_RESULTS / "hp_search_results.csv"
    training_path = STORED_RESULTS / "training_results.csv"
    yaml_paths = sorted((CONFIG_DIR / "pattern").glob("*.yaml"))
    runs = load_runs()
    checks: List[Dict[str, object]] = []

    def check(name, observed, expected):
        checks.append({"check": name, "observed": observed, "published_or_expected": expected, "pass": observed == expected})

    # --- all runs, parsed -------------------------------------------------------------------------------------
    write_csv(out_dir / "stage1_runs.csv",
              ["experiment_id", "descriptor", "encoder", "channels", "kernel", "attention", "smoothing", "test_loss", "test_accuracy"],
              [[r["experiment_id"], r["descriptor"], r["encoder"], r["channels"], r["kernel"], r["attention"],
                r["smoothing"], r["test_loss"], r["test_accuracy"]] for r in runs])
    check("rows in hp_search_results.csv", len(runs), PUBLISHED["n_runs"])
    check("all test_status True", all(r["test_status"] == "True" for r in runs), True)

    # --- top-10 lists: the same pandas expressions as main_training.py ------------------------------------------
    df = pd.read_csv(csv_path)
    df = df[df["test_status"] == True]  # noqa: E712  (same expression as main_training.py)
    top_acc = df.sort_values("test_metric", ascending=False).head(TOP_K)
    top_loss = df.sort_values("test_loss", ascending=True).head(TOP_K)
    selected = pd.concat([top_acc, top_loss]).drop_duplicates(subset=["experiment_id"])
    for name, frame in (("stage1_top10_accuracy.csv", top_acc), ("stage1_top10_loss.csv", top_loss)):
        write_csv(out_dir / name, ["rank", "experiment_id", "test_accuracy", "test_loss"],
                  [[i + 1, row.experiment_id, float(row.test_metric), float(row.test_loss)] for i, row in enumerate(frame.itertuples())])
    selected_ids = selected["experiment_id"].tolist()
    stored_ids = [r["experiment_id"] for r in read_csv_dicts(training_path)]
    overlap = len(set(top_acc["experiment_id"]) & set(top_loss["experiment_id"]))
    write_csv(out_dir / "selection_check.csv", ["experiment_id", "in_top10_accuracy", "in_top10_loss", "in_training_results_csv"],
              [[i, i in set(top_acc["experiment_id"]), i in set(top_loss["experiment_id"]), i in set(stored_ids)]
               for i in sorted(set(selected_ids) | set(stored_ids))])
    check("union of the two top-10 lists", len(selected_ids), PUBLISHED["n_union"])
    check("overlap of the two top-10 lists", overlap, PUBLISHED["n_overlap"])
    check("union equals the 14 ids of training_results.csv", set(selected_ids) == set(stored_ids), True)
    sorted_acc = sorted((r["test_accuracy"] for r in runs), reverse=True)
    ties_at_cut = sorted_acc[TOP_K - 1] == sorted_acc[TOP_K]

    # --- medians --------------------------------------------------------------------------------------------------
    medians = []
    for smoothing in (False, True):
        sel = [r for r in runs if r["smoothing"] == smoothing]
        medians.append([smoothing, len(sel), 100 * statistics.median(r["test_accuracy"] for r in sel),
                        statistics.median(r["test_loss"] for r in sel)])
    write_csv(out_dir / "stage1_medians_by_smoothing.csv", ["smoothing", "n_runs", "median_accuracy_pct", "median_loss"], medians)
    check("median accuracy, smoothing off (%)", round(medians[0][2], 2), PUBLISHED["median_acc_smooth_false_pct"])
    check("median accuracy, smoothing on (%)", round(medians[1][2], 2), PUBLISHED["median_acc_smooth_true_pct"])
    check("median loss, smoothing off", round(medians[0][3], 4), PUBLISHED["median_loss_smooth_false"])
    check("median loss, smoothing on", round(medians[1][3], 4), PUBLISHED["median_loss_smooth_true"])

    by_descriptor = collections.defaultdict(list)
    for r in runs:
        by_descriptor[r["descriptor"]].append(r)
    write_csv(out_dir / "stage1_descriptor_medians.csv", ["descriptor", "encoder", "n_runs", "median_accuracy", "max_accuracy", "median_loss"],
              [[d, v[0]["encoder"], len(v), statistics.median(x["test_accuracy"] for x in v), max(x["test_accuracy"] for x in v),
                statistics.median(x["test_loss"] for x in v)]
               for d, v in sorted(by_descriptor.items(), key=lambda kv: -statistics.median(x["test_accuracy"] for x in kv[1]))])

    # --- grid composition from the YAML files (468 = 6 x 48 + 5 x 36) ----------------------------------------------
    yaml_rows = collections.Counter()
    epochs, kan_types, id_mismatch = collections.Counter(), collections.Counter(), 0
    for path in yaml_paths:
        with open(path) as handle:
            config = yaml.safe_load(handle)
        features = config["data"]["features"]
        descriptor = features[0] if features else "raw"
        yaml_rows[(descriptor, config["encoder"]["type"])] += 1
        epochs[config["epochs"]] += 1
        kan_types[config["kan"]["type"]] += 1
        id_mismatch += config["experiment_id"] != path.stem
    composition = []
    for descriptor in DESCRIPTORS_1D + DESCRIPTORS_2D:
        encoder = "2D" if descriptor in DESCRIPTORS_2D else "1D"
        composition.append([descriptor, encoder, yaml_rows[(descriptor, encoder)], len(by_descriptor[descriptor]),
                            "2 channels x 3 kernels x 3 attentions x 2 smoothing = 36" if encoder == "2D"
                            else "2 channels x 3 kernels x 4 attentions x 2 smoothing = 48"])
    total_1d = sum(r[2] for r in composition if r[1] == "1D")
    total_2d = sum(r[2] for r in composition if r[1] == "2D")
    composition += [["TOTAL 1D descriptors", "1D", total_1d, sum(r[3] for r in composition if r[1] == "1D"), "6 x 48"],
                    ["TOTAL 2D descriptors", "2D", total_2d, sum(r[3] for r in composition if r[1] == "2D"), "5 x 36"],
                    ["TOTAL", "", total_1d + total_2d, len(runs), "468 = 288 + 180"]]
    reading_372 = sum(len(v) for d, v in by_descriptor.items() if d not in ("raw", "ticks"))
    composition.append(["reading without raw and ticks (4 x 48 + 5 x 36)", "", reading_372, reading_372, "the 372 of the published table"])
    write_csv(out_dir / "stage1_grid_composition.csv", ["descriptor", "encoder", "n_yaml_files", "n_csv_rows", "factorization"], composition)
    check("YAML files in configs/pattern", len(yaml_paths), 468)
    check("grid = 6 x 48 + 5 x 36", (total_1d, total_2d), (288, 180))
    check("rows without raw and ticks (published table)", reading_372, 372)
    check("epochs per YAML (all 1)", dict(epochs), {1: 468})
    check("experiment_id equals file stem", id_mismatch, 0)

    # --- paired whole-grid analysis (all 468 runs, no selection) ----------------------------------------------------
    rows = []
    runs_1d = [r for r in runs if r["encoder"] == "1D"]
    runs_2d = [r for r in runs if r["encoder"] == "2D"]
    for label, subset, attentions in (("1D", runs_1d, ("posenc", "se", "convattn")), ("2D", runs_2d, ("axial", "mhsa"))):
        for attention in attentions:
            rows.append((f"{label} attention {attention} vs none", paired(subset, "attention", "None", attention, ["descriptor", "channels", "kernel", "smoothing"])))
    for label, subset, kernels in (("1D", runs_1d, ("3", "5", "7")), ("2D", runs_2d, ("[3, 3]", "[5, 5]", "[7, 7]"))):
        for a, b in ((kernels[0], kernels[1]), (kernels[1], kernels[2]), (kernels[0], kernels[2])):
            rows.append((f"{label} kernel {a} -> {b}", paired(subset, "kernel", a, b, ["descriptor", "channels", "attention", "smoothing"])))
    for label, subset in (("1D", runs_1d), ("2D", runs_2d)):
        rows.append((f"{label} channels 16 -> 32", paired(subset, "channels", 16, 32, ["descriptor", "kernel", "attention", "smoothing"])))
        rows.append((f"{label} smoothing off -> on", paired(subset, "smoothing", False, True, ["descriptor", "kernel", "attention", "channels"])))
    for other in ("tempogram", "tempogram_ratio", "tssd", "trh"):
        rows.append((f"2D descriptor rp -> {other}", paired(runs_2d, "descriptor", "rp", other, ["channels", "kernel", "attention", "smoothing"])))
    for other in ("raw", "beats", "ticks", "bpm_estimates", "rh"):
        rows.append((f"1D descriptor onset_strength -> {other}", paired(runs_1d, "descriptor", "onset_strength", other, ["channels", "kernel", "attention", "smoothing"])))
    write_csv(out_dir / "stage1_paired_analysis.csv",
              ["comparison", "n_pairs", "other_better", "other_worse", "ties", "median_delta_accuracy", "ci95_low", "ci95_high",
               "median_delta_loss", "sign_test_p"],
              [[name, r["n_pairs"], r["other_better"], r["other_worse"], r["ties"], r["median_delta_accuracy"], r["ci95_low"],
                r["ci95_high"], r["median_delta_loss"], r["sign_test_p"]] for name, r in rows])

    # --- scalar summary ---------------------------------------------------------------------------------------------
    accuracies = [r["test_accuracy"] for r in runs]
    losses = [r["test_loss"] for r in runs]
    summary = {
        "n_runs": len(runs),
        "ln3": LN3,
        "n_runs_accuracy_le_0.43": sum(a <= 0.43 for a in accuracies),
        "n_runs_loss_within_0.03_of_ln3": sum(abs(l - LN3) <= 0.03 for l in losses),
        "n_runs_accuracy_ge_0.55": sum(a >= 0.55 for a in accuracies),
        "median_accuracy_all": statistics.median(accuracies),
        "max_accuracy": max(accuracies),
        "all_test_accuracies_are_k_over_561": all(abs(a * 561 - round(a * 561)) < 1e-9 for a in accuracies),
        "tie_at_top10_accuracy_cut": ties_at_cut,
        "overlap_of_top10_lists": overlap,
        "selected_ids_equal_training_results": set(selected_ids) == set(stored_ids),
        "kan_types_in_yaml": dict(kan_types),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "statistic": "median of paired accuracy differences"},
        "note": "Stage-1 test accuracy is measured on a partition re-drawn at evaluation time, after one epoch (see docs/PROTOCOL.md).",
        "checks": checks,
    }
    write_json(out_dir / "stage1_summary.json", summary)
    outputs = ["stage1_runs.csv", "stage1_top10_accuracy.csv", "stage1_top10_loss.csv", "selection_check.csv", "stage1_medians_by_smoothing.csv",
               "stage1_descriptor_medians.csv", "stage1_grid_composition.csv", "stage1_paired_analysis.csv", "stage1_summary.json"]
    return {"script": "analysis/stage1_grid.py",
            "inputs": [file_record(csv_path), file_record(training_path), {"configs/pattern/*.yaml": directory_record(yaml_paths)}],
            "outputs": [file_record(out_dir / name) for name in outputs],
            "parameters": {"top_k": TOP_K, "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED},
            "checks": checks}


if __name__ == "__main__":
    result = run()
    for item in result["checks"]:
        print(("PASS " if item["pass"] else "DIFF ") + f"{item['check']}: {item['observed']} (expected {item['published_or_expected']})")
