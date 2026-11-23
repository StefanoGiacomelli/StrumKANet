"""Stage 2 (14 retrained models): regenerate the table, validation intervals, curves and wall-clock from the stored files.

Inputs : experiments/results/training_results.csv, experiments/results/hp_search_results.csv,
         experiments/logs/pattern/<run>/version_0/events.out.tfevents.* (fit file first, test file second)
Outputs: results/stage2_table.csv, stage2_training_curves.csv, stage2_training_time.csv, stage2_summary.json
Nothing in experiments/ is modified. "val_acc best" is the best epoch value (the "Val Acc." column of the IS2 2025 paper);
"val_acc last" is the final-epoch value (not used for selection). The "test" columns were measured on a partition that
is re-drawn at evaluation time (about 80 % training overlap, see docs/PROTOCOL.md); the columns marked *model* below
(implied training accuracy, accuracy bound, predicted loss) apply that overlap model and are labelled as such.
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.common import (DERIVED_RESULTS, STORED_LOGS, STORED_RESULTS, consistent_denominators, file_record, parse_experiment_id,
                             pearson, read_csv_dicts, sanitize_id, wilson_interval, write_csv, write_json)
from analysis.tfevents_reader import read_scalars

OVERLAP_FRACTION = 0.8        # expected share of the re-drawn test partition that was training data (0.8 = split ratio)
# "Training Time" cells printed in the IS2 2025 paper (Table VII of the draft in docs/), transcribed for the comparison only.
PUBLISHED_TRAINING_TIME = {
    "ENC_chs32_ks_7-7_att_None_smooth_True_rp": (30, 49), "ENC_chs32_ks_7-7_att_None_smooth_False_rp": (34, 42),
    "ENC_chs16_ks_7-7_att_None_smooth_True_rp": (44, 32), "ENC_chs32_ks_3-3_att_None_smooth_False_rp": (43, 20),
    "ENC_chs32_ks_5-5_att_None_smooth_False_rp": (40, 13), "ENC_chs16_ks_3-3_att_None_smooth_False_rp": (35, 2),
    "ENC_chs16_ks_5-5_att_None_smooth_False_rp": (27, 11), "ENC_chs16_ks_7_att_convattn_smooth_False_onset_strength": (29, 35),
    "ENC_chs16_ks_5-5_att_None_smooth_True_rp": (32, 10)}
MAJORITY_RECORDING_RATE = 99 / 255        # VERIFIED on the released archive: 255 distinct audio files (classes 87 / 99 / 69); the CSV has 257 rows (89 / 99 / 69), 99 / 257 = 0.385 counts two repeated rows twice
MAJORITY_ROW_RATE = 99 / 257
POOL_EXCERPTS, TRAIN_EXCERPTS, TEST_EXCERPTS = 5604, 4483, 561     # VERIFIED: PopMusicDataset on the released archive (earlier reading from the k/560 and k/561 lattices: 5,605 and 4,484)


def event_files(run_id: str) -> List[Path]:
    folder = STORED_LOGS / run_id / "version_0"
    return sorted(folder.glob("events.out.tfevents.*"), key=lambda p: int(p.name.split(".")[3]))


def load_run(run_id: str) -> Dict[str, object]:
    files = event_files(run_id)
    if len(files) != 2:
        raise RuntimeError(f"{run_id}: expected a fit and a test event file, found {len(files)}")
    fit, fit_times = read_scalars(str(files[0]))
    test, _ = read_scalars(str(files[1]))
    return {"fit": fit, "test": test, "fit_seconds": fit_times[-1] - fit_times[0], "files": files}


def run(out_dir: Path = DERIVED_RESULTS) -> Dict[str, object]:
    training_path = STORED_RESULTS / "training_results.csv"
    stage1_path = STORED_RESULTS / "hp_search_results.csv"
    stored = read_csv_dicts(training_path)
    stage1 = {r["experiment_id"]: float(r["test_metric"]) for r in read_csv_dicts(stage1_path)}
    checks: List[Dict[str, object]] = []

    def check(name, observed, expected):
        checks.append({"check": name, "observed": observed, "published_or_expected": expected, "pass": observed == expected})

    rows, curves, timing, input_files = [], [], [], [training_path, stage1_path]
    val_values, per_run = [], []
    for record in stored:
        experiment_id = record["experiment_id"]
        run_id = sanitize_id(experiment_id)
        info = parse_experiment_id(experiment_id)
        data = load_run(run_id)
        input_files += data["files"] + [STORED_LOGS / run_id / "version_0" / "hparams.yaml"]
        fit, test = data["fit"], data["test"]
        val_acc = [v for _, v, _ in fit["val_acc"]]
        val_loss = [v for _, v, _ in fit["val_loss"]]
        train_loss = [v for _, v, _ in fit["train_loss"]]
        stop_epoch = int(record["stopped_epoch"])
        test_acc, test_loss = float(record["test_metric"]), float(record["test_loss"])
        best = max(val_acc)
        best_index = val_acc.index(best)               # first epoch reaching the maximum (strict improvement, as EarlyStopping)
        val_values += val_acc
        test_step = test["test_acc"][-1][0]
        per_run.append({"run_id": run_id, "val_acc": val_acc, "val_loss": val_loss, "train_loss": train_loss, "stop_epoch": stop_epoch,
                        "best_index": best_index, "test_acc": test_acc, "test_loss": test_loss, "info": info,
                        "test_step": test_step, "test_acc_event": test["test_acc"][-1][1], "fit_seconds": data["fit_seconds"]})
        for epoch, (a, l) in enumerate(zip(val_acc, val_loss)):
            curves.append([run_id, epoch, a, l, train_loss[epoch] if len(train_loss) == len(val_acc) else float("nan")])
        wall = data["fit_seconds"]
        timing.append([run_id, stop_epoch, wall, wall / stop_epoch, f"{int(wall // 3600)}:{int(wall % 3600 // 60):02d}:{int(wall % 60):02d}",
                       "" if run_id not in PUBLISHED_TRAINING_TIME else 60 * PUBLISHED_TRAINING_TIME[run_id][0] + PUBLISHED_TRAINING_TIME[run_id][1],
                       "" if run_id not in PUBLISHED_TRAINING_TIME else wall / (60 * PUBLISHED_TRAINING_TIME[run_id][0] + PUBLISHED_TRAINING_TIME[run_id][1])])

    n_val_candidates = consistent_denominators(val_values, range(400, 800), 1e-3)
    n_test_candidates = consistent_denominators([p["test_acc"] for p in per_run], range(400, 800), 1e-6)
    n_val = n_val_candidates[0] if n_val_candidates else None
    n_test = n_test_candidates[0] if n_test_candidates else None
    check("validation accuracies are k/n for n =", n_val_candidates, [560])
    check("test accuracies are k/n for n =", n_test_candidates, [561])

    # --- per-run table -----------------------------------------------------------------------------------------------
    header = ["experiment_id", "run_id", "descriptor", "encoder", "channels", "kernel", "attention", "smoothing", "stop_epoch", "best_epoch_index",
              "stop_minus_best", "n_validation_epochs", "val_acc_best", "val_acc_best_ci_low", "val_acc_best_ci_high", "val_acc_last",
              "val_acc_last_ci_low", "val_acc_last_ci_high", "val_acc_epoch0", "val_loss_last", "train_loss_last", "test_accuracy",
              "test_ci_low", "test_ci_high", "test_loss", "test_minus_val_best", "above_0.90", "model_implied_train_acc", "model_acc_bound",
              "model_pred_test_loss", "stage1_test_accuracy", "stage1_minus_epoch0_val", "test_logged_step", "steps_per_epoch_implied"]
    table, summary_rows = [], []
    for record, p in zip(stored, per_run):
        va = p["val_acc"]
        best, last, first = max(va), va[-1], va[0]
        k_best, k_last, k_test = round(best * n_val), round(last * n_val), round(p["test_acc"] * n_test)
        ci_best, ci_last, ci_test = wilson_interval(k_best, n_val), wilson_interval(k_last, n_val), wilson_interval(k_test, n_test)
        implied = (p["test_acc"] - (1 - OVERLAP_FRACTION) * last) / OVERLAP_FRACTION
        bound = OVERLAP_FRACTION + (1 - OVERLAP_FRACTION) * last
        pred_loss = OVERLAP_FRACTION * p["train_loss"][-1] + (1 - OVERLAP_FRACTION) * p["val_loss"][-1]
        stage1_acc = stage1[record["experiment_id"]]
        row = [record["experiment_id"], p["run_id"], p["info"]["descriptor"], p["info"]["encoder"], p["info"]["channels"], p["info"]["kernel"],
               p["info"]["attention"], p["info"]["smoothing"], p["stop_epoch"], p["best_index"], p["stop_epoch"] - p["best_index"], len(va),
               best, ci_best[0], ci_best[1], last, ci_last[0], ci_last[1], first, p["val_loss"][-1], p["train_loss"][-1], p["test_acc"],
               ci_test[0], ci_test[1], p["test_loss"], p["test_acc"] - best, p["test_acc"] > 0.90, implied, bound, pred_loss, stage1_acc,
               stage1_acc - first, p["test_step"], p["test_step"] / p["stop_epoch"]]
        table.append(row)
        summary_rows.append(dict(zip(header, row)))
    write_csv(out_dir / "stage2_table.csv", header, table)
    write_csv(out_dir / "stage2_training_curves.csv", ["run_id", "epoch", "val_acc", "val_loss", "train_loss_epoch"], curves)
    write_csv(out_dir / "stage2_training_time.csv",
              ["run_id", "stop_epoch", "fit_wall_clock_s", "seconds_per_epoch", "fit_wall_clock_h_mm_ss", "published_table_time_s", "logged_over_published"], timing)

    # --- summary --------------------------------------------------------------------------------------------------------
    S = summary_rows
    gaps = [r["test_minus_val_best"] for r in S]
    rp = [r for r in S if r["descriptor"] == "rp"]
    free = [r for r in S if r["attention"] == "None"]
    free_rp = [r for r in free if r["descriptor"] == "rp"]
    mae_pred = statistics.mean(abs(r["model_pred_test_loss"] - r["test_loss"]) for r in S)
    mae_h0 = statistics.mean(abs(r["val_loss_last"] - r["test_loss"]) for r in S)
    within = [r for r in S if abs(r["test_accuracy"] - r["model_acc_bound"]) <= 0.02]
    e6 = [r["stage1_minus_epoch0_val"] for r in S]
    se_e6 = statistics.stdev(e6) / math.sqrt(len(e6))
    by_descriptor = {}
    for d in sorted({r["descriptor"] for r in S}):
        sel = [r for r in S if r["descriptor"] == d]
        by_descriptor[d] = {"n": len(sel), "mean_val_acc_best": statistics.mean(r["val_acc_best"] for r in sel),
                            "mean_val_acc_last": statistics.mean(r["val_acc_last"] for r in sel),
                            "mean_test_accuracy": statistics.mean(r["test_accuracy"] for r in sel)}
    summary = {
        "n_runs": len(S), "n_validation_denominator": n_val, "n_test_denominator": n_test,
        "stop_minus_best_values": sorted({r["stop_minus_best"] for r in S}),
        "n_validation_epochs_equal_stop_epoch": all(r["n_validation_epochs"] == r["stop_epoch"] for r in S),
        "steps_per_epoch_implied": sorted({r["steps_per_epoch_implied"] for r in S}),
        "steps_per_epoch_if_batch_32_and_4483_training_excerpts": math.ceil(TRAIN_EXCERPTS / 32),
        "test_minus_val_best": {"mean": statistics.mean(gaps), "min": min(gaps), "max": max(gaps),
                                "rp_min": min(r["test_minus_val_best"] for r in rp), "rp_max": max(r["test_minus_val_best"] for r in rp)},
        "rp_val_acc_best_range": [min(r["val_acc_best"] for r in rp), max(r["val_acc_best"] for r in rp)],
        "rp_val_acc_last_range": [min(r["val_acc_last"] for r in rp), max(r["val_acc_last"] for r in rp)],
        "test_accuracy_range": [min(r["test_accuracy"] for r in S), max(r["test_accuracy"] for r in S)],
        "n_above_0.90": sum(r["above_0.90"] for r in S), "n_above_0.90_rp": sum(r["above_0.90"] and r["descriptor"] == "rp" for r in S),
        "train_loss_last_range_unsmoothed_above_0.90": [min(r["train_loss_last"] for r in S if r["above_0.90"] and not r["smoothing"]),
                                                        max(r["train_loss_last"] for r in S if r["above_0.90"] and not r["smoothing"])],
        "train_loss_last_range_smoothed_above_0.90": [min(r["train_loss_last"] for r in S if r["above_0.90"] and r["smoothing"]),
                                                      max(r["train_loss_last"] for r in S if r["above_0.90"] and r["smoothing"])],
        "loss_signature_model": {"description": "test_loss predicted by 0.8*train_loss_last + 0.2*val_loss_last",
                                 "mae_predictor": mae_pred, "mae_exchangeable_test_equals_val_loss_last": mae_h0,
                                 "pearson_r": pearson([r["model_pred_test_loss"] for r in S], [r["test_loss"] for r in S])},
        "accuracy_bound_model": {"description": "test_acc within +-0.02 of 0.8 + 0.2*val_acc_last", "n_within": len(within),
                                 "implied_train_acc_range_within": [min(r["model_implied_train_acc"] for r in within), max(r["model_implied_train_acc"] for r in within)]},
        "stage1_test_minus_stage2_epoch0_val": {"mean": statistics.mean(e6), "se": se_e6, "n_positive": sum(x > 0 for x in e6), "n": len(e6),
                                                "note": "mixes contamination with the winner's curse of selecting the 14 by their Stage-1 test scores"},
        "median_populations": {
            "all_14": {"accuracy": statistics.median(r["test_accuracy"] for r in S), "loss": statistics.median(r["test_loss"] for r in S)},
            "attention_free_11": {"n": len(free), "accuracy": statistics.median(r["test_accuracy"] for r in free), "loss": statistics.median(r["test_loss"] for r in free)},
            "rp_attention_free_8": {"n": len(free_rp), "accuracy": statistics.median(r["test_accuracy"] for r in free_rp), "loss": statistics.median(r["test_loss"] for r in free_rp)}},
        "by_descriptor": by_descriptor,
        "logged_wall_clock_total_hours": sum(p["fit_seconds"] for p in per_run) / 3600,
        "logged_over_published_training_time": {"min": min(t[6] for t in timing if t[6] != ""), "max": max(t[6] for t in timing if t[6] != "")},
        "majority_class_rate_recording_level": {"value": MAJORITY_RECORDING_RATE, "value_over_csv_rows": MAJORITY_ROW_RATE, "source": "VERIFIED on the released archive: 99 of 255 distinct audio files (99 of 257 CSV rows); the excerpt-level rate is 2,070 / 5,604 = 0.369"},
        "overlap_model": {"fraction": OVERLAP_FRACTION, "expected_overlap_excerpts": TEST_EXCERPTS * TRAIN_EXCERPTS / POOL_EXCERPTS,
                          "sd_overlap_excerpts": math.sqrt(TEST_EXCERPTS * (TRAIN_EXCERPTS / POOL_EXCERPTS) * (1 - TRAIN_EXCERPTS / POOL_EXCERPTS) * (POOL_EXCERPTS - TEST_EXCERPTS) / (POOL_EXCERPTS - 1))},
        "checks": checks}

    check("14 runs", len(S), 14)
    check("stop epoch = best epoch index + 21 in every run", summary["stop_minus_best_values"], [21])
    check("one validation value per epoch", summary["n_validation_epochs_equal_stop_epoch"], True)
    check("141 optimizer steps per epoch (B = 32 with 4,483 training excerpts)", summary["steps_per_epoch_implied"], [141.0])
    check("nine models above 0.90 (8 RP + 1 OS)", (summary["n_above_0.90"], summary["n_above_0.90_rp"]), (9, 8))
    write_json(out_dir / "stage2_summary.json", summary)
    outputs = ["stage2_table.csv", "stage2_training_curves.csv", "stage2_training_time.csv", "stage2_summary.json"]
    return {"script": "analysis/stage2_table.py",
            "inputs": [file_record(p) for p in input_files],
            "outputs": [file_record(out_dir / name) for name in outputs],
            "parameters": {"overlap_fraction": OVERLAP_FRACTION, "n_validation": n_val, "n_test": n_test},
            "checks": checks}


if __name__ == "__main__":
    result = run()
    for item in result["checks"]:
        print(("PASS " if item["pass"] else "DIFF ") + f"{item['check']}: {item['observed']} (expected {item['published_or_expected']})")
