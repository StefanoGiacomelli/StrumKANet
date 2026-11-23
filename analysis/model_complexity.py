"""Parameter counts and multiply-accumulate (MAC) budget of the 14 stored Stage-2 models (needs PyTorch, no dataset).

Inputs : experiments/results/training_results.csv, experiments/configs/pattern/*.yaml, experiments/results/training_log.txt
Outputs: results/stage2_model_complexity.csv
Each model is instantiated with ``model.init_modules`` on a synthetic input of the descriptor's shape (the number of
parameters does not depend on the input length). Columns:
  * *_params / *_log: parameters by instantiation versus the Lightning summary stored in training_log.txt;
  * analytic_*: arithmetic MAC budget per excerpt (weights-only MACs of the convolutions, attention and the head
    ``3 * D`` weighted sums; the element-wise wavelet evaluations are excluded);
  * counted_macs: multiply-accumulates of the convolution / matrix-multiply operations executed by one forward pass at
    batch size 1, from ``torch.utils.flop_counter`` (FLOPs / 2). It includes the unused ``F.linear(x, weight1)`` of
    WavKAN (another ``3 * D`` MACs), so it is comparable to the analytic total only up to that term (which coincides in size).
Assumptions: C_in = 1; 'same' convolutions; T = 938 frames for onset strength and tempogram ratio (480,000 samples,
hop 512, centred; derived, not stored); RH length 60; RP 24 x 60; MHSA on 32 x 32 tokens.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Dict

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.common import CONFIG_DIR, DERIVED_RESULTS, ROOT, STORED_RESULTS, file_record, parse_experiment_id, read_csv_dicts, sanitize_id, write_csv

FRAMES = 938
INPUT_SHAPES = {"rp": (24, 60), "rh": (60,), "tssd": (24, 49), "trh": (60, 7), "onset_strength": (FRAMES,),
                "tempogram_ratio": (13, FRAMES), "tempogram": (384, FRAMES)}


def parse_summary_table(log_text: str) -> Dict[str, Dict[str, str]]:
    """Encoder and head parameter cells of the Lightning model summary of every run in training_log.txt."""
    runs: Dict[str, Dict[str, str]] = {}
    current = None
    for line in re.split(r"[\r\n]", log_text):
        started = re.search(r"Starting: (\S+)", line)
        if started:
            current = started.group(1)
            runs[current] = {}
        cell = re.match(r"\s*\d\s*\|\s*(encoder|kan)\s*\|\s*\w+\s*\|\s*([0-9.]+\s*[KM]?)\s*\|", line)
        if cell and current and cell.group(1) not in runs[current]:
            runs[current][cell.group(1)] = cell.group(2).replace(" ", "")
    return runs


def lightning_count(number: int) -> str:
    """Parameter count formatted like the Lightning model summary (e.g. 832, 1.7K, 786K, 1.6M)."""
    labels = [" ", "K", "M", "B", "T"]
    digits = int(math.floor(math.log10(number))) + 1 if number > 0 else 1
    groups = int(math.ceil(digits / 3))
    scaled = number * 10 ** (-3 * (groups - 1))
    index = groups - 1
    text = f"{int(scaled):,d} {labels[index]}" if index < 1 or scaled >= 100 else f"{scaled:,.1f} {labels[index]}"
    return text.replace(" ", "")


def analytic_macs(info: Dict[str, object], descriptor: str) -> Dict[str, int]:
    channels = int(info["channels"])
    kernel = int(str(info["kernel"]).strip("[]").split(",")[0].split("-")[0])
    attention = info["attention"]
    shape = INPUT_SHAPES[descriptor]
    positions = shape[0] if len(shape) == 1 else shape[0] * shape[1]
    if attention == "mhsa":
        positions = 32 * 32                          # AdaptiveAvgPool2d((32, 32)) before the attention
    conv = channels * (kernel ** (2 if info["encoder"] == "2D" else 1)) * positions
    att = 0
    if attention == "convattn":
        att = channels * 3 * positions + channels * channels * positions        # depthwise (k = 3) + pointwise
    elif attention == "mhsa":
        att = 768 * 1024 + 2 * 8 * 1024 * 1024 * 32 + 256 * 1024               # QKV, scores + values, output projection
    head = 3 * channels * 4096
    return {"conv": conv, "attention": att, "head": head, "total": conv + att + head, "embedding": channels * 4096}


def run(out_dir: Path = DERIVED_RESULTS) -> Dict[str, object]:
    import torch
    import yaml
    from torch.utils.flop_counter import FlopCounterMode

    sys.path.insert(0, str(ROOT))
    from model import init_modules

    stored = read_csv_dicts(STORED_RESULTS / "training_results.csv")
    log_path = STORED_RESULTS / "training_log.txt"
    summary = parse_summary_table(log_path.read_text(errors="replace"))
    rows, inputs, checks = [], [log_path, STORED_RESULTS / "training_results.csv"], []
    for record in stored:
        experiment_id = record["experiment_id"]
        run_id = sanitize_id(experiment_id)
        info = parse_experiment_id(experiment_id)
        descriptor = info["descriptor"]
        config_path = CONFIG_DIR / "pattern" / f"{experiment_id}.yaml"
        inputs.append(config_path)
        with open(config_path) as handle:
            config = yaml.safe_load(handle)
        shape = INPUT_SHAPES[descriptor]
        torch.manual_seed(0)
        encoder, head = init_modules(config, torch.randn(2, 1, *shape))
        encoder_params = sum(p.numel() for p in encoder.parameters())
        head_params = sum(p.numel() for p in head.parameters())
        embedding = encoder(torch.randn(2, 1, *shape)).shape[1]
        encoder.eval()
        head.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            head(torch.randn(1, 1, *shape), encoder=encoder)
        counted = counter.get_total_flops() / 2
        analytic = analytic_macs(info, descriptor)
        log_cells = summary.get(run_id, {})
        head_log = log_cells.get("kan", "")
        encoder_log = log_cells.get("encoder", "")
        encoder_ok = lightning_count(encoder_params) == encoder_log
        head_log_ok = lightning_count(head_params) == head_log
        rows.append([run_id, descriptor, info["encoder"], info["channels"], info["attention"], embedding, encoder_params, log_cells.get("encoder", ""),
                     head_params, 12 * embedding + 6, head_log, 9 * embedding + 6, encoder_params + head_params, analytic["conv"], analytic["attention"],
                     analytic["head"], analytic["total"], int(counted), int(counted) - (analytic["conv"] + analytic["attention"] + analytic["head"]),
                     encoder_ok, head_log_ok])
    write_csv(out_dir / "stage2_model_complexity.csv",
              ["run_id", "descriptor", "encoder_type", "channels", "attention", "embedding_dim_D", "encoder_params", "encoder_params_log", "head_params",
               "head_params_formula_12D_plus_6", "head_params_log", "head_effective_params_9D_plus_6", "total_params", "analytic_macs_conv",
               "analytic_macs_attention", "analytic_macs_head_3D", "analytic_macs_total", "counted_macs_conv_matmul", "counted_minus_analytic_total",
               "encoder_params_match_log", "head_params_match_log"], rows)
    checks.append({"check": "encoder parameters equal the Lightning summary of training_log.txt (14 runs)", "observed": sum(r[-2] for r in rows),
                   "published_or_expected": 14, "pass": sum(r[-2] for r in rows) == 14})
    checks.append({"check": "head parameters equal 12 D + 6 and match the logged summary (14 runs)",
                   "observed": sum(r[8] == r[9] and r[-1] for r in rows), "published_or_expected": 14,
                   "pass": sum(r[8] == r[9] and r[-1] for r in rows) == 14})
    return {"script": "analysis/model_complexity.py",
            "inputs": [file_record(p) for p in inputs],
            "outputs": [file_record(out_dir / "stage2_model_complexity.csv")],
            "parameters": {"frames": FRAMES, "torch": torch.__version__},
            "checks": checks}


if __name__ == "__main__":
    result = run()
    for item in result["checks"]:
        print(("PASS " if item["pass"] else "DIFF ") + f"{item['check']}: {item['observed']} (expected {item['published_or_expected']})")
