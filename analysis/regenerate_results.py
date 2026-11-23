"""Regenerate every reported number that can be rebuilt from the stored experiment files, with a provenance record.

Usage (from any directory)::

    python analysis/regenerate_results.py [--out results] [--skip-torch]

Steps: analysis/stage1_grid.py, analysis/stage2_table.py and, when PyTorch is importable, analysis/model_complexity.py.
Writes results/*.csv, results/*.json and results/PROVENANCE.json (script, inputs and outputs with SHA-256, parameters,
environment). The CSV/JSON outputs are deterministic (seeded bootstrap, sorted pairing), so re-running leaves their
hashes unchanged; only ``generated_utc`` in PROVENANCE.json changes. Nothing under experiments/ is modified.
"""
from __future__ import annotations

import argparse
import datetime
import importlib
import platform
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import stage1_grid, stage2_table
from analysis.common import DERIVED_RESULTS, ROOT, sha256_file, write_json


def package_version(name: str) -> str:
    try:
        return getattr(importlib.import_module(name), "__version__", "unknown")
    except ImportError:
        return "not installed"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(DERIVED_RESULTS), help="output directory (default: results/)")
    parser.add_argument("--skip-torch", action="store_true", help="skip the PyTorch-based model-complexity step")
    args = parser.parse_args(argv)
    out_dir = Path(args.out).resolve()

    steps = [stage1_grid, stage2_table]
    if not args.skip_torch:
        try:
            import torch  # noqa: F401
            from analysis import model_complexity
            steps.append(model_complexity)
        except ImportError:
            print("[INFO] PyTorch not importable: model-complexity step skipped")
    records, failed = [], 0
    for step in steps:
        record = step.run(out_dir)
        script = ROOT / record["script"]
        record["script_sha256"] = sha256_file(script)
        records.append(record)
        print(f"[{record['script']}] wrote {len(record['outputs'])} files")
        for item in record["checks"]:
            failed += not item["pass"]
            print(("  PASS " if item["pass"] else "  DIFF ") + f"{item['check']}: {item['observed']} (expected {item['published_or_expected']})")
    provenance = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": "python analysis/regenerate_results.py" + ("" if not args.skip_torch else " --skip-torch"),
        "repository": "StrumKANet; stored inputs under experiments/ are read-only for these scripts",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "pandas": package_version("pandas"), "pyyaml": package_version("yaml"), "torch": package_version("torch")},
        "steps": records,
        "notes": ["Derived tables only: nothing here replaces or alters an original result in experiments/results/.",
                  "Bootstrap intervals are seeded and the pairing order is sorted, so they are reproducible bit for bit.",
                  "Columns marked 'model' in stage2_table.csv apply the 80 % training-overlap model of docs/PROTOCOL.md; they are not measurements."],
    }
    write_json(out_dir / "PROVENANCE.json", provenance)
    print(f"[done] {sum(len(r['outputs']) for r in records)} files + PROVENANCE.json in {out_dir}; checks with a difference: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
