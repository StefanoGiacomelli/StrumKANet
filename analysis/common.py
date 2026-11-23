"""Shared helpers of the regeneration scripts (standard library only; pandas/PyYAML are imported by the steps that need them)."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
STORED_RESULTS = ROOT / "experiments" / "results"
STORED_LOGS = ROOT / "experiments" / "logs" / "pattern"
CONFIG_DIR = ROOT / "experiments" / "configs"
DERIVED_RESULTS = ROOT / "results"

ID_PATTERN = re.compile(r"^ENC_chs(\d+)_ks_(.+?)_att_(None|posenc|se|convattn|axial|mhsa)_smooth_(True|False)_(.+)$")
_BRACKET_PAIR = re.compile(r"\[(\d+),\s*(\d+)\]")
DESCRIPTORS_2D = ("tempogram", "tempogram_ratio", "rp", "tssd", "trh")
DESCRIPTORS_1D = ("raw", "onset_strength", "beats", "ticks", "bpm_estimates", "rh")
LN3 = math.log(3.0)


def sanitize_id(experiment_id: str) -> str:
    """Same rule as ``main_training.sanitize_experiment_id``: ``[7, 7]`` becomes ``7-7``."""
    return _BRACKET_PAIR.sub(r"\1-\2", experiment_id)


def parse_experiment_id(experiment_id: str) -> Dict[str, object]:
    """Split an experiment id into its factors (works with the stored and the sanitised spelling)."""
    match = ID_PATTERN.match(experiment_id) or ID_PATTERN.match(
        re.sub(r"(\d+)-(\d+)", r"[\1, \2]", experiment_id, count=1))
    if match is None:
        raise ValueError(f"unrecognised experiment id: {experiment_id}")
    channels, kernel, attention, smoothing, descriptor = match.groups()
    return {"experiment_id": experiment_id,
            "descriptor": descriptor,
            "encoder": "2D" if descriptor in DESCRIPTORS_2D else "1D",
            "channels": int(channels),
            "kernel": kernel,
            "attention": attention,
            "smoothing": smoothing == "True"}


# ----------------------------------------------------------------------------- statistics
def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval of a binomial proportion."""
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test (ties excluded)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1))
    return min(1.0, 2.0 * tail / 2 ** n)


def bootstrap_median_ci(values: Sequence[float], resamples: int = 2000, seed: int = 1) -> Tuple[float, float]:
    """Percentile bootstrap 95 % interval of the median (seeded; the same algorithm is used for every interval of these scripts)."""
    rng = random.Random(seed)
    medians = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        medians.append(statistics.median(sample))
    medians.sort()
    return medians[int(0.025 * resamples)], medians[int(0.975 * resamples)]


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    mx, my = statistics.mean(x), statistics.mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def consistent_denominators(values: Iterable[float], candidates: Iterable[int], tolerance: float) -> List[int]:
    """Denominators n for which every value is k/n within ``tolerance`` (in units of 1/n)."""
    values = list(values)
    return [n for n in candidates if all(abs(v * n - round(v * n)) <= tolerance for v in values)]


# ----------------------------------------------------------------------------- files
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Path relative to the repository root when inside it, absolute otherwise (e.g. a temporary output directory)."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def file_record(path: Path) -> Dict[str, object]:
    return {"path": display_path(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def directory_record(paths: Sequence[Path]) -> Dict[str, object]:
    """One record for a set of files: count plus the SHA-256 of the sorted ``name:hash`` lines."""
    lines = sorted(f"{p.name}:{sha256_file(p)}" for p in paths)
    return {"files": len(paths), "sha256_of_sorted_name_hash_lines": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
            "example": display_path(paths[0]) if paths else None}


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([format_number(v) for v in row])


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def format_number(value: object) -> object:
    """Deterministic float formatting (12 significant digits) so that regenerated files hash identically."""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.12g}"
    return value


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def optional_float(value: Optional[str]) -> float:
    return float(value) if value not in (None, "") else float("nan")
