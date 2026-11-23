"""Facts of the released Strummin' Dataset as the pipeline sees it (CSV header, recordings, duplicates, performers, audio format, excerpts).

Usage (needs the dataset; run from the repository root)::

    python analysis/dataset_facts.py [--csv ./dataset/dataset.csv] [--audio ./dataset/file_audio/] [--out results/recording_disjoint_2026/dataset_facts.json]

Every number is computed from the files (nothing is copied from the Zenodo description) with the repository's own ``PopMusicDataset``,
so the excerpt count is exactly the pool of ``data.py`` / ``main_recording_disjoint.py``.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf    # noqa: E402

from analysis.common import write_json    # noqa: E402
from feature_cache import grouping_keys, index_dataset, missing_audio    # noqa: E402
from globals import AUDIO_FOLDER, BATCH_SIZE, CSV_PATH, SPLIT_RATIO    # noqa: E402


def file_id(row: Dict[str, str]) -> str:
    return f"{row['performer'].strip().replace(' ', '_')}-{row['title'].strip().replace(' ', '_')}"


def compute(csv_path: str, audio_folder: str) -> Dict[str, object]:
    raw = Path(csv_path).read_bytes()
    text = raw.decode("utf-8-sig")
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    downloaded = [r for r in rows if r["downloaded"].strip().lower() == "true"]
    ids = [file_id(r) for r in downloaded]
    id_counts = collections.Counter(ids)
    url_rows = collections.Counter(r["yt_url"].strip() for r in downloaded)
    url_titles: Dict[str, set] = collections.defaultdict(set)
    for r in downloaded:
        url_titles[r["yt_url"].strip()].add(f"{r['performer'].strip()} - {r['title'].strip()}")
    performers = collections.Counter(" ".join(r["performer"].casefold().split()) for r in downloaded)
    performer_classes: Dict[str, set] = collections.defaultdict(set)
    for r in downloaded:
        performer_classes[" ".join(r["performer"].casefold().split())].add(r["label"].strip())
    formats = collections.Counter()
    duration_of: Dict[str, float] = {}
    for identifier in id_counts:
        path = Path(audio_folder) / f"{identifier}.wav"
        if path.is_file():
            info = sf.info(str(path))
            duration_of[identifier] = info.frames / info.samplerate
    for identifier in ids:                                                              # over rows: a repeated row counts twice, as in the dataset statistics
        path = Path(audio_folder) / f"{identifier}.wav"
        if path.is_file():
            info = sf.info(str(path))
            formats[f"{info.samplerate} Hz, {info.channels} channel(s), {info.subtype}, {info.format}"] += 1
    durations = [duration_of[i] for i in ids if i in duration_of]
    unique_durations = list(duration_of.values())
    seen = set(duration_of)
    dataset = index_dataset(csv_path, audio_folder, 10.0, "pattern")
    groups, labels = dataset.partition_keys()
    excerpt_ids = [dataset.excerpt_id(i) for i in range(len(dataset))]
    per_recording = collections.Counter(groups)
    n = len(dataset)
    train, val = int(SPLIT_RATIO[0] * n), int(SPLIT_RATIO[1] * n)
    recording_keys = grouping_keys(dataset, "recording")
    label_of_recording = {g: l for g, l in zip(groups, labels)}
    hms = lambda s: f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}"
    return {
        "dataset_csv": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "header_line": text.splitlines()[0], "columns": header,
                        "column_index": {c: i for i, c in enumerate(header)}, "line_terminator": "CRLF" if b"\r\n" in raw else "LF", "rows": len(rows),
                        "downloaded_values": dict(collections.Counter(r["downloaded"] for r in rows)),
                        "label_values_by_row": dict(sorted(collections.Counter(r["label"] for r in rows).items())),
                        "ambiguity_values": dict(collections.Counter(r["ambiguity"] for r in rows))},
        "recordings": {"rows": len(downloaded), "distinct_audio_files": len(id_counts), "rows_sharing_a_file": {k: v for k, v in id_counts.items() if v > 1},
                       "urls_shared_by_several_rows": {u: sorted(url_titles[u]) for u, c in url_rows.items() if u and c > 1},
                       "distinct_recordings_after_merging_shared_urls": len(set(recording_keys)),
                       "downloaded_rows_without_wav": missing_audio(csv_path, audio_folder), "wav_files_in_folder": len(list(Path(audio_folder).glob("*.wav")))},
        "classes": {"rows": dict(sorted(collections.Counter(int(r["label"]) for r in downloaded).items())),
                    "distinct_audio_files": dict(sorted(collections.Counter(label_of_recording.values()).items()))},
        "performers": {"distinct_casefolded": len(performers), "with_two_or_more_recordings": sum(1 for v in performers.values() if v >= 2),
                       "recordings_by_those_performers": sum(v for v in performers.values() if v >= 2), "max_recordings_of_one_performer": max(performers.values()),
                       "performers_with_more_than_one_class": sum(1 for s in performer_classes.values() if len(s) > 1)},
        "audio": {"formats_over_rows": dict(formats), "duration_total_over_rows_s": sum(durations), "duration_total_over_rows_hms": hms(sum(durations)),
                  "duration_mean_over_rows_s": statistics.mean(durations), "duration_min_s": min(durations), "duration_max_s": max(durations),
                  "duration_total_distinct_files_hms": hms(sum(unique_durations)), "audio_bytes_distinct_files": sum((Path(audio_folder) / f"{i}.wav").stat().st_size for i in seen)},
        "excerpts": {"window_s": 10.0, "rule": "windows 1 .. W-2 of the W complete windows (first and last dropped); recordings with W <= 2 give none",
                     "pool_size_N": n, "distinct_excerpt_ids": len(set(excerpt_ids)), "excerpts_per_class": dict(sorted(collections.Counter(labels).items())),
                     "per_row_min": min(per_recording.values()), "per_row_median": statistics.median(per_recording.values()), "per_row_max": max(per_recording.values()),
                     "per_row_mean": n / len(per_recording), "excerpt_level_majority_rate": max(collections.Counter(labels).values()) / n,
                     "recording_level_majority_rate_distinct_files": max(collections.Counter(label_of_recording.values()).values()) / len(label_of_recording)},
        "legacy_partition_sizes": {"train": train, "val": val, "test": n - train - val, "steps_per_epoch_batch_32": math.ceil(train / BATCH_SIZE)},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--audio", default=AUDIO_FOLDER)
    parser.add_argument("--out", default=None, help="write the facts as JSON (default: print only)")
    args = parser.parse_args(argv)
    facts = compute(args.csv, args.audio)
    if args.out:
        write_json(Path(args.out), facts)
    print(json.dumps(facts, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
