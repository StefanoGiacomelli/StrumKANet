import csv
import os
import soundfile as sf
from pathlib import Path
from collections import defaultdict
import numpy as np
from datetime import timedelta

# === CONFIG ===
CSV_PATH = './dataset/dataset.csv'
AUDIO_FOLDER = './dataset/file_audio'

# === INIT ===
durations = []
labels_count = defaultdict(int)
missing_files = []

# === PARSE CSV ===
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        title = row[0].strip().replace(" ", "_")
        performer = row[1].strip().replace(" ", "_")
        label = row[6].strip()
        downloaded = row[8].strip()

        if downloaded.lower() != 'true':
            continue

        filename = f"{performer}-{title}.wav"
        filepath = Path(AUDIO_FOLDER) / filename

        if filepath.exists():
            try:
                audio, sr = sf.read(filepath)
                duration_sec = len(audio) / sr
                durations.append(duration_sec)
                labels_count[label] += 1
            except Exception as e:
                print(f"[ERROR] Cannot read {filepath}: {e}")
        else:
            missing_files.append(filename)

# === STATS ===
n_total = len(durations)
dur_mean = np.mean(durations) if durations else 0.0
dur_total_sec = np.sum(durations)
dur_total_fmt = str(timedelta(seconds=int(dur_total_sec)))

# === REPORT ===
print(f"Total valid samples: {n_total}")
print(f"Average duration per file: {dur_mean:.2f} seconds")
print(f"⏱Total duration: {dur_total_fmt} (hh:mm:ss)")

print("\nSamples per class:")
for label in sorted(labels_count.keys()):
    print(f"  Class {label}: {labels_count[label]} samples")

if missing_files:
    print(f"\nMissing {len(missing_files)} audio files:")
    for f in missing_files:
        print(f"  - {f}")