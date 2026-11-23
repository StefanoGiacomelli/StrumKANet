# Recording-disjoint re-evaluation (new run, 27 to 28 October 2026)

First execution of `main_recording_disjoint.py` on the released Strummin' dataset (Zenodo 10.5281/zenodo.15862786). **Every number here is a new run**; it is shown next to, never in place of, the stored 2025 results in `experiments/results/` and `results/*.csv`. Protocol, results, reading and caveats: [`docs/PROTOCOL.md`](../../docs/PROTOCOL.md), section 12.

| Folder | Grouping unit of the folds | Models | Runs |
|---|---|---|---|
| `primary_seed42/` | recording (rows sharing a YouTube URL are one recording), partition seed 42 | nine stored configurations x {WavKAN, MLP, linear head}, majority and logistic-regression baselines | 135 |
| `control_performer_seed42/` | performer (all recordings of a performer held out together; stricter) | WavKAN head, nine configurations, baselines | 45 |
| `control_excerpt_seed42/` | excerpt (siblings of a held-out excerpt are in training; looser) | WavKAN head, nine configurations, baselines | 45 |
| `recording_seed43/`, `recording_seed44/` | recording, partition seeds 43 and 44 | WavKAN head, three configurations, baselines | 15 each |
| `report/` | tables built by `analysis/recording_disjoint_report.py` from the folders above | | |
| `variants/` | recipe and descriptor variants under the same folds (`main_variants.py`; partition seeds 42 to 44, controls); see [`variants/README.md`](variants/README.md) | recipe variants of two RP networks, two tempogram-ratio networks, heads on descriptor vectors, logistic regressions | 290 training runs, 95 method evaluations |
| `dataset_facts.json` | facts of the released corpus computed by `analysis/dataset_facts.py` (header of `dataset.csv`, 257 rows = 255 audio files, mono audio, N = 5,604 excerpts, performers) | | |

Each run folder holds `summary.csv` (accuracy with recording-level cluster-bootstrap interval, recording-level accuracy, balanced accuracy, macro-F1, recall per class), `folds.csv` (per fold), `confusion_matrices.json` (pooled out-of-fold counts, rows = true class), `partition_k5_seed<seed>.json` (the folds: training, validation and test groups of every fold), `configs.json` (the configurations as run), `validation_curves.csv` (per-epoch validation accuracy and loss of every run), `predictions/<method>.csv.gz` (one row per held-out excerpt: id, recording, label, prediction, three probabilities, fold) and `PROVENANCE.json` (command, parameters, inputs and code SHA-256, `rp_extract` commit and hash, hardware, load, every run with device, threads, timestamps and fit time).

Notes on the files:

* The SHA-256 values listed in `PROVENANCE.json` (`outputs`) refer to the uncompressed prediction files: `gunzip -c predictions/<name>.csv.gz | shasum -a 256`. All 96 recorded outputs were verified against the files in this folder. `validation_curves.csv` and `report/` were produced afterwards (`analysis/recording_disjoint_report.py --write-curves`) and are not listed there; `SHA256SUMS` lists every file of this folder (the subfolder `variants/` included).
* The copied `PROVENANCE.json` files differ from the originals only in the absolute paths of the machine that ran the job, which are replaced by `<scratch>`.
* The per-run result files (`runs/*.json`, 56 MB, with the per-excerpt probabilities that are also in `predictions/`), the worker logs and the model checkpoints (4.9 GB) are not part of the repository.
* The two repeated rows of `dataset.csv` (Seal - Crazy; Pinguini tattici nucleari - Ringo Starr) contribute their excerpts twice, always in the same fold.
