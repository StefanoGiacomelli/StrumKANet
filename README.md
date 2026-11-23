# StrumKANet: A Kolmogorov–Arnold NAS Framework for Strumming Pattern Recognition

<p align="center">
  <img src="assets/disim_logo.png" alt="DISIM - Department of Information Engineering, Computer Science and Mathematics" height="90">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/univaq_logo.png" alt="University of L'Aquila" height="90">
</p>

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC--BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10 | 3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/)
[![Dataset DOI](https://img.shields.io/badge/dataset%20DOI-10.5281%2Fzenodo.15862786-blue)](https://doi.org/10.5281/zenodo.15862786)

**StrumKANet** is an open research framework for the classification of categorical guitar strumming patterns from pop/rock audio recordings. It is a PyTorch/Lightning pipeline that turns 10-second excerpts into rhythm descriptors, encodes them with small one- and two-dimensional convolutional encoders (with optional attention) and classifies them with a wavelet Kolmogorov–Arnold (Wav-KAN) head, together with the scripted 468-configuration NAS/HPO campaign, its stored logs and results, the scripts that regenerate the reported tables and analyses from those files, and a recording-disjoint re-evaluation of the selected models (run in 2026). It supports the IEEE IS2 2025 paper *A Kolmogorov Arnold Network NAS Framework for Strumming Pattern Recognition in Technology-Enhanced Pop/Rock Music Education* ([DOI](https://doi.org/10.1109/IS264627.2025.11284580)). The work is part of the *Strummin' Gym* platform for technology-enhanced guitar learning (*Phase 2*: extracting a consistent strumming pattern from real recordings, a mid-level rhythmic target above beat/onset tracking and below symbolic transcription).

## What you can do

- Classify 10-s excerpts into three strumming-pattern classes with the stored model family (descriptor → encoder → WavKAN head).
- Compute rhythm descriptors: onset strength, tempogram, tempogram ratio, beats (LibROSA); Rhythm Patterns, Rhythm Histogram, TSSD, TRH (`rp_extract`); ticks and BPM estimates (Essentia); or feed the raw waveform (mono in the released corpus).
- Reproduce or extend the 468-configuration grid (`experiments/hyper_params_space.py`) and run the two scripted stages (`main_hp_nas.py`: one-epoch screen; `main_training.py`: the 14 selected models).
- **Regenerate the reported numbers from the stored files** (Stage-1 and Stage-2 tables, the 468-run paired analysis, validation intervals) into `results/*.csv` with a provenance record: `python analysis/regenerate_results.py`.
- Run dataset-free smoke tests of the model, the loss, the WavKAN unit, the partitions and the analysis chain: `python tests/test_smoke.py`.
- Compare training recipes and descriptors under the same recording-disjoint folds (`main_variants.py`, run in 2026): descriptor augmentation, averaging of several RP views, heads on tempo-normalized descriptor vectors, the stored tempogram-ratio networks, logistic regressions on cached descriptors, with paired intervals against the stored predictions (`analysis/variants_report.py`; results in `results/recording_disjoint_2026/variants/`, summary below).
- Use the protocol options added after the 2025 release (all **default-off**; the stored 2025 protocol stays the default): a seeded or recording-disjoint split drawn once, best-checkpoint evaluation, saved test predictions and confusion matrices, a CPU option, and a one-command **recording-disjoint re-evaluation** (`main_recording_disjoint.py`), executed in 2026 (results in `results/recording_disjoint_2026/`, summary below).

## Supported items

| Item | Scope (verified in the code and the stored files) |
|---|---|
| Task | 3 classes: 0 = binary pulse with binary subdivision, 1 = binary pulse with quaternary (quarters) subdivision, 2 = ternary pulse (`globals.py`) |
| Data | Strummin' Dataset, Zenodo [10.5281/zenodo.15862786](https://doi.org/10.5281/zenodo.15862786): 257 rows in `dataset.csv` (89 / 99 / 69; 255 distinct audio files), WAV **mono** 48 kHz 16-bit (verified on the archive), 5,604 ten-second excerpts, CC BY 4.0 (record metadata); the three class labels (one per recording) were validated by Marco Pennese, a music teacher and co-author of the dataset |
| Descriptors | `raw` (waveform, 1 × 480,000 for the released mono files), `onset_strength` (1 × 938), `tempogram` (1 × 384 × 938), `tempogram_ratio` (1 × 13 × 938), `beats`, `ticks`, `bpm_estimates` (variable length), `rp` (1 × 24 × 60), `rh` (1 × 60), `tssd` (1 × 24 × 49), `trh` (1 × 60 × 7); 938 frames per excerpt is derived, not stored |
| Encoders | `ConvEncoder1D` (pool to 4,096) and `ConvEncoder2D` (pool to 64 × 64), 16 or 32 channels, kernels 3/5/7 (2D: 3×3, 5×5, 7×7), 128 to 2,689 parameters in the 14 stored models |
| Attention | 1D: sinusoidal positional encoding, squeeze-and-excitation (r = 16), depthwise-separable convolutional attention (`convattn`, kernel 3); 2D: axial (height axis only) and multi-head self-attention (8 heads, d_h = 32, 32 × 32 tokens) |
| Head | `PatternKAN`: one Wav-KAN layer (mexican-hat wavelet, one wavelet per edge, batch normalization), 786,438 or 1,572,870 parameters (`12 D + 6`, D = 65,536 / 131,072) |
| Loss | cross-entropy, or the tolerance-smoothed likelihood (τ = 0.3; classes 0 and 1 mutually tolerated) |

## Installation

Python 3.10 or 3.11. The 2025 campaign ran on a Linux host (Python 3.11, one CUDA GPU, Lightning 2.x; other versions were not recorded). The smoke tests and the regeneration scripts were run in the environment listed in [`requirements-tested.txt`](requirements-tested.txt) (Python 3.10.11, CPU).

```bash
git clone https://github.com/StefanoGiacomelli/StrumKANet.git
cd StrumKANet
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # core: model, data, entry points, analysis, tests
# optional, only for the descriptors that need them (see requirements-features.txt):
pip install -r requirements-features.txt   # essentia: 'ticks', 'bpm_estimates'
git clone https://github.com/tuwien-musicir/rp_extract.git   # 'rp', 'rh', 'tssd', 'trh' (clone into the repository root)
```

Dataset (needed only to train or to re-evaluate; **not** needed for the tests and the analysis of the stored files). The archive is 21.6 GB (Zenodo md5 `7eb77b79a6a05839cf671039acd8f154`):

```bash
wget -O strummin_dataset.zip "https://zenodo.org/records/15862786/files/strummin_dataset.zip?download=1"
# or: curl -L -o strummin_dataset.zip "https://zenodo.org/records/15862786/files/strummin_dataset.zip?download=1"
```

Unzip and move the `./dataset` folder (`dataset.csv`, `file_audio/`, `analysis/`) into the repository root. **Notes (verified on the archive):** (i) the archive also holds older copies of `data.py`, `data_processing.py`, `globals.py` and `requirements.txt` at its root: do not copy them over the files of this repository; (ii) it stores UTF-8 file names without the zip UTF-8 flag, so extract it with `unzip` or `7z`; Python's `zipfile` decodes the 22 names with accents (for example `Modà-Come_un_pittore.wav`) as CP437 (`Mod├á-...`) and the data class then skips those recordings with only a printed warning (`main_recording_disjoint.py` stops with an error instead; repair an extracted folder by renaming each file with `name.encode("cp437").decode("utf-8")`); (iii) the audio is **mono** (48 kHz, 16-bit) although the Zenodo description says two channels; the header of `dataset.csv` is `title,performer,yt_url,bpm_marco,bpm_tunebat,bpm_songBPM,label,ambiguity,downloaded`. All scripts assume the repository root as working directory (`./experiments`, `./dataset`).

Baseline checkpoints (the **best-validation** checkpoints of the 14 retrained models; restricted access): [request access](https://univaq-my.sharepoint.com/:f:/g/personal/stefano_giacomelli_graduate_univaq_it/Etkw9KM2gqJIprz_hfFaYLsBd_aCHeB7WhYfSEq8nmR_jQ?e=hUGPFn). They reproduce the validation column, not the weights behind the reported test numbers (see *Current boundaries*).

> Device note: the model cannot run on Apple MPS as it is (`AdaptiveAvgPool` with non-divisible sizes is unsupported there). Use `--accelerator cpu` on a Mac; the legacy scripts default to `accelerator='auto'`. Measured in 2026: with the pools replaced by matrix products MPS is only 1.3 to 1.5 times faster than one 10-thread CPU process for the WavKAN head and gives the same aggregate throughput as five CPU processes (`--workers 5 --threads 2`), so the 2026 re-evaluation ran on the CPU. On the CPU build used there (PyTorch 2.13, macOS arm64) `--no-nnpack` makes the one-dimensional onset-strength model 10 times faster.

## Quick start

```bash
python tests/test_smoke.py                    # no dataset needed; expected: "OK" (all tests pass)
python analysis/regenerate_results.py         # stored files -> results/*.csv + results/PROVENANCE.json (seconds)
python main_hp_nas.py                         # Stage 1: 468 configurations x 1 epoch (needs dataset + rp_extract)
python main_training.py                       # Stage 2: the 14 selected models (needs Stage-1 results)
./start_logger.sh                             # TensorBoard on ./experiments/logs
```

`analysis/regenerate_results.py` prints the written files and checks the key reported numbers (for example, the 14 Stage-2 identifiers equal the union of the two Stage-1 top-10 lists, and the 468 grid is 6 × 48 + 5 × 36).

## Repository layout

```text
StrumKANet/
├── main_hp_nas.py                 # Stage 1: NAS/HPO screen (468 configurations x 1 epoch)
├── main_training.py               # Stage 2: top-10 by test accuracy + top-10 by test loss -> 14 models, early stopping
├── main_recording_disjoint.py     # (new, optional) recording-disjoint re-evaluation, one command (K-fold, controls, parallel workers)
├── main_variants.py               # (new, optional) recipe / descriptor variants, vector models and classical models under the same folds
├── model.py                       # encoders, attention modules, KAN/MLP heads, Lightning module
├── WavKAN.py                      # wavelet KAN layer (adapted from the Wav-KAN sample code; see its header)
├── data.py                        # Dataset / LightningDataModule (legacy split + default-off options)
├── data_processing.py             # rhythm descriptors (LibROSA, rp_extract, Essentia)
├── data_analysis.py               # dataset statistics (needs the dataset)
├── splits.py  evaluation.py  feature_cache.py   # (new) partitions, metrics/confusion matrices, descriptor cache
├── globals.py                     # global parameters (seed, paths, batch size, optimizer)
├── start_logger.sh                # TensorBoard launcher
├── experiments/
│   ├── hyper_params_space.py      # generates the YAML grid (do not run casually: rewrites configs and cleans the CSV)
│   ├── configs/pattern/           # the 468 YAML configurations of the grid
│   ├── configs/bpm/               # 234 generated BPM-regression configurations, never run (kept, regenerable)
│   ├── logs/pattern/              # TensorBoard logs of the 14 Stage-2 runs
│   └── results/                   # ORIGINAL stored results: hp_search_results.csv, training_results.csv, training_log.txt, figures, analysis scripts
├── analysis/                      # (new) regeneration scripts for the stored tables, recording_disjoint_report.py, variants_report.py
├── results/                       # (new) DERIVED tables regenerated from the stored files + PROVENANCE.json
│   └── recording_disjoint_2026/ # (new) the 2026 recording-disjoint re-evaluation: per-step tables, predictions (gzip), partitions, provenance, report/
│       └── variants/            # (new) variants of the recipe and of the descriptors under the same folds (seeds 42 to 44, controls)
├── tests/test_smoke.py            # (new) dataset-free smoke tests
├── assets/                        # logos
└── requirements*.txt  CITATION.cff  CHANGELOG.md  LICENSE
```

`experiments/results/` holds the **original, unmodified outputs** of the 2025 campaign; `results/` holds **derived** tables that the scripts rebuild from them and, in `results/recording_disjoint_2026/`, the new run of 2026 (labelled as such; analysis scripts: `analysis/recording_disjoint_report.py`, `analysis/dataset_facts.py`).

## Reproducibility

| What | Can it be re-run here? | Notes |
|---|---|---|
| Tables, medians, intervals, paired analysis, validation curves | **yes**, from the stored CSV / TensorBoard files | `python analysis/regenerate_results.py`; standard library plus `pandas`, `pyyaml` (seconds; PyTorch only for the model-complexity table); hashes in `results/PROVENANCE.json` |
| Smoke tests (model, loss, WavKAN, partitions, parallel workers, report scripts, variants) | **yes**, no dataset | `python tests/test_smoke.py` (40 tests, under a minute on CPU) |
| Stage 1 / Stage 2 as executed in 2025 | not locally | dataset (21.6 GB), `rp_extract`, Essentia, a CUDA host; Stage 2 took 38.5 h of logged wall-clock in one process (4.0 min per epoch for the RP and RH models, 6.6 min for the onset-strength and tempogram-ratio models, descriptors computed on the fly); Stage-1 logs were not kept |
| The 14 checkpoints | on request (restricted link) | best-validation weights; the weights behind the test numbers (final epoch) were never saved |
| Seeds | `SEED = 42`, set once per experiment | the partitions were **not** seeded (see below) |
| Recording-disjoint re-evaluation | **yes**, run in 2026 (255 runs, 5.7 h wall-clock on a laptop CPU) | `main_recording_disjoint.py` (below); results and provenance in `results/recording_disjoint_2026/` |
| Variants of the recipe and of the descriptors | **yes**, run in 2026 (290 training runs, 16.5 process-hours of training on a laptop CPU, plus logistic regressions) | `main_variants.py` (below); results and provenance in `results/recording_disjoint_2026/variants/` |

### Recording-disjoint re-evaluation (run in 2026)

```bash
python main_recording_disjoint.py --heads kan mlp linear --workers 5 --threads 2 --no-nnpack    # nine configurations, 5 folds of whole recordings (builds the descriptor cache if missing)
python main_recording_disjoint.py --benchmark                                                    # no dataset: measures seconds per epoch here and prints a runtime estimate
python analysis/recording_disjoint_report.py --run results/recording_disjoint_2026/primary_seed42 --control-excerpt results/recording_disjoint_2026/control_excerpt_seed42 --control-performer results/recording_disjoint_2026/control_performer_seed42
```

The command evaluates the stored configurations (default: the nine whose stored test accuracy exceeds 0.90, eight RP and one onset strength) on **held-out recordings**: class-stratified K-fold of whole recordings (default 5, seed 42; rows sharing a YouTube URL are one recording), a recording-disjoint validation part (one eighth of the training recordings) for early stopping (patience 20, at most 50 epochs, as in Stage 2), testing with the best-validation checkpoint, the stored WavKAN head and, with `--heads`, an MLP and a linear head on the same encoder, a majority-class and a logistic-regression baseline on the same descriptors, accuracy with recording-level (cluster) bootstrap intervals, per-class metrics and confusion matrices. `--grouping performer` and `--grouping excerpt` run the same pipeline with stricter (all recordings of a performer) and looser (excerpt-level) folds. Outputs per run directory (`--out`, default `./results/recording_disjoint`): `summary.csv`, `folds.csv`, `confusion_matrices.json`, `predictions/`, the partition JSON and `PROVENANCE.json` (inputs, code and `rp_extract` hashes, hardware, load, every run); checkpoints go to `experiments/checkpoints/recording_disjoint/`, descriptors to `dataset/feature_cache/`. An interrupted command resumes from the stored runs. Configuration ids contain spaces and brackets: quote them, e.g. `--configs "ENC_chs32_ks_[7, 7]_att_None_smooth_False_rp"`. Every number it produces is a **new run** to be reported next to the stored ones, never in place of them.

**Result of the 2026 run** (partition seed 42; 255 runs; accuracy over all 5,604 excerpts, each held out once; 95 % intervals resample whole recordings):

| Protocol | RP + WavKAN head (mean of 8 configurations) | logistic regression on RP | majority class |
|---|---|---|---|
| stored 2025 "test" (final epoch, 80 % of the test excerpts were training excerpts) | 0.907 to 0.943 | - | - |
| stored 2025 validation (excerpt-level, best epoch, n = 560) | 0.684 to 0.720 | - | - |
| excerpt-level folds (new control) | 0.666 [0.645, 0.687] | 0.593 [0.572, 0.614] | 0.369 |
| **recording-disjoint folds (new)** | **0.460 [0.430, 0.489]** (per configuration 0.438 to 0.476) | 0.441 [0.415, 0.467] | 0.367 [0.305, 0.430] |
| performer-disjoint folds (new control) | 0.456 [0.426, 0.485] | 0.433 [0.405, 0.460] | 0.368 |

* On recordings the models have never seen, the stored RP models classify 0.438 to 0.476 of the excerpts correctly, 9.3 points above the majority class (paired difference +0.093, 95 % interval +0.020 to +0.163) and 0.21 below the same pipeline with excerpt-level folds; the in-corpus values of the stored test columns and the excerpt-level validation values do not transfer to unseen recordings. The onset-strength model is at the majority level (0.388).
* The WavKAN head is not distinguishable from a logistic regression on the flattened RP descriptor (+0.018, interval -0.007 to +0.041) or from a linear head on the same encoder (+0.012, interval -0.005 to +0.028); an MLP head on the same encoder is better (+0.028, interval +0.010 to +0.046). The ternary class is recognized at chance level (recall 0.32; binary 0.43, quaternary 0.59). Holding out all recordings of a performer gives the same accuracy as holding out recordings (+0.004, interval -0.022 to +0.030), and other partition seeds give the same picture (three configurations, seeds 42 to 44: standard deviation 0.007 to 0.022).
* Scope: nine stored configurations (eight attention-free RP encoders and one onset-strength encoder with the convolutional attention module; selected with the in-corpus test accuracy), one training seed per fold, 255 distinct recordings (intervals about ±0.03). The run does not test the attention, kernel and channel comparisons of the 468-run screening grid, which remain screening results.
* Cost: 255 runs in 5.7 h wall-clock on an Apple M4 Pro laptop shared with other jobs (5 processes of 2 threads at `nice -n 5`, 27 process-hours of training, 1-minute load average 10.3 on 14 cores); the earlier single-process estimates for the same runs (`--benchmark`, NNPACK on, lightly or heavily loaded machine) were 11 to 28 h.

### Variants of the recipe and of the descriptors (run in 2026)

```bash
python main_variants.py --build-cache --build-views                      # all fixed-size descriptors and the extra RP caches (dataset and rp_extract needed)
python main_variants.py --variants base aug views --configs "ENC_chs16_ks_[5, 5]_att_None_smooth_False_rp" --heads kan mlp --workers 8 --threads 1 --no-nnpack --out results/variants_seed42
python main_variants.py --variants base bal --configs VEC_tros --heads kan mlp linear --workers 4 --threads 1 --out results/variants_seed42        # heads on a tempo-normalized descriptor vector
python main_variants.py --classical lr_rp lr_tr lr_osscale lr_tros --out results/variants_seed42                                              # logistic regressions on cached descriptors
python analysis/variants_report.py --variants results/variants_seed42 --stored results/recording_disjoint_2026/primary_seed42                  # paired intervals against the stored predictions
```

Same folds and protocol as above, one ingredient changed at a time (class-weighted loss, label smoothing, descriptor augmentation, mixup, averaging of several RP views, early stopping on the validation loss, other RP normalizations, heads on descriptor vectors, other descriptors, seed ensembles), every method compared recording by recording with the stored predictions of the same configuration. Held-out accuracy at partition seed 42 against the stored RP + WavKAN (mean of the eight configurations **0.460 [0.430, 0.489]**, ternary recall 0.32); the 95 % interval of the paired difference excludes zero for every row (full tables, controls, decision rule, caveats and the variants without effect; files: [`results/recording_disjoint_2026/variants/`](results/recording_disjoint_2026/variants)):

| Model | Accuracy [95 % interval] | Balanced acc. | Ternary recall | Difference to the stored RP + WavKAN (0.460) |
|---|---|---|---|---|
| RP, 16 ch 5x5, WavKAN head (stored head), augmentation | 0.504 [0.468, 0.541] | 0.499 | 0.45 | +0.045 [+0.017, +0.073] (+0.066 against the same network without it) |
| RP, 16 ch 5x5, MLP head, augmentation + three views (†) | 0.589 [0.550, 0.627] | 0.583 | 0.53 | +0.129 [+0.096, +0.162] |
| tempogram-ratio network, 16 ch 5x5, WavKAN head (stored configuration, not among the nine) | 0.608 [0.573, 0.641] | 0.612 | 0.65 | +0.149 [+0.114, +0.184] |
| tempogram-ratio network, 32 ch 7x7, MLP head | 0.636 [0.599, 0.674] | 0.638 | 0.66 | +0.177 [+0.141, +0.212] |
| MLP head on 58 tempo-normalized values (tempogram-ratio statistics + onset scale features) | 0.642 [0.604, 0.679] | 0.646 | 0.69 | +0.183 [+0.144, +0.221] |
| logistic regression on the 26 tempogram-ratio statistics | 0.638 [0.598, 0.676] | 0.643 | 0.69 | +0.179 [+0.142, +0.217] |
| **logistic regression on the 58 tempo-normalized values** | **0.664 [0.628, 0.697]** | **0.667** | **0.69** | **+0.204 [+0.167, +0.241]** |

† Decided after seeing the seed-42 results and not run on the other partitions; augmentation and three views were each replicated there.

* The limit of the stored RP networks on unseen recordings is largely the descriptor: tempogram-ratio statistics and onset scale features are relative to the tempo and carry no timbre, and a logistic regression on them recognizes the ternary class (recall 0.69 against 0.32), reaches a recording-level accuracy of 0.81 (against 0.50), and loses only 0.05 between excerpt-level and recording-disjoint folds (RP: 0.15). The result replicates on the partition seeds 43 and 44 (0.655 and 0.656 against 0.435 and 0.422 for the logistic regression on RP), is unchanged when all recordings of a performer are held out (0.653), and falls to the chance level when the labels are permuted among the recordings (0.33 to 0.38). The stored networks on the tempogram ratio (not among the nine; in-corpus test accuracy 0.759 and 0.865) reach 0.608 to 0.636, the 16-channel one with the WavKAN head also on the other two partitions (0.587 and 0.599; mean difference to the stored RP + WavKAN over the three partitions +0.133 [+0.102, +0.165]); the augmentation of the RP networks replicates too (mean difference over the three partitions +0.091 [+0.066, +0.117] for the MLP head and +0.053 [+0.029, +0.078] for the stored WavKAN head, whose interval includes zero on one partition).
* Augmentation of the RP descriptor (tempo, Bark-axis and gain jitter) raises the MLP head of two RP networks by +0.065 [+0.040, +0.092] over the stored MLP head and the stored WavKAN head of the 16-channel network by +0.066 [+0.026, +0.107]; averaging the logits of three RP views of the excerpt adds +0.029 [+0.011, +0.048]; both together stay below the tempo-normalized descriptors.
* No gain in accuracy: class-weighted loss (it moves the ternary recall of the logistic regression from 0.69 to 0.75 at the same accuracy), early stopping on the validation loss, label smoothing and five-seed ensembles on the vector models; the WavKAN head is again not better than an MLP or a linear head (58 values: 0.565 against 0.642 and 0.617), and a logistic regression fitted to convergence (0.664) beats all three heads trained with the stored recipe.
* Scope: 255 recordings (intervals about ±0.03 to ±0.04), one training seed per fold, the variants were compared on the same recordings (a few were decided after seeing earlier results: listed in section 13.1), the stored 2025 results and the defaults of the existing scripts are unchanged.

## Current boundaries

These are facts of the stored protocol, stated so that the numbers can be read correctly:

- **Validation accuracy is an excerpt-level held-out estimate, not a recording-disjoint one.** It is measured on excerpts never used for gradients (best epoch 0.684 to 0.720 for the RP models, n = 560, against a recording-level majority-class rate of 0.388, 99 of 255 distinct recordings), but the other excerpts of the same recordings were in training. The 2026 excerpt-level folds of the same models give 0.650 to 0.682, the recording-disjoint folds 0.438 to 0.476.
- **The "test" columns come from a partition re-drawn at evaluation time.** The split is an unseeded excerpt-level `random_split` that Lightning draws again at `fit` and at `test`, so about 80 % of the test excerpts (448.8 ± 9.0 of 561) were training excerpts of the same run, and the test weights are the final-epoch ones. The Stage-2 test accuracies (up to 0.943) therefore measure how completely the descriptor–encoder–WavKAN chain **fits the corpus**; they are not recording-disjoint generalization estimates.
- **Selection used test metrics.** The 14 retrained models are the union of the top 10 by Stage-1 test accuracy and the top 10 by Stage-1 test loss.
- **The stored runs are excerpt-level, not recording-disjoint.** Excerpts of one recording appear in training, validation and test of the same run (partition indices and test weights were never stored). The 2026 recording-disjoint re-evaluation (above) is the generalization estimate for unseen recordings: 0.438 to 0.476 for the RP models (mean 0.460), 0.388 for the onset-strength model, against a majority-class rate of 0.367.
- **What the stored evidence does support:** the unselected 468-run grid (paired by descriptor, channels, kernel and smoothing) shows that attention in the 2D encoders lowers accuracy (axial -8.7 pp, MHSA -14.6 pp), that RP is the best 2D descriptor, and that kernel width, channel width and smoothing have no detectable effect; the model family fits the corpus almost completely (9 of 14 models above 0.90; final training loss 0.04 to 0.15 for the six unsmoothed ones and 0.30 to 0.33 for the three smoothed ones, whose loss has a higher floor). The stored files contain no MLP-head ablation, per-class metric or baseline other than the majority rate; the 2026 run adds them for the nine configurations (WavKAN, MLP and linear heads, logistic regression, per-class recall, confusion matrices).
- **Reading the losses and the times.** The test loss of smoothed models is the tolerance-smoothed likelihood, not cross-entropy; smoothed and unsmoothed losses are not comparable. The "Stop epoch" is time-to-best plus patience (best index + 21 in all 14 runs). The "Training time" column printed in the IS2 2025 paper (27 to 45 min) is 3.35 to 5.13 times shorter than the wall-clock logged for the same nine runs (1 h 31 min to 3 h 15 min) and cannot be reproduced from the stored files; the logged times, which cover every epoch up to the early-stopping epoch, are the reference; they mostly reflect the on-the-fly descriptor extraction and the stop epoch (the time per epoch depends on the descriptor, not on the encoder) and not the cost of the models.
- **Descriptor details kept for fidelity** (defaults of `rhythm_features` are unchanged): excerpts at 48 kHz are analysed with the 44.1 kHz default and no resampling; `onset_strength` runs with `max_size = 1` (no maximum filter); `rp_extract` analyses a single 5.46 s segment of each 10-s excerpt.
- **Environment:** the environment of the tests and of the regeneration scripts (`requirements-tested.txt`) had no dataset, checkpoints, `rp_extract` or Essentia; the 2026 re-evaluation ran with the dataset and `rp_extract` (commit `f5e6fe6`, no Essentia: it is not needed for the nine configurations); Apple MPS is unsupported; `torchaudio` 2.9 or later no longer provides `torchaudio.info` (the data module falls back to `soundfile`).
- **Third-party code:** `WavKAN.py` is adapted from the Wav-KAN sample code (see its header); `rp_extract` (TU Wien) is cloned at run time and is not redistributed here.

## Funding

Research activities were mainly funded under the Ministerial Decree (DM) 118/2023, Mission 4, Component 1, Investment 4.1 of the National Recovery and Resilience Plan (PNRR) – “PNRR Research” – CUP: E11I23000100001.

Research activities and publications were also partially supported by the European Union under the Italian National Recovery and Resilience Plan (PNRR) of NextGenerationEU, within the partnership on “Telecommunications of the Future” (PE00000001 – RESTART) (the funder stated in the IS2 2025 paper).

<p align="center">
  <img src="assets/italiadomani_logo.png" alt="Italia Domani" height="60">
  &nbsp;&nbsp;
  <img src="assets/next_generation_eu_logo.png" alt="Funded by the European Union - NextGenerationEU" height="60">
  &nbsp;&nbsp;
  <img src="assets/mur_blue_logo.jpg" alt="Ministero dell'Università e della Ricerca" height="60">
</p>

## Related publications and how to cite

The authors of the papers below, and of [`CITATION.cff`](CITATION.cff), are Marco Pennese, Stefano Giacomelli and Claudia Rinaldi. If you use this repository, please cite the IS2 2025 paper (preferred citation); if you also use the data, cite the Strummin' Dataset.

- Pennese, Giacomelli, Rinaldi, *A Kolmogorov Arnold Network NAS Framework for Strumming Pattern Recognition in Technology-Enhanced Pop/Rock Music Education*, 2025 IEEE 6th International Symposium on the Internet of Sounds (IS2), pp. 1–10, DOI [10.1109/IS264627.2025.11284580](https://doi.org/10.1109/IS264627.2025.11284580) (**preferred citation**, also in [`CITATION.cff`](CITATION.cff)).
- Pennese, Giacomelli, Rinaldi, *The Strummin' Dataset*, Zenodo, DOI [10.5281/zenodo.15862786](https://doi.org/10.5281/zenodo.15862786) (cite it when you use the data).
- Pennese, Giacomelli, Rinaldi, *From Neural Rhythm Classification to Embodied Sense-Making: Strummin' Gym as a Browser-Based IoMusT Environment for Guitar Strumming Practice*, 2026 IEEE 7th International Symposium on the Internet of Sounds (IS2), Cannes, France, October 2026. Accepted for presentation (no DOI yet).

```bibtex
@inproceedings{pennese2025strumkan,
  author    = {Pennese, Marco and Giacomelli, Stefano and Rinaldi, Claudia},
  title     = {{A Kolmogorov Arnold Network NAS Framework for Strumming Pattern Recognition in Technology-Enhanced Pop/Rock Music Education}},
  booktitle = {{2025 IEEE 6th International Symposium on the Internet of Sounds (IS2)}},
  pages     = {1--10},
  year      = {2025},
  doi       = {10.1109/IS264627.2025.11284580}
}

@misc{pennese2025strummin_dataset,
  author    = {Pennese, Marco and Giacomelli, Stefano and Rinaldi, Claudia},
  title     = {{The Strummin' Dataset: An International Pop/Rock Curated Audio Selection for Strumming Patterns Recognition}},
  month     = jul,
  year      = {2025},
  publisher = {Zenodo},
  version   = {version\_1},
  doi       = {10.5281/zenodo.15862786},
  url       = {https://doi.org/10.5281/zenodo.15862786}
}

@inproceedings{pennese2026strumming_gym,
  author    = {Pennese, Marco and Giacomelli, Stefano and Rinaldi, Claudia},
  title     = {{From Neural Rhythm Classification to Embodied Sense-Making: Strummin' Gym as a Browser-Based IoMusT Environment for Guitar Strumming Practice}},
  booktitle = {{2026 IEEE 7th International Symposium on the Internet of Sounds (IS2)}},
  address   = {Cannes, France},
  year      = {2026},
  month     = Oct,
  note      = {Accepted for presentation},
  url       = {https://internetofsounds2026.ieee-internetofsounds.org/}
}
```

## License

The code, the documentation and the derived result tables of this repository are released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license; the full legal code is in [`LICENSE`](LICENSE). The Strummin' Dataset is not part of this repository: it is distributed on Zenodo under CC BY 4.0 (record metadata). Third-party components keep their own terms and are not relicensed here: `WavKAN.py` is adapted from the Wav-KAN sample code (see its header), and `rp_extract` (TU Wien) is cloned at run time and is not redistributed.

## Contact

**Stefano Giacomelli**<br>
ICT - Ph.D. Candidate<br>
Department of Information Engineering, Computer Science and Mathematics (DISIM)<br>
University of L'Aquila, Italy

- **Email:** [stefano.giacomelli@graduate.univaq.it](mailto:stefano.giacomelli@graduate.univaq.it)
- **GitHub:** [StefanoGiacomelli](https://github.com/StefanoGiacomelli)
- **ORCID:** [0009-0009-0438-1748](https://orcid.org/0009-0009-0438-1748)
- **Google Scholar:** [Profile](https://scholar.google.com/citations?user=l-n0hl4AAAAJ&hl=en)
