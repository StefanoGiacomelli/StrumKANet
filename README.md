# StrumKANet: A Kolmogorov–Arnold NAS Framework for Strumming Pattern Recognition

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC--BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**StrumKANet** is an open research framework for the classification of categorical guitar strumming patterns from pop/rock audio recordings.  
This repository contains:

- A curated dataset interfac for Strummin' Dataset: annotated pop/rock songs with human-validated rhythmic labels.
- A full PyTorch training pipeline using Kolmogorov–Arnold Networks (KANs) and Convolutional-Attentive encoders.
- Feature extraction (Pre-Processing) scripts and reproducible experiment configurations.


## Project Overview

This work is part of the *Strummin'* Gym platform, which supports technology-enhanced guitar learning.  
In particular, this repository focuses on *Phase 2* of the framework: extracting a **consistent strumming pattern** from real-world pop/rock recordings. Unlike classic tasks such as beat tracking or onset detection, this work targets the **mid-level rhythmic structures** specific to guitar accompaniment. The task is formulated as a **categorical classification** over a reduced set of pedagogically-relevant patterns (binary, quaternary, ternary).

## Repository Structure

```bash
StrumKANet/
├── dataset/
│   ├── dataset.csv                 # Metadata and annotations
│   ├── file_audio/                 # WAV files named as performer-title.wav
│   ├── features/                   # Optional extracted features (.npz) and features plot (.svg)
├── data_processing.py              # Feature extraction and segmentation
├── data.py                         # PyTorch Dataset and LightningDataModule classes
├── experiments/
│   ├── configs/                    # HPO-NAS configurations
│   │   ├── bpm/                    # TO_DO
│   │   ├── patterns/               # Experiments configurations
│   ├── logs/                       # Top-14 training logs
│   ├── results/                    
│   │   ├── ..._figures/            # Plots for paper
│   │   ├── hp_search_analysis.py   # Statistical analysis of HPO-NAS experiment results
│   │   ├── hp_search_results.csv   # Results log for HPO-NAS experiments
│   │   ├── training_analysis.py    # Statistical analysis of Top-14 training pipeline results
│   │   ├── training_results.csv    # Results log for Top-14 training pipelines
│   │   ├── training_log.txt        # Top-14 training pipeline console logs
│   ├── hyper_params_space.py       # Script for generating configs stored in ./experiments/config/...                    
├── model.py                        # ConvAtt Encoders and KAN-based models (PyTorch and Lightning AI)
├── main_hp_nas.py                  # NAS-HPO experiments script
├── main_training.py                # Baseline raining script
├── requirements.txt
├── start_logger.sh                 # Script for monitoring Training pipelines in real-time (or for logs analysis)
├── globals.py                      # Environment global parameters
├── WavKAN.py                       # Wavelet-based Kolmogorov-Arnold Network (PyTorch)
└── README.md
```

## How to Use

1. Clone repository:
    ```bash
    git clone https://github.com/StefanoGiacomelli/StrumKANet.git
    ```
    Move inside

2. Download and add dataset contents to the workspace:
    ```bash
    wget -O strummin_dataset.zip "https://zenodo.org/records/15862786/files/strummin_dataset.zip?download=1"
    ```
    or
    ```bash
    curl -L -o strummin_dataset.zip "https://zenodo.org/records/15862786/files/strummin_dataset.zip?download=1"
    ```
    unzip and move the ```"./dataset"``` folder (and all its contents) inside the workspace. 
    **NOTE**: inside the folder we provide ```data_processing.py``` and ```data.py``` for third-usage cases (you can safely delete them)

3. Clone ```rp_extract``` repository inside the main workspace:
    ```bash
    git clone https://github.com/tuwien-musicir/rp_extract.git
    ```

3. You are ready!

## Baseline Checkpoint Links

To_Do

## 📄 License

This dataset and code are released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

## 📚 Citation

If you use this code or dataset, please cite:

```bibtex
@article{pennese2025strumkan,
    author    = {Pennese, Marco and Giacomelli, Stefano and Rinaldi, Claudia},
    title     = {A Kolmogorov Arnold Network NAS Framework for Strumming Pattern Recognition in Technology-Enhanced Pop/Rock Music Education},
    note      = {under peer-review for IEEE International Symposium on the Internet of Sounds (IS2)},
    year      = {2025}
}

@dataset{pennese_2025_15862786,
    author    = {Pennese, Marco and Giacomelli, Stefano and Rinaldi, Claudia},
    title     = {The Strummin' Dataset: an international pop/rock curated audio selection for strumming patterns recognition},
    month     = jul,
    year      = 2025,
    publisher = {Zenodo},
    version   = {version\_1},
    doi       = {10.5281/zenodo.15862786},
    url       = {https://doi.org/10.5281/zenodo.15862786},
}
```
