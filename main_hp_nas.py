import os
from functools import partial
import yaml
import csv
import torch
import numpy as np
import random
from glob import glob
from datetime import datetime
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from data import PopMusicDataModule
from data_processing import rhythm_features
from model import Model, init_modules
from globals import SEED, MAIN_FOLDER, CSV_PATH, AUDIO_FOLDER, SPLIT_RATIO, SHUFFLE, BATCH_SIZE, NUM_WORKERS, FEATURES_MAP, LR


torch.set_float32_matmul_precision('high')


# -------------------- CONFIG --------------------
CONFIG_DIR = "./experiments/configs/pattern"            # or ./bpm
LOGS_DIR = os.path.join(MAIN_FOLDER, "logs")
RESULTS_DIR = os.path.join(MAIN_FOLDER, "results")
CSV_RESULTS = os.path.join(RESULTS_DIR, "hp_search_results.csv")


# --------------------- SEED ---------------------
def set_all_seeds(seed=42):
    seed_everything(seed, workers=True)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------- CSV HANDLING -----------------
def load_results(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["experiment_id", "test_loss", "test_metric", "test_status"])
        return {}
    
    results = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["experiment_id"]] = row
    return results


def append_result_to_csv(csv_path, result_row):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(result_row)


# ----------------- MODEL HANDLING -----------------
def get_extractors(features):
    if features == [] or features is None:
        return []
    if isinstance(features, str):
        features = [features]
    
    extractors = set()
    for feature in features:
        if feature in FEATURES_MAP:
            extractor = FEATURES_MAP[feature]
            extractors.add(extractor)
    
    return list(extractors)


def configure_features_and_preprocessor(config_data: dict):
    raw_features = config_data.get("features", None)
    features = None if isinstance(raw_features, str) and raw_features.lower() == "none" else raw_features

    preprocessor = None
    if config_data.get("preprocessor", False):
        if features is None:
            raise ValueError("You enabled 'preprocessor' but did not specify any 'features'.")
        extractors = get_extractors(features)
        preprocessor = partial(rhythm_features, extractors=extractors)
    else:
        if features is not None:
            print("[WARNING] 'features' were specified but 'preprocessor' is disabled. Features will be ignored.")
            features = None

    return features, preprocessor


# ------------------- MAIN LOOP -------------------
def main():
    #set_all_seeds(SEED)
    
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    existing_results = load_results(CSV_RESULTS)
    config_files = sorted(glob(os.path.join(CONFIG_DIR, "*.yaml")))

    for config_file in config_files:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        exp_id = config["experiment_id"]
        task = config["task"]
        total_experiments = len(config_files)
        current_experiment = config_files.index(config_file) + 1
        print(f"[INFO] Starting Experiment {current_experiment}/{total_experiments}: {exp_id}")

        if exp_id in existing_results and existing_results[exp_id]["test_status"] == "True":
            print(f"[SKIP] {exp_id} already completed.")
            continue

        try:
            set_all_seeds(SEED)
            
            logger = TensorBoardLogger(save_dir=os.path.join(LOGS_DIR, task), name=exp_id)

            features, preprocessor = configure_features_and_preprocessor(config["data"])
            data_module = PopMusicDataModule(csv_path=CSV_PATH,
                                             audio_folder=AUDIO_FOLDER,
                                             features=features,
                                             segment_duration=config["data"]["segment_duration"],
                                             label_type=config["data"]["label_type"],
                                             bpm_source=config["data"]["bpm_source"],
                                             preprocessor=preprocessor,
                                             split_ratio=SPLIT_RATIO,
                                             shuffle=SHUFFLE,
                                             batch_size=BATCH_SIZE,
                                             num_workers=NUM_WORKERS)
            data_module.setup()

            # Get one batch for modules init inference
            sample_batch = next(iter(data_module.train_dataloader()))
            sample_input = sample_batch["input"]

            # Initialize NNs modules
            encoder_module, kan_module = init_modules(config, sample_input)
            
            model = Model(encoder_module=encoder_module,
                          kan_module=kan_module,
                          downstream_task=config["task"],
                          lr=LR,
                          config=config)

            trainer = Trainer(accelerator='auto',
                              max_epochs=config["epochs"],
                              logger=logger,
                              enable_checkpointing=False,
                              enable_progress_bar=True,
                              log_every_n_steps=10)

            # Train/Dev/Test
            trainer.fit(model, datamodule=data_module)
            trainer.test(model, datamodule=data_module)
            
            # Store Results
            test_loss = model.test_metrics_summary.get("test_loss", float("nan"))
            test_metric = model.test_metrics_summary.get("test_metric", float("nan"))
            append_result_to_csv(CSV_RESULTS, [exp_id, test_loss, test_metric, True])
            
            print(f"[SUCCESS] {exp_id}: Completed.")

        except Exception as e:
            append_result_to_csv(CSV_RESULTS, [exp_id, "", "", False])
            print(e)


if __name__ == "__main__":
    main()
