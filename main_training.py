import os
import csv
import re
import yaml
import pandas as pd
import torch
import random
import numpy as np
from glob import glob
from functools import partial
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from data import PopMusicDataModule
from data_processing import rhythm_features
from model import Model, init_modules
from globals import SEED, MAIN_FOLDER, CSV_PATH, AUDIO_FOLDER, SPLIT_RATIO, SHUFFLE, BATCH_SIZE, NUM_WORKERS, FEATURES_MAP, LR


torch.set_float32_matmul_precision('high')


CONFIG_DIR = "./experiments/configs/pattern"
LOGS_DIR = os.path.join(MAIN_FOLDER, "logs")
RESULTS_DIR = os.path.join(MAIN_FOLDER, "results")
CSV_NAS_RESULTS = os.path.join(RESULTS_DIR, "hp_search_results.csv")
CSV_TRAINING_RESULTS = os.path.join(RESULTS_DIR, "training_results.csv")
TOP_K = 10


# -------------------- UTILS --------------------
def set_all_seeds(seed=42):
    seed_everything(seed, workers=True)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_experiment_id(exp_id):
    # Replace [num, num] with num-num
    exp_id = re.sub(r"\[(\d+),\s*(\d+)\]", r"\1-\2", exp_id)
    return exp_id


def append_training_result(csv_path, exp_id, test_loss, test_metric, stopped_epoch):
    header = ["experiment_id", "test_loss", "test_metric", "stopped_epoch"]
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([exp_id, test_loss, test_metric, stopped_epoch])


def load_completed_experiments(csv_path):
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    return set(df["experiment_id"].tolist())


# ----------------- FEATURE HANDLING -----------------
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
    return features, preprocessor


# ----------------- MAIN -----------------
def main():
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(CSV_NAS_RESULTS)
    df = df[df["test_status"] == True]

    top_acc = df.sort_values("test_metric", ascending=False).head(TOP_K)
    top_loss = df.sort_values("test_loss", ascending=True).head(TOP_K)
    selected = pd.concat([top_acc, top_loss]).drop_duplicates(subset=["experiment_id"])
    experiment_ids = selected["experiment_id"].tolist()

    config_files = sorted(glob(os.path.join(CONFIG_DIR, "*.yaml")))
    config_map = {os.path.splitext(os.path.basename(p))[0]: p for p in config_files}

    completed_experiments = load_completed_experiments(CSV_TRAINING_RESULTS)

    # Check if experiment needs to be skipped
    for i, exp_id in enumerate(experiment_ids):
        exp_id_sanitized = sanitize_experiment_id(exp_id)
        if exp_id_sanitized in completed_experiments:
            print(f"[SKIP] {exp_id_sanitized} already completed.")
            continue

        print(f"[{i+1}/{len(experiment_ids)}] Starting: {exp_id_sanitized}")
        config_path = config_map.get(exp_id)
        if config_path is None:
            print(f"[SKIP] Config not found for {exp_id}")
            continue

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        try:
            set_all_seeds(SEED)

            logger = TensorBoardLogger(save_dir=os.path.join(LOGS_DIR, config["task"]), name=exp_id_sanitized)

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

            batch = next(iter(data_module.train_dataloader()))
            encoder_module, kan_module = init_modules(config, batch["input"])

            model = Model(encoder_module=encoder_module,
                          kan_module=kan_module,
                          downstream_task=config["task"],
                          lr=LR,
                          config=config)

            checkpoint_cb = ModelCheckpoint(dirpath=os.path.join(MAIN_FOLDER, "checkpoints"),
                                            filename=f"{exp_id_sanitized}" + "-{epoch:02d}-{val_acc:.4f}",
                                            monitor="val_acc",
                                            mode="max",
                                            save_top_k=1,
                                            save_weights_only=False)

            early_stop_cb = EarlyStopping(monitor="val_acc", patience=20, mode="max", verbose=True)

            trainer = Trainer(accelerator='auto',
                              max_epochs=50,
                              callbacks=[checkpoint_cb, early_stop_cb],
                              logger=logger,
                              enable_checkpointing=True,
                              enable_progress_bar=True,
                              log_every_n_steps=10)

            trainer.fit(model, datamodule=data_module)
            trainer.test(model, datamodule=data_module)

            
            test_summary = model.test_metrics_summary
            stopped_epoch = trainer.current_epoch
            append_training_result(CSV_TRAINING_RESULTS,
                                   exp_id,
                                   test_summary.get("test_loss", -1.0),
                                   test_summary.get("test_metric", -1.0),
                                   stopped_epoch)

            print(f"[SUCCESS] {exp_id_sanitized}: Completed.")

        except Exception as e:
            print(f"[ERROR] {exp_id_sanitized}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
