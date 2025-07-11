SEED = 42

# EXPERIMENTS
MAIN_FOLDER = "./experiments"

# DATASET & PRE-PROCESSING
CSV_PATH = "./dataset/dataset.csv"
AUDIO_FOLDER = "./dataset/file_audio/"
BATCH_SIZE = 32
SPLIT_RATIO = (0.8, 0.1, 0.1)
SHUFFLE = True
NUM_WORKERS = 4                                         # 0 for no multiprocessing

SAMPLE_RATE = 48000
AUDIO_EXT = ".wav"                                      # ...used by data_processing.py
SVG_FOLDER = "./dataset/analysis/"                      # ...used by data_processing.py    
EXTRACTORS = ["librosa", "rp_extract", "essentia"]      # ...used by data_processing.py

FEATURES_MAP = {"onset_strength": "librosa",
                "tempogram": "librosa",
                "tempogram_ratio": "librosa",
                "beats": "librosa",
                "bpm_librosa": "librosa",
                "bpm_essentia": "essentia",
                "ticks": "essentia",
                "confidence": "essentia",
                "bpm_estimates": "essentia",
                "rp": "rp_extract",
                "rh": "rp_extract",
                "tssd": "rp_extract",
                "trh": "rp_extract"}

# MODEL
LR = 5e-4                                               # Adam learning rate
BETAS = (0.9, 0.999)                                    # Adam beta_1 and beta_2 parameters
WEIGHT_DECAY = 1e-4                                     # Weight decay (L2 regularization)

# MODEL METRICS
NUM_CLASSES = 3                                         # 0 = binary pulse - binary subdivision, 1 = binary pulse - quarters subdivision
                                                        # 2 = ternary pulse
DROP_PROB = 0.2
