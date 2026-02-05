# config.py
from typing import Tuple
import time
from pathlib import Path

############################
######### Basic Settings #########
############################
SEED: int = int(time.time()) % 10000 

# Data/Model Basic Configuration
N_BANDS: int = 5
N_CHANNELS: int = 32
IMG_H: int = 100
IMG_W: int = 100

# Optional: select a subset of band indices to use (e.g. [2,3,4])
# Set to None to use all bands.
SELECT_BANDS = [2, 3, 4]  # use bands indices 2,3,4 (3-band mode)

EPOCHS = 3
TRIAL = 4
LR = 1e-3
RATIO_VALID = 0.2
RATIO_TEST = 0.1
BATCH_SIZE = 64
LOSS_FN: str = "BCEWithLogitsLoss"

# Save Related Settings
SAVE_EVERY = 5

############################
######### Model Related #########
############################
# Mapper 종류: "rescnn" "cnn" "dilatedcnn" "pyramidcnn" "sharedcnn"
MAPPER_NAME: str = "rescnn"

# shared MLP
SHARED_OUT_DIMS: int = 128
HIDDEN_DIMS: Tuple[int, int, int] = (256, 256, 256)

# Number of classifiers (heads)
NUM_HEADS: int = 7

# Paths
# DATA_DIR = "data_torch_small"
DIR_DATA = Path("data")
DIR_INFOVEC = Path("data_infovec/InfoVec.npy")
DIR_OUTPUT = Path("results")
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
    