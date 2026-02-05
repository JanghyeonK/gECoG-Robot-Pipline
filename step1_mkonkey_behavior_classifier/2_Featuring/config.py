from pathlib import Path

# =====================================================
# ---------------- Global Flags ----------------
# =====================================================
FLAG_TIME = False                # Print Wavelet execution time
FLAG_TEST_LOAD = False           # Test loading of saved files

FLAG_IMAGE_SAVE = True           # Save preview images
INTERVAL_IMAGE_SAVE = 3000       # Image save interval (in windows)

FLAG_MONITOR = True              # Print progress
INTERVAL_MONITOR = 500           # Progress print interval (in windows)

# =====================================================
# ---------------- Data & Window Settings ----------------
# =====================================================
NUM_PROC = 3                     # Number of parallel processes
SIZE_BATCH = 1000                # Number of windows to save at once
SIZE_WINDOW = 250                # ECoG window size (samples)
SIZE_STRIDE = 25                 # ECoG stride
SIZE_IMG = 100                   # Wavelet image size
NUM_CLASS = 7

# =====================================================
# ---------------- Featuring Settings ----------------
# =====================================================
BAND_NAME = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_HZ = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 80)]
FS = 256
NUM_FREQ_PER_BAND = 20

# =====================================================
# ---------------- Directory Setup ----------------
# =====================================================
DIR_LABEL = Path("data/label")
DIR_ECOG  = Path("data/ecog")
#DIR_OUT   = Path("result")

# DIR_SLICED = Path("data/ecog_sliced.mat")
# DIR_OUT = Path("/media/hanlab/새 볼륨/janghyeonk/2.classifier/data_torch_V0")

DIR_SLICED = Path("data/ecog_sliced.mat")
DIR_OUT = Path("result/data_251230")
#DIR_OUT = Path("/media/hanlab/새 볼륨/janghyeonk/2.classifier/data_torch_V1")

# DIR_SLICED = Path("data/ecog_sliced_AAFT_V2.mat")
# DIR_OUT = Path("/media/hanlab/새 볼륨/janghyeonk/2.classifier/data_torch_V2")

DIR_OUT.mkdir(exist_ok=True, parents=True)

# =====================================================
# ---------------- Realtime Settings ----------------
# =====================================================
UDP_IP = "127.0.0.1"
UDP_PORT = 12345
UDP_PACKET_FORMAT = "=32H H H H H"
N_CHANNEL = 32

ZMQ_PUB_PORT = 5555
