from pathlib import Path
# =====================================================
# ---------------- Global Flags ----------------
# =====================================================
FLAG_TIME = False                # Print wavelet execution time
FLAG_TEST_LOAD = False           # Test if saved files can be loaded

FLAG_IMAGE_SAVE = True           # Whether to save preview images
INTERVAL_IMAGE_SAVE = 100          # Image save interval (window units)

FLAG_MONITOR = True              # Whether to print progress
INTERVAL_MONITOR = 100           # Progress print interval (window units)

# =====================================================
# ---------------- Data & Window Settings ----------------
# =====================================================
NUM_PROC = 1                     # Number of parallel processes
SIZE_BATCH = 1000                # Number of windows to save at once
SIZE_WINDOW = 250                # ECoG window size (samples)
SIZE_STRIDE = 25                 # ECoG stride
SIZE_IMG = 100                   # Wavelet image size

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
DIR_OUT   = Path("result")
DIR_OUT.mkdir(exist_ok=True, parents=True)

# =====================================================
# ---------------- Realtime Settings ----------------
# =====================================================
UDP_IP = "127.0.0.1"
UDP_PORT = 12345
UDP_PACKET_FORMAT = "=32H H H H H"
N_CHANNEL = 32

ZMQ_PUB_PORT = 5555
