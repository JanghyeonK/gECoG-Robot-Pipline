import random, torch
from multiprocessing import Pool
import torch.multiprocessing as mp
from functions.wavelet import preload_wavelets
from functions.dataset_builder import *
from config import *
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_data():
    mp.set_start_method('spawn', force=True)

    preload_wavelets(fs=FS, band_list=BAND_HZ, n_freqs_per_band=NUM_FREQ_PER_BAND, device=str(DEVICE))

    csv_files = sorted(list(DIR_LABEL.glob("*.csv")))
    mat_files = sorted(list(DIR_ECOG.glob("*.mat")))
    num_file = min(len(csv_files), len(mat_files))

    file_pairs = [(csv_files[i], mat_files[i], i + 1, DEVICE) for i in range(num_file)]

    print(f"🧠 Found {num_file} files")
    print(f"🧠 Using {NUM_PROC} parallel processes")

    all_results = []
    for i in range(0, num_file, NUM_PROC):
        batch = file_pairs[i:i+NUM_PROC]
        print(f"\n🚀 Processing batch: {[x[2] for x in batch]}")

        with Pool(processes=NUM_PROC) as pool:
            results = pool.map(process_file, batch)

        all_results.extend(results)
        print("✅ Batch finished")

def run_sliced_data():
    global DEVICE
    mp.set_start_method('spawn', force=True)

    preload_wavelets(fs=FS, band_list=BAND_HZ,
                     n_freqs_per_band=NUM_FREQ_PER_BAND,
                     device=str(DEVICE))

    # Create task list for each class
    args_list = [(str(DIR_SLICED), cls, DEVICE) for cls in range(NUM_CLASS)]

    print(f"🧠 Using sliced file: {DIR_SLICED}")
    print(f"🧠 Using {NUM_PROC} parallel processes")

    all_results = []
    for i in range(0, len(args_list), NUM_PROC):
        batch = args_list[i:i+NUM_PROC]
        print(f"\n🚀 Processing class batch: {[x[1] for x in batch]}")

        with Pool(processes=NUM_PROC) as pool:
            results = pool.map(process_sliced_file, batch)

        all_results.extend(results)
        print("✅ Batch finished")

    print("\n🎉 All classes done.")
    print(all_results)

# other config values...
if __name__ == "__main__":
    # Reproducibility
    seed = random.randint(0, 10000)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run_data()