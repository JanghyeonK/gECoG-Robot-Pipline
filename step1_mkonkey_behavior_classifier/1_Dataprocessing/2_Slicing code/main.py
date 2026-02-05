import random, torch
import pandas as pd
from scipy.io import loadmat, savemat
from functions.label_utils import *
from config import *

def run_offline_slicing():
    # -------------------------------------------------
    # 0. Prepare file list
    # -------------------------------------------------
    csv_files = sorted(list(DIR_LABEL.glob("*.csv")))
    mat_files = sorted(list(DIR_ECOG.glob("*.mat")))
    num_file = min(len(csv_files), len(mat_files))
    file_pairs = [(csv_files[i], mat_files[i], i + 1) for i in range(num_file)]
    print(f"🧠 Found {num_file} files")

    # -------------------------------------------------
    # 1. Prepare ECoG/label list by class
    #    (Classes 0-6: based on label_utils.map_behavior_labels)
    # -------------------------------------------------
    NUM_CLASS = 7
    ecog_by_class = {c: [] for c in range(NUM_CLASS)}  
    label_by_class = {c: [] for c in range(NUM_CLASS)}

    idx_global = 0

    # -------------------------------------------------
    # 2. File-by-file loop
    # -------------------------------------------------
    for csv_path, mat_path, file_id in file_pairs:
        print(f"\n📂 Slicing - {csv_path.name}, {mat_path.name}")

        # -------- (1) Load and preprocess labels --------
        T = pd.read_csv(csv_path, header=None, names=["Time", "Behavior"])

        labels_np, bad_idx = map_behavior_labels(T["Behavior"])
        if len(bad_idx) > 0:
            # Remove incorrectly mapped label rows
            T = T.drop(bad_idx).reset_index(drop=True)
            labels = torch.tensor(
                [l for i, l in enumerate(labels_np) if i not in bad_idx],
                dtype=torch.int32
            )
        else:
            labels = torch.tensor(labels_np, dtype=torch.int32)

        # -------- (2) Load ECoG --------
        S = loadmat(mat_path)
        ecog_all = torch.tensor(S["nD_concat"], dtype=torch.float32)  # (N_t, N_ch)
        ecog_t = torch.tensor(S["nD_t"]).squeeze()                    # (N_t,)
        N_ch = ecog_all.shape[1]

        # -------- (3) Expand labels to match ECoG time --------
        Label_full_np = map_labels_to_ecog_time(ecog_t, T["Time"], labels)
        Label_full = torch.tensor(Label_full_np, dtype=torch.int32)
        N_t = len(Label_full)

        # -------- (4) Generate window indices --------
        idx_win_list = list(range(SIZE_WINDOW + 1,N_t -  SIZE_WINDOW - 1, SIZE_STRIDE))
        total_windows = len(idx_win_list)
        print(f"  → {total_windows} windows in this file")

        # -------- (5) Window slicing and collect by class --------
        for idx_win in idx_win_list:
            idx_global += 1

            idx_start = idx_win - SIZE_WINDOW - 1
            idx_end   = idx_win + SIZE_WINDOW  # [idx_start, idx_end)

            # ECoG window shape = (2*SIZE_WINDOW+1, N_ch)
            ecog = ecog_all[idx_start:idx_end, :].cpu().numpy()

            # Label at center timepoint
            label = int(Label_full[idx_win].item())

            # Skip invalid labels
            if (label < 0) or (label >= NUM_CLASS):
                continue

            ecog_by_class[label].append(ecog)
            label_by_class[label].append(label)

    # -------------------------------------------------
    # 3. Print overall statistics (optional)
    # -------------------------------------------------
    print("\n✅ Slicing finished.")
    for c in range(NUM_CLASS):
        print(f"Class {c}: {len(ecog_by_class[c])} windows")

    # Return for later use in AAFT or saving
    return ecog_by_class, label_by_class


def save_for_matlab(ecog_by_class, save_path="ecog_sliced.mat"):
    """
    ecog_by_class: dict[int -> list of (T_window, N_ch) np.array]
    save_path    : Path to .mat file to save
    """
    mat_dict = {}
    for c, windows in ecog_by_class.items():
        if len(windows) == 0:
            continue

        # windows: list of (T_window, N_ch)
        # -> X: (T_window, N_ch, N_win_c)
        X = np.stack(windows, axis=2).astype(np.float32)
        mat_dict[f"X_class{c}"] = X

    # Metadata can also be saved together if needed
    mat_dict["NUM_CLASS"] = np.int32(len(ecog_by_class))

    savemat(save_path, mat_dict)
    print(f"💾 Saved for MATLAB: {save_path}")


if __name__ == "__main__":
    # Reproducibility
    seed = random.randint(0, 10000)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ecog_by_class, label_by_class = run_offline_slicing()
    save_for_matlab(ecog_by_class, "result/ecog_sliced.mat")


