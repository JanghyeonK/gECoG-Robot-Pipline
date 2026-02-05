import pandas as pd
import torch, h5py
import torch.nn.functional as F
from scipy.io import loadmat
from multiprocessing import Pool
import torch.multiprocessing as mp
from pathlib import Path
import numpy as np

from functions.label_utils import map_behavior_labels, map_labels_to_ecog_time
from functions.io_utils import async_save, save_preview_image
from functions.filters import notch_filter, bandpass_filter, car
from functions.wavelet import torch_cwt_band_multi, preload_wavelets
from config import *

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


def process_file(args):
    csv_path, mat_path, file_id, device = args

    print(f"\n▶ Processing file {file_id}: {csv_path.name}, {mat_path.name}")

    # --- Load Labels ---
    T = pd.read_csv(csv_path, header=None, names=["Time", "Behavior"])
    labels, bad_idx = map_behavior_labels(T["Behavior"])
    if len(bad_idx) > 0:
        T = T.drop(bad_idx).reset_index(drop=True)
        labels = torch.tensor([l for i, l in enumerate(labels) if i not in bad_idx], dtype=torch.int32)

    # --- Load ECoG ---
    S = loadmat(mat_path)
    ecog_all = torch.tensor(S["nD_concat"], dtype=torch.float32)
    ecog_t = torch.tensor(S["nD_t"]).squeeze()
    N_ch = ecog_all.shape[1]

    # --- Align Labels with ECoG Time ---
    Label_full = torch.tensor(map_labels_to_ecog_time(ecog_t, T["Time"], labels))
    N_t = len(Label_full)

    # --- Window Index Generation ---
    idx_win_list = list(range(SIZE_WINDOW + 1, N_t - SIZE_WINDOW - 1, SIZE_STRIDE))
    total_windows = len(idx_win_list)

    # --- Buffers ---
    X = torch.zeros((SIZE_BATCH, len(BAND_NAME), N_ch, SIZE_IMG, SIZE_IMG), dtype=torch.float32)
    Y = torch.zeros((SIZE_BATCH,), dtype=torch.uint16)

    idx_global = 0
    idx_batch = 0

    for idx_win in idx_win_list:
        idx_global += 1
        idx_start = idx_win - SIZE_WINDOW - 1
        idx_end = idx_win + SIZE_WINDOW

        y = Label_full[idx_win]
        pos = (idx_global - 1) % SIZE_BATCH

        # ----- Convert to numpy from CPU -----
        ecog_np = ecog_all[idx_start:idx_end, :].cpu().numpy()

        # ----- Notch filters -----
        ecog_np = notch_filter(ecog_np, FS, 60)
        ecog_np = notch_filter(ecog_np, FS, 120)

        # ----- Band-pass filter (1–120 Hz) -----
        ecog_np = bandpass_filter(ecog_np, FS, 1, 120)

        # ----- Common Average Reference -----
        ecog_np = car(ecog_np)

        # ----- Convert to GPU Tensor -----
        ecog = torch.tensor(ecog_np, device=device,
                            dtype=torch.float32)

        # --- Wavelet ---
        with torch.no_grad():
            coeffs_all = torch_cwt_band_multi(
                ecog.T, FS, BAND_HZ,
                n_freqs_per_band=NUM_FREQ_PER_BAND,
                device=ecog.device,
                use_half=False,
            )

            for idx_band, e in enumerate(coeffs_all):
                for idx_ch in range(N_ch):
                    e_ch = e[idx_ch]
                    e_min, e_max = e_ch.min(), e_ch.max()
                    e_norm = (e_ch - e_min) / (e_max - e_min + 1e-8)
                    e_norm = e_norm.unsqueeze(0).unsqueeze(0)
                    x = F.interpolate(e_norm, size=(SIZE_IMG, SIZE_IMG),
                                      mode='bilinear', align_corners=False)
                    X[pos, idx_band, idx_ch] = x.squeeze().clamp(0, 1)

        Y[pos] = y.to(torch.uint16)

        # --- Monitor ---
        if FLAG_MONITOR and (idx_global % INTERVAL_MONITOR == 0):
            progress = idx_global / total_windows * 100
            print(f"[file {file_id:02d}] progress: {progress:6.2f}% ({idx_global}/{total_windows})")


        # --- Image Save ---
        if FLAG_IMAGE_SAVE and (idx_global % INTERVAL_IMAGE_SAVE == 0):
            single_win = X[pos].unsqueeze(0).clone()   # shape: (1, Band, Ch, H, W)
            save_preview_image(single_win, file_id, idx_global, DIR_OUT)

        # --- Batch Save ---
        if (pos + 1) == SIZE_BATCH:
            idx_batch += 1
            fx = DIR_OUT / f"X_f{file_id:02d}_b{idx_batch:03d}.pt"
            fy = DIR_OUT / f"Y_f{file_id:02d}_b{idx_batch:03d}.pt"
            async_save(X.clone(), Y.clone(), fx, fy)
            X.zero_()
            Y.zero_()
            print(f"✅ Saved batch {idx_batch:03d} (file {file_id})")


    # --- Save Remaining ---
    remain = idx_global % SIZE_BATCH
    if remain > 0:
        idx_batch += 1
        fx = DIR_OUT / f"X_f{file_id:02d}_b{idx_batch:03d}.pt"
        fy = DIR_OUT / f"Y_f{file_id:02d}_b{idx_batch:03d}.pt"
        async_save(X[:remain].clone(), Y[:remain].clone(), fx, fy)
        print(f"✅ Saved final batch (remain={remain})")


    print(f"🎯 Done file {file_id}. Total windows={idx_global}, batches={idx_batch}")

    return {
        "file_id": file_id,
        "N_ch": N_ch,
        "total_windows": idx_global,
        "total_batches": idx_batch,
    }

def process_sliced_file(args):
    """
    Read 501x32xN windows for a specific class from ecog_sliced(.mat, v7.3),
    filter, convert to wavelet images, and save as .pt files.
    """
    mat_path, class_id, device = args
    mat_path = str(mat_path)

    print(f"\n▶ Processing CLASS {class_id} from {Path(mat_path).name}")

    try:
        S = loadmat(mat_path)
        data = np.array(S[f"X_class{class_id}"], dtype="float32")
        data = np.transpose(data, (2, 1, 0))

    except Exception:
            with h5py.File(mat_path, "r") as f:
                key_aaft = f"X_class{class_id}_AAFT"
                key_raw  = f"X_class{class_id}"

                if key_aaft in f:
                    dset = f[key_aaft]
                    print(f"  -> using key: {key_aaft}")
                elif key_raw in f:
                    dset = f[key_raw]
                    print(f"  -> using key: {key_raw}")
                else:
                    raise KeyError(f"Cannot find {key_aaft} or {key_raw} in {mat_path}")

                # MATLAB -v7.3: shape comes as is (T, Ch, N)
                data = dset[...]           # numpy array
    
    data = np.asarray(data, dtype="float32")
    N_data, N_ch, N_T = data.shape

    # ----- 2) Buffer preparation (same structure as original process_file) -----
    X = torch.zeros((SIZE_BATCH, len(BAND_NAME), N_ch, SIZE_IMG, SIZE_IMG),
                    dtype=torch.float32)
    Y = torch.zeros((SIZE_BATCH,), dtype=torch.uint16)

    idx_global = 0
    idx_batch  = 0

    # ----- 3) Window loop: already sliced to 501x32 -----
    for idx in range(N_data):
        idx_global += 1
        pos = (idx_global - 1) % SIZE_BATCH

        # (T, Ch)
        ecog_np = data[idx, :, :].T 

        # ----- Filtering (notch + bandpass + CAR) -----
        ecog_np = notch_filter(ecog_np, FS, 60)
        ecog_np = notch_filter(ecog_np, FS, 120)
        ecog_np = bandpass_filter(ecog_np, FS, 1, 120)
        ecog_np = car(ecog_np)

        # GPU tensor 변환
        ecog = torch.tensor(ecog_np, device=device, dtype=torch.float32)

        # ----- Wavelet -----
        with torch.no_grad():
            # input ecog: (N_ch, T)
            coeffs_all = torch_cwt_band_multi(
                ecog.T, FS, BAND_HZ,
                n_freqs_per_band=NUM_FREQ_PER_BAND,
                device=device,
                use_half=False,
            )

            for idx_band, e in enumerate(coeffs_all):
                for idx_ch in range(N_ch):
                    e_ch = e[idx_ch]                      # (N_freq, T')
                    e_min, e_max = e_ch.min(), e_ch.max()
                    e_norm = (e_ch - e_min) / (e_max - e_min + 1e-8)
                    e_norm = e_norm.unsqueeze(0).unsqueeze(0)
                    x = F.interpolate(
                        e_norm, size=(SIZE_IMG, SIZE_IMG),
                        mode="bilinear", align_corners=False
                    )
                    X[pos, idx_band, idx_ch] = x.squeeze().clamp(0, 1)

        # 이 윈도우의 label = class_id
        Y[pos] = int(class_id)

        # ----- Monitor -----
        if FLAG_MONITOR and (idx_global % INTERVAL_MONITOR == 0):
            progress = idx_global / N_data * 100.0
            print(f"[class {class_id:02d}] progress: {progress:6.2f}% "
                  f"({idx_global}/{N_data})")

        # ----- Preview image 저장 -----
        if FLAG_IMAGE_SAVE and (idx_global % INTERVAL_IMAGE_SAVE == 0):
            single_win = X[pos].unsqueeze(0).clone()
            save_preview_image(single_win, class_id, idx_global, DIR_OUT)

        # ----- Batch 저장 -----
        if (pos + 1) == SIZE_BATCH:
            idx_batch += 1
            fx = DIR_OUT / f"X_c{class_id:02d}_b{idx_batch:03d}.pt"
            fy = DIR_OUT / f"Y_c{class_id:02d}_b{idx_batch:03d}.pt"
            async_save(X.clone(), Y.clone(), fx, fy)
            X.zero_()
            Y.zero_()
            print(f"✅ Saved batch {idx_batch:03d} (class {class_id})")

    # ----- 4) Save the remaining -----
    remain = idx_global % SIZE_BATCH
    if remain > 0:
        idx_batch += 1
        fx = DIR_OUT / f"X_c{class_id:02d}_b{idx_batch:03d}.pt"
        fy = DIR_OUT / f"Y_c{class_id:02d}_b{idx_batch:03d}.pt"
        async_save(X[:remain].clone(), Y[:remain].clone(), fx, fy)
        print(f"✅ Saved final batch for class {class_id} (remain={remain})")

    print(f"🎯 Done CLASS {class_id}. Total windows={idx_global}, batches={idx_batch}")

    return {
        "class_id": class_id,
        "N_ch": N_ch,
        "total_windows": idx_global,
        "total_batches": idx_batch,
    }