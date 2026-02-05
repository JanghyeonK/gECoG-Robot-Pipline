import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import time
from datetime import timedelta

from config import *
from models import MultiTaskBinaryModel
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from functions.data_loader import *
from functions.utils import *

torch.backends.cudnn.benchmark = True

def main(X_all=None, Y_all=None):
    seed_everything(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ######### 1) Load Data #########
    if X_all is None or Y_all is None:
        X_all, Y_all, num_bands, num_channels, classes, counts, num_classes = load_data(DIR_DATA)
    else:
        # derive metadata from provided tensors
        num_bands = X_all.shape[1]
        num_channels = X_all.shape[2]
        classes, counts = torch.unique(Y_all, return_counts=True)
        num_classes = len(classes)
    
    # Optionally select a subset of bands (e.g. [2,3,4]) — handled per-sample in dataset
    try:
        if SELECT_BANDS is not None:
            print(f"[DATA] Will use selected bands {list(SELECT_BANDS)} (applied per-sample in dataloader)")
    except NameError:
        pass

    # Keep `Y_all` as integer labels (0..6) for stratified splitting.
    # If Y_all is already multi-hot, ensure float; otherwise keep ints.
    if Y_all.dim() > 1:
        Y_all = Y_all.float()
    print(f"[DATA] X_all: {tuple(X_all.shape)}, Y_all: {tuple(Y_all.shape)}")

    ######### 2) DataLoader #########
    train_dl, val_dl, test_dl, train_ds, val_ds, test_ds = make_dataloaders(
        X_all, Y_all, RATIO_VALID, RATIO_TEST, BATCH_SIZE, select_bands=SELECT_BANDS
    )

    ######### 3) Model, Optimizer, Criterion, Scheduler #########
    effective_n_bands = num_bands if (('SELECT_BANDS' not in globals()) or (SELECT_BANDS is None)) else len(SELECT_BANDS)
    model = MultiTaskBinaryModel(
        n_bands=effective_n_bands,
        n_channels=N_CHANNELS,
        shared_out_dims=SHARED_OUT_DIMS,
        hidden_dims=HIDDEN_DIMS,
        num_heads=NUM_HEADS,
        mapper_name=MAPPER_NAME,
    ).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # Compute per-head pos_weight from training labels for BCEWithLogitsLoss
    try:
        Y_train = train_ds.Y[train_ds.indices]
        # If Y_train is 1D integers (0..6), convert to one-hot for per-head counts
        if Y_train.dim() == 1:
            Y_train_oh = torch.nn.functional.one_hot(Y_train.long(), num_classes=NUM_HEADS).float()
        else:
            Y_train_oh = Y_train.float()

        # per-head positive counts
        pos = Y_train_oh.sum(dim=0)
        neg = Y_train_oh.size(0) - pos
        # Avoid division by zero; if a head has zero positives, set pos to 1 to get large weight
        pos_safe = torch.where(pos == 0, torch.ones_like(pos), pos)
        pos_weight = (neg / (pos_safe + 1e-8))
        pos_weight = torch.clamp(pos_weight, min=1e-3, max=100.0).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"[LOSS] Using BCEWithLogitsLoss with pos_weight={pos_weight}")
    except Exception as e:
        print(f"[WARN] Failed to compute pos_weight from training labels: {e}. Using unweighted BCE.")
        criterion = nn.BCEWithLogitsLoss()

    ######### 4) Load Info Vectors #########
    info_path = DIR_INFOVEC
    try:
        vecs = np.load(info_path)
        vecs_np = np.asarray(vecs)

        orig_bands = num_bands
        # determine selected bands
        try:
            sel = list(SELECT_BANDS) if (('SELECT_BANDS' in globals()) and (SELECT_BANDS is not None)) else list(range(orig_bands))
        except NameError:
            sel = list(range(orig_bands))

        # normalize shapes and bring heads to axis 0
        if vecs_np.ndim == 1:
            if vecs_np.size == NUM_HEADS * orig_bands * N_CHANNELS:
                vecs_np = vecs_np.reshape(NUM_HEADS, orig_bands, N_CHANNELS)
            else:
                raise ValueError(f"Unsupported flat InfoVec shape: {vecs_np.shape}")

        if vecs_np.ndim == 2:
            if vecs_np.shape[0] == NUM_HEADS and vecs_np.shape[1] == orig_bands * N_CHANNELS:
                vecs_np = vecs_np.reshape(NUM_HEADS, orig_bands, N_CHANNELS)
            else:
                raise ValueError(f"Unsupported InfoVec 2D shape: {vecs_np.shape}")

        if vecs_np.ndim == 3:
            # If head axis is not axis 0, move it to 0. Detect which axis equals NUM_HEADS.
            shp = vecs_np.shape
            head_axes = [i for i, s in enumerate(shp) if s == NUM_HEADS]
            if len(head_axes) == 0:
                # fallback: if last dim equals NUM_HEADS, move it
                if shp[-1] == NUM_HEADS:
                    vecs_np = np.moveaxis(vecs_np, -1, 0)
                else:
                    raise ValueError(f"No axis matches NUM_HEADS in InfoVec shape {shp}")
            else:
                if head_axes[0] != 0:
                    vecs_np = np.moveaxis(vecs_np, head_axes[0], 0)

            # Now vecs_np has shape (NUM_HEADS, dim1, dim2) where dim1/dim2 are bands and channels in some order.
            # Ensure axes order is (NUM_HEADS, orig_bands, N_CHANNELS)
            if vecs_np.shape[1] == orig_bands and vecs_np.shape[2] == N_CHANNELS:
                pass
            elif vecs_np.shape[1] == N_CHANNELS and vecs_np.shape[2] == orig_bands:
                vecs_np = vecs_np.transpose(0, 2, 1)
            else:
                # shapes don't match expected; try to infer by matching values
                if vecs_np.shape[1] == orig_bands:
                    # assume third is channels
                    pass
                elif vecs_np.shape[2] == orig_bands:
                    vecs_np = vecs_np.transpose(0, 2, 1)
                else:
                    raise ValueError(f"Cannot infer bands/channels axes in InfoVec shape: {vecs_np.shape}")

            # select bands and flatten
            vecs_sel = vecs_np[:, sel, :]
            vecs_flat = vecs_sel.reshape(NUM_HEADS, -1)
            vecs_t = torch.from_numpy(vecs_flat).float().to(device)
            model.set_info_vectors(vecs_t)
            print(f"[INFO] Loaded info_vectors from {info_path}, original shape={tuple(shp)}, used bands={sel}, final shape={tuple(vecs_flat.shape)}")
        else:
            raise ValueError(f"Unsupported InfoVec array shape: {vecs_np.shape}")
    except Exception as e:
        print(f"[WARN] Failed to load {info_path}: {e}. Using all-ones info_vectors.")

    ######### 5) Training Loop #########
    # Record train start time
    t0 = time.perf_counter()
    os.makedirs(DIR_OUTPUT, exist_ok=True)
    best_val = float("inf")
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    for epoch in range(1, EPOCHS + 1):
        # Record epoch start time
        t_ep = time.perf_counter()
        # ---------- Train ----------
        model.train()
        running_train_loss = 0.0
        running_train_correct = 0
        seen = 0

        pbar = tqdm(train_dl, desc=f"Train E{epoch:02d}", leave=False)
        for xb, yb in pbar:
            ## Ready data
            xb = xb.to(device, non_blocking=True)
            if yb.dim() == 1:
                yb = torch.nn.functional.one_hot(yb.long(), num_classes=NUM_HEADS).float().to(device, non_blocking=True)
            else:
                yb = yb.float().to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            ## Train Step
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits, _ = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            ## Logging
            running_train_loss += loss.item() * xb.size(0)
            seen += xb.size(0)
            preds = (logits.sigmoid() > 0.5).float()
            running_train_correct += (preds.eq(yb).float().mean(dim=1).sum().item()) 
            avg_loss = running_train_loss / seen
            avg_acc = running_train_correct / seen
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")
        train_loss = running_train_loss / len(train_dl.dataset)
        train_acc  = running_train_correct / len(train_dl.dataset)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ---------- Validation ----------
        model.eval()
        running_val_loss = 0.0
        running_val_correct = 0
        seen_val = 0

        with torch.no_grad():
            pbar_v = tqdm(val_dl, desc=f"Valid E{epoch:02d}", leave=False)
            for xb, yb in pbar_v:
                # Validation Step
                xb = xb.to(device, non_blocking=True)
                if yb.dim() == 1:
                    yb = torch.nn.functional.one_hot(yb.long(), num_classes=NUM_HEADS).float().to(device, non_blocking=True)
                else:
                    yb = yb.float().to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    logits, _ = model(xb)
                    loss = criterion(logits, yb)
                # Logging
                running_val_loss += loss.item() * xb.size(0)
                seen_val += xb.size(0)
                preds = (logits.sigmoid() > 0.5).float()
                running_val_correct += (preds.eq(yb).float().mean(dim=1).sum().item())
                avg_vloss = running_val_loss / seen_val
                avg_vacc  = running_val_correct / seen_val
                pbar_v.set_postfix(loss=f"{avg_vloss:.4f}", acc=f"{avg_vacc:.4f}")
        val_loss = running_val_loss / len(val_dl.dataset)
        val_acc  = running_val_correct / len(val_dl.dataset)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"[E{epoch:02d}] "
        f"train_loss={train_loss:.6f}, train_acc={train_acc:.4f}, "
        f"val_loss={val_loss:.6f}, val_acc={val_acc:.4f}")

        # ---------- Checkpointing ----------
        # 1) Save every K epochs
        if epoch % SAVE_EVERY == 0:
            ckpt_path = os.path.join(DIR_OUTPUT, f"epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "train_losses": train_losses, 
                "val_losses":   val_losses,   
                "train_accs":   train_accs,
                "val_accs":     val_accs, 
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "train_acc":  train_acc,
                "val_acc":    val_acc,
            }, ckpt_path)
            print(f"[CKPT] saved: {ckpt_path}")

        # 2) Save best (lowest val) model separately
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(DIR_OUTPUT, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "train_losses": train_losses,
                "val_losses":   val_losses,
                "train_accs":   train_accs,
                "val_accs":     val_accs,
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "train_acc":  train_acc,
                "val_acc":    val_acc,
                "best_val":   best_val,
            }, best_path)

        # Record epoch end time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ep_sec = time.perf_counter() - t_ep
        print(f"[E{epoch:02d}] elapsed={timedelta(seconds=int(ep_sec))} ({ep_sec:.2f}s)")

    # Synchronize CUDA after computation and calculate elapsed time
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    print(f"[TOTAL] training time = {timedelta(seconds=int(elapsed))} ({elapsed:.2f} sec)")

    # Final
    ckpt_path = os.path.join(DIR_OUTPUT, f"epoch_{epoch:03d}_Final.pt")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "train_losses": train_losses, 
        "val_losses":   val_losses,   
        "train_accs":   train_accs,
        "val_accs":     val_accs, 
        "train_loss": train_loss,
        "val_loss":   val_loss,
        "train_acc":  train_acc,
        "val_acc":    val_acc,
        "elapsed": elapsed,
    }, ckpt_path)
    print(f"[CKPT] saved: {ckpt_path}")


if __name__ == "__main__":
    main()