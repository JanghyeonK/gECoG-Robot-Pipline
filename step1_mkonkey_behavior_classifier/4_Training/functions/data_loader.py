import torch, gc
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import torch
import platform

# ------------------------------------------------------------
# 1. List PT files
# ------------------------------------------------------------
def list_pt_files(data_dir):
    """List and sort X_*.pt, Y_*.pt files in data_dir"""
    x_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)
                      if f.startswith("X_") and f.endswith(".pt")])
    y_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)
                      if f.startswith("Y_") and f.endswith(".pt")])
    return x_files, y_files


# ------------------------------------------------------------
# 2. Load PT files (multiple files -> one tensor)
# ------------------------------------------------------------
def load_all_pt(x_files, y_files):
    count_test = None
    """
    Preallocate version: use Y_all to estimate total length,
    and X_1 to get shape for memory allocation.
    """
    print("🔍 Step 1: Counting total samples from Y files...")
    total_N = 0
    Y_all_list = []
    count_loaded = 0
    for y_path in y_files:
        try:
            Y = torch.load(y_path, map_location="cpu").squeeze()
            total_N += len(Y)
            Y_all_list.append(Y)
            count_loaded += 1

            if count_loaded == count_test:
                break  # For quicker testing; remove this line in production

        except Exception as e:
            print(f"⚠️ Skipping {y_path} (reason: {e})")
    if total_N == 0:
        raise RuntimeError("❌ No valid Y files loaded.")
    Y_all = torch.cat(Y_all_list, dim=0)
    del Y_all_list
    gc.collect()

    print(f"✅ Total samples: {total_N}")

    # --- Step 2: Inspect first X file for shape
    print("🔍 Step 2: Reading first X to get shape...")
    first_X = torch.load(x_files[0], map_location="cpu")
    shape_single = first_X.shape[1:]  # (C,H,W)
    dtype = first_X.dtype
    del first_X
    gc.collect()

    print(f"✅ Single sample shape: {shape_single}, dtype: {dtype}")

    # --- Step 3: Preallocate memory
    print("🔧 Step 3: Preallocating memory...")
    X_all = torch.empty((total_N, *shape_single), dtype=dtype)
    print(f"   X_all allocated: {tuple(X_all.shape)}")

    # --- Step 4: Fill memory chunk-by-chunk
    offset = 0
    count_loaded = 0
    for x_path in x_files:
        try:
            print(f"⏳ Loading {x_path} at offset {offset}...")
            X = torch.load(x_path, map_location="cpu")
            n = X.shape[0]
            X_all[offset:offset+n] = X  # ✅ in-place copy
            offset += n
            count_loaded += 1

            if count_loaded == count_test:
                break  # For quicker testing; remove this line in production

            del X
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️ Failed to load {x_path}: {e}")
            continue

    print(f"✅ Successfully loaded {count_loaded} / {len(x_files)} X files.")
    print(f"   Final shapes: X_all={tuple(X_all.shape)}, Y_all={tuple(Y_all.shape)}")

    return X_all, Y_all

# ------------------------------------------------------------
# 3. Load all data and print basic information
# ------------------------------------------------------------
def load_data(data_dir):
    """Load all PT data + split into TensorDataset"""

    x_files, y_files = list_pt_files(data_dir)
    X_all, Y_all = load_all_pt(x_files, y_files)

    # Y 타입 변환 / X는 입력 시 float32로 변환
    if Y_all.dtype != torch.long:    Y_all = Y_all.long()

    # Automatically extract data structure
    num_samples  = X_all.shape[0]
    num_bands    = X_all.shape[1]
    num_channels = X_all.shape[2]
    img_h, img_w = X_all.shape[3], X_all.shape[4]
    classes, counts = torch.unique(Y_all, return_counts=True)
    num_classes = len(classes)

    print(f"✅ X: {num_samples} samples, {num_bands} bands, {num_channels} channels")
    print(f"✅ Y: {num_classes} classes")
    for c, n in zip(classes.tolist(), counts.tolist()):
        print(f"   Class {c}: {n} samples")

    return X_all, Y_all, num_bands, num_channels, classes, counts, num_classes


from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ============================================================
# 🧩 LazyFloatDataset: on-the-fly float conversion Dataset
# ============================================================
class LazyFloatSubset(Dataset):
    def __init__(self, X, Y, indices, normalize=True, select_bands=None):
        self.X = X
        self.Y = Y
        self.indices = indices
        self.normalize = normalize
        self.select_bands = select_bands

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = self.X[real_idx]
        y = self.Y[real_idx]

        # Per-sample band selection (avoid casting entire X array)
        if self.select_bands is not None:
            sel = list(self.select_bands)
            if len(sel) == 0:
                pass
            else:
                # if contiguous, use slice (faster)
                if max(sel) - min(sel) + 1 == len(sel) and sorted(sel) == sel:
                    lo, hi = min(sel), max(sel)
                    x = x[lo:hi+1]
                else:
                    idx_tensor = torch.tensor(sel, dtype=torch.long)
                    x = torch.index_select(x, 0, idx_tensor)

        # Convert to float and normalize only for this sample
        if x.dtype == torch.uint16:
            x = x.to(torch.float32)
            if self.normalize:
                x.div_(65535.0)

        return x, y

# ============================================================
# ⚙️ DataLoader creation function
# ============================================================
def make_dataloaders(X_bandch, Y_binary, ratio_val, ratio_test, batch_size, select_bands=None):
    """
    Input:
      - X_bandch : (N, 1, H, W) [uint16 possible]
      - Y_binary : (N,)
    """
    import os
    
    N = len(Y_binary)
    stratify = Y_binary.numpy() if len(torch.unique(Y_binary)) > 1 else None

    # --------------------------------------------------------
    # 1️⃣ Split train+val / test
    # --------------------------------------------------------
    idx_trainval, idx_test = train_test_split(
        range(N),
        test_size=ratio_test,
        stratify=stratify,
    )

    # --------------------------------------------------------
    # 2️⃣ Split train / val
    # --------------------------------------------------------
    stratify_trainval = Y_binary[idx_trainval].numpy() if len(torch.unique(Y_binary)) > 1 else None
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=ratio_val / (1 - ratio_test),
        stratify=stratify_trainval,
    )

    # --------------------------------------------------------
    # ✅ Use LazyFloatSubset — no indexing here!
    # --------------------------------------------------------
    train_ds = LazyFloatSubset(X_bandch, Y_binary, idx_train, normalize=True, select_bands=select_bands)
    val_ds   = LazyFloatSubset(X_bandch, Y_binary, idx_val,   normalize=True, select_bands=select_bands)
    test_ds  = LazyFloatSubset(X_bandch, Y_binary, idx_test,  normalize=True, select_bands=select_bands)

    # --------------------------------------------------------
    # Create DataLoader (OS-specific optimization)
    # --------------------------------------------------------
    is_child_process = 'DATALOADER_WORKERS' in os.environ
    is_windows = platform.system() == "Windows"
    
    if is_child_process:
        # Child process: minimize all resources
        num_workers_train = 0
        num_workers_val = 0
        num_workers_test = 0
        pin_memory_val = False
        prefetch_factor = 2
        print("[DataLoader] ⚠️  Child process detected → num_workers=0, pin_memory=False, prefetch_factor=2")
    elif is_windows:
        # Windows: spawn overhead large → num_workers=0
        num_workers_train = 0
        num_workers_val = 0
        num_workers_test = 0
        pin_memory_val = False
        prefetch_factor = 2
        print("[DataLoader] Windows detected → using num_workers=0, pin_memory=False")
    else:
        # Linux/Unix (single process): fork method fast → use workers
        num_workers_train = 8
        num_workers_val = 4
        num_workers_test = 2
        pin_memory_val = True
        prefetch_factor = 4
        print("[DataLoader] Linux (single process) → using num_workers=8/4/2, pin_memory=True, prefetch_factor=4")
    
    loader_kwargs = dict(
        batch_size=batch_size,
        pin_memory=pin_memory_val,
        prefetch_factor=prefetch_factor,
    )
    
    print(f"[DataLoader] Creating train_loader with {num_workers_train} workers...")
    train_loader = DataLoader(train_ds, shuffle=True, num_workers=num_workers_train, **loader_kwargs)
    print(f"[DataLoader] Creating val_loader with {num_workers_val} workers...")
    val_loader   = DataLoader(val_ds,   shuffle=False, num_workers=num_workers_val, **loader_kwargs)
    print(f"[DataLoader] Creating test_loader with {num_workers_test} workers...")
    test_loader  = DataLoader(test_ds,  shuffle=False, num_workers=num_workers_test, **loader_kwargs)

    print(f"[DataLoader] ✅ All dataloaders created successfully")
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


def balance_classes(X_all, Y_all, max_factor=2):
    """
    Examine data count for each class and
    balance to {max_factor}x the least frequent class
    """
    unique_classes = torch.unique(Y_all)
    counts = {int(c.item()): int((Y_all == c).sum().item()) for c in unique_classes}
    print(f"Original class distribution: {counts}")

    min_count = min(counts.values())
    target_count = min_count * max_factor

    X_bal_list, Y_bal_list = [], []
    for c in unique_classes:
        idx = torch.nonzero(Y_all == c, as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx))]  # shuffle
        n_select = min(len(idx), target_count)
        selected = idx[:n_select]

        X_bal_list.append(X_all[selected])
        Y_bal_list.append(Y_all[selected])

    X_bal = torch.cat(X_bal_list, dim=0)
    Y_bal = torch.cat(Y_bal_list, dim=0)

    print(f"After balancing, class distribution:")
    new_counts = {int(c.item()): int((Y_bal == c).sum().item()) for c in unique_classes}
    print(new_counts)

    return X_bal, Y_bal