import os
import threading
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from config import DIR_OUT, FLAG_TEST_LOAD

def async_save(X, Y, fx: Path, fy: Path):
    """Async-safe save (tmp → atomic replace)."""
    def _save():
        X_cpu = X.detach().to("cpu", non_blocking=True)
        Y_cpu = Y.detach().to("cpu", non_blocking=True)

        X_uint16 = (X_cpu.clamp(0, 1) * 65535).to(torch.uint16)
        Y_uint16 = Y_cpu.to(torch.uint16)

        fx_pt = fx.with_suffix('.pt')
        fy_pt = fy.with_suffix('.pt')

        tmp_fx = fx_pt.with_suffix('.pt.tmp')
        tmp_fy = fy_pt.with_suffix('.pt.tmp')

        with open(tmp_fx, "wb") as f:
            torch.save(X_uint16, f)
            f.flush()
            os.fsync(f.fileno())

        with open(tmp_fy, "wb") as f:
            torch.save(Y_uint16, f)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_fx, fx_pt)
        os.replace(tmp_fy, fy_pt)

        if FLAG_TEST_LOAD:
            try:
                test = torch.load(fx_pt, map_location='cpu')
                assert isinstance(test, torch.Tensor)
                print(f"✅ Load test passed for {fx_pt}")
            except Exception as e:
                print(f"❌ Load test failed for {fx_pt}: {e}")

    t = threading.Thread(target=_save, daemon=True)
    t.start()
    t.join()


def save_preview_image(X_batch, file_id: int, idx_global: int, out_dir: Path = None):
    """Save preview of (Band x Channel) grid.

    X_batch: (N, Band, Ch, H, W)
    """
    if out_dir is None:
        out_dir = DIR_OUT

    img = X_batch[0].cpu().numpy()
    n_band, n_ch, H, W = img.shape
    grid = np.zeros((n_band * H, n_ch * W))

    for b in range(n_band):
        for c in range(n_ch):
            grid[b * H:(b + 1) * H, c * W:(c + 1) * W] = img[b, c]

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(16, 3))
    plt.imshow(grid, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title(f"File {file_id} - Window {idx_global}")
    out_path = out_dir / f"preview_f{file_id:02d}_w{idx_global:06d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"🖼️ Preview saved: {out_path}")
