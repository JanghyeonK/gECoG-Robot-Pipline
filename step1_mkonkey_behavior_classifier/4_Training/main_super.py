import sys
import time
from pathlib import Path

import config
import torch
import main
from functions.data_loader import load_data


def run_trials(n_trials: int = 10):
    base = Path(config.DIR_OUTPUT)
    base.mkdir(parents=True, exist_ok=True)
    # Load data once
    print("[SUPER] Loading data once for all trials...")
    X_all, Y_all, num_bands, num_channels, classes, counts, num_classes = load_data(config.DIR_DATA)
    print(f"[SUPER] Loaded X_all={tuple(X_all.shape)}, Y_all={tuple(Y_all.shape)}")
    
    for t in range(1, n_trials + 1):
        out = base / f"trial_{t:02d}"
        out.mkdir(parents=True, exist_ok=True)
        # override config output dir for this run
        config.DIR_OUTPUT = out
        # set a fresh seed for each trial
        new_seed = int(time.time()) % 1000000
        config.SEED = new_seed

        # set band selection: first half use all bands, second half use config.SELECT_BANDS
        half = n_trials // 2
        if t <= half:
            sel = None
        else:
            sel = config.SELECT_BANDS

        # also set these values in the imported main module so main.main() sees them
        try:
            main.SEED = new_seed
            main.DIR_OUTPUT = out
            main.SELECT_BANDS = sel
        except Exception:
            pass

        print(f"[SUPER] Starting trial {t}/{n_trials} -> {out} (SEED={new_seed}, SELECT_BANDS={sel})")
        start = time.time()
        try:
            main.main(X_all=X_all, Y_all=Y_all)
        except Exception as e:
            print(f"[SUPER] Trial {t} failed: {e}")
        finally:
            # best-effort GPU cleanup between trials
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        dur = time.time() - start
        print(f"[SUPER] Finished trial {t} ({dur:.1f}s)\n")


if __name__ == "__main__":
    n = config.TRIAL
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except Exception:
            pass
    run_trials(n)
