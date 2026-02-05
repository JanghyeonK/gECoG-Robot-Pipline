import torch
import torch.nn.functional as F
import math

# =====================================================
# 🧠 Global Wavelet Cache
# =====================================================
WAVELET_CACHE = {}

# =====================================================
# 🧩 Morlet Wavelet Generator (with cache)
# =====================================================
def morlet_wavelet(f, fs, sigma=6.0, n_sigma=6, device="cuda", dtype=torch.float16):
    """Generate Morlet wavelet for frequency f and cache it."""
    key = (round(float(f), 5), fs, str(device), str(dtype))
    if key in WAVELET_CACHE:
        return WAVELET_CACHE[key]

    sigma_t = sigma / (2 * math.pi * f)
    base_len = int(2 * n_sigma * sigma_t * fs)
    if base_len % 2 != 0:
        base_len += 1 # Make it even

    t = torch.linspace(-n_sigma * sigma_t, n_sigma * sigma_t, base_len,
                       device=device, dtype=dtype)
    w = torch.exp(2j * math.pi * f * t) * torch.exp(-t**2 / (2 * sigma_t**2))
    w = w / (w.abs().sum() + 1e-8)
    w_complex = w.to(torch.complex64 if dtype == torch.float32 else torch.complex32)

    WAVELET_CACHE[key] = w_complex
    return w_complex


# =====================================================
# ⚡ GPU Continuous Wavelet Transform (CWT)
# =====================================================
def torch_cwt_band_multi(signal_batch, fs, band_list, n_freqs_per_band=32,
                         device="cuda", use_half=True):
    """Multi-band CWT on GPU using convolution (Morlet wavelet) with caching.

    signal_batch: (N_ch, T)
    return: list of bands, each (N_ch, N_freq, T')
    """
    dtype_real = torch.float16 if use_half else torch.float32
    dtype_complex = torch.complex32 if use_half else torch.complex64

    signal_batch = signal_batch.to(dtype=dtype_complex, device=device)
    N_ch, T = signal_batch.shape
    signal_batch = signal_batch.unsqueeze(0)  # (1, N_ch, T)

    all_coeffs = []
    for (fmin, fmax) in band_list:
        freqs = torch.linspace(fmin, fmax, n_freqs_per_band, device=device, dtype=torch.float32)
        coeffs_band = []

        for f in freqs:
            wavelet = morlet_wavelet(f.item(), fs, device=device, dtype=dtype_real)
            wavelet = wavelet.conj().flip(0).view(1, 1, -1)
            wavelet = wavelet.repeat(N_ch, 1, 1)

            coef = F.conv1d(signal_batch, wavelet, padding=wavelet.shape[-1] // 2, groups=N_ch)
            coeffs_band.append(coef.squeeze(0))

        max_T = max(c.shape[-1] for c in coeffs_band)
        coeffs_band = [F.pad(c, (0, max_T - c.shape[-1])) for c in coeffs_band]

        coeffs_band = torch.stack(coeffs_band, dim=0)      # (N_freq, N_ch, T)
        coeffs_band = coeffs_band.permute(1, 0, 2).abs()   # (N_ch, N_freq, T)
        coeffs_band = coeffs_band.to(torch.float32)
        all_coeffs.append(coeffs_band)

    return all_coeffs


# =====================================================
# ⚙️ Wavelet Cache Preloader
# =====================================================
def preload_wavelets(fs=256, band_list=None, n_freqs_per_band=20, device="cuda", dtype=torch.float16):
    """Precompute and cache all wavelets used for CWT."""
    if band_list is None:
        band_list = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 80)]

    for (fmin, fmax) in band_list:
        freqs = torch.linspace(fmin, fmax, n_freqs_per_band, device=device)
        for f in freqs:
            morlet_wavelet(float(f), fs, device=device, dtype=dtype)

    print(f"✅ Preloaded {len(WAVELET_CACHE)} wavelets into cache (device={device}).")
