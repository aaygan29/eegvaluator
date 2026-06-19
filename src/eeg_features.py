import numpy as np

try:
    from scipy.signal import welch
except ImportError as e:
    raise ImportError("scipy is required for EEG feature extraction") from e


FREQ_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def _to_array(signal):
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be a 1D sequence")
    return x


def bandpower(signal, sfreq=256, band=(0.5, 4), window_sec=2.0):
    x = _to_array(signal)
    nperseg = min(len(x), int(window_sec * sfreq))
    if nperseg < 4:
        raise ValueError("signal too short for spectral estimation")

    freqs, psd = welch(x, fs=sfreq, nperseg=nperseg)
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    if not np.any(idx):
        return 0.0
    return float(np.trapz(psd[idx], freqs[idx]))


def total_power(signal, sfreq=256, fmin=0.5, fmax=45.0, window_sec=2.0):
    return bandpower(signal, sfreq=sfreq, band=(fmin, fmax), window_sec=window_sec)


def relative_bandpower_features(signal, sfreq=256):
    x = _to_array(signal)
    total = total_power(x, sfreq=sfreq)
    total = total if total > 0 else 1e-12

    feats = {}
    for name, band in FREQ_BANDS.items():
        bp = bandpower(x, sfreq=sfreq, band=band)
        feats[f"{name}_power"] = bp
        feats[f"{name}_rel_power"] = bp / total

    feats["total_power"] = total
    feats["mean"] = float(np.mean(x))
    feats["std"] = float(np.std(x))
    feats["rms"] = float(np.sqrt(np.mean(x ** 2)))
    return feats
