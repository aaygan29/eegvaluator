import numpy as np


def add_gaussian_noise(signal, noise_level=0.1, seed=None):
    rng = np.random.default_rng(seed)
    x = np.asarray(signal, dtype=float)
    scale = np.std(x) if np.std(x) > 0 else 1.0
    noise = rng.normal(0, noise_level * scale, size=len(x))
    return (x + noise).tolist()


def compute_signal_energy(signal):
    x = np.asarray(signal, dtype=float)
    return float(np.mean(x ** 2))


def compute_snr(clean_signal, noisy_signal):
    clean = np.asarray(clean_signal, dtype=float)
    noisy = np.asarray(noisy_signal, dtype=float)
    noise = noisy - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return float(10 * np.log10(signal_power / noise_power))
