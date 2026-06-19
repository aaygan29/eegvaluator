import numpy as np
from eeg_features import relative_bandpower_features


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict_probability(signal, sfreq=256):
    feats = relative_bandpower_features(signal, sfreq=sfreq)

    score = (
        1.5 * feats["alpha_rel_power"]
        + 1.2 * feats["beta_rel_power"]
        + 0.8 * feats["theta_rel_power"]
        + 0.5 * feats["rms"]
        - 0.3 * feats["gamma_rel_power"]
        - 0.2 * abs(feats["mean"])
    )

    return float(sigmoid(score))


def predict_label(signal, threshold=0.5, sfreq=256):
    prob = predict_probability(signal, sfreq=sfreq)
    return "real" if prob >= threshold else "noise", prob


def accuracy(y_true, y_pred):
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def brier_score(y_true, y_prob):
    y = np.array([1 if label == "real" else 0 for label in y_true], dtype=float)
    p = np.array(y_prob, dtype=float)
    return float(np.mean((p - y) ** 2))


def calibration_error(y_true, y_prob, n_bins=5):
    y = np.array([1 if label == "real" else 0 for label in y_true], dtype=float)
    p = np.array(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(y)
    err = 0.0

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (p >= bins[i]) & (p < bins[i + 1])
        else:
            mask = (p >= bins[i]) & (p <= bins[i + 1])

        if mask.sum() == 0:
            continue

        acc = y[mask].mean()
        conf = p[mask].mean()
        err += (mask.sum() / total) * abs(acc - conf)

    return float(err)
