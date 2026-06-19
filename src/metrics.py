import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict_probability(signal):
    x = np.asarray(signal, dtype=float)
    energy = np.mean(x ** 2)
    variability = np.std(x)
    score = 2.0 * energy + 0.5 * variability - 1.0
    return float(sigmoid(score))


def predict_label(signal, threshold=0.5):
    prob = predict_probability(signal)
    return 'real' if prob >= threshold else 'noise', prob


def accuracy(y_true, y_pred):
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def brier_score(y_true, y_prob):
    y = np.array([1 if label == 'real' else 0 for label in y_true], dtype=float)
    p = np.array(y_prob, dtype=float)
    return float(np.mean((p - y) ** 2))


def calibration_error(y_true, y_prob, n_bins=5):
    y = np.array([1 if label == 'real' else 0 for label in y_true], dtype=float)
    p = np.array(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(y)
    err = 0.0
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1]) if i < n_bins - 1 else (p >= bins[i]) & (p <= bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y[mask].mean()
        conf = p[mask].mean()
        err += (mask.sum() / total) * abs(acc - conf)
    return float(err)
