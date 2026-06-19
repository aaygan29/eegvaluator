import json
from pathlib import Path

import pandas as pd

from load_data import load_eeg_csv
from noise_utils import add_gaussian_noise, compute_snr
from metrics import predict_label, accuracy, brier_score, calibration_error


def run(noise_levels=(0.0, 0.1, 0.2, 0.4), seed=42):
    df = load_eeg_csv()
    rows = []

    for noise_level in noise_levels:
        for _, row in df.iterrows():
            clean_signal = row['signal']
            noisy_signal = add_gaussian_noise(clean_signal, noise_level=noise_level, seed=seed)
            pred, prob = predict_label(noisy_signal)
            rows.append({
                'sample_id': row['sample_id'],
                'true_label': row['label'],
                'pred_label': pred,
                'prob_real': prob,
                'noise_level': noise_level,
                'snr': compute_snr(clean_signal, noisy_signal),
            })

    out = pd.DataFrame(rows)
    Path('results').mkdir(exist_ok=True)
    out.to_csv('results/eval_results.csv', index=False)

    summary_rows = []
    for noise_level, group in out.groupby('noise_level'):
        summary_rows.append({
            'noise_level': noise_level,
            'accuracy': accuracy(group['true_label'], group['pred_label']),
            'brier_score': brier_score(group['true_label'], group['prob_real']),
            'calibration_error': calibration_error(group['true_label'], group['prob_real']),
            'avg_snr': float(group['snr'].mean()),
        })

    summary = pd.DataFrame(summary_rows).sort_values('noise_level')
    summary.to_csv('results/summary.csv', index=False)

    with open('results/summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary.to_dict(orient='records'), f, indent=2)

    print(summary)


if __name__ == '__main__':
    run()
