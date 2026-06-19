# eegvaluator project

# EEG Noise Evals

This project tests whether a model can tell real EEG-like signals from noise and whether its confidence changes as the signal gets more corrupted.

## What it does

- Loads EEG samples from a CSV file.
- Creates controlled noisy versions of each sample.
- Runs a classifier or heuristic scorer.
- Measures accuracy, confidence, calibration, and noise robustness.
- Saves results for analysis.

## Folder structure

```text
eeg-noise-evals/
├── data/
│   └── eeg_samples.csv
├── src/
│   ├── __init__.py
│   ├── load_data.py
│   ├── noise_utils.py
│   ├── metrics.py
│   ├── run_eval.py
│   └── analysis.py
├── results/
├── README.md
└── requirements.txt
```

## How to run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Put EEG samples into `data/eeg_samples.csv`.

3. Run evaluation:
   ```bash
   python src/run_eval.py
   ```

4. Run analysis:
   ```bash
   python src/analysis.py
   ```

## Input format

CSV Files:

- `sample_id`
- `signal`
- `label`

Where `signal` is a serialized list of numbers, and `label` is `real` or `noise`.

## Output files

- `results/eval_results.csv`
- `results/summary.json`
- `results/accuracy_by_noise.png`
- `results/calibration_by_noise.png`

## Metrics

- Accuracy
- Confidence
- Calibration error
- Robustness under added noise
