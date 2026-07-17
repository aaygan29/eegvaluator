# eegvaluator

Tests whether an EEG classifier can tell real signals from noise, and how its confidence and calibration change as the signal is progressively corrupted, not just its clean-data accuracy.

## What it does

- Loads EEG samples from a CSV file.
- Creates controlled noisy versions of each sample.
- Runs a classifier or heuristic scorer.
- Measures accuracy, confidence, calibration, and noise robustness.
- Saves results for analysis.

## Data & grounding

- Input is a CSV of EEG samples (`data/eeg_samples.csv`); noise is added synthetically at controlled levels.
- The idea is that a trustworthy classifier should degrade gracefully and stay well-calibrated as signal quality drops, so this measures robustness and calibration rather than peak accuracy.

## License

MIT — see [LICENSE](LICENSE).
