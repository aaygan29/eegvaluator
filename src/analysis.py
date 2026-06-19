from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def analyze(path='results/summary.csv'):
    df = pd.read_csv(path)
    Path('results').mkdir(exist_ok=True)

    ax = df.plot(x='noise_level', y='accuracy', marker='o', legend=False, title='Accuracy vs noise level')
    ax.set_xlabel('Noise level')
    ax.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('results/accuracy_by_noise.png', dpi=200)
    plt.close()

    ax = df.plot(x='noise_level', y='calibration_error', marker='o', legend=False, title='Calibration error vs noise level')
    ax.set_xlabel('Noise level')
    ax.set_ylabel('Calibration error')
    plt.tight_layout()
    plt.savefig('results/calibration_by_noise.png', dpi=200)
    plt.close()

    print(df)


if __name__ == '__main__':
    analyze()
