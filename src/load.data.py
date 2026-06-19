import ast
from pathlib import Path
import pandas as pd


def load_eeg_csv(path='data/eeg_samples.csv'):
    path = Path(path)
    df = pd.read_csv(path)
    required = {'sample_id', 'signal', 'label'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    df['signal'] = df['signal'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df
