"""
03_merge_data.py
----------------
Merges the macroeconomic series with the MPUI and writes the master
dataset used throughout the analysis.

The FOMC meets roughly eight times a year, so the MPUI is forward-filled
into months without a meeting: the most recent statement represents the
committee's communication stance until superseded.

Inputs
  data/macro_data.csv
  data/mpui_monthly.csv

Output
  data/master_dataset.csv   (299 monthly observations, 2000-2024)
"""

import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')


def load(name):
    df = pd.read_csv(os.path.join(DATA, name))
    first = df.columns[0]
    df[first] = pd.to_datetime(df[first])
    df = df.set_index(first)
    df.index = df.index.to_period('M').to_timestamp()
    df.index.name = 'date'
    # Guard against duplicate months from either source
    return df[~df.index.duplicated(keep='first')]


def main():
    macro = load('macro_data.csv')
    mpui  = load('mpui_monthly.csv')

    print(f'Macro : {macro.shape}')
    print(f'MPUI  : {mpui.shape}')

    df = macro.join(mpui, how='left')

    for col in ['mpui', 'mpui_adjusted', 'total_words']:
        if col in df.columns:
            df[col] = df[col].ffill()

    df = df.dropna().round(4)

    out = os.path.join(DATA, 'master_dataset.csv')
    df.to_csv(out)

    print(f'\nMaster dataset: {df.shape}')
    print(f'Range: {df.index[0].date()} to {df.index[-1].date()}')
    print(f'Saved to {out}\n')

    cols = ['sp500_return', 'fed_funds_rate', 'inflation',
            'ip_growth', 'mpui', 'mpui_adjusted']
    desc = df[cols].describe().round(4)
    desc.loc['skewness'] = df[cols].skew().round(4)
    desc.loc['kurtosis'] = df[cols].kurt().round(4)
    print('Descriptive statistics:')
    print(desc.to_string())
    print('\nCorrelation matrix:')
    print(df[cols].corr().round(4).to_string())


if __name__ == '__main__':
    main()
