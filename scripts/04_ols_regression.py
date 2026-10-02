"""
04_ols_regression.py
--------------------
Augmented Dickey-Fuller unit root tests and OLS regressions of monthly
S&P 500 returns on the MPUI, the Federal Funds Rate, inflation and
industrial production growth.

Tests H1 (MPUI negatively associated with returns) and H2 (MPUI adds
explanatory power beyond the controls). Both are rejected.

Input  : data/master_dataset.csv
Outputs: printed to console; model objects returned by main()
"""

import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')

CONTROLS = ['fed_funds_rate', 'inflation', 'ip_growth']


def unit_root_tests(df, variables):
    print(f'\n{"Variable":<20}{"ADF":>10}{"p-value":>10}'
          f'{"Lags":>7}{"Stationary":>12}')
    print('-' * 59)
    for v in variables:
        stat, p, lags = adfuller(df[v].dropna(), autolag='AIC')[:3]
        print(f'{v:<20}{stat:>10.3f}{p:>10.3f}{lags:>7}'
              f'{"Yes" if p < 0.05 else "No":>12}')


def report(model, name, key):
    print(f'\n{name}')
    print(f'  coefficient : {model.params[key]:>9.3f}')
    print(f'  std error   : {model.bse[key]:>9.3f}')
    print(f'  t-statistic : {model.tvalues[key]:>9.3f}')
    print(f'  p-value     : {model.pvalues[key]:>9.4f}')
    print(f'  R-squared   : {model.rsquared:>9.4f}')
    print(f'  adj R-sq    : {model.rsquared_adj:>9.4f}')
    print(f'  N           : {int(model.nobs):>9}')


def main():
    df = pd.read_csv(os.path.join(DATA, 'master_dataset.csv'),
                     parse_dates=['date'], index_col='date')
    print(f'Loaded {len(df)} observations')

    print('\n' + '=' * 59)
    print('AUGMENTED DICKEY-FULLER UNIT ROOT TESTS')
    print('=' * 59)
    unit_root_tests(df, ['sp500_return'] + CONTROLS +
                        ['mpui', 'mpui_adjusted'])

    y = df['sp500_return']

    print('\n' + '=' * 59)
    print('OLS REGRESSIONS — S&P 500 RETURNS (HC3 robust)')
    print('=' * 59)

    # Model 1: baseline with MPUI
    X1 = sm.add_constant(df[['fed_funds_rate', 'mpui',
                             'inflation', 'ip_growth']])
    m1 = sm.OLS(y, X1).fit(cov_type='HC3')
    report(m1, 'Model 1 — baseline with MPUI', 'mpui')

    # Model 2: benchmark without MPUI (H2 comparison)
    X2 = sm.add_constant(df[CONTROLS])
    m2 = sm.OLS(y, X2).fit(cov_type='HC3')
    print(f'\nModel 2 — without MPUI')
    print(f'  R-squared   : {m2.rsquared:>9.4f}')
    print(f'  adj R-sq    : {m2.rsquared_adj:>9.4f}')

    # Model 3: length-adjusted MPUI
    X3 = sm.add_constant(df[['fed_funds_rate', 'mpui_adjusted',
                             'inflation', 'ip_growth']])
    m3 = sm.OLS(y, X3).fit(cov_type='HC3')
    report(m3, 'Model 3 — length-adjusted MPUI', 'mpui_adjusted')

    print(f'\nIncremental R-squared from MPUI: '
          f'{m1.rsquared - m2.rsquared:.4f}')
    print('H2 is tested by the t-statistic on the MPUI coefficient, '
          'not by the R-squared comparison.')

    print('\n' + '=' * 59)
    print('SUB-SAMPLE REGRESSIONS — RETURNS')
    print('=' * 59)
    regimes = {
        'Pre-Crisis (2000-2007)':    ('2000-01-01', '2007-12-31'),
        'Crisis/ZLB (2008-2015)':    ('2008-01-01', '2015-12-31'),
        'Normalisation (2016-2019)': ('2016-01-01', '2019-12-31'),
        'COVID/Tight (2020-2024)':   ('2020-01-01', '2024-12-31'),
    }
    print(f'\n{"Period":<28}{"Coef":>10}{"p-value":>10}'
          f'{"R2":>9}{"N":>6}')
    print('-' * 63)
    for label, (s, e) in regimes.items():
        sub = df[s:e]
        X = sm.add_constant(sub[['fed_funds_rate', 'mpui',
                                 'inflation', 'ip_growth']])
        m = sm.OLS(sub['sp500_return'], X).fit(cov_type='HC3')
        print(f'{label:<28}{m.params["mpui"]:>10.2f}'
              f'{m.pvalues["mpui"]:>10.4f}{m.rsquared:>9.4f}'
              f'{int(m.nobs):>6}')

    return m1, m2, m3


if __name__ == '__main__':
    main()
