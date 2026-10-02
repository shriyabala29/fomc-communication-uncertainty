"""
06_robustness.py
----------------
Eleven robustness specifications for the MPUI-volatility relationship,
plus influence diagnostics and a formal Wald test of coefficient
equality across monetary policy regimes.

Specifications
   (1) Baseline
   (2) One-month lagged MPUI          — temporal alignment
   (3) Excluding the financial crisis
   (4) Excluding the COVID-19 shock
   (5) Excluding both
   (6) Length-adjusted MPUI           — normalisation method
   (7) With macroeconomic controls    — omitted variables
   (8) Loughran-McDonald terms only   — custom vocabulary
   (9) Excluding |z| > 2.5            — high-leverage observations
  (10) Excluding Cook's D outliers
  (11) First-statement retention      — multi-meeting convention

Inputs : data/master_dataset.csv, data/conditional_volatility.csv
Output : printed to console
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')

GFC   = ('2008-09-01', '2009-06-30')
COVID = ('2020-02-01', '2020-06-30')


def fit(y, X, key):
    """Return coefficient, SE, HC3 p-value, HAC p-value, R2, N."""
    h = sm.OLS(y, X).fit(cov_type='HC3')
    n = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
    return (h.params[key], h.bse[key], h.pvalues[key],
            n.pvalues[key], h.rsquared, int(h.nobs))


def line(label, r):
    print(f'{label:<34}{r[0]:>9.2f}{r[1]:>9.3f}'
          f'{r[2]:>9.4f}{r[3]:>9.4f}{r[4]:>8.4f}{r[5]:>6}')


def main():
    df  = pd.read_csv(os.path.join(DATA, 'master_dataset.csv'),
                      parse_dates=['date'], index_col='date')
    vol = pd.read_csv(os.path.join(DATA,
                      'conditional_volatility.csv'),
                      parse_dates=['date'], index_col='date')

    d = df.join(vol[['cond_vol']], how='inner').dropna()
    print(f'Loaded {len(d)} observations\n')

    print(f'{"Specification":<34}{"Coef":>9}{"SE":>9}'
          f'{"p(HC3)":>9}{"p(HAC)":>9}{"R2":>8}{"N":>6}')
    print('-' * 84)

    # (1) Baseline
    line('(1) Baseline',
         fit(d['cond_vol'], sm.add_constant(d['mpui']), 'mpui'))

    # (2) Lagged MPUI
    lag = d.copy()
    lag['mpui_lag'] = lag['mpui'].shift(1)
    lag = lag.dropna(subset=['mpui_lag'])
    line('(2) Lagged MPUI (t-1)',
         fit(lag['cond_vol'],
             sm.add_constant(lag['mpui_lag']), 'mpui_lag'))

    # (3)-(5) Crisis exclusions
    mg = ~((d.index >= GFC[0])   & (d.index <= GFC[1]))
    mc = ~((d.index >= COVID[0]) & (d.index <= COVID[1]))
    for mask, label in [(mg, '(3) Excluding GFC'),
                        (mc, '(4) Excluding COVID-19'),
                        (mg & mc, '(5) Excluding both')]:
        s = d[mask]
        line(label,
             fit(s['cond_vol'], sm.add_constant(s['mpui']), 'mpui'))

    # (6) Length-adjusted
    line('(6) Length-adjusted MPUI',
         fit(d['cond_vol'],
             sm.add_constant(d['mpui_adjusted']), 'mpui_adjusted'))

    # (7) Macro controls
    line('(7) With macro controls',
         fit(d['cond_vol'],
             sm.add_constant(d[['mpui', 'fed_funds_rate',
                                'inflation', 'ip_growth']]), 'mpui'))

    # (9)-(10) Influence diagnostics
    base = sm.OLS(d['cond_vol'],
                  sm.add_constant(d['mpui'])).fit()
    infl = base.get_influence()
    d = d.copy()
    d['cooks_d'] = infl.cooks_distance[0]
    d['z'] = (d['mpui'] - d['mpui'].mean()) / d['mpui'].std()

    trim = d[d['z'].abs() <= 2.5]
    line('(9) Excluding |z| > 2.5',
         fit(trim['cond_vol'],
             sm.add_constant(trim['mpui']), 'mpui'))

    cook = d[d['cooks_d'] <= 4 / len(d)]
    line("(10) Excluding Cook's D outliers",
         fit(cook['cond_vol'],
             sm.add_constant(cook['mpui']), 'mpui'))

    print('\nObservations dropped at |z| > 2.5:')
    for dt, row in d[d['z'].abs() > 2.5].iterrows():
        print(f'  {dt.date()}  MPUI {row["mpui"]:.4f}  '
              f'z {row["z"]:+.2f}')

    # ── Regime interaction and Wald test ─────────────────────
    print('\n' + '=' * 60)
    print('REGIME INTERACTION MODEL AND WALD TEST')
    print('=' * 60)

    d['d1'] = ((d.index >= '2008-01-01') &
               (d.index <= '2015-12-31')).astype(int)
    d['d2'] = ((d.index >= '2016-01-01') &
               (d.index <= '2019-12-31')).astype(int)
    d['d3'] = (d.index >= '2020-01-01').astype(int)
    for k in '123':
        d[f'mx{k}'] = d['mpui'] * d[f'd{k}']

    X = sm.add_constant(d[['mpui', 'd1', 'd2', 'd3',
                           'mx1', 'mx2', 'mx3']])
    hc3 = sm.OLS(d['cond_vol'], X).fit(cov_type='HC3')
    hac = sm.OLS(d['cond_vol'], X).fit(
        cov_type='HAC', cov_kwds={'maxlags': 6})

    restriction = 'mx1 = 0, mx2 = 0, mx3 = 0'
    w1 = hc3.wald_test(restriction, use_f=True, scalar=True)
    w2 = hac.wald_test(restriction, use_f=True, scalar=True)

    print(f'\nH0: the MPUI coefficient is equal across all four '
          f'regimes\n')
    print(f'Wald test (HC3) : F = {w1.statistic:.3f}, '
          f'p = {w1.pvalue:.4f}')
    print(f'Wald test (HAC) : F = {w2.statistic:.3f}, '
          f'p = {w2.pvalue:.4f}')

    base_coef = hc3.params['mpui']
    print(f'\nImplied total effect by regime:')
    print(f'  Pre-Crisis       {base_coef:>9.2f}')
    for k, name in [('1', 'Crisis/ZLB'),
                    ('2', 'Normalisation'),
                    ('3', 'COVID/Tightening')]:
        print(f'  {name:<16} '
              f'{base_coef + hc3.params[f"mx{k}"]:>9.2f}')


if __name__ == '__main__':
    main()
