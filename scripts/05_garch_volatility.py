"""
05_garch_volatility.py
----------------------
Two-step volatility analysis.

Step 1: estimate GARCH(1,1) on monthly S&P 500 returns and extract the
        conditional volatility series.
Step 2: regress conditional volatility on the MPUI, reporting both HC3
        and Newey-West HAC standard errors.

Tests H3 (MPUI positively associated with conditional volatility). The
relationship is significant but negative, so H3 is rejected in its
hypothesised direction.

The conditional volatility series is a generated dependent variable, so
second-stage standard errors do not account for first-stage estimation
uncertainty. A single-step GARCH-X specification would avoid this.

Input  : data/master_dataset.csv
Output : data/conditional_volatility.csv
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')


def main():
    df = pd.read_csv(os.path.join(DATA, 'master_dataset.csv'),
                     parse_dates=['date'], index_col='date')
    returns = df['sp500_return']

    print('=' * 60)
    print('STEP 1 — GARCH(1,1) ESTIMATION')
    print('=' * 60)

    model = arch_model(returns, vol='GARCH', p=1, q=1,
                       dist='normal', rescale=False)
    res = model.fit(disp='off')
    print(res.summary())

    a = res.params['alpha[1]']
    b = res.params['beta[1]']
    cv = res.conditional_volatility

    print(f'\nalpha + beta         : {a + b:.4f}')
    print(f'Half-life (months)   : '
          f'{np.log(0.5) / np.log(a + b):.1f}')
    print(f'Mean cond volatility : {cv.mean():.4f}')
    print(f'Max cond volatility  : {cv.max():.4f} '
          f'({cv.idxmax().date()})')

    out = pd.DataFrame({
        'sp500_return': returns,
        'cond_vol':     cv,
        'mpui':         df['mpui'],
    })
    path = os.path.join(DATA, 'conditional_volatility.csv')
    out.to_csv(path)
    print(f'\nSaved to {path}')

    print('\n' + '=' * 60)
    print('STEP 2 — VOLATILITY REGRESSION')
    print('=' * 60)

    d = df.join(out[['cond_vol']], how='inner').dropna()
    y = d['cond_vol']
    X = sm.add_constant(d['mpui'])

    hc3 = sm.OLS(y, X).fit(cov_type='HC3')
    hac = sm.OLS(y, X).fit(cov_type='HAC',
                           cov_kwds={'maxlags': 6})

    print(f'\n{"":<16}{"HC3":>14}{"Newey-West HAC":>18}')
    print('-' * 48)
    for lab, key in [('Coefficient', 'params'),
                     ('Std error',   'bse'),
                     ('z-statistic', 'tvalues'),
                     ('p-value',     'pvalues')]:
        v1 = getattr(hc3, key)['mpui']
        v2 = getattr(hac, key)['mpui']
        print(f'{lab:<16}{v1:>14.4f}{v2:>18.4f}')
    print(f'{"R-squared":<16}{hc3.rsquared:>14.4f}'
          f'{hac.rsquared:>18.4f}')
    print(f'{"N":<16}{int(hc3.nobs):>14}{int(hac.nobs):>18}')

    corr = d['mpui'].corr(d['cond_vol'])
    dw   = durbin_watson(hc3.resid)
    lb   = acorr_ljungbox(hc3.resid, lags=[6, 12], return_df=True)

    print(f'\nCorrelation (MPUI, cond vol) : {corr:.4f}')
    print(f'Durbin-Watson                : {dw:.4f}')
    print('\nLjung-Box test on residuals:')
    print(lb.round(4).to_string())
    print('\nSerial correlation is strong, as expected given the '
          'persistence of conditional volatility. The HAC estimates '
          'are therefore preferred.')

    # Economic magnitude
    beta = hc3.params['mpui']
    sd   = d['mpui'].std()
    mean = d['cond_vol'].mean()
    print(f'\nEconomic magnitude:')
    print(f'  beta x sd(MPUI)     : {beta * sd:.4f} pp')
    print(f'  as % of mean cond vol: {abs(beta * sd) / mean * 100:.1f}%')

    print('\n' + '=' * 60)
    print('SUB-SAMPLE VOLATILITY REGRESSIONS')
    print('=' * 60)
    regimes = {
        'Pre-Crisis (2000-2007)':    ('2000-01-01', '2007-12-31'),
        'Crisis/ZLB (2008-2015)':    ('2008-01-01', '2015-12-31'),
        'Normalisation (2016-2019)': ('2016-01-01', '2019-12-31'),
        'COVID/Tight (2020-2024)':   ('2020-01-01', '2024-12-31'),
    }
    print(f'\n{"Period":<28}{"Coef":>10}{"p(HC3)":>10}'
          f'{"p(HAC)":>10}{"R2":>9}{"N":>6}')
    print('-' * 73)
    for label, (s, e) in regimes.items():
        sub = d[s:e]
        Xs = sm.add_constant(sub['mpui'])
        h = sm.OLS(sub['cond_vol'], Xs).fit(cov_type='HC3')
        n = sm.OLS(sub['cond_vol'], Xs).fit(
            cov_type='HAC', cov_kwds={'maxlags': 6})
        print(f'{label:<28}{h.params["mpui"]:>10.2f}'
              f'{h.pvalues["mpui"]:>10.4f}{n.pvalues["mpui"]:>10.4f}'
              f'{h.rsquared:>9.4f}{int(h.nobs):>6}')


if __name__ == '__main__':
    main()
