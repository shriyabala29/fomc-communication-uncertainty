"""
01_collect_data.py
------------------
Downloads S&P 500 prices from Yahoo Finance and macroeconomic series
from the FRED API, then writes a combined monthly dataset.

Requires a free FRED API key:
https://fred.stlouisfed.org/docs/api/api_key.html

Output: data/macro_data.csv
"""

import os
import pandas as pd
import requests
import yfinance as yf

# ── Configuration ────────────────────────────────────────────
API_KEY = 'YOUR_FRED_API_KEY_HERE'

START = '2000-01-01'
END   = '2024-12-31'

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
os.makedirs(DATA, exist_ok=True)


# ── S&P 500 from Yahoo Finance ───────────────────────────────
def get_sp500():
    raw = yf.download('^GSPC', start=START, end=END,
                      interval='1mo', auto_adjust=True,
                      progress=False)
    close = raw['Close'].copy()
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    returns = close.pct_change() * 100
    returns.name = 'sp500_return'
    df = returns.to_frame().dropna()
    df.index = pd.to_datetime(df.index).to_period('M').to_timestamp()
    return df


# ── FRED series via API ──────────────────────────────────────
def get_fred(series_id, api_key, start=START, end=END):
    url = (
        'https://api.stlouisfed.org/fred/series/observations'
        f'?series_id={series_id}'
        f'&api_key={api_key}'
        f'&observation_start={start}'
        f'&observation_end={end}'
        '&frequency=m&file_type=json'
    )
    r = requests.get(url, timeout=60)
    obs = r.json()['observations']
    df = pd.DataFrame(obs)[['date', 'value']]
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    s = df.set_index('date')['value']
    s.name = series_id
    return s


def main():
    if API_KEY == 'YOUR_FRED_API_KEY_HERE':
        raise SystemExit(
            'Set API_KEY to your own FRED key before running. '
            'Get one free at '
            'https://fred.stlouisfed.org/docs/api/api_key.html'
        )

    print('Downloading S&P 500...')
    sp500 = get_sp500()
    print(f'  {len(sp500)} observations')

    print('Downloading FRED series...')
    ffr = get_fred('DFF',      API_KEY)   # Federal Funds Rate
    cpi = get_fred('CPIAUCSL', API_KEY)   # CPI
    ip  = get_fred('INDPRO',   API_KEY)   # Industrial Production

    macro = pd.DataFrame({
        'fed_funds_rate':        ffr,
        'cpi':                   cpi,
        'industrial_production': ip,
    })
    macro.index = (pd.to_datetime(macro.index)
                     .to_period('M').to_timestamp())

    macro['inflation'] = macro['cpi'].pct_change() * 100
    macro['ip_growth'] = (macro['industrial_production']
                          .pct_change() * 100)

    keep = ['fed_funds_rate', 'inflation', 'ip_growth']
    df = sp500.join(macro[keep], how='inner').dropna()
    df = df[START:END].round(4)

    out = os.path.join(DATA, 'macro_data.csv')
    df.to_csv(out)

    print(f'\nSaved {len(df)} observations to {out}')
    print(f'Range: {df.index[0].date()} to {df.index[-1].date()}')
    print(df.describe().round(4))


if __name__ == '__main__':
    main()
