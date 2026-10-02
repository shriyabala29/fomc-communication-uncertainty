"""
02_build_mpui.py
----------------
Constructs the Monetary Policy Uncertainty Index (MPUI) from FOMC
post-meeting statements using a three-layer lexicon:

  1. Loughran-McDonald uncertainty category   (297 terms)
  2. Loughran-McDonald weak modal category    (27 terms)
  3. Custom FOMC-specific vocabulary          (113 terms)

After deduplication the master lexicon contains 381 unique terms.

Inputs
  data/fomc_statements/YYYY_MM_DD.txt   (208 files)
  data/Loughran-McDonald_MasterDictionary.csv
  vocabulary/custom_fomc_terms.txt

Output
  data/mpui_monthly.csv

The Loughran-McDonald dictionary is not redistributed here.
Download it from https://sraf.nd.edu/loughranmcdonald-master-dictionary/
and place it in data/.
"""

import os
import string
import pandas as pd

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, 'data')
VOCAB = os.path.join(BASE, 'vocabulary')
STMT  = os.path.join(DATA, 'fomc_statements')


# ── Load the three lexicon layers ────────────────────────────
def load_lexicon():
    lm_file = None
    for f in os.listdir(DATA):
        if 'oughran' in f or 'asterDictionary' in f:
            lm_file = os.path.join(DATA, f)
            break
    if lm_file is None:
        raise SystemExit(
            'Loughran-McDonald dictionary not found in data/. '
            'Download from https://sraf.nd.edu/'
        )

    lm = pd.read_csv(lm_file)
    uncertainty = set(lm[lm['Uncertainty'] > 0]['Word'].str.lower())
    weak_modal  = set(lm[lm['Weak_Modal'] > 0]['Word'].str.lower())

    with open(os.path.join(VOCAB, 'custom_fomc_terms.txt')) as f:
        custom = {line.strip().lower()
                  for line in f if line.strip()
                  and not line.startswith('#')}

    master = uncertainty | weak_modal | custom

    print(f'LM uncertainty : {len(uncertainty)}')
    print(f'LM weak modal  : {len(weak_modal)}')
    print(f'Custom FOMC    : {len(custom)}')
    print(f'Master lexicon : {len(master)} unique terms')
    return master


# ── Index construction ───────────────────────────────────────
def compute_mpui(text, lexicon):
    """Proportion of lexicon terms in a statement."""
    words = text.lower().split()
    words = [w.strip(string.punctuation) for w in words]
    words = [w for w in words if w and not w.isdigit()]
    if len(words) < 50:
        return None, 0, len(words)
    count = sum(1 for w in words if w in lexicon)
    return round(count / len(words), 6), count, len(words)


def parse_date(stem):
    """Parse YYYY_MM_DD, correcting any swapped month/day."""
    parts = stem.split('_')
    if len(parts) != 3:
        return None
    y, m, d = parts
    if int(m) > 12:
        m, d = d, m
    return pd.Timestamp(f'{y}-{m}-{d}')


def main():
    lexicon = load_lexicon()

    rows = []
    for fn in sorted(os.listdir(STMT)):
        if not fn.endswith('.txt'):
            continue
        date = parse_date(fn[:-4])
        if date is None:
            continue
        with open(os.path.join(STMT, fn),
                  encoding='utf-8', errors='ignore') as f:
            text = f.read()
        mpui, count, total = compute_mpui(text, lexicon)
        if mpui is None:
            continue
        rows.append({
            'date': date, 'mpui': mpui,
            'uncertainty_count': count, 'total_words': total,
        })

    df = pd.DataFrame(rows).sort_values('date')
    print(f'\nProcessed {len(df)} statements')

    # Length-adjusted variant: scales by rolling average length
    df['rolling_avg_words'] = (df['total_words']
                               .rolling(12, min_periods=1).mean())
    df['mpui_adjusted'] = (df['mpui'] *
                           df['rolling_avg_words'] /
                           df['total_words']).round(6)

    # Collapse to monthly. In months with more than one meeting the
    # later statement is retained: it supersedes the earlier one and
    # represents the committee's operative stance at month end.
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    monthly = (df.drop_duplicates('month', keep='last')
                 .set_index('month')[['mpui', 'mpui_adjusted',
                                      'total_words']])
    monthly.index.name = 'date'

    out = os.path.join(DATA, 'mpui_monthly.csv')
    monthly.to_csv(out)

    print(f'Saved {len(monthly)} monthly values to {out}')
    print(monthly['mpui'].describe().round(6))
    print('\nHighest MPUI:')
    print(monthly.nlargest(3, 'mpui')[['mpui']].round(4))
    print('\nLowest MPUI:')
    print(monthly.nsmallest(3, 'mpui')[['mpui']].round(4))


if __name__ == '__main__':
    main()
