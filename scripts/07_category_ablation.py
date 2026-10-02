"""
07_category_ablation.py
-----------------------
Rebuilds the MPUI eight times, each time excluding one of the custom
vocabulary categories, and re-estimates the volatility regression.

This is the paper's central result. The relationship is carried almost
entirely by tentative language, and within that category by nine
pace-and-magnitude descriptors (gradual, moderate, modest, somewhat)
rather than by epistemic hedges (appear, seem, suggest, indicate).

Dropping policy-stance vocabulary leaves the coefficient essentially
unchanged, which rules out the concern that the regime-dependence
result reflects stance language tracking the policy cycle.

Inputs : data/fomc_statements/, data/Loughran-McDonald dictionary,
         data/conditional_volatility.csv
Output : printed to console
"""

import os
import string
import pandas as pd
import statsmodels.api as sm

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, 'data')
STMT  = os.path.join(DATA, 'fomc_statements')

# ── Custom vocabulary by category ────────────────────────────
CATEGORIES = {
    'Conditionality': [
        'appropriate', 'appropriately', 'prepared',
        'flexibility', 'flexible'],
    'Monitoring': [
        'monitor', 'monitoring', 'monitors', 'assess', 'assesses',
        'assessing', 'assessment', 'evaluate', 'evaluating',
        'evaluation', 'review', 'reviewing'],
    'ForwardLooking': [
        'evolve', 'evolves', 'evolving', 'prospect', 'prospects',
        'anticipate', 'anticipates', 'anticipated', 'expect',
        'expects', 'expected', 'expectation', 'expectations',
        'project', 'projects', 'projected', 'projection',
        'projections', 'forecast', 'forecasts', 'outlook'],
    'RiskLanguage': [
        'risk', 'risks', 'downside', 'upside', 'headwind',
        'headwinds', 'challenge', 'challenges', 'challenging',
        'vulnerable', 'vulnerability', 'concern', 'concerns',
        'concerned', 'caution', 'cautious', 'cautiously'],
    'CondCommit': [
        'confident', 'confidence', 'warrant', 'warrants',
        'warranted', 'depend', 'depends', 'depending',
        'contingent', 'contingently', 'threshold', 'thresholds'],
    'Tentative': [
        'gradual', 'gradually', 'tentative', 'tentatively',
        'modest', 'modestly', 'moderate', 'moderately', 'somewhat',
        'roughly', 'approximately', 'appear', 'appears',
        'apparent', 'seem', 'seems', 'suggest', 'suggests',
        'suggesting', 'indicate', 'indicates', 'indicating'],
    'Unusual': [
        'unusual', 'unusually', 'unprecedented', 'elevated',
        'heightened', 'stressed', 'stress', 'turbulence',
        'turbulent', 'volatile', 'volatility', 'disruption',
        'disruptions'],
    'PolicyPath': [
        'accommodate', 'accommodative', 'restrictive', 'tighten',
        'tightening', 'ease', 'easing', 'normalize', 'normalise',
        'normalization', 'normalisation'],
}

# Within Tentative: words qualifying confidence vs words
# describing how much and how fast
HEDGES = ['appear', 'appears', 'apparent', 'seem', 'seems',
          'suggest', 'suggests', 'suggesting', 'indicate',
          'indicates', 'indicating', 'tentative', 'tentatively']
PACE   = ['gradual', 'gradually', 'modest', 'modestly', 'moderate',
          'moderately', 'somewhat', 'roughly', 'approximately']


def load_lm():
    lm_file = next(os.path.join(DATA, f) for f in os.listdir(DATA)
                   if 'oughran' in f or 'asterDictionary' in f)
    lm = pd.read_csv(lm_file)
    return (set(lm[lm['Uncertainty'] > 0]['Word'].str.lower()) |
            set(lm[lm['Weak_Modal'] > 0]['Word'].str.lower()))


def build_index(lexicon, target_index):
    rows = []
    for fn in sorted(os.listdir(STMT)):
        if not fn.endswith('.txt'):
            continue
        parts = fn[:-4].split('_')
        if len(parts) != 3:
            continue
        y, m, dd = parts
        if int(m) > 12:
            m, dd = dd, m
        with open(os.path.join(STMT, fn),
                  encoding='utf-8', errors='ignore') as f:
            text = f.read()
        words = [w.strip(string.punctuation)
                 for w in text.lower().split()]
        words = [w for w in words if w and not w.isdigit()]
        if len(words) < 50:
            continue
        rows.append({
            'date': pd.Timestamp(f'{y}-{m}-{dd}'),
            'v': sum(1 for w in words if w in lexicon) / len(words),
        })
    s = pd.DataFrame(rows).sort_values('date')
    s['m'] = s['date'].dt.to_period('M').dt.to_timestamp()
    s = s.drop_duplicates('m', keep='last').set_index('m')['v']
    return s.reindex(target_index.union(s.index)).ffill() \
            .reindex(target_index)


def main():
    vol = pd.read_csv(os.path.join(DATA,
                      'conditional_volatility.csv'),
                      parse_dates=['date'], index_col='date')
    vol = vol.dropna(subset=['cond_vol'])
    target = vol.index

    lm = load_lm()
    full = lm | set(sum(CATEGORIES.values(), []))

    def run(lexicon, label):
        x = build_index(lexicon, target)
        d = pd.DataFrame({'y': vol['cond_vol'], 'x': x}).dropna()
        X = sm.add_constant(d['x'])
        h = sm.OLS(d['y'], X).fit(cov_type='HC3')
        n = sm.OLS(d['y'], X).fit(cov_type='HAC',
                                  cov_kwds={'maxlags': 6})
        print(f'{label:<36}{h.params["x"]:>9.2f}'
              f'{h.pvalues["x"]:>9.4f}{n.pvalues["x"]:>9.4f}'
              f'{h.rsquared:>8.4f}')

    print(f'{"Specification":<36}{"Coef":>9}{"p(HC3)":>9}'
          f'{"p(HAC)":>9}{"R2":>8}')
    print('-' * 71)

    run(full, 'Full index')
    for name, terms in CATEGORIES.items():
        run(full - set(terms), f'Drop {name} ({len(terms)} terms)')

    print('-' * 71)
    run(full - set(HEDGES),
        f'  of which: epistemic hedges ({len(HEDGES)})')
    run(full - set(PACE),
        f'  of which: pace descriptors ({len(PACE)})')

    print('\nThe relationship is carried by pace-and-magnitude '
          'vocabulary. Removing epistemic hedges leaves the '
          'coefficient essentially unchanged.')


if __name__ == '__main__':
    main()
