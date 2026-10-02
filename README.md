# Monetary Policy Uncertainty and Equity Volatility

Textual analysis of 208 FOMC statements (2000–2024) testing whether
dictionary-based uncertainty indices measure what they claim to.

MSc Asset Pricing dissertation, King's Business School, 2026.

---

## Summary

I construct a Monetary Policy Uncertainty Index (MPUI) from FOMC
post-meeting statements using the Loughran-McDonald financial lexicon
plus a custom FOMC-specific vocabulary, and test it against S&P 500
returns and GARCH(1,1) conditional volatility across 299 monthly
observations.

**No relationship with returns** is found in any specification.

**The index is significantly negatively associated with conditional
volatility** (β = −50.98, p < 0.001, R² = 0.142), a relationship robust
to lagging, crisis exclusion, alternative normalisation, macroeconomic
controls and influence diagnostics.

**But rebuilding the index from validated Loughran-McDonald uncertainty
terms alone eliminates it entirely** (β = −12.20, p = 0.205, R² = 0.006),
and a category ablation locates most of the effect in nine
pace-and-magnitude descriptors — *gradual*, *moderate*, *modest*,
*somewhat* — rather than in hedging or uncertainty vocabulary.

The index appears to track the linguistic register of steady economic
conditions rather than monetary policy uncertainty. Constructions such
as "expanding at a moderate pace" are dense in statements written during
calm periods and fall away during acute stress, when statements become
short and declarative.

---

## Methods

| Component | Approach |
|---|---|
| Index construction | Three-layer lexicon, 381 unique terms, normalised by statement length |
| Returns | OLS with HC3 heteroskedasticity-robust standard errors |
| Volatility | Two-step: GARCH(1,1) conditional variance, then OLS with HC3 and Newey-West HAC |
| Robustness | Eleven specifications including lagged, crisis-excluded, length-adjusted, macro controls, alternative lexicon, retention convention |
| Diagnostics | ADF unit root tests, Durbin-Watson, Ljung-Box, Cook's distance, DFBETAs |
| Regime analysis | Interaction model with Wald test of coefficient equality across four monetary policy regimes |
| Ablation | Index rebuilt eight times, each excluding one vocabulary category |

---

## Repository structure

```
scripts/
  01_collect_data.py        FRED API and Yahoo Finance download
  02_build_mpui.py          Index construction from FOMC text
  03_merge_data.py          Merge and forward-fill to monthly
  04_ols_regression.py      Unit root tests and return regressions
  05_garch_volatility.py    GARCH(1,1) and volatility regression
  06_robustness.py          Eleven specifications, Wald test
  07_category_ablation.py   Vocabulary category ablation

vocabulary/
  custom_fomc_terms.txt     The 113 custom terms, grouped by category

data/
  master_dataset.csv        299 monthly observations, 2000–2024

figures/
  figure1_variables.png     Time series of key variables
  figure2_mpui.png          MPUI, length-adjusted MPUI, statement length
  figure3_volatility.png    Conditional volatility and MPUI scatter
```

---

## Running it

Requires Python 3.11 or later.

```
pip install pandas numpy statsmodels arch matplotlib yfinance requests
```

Two external inputs are needed and are not included here:

**FRED API key** — free, from
https://fred.stlouisfed.org/docs/api/api_key.html
Set it at the top of `01_collect_data.py`.

**Loughran-McDonald Master Dictionary** — from
https://sraf.nd.edu/loughranmcdonald-master-dictionary/
Place the CSV in `data/`.

**FOMC statements** — 208 post-meeting statements from
federalreserve.gov, saved as `data/fomc_statements/YYYY_MM_DD.txt`.

Scripts run in numerical order.

---

## Data sources

| Series | Source |
|---|---|
| S&P 500 | Yahoo Finance (^GSPC) |
| Federal Funds Rate | FRED (DFF) |
| CPI | FRED (CPIAUCSL) |
| Industrial Production | FRED (INDPRO) |
| FOMC statements | federalreserve.gov |
| Financial lexicon | Loughran & McDonald (2011, 2016) |

---

## Limitations

The Federal Funds Rate is endogenous, so coefficients are associations
rather than structural effects; futures-based policy surprises following
Kuttner (2001) would improve identification. The two-step volatility
design treats a generated series as an observed dependent variable; a
single-step GARCH-X specification would avoid this. The custom
vocabulary reflects researcher judgement, which is the subject of the
ablation rather than an assumption of it.
