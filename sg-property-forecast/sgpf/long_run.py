"""Explainable long-run (3-10 year) price scenarios: prices grow with incomes.

    price in N years = price today x (1 + income growth over N years) x historical deviation

* income = nominal GDP per resident (SingStat GDP / population). Over several years what
  buyers can pay is anchored by what they earn, and it is the single strongest driver of
  10-year price growth in Singapore's history (correlation +0.86).
* The central forecast has NO fitted parameters: price growth = income growth.
  Income growth is the scenario input (bear / base / bull).
* The range comes from how far prices historically ran ahead of or behind incomes over the
  same horizon ("excess growth" = price growth - income growth), measured on every window
  since 1985. It captures everything the rule leaves out: rate cycles, cooling measures,
  supply gluts, crises.

Tested and rejected (made out-of-sample errors worse, see README): fitting the income
elasticity freely, a valuation mean-reversion term, and the supply pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import fetch_price_index
from .macro import fetch_macro

EVAL_START = "1985Q4"  # first origin with 10 years of income history for the "guessed" test
LIKELY, WIDE = (0.10, 0.90), (0.05, 0.95)


def build_panel(series: str = "All Residential", h: int = 40) -> pd.DataFrame:
    """Quarterly panel with realised h-quarter price and income growth (log)."""
    P = fetch_price_index()[series]
    q = pd.period_range("1975Q1", P.index[-1], freq="Q")
    m = fetch_macro().reindex(q)
    pop = np.exp(np.log(m.population).interpolate(limit_area="inside")).ffill()
    log_inc = np.log(m.gdp_nominal.rolling(4).sum() / pop)
    log_p = np.log(P.reindex(q))
    d = pd.DataFrame({"log_price": log_p, "log_income": log_inc}, index=q)
    d["growth"] = log_p.shift(-h) - log_p
    d["income_growth"] = log_inc.shift(-h) - log_inc
    # What you'd have guessed at the time: last decade's income pace, scaled to horizon.
    d["income_guess"] = (log_inc - log_inc.shift(40)) / 40 * h
    d["excess"] = d.growth - d.income_growth
    return d


def evaluate(h: int, series: str = "All Residential") -> dict:
    """Out-of-sample errors (in percentage points of total growth) for one horizon."""
    d = build_panel(series, h).dropna(subset=["growth", "income_growth", "income_guess"])
    d = d[d.index >= pd.Period(EVAL_START, "Q")]
    pct = lambda x: (np.exp(x) - 1) * 100
    act = pct(d.growth)
    out = {"years": h / 4, "n_windows": len(d), "independent_windows": round(len(d) / h, 1),
           "first": str(d.index[0]), "last": str(d.index[-1])}
    for name, pred in {"flat": act * 0, "income_known": pct(d.income_growth),
                       "income_guessed": pct(d.income_guess)}.items():
        e = pred - act
        out[f"{name}_MAE"] = e.abs().mean()
        out[f"{name}_median"] = e.abs().median()
        out[f"{name}_p90"] = e.abs().quantile(0.9)
    # Band coverage: band built from windows that do not overlap the one being tested.
    for label, (lo_q, hi_q) in {"likely": LIKELY, "wide": WIDE}.items():
        hits = []
        for t in d.index:
            other = d.excess[(d.index <= t - h) | (d.index >= t + h)]
            lo, hi = other.quantile([lo_q, hi_q])
            hits.append(lo <= d.excess[t] <= hi)
        out[f"{label}_band_coverage"] = float(np.mean(hits))
    return out


def excess_quantiles(h: int, series: str = "All Residential") -> pd.Series:
    d = build_panel(series, h).dropna(subset=["excess"])
    d = d[d.index >= pd.Period(EVAL_START, "Q")]
    return d.excess.quantile([WIDE[0], LIKELY[0], 0.5, LIKELY[1], WIDE[1]])


def scenarios(income_growth: dict[str, float], years: int = 10,
              series: str = "All Residential") -> pd.DataFrame:
    """Yearly index paths for each scenario's annual income growth (e.g. {"base": 0.045}).

    Columns per scenario: central, likely_lo/hi (10-90th pct), wide_lo/hi (5-95th pct).
    """
    P = fetch_price_index()[series].dropna()
    p0, t0 = P.iloc[-1], P.index[-1]
    q = {y: excess_quantiles(4 * y, series) for y in range(1, years + 1)}
    rows = []
    for name, g in income_growth.items():
        for y in range(1, years + 1):
            inc = y * np.log1p(g)
            lvl = lambda x: p0 * np.exp(inc + x)
            qq = q[y].values
            rows.append({"scenario": name, "income_growth_pa": g, "years": y,
                         "quarter": str(t0 + 4 * y), "central": lvl(0),
                         "wide_lo": lvl(qq[0]), "likely_lo": lvl(qq[1]),
                         "likely_hi": lvl(qq[3]), "wide_hi": lvl(qq[4])})
    return pd.DataFrame(rows)


def income_history(series: str = "All Residential") -> pd.Series:
    """Annualised income growth (%) for each decade, by start year, for anchoring scenarios."""
    d = build_panel(series, 40)
    g = (np.exp(d.income_growth / 10) - 1) * 100
    return g[g.index.quarter == 2].dropna().round(1)
