"""Explainable long-run (3-10 year) price scenarios: prices grow with incomes.

    price in N years = price today x (1 + income growth over N years) x historical deviation
    income growth    = real income growth + inflation (CPI)

Results come in two units:
* future dollars ("nominal"): the price tag in that year.
* today's dollars ("real"): the nominal price divided by cumulative CPI inflation, i.e.
  purchasing power. In today's dollars the rule reads: prices grow with REAL incomes, so the
  inflation assumption changes the price tag but not the real result.

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
    log_cpi = np.log(m.cpi)
    d["inflation"] = log_cpi.shift(-h) - log_cpi  # realised h-quarter CPI inflation (log)
    d["real_growth"] = d.growth - d.inflation
    d["real_income_growth"] = d.income_growth - d.inflation
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
    # Same test in today's dollars (inflation-adjusted). Baseline: prices just keep pace with
    # inflation. The rule's real error equals its nominal error deflated by realised CPI.
    act_r = pct(d.real_growth)
    for name, pred in {"real_flat": act_r * 0, "real_income_known": pct(d.real_income_growth)}.items():
        e = pred - act_r
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


def scenarios(real_income_growth: dict[str, float], inflation: float, years: int = 10,
              series: str = "All Residential") -> pd.DataFrame:
    """Yearly index paths per scenario, e.g. scenarios({"base": 0.029}, inflation=0.016).

    Nominal income growth = (1 + real income growth) x (1 + inflation) - 1.
    Columns: central / likely_lo / likely_hi / wide_lo / wide_hi in future dollars, the same
    with a `real_` prefix in today's dollars, and total % changes in both units.
    """
    P = fetch_price_index()[series].dropna()
    p0, t0 = P.iloc[-1], P.index[-1]
    q = {y: excess_quantiles(4 * y, series) for y in range(1, years + 1)}
    rows = []
    for name, g_real in real_income_growth.items():
        g_nom = (1 + g_real) * (1 + inflation) - 1
        for y in range(1, years + 1):
            inc = y * np.log1p(g_nom)
            deflator = (1 + inflation) ** y
            row = {"scenario": name, "real_income_growth_pa": g_real, "inflation_pa": inflation,
                   "nominal_income_growth_pa": g_nom, "years": y, "quarter": str(t0 + 4 * y)}
            qq = q[y]
            for col, x in {"wide_lo": qq.iloc[0], "likely_lo": qq.iloc[1], "central": 0.0,
                           "likely_hi": qq.iloc[3], "wide_hi": qq.iloc[4]}.items():
                row[col] = p0 * np.exp(inc + x)
                row[f"real_{col}"] = row[col] / deflator
            row["change_pct"] = (row["central"] / p0 - 1) * 100
            row["real_change_pct"] = (row["real_central"] / p0 - 1) * 100
            rows.append(row)
    return pd.DataFrame(rows)


def decade_history(series: str = "All Residential") -> pd.DataFrame:
    """Annualised growth (%) over each 10-year window, by start quarter (Q2 of each year):
    nominal and real income, CPI inflation, nominal and real prices."""
    d = build_panel(series, 40)
    ann = lambda x: (np.exp(x / 10) - 1) * 100
    t = pd.DataFrame({"income_nominal": ann(d.income_growth),
                      "inflation": ann(d.inflation),
                      "income_real": ann(d.real_income_growth),
                      "price_nominal": ann(d.growth),
                      "price_real": ann(d.real_growth)})
    return t[t.index.quarter == 2].dropna().round(2)


def scenario_anchors(since: int = 1990, series: str = "All Residential") -> dict[str, float]:
    """Bear/base/bull real income growth = weakest/median/strongest decade since `since`;
    inflation = median decade. Returned as fractions."""
    h = decade_history(series)
    h = h[h.index.year >= since]
    r = h.income_real
    return {"bear": r.min() / 100, "base": r.median() / 100, "bull": r.max() / 100,
            "inflation": h.inflation.median() / 100}
