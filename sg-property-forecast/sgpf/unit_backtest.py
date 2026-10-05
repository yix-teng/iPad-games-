"""Backtest the unit forecast at horizons of 1-20 years.

For each origin (end of year T), using only data available at T:
  1. value units with the transparent model fitted on sales up to T,
  2. market path: short-term index model (Non-Landed) for years 1-2, then the trailing
     10-year nominal income growth (what you would have assumed at T),
  3. age/lease relative performance from the table built on data up to T.
Then compare with the actual resale prices of those units 1-20 years later. Baselines: no
growth (value at T), and valuation x market path without the age/lease adjustment.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .data import fetch_price_index
from .index_model import forecast
from .long_run import build_panel
from .unit_forecast import _rel_lookup
from .unit_model import (AGE_BINS, AGE_LABELS, TransparentModel, relative_performance)


def _market_fn(T: pd.Timestamp):
    """Market growth multiplier as a function of fractional years after T."""
    s = fetch_price_index()["Non-Landed"].dropna()
    tq = pd.Period(T, "Q")
    s = s[s.index <= tq]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = forecast(s, "ensemble", horizons=[4, 8]).table
    l1 = np.log(fc[fc.h == 4].point.iloc[0] / s.iloc[-1])
    l2 = np.log(fc[fc.h == 8].point.iloc[0] / s.iloc[-1])
    li = build_panel("Non-Landed").log_income.dropna()
    li = li[li.index <= tq]
    g = (li.iloc[-1] - li.iloc[-41]) / 10  # trailing 10-yr log income growth per year

    def f(x):
        x = np.asarray(x, dtype=float)
        return np.exp(np.where(x <= 1, x * l1,
                      np.where(x <= 2, l1 + (x - 1) * (l2 - l1), l2 + (x - 2) * g)))
    return f, float(np.exp(l1) - 1), float(np.exp(g) - 1)


def run_origin(d: pd.DataFrame, year: int, max_h: int = 20) -> pd.DataFrame:
    T = pd.Timestamp(f"{year}-12-31")
    hist = d[d.date <= T]
    later = d[(d.date > T) & (d.date <= T + pd.DateOffset(years=max_h, months=6))
              & (d.sale_type == "resale") & d.slug.isin(hist.slug.unique())].copy()
    if later.empty:
        return later
    A = TransparentModel().fit(hist)
    market, g1, g_inc = _market_fn(T)
    try:
        _, tab = relative_performance(hist, A)
    except Exception:
        tab = None

    # The unit as it was at T: same project, stack, floor and size, valued as a resale.
    x = (later.date - T).dt.days.values / 365.25
    yrs_back = (later.date - T).dt.days / 365.25
    at_T = later.copy()
    at_T["age"] = later.age - yrs_back
    at_T["lease_left"] = later.lease_left - yrs_back
    age_bin = pd.cut(at_T.age, AGE_BINS, labels=AGE_LABELS).astype(object).fillna("unknown")
    at_T["ec_age_bin"] = np.where(at_T.is_ec == 1, age_bin, "not EC")
    log_v, _ = A.predict(at_T)

    rel = np.zeros(len(at_T))
    if tab is not None and len(tab):
        cache = {}
        for i, (ten, age, lease, xi) in enumerate(zip(at_T.tenure_type, at_T.age,
                                                       at_T.lease_left, x)):
            tot = 0.0
            if np.isnan(age):  # completion year unknown: no age/lease adjustment
                continue
            for k in range(int(round(xi))):
                key = (ten, int(max(age, 0) + k) // 5, None if np.isnan(lease) else
                       int(lease - k) // 5)
                if key not in cache:
                    try:
                        cache[key] = np.log1p(_rel_lookup(tab, ten, max(age, 0) + k,
                                                          lease - k)[0] / 100)
                    except Exception:
                        cache[key] = 0.0
                tot += cache[key]
            rel[i] = tot
    return pd.DataFrame({
        "origin": year, "years": x, "h": np.clip(np.round(x).astype(int), 1, max_h),
        "slug": later.slug.values, "actual": later.log_psf.values,
        "flat": log_v, "market": log_v + np.log(market(x)),
        "full": log_v + np.log(market(x)) + rel,
        "market_g1": g1, "income_g": g_inc})


def summarise(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, g in bt.groupby("h"):
        r = {"years": h, "sales": len(g), "origins": g.origin.nunique(),
             "first_origin": g.origin.min(), "last_origin": g.origin.max()}
        for m in ("flat", "market", "full"):
            e = np.exp(g[m] - g.actual) - 1  # predicted / actual - 1
            r[f"{m}_median_err"] = e.abs().median() * 100
            if m == "full":
                r["full_within_10"] = (e.abs() < 0.10).mean()
                r["full_within_20"] = (e.abs() < 0.20).mean()
                r["full_bias"] = e.median() * 100
                # range of actual/predicted that held 80% of outcomes
                ratio = np.exp(g.actual - g[m])
                r["range80_lo"], r["range80_hi"] = (np.quantile(ratio, [0.1, 0.9]) - 1) * 100
                # same, but each origin weighted equally (big cohorts don't dominate)
                r["full_median_err_by_origin"] = g.assign(a=e.abs()).groupby("origin") \
                    .a.median().median() * 100
        rows.append(r)
    return pd.DataFrame(rows).set_index("years")
