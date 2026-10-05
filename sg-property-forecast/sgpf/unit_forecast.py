"""Forecast one condo unit: today's value x market path x how units like it kept up.

    value in N years = value today                    (transparent unit model, recent sales)
                     x market growth to year N        (URA Non-Landed index: short-term model
                                                       for 1-2 yrs, income scenarios for 3-10)
                     x relative performance           (how condos of this age / lease kept up
                                                       with their region, per year)

Uncertainty has three independent sources, combined in log terms (root-sum-of-squares):
valuation error today, market range, and how far individual projects stray from their
age/lease group.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from .data import DATA_DIR, fetch_price_index
from .index_model import backtest, forecast
from .long_run import scenario_anchors, scenarios
from .unit_model import (AREA_BINS, AREA_LABELS, FLOOR_BINS, FLOOR_LABELS, REL_AGE_BINS,
                         REL_AGE_LABELS, REL_LEASE_BINS, REL_LEASE_LABELS, RULES, explain_gbm,
                         parse_floor, predict_gbm)

ARTIFACTS = DATA_DIR / "propertynoob" / "unit_model.pkl"
Z80 = 1.2816  # 80% two-sided normal multiplier
HORIZONS = (1, 2, 3, 5, 10)


def market_path(series: str = "Non-Landed", horizons=HORIZONS) -> pd.DataFrame:
    """Market growth multipliers (central and 80% band) per horizon, plus CPI deflator."""
    cache = DATA_DIR / f"market_path_{series.replace(' ', '_')}.csv"
    s = fetch_price_index()[series].dropna()
    if cache.exists():
        mp = pd.read_csv(cache)
        if mp.attrs_quarter.iloc[0] == str(s.index[-1]):
            return mp
    rows, p0 = [], s.iloc[-1]
    # Years 1-2: short-term index model. Later years continue from its year-2 value at the
    # base-case income growth, so the path has no jump where the sources hand over; the
    # range at each horizon is the long-run model's historical range for that horizon.
    bt = backtest(s, horizons=[4, 8], models=["ensemble"])
    fc = forecast(s, "ensemble", horizons=[4, 8], bt=bt).table
    short = {h: fc[fc.h == 4 * h].iloc[0] for h in (1, 2)}
    a = scenario_anchors(series=series)
    g_nom = (1 + a["base"]) * (1 + a["inflation"]) - 1
    sc = scenarios({"base": a["base"]}, a["inflation"], years=max(horizons), series=series)
    for h in horizons:
        if h <= 2:
            r = short[h]
            rows.append({"years": h, "source": "short-term index model",
                         "central": r.point / p0, "lo": r.lo80 / p0, "hi": r.hi80 / p0})
        else:
            c = short[2].point / p0 * (1 + g_nom) ** (h - 2)
            r = sc[sc.years == h].iloc[0]
            rows.append({"years": h, "source": "short-term to year 2, then base income growth",
                         "central": c, "lo": c * r.likely_lo / r.central,
                         "hi": c * r.likely_hi / r.central})
    mp = pd.DataFrame(rows)
    mp["inflation_pa"] = a["inflation"]
    mp["attrs_quarter"] = str(s.index[-1])
    mp.to_csv(cache, index=False)
    return mp


def _rel_lookup(tab: pd.DataFrame, tenure: str, age: float, lease_left: float,
                min_projects: int = 5) -> tuple[float, float, str]:
    """(mean % per year, std % per year, which group was used) for a unit's age/lease."""
    age_bin = pd.cut([max(age, 0)], REL_AGE_BINS, labels=REL_AGE_LABELS, right=False)[0]
    if tenure == "leasehold":
        lease_bin = pd.cut([lease_left], REL_LEASE_BINS, labels=REL_LEASE_LABELS)[0]
    else:
        lease_bin = "freehold"
    key = (tenure, age_bin, lease_bin)
    if key in tab.index and tab.loc[key, "projects"] >= min_projects:
        r = tab.loc[key]
        return float(r["mean"]), float(r["std"]), f"{tenure}, age {age_bin}, lease {lease_bin}"
    pooled = tab.xs(tenure, level="tenure")
    pooled = pooled[pooled.index.get_level_values("age_bin") == age_bin]
    if pooled.projects.sum() >= min_projects:
        w = pooled["count"]
        return (float((pooled["mean"] * w).sum() / w.sum()),
                float((pooled["std"] * w).sum() / w.sum()), f"{tenure}, age {age_bin}")
    allt = tab.xs(tenure, level="tenure")
    w = allt["count"]
    return (float((allt["mean"] * w).sum() / w.sum()),
            float((allt["std"] * w).sum() / w.sum()), f"{tenure} (all ages)")


def unit_row(d: pd.DataFrame, slug: str, unit: str, area_sqft: float | None = None):
    """Build a feature row for a unit, using the project's attributes from its latest sale."""
    proj = d[d.slug == slug]
    if proj.empty:
        raise ValueError(f"no transactions for project '{slug}'")
    row = proj.sort_values("date").iloc[-1].copy()
    floor = parse_floor(unit)
    stack = unit.split("-")[-1] if "-" in unit else None
    if area_sqft is None:  # same stack usually means the same layout
        same = proj[proj["stack"] == stack]
        area_sqft = float((same if len(same) else proj).area_sqft.median())
    asof = d.date.max()
    yrs_elapsed = (asof - row.date).days / 365.25
    row["unit"], row["floor"], row["stack"], row["area_sqft"] = unit, floor, stack, area_sqft
    row["floor_bin"] = pd.cut([floor], FLOOR_BINS, labels=FLOOR_LABELS)[0]
    row["area_bin"] = pd.cut([area_sqft], AREA_BINS, labels=AREA_LABELS)[0]
    row["sale_type"] = "resale"
    row["date"], row["month"] = asof, asof.to_period("M")
    row["age"] = row.age + yrs_elapsed
    if row.tenure_type == "leasehold":
        row["lease_left"] = row.lease_left - yrs_elapsed
    return row


def forecast_unit(slug: str, unit: str, area_sqft: float | None = None,
                  horizons=HORIZONS) -> dict:
    art = pickle.loads(ARTIFACTS.read_bytes())
    A, B, d, tab, acc = art["A"], art["B"], art["data"], art["rel_table"], art["accuracy"]
    row = unit_row(d, slug, unit, area_sqft)
    frame = pd.DataFrame([row])
    log_a, _ = A.predict(frame)
    frame = frame.assign(pred_A=log_a, pred_B=predict_gbm(B, frame))
    psf_a, psf_b = float(np.exp(frame.pred_A.iloc[0])), float(np.exp(frame.pred_B.iloc[0]))
    rule = acc["chosen_rule"]
    psf = float(np.exp(np.asarray(RULES[rule](frame))[0]))
    value = psf * row.area_sqft
    # 10th/90th percentile of log(actual / predicted) for resales in the test window.
    v_lo, v_hi = acc["chosen_resale_log_resid_q10"], acc["chosen_resale_log_resid_q90"]
    mp = market_path(horizons=horizons)
    out = []
    for _, m in mp.iterrows():
        h = int(m.years)
        rel_log, rel_var, groups = 0.0, 0.0, []
        for k in range(h):
            mean, std, grp = _rel_lookup(tab, row.tenure_type, row.age + k,
                                         row.lease_left - k if row.tenure_type == "leasehold"
                                         else np.nan)
            rel_log += np.log1p(mean / 100)
            # `std` is the spread of per-year averages over 5-year spans, so one project's
            # 5-year deviation has sd 5*std; spread evenly that is 5*std^2 variance per year.
            rel_var += 5 * (std / 100) ** 2
            groups.append(grp)
        rel_sd = np.sqrt(rel_var)
        central = value * m.central * np.exp(rel_log)
        lo_w = np.sqrt(v_lo ** 2 + np.log(m.lo / m.central) ** 2 + (Z80 * rel_sd) ** 2)
        hi_w = np.sqrt(v_hi ** 2 + np.log(m.hi / m.central) ** 2 + (Z80 * rel_sd) ** 2)
        defl = (1 + m.inflation_pa) ** h
        out.append({"years": h, "market_source": m.source,
                    "market_growth_pct": (m.central - 1) * 100,
                    "relative_pct": (np.exp(rel_log) - 1) * 100,
                    "value": central, "low": central * np.exp(-lo_w),
                    "high": central * np.exp(hi_w), "value_todays_dollars": central / defl,
                    "low_todays_dollars": central * np.exp(-lo_w) / defl,
                    "high_todays_dollars": central * np.exp(hi_w) / defl,
                    "age_lease_group": groups[-1]})
    return {"row": row, "psf": psf, "psf_A": psf_a, "psf_B": psf_b, "rule": rule,
            "value_now": value, "value_now_low": value * np.exp(v_lo),
            "value_now_high": value * np.exp(v_hi), "explain_A": A.explain(row),
            "explain_B": explain_gbm(B, row), "forecast": pd.DataFrame(out), "accuracy": acc}
