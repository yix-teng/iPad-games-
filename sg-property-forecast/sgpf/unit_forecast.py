"""Forecast one condo unit: today's value x market path, with ranges from a backtest.

    value in N years = value today                 (unit valuation model)
                     x market growth to year N     (URA Non-Landed index: average of the
                                                    short-term model for years 1-2 then base
                                                    income growth, and TimesFM 3.0)

The 80% range at each horizon is how far actual resale prices landed from forecasts made
this way in the 1997-2025 backtest with LightGBM valuations refitted at each start year
(outputs/unit_backtest_by_horizon.csv). It covers valuation and market error together.

An age/lease adjustment (how condos of the unit's age and lease kept up with their region)
was tested and made the backtest worse at every horizon from 3 years, so it is not applied;
the table is still produced by run_unit_model.py for information.
"""
from __future__ import annotations

import pickle
from pathlib import Path

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
HORIZONS = (1, 2, 3, 5, 10, 15, 20)
BACKTEST = Path(__file__).resolve().parent.parent / "outputs" / "unit_backtest_by_horizon.csv"


def market_path(series: str = "Non-Landed", horizons=HORIZONS) -> pd.DataFrame:
    """Market growth multipliers per horizon: the average (geometric mean) of
    * current path: short-term index model for years 1-2, then base-case income growth, and
    * TimesFM 3.0's forecast of the index (sgpf.market_timesfm).
    Also returns both components and the CPI assumption."""
    from .market_timesfm import timesfm_growth, timesfm_log_path
    cache = DATA_DIR / f"market_path_{series.replace(' ', '_')}.csv"
    s = fetch_price_index()[series].dropna()
    p0, t0 = s.iloc[-1], s.index[-1]
    if cache.exists():
        mp = pd.read_csv(cache)
        if mp.attrs_quarter.iloc[0] == str(t0) and "timesfm" in mp and \
                set(mp.years) >= set(horizons):
            return mp[mp.years.isin(horizons)].reset_index(drop=True)
    bt = backtest(s, horizons=[4, 8], models=["ensemble"])
    fc = forecast(s, "ensemble", horizons=[4, 8], bt=bt).table
    l1 = np.log(fc[fc.h == 4].point.iloc[0] / p0)
    l2 = np.log(fc[fc.h == 8].point.iloc[0] / p0)
    a = scenario_anchors(series=series)
    g = np.log((1 + a["base"]) * (1 + a["inflation"]))

    def current(x):
        x = np.asarray(x, dtype=float)
        return np.exp(np.where(x <= 1, x * l1,
                      np.where(x <= 2, l1 + (x - 1) * (l2 - l1), l2 + (x - 2) * g)))

    path = timesfm_log_path(t0, series)
    rows = []
    for h in sorted(set(horizons) | set(range(1, 21))):
        c, tf = float(current(h)), float(timesfm_growth(path, h))
        rows.append({"years": h, "central": np.sqrt(c * tf), "current": c, "timesfm": tf,
                     "source": "average of current path and TimesFM 3.0"})
    mp = pd.DataFrame(rows)
    mp["inflation_pa"] = a["inflation"]
    mp["attrs_quarter"] = str(t0)
    mp.to_csv(cache, index=False)
    return mp[mp.years.isin(horizons)].reset_index(drop=True)


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
    row["slug_stack"] = f"{slug}|{stack}"  # this unit's stack, not the latest sale's
    row["floor_bin"] = pd.cut([floor], FLOOR_BINS, labels=FLOOR_LABELS)[0]
    row["area_bin"] = pd.cut([area_sqft], AREA_BINS, labels=AREA_LABELS)[0]
    row["sale_type"] = "resale"
    row["date"], row["month"] = asof, asof.to_period("M")
    row["age"] = row.age + yrs_elapsed
    if row.tenure_type == "leasehold":
        row["lease_left"] = row.lease_left - yrs_elapsed
    return row


OUT = Path(__file__).resolve().parent.parent / "outputs"
MIN_LAUNCHES = 200


BAND_FILES = {"resale": "bands_resale.csv", "launch phase": "bands_launch_phase.csv",
              "brand-new launch": "bands_brand_new_launch.csv",
              "unknown completed project": "bands_unknown_completed.csv"}


def _range_table(mode: str, is_ec: bool) -> pd.DataFrame:
    """Backtest error and bands by horizon for the pricing path used (run_band_calibration.py).
    Columns: err, within_10, lo/hi (plausible, 10th-90th), lo50/hi50 (likely, 25th-75th),
    groups (start years or launches)."""
    f = BAND_FILES[mode]
    if is_ec and mode in ("resale", "launch phase"):
        f = "bands_ec.csv"
    t = pd.read_csv(OUT / f, index_col="years")
    t = t[t.index >= 1]
    if mode == "brand-new launch":
        # Few units resell within 2 years of launch (before completion), so horizons backed by
        # fewer than 200 launches use the nearest horizon that has enough.
        t = t[t.groups >= MIN_LAUNCHES]
    return t.rename(columns={"typical_err": "err", "range80_lo": "lo", "range80_hi": "hi",
                             "range50_lo": "lo50", "range50_hi": "hi50"})


def _today_range(mode: str, acc: dict) -> dict:
    """Typical error and bands (in %) for today's price, with their source."""
    if mode == "brand-new launch":
        r = pd.read_csv(OUT / "launch_pricing_backtest.csv", index_col=0).loc[
            "all launches 2000-2026"]
        return {"err": r.typical_err, "lo": r.range80_lo, "hi": r.range80_hi,
                "lo50": r.range50_lo, "hi50": r.range50_hi,
                "source": f"{int(r.launches):,} launches since 2000"}
    if mode == "unknown completed project":
        r = pd.read_csv(OUT / BAND_FILES[mode], index_col="years").loc[0]
        return {"err": r.typical_err, "lo": r.range80_lo, "hi": r.range80_hi,
                "lo50": r.range50_lo, "hi50": r.range50_hi,
                "source": f"{int(r.sales):,} resales of {int(r.groups):,} projects the model "
                          "had not seen (2008, 2013, 2018)"}
    v = pd.read_csv(OUT / "unit_valuation_by_segment.csv")
    if mode == "launch phase":
        r = v[(v.segment == "sale_type") & (v.group == "new")].iloc[0]
        m, src = "transparent", "new sales, Apr-Sep 2026 test"
    else:
        r = v[(v.segment == "sale_type") & (v.group == "resale")].iloc[0]
        m, src = "lightgbm", f"resales, {acc['test_from']}..{acc['test_to']} test"
    return {"err": r[f"{m}_err"], "lo": r[f"{m}_range80_lo"], "hi": r[f"{m}_range80_hi"],
            "lo50": r[f"{m}_range50_lo"], "hi50": r[f"{m}_range50_hi"],
            "source": f"{int(r.sales):,} {src}"}


def ec_shift(A, age_now: float, years: float) -> float:
    """Change in the EC age effect (log) between now and `years` ahead: captures the jumps at
    the 5-year minimum occupation period and 10-year full privatisation."""
    from .unit_model import AGE_BINS, AGE_LABELS
    b = pd.cut([age_now, age_now + years], AGE_BINS, labels=AGE_LABELS).astype(object)
    return A.effect("ec_age", b[1]) - A.effect("ec_age", b[0])


def new_project_row(d, A, unit: str, area_sqft: float, postal: str, tenure: str,
                    is_ec: bool, top_year: int, district: str | None = None):
    """Feature row for a unit in a project with no transactions yet."""
    from .geo import geocode_postals, haversine_km
    from .unit_model import region_of
    g = geocode_postals([postal])
    if g.empty:
        raise ValueError(f"postal code {postal} not found on OneMap")
    lat, lon = float(g.lat.iloc[0]), float(g.lon.iloc[0])
    if district is None:  # district of the nearest known project
        p = d.groupby("slug").agg(lat=("lat", "first"), lon=("lon", "first"),
                                  district=("district", "first")).dropna()
        district = p.district.iloc[int(np.argmin(haversine_km(lat, lon, p.lat.values,
                                                              p.lon.values)))]
    asof = d.date.max()
    age = asof.year + (asof.dayofyear - 1) / 365.25 - top_year
    floor = parse_floor(unit)
    from .unit_model import AGE_BINS, AGE_LABELS
    age_bin = pd.cut([age], AGE_BINS, labels=AGE_LABELS)[0]
    return pd.Series({
        "slug": None, "unit": unit, "floor": floor, "area_sqft": area_sqft,
        "lat": lat, "lon": lon, "district": district, "region": region_of(district),
        "tenure_type": tenure, "is_ec": int(is_ec), "top_year": top_year, "age": age,
        "lease_left": 99 - max(age, 0) if tenure == "leasehold" else np.nan,
        "floor_bin": pd.cut([floor], FLOOR_BINS, labels=FLOOR_LABELS)[0],
        "area_bin": pd.cut([area_sqft], AREA_BINS, labels=AREA_LABELS)[0],
        "ec_age_bin": age_bin if is_ec else "not EC", "sale_type": "new",
        "date": asof, "month": asof.to_period("M"), "stack": unit.split("-")[-1]})


def forecast_unit(slug: str | None, unit: str, area_sqft: float | None = None,
                  horizons=HORIZONS, new_project: dict | None = None) -> dict:
    """Forecast a unit. Four pricing paths, chosen automatically:

    * brand-new launch  project has no transactions and is not yet completed (pass
                        `new_project` with postal, tenure, is_ec, top_year and optionally
                        district): priced from comparable launches within 3 km in the last
                        12 months (sgpf.launch).
    * unknown completed project  no transactions in the data but already completed
                        (top_year in the past): LightGBM on location and building features.
    * launch phase      project is selling new units and has no resales yet: today's price
                        is the transparent model's new-sale price (its own recent launch
                        prices); forecasts use the LightGBM resale value.
    * resale            everything else: LightGBM resale value.
    ECs additionally get the EC age adjustment in forecasts.
    """
    from .launch import base_log_psf, comparables, effects
    art = pickle.loads(ARTIFACTS.read_bytes())
    A, B, d, acc = art["A"], art["B"], art["data"], art["accuracy"]
    comps, explain_A, explain_B = None, None, None
    if slug is None or slug not in set(d.slug):
        if not new_project:
            raise ValueError(f"'{slug}' has no transactions: pass new_project details "
                             "(postal, tenure, is_ec, top_year) to price it from launches")
        if area_sqft is None:
            raise ValueError("area_sqft is required for a project with no transactions")
        row = new_project_row(d, A, unit, area_sqft, **new_project)
    if (slug is None or slug not in set(d.slug)) and row.age >= 0:
        # Completed project the data has never seen: LightGBM on location and building
        # features (tested on unseen projects in run_band_calibration.py).
        from .geo import location_features
        mode = "unknown completed project"
        loc = location_features(pd.DataFrame({"lat": [row.lat], "lon": [row.lon]}))
        row["mrt_km"], row["cbd_km"] = loc.mrt_km.iloc[0], loc.cbd_km.iloc[0]
        row["sale_type"], row["slug"] = "resale", "__unseen__"
        value = float(np.exp(predict_gbm(B, pd.DataFrame([row]))[0])) * area_sqft
        price_now, psf = value, value / area_sqft
        explain_B = explain_gbm(B, row)
    elif slug is None or slug not in set(d.slug):
        mode = "brand-new launch"
        comps = comparables(A, d, row.lat, row.lon, row.district, row.tenure_type,
                            row.is_ec, d.date.max() + pd.Timedelta(days=1))
        if comps.empty:
            raise ValueError("no comparable launches in the last 12 months nearby")
        base = base_log_psf(comps)
        frame = pd.DataFrame([row])
        price_now = float(np.exp(base + effects(A, frame)[0])) * area_sqft
        resale_row = row.copy(); resale_row["sale_type"] = "resale"
        value = float(np.exp(base + effects(A, pd.DataFrame([resale_row]))[0])) * area_sqft
        psf = price_now / area_sqft
    if slug is not None and slug in set(d.slug):
        row = unit_row(d, slug, unit, area_sqft)
        proj = d[d.slug == slug]
        recent_new = proj[(proj.sale_type == "new")
                          & (proj.date > d.date.max() - pd.DateOffset(months=12))]
        mode = ("launch phase" if (proj.sale_type == "resale").sum() == 0 and len(recent_new)
                else "resale")
        frame = pd.DataFrame([row])
        log_a, _ = A.predict(frame)
        frame = frame.assign(pred_A=log_a, pred_B=predict_gbm(B, frame))
        value = float(np.exp(frame.pred_B.iloc[0])) * row.area_sqft  # resale value (LightGBM)
        if mode == "launch phase":
            new_frame = frame.assign(sale_type="new")
            psf = float(np.exp(A.predict(new_frame)[0][0]))
            price_now = psf * row.area_sqft
        else:
            price_now, psf = value, value / row.area_sqft
        explain_A, explain_B = A.explain(row), explain_gbm(B, row)
    is_ec = bool(row.is_ec)
    now = _today_range(mode, acc)
    rt = _range_table(mode, is_ec)
    mp = market_path(horizons=horizons)
    out = []
    for _, m in mp.iterrows():
        h = int(m.years)
        adj = ec_shift(A, row.age, h) if is_ec else 0.0
        central = value * m.central * np.exp(adj)
        b = rt.loc[rt.index[np.argmin(np.abs(rt.index - h))]]  # nearest backtested horizon
        defl = (1 + m.inflation_pa) ** h
        out.append({"years": h, "market_growth_pct": (m.central - 1) * 100,
                    "current_path_pct": (m.current - 1) * 100,
                    "timesfm_pct": (m.timesfm - 1) * 100,
                    "ec_adjust_pct": (np.exp(adj) - 1) * 100, "value": central,
                    "low": central * (1 + b.lo / 100), "high": central * (1 + b.hi / 100),
                    "low50": central * (1 + b.lo50 / 100),
                    "high50": central * (1 + b.hi50 / 100),
                    "value_todays_dollars": central / defl,
                    "backtest_err_pct": b.err, "backtest_within_10pct": b.within_10,
                    "backtest_groups": int(b.groups), "backtest_years": int(b.name)})
    return {"mode": mode, "row": row, "psf": psf, "price_now": price_now,
            "resale_value_now": value, "now_err": now["err"],
            "price_now_low": price_now * (1 + now["lo"] / 100),
            "price_now_high": price_now * (1 + now["hi"] / 100),
            "price_now_low50": price_now * (1 + now["lo50"] / 100),
            "price_now_high50": price_now * (1 + now["hi50"] / 100),
            "now_source": now["source"],
            "comparables": comps, "explain_A": explain_A, "explain_B": explain_B,
            "forecast": pd.DataFrame(out), "accuracy": acc}
