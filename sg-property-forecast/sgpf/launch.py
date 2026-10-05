"""Price units in brand-new launches from comparable recent launches nearby.

    base psf = weighted median of new-sale prices in OTHER projects within 3 km (fallback:
               same district) in the 12 months before `asof`, same tenure and EC status,
               after removing floor, size, EC-age and sale-type effects
    unit psf = base psf x this unit's floor, size, EC-age and sale-type effects

Backtest on 1,551 launches since 2000: 11.4% typical error at launch (run_launch_backtest.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import haversine_km

TERMS = (("floor", "floor_bin"), ("area", "area_bin"), ("ec_age", "ec_age_bin"),
         ("sale_type", "sale_type"))
RADIUS_KM, WINDOW_MONTHS = 3.0, 12


def effects(A, df: pd.DataFrame) -> np.ndarray:
    s = np.zeros(len(df))
    for term, col in TERMS:
        s += df[col].astype(object).map(str).map(lambda l, t=term: A.effect(t, l)).values
    return s


def comparables(A, d: pd.DataFrame, lat, lon, district, tenure, is_ec, asof,
                exclude_slug=None) -> pd.DataFrame:
    """Comparable launches (one row per project) with their adjusted base log psf."""
    w = d[(d.sale_type == "new") & (d.date < asof)
          & (d.date >= asof - pd.DateOffset(months=WINDOW_MONTHS))]
    if exclude_slug is not None:
        w = w[w.slug != exclude_slug]
    w = w[(w.tenure_type == tenure) & (w.is_ec == is_ec)]
    if w.empty:
        return w.iloc[:0]
    w = w.assign(base=w.log_psf.values - effects(A, w))
    c = w.groupby("slug").agg(base=("base", "median"), sales=("base", "size"),
                              lat=("lat", "first"), lon=("lon", "first"),
                              district=("district", "first"))
    c["km"] = haversine_km(lat, lon, c.lat.values, c.lon.values) if pd.notna(lat) else np.nan
    near = c[c.km <= RADIUS_KM].assign(how=f"within {RADIUS_KM:g} km")
    if near.empty:
        near = c[c.district == district].assign(how="same district")
    return near.sort_values("km")


def base_log_psf(comps: pd.DataFrame) -> float:
    return float(np.average(comps.base, weights=np.sqrt(comps.sales)))
