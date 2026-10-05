"""Backtest pricing of brand-new launches from comparable recent launches nearby.

For every project launched 2000 onward: take its new sales in the first 3 months after
launch; predict each from new-sale prices of OTHER projects within 3 km (fallback: same
district) in the 12 months before the launch, adjusted for floor, size, EC age and sale type
(lookup tables from the transparent model fitted on all data: a small look-ahead in those
adjustment tables only). Comparable projects must match on tenure and EC status.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sgpf.geo import haversine_km
from sgpf.unit_forecast import ARTIFACTS

art = pickle.loads(ARTIFACTS.read_bytes())
A, d = art["A"], art["data"]
TERMS = (("floor", "floor_bin"), ("area", "area_bin"), ("ec_age", "ec_age_bin"),
         ("sale_type", "sale_type"))


def effects(df):
    s = np.zeros(len(df))
    for term, col in TERMS:
        s += df[col].astype(object).map(str).map(lambda l, t=term: A.effect(t, l)).values
    return s


new = d[d.sale_type == "new"].copy()
new["base"] = new.log_psf.values - effects(new)
launch = new.groupby("slug").date.min()
proj = d.groupby("slug").agg(lat=("lat", "first"), lon=("lon", "first"),
                             district=("district", "first"), tenure=("tenure_type", "first"),
                             is_ec=("is_ec", "max"), region=("region", "first"))
rows = []
for slug, L in launch[launch >= "2000-01-01"].items():
    p = proj.loc[slug]
    window = new[(new.date < L) & (new.date >= L - pd.DateOffset(months=12))
                 & (new.slug != slug)]
    c = window.groupby("slug").agg(base=("base", "median"), n=("base", "size")).join(proj)
    c = c[(c.tenure == p.tenure) & (c.is_ec == p.is_ec)]
    near = c
    if pd.notna(p.lat):
        near = c[haversine_km(p.lat, p.lon, c.lat.values, c.lon.values) <= 3.0]
    how = "within 3 km"
    if near.empty:
        near, how = c[c.district == p.district], "same district"
    if near.empty:
        continue
    est = np.average(near.base, weights=np.sqrt(near.n))
    first = new[(new.slug == slug) & (new.date < L + pd.DateOffset(months=3))]
    pred = est + effects(first)
    rows.append(pd.DataFrame({"slug": slug, "launch_year": L.year, "how": how,
                              "comparables": len(near), "region": p.region,
                              "tenure": p.tenure, "is_ec": p.is_ec,
                              "err": np.exp(pred - first.log_psf.values) - 1}))
r = pd.concat(rows)

def stats(g):
    e = g.err
    lo, lo50, hi50, hi = (np.quantile(1 / (1 + e), [0.1, 0.25, 0.75, 0.9]) - 1) * 100
    return pd.Series({"launches": g.slug.nunique(), "sales": len(g),
                      "typical_err": e.abs().median() * 100, "within_10": (e.abs() < .1).mean(),
                      "range80": f"{lo:+.0f}% to {hi:+.0f}%", "bias": e.median() * 100,
                      "range80_lo": lo, "range50_lo": lo50, "range50_hi": hi50,
                      "range80_hi": hi})

out = [stats(r).rename("all launches 2000-2026")]
for seg in ("tenure", "region", "is_ec", "how"):
    for k, g in r.groupby(seg):
        out.append(stats(g).rename(f"{seg}={k}"))
r["period"] = pd.cut(r.launch_year, [1999, 2007, 2013, 2019, 2026],
                     labels=["2000-07", "2008-13", "2014-19", "2020-26"]).astype(str)
for k, g in r.groupby("period"):
    out.append(stats(g).rename(f"launched {k}"))
res = pd.DataFrame(out)
res.round(2).to_csv(Path("outputs") / "launch_pricing_backtest.csv")
pd.set_option("display.width", 200)
print(res.round(1).to_string())
