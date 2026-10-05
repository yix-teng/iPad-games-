"""Backtests for (1) new-launch forecasts and (2) the EC age adjustment.

1. Every launch from 2000: priced from comparable launches in the 12 months before launch
   (sgpf.launch), grown with the market path known at the previous quarter end, compared with
   the actual resale prices of its units 1-20 years later. Floor/size/sale-type adjustments
   use the all-data lookup tables (small look-ahead in those tables only).
2. EC adjustment: in the unit backtest, EC forecasts are multiplied by the change in the EC
   age effect between the start year and the sale date, with the EC age table re-estimated at
   each start year from data up to then (no look-ahead).
"""
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sgpf.launch import base_log_psf, comparables, effects
from sgpf.unit_backtest import _market_fn
from sgpf.unit_forecast import ARTIFACTS
from sgpf.unit_model import AGE_BINS, AGE_LABELS, TransparentModel

warnings.filterwarnings("ignore")
out = Path("outputs")
art = pickle.loads(ARTIFACTS.read_bytes())
A, d = art["A"], art["data"]


def stats(g, col, label):
    rows = []
    for h, x in g.groupby("h"):
        e = np.exp(x[col] - x.actual) - 1
        lo, hi = (np.quantile(np.exp(x.actual - x[col]), [0.1, 0.9]) - 1) * 100
        rows.append({"years": h, "method": label, "sales": len(x),
                     "groups": x.group.nunique(), "typical_err": e.abs().median() * 100,
                     "within_10": (e.abs() < .1).mean(), "within_20": (e.abs() < .2).mean(),
                     "range80_lo": lo, "range80_hi": hi, "bias": e.median() * 100})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 1. launches
proj = d.groupby("slug").agg(lat=("lat", "first"), lon=("lon", "first"),
                             district=("district", "first"), tenure=("tenure_type", "first"),
                             is_ec=("is_ec", "max"))
launch = d[d.sale_type == "new"].groupby("slug").date.min()
markets = {}
parts = []
for slug, L in launch[launch >= "2000-01-01"].items():
    p = proj.loc[slug]
    c = comparables(A, d, p.lat, p.lon, p.district, p.tenure, p.is_ec, L, exclude_slug=slug)
    if c.empty:
        continue
    base = base_log_psf(c)
    q = pd.Period(L, "Q") - 1
    if q not in markets:
        markets[q] = _market_fn(q.end_time.normalize())[0]
    T = q.end_time.normalize()
    rs = d[(d.slug == slug) & (d.sale_type == "resale") & (d.date > T)]
    rs = rs[(rs.date - T).dt.days / 365.25 <= 20.5]
    if rs.empty:
        continue
    x = (rs.date - T).dt.days.values / 365.25
    # unit effects with the EC age at launch (no EC adjustment) and at resale (with it)
    at_launch = rs.assign(ec_age_bin=np.where(rs.is_ec == 1, "pre-TOP", "not EC"))
    parts.append(pd.DataFrame({
        "group": slug, "is_ec": p.is_ec, "tenure": p.tenure, "x": x,
        "h": np.clip(np.round(x).astype(int), 1, 20), "actual": rs.log_psf.values,
        "launch_plain": base + effects(A, at_launch) + np.log(markets[q](x)),
        "launch_ec_adj": base + effects(A, rs) + np.log(markets[q](x))}))
lb = pd.concat(parts)
lb.to_pickle("data/propertynoob/launch_backtest.pkl")
res1 = pd.concat([stats(lb[lb.is_ec == 0], "launch_plain", "launch, condo/apartment"),
                  stats(lb[lb.is_ec == 1], "launch_plain", "launch, EC, no EC adjustment"),
                  stats(lb[lb.is_ec == 1], "launch_ec_adj", "launch, EC, with EC adjustment"),
                  stats(lb[lb.tenure == "freehold"], "launch_plain", "launch, freehold"),
                  stats(lb[(lb.tenure == "leasehold") & (lb.is_ec == 0)], "launch_plain",
                        "launch, leasehold condo")])
res1.round(2).to_csv(out / "launch_forecast_backtest.csv", index=False)
print("1. New-launch forecasts: launches", lb.group.nunique(), "resales", len(lb))

# ---------------------------------------------------------------- 2. EC adjustment
bt = pd.read_pickle("data/propertynoob/unit_backtest.pkl")
top = d.groupby("slug").agg(top_year=("top_year", "first"), is_ec=("is_ec", "max"))
bt = bt.join(top, on="slug")
ec = bt[bt.is_ec == 1].copy()
ec["group"] = ec.origin
ec_adj = np.zeros(len(ec))
for o in sorted(ec.origin.unique()):
    m = (ec.origin == o).values
    hist = d[d.date <= pd.Timestamp(f"{o}-12-31")]
    if hist.is_ec.sum() < 200:
        continue
    Ao = TransparentModel().fit(hist)
    age0 = (o + 1) - ec.top_year.values[m]
    age1 = age0 + ec.years.values[m]
    b0 = pd.cut(age0, AGE_BINS, labels=AGE_LABELS).astype(object)
    b1 = pd.cut(age1, AGE_BINS, labels=AGE_LABELS).astype(object)
    f = np.vectorize(lambda l: Ao.effect("ec_age", l))
    ec_adj[m] = f(b1) - f(b0)
    print(f"  EC table as of {o}: " + ", ".join(f"{k} {v:+.1f}%" for k, v in
                                                   Ao.table("ec_age").items()), flush=True)
ec["ec_adjusted"] = ec.market_gbm + np.nan_to_num(ec_adj)
res2 = pd.concat([stats(ec, "market_gbm", "EC resale, current"),
                  stats(ec, "ec_adjusted", "EC resale, with EC adjustment")])
res2.round(2).to_csv(out / "ec_backtest.csv", index=False)

pd.set_option("display.width", 200); pd.set_option("display.max_rows", 300)
for r in (res1, res2):
    print(r[r.years.isin([1, 2, 3, 5, 7, 10, 15, 20])].round(1).to_string(index=False))
