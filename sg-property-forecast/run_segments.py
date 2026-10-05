"""Backtest accuracy by segment: tenure, new launch vs completed, trading history, region, EC.

Segments are defined as at each start year (what was known then):
* stage            "new launch / under construction" if the project had not reached TOP by T
* tenure           freehold (incl. 999-yr) vs leasehold
* resale_history   number of resales in the project up to T (0 / 1-9 / 10-49 / 50+)
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sgpf.unit_forecast import ARTIFACTS

out = Path("outputs")
art = pickle.loads(ARTIFACTS.read_bytes())
d = art["data"]
bt = pd.read_pickle("data/propertynoob/unit_backtest.pkl")
M = "market_gbm"

proj = d.groupby("slug").agg(tenure=("tenure_type", "first"), top_year=("top_year", "first"),
                             region=("region", "first"), is_ec=("is_ec", "max"))
bt = bt.join(proj, on="slug")
# Resales per project up to each start year.
rs = d[d.sale_type == "resale"].assign(y=d.date.dt.year).groupby(["slug", "y"]).size()
rs = rs.groupby(level=0).cumsum().rename("n").reset_index()
hist = {}
for o in bt.origin.unique():
    hist[o] = rs[rs.y <= o].groupby("slug").n.last()
bt["resales_before"] = [hist[o].get(s, 0) for o, s in zip(bt.origin, bt.slug)]
bt["resale_history"] = pd.cut(bt.resales_before, [-1, 0, 9, 49, 1e9],
                              labels=["0 resales", "1-9", "10-49", "50+"]).astype(str)
bt["stage"] = np.where(bt.top_year > bt.origin, "new launch / under construction",
                       np.where(bt.top_year.notna(), "completed", "unknown TOP"))
bt["type"] = np.where(bt.is_ec == 1, "EC", "condo/apartment")

def stats(g):
    e = np.exp(g[M] - g.actual) - 1
    ratio = np.exp(g.actual - g[M])
    lo, hi = (np.quantile(ratio, [0.1, 0.9]) - 1) * 100
    return pd.Series({"sales": len(g), "start_years": g.origin.nunique(),
                      "typical_err": e.abs().median() * 100, "within_10": (e.abs() < .1).mean(),
                      "range80_lo": lo, "range80_hi": hi, "bias": e.median() * 100})

rows = []
for seg in ("stage", "tenure", "resale_history", "region", "type"):
    for h in (1, 3, 5, 10):
        for k, g in bt[bt.h == h].groupby(seg):
            if len(g) >= 500:
                rows.append({"segment": seg, "group": k, "years": h, **stats(g)})
res = pd.DataFrame(rows)
res.round(2).to_csv(out / "unit_backtest_by_segment.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(res.round(1).to_string(index=False))
