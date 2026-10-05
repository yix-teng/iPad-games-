"""Recompute every unit-forecast range table with the adopted market path: the average of the
current path and TimesFM 3.0 (sgpf.market_timesfm). The income-rule tables are kept as
*_income_rule.csv for reference.

Needs run_unit_backtest.py and run_ec_launch_backtest.py outputs (unit-level pickles in
data/propertynoob/) and TimesFM installed. EC tables are re-estimated per start year (~20 min).
"""
import pickle
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sgpf.market_timesfm import averaged, timesfm_log_path
from sgpf.unit_backtest import _market_fn, summarise
from sgpf.unit_forecast import ARTIFACTS
from sgpf.unit_model import AGE_BINS, AGE_LABELS, TransparentModel

warnings.filterwarnings("ignore")
OUT = Path("outputs")
d = pickle.loads(ARTIFACTS.read_bytes())["data"]
_paths = {}


def avg_fn(q: pd.Period):
    """Averaged market growth function as known at quarter q."""
    if q not in _paths:
        cur, _, _ = _market_fn(q.end_time.normalize())
        _paths[q] = (cur, averaged(cur, timesfm_log_path(q)))
    return _paths[q]


def stats(g, col, label, group_col):
    rows = []
    for h, x in g.groupby("h"):
        e = np.exp(x[col] - x.actual) - 1
        lo, hi = (np.quantile(np.exp(x.actual - x[col]), [0.1, 0.9]) - 1) * 100
        rows.append({"years": h, "method": label, "sales": len(x),
                     "groups": x[group_col].nunique(), "typical_err": e.abs().median() * 100,
                     "within_10": (e.abs() < .1).mean(), "within_20": (e.abs() < .2).mean(),
                     "range80_lo": lo, "range80_hi": hi, "bias": e.median() * 100})
    return pd.DataFrame(rows)


# 1. Resale and launch-phase tables (unit backtest, start years 1997-2025)
bt = pd.read_pickle("data/propertynoob/unit_backtest.pkl")
bt["market_avg"] = np.nan
for y in sorted(bt.origin.unique()):
    k = (bt.origin == y).values
    _, avg = avg_fn(pd.Period(f"{y}Q4", "Q"))
    bt.loc[k, "market_avg"] = bt.flat_gbm.values[k] + np.log(avg(bt.years.values[k]))
for f in ("unit_backtest_by_horizon.csv", "unit_backtest_launch_phase.csv"):
    ref = OUT / f.replace(".csv", "_income_rule.csv")
    if not ref.exists():
        shutil.copy(OUT / f, ref)
summarise(bt, "market_avg").round(2).to_csv(OUT / "unit_backtest_by_horizon.csv")
top = d.groupby("slug").agg(top_year=("top_year", "first"), is_ec=("is_ec", "max"))
bt = bt.join(top, on="slug")
lp = bt[(bt.top_year > bt.origin) & (bt.is_ec == 0)]
summarise(lp, "market_avg").round(2).to_csv(OUT / "unit_backtest_launch_phase.csv")
print("resale and launch-phase tables done", flush=True)

# 2. EC table (EC age adjustment re-estimated at each start year)
ec = bt[bt.is_ec == 1].copy()
ec_adj = np.zeros(len(ec))
for o in sorted(ec.origin.unique()):
    m = (ec.origin == o).values
    hist = d[d.date <= pd.Timestamp(f"{o}-12-31")]
    if hist.is_ec.sum() < 200:
        continue
    Ao = TransparentModel().fit(hist)
    age0 = (o + 1) - ec.top_year.values[m]
    b0 = pd.cut(age0, AGE_BINS, labels=AGE_LABELS).astype(object)
    b1 = pd.cut(age0 + ec.years.values[m], AGE_BINS, labels=AGE_LABELS).astype(object)
    f = np.vectorize(lambda l: Ao.effect("ec_age", l))
    ec_adj[m] = f(b1) - f(b0)
    print(f"  EC table {o}", flush=True)
ec["ec_avg"] = ec.market_avg + np.nan_to_num(ec_adj)
ect = pd.read_csv(OUT / "ec_backtest.csv")
ect = ect[ect.method != "EC resale, with EC adjustment, TimesFM average"]
ect = pd.concat([ect, stats(ec, "ec_avg", "EC resale, with EC adjustment, TimesFM average",
                            "origin")])
ect.round(2).to_csv(OUT / "ec_backtest.csv", index=False)

# 3. Brand-new launch table
lb = pd.read_pickle("data/propertynoob/launch_backtest.pkl")
launch = d[d.sale_type == "new"].groupby("slug").date.min()
lb["launch_avg"] = np.nan
for slug, g in lb.groupby("group"):
    q = pd.Period(launch[slug], "Q") - 1
    cur, avg = avg_fn(q)
    k = (lb.group == slug).values
    lb.loc[k, "launch_avg"] = (lb.launch_plain.values[k] - np.log(cur(lb.x.values[k]))
                               + np.log(avg(lb.x.values[k])))
lt = pd.read_csv(OUT / "launch_forecast_backtest.csv")
lt = lt[lt.method != "launch, condo/apartment, TimesFM average"]
lt = pd.concat([lt, stats(lb[lb.is_ec == 0], "launch_avg",
                          "launch, condo/apartment, TimesFM average", "group")])
lt.round(2).to_csv(OUT / "launch_forecast_backtest.csv", index=False)

# 4. Before/after summary
old_r = pd.read_csv(OUT / "unit_backtest_by_horizon_income_rule.csv", index_col="years")
new_r = pd.read_csv(OUT / "unit_backtest_by_horizon.csv", index_col="years")
old_l = pd.read_csv(OUT / "unit_backtest_launch_phase_income_rule.csv", index_col="years")
new_l = pd.read_csv(OUT / "unit_backtest_launch_phase.csv", index_col="years")
e_old = ect[ect.method == "EC resale, with EC adjustment"].set_index("years")
e_new = ect[ect.method == "EC resale, with EC adjustment, TimesFM average"].set_index("years")
l_old = lt[(lt.method == "launch, condo/apartment") & (lt.groups >= 200)].set_index("years")
l_new = lt[(lt.method == "launch, condo/apartment, TimesFM average")
           & (lt.groups >= 200)].set_index("years")
rows = []
for h in (1, 2, 3, 5, 7, 10, 15, 20):
    pick = lambda t, c: t.loc[t.index[np.argmin(np.abs(t.index - h))], c]
    rows.append({"years": h,
                 "resale_before": old_r.loc[h, "market_gbm_median_err"],
                 "resale_after": new_r.loc[h, "market_avg_median_err"],
                 "resale_range_after": f"{new_r.loc[h, 'range80_lo']:+.0f}% to {new_r.loc[h, 'range80_hi']:+.0f}%",
                 "launch_phase_before": old_l.loc[h, "market_gbm_median_err"],
                 "launch_phase_after": new_l.loc[h, "market_avg_median_err"],
                 "ec_before": e_old.loc[h, "typical_err"], "ec_after": e_new.loc[h, "typical_err"],
                 "brand_new_before": pick(l_old, "typical_err"),
                 "brand_new_after": pick(l_new, "typical_err")})
s = pd.DataFrame(rows).set_index("years")
s.round(2).to_csv(OUT / "timesfm_adoption_summary.csv")
pd.set_option("display.width", 220)
print(s.round(1).to_string())
