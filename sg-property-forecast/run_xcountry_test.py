"""Cross-country test: can OECD house-price history improve Singapore's 3-20 year market path?

Current rule: from year 2, prices grow with incomes (price-to-income ratio, PTI, stays flat).
Cross-country model: predict the change in log PTI over h years (h = 1..20) from three
predictors, pooled over OECD countries (quarterly price-to-income ratio, aggregates excluded):
  gap   log PTI minus the country's own trailing mean (at least 10 years of history)
  mom3  change in log PTI over the past 3 years
  mom1  change in log PTI over the past year
For each Singapore start year T, training uses only country-quarters whose h-year outcome was
known by T. Applied to Singapore's own PTI (URA Non-Landed index / income per resident):
  path(x) = current path(x) x exp(pred(x) - pred(2))   for x > 2 years; current path otherwise.
Adoption rule (set before running): better than current at 3, 5 and 10 years and in most
start years there.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sgpf.data import fetch_price_index
from sgpf.long_run import build_panel
from sgpf.unit_backtest import _market_fn

warnings.filterwarnings("ignore")
X = Path("data/xcountry")
if not (X / "oecd_hp.csv").exists():
    raise SystemExit("download OECD data first (see README)")
o = pd.read_csv(X / "oecd_hp.csv", low_memory=False)
o = o[(o.FREQ == "Q") & (o.MEASURE == "HPI_YDH")]
o = o[~o.REF_AREA.str.match(r"^(OECD|EA|EU)")]
o["q"] = pd.PeriodIndex(o.TIME_PERIOD.str.replace("-", ""), freq="Q")
pti = {c: np.log(g.set_index("q").OBS_VALUE.astype(float).sort_index())
       for c, g in o.groupby("REF_AREA")}


def predictors(s: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame(index=s.index)
    f["gap"] = s - s.expanding(40).mean()
    f["mom3"] = s - s.shift(12)
    f["mom1"] = s - s.shift(4)
    return f


rows = []
for c, s in pti.items():
    f = predictors(s)
    for h in range(1, 21):
        tgt = s.shift(-4 * h) - s
        d = f.assign(y=tgt, h=h, country=c, end=[t + 4 * h for t in s.index]).dropna()
        rows.append(d.reset_index(names="q"))
P = pd.concat(rows)
print(f"OECD panel: {P.country.nunique()} countries, {len(P):,} country-quarter-horizon rows")

# Singapore PTI from the URA Non-Landed index and income per resident.
s = fetch_price_index()["Non-Landed"].dropna()
li = build_panel("Non-Landed").log_income
sg = (np.log(s) - li.reindex(s.index)).dropna()
sgf = predictors(sg)

bt = pd.read_pickle("data/propertynoob/unit_backtest.pkl")
coefs, preds, idx_rows = [], {}, []
for y in sorted(bt.origin.unique()):
    T = pd.Period(f"{y}Q4", "Q")
    if T not in sgf.index or sgf.loc[T].isna().any():
        continue
    xT = sgf.loc[T]
    pr = np.zeros(21)
    for h in range(1, 21):
        tr = P[(P.h == h) & (P.end <= T)]
        if len(tr) < 200:
            pr[h] = np.nan
            continue
        A = np.c_[np.ones(len(tr)), tr.gap, tr.mom3, tr.mom1]
        b = np.linalg.lstsq(A, tr.y, rcond=None)[0]
        pr[h] = b @ np.r_[1, xT.gap, xT.mom3, xT.mom1]
        coefs.append({"origin": y, "h": h, "n": len(tr), "countries": tr.country.nunique(),
                      "intercept": b[0], "gap": b[1], "mom3": b[2], "mom1": b[3]})
    if np.isnan(pr[2:]).any():
        continue
    cur, _, _ = _market_fn(pd.Timestamp(f"{y}-12-31"))
    preds[y] = (pr, cur)
    for h in (3, 5, 10):
        if T + 4 * h <= s.index[-1]:
            act = np.log(s[T + 4 * h] / s[T])
            idx_rows.append({"origin": y, "years": h, "actual": act,
                             "current": float(np.log(cur(h))),
                             "xcountry": float(np.log(cur(h))) + pr[h] - pr[2]})
    print(f"{y}: SG PTI gap {xT.gap:+.2f}, mom3 {xT.mom3:+.2f} -> PTI change yrs 2-10 "
          f"{np.expm1(pr[10] - pr[2]):+.0%} | 10y market: current {cur(10) - 1:+.0%}, "
          f"cross-country {cur(10) * np.exp(pr[10] - pr[2]) - 1:+.0%}", flush=True)

g = bt[bt.origin.isin(preds)].copy()
x = g.years.values
adj = np.zeros(len(g)); lcur = np.zeros(len(g))
for y, (pr, cur) in preds.items():
    k = (g.origin == y).values
    lcur[k] = np.log(cur(x[k]))
    adj[k] = np.where(x[k] > 2, np.interp(x[k], np.arange(21), pr) - pr[2], 0.0)
g["current"] = g.flat_gbm + lcur
g["xcountry"] = g.current + adj
err = lambda col, d: (np.abs(np.exp(d[col] - d.actual) - 1)).median() * 100
res = []
for h, d in g.groupby("h"):
    a = (np.abs(np.exp(d.current - d.actual) - 1)).groupby(d.origin).median()
    b = (np.abs(np.exp(d.xcountry - d.actual) - 1)).groupby(d.origin).median()
    res.append({"years": h, "start_years": d.origin.nunique(), "current": err("current", d),
                "xcountry": err("xcountry", d), "better_in": f"{int((b < a).sum())}/{len(a)}"})
res = pd.DataFrame(res).set_index("years")
res.round(2).to_csv("outputs/xcountry_unit_test.csv")
pd.DataFrame(coefs).round(4).to_csv("outputs/xcountry_coefficients.csv", index=False)
ix = pd.DataFrame(idx_rows)
rm = lambda c, d: np.sqrt(((np.expm1(d[c]) - np.expm1(d.actual)) ** 2).mean()) * 100
ixs = ix.groupby("years").apply(lambda d: pd.Series(
    {"start_years": len(d), "current": rm("current", d), "xcountry": rm("xcountry", d)}))
ixs.round(2).to_csv("outputs/xcountry_index_test.csv")
c = pd.DataFrame(coefs)
print("\nLatest coefficients (log PTI change over h years):")
print(c[(c.origin == c.origin.max()) & c.h.isin([3, 5, 10, 20])].round(3).to_string(index=False))
print("\nIndex growth forecast vs actual (RMSE, % points):"); print(ixs.round(1).to_string())
print("\nUnit backtest (typical error, % of actual price):")
print(res.loc[[h for h in (1, 2, 3, 5, 7, 10, 15, 20) if h in res.index]].round(1).to_string())
