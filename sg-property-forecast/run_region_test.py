"""Can region-specific market paths (CCR/RCR/OCR) be forecast well enough to help?

Region index: quality-adjusted market level by region from the transparent model (all data;
small look-ahead in the index history only). National: URA Non-Landed index.
Relative level rel_r = log(region index) - log(national).
Predictors at start date T (pre-specified): gap = rel_r(T) - its 10-year trailing mean
(reversion), mom = rel_r(T) - rel_r(T - 3 yrs) (momentum).
Walk-forward: for each start year, OLS per horizon on region-quarters whose outcome was known
by T; prediction adds to the current forecast for units in that region.
"""
import pickle

import numpy as np
import pandas as pd

from sgpf.data import fetch_price_index
from sgpf.unit_forecast import ARTIFACTS

art = pickle.loads(ARTIFACTS.read_bytes()); A, d = art["A"], art["data"]
nat = np.log(fetch_price_index()["Non-Landed"].dropna())
rel = {}
for r in ("CCR", "RCR", "OCR"):
    s = np.log(A.market_index(r))
    rel[r] = (s - nat.reindex(s.index)).dropna()
q_all = sorted(set().union(*[s.index for s in rel.values()]))

def feats(r, t):
    s = rel[r][rel[r].index <= t]
    if len(s) < 52:
        return None
    return s.iloc[-1] - s.iloc[-41:-1].mean(), s.iloc[-1] - s.iloc[-13]

samples = []  # (region, t, h, gap, mom, target)
for r, s in rel.items():
    for t in s.index:
        f = feats(r, t)
        if f is None:
            continue
        for h in range(1, 11):
            if t + 4 * h in s.index:
                samples.append((r, t, h, *f, s[t + 4 * h] - s[t]))
S = pd.DataFrame(samples, columns=["region", "t", "h", "gap", "mom", "target"])

bt = pd.read_pickle("data/propertynoob/unit_backtest.pkl")
bt = bt.join(d.groupby("slug").region.first(), on="slug")
bt["hh"] = np.clip(np.round(bt.years).astype(int), 1, 10)
bt["rel_adj"] = np.nan
coefs = []
for o in sorted(bt.origin.unique()):
    T = pd.Period(f"{o}Q4", "Q")
    for h in range(1, 11):
        tr = S[(S.h == h) & (S.t + 4 * h <= T)]
        if len(tr) < 40:
            continue
        X = np.c_[np.ones(len(tr)), tr.gap, tr.mom]
        b = np.linalg.lstsq(X, tr.target, rcond=None)[0]
        coefs.append({"origin": o, "h": h, "intercept": b[0], "gap": b[1], "mom": b[2]})
        for r in rel:
            f = feats(r, T)
            if f is None:
                continue
            pred = b[1] * f[0] + b[2] * f[1]  # no intercept: national path stays as is
            m = (bt.origin == o) & (bt.hh == h) & (bt.region == r)
            bt.loc[m, "rel_adj"] = pred
g = bt.dropna(subset=["rel_adj"])
g = g.assign(regional=g.market_gbm + g.rel_adj)
err = lambda c, x: (np.abs(np.exp(x[c] - x.actual) - 1)).median() * 100
rows = []
for h, x in g.groupby("h"):
    if h > 20:
        continue
    better = sum(err("regional", y) < err("market_gbm", y) for _, y in x.groupby("origin"))
    rows.append({"years": h, "start_years": x.origin.nunique(), "current": err("market_gbm", x),
                 "regional_path": err("regional", x),
                 "better_in": f"{better}/{x.origin.nunique()}"})
res = pd.DataFrame(rows).set_index("years")
res.round(2).to_csv("outputs/region_path_test.csv")
print(f"Walk-forward from start year {g.origin.min()} (typical error):")
print(res.loc[[h for h in (1, 2, 3, 5, 7, 10, 15) if h in res.index]].round(1).to_string())
c = pd.DataFrame(coefs); print("\nLatest coefficients (per horizon):")
print(c[c.origin == c.origin.max()].round(2).to_string(index=False))
