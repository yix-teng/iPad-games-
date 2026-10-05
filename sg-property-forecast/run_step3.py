"""Step 3 review: do leading indicators improve the short-term market path?

1. Index backtest (URA Non-Landed): every quarter from 2010Q1, 1-8 quarters ahead.
2. Unit backtest re-scored: each start year's valuations (already in the backtest) are grown
   with the market path from each model; years 1-2 come from the index model, later years
   continue at trailing income growth exactly as before.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from sgpf.data import fetch_price_index
from sgpf.index_macro import backtest, predict_all
from sgpf.long_run import build_panel

out = Path("outputs")
s = fetch_price_index()["Non-Landed"].dropna()
MODELS = ["ensemble", "ensemble_macro", "ensemble_macro_all_h", "ensemble_macro_cool",
          "ensemble_macro_cool_all_h", "arima", "ridge", "ridge_macro", "ridge_macro_cool"]

# 1. Index backtest
ib = backtest(s, s.index[s.index >= pd.Period("2010Q1", "Q")])
ib["err"] = (ib.pred - ib.actual) * 100
tab = ib.pivot_table(index="h", columns="model", values="err",
                     aggfunc=lambda e: np.sqrt((e ** 2).mean()))[MODELS]
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
print("1. Index backtest, Non-Landed, origins 2010Q1+: RMSE of growth forecast (% points)")
print(tab.round(2).to_string())
w = ib.pivot_table(index=["origin", "h"], columns="model", values="err").abs()
print("   share of origins where model beats current ensemble:")
print((w[MODELS[1:5]].lt(w["ensemble"], axis=0)).groupby(level="h").mean().round(2).to_string())
tab.round(3).to_csv(out / "index_backtest_step3.csv")

# 2. Unit backtest re-scored with each market path
bt = pd.read_pickle("data/propertynoob/unit_backtest.pkl")
li = build_panel("Non-Landed").log_income.dropna()
paths = {}
for y in sorted(bt.origin.unique()):
    t = pd.Period(f"{y}Q4", "Q")
    p4, p8 = predict_all(s, t, 4), predict_all(s, t, 8)
    lt = li[li.index <= t]
    g = (lt.iloc[-1] - lt.iloc[-41]) / 10
    paths[y] = {m: (p4[m], p8[m], g) for m in MODELS[:5]}

def path(x, l1, l2, g):
    return np.where(x <= 1, x * l1, np.where(x <= 2, l1 + (x - 1) * (l2 - l1), l2 + (x - 2) * g))

rows = []
for m in MODELS[:5]:
    l1 = bt.origin.map(lambda y: paths[y][m][0]); l2 = bt.origin.map(lambda y: paths[y][m][1])
    g = bt.origin.map(lambda y: paths[y][m][2])
    bt[m] = bt.flat + path(bt.years.values, l1.values, l2.values, g.values)
for h, gr in bt[bt.h <= 5].groupby("h"):
    r = {"years": h, "start_years": gr.origin.nunique(), "sales": len(gr)}
    base = (np.exp(gr["ensemble"] - gr.actual) - 1).abs()
    for m in MODELS[:5]:
        e = (np.exp(gr[m] - gr.actual) - 1).abs()
        r[m] = e.median() * 100
        if m != "ensemble":
            r[f"{m}_better_in"] = f"{int((e.groupby(gr.origin).median() < base.groupby(gr.origin).median()).sum())}/{gr.origin.nunique()}"
    rows.append(r)
ut = pd.DataFrame(rows).set_index("years")
print("\n2. Unit backtest re-scored (typical error, % of actual price), start years 1997-2025:")
print(ut.round(2).to_string())
ut.round(3).to_csv(out / "unit_backtest_step3.csv")
