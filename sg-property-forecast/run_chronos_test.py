"""Test a pretrained deep-learning forecaster (Amazon Chronos-Bolt Base) as the market path.

At each start year (Q4), Chronos sees only the URA Non-Landed index up to then and forecasts
80 quarters ahead (median path). Variants scored in the unit backtest (LightGBM valuations,
actual resales up to 20 years later), against the current path:
  chronos  Chronos path for all years
  chain    current path for years 1-2, then Chronos's growth from year 2 onward
  average  average (in log terms) of the current and Chronos paths
Adoption rule (set before running): better than current at 3, 5 and 10 years, and in the
majority of start years at those horizons.
Requires: pip install torch chronos-forecasting
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

from sgpf.data import fetch_price_index
from sgpf.unit_backtest import _market_fn

warnings.filterwarnings("ignore")
torch.manual_seed(0)
pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map="cpu",
                                           torch_dtype=torch.float32)
s = fetch_price_index()["Non-Landed"].dropna()
bt = pd.read_pickle("data/propertynoob/unit_backtest.pkl")
H = 80

paths, idx_rows = {}, []
for y in sorted(bt.origin.unique()):
    T = pd.Period(f"{y}Q4", "Q")
    hist = s[s.index <= T]
    _, med = pipe.predict_quantiles(torch.tensor(hist.values, dtype=torch.float32),
                                    prediction_length=H, quantile_levels=[0.5])
    chron = np.r_[0.0, np.log(med[0].numpy() / hist.iloc[-1])]  # log growth at 0..80 quarters
    cur, _, _ = _market_fn(pd.Timestamp(f"{y}-12-31"))
    paths[y] = (chron, cur)
    for h in (3, 5, 10):
        if T + 4 * h <= s.index[-1]:
            act = np.log(s[T + 4 * h] / s[T])
            idx_rows.append({"origin": y, "years": h, "actual": act, "chronos": chron[4 * h],
                             "current": float(np.log(cur(h)))})
    print(f"{y}: Chronos 3y {np.expm1(chron[12]):+.1%}, 10y {np.expm1(chron[40]):+.1%} | "
          f"current 3y {cur(3) - 1:+.1%}, 10y {cur(10) - 1:+.1%}", flush=True)


def chron_at(c, x):
    return np.interp(x * 4, np.arange(H + 1), c)


x = bt.years.values
lc = np.empty(len(bt)); lcur = np.empty(len(bt)); lc2 = np.empty(len(bt)); lcur2 = np.empty(len(bt))
for y, (c, cur) in paths.items():
    m = (bt.origin == y).values
    lc[m] = chron_at(c, x[m]); lcur[m] = np.log(cur(x[m]))
    lc2[m] = chron_at(c, 2.0); lcur2[m] = np.log(cur(2.0))
bt["current"] = bt.flat_gbm + lcur
bt["chronos"] = bt.flat_gbm + lc
bt["chain"] = bt.flat_gbm + np.where(x <= 2, lcur, lcur2 + lc - lc2)
bt["average"] = bt.flat_gbm + (lcur + lc) / 2

err = lambda c, g: (np.abs(np.exp(g[c] - g.actual) - 1)).median() * 100
rows = []
for h, g in bt.groupby("h"):
    r = {"years": h, "start_years": g.origin.nunique(), "current": err("current", g)}
    base = (np.abs(np.exp(g.current - g.actual) - 1)).groupby(g.origin).median()
    for v in ("chronos", "chain", "average"):
        r[v] = err(v, g)
        e = (np.abs(np.exp(g[v] - g.actual) - 1)).groupby(g.origin).median()
        r[f"{v}_better_in"] = f"{int((e < base).sum())}/{len(base)}"
    rows.append(r)
res = pd.DataFrame(rows).set_index("years")
res.round(2).to_csv(Path("outputs") / "chronos_unit_test.csv")
ix = pd.DataFrame(idx_rows)
ix_s = ix.groupby("years").apply(lambda g: pd.Series({
    "start_years": len(g),
    "current_rmse_pts": np.sqrt(((np.expm1(g.current) - np.expm1(g.actual)) ** 2).mean()) * 100,
    "chronos_rmse_pts": np.sqrt(((np.expm1(g.chronos) - np.expm1(g.actual)) ** 2).mean()) * 100}))
ix_s.round(2).to_csv(Path("outputs") / "chronos_index_test.csv")
pd.set_option("display.width", 200)
print("\nIndex growth forecast vs actual (RMSE, % points of total growth):")
print(ix_s.round(1).to_string())
print("\nUnit backtest (typical error, % of actual price):")
print(res.loc[[h for h in (1, 2, 3, 5, 7, 10, 15, 20) if h in res.index]].round(1).to_string())
