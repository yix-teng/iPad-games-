"""Test Google TimesFM as the market path (same design and adoption rule as run_chronos_test.py).

Variants (set before running), each seeing only data up to the start year (Q4):
  timesfm25      TimesFM 2.5 (200M), URA Non-Landed index only
  timesfm3       TimesFM 3.0, index only
  timesfm3_cov   TimesFM 3.0, index plus past income per resident and CPI as covariates
Scored as: the model's path for all years, current path for years 1-2 then the model, and
the average of current and model paths. Adoption: better than current at 3, 5 and 10 years
and in the majority of start years there.
Requires: pip install torch "git+https://github.com/google-research/timesfm.git"
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import timesfm

from sgpf.data import fetch_price_index
from sgpf.long_run import build_panel
from sgpf.macro import fetch_macro
from sgpf.unit_backtest import _market_fn

warnings.filterwarnings("ignore")
H = 80
s = fetch_price_index()["Non-Landed"].dropna()
li = build_panel("Non-Landed").log_income
cpi = np.log(fetch_macro().cpi)
cpi.index = pd.PeriodIndex(cpi.index, freq="Q")

m25 = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
m25.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=128, normalize_inputs=True,
                                   use_continuous_quantile_head=True,
                                   force_flip_invariance=True, infer_is_positive=True,
                                   fix_quantile_crossing=True))
m3 = timesfm.TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cpu")


def covariates(idx):
    c = pd.DataFrame({"income": li.reindex(idx), "cpi": cpi.reindex(idx)}).bfill().ffill()
    c = (c - c.mean()) / c.std()
    return c.values.T.astype(np.float32)  # (n_covariates, T)


def forecast(name, hist):
    x = hist.values.astype(np.float32)
    if name == "timesfm25":
        p, _ = m25.forecast(horizon=H, inputs=[x])
        out = np.asarray(p)[0]
    elif name == "timesfm3":
        out = np.asarray(m3.predict(context=x, horizon=H).forecast).ravel()
    else:
        out = np.asarray(m3.predict(context=x, horizon=H,
                                    past_only_covariates=covariates(hist.index)).forecast).ravel()
    return np.r_[0.0, np.log(np.maximum(out[:H], 1e-6) / x[-1])]


MODELS = ["timesfm25", "timesfm3", "timesfm3_cov"]
bt = pd.read_pickle("data/propertynoob/unit_backtest.pkl")
paths, idx_rows = {}, []
for y in sorted(bt.origin.unique()):
    T = pd.Period(f"{y}Q4", "Q")
    hist = s[s.index <= T]
    cur, _, _ = _market_fn(pd.Timestamp(f"{y}-12-31"))
    paths[y] = {m: forecast(m, hist) for m in MODELS}
    paths[y]["cur"] = cur
    for h in (3, 5, 10):
        if T + 4 * h <= s.index[-1]:
            r = {"origin": y, "years": h, "actual": np.log(s[T + 4 * h] / s[T]),
                 "current": float(np.log(cur(h)))}
            r.update({m: paths[y][m][4 * h] for m in MODELS})
            idx_rows.append(r)
    print(f"{y}: 10y " + ", ".join(f"{m} {np.expm1(paths[y][m][40]):+.0%}" for m in MODELS)
          + f" | current {cur(10) - 1:+.0%}", flush=True)

x = bt.years.values
lcur = np.empty(len(bt)); lcur2 = np.empty(len(bt))
lm = {m: np.empty(len(bt)) for m in MODELS}; lm2 = {m: np.empty(len(bt)) for m in MODELS}
for y, p in paths.items():
    k = (bt.origin == y).values
    lcur[k] = np.log(p["cur"](x[k])); lcur2[k] = np.log(p["cur"](2.0))
    for m in MODELS:
        lm[m][k] = np.interp(x[k] * 4, np.arange(H + 1), p[m])
        lm2[m][k] = np.interp(8.0, np.arange(H + 1), p[m])
bt["current"] = bt.flat_gbm + lcur
variants = []
for m in MODELS:
    bt[m] = bt.flat_gbm + lm[m]
    bt[f"{m}_chain"] = bt.flat_gbm + np.where(x <= 2, lcur, lcur2 + lm[m] - lm2[m])
    bt[f"{m}_avg"] = bt.flat_gbm + (lcur + lm[m]) / 2
    variants += [m, f"{m}_chain", f"{m}_avg"]

err = lambda c, g: (np.abs(np.exp(g[c] - g.actual) - 1)).median() * 100
rows = []
for h, g in bt.groupby("h"):
    base = (np.abs(np.exp(g.current - g.actual) - 1)).groupby(g.origin).median()
    r = {"years": h, "start_years": g.origin.nunique(), "current": err("current", g)}
    for v in variants:
        e = (np.abs(np.exp(g[v] - g.actual) - 1)).groupby(g.origin).median()
        r[v] = err(v, g)
        r[f"{v}_better_in"] = f"{int((e < base).sum())}/{len(base)}"
    rows.append(r)
res = pd.DataFrame(rows).set_index("years")
res.round(2).to_csv(Path("outputs") / "timesfm_unit_test.csv")
ix = pd.DataFrame(idx_rows)
rm = lambda c, g: np.sqrt(((np.expm1(g[c]) - np.expm1(g.actual)) ** 2).mean()) * 100
ix_s = ix.groupby("years").apply(lambda g: pd.Series(
    {"start_years": len(g), "current": rm("current", g), **{m: rm(m, g) for m in MODELS}}))
ix_s.round(2).to_csv(Path("outputs") / "timesfm_index_test.csv")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
print("\nIndex growth forecast vs actual (RMSE, % points of total growth):")
print(ix_s.round(1).to_string())
keep = [h for h in (1, 2, 3, 5, 7, 10, 15, 20) if h in res.index]
print("\nUnit backtest, typical error (% of actual price):")
print(res.loc[keep, ["start_years", "current"] + variants].round(1).to_string())
print("\nStart years better than current:")
print(res.loc[keep, [f"{v}_better_in" for v in variants]].to_string())
