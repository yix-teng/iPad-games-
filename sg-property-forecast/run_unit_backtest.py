"""Backtest unit forecasts 1-20 years ahead against actual resale prices.

    python run_unit_backtest.py              # origins 1997-2025; resumable (~2-3 h with LightGBM)
    python run_unit_backtest.py 2010 2012    # a subset of origins

Each origin's results are cached in data/propertynoob/backtest_parts/ (unit-level, not
committed); delete that folder to start over after changing the models.
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

from sgpf.unit_backtest import run_origin, summarise
from sgpf.unit_forecast import ARTIFACTS

out = Path("outputs"); out.mkdir(exist_ok=True)
parts_dir = Path("data/propertynoob/backtest_parts"); parts_dir.mkdir(parents=True, exist_ok=True)
d = pickle.loads(ARTIFACTS.read_bytes())["data"]
first, last = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (1997, 2025)
for y in range(first, last + 1):
    f = parts_dir / f"{y}.pkl"
    if f.exists():
        continue
    r = run_origin(d, y)
    r.to_pickle(f)
    print(f"origin {y}: {len(r):,} later resales tested"
          + (f", market 1y {r.market_g1.iloc[0]:+.1%}, income {r.income_g.iloc[0]:+.1%}/yr"
             if len(r) else ""), flush=True)
bt = pd.concat([pd.read_pickle(f) for f in sorted(parts_dir.glob("*.pkl"))])
bt.to_pickle(Path("data/propertynoob/unit_backtest.pkl"))
method = "market_gbm" if bt.get("market_gbm", pd.Series(dtype=float)).notna().any() else "market"
s = summarise(bt, method)
s.round(2).to_csv(out / "unit_backtest_by_horizon.csv")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print(f"Ranges and within-x% columns are for: {method}")
print(s.round(1).to_string())
