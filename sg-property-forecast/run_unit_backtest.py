"""Backtest unit forecasts 1-20 years ahead against actual resale prices.

    python run_unit_backtest.py          # origins 1997-2025, ~1 hour
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

from sgpf.unit_backtest import run_origin, summarise
from sgpf.unit_forecast import ARTIFACTS

out = Path("outputs"); out.mkdir(exist_ok=True)
d = pickle.loads(ARTIFACTS.read_bytes())["data"]
first, last = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (1997, 2025)
parts = []
for y in range(first, last + 1):
    r = run_origin(d, y)
    print(f"origin {y}: {len(r):,} later resales tested"
          + (f", market 1y {r.market_g1.iloc[0]:+.1%}, income {r.income_g.iloc[0]:+.1%}/yr"
             if len(r) else ""), flush=True)
    parts.append(r)
bt = pd.concat(parts)
bt.to_pickle(Path("data/propertynoob/unit_backtest.pkl"))  # unit-level: not committed
s = summarise(bt)
s.round(2).to_csv(out / "unit_backtest_by_horizon.csv")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
print(s.round(1).to_string())
