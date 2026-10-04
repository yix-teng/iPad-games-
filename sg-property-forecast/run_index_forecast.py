"""Backtest models on the URA private residential price index and forecast 8 quarters ahead.

    python run_index_forecast.py [--series "All Residential"] [--refresh]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sgpf.data import fetch_price_index
from sgpf.index_model import backtest, feature_importance, forecast, score

p = argparse.ArgumentParser()
p.add_argument("--series", default="All Residential",
               choices=["All Residential", "Landed", "Non-Landed"])
p.add_argument("--model", default="ensemble")
p.add_argument("--start", default="2010Q1", help="first backtest origin")
p.add_argument("--refresh", action="store_true", help="re-download from data.gov.sg")
a = p.parse_args()

out = Path("outputs"); out.mkdir(exist_ok=True)
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 100)
s = fetch_price_index(a.refresh)[a.series].dropna()
print(f"{a.series}: {s.index[0]}..{s.index[-1]}, latest index {s.iloc[-1]}")

bt = backtest(s, start=a.start)
bt.to_csv(out / "backtest_predictions.csv", index=False)
sc = score(bt)
print("\nBacktest (expanding window, origins from", a.start, ")\n", sc)
sc.to_csv(out / "backtest_scores.csv")

fc = forecast(s, a.model, bt=bt)
print(f"\nForecast ({a.model}):\n", fc.table)
fc.table.to_csv(out / "forecast.csv")
print("\nTop features (LightGBM, h=4):\n", feature_importance(s).head(8).round(4))

fig, ax = plt.subplots(figsize=(10, 5))
hist = s["2005Q1":]
ax.plot(hist.index.to_timestamp(), hist.values, color="#1f4e79", label=f"URA PPI – {a.series}")
t = fc.table
x = [s.index[-1].to_timestamp()] + list(t.index.to_timestamp())
pt = [s.iloc[-1]] + list(t.point)
ax.plot(x, pt, color="#c0504d", label=f"Forecast ({a.model})")
ax.fill_between(x[1:], t.lo95, t.hi95, color="#c0504d", alpha=0.12, label="95% interval")
ax.fill_between(x[1:], t.lo80, t.hi80, color="#c0504d", alpha=0.25, label="80% interval")
ax.set_ylabel("Index (2009Q1 = 100)"); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
ax.set_title("Singapore private residential price index forecast")
fig.tight_layout(); fig.savefig(out / "forecast.png", dpi=130)
print("\nSaved outputs/backtest_scores.csv, outputs/forecast.csv, outputs/forecast.png")
