"""Explainable 3-10 year price scenarios with historically measured error and range.

    python run_long_run.py [--bear 0.03 --base 0.045 --bull 0.06] [--series Non-Landed]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sgpf.data import fetch_price_index
from sgpf.long_run import build_panel, evaluate, income_history, scenarios

p = argparse.ArgumentParser()
p.add_argument("--series", default="All Residential",
               choices=["All Residential", "Landed", "Non-Landed"])
# Defaults: weakest, median and strongest 10-year income growth among decades starting 1990+.
p.add_argument("--bear", type=float, default=0.035, help="annual nominal income growth")
p.add_argument("--base", type=float, default=0.046)
p.add_argument("--bull", type=float, default=0.059)
a = p.parse_args()
out = Path("outputs"); out.mkdir(exist_ok=True)
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 20)

hist = income_history(a.series)
print("Income growth per year (nominal GDP per resident) in each 10-year window since:")
print("  " + ", ".join(f"{p.year}: {v}%" for p, v in hist.iloc[::4].items()))
d = build_panel(a.series)
recent = (np.exp(d.log_income.iloc[-1] - d.log_income.iloc[-41]) ** 0.1 - 1) * 100
print(f"  last 10 years: {recent:.1f}% per year\n")

ev = pd.DataFrame([evaluate(h, a.series) for h in (12, 20, 40)]).set_index("years")
print("Out-of-sample accuracy, average miss in percentage points of total growth:")
print(ev.round(2).T)
ev.to_csv(out / "long_run_accuracy.csv")

sc = scenarios({"bear": a.bear, "base": a.base, "bull": a.bull}, series=a.series)
sc.round(1).to_csv(out / "long_run_scenarios.csv", index=False)
print("\nScenarios (index level):")
show = sc[sc.years.isin([3, 5, 10])].copy()
show["income_growth_pa"] = show.income_growth_pa.map("{:.1%}".format)
print(show.round(1).to_string(index=False))

P = fetch_price_index()[a.series].dropna()["2000Q1":]
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(P.index.to_timestamp(), P.values, color="#1f4e79", label=f"URA PPI – {a.series}")
x0 = P.index[-1].to_timestamp()
base = sc[sc.scenario == "base"]
xs = [x0] + list(pd.PeriodIndex(base.quarter, freq="Q").to_timestamp())
p0 = P.iloc[-1]
ax.fill_between(xs, [p0] + list(base.wide_lo), [p0] + list(base.wide_hi), color="#c0504d",
                alpha=0.12, label="Base: wide range (historical 5–95%)")
ax.fill_between(xs, [p0] + list(base.likely_lo), [p0] + list(base.likely_hi), color="#c0504d",
                alpha=0.25, label="Base: likely range (historical 10–90%)")
for name, style in (("bear", ":"), ("base", "-"), ("bull", "--")):
    s = sc[sc.scenario == name]
    ax.plot(xs, [p0] + list(s.central), color="#c0504d", ls=style,
            label=f"{name.title()}: income +{s.income_growth_pa.iloc[0]:.1%}/yr")
ax.set_ylabel("Index (2009Q1 = 100)"); ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
ax.set_title("Singapore private residential prices: 10-year scenarios (prices track incomes)")
fig.tight_layout(); fig.savefig(out / "long_run_scenarios.png", dpi=130)
print("\nSaved outputs/long_run_accuracy.csv, long_run_scenarios.csv, long_run_scenarios.png")
