"""Explainable 3-10 year price scenarios, in future dollars and today's dollars.

    python run_long_run.py                                  # anchors taken from history
    python run_long_run.py --base 0.03 --inflation 0.02     # your own assumptions
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sgpf.data import fetch_price_index
from sgpf.long_run import decade_history, evaluate, scenario_anchors, scenarios

anchors = scenario_anchors()  # weakest / median / strongest decade since 1990
p = argparse.ArgumentParser()
p.add_argument("--series", default="All Residential",
               choices=["All Residential", "Landed", "Non-Landed"])
p.add_argument("--bear", type=float, default=anchors["bear"], help="real income growth p.a.")
p.add_argument("--base", type=float, default=anchors["base"])
p.add_argument("--bull", type=float, default=anchors["bull"])
p.add_argument("--inflation", type=float, default=anchors["inflation"], help="CPI inflation p.a.")
a = p.parse_args()
out = Path("outputs"); out.mkdir(exist_ok=True)
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 20)

hist = decade_history(a.series)
print("History: average % per year over each 10-year window, by start year")
show = hist.iloc[::2].copy(); show.index = show.index.year
print(show.to_string())
since = hist[hist.index.year >= 1990]
print("Decades starting 1990+ (min / median / max):")
print(since.agg(["min", "median", "max"]).round(1).to_string(), "\n")

ev = pd.DataFrame([evaluate(h, a.series) for h in (12, 20, 40)]).set_index("years")
print("Out-of-sample accuracy, average miss in percentage points of total growth:")
print(ev.round(2).T.to_string())
ev.to_csv(out / "long_run_accuracy.csv")

sc = scenarios({"bear": a.bear, "base": a.base, "bull": a.bull}, a.inflation, series=a.series)
sc.round(3).to_csv(out / "long_run_scenarios.csv", index=False)
fmt = sc[sc.years.isin([3, 5, 10])].copy()
for c in ("real_income_growth_pa", "inflation_pa", "nominal_income_growth_pa"):
    fmt[c] = fmt[c].map("{:.1%}".format)
cols = ["scenario", "real_income_growth_pa", "inflation_pa", "nominal_income_growth_pa", "quarter"]
print("\nFuture dollars (index level; the price tag in that year):")
print(fmt[cols + ["central", "likely_lo", "likely_hi", "change_pct"]].round(1).to_string(index=False))
print("\nToday's dollars (index level deflated by assumed CPI; purchasing power):")
print(fmt[cols + ["real_central", "real_likely_lo", "real_likely_hi", "real_change_pct"]]
      .round(1).to_string(index=False))

sens = []
for name, infl in (("low", since.inflation.min() / 100), ("median", a.inflation),
                   ("high", since.inflation.max() / 100)):
    r = scenarios({"base": a.base}, infl, series=a.series).iloc[-1]
    sens.append({"inflation": f"{name} ({infl:.1%})", "10y_change_future_$": round(r.change_pct, 1),
                 "10y_change_todays_$": round(r.real_change_pct, 1)})
print("\nBase case, 10 years, sensitivity to inflation:")
print(pd.DataFrame(sens).to_string(index=False))

P = fetch_price_index()[a.series].dropna()["2000Q1":]
p0, x0 = P.iloc[-1], P.index[-1].to_timestamp()
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, pre, title in ((axes[0], "", "Future dollars (price tag)"),
                       (axes[1], "real_", f"Today's dollars (after {a.inflation:.1%}/yr inflation)")):
    ax.plot(P.index.to_timestamp(), P.values, color="#1f4e79", label=f"URA PPI – {a.series}")
    base = sc[sc.scenario == "base"]
    xs = [x0] + list(pd.PeriodIndex(base.quarter, freq="Q").to_timestamp())
    ax.fill_between(xs, [p0] + list(base[pre + "wide_lo"]), [p0] + list(base[pre + "wide_hi"]),
                    color="#c0504d", alpha=0.12, label="Base: wide range (historical 5–95%)")
    ax.fill_between(xs, [p0] + list(base[pre + "likely_lo"]),
                    [p0] + list(base[pre + "likely_hi"]), color="#c0504d", alpha=0.25,
                    label="Base: likely range (historical 10–90%)")
    for name, style in (("bear", ":"), ("base", "-"), ("bull", "--")):
        s = sc[sc.scenario == name]
        ax.plot(xs, [p0] + list(s[pre + "central"]), color="#c0504d", ls=style,
                label=f"{name.title()}: real income +{s.real_income_growth_pa.iloc[0]:.1%}/yr")
    ax.set_title(title); ax.grid(alpha=0.3)
axes[0].set_ylabel("Index (2009Q1 = 100)"); axes[0].legend(loc="upper left", fontsize=8)
fig.suptitle("Singapore private residential prices: 10-year scenarios (prices track incomes)")
fig.tight_layout(); fig.savefig(out / "long_run_scenarios.png", dpi=130)
print("\nSaved outputs/long_run_accuracy.csv, long_run_scenarios.csv, long_run_scenarios.png")
