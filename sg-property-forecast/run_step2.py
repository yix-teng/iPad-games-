"""Step 2 review: do project-level features predict how units deviate from the market path?

Walk-forward: for each start year T, a ridge model is trained on earlier backtest forecasts
whose outcomes were known by T (start year + years ahead <= T). The target is each
forecast's error after removing that start year's average error at that horizon, so the
model can only learn unit-level differences, not market corrections (step 3).

    python run_step2.py      # needs run_unit_model.py and run_unit_backtest.py outputs
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from sgpf.propertynoob import load
from sgpf.unit_features import FEATURES, design, features_asof, project_table
from sgpf.unit_forecast import ARTIFACTS
from sgpf.unit_model import project_year_levels

BASE = "market"  # current forecast: transparent valuation x market path
out = Path("outputs")
art = pickle.loads(ARTIFACTS.read_bytes())
A, d = art["A"], art["data"]
info, _ = load()
bt = pd.read_pickle("data/propertynoob/unit_backtest.pkl")
proj = project_table(info, d)
levels = project_year_levels(d, A)

feats = []
for y in sorted(bt.origin.unique()):
    f = features_asof(proj, d, y, levels)
    feats.append(f.assign(origin=y).reset_index())
F = pd.concat(feats)
bt = bt.merge(F, on=["origin", "slug"], how="left")
bt["resid"] = bt.actual - bt[BASE]
bt["target"] = bt.resid - bt.groupby(["origin", "h"]).resid.transform("mean")
bt["sale_year"] = bt.origin + bt.years

PER_YEAR = __import__("sys").argv[1:] != ["--level-only"]
rng = np.random.default_rng(0)
preds = []
for T in sorted(bt.origin.unique()):
    tr = bt[(bt.origin < T) & (bt.sale_year <= T)]
    te = bt[bt.origin == T]
    if len(tr) < 20000 or tr.origin.nunique() < 3:
        continue
    if len(tr) > 400_000:
        tr = tr.iloc[rng.choice(len(tr), 400_000, replace=False)]
    Xtr, names, stats = design(tr, per_year=PER_YEAR)
    m = Ridge(alpha=10.0).fit(Xtr, tr.target.values)
    Xte, _, _ = design(te, stats, per_year=PER_YEAR)
    adj = Xte @ m.coef_  # feature effects only, no intercept
    preds.append(te[["origin", "h", "actual", BASE]].assign(step2=te[BASE].values + adj))
P = pd.concat(preds)

err = lambda col, g: (np.exp(g[col] - g.actual) - 1).abs().median() * 100
rows = []
for h, g in P.groupby("h"):
    by_o = g.groupby("origin")
    better = sum(err("step2", x) < err(BASE, x) for _, x in by_o)
    rows.append({"years": h, "origins": g.origin.nunique(), "sales": len(g),
                 "current": err(BASE, g), "step2": err("step2", g),
                 "step2_better_in": f"{better}/{g.origin.nunique()}"})
res = pd.DataFrame(rows).set_index("years")
suffix = "" if PER_YEAR else "_level_only"
res.round(2).to_csv(out / f"unit_backtest_step2_features{suffix}.csv")
pd.set_option("display.width", 160)
print(("Level + per-year effects. " if PER_YEAR else "Level effects only. ")
      + f"Walk-forward, start years {P.origin.min()}-{P.origin.max()} "
      f"(typical error, % of actual price):")
print(res.round(1).to_string())

# Final fit on everything, for explanation: effect of +1 standard deviation of each feature.
X, names, stats = design(bt, per_year=PER_YEAR)
m = Ridge(alpha=10.0).fit(X, bt.target.values)
coef = pd.Series(m.coef_, index=names)
expl = pd.DataFrame({"feature": FEATURES, "1_sd_equals": stats[1].round(2).values,
                     "level_effect_pct": (np.exp(coef.iloc[:len(FEATURES)].values) - 1) * 100})
if PER_YEAR:
    expl["per_year_effect_pct"] = (np.exp(coef.iloc[len(FEATURES):].values) - 1) * 100
expl.round(2).to_csv(out / f"unit_step2_feature_effects{suffix}.csv", index=False)
print("\nEffect of +1 standard deviation of each feature on price vs the market path"
      " (all start years):")
print(expl.round(2).to_string(index=False))
