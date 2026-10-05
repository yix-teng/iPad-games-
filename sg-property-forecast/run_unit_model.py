"""Fit and test the unit valuation models on propertynoob data; save artifacts for forecasts.

    python -m sgpf.propertynoob      # collect data first (resumable, ~2.5 h)
    python run_unit_model.py
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sgpf.geo import geocode_postals, location_features, postal_from_address
from sgpf.propertynoob import load
from sgpf.unit_forecast import ARTIFACTS
from sgpf.unit_model import (TransparentModel, evaluate, fit_gbm, prepare,
                             relative_performance)

out = Path("outputs"); out.mkdir(exist_ok=True)
pd.set_option("display.width", 160); pd.set_option("display.max_rows", 200)
info, tx = load()
print(f"{len(info):,} projects, {len(tx):,} transactions collected")
loc = location_features(geocode_postals(info.Address.map(postal_from_address).dropna()))
d = prepare(info, tx, loc)
print(f"{len(d):,} usable sales in {d.slug.nunique():,} projects, {d.date.min():%Y-%m}"
      f" to {d.date.max():%Y-%m}")

res, A_test, B_test, test = evaluate(d)
print(json.dumps(res, indent=1))
(out / "unit_model_accuracy.json").write_text(json.dumps(res, indent=1))

# Final models on all data.
A = TransparentModel().fit(d)
vcut = d.month.max() - 6
B = fit_gbm(d[d.month <= vcut], d[d.month > vcut])
pairs, tab = relative_performance(d, A)
tab.to_csv(out / "unit_relative_performance.csv")
print("\nHow condos kept up with their region (% per year, 5-year spans):\n", tab)
tables = {t: A.table(t) for t in ("floor", "area", "ec_age", "sale_type")}
pd.concat(tables, names=["term", "level"]).rename("effect_pct").to_csv(out / "unit_effects.csv")
for t, s in tables.items():
    print(f"\n{t} effect (% vs reference):\n", s.to_string())

ARTIFACTS.write_bytes(pickle.dumps({"A": A, "B": B, "data": d, "rel_table": tab,
                                    "accuracy": res}))
print(f"\nSaved {ARTIFACTS} (not committed) and outputs/unit_*.csv/json")
