"""Valuation-today accuracy by segment (out-of-time: train to 2026-03, test 2026-04..09).

Includes projects with no sales before the cutoff (brand-new launches): the transparent
model cannot price those (no project premium), LightGBM falls back on location and the
other features.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sgpf.unit_forecast import ARTIFACTS
from sgpf.unit_model import TransparentModel, fit_gbm, predict_gbm

out = Path("outputs")
d = pickle.loads(ARTIFACTS.read_bytes())["data"]
cut = d.month.max() - 6
train, test = d[d.month <= cut], d[d.month > cut].copy()
A = TransparentModel().fit(train)
B = fit_gbm(train[train.month <= cut - 6], train[train.month > cut - 6])
test["pred_A"], seen = A.predict(test)
test["pred_B"] = predict_gbm(B, test, at_month=str(cut))
test["project_seen"] = seen
prior = train.groupby("slug").size()
prior_rs = train[train.sale_type == "resale"].groupby("slug").size()
test["sales_before"] = test.slug.map(prior).fillna(0)
test["resales_before"] = test.slug.map(prior_rs).fillna(0)
test["history"] = pd.cut(test.sales_before, [-1, 0, 9, 49, 1e9],
                         labels=["no sales before (brand-new)", "1-9", "10-49", "50+"]).astype(str)
test["resale_history"] = pd.cut(test.resales_before, [-1, 0, 9, 49, 1e9],
                                labels=["0 resales", "1-9", "10-49", "50+"]).astype(str)
test["type"] = np.where(test.is_ec == 1, "EC", "condo/apartment")
# What the tool uses: transparent for new sales & ECs is NOT the chosen rule; LightGBM is.
rows = []
for seg in ("sale_type", "tenure_type", "history", "resale_history", "region", "type"):
    for k, g in test.groupby(seg):
        r = {"segment": seg, "group": k, "sales": len(g)}
        for m, col in (("lightgbm", "pred_B"), ("transparent", "pred_A")):
            gg = g if m == "lightgbm" else g[g.project_seen]
            if len(gg) == 0:
                r[f"{m}_err"] = np.nan
                continue
            e = np.exp(gg[col] - gg.log_psf) - 1
            r[f"{m}_err"] = e.abs().median() * 100
            r[f"{m}_within_10"] = (e.abs() < .1).mean()
            lo, lo50, hi50, hi = (np.quantile(np.exp(gg.log_psf - gg[col]),
                                              [.1, .25, .75, .9]) - 1) * 100
            r[f"{m}_range80_lo"], r[f"{m}_range80_hi"] = lo, hi
            r[f"{m}_range50_lo"], r[f"{m}_range50_hi"] = lo50, hi50
        rows.append(r)
res = pd.DataFrame(rows)
res.round(2).to_csv(out / "unit_valuation_by_segment.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(1).to_string(index=False))
