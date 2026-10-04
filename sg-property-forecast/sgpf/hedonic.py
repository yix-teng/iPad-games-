"""Transaction-level (hedonic) model: predict price per sqm of individual URA transactions.

Answers "what is this unit worth / what will it be worth", complementing the index model
which answers "where is the market going". Validation is out-of-time: train on older
transactions, test on the most recent months.
"""
from __future__ import annotations

import re

import lightgbm as lgb
import numpy as np
import pandas as pd

CATEGORICAL = ["market_segment", "property_type", "type_of_sale", "type_of_area",
               "district", "tenure_type", "project"]
NUMERIC = ["area", "floor_mid", "lease_remaining", "lease_start_year", "x", "y", "t_months"]


def _tenure(t: str, sale_year: float):
    """'Freehold' / '99 yrs lease commencing from 2015' -> (type, remaining yrs, start year)."""
    t = str(t)
    if "freehold" in t.lower():
        return "freehold", 999.0, np.nan
    m = re.search(r"(\d+)\s*yrs?.*?(\d{4})", t)
    if not m:
        return "unknown", np.nan, np.nan
    yrs, start = int(m.group(1)), int(m.group(2))
    kind = "999yr" if yrs >= 900 else f"{yrs}yr"
    return kind, yrs - (sale_year - start), float(start)


def _floor_mid(fr: str) -> float:
    """'06-10' -> 8, 'B1-B5' -> -3, '-' (landed) -> NaN."""
    nums = [(-int(b) if b else int(n)) for b, n in re.findall(r"(?:B(\d+))|(\d+)", str(fr))]
    return float(np.mean(nums)) if nums else np.nan


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # URA contractDate is MMYY, e.g. '0324' = Mar 2024.
    cd = d["contract_date"].astype(str).str.zfill(4)
    d["sale_date"] = pd.to_datetime("20" + cd.str[2:] + "-" + cd.str[:2] + "-01")
    d["sale_year"] = d.sale_date.dt.year + (d.sale_date.dt.month - 1) / 12
    for c in ("price", "area", "x", "y", "no_of_units"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "no_of_units" in d:  # bulk deals distort per-unit prices
        d = d[d.no_of_units.fillna(1) == 1]
    ten = [_tenure(t, y) for t, y in zip(d["tenure"], d["sale_year"])]
    d["tenure_type"], d["lease_remaining"], d["lease_start_year"] = map(list, zip(*ten))
    d["floor_mid"] = d["floor_range"].map(_floor_mid)
    d["t_months"] = (d.sale_date.dt.year - 2000) * 12 + d.sale_date.dt.month
    d["psm"] = d.price / d.area
    d = d[(d.psm > 0) & d.psm.notna()]
    for c in CATEGORICAL:
        if c not in d:
            d[c] = "NA"
        d[c] = d[c].astype(str).astype("category")
    for c in NUMERIC:
        if c not in d:
            d[c] = np.nan
    return d.reset_index(drop=True)


def train_eval(df: pd.DataFrame, test_months: int = 6, val_months: int = 6, seed: int = 0):
    """Out-of-time split: train | validation (early stopping) | test (last `test_months`).

    Returns (model, metrics dict, test frame with predictions).
    """
    d = prepare(df)
    cutoff = d.sale_date.max() - pd.DateOffset(months=test_months)
    val_cut = cutoff - pd.DateOffset(months=val_months)
    tr, va = d[d.sale_date <= val_cut], d[(d.sale_date > val_cut) & (d.sale_date <= cutoff)]
    te = d[d.sale_date > cutoff]
    X = lambda f: f[CATEGORICAL + NUMERIC]
    model = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                              min_child_samples=20, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, cat_smooth=20, random_state=seed,
                              verbose=-1)
    model.fit(X(tr), np.log(tr.psm), eval_set=[(X(va), np.log(va.psm))],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    te = te.assign(pred_psm=np.exp(model.predict(X(te))))
    ape = (te.pred_psm - te.psm).abs() / te.psm
    metrics = {"n_train": len(tr), "n_val": len(va), "n_test": len(te), "cutoff": str(cutoff.date()),
               "MAPE_pct": round(float(ape.mean()) * 100, 2),
               "median_APE_pct": round(float(ape.median()) * 100, 2),
               "within_10pct": round(float((ape < 0.10).mean()), 3)}
    return model, metrics, te


def predict_future(model, units: pd.DataFrame, index_growth: float = 0.0) -> pd.Series:
    """Price per sqm for `units` (same columns as transactions), rolled forward by the index.

    Tree models cannot extrapolate time trends, so the hedonic model prices each unit at the
    latest market level and `index_growth` (e.g. the index model's forecast growth for the
    target quarter, as a fraction) rolls it forward: psm_future = psm_now * (1 + growth).
    """
    d = prepare(units)
    return pd.Series(np.exp(model.predict(d[CATEGORICAL + NUMERIC])) * (1 + index_growth),
                     index=d.index, name="pred_psm")
