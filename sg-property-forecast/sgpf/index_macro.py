"""Step 3: leading indicators for the short-term market path.

Adds to the ridge half of the index ensemble, as known at each quarter:
* sgs_10y, sgs_10y_chg_4q    10-year SGS yield (mortgage-rate proxy) and its 1-year change
* pipeline_share, ..._chg_4q  private homes in the pipeline / existing stock (lagged 1 quarter,
                              to allow for publication)
* vacancy                    vacant / existing stock (lagged 1 quarter)
Variant "cool" also keeps the cooling-measure features (dropped by the plain ridge).

Macro data starts 1998Q2, so these models train on 1999 onward; where too little data exists
the ensembles fall back to the current ones.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .index_model import REGIME_FEATURES, make_features, make_target, m_arima, m_ridge
from .macro import fetch_macro

MACRO = ["sgs_10y", "sgs_10y_chg_4q", "pipeline_share", "pipeline_share_chg_4q", "vacancy"]
MIN_ROWS = 30


def macro_features(index: pd.PeriodIndex) -> pd.DataFrame:
    m = fetch_macro()
    m.index = pd.PeriodIndex(m.index, freq="Q")
    f = pd.DataFrame(index=m.index)
    f["sgs_10y"] = m.sgs_10y
    f["sgs_10y_chg_4q"] = m.sgs_10y - m.sgs_10y.shift(4)
    pipe = (m.pipeline_landed + m.pipeline_nonlanded) / m.stock_units
    f["pipeline_share"] = pipe.shift(1)
    f["pipeline_share_chg_4q"] = (pipe - pipe.shift(4)).shift(1)
    f["vacancy"] = (m.vacant_units / m.stock_units).shift(1)
    return f.reindex(index)


def _frames(s: pd.Series, h: int, end: pd.Period, with_macro: bool):
    X = make_features(s)
    if with_macro:
        X = X.join(macro_features(X.index))
    y = make_target(s, h)
    mask = (X.index >= pd.Period("1990Q1", "Q")) & (X.index <= end - h)
    d = X[mask].assign(_y=y[mask]).dropna()
    return d.drop(columns="_y"), d["_y"], X


def _ridge(Xtr, ytr, Xp, drop):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
    return float(m.fit(Xtr.drop(columns=drop), ytr).predict(Xp.drop(columns=drop))[0])


def predict_all(s: pd.Series, t: pd.Period, h: int) -> dict:
    """Log growth forecasts from t to t+h for current and step-3 models, using s[:t] only."""
    hist = s[:t]
    Xtr, ytr, Xall = _frames(hist, h, t, with_macro=False)
    Xp = Xall.loc[[t]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arima = float(m_arima(Xtr, ytr, Xp, hist, h)[0])
    ridge = float(m_ridge(Xtr, ytr, Xp, hist, h)[0])
    out = {"arima": arima, "ridge": ridge,
           "ensemble": (arima + ridge) / 2 if h <= 4 else arima}
    Mtr, mtr, Mall = _frames(hist, h, t, with_macro=True)
    Mp = Mall.loc[[t]]
    if len(Mtr) >= MIN_ROWS and Mp[MACRO].notna().all(axis=1).iloc[0]:
        out["ridge_macro"] = _ridge(Mtr, mtr, Mp, REGIME_FEATURES)
        out["ridge_macro_cool"] = _ridge(Mtr, mtr, Mp, [])
    else:  # not enough macro history yet: fall back to the current ridge
        out["ridge_macro"] = out["ridge_macro_cool"] = ridge
    for v in ("ridge_macro", "ridge_macro_cool"):
        # Same structure as the current ensemble (macro ridge replaces ridge for h <= 4) ...
        out[f"ensemble_{v[6:]}"] = (arima + out[v]) / 2 if h <= 4 else arima
        # ... and a version that also uses it for 5-8 quarters ahead.
        out[f"ensemble_{v[6:]}_all_h"] = (arima + out[v]) / 2
    return out


def backtest(s: pd.Series, origins, horizons=range(1, 9)) -> pd.DataFrame:
    rows = []
    for t in origins:
        for h in horizons:
            if t + h > s.index[-1]:
                continue
            p = predict_all(s, t, h)
            actual = float(np.log(s[t + h] / s[t]))
            rows += [{"origin": t, "h": h, "model": k, "pred": v, "actual": actual}
                     for k, v in p.items()]
    return pd.DataFrame(rows)
