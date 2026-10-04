"""Forecast the URA private residential price index with direct multi-horizon models.

For each horizon h, target y_h(t) = log(I[t+h] / I[t]); features use only data up to t.
Models are compared against a random-walk baseline in an expanding-window backtest.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

from .data import COOLING_MEASURES


def make_features(s: pd.Series) -> pd.DataFrame:
    """Features at quarter t from a log-index series (no look-ahead)."""
    r = np.log(s).diff()
    f = pd.DataFrame(index=s.index)
    for k in (1, 2, 3, 4, 6, 8):
        f[f"ret_lag{k}"] = r.shift(k - 1)
    for w in (4, 8, 12):
        f[f"mom_{w}q"] = np.log(s / s.shift(w))
        f[f"vol_{w}q"] = r.rolling(w).std()
    # Distance from long-run trend (mean reversion signal).
    f["gap_trend_20q"] = np.log(s) - np.log(s).rolling(20).mean()
    f["drawdown"] = np.log(s / s.cummax())
    f["quarter"] = s.index.quarter
    ends = s.index.to_timestamp(how="end")
    cm = COOLING_MEASURES.values
    n_before = np.searchsorted(cm, ends.values, side="right")
    last = np.where(n_before > 0, cm[np.maximum(n_before - 1, 0)], np.datetime64("NaT"))
    f["q_since_cooling"] = np.clip(
        ((ends.values - last) / np.timedelta64(91, "D")).astype(float), 0, 40)
    f["q_since_cooling"] = f["q_since_cooling"].fillna(40)
    f["cooling_last_4q"] = (f["q_since_cooling"] < 4).astype(int)
    return f


def make_target(s: pd.Series, h: int) -> pd.Series:
    return np.log(s.shift(-h) / s)


# ---- models: each takes (X_train, y_train, X_pred, series_train, h) -> prediction of y_h ----

def m_random_walk(X, y, Xp, s, h):
    return np.zeros(len(Xp))


def m_drift(X, y, Xp, s, h):
    r = np.log(s).diff().dropna().iloc[-40:]  # 10-year average quarterly growth
    return np.full(len(Xp), r.mean() * h)


def m_arima(X, y, Xp, s, h):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = np.log(s).diff().dropna().iloc[-80:]
        fit = ARIMA(r.values, order=(2, 0, 1)).fit()
        return np.full(len(Xp), fit.forecast(h).sum())


# Cooling-measure features are near-constant before ~2012 (first measure 2009), so a linear
# model extrapolates them wildly in early backtest windows. Trees are immune; ridge drops them.
REGIME_FEATURES = ["q_since_cooling", "cooling_last_4q"]


def m_ridge(X, y, Xp, s, h):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
    return m.fit(X.drop(columns=REGIME_FEATURES), y).predict(Xp.drop(columns=REGIME_FEATURES))


LGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  verbose=-1)


def m_lgbm(X, y, Xp, s, h):
    return lgb.LGBMRegressor(**LGB_PARAMS, random_state=0).fit(X, y).predict(Xp)


def m_ensemble(X, y, Xp, s, h):
    """ARIMA + ridge for short horizons; ARIMA alone beyond a year.

    With ~140 quarterly training rows, direct ML models overfit at long horizons (8-quarter
    overlapping targets give only ~17 independent observations), so the backtest favours
    the parsimonious ARIMA there.
    """
    if h <= 4:
        return (m_ridge(X, y, Xp, s, h) + m_arima(X, y, Xp, s, h)) / 2
    return m_arima(X, y, Xp, s, h)


MODELS = {"random_walk": m_random_walk, "drift": m_drift, "arima": m_arima,
          "ridge": m_ridge, "lgbm": m_lgbm, "ensemble": m_ensemble}


def _train_frame(s: pd.Series, h: int, end: pd.Period, min_start: str = "1990Q1"):
    X = make_features(s)
    y = make_target(s, h)
    # Only targets fully observed by `end` may be used for training.
    mask = (X.index >= pd.Period(min_start, "Q")) & (X.index <= end - h)
    d = X[mask].assign(_y=y[mask]).dropna()
    return d.drop(columns="_y"), d["_y"], X


def backtest(s: pd.Series, horizons=range(1, 9), start="2010Q1", models=None) -> pd.DataFrame:
    """Expanding-window backtest. Returns one row per (origin, horizon, model)."""
    models = models or list(MODELS)
    rows = []
    origins = s.index[(s.index >= pd.Period(start, "Q"))]
    for t in origins:
        hist = s[:t]
        for h in horizons:
            if t + h > s.index[-1]:
                continue
            Xtr, ytr, Xall = _train_frame(hist, h, t)
            Xp = Xall.loc[[t]]
            actual = np.log(s[t + h] / s[t])
            for name in models:
                pred = float(MODELS[name](Xtr, ytr, Xp, hist, h)[0])
                rows.append({"origin": t, "h": h, "model": name,
                             "pred": pred, "actual": actual,
                             "pred_level": s[t] * np.exp(pred), "actual_level": s[t + h]})
    return pd.DataFrame(rows)


def score(bt: pd.DataFrame) -> pd.DataFrame:
    e = bt.assign(err=bt.pred - bt.actual,
                  ape=(bt.pred_level - bt.actual_level).abs() / bt.actual_level,
                  hit=np.sign(bt.pred) == np.sign(bt.actual))
    g = e.groupby(["h", "model"])
    out = pd.DataFrame({
        "MAE_pct": g.err.apply(lambda x: x.abs().mean() * 100),
        "RMSE_pct": g.err.apply(lambda x: np.sqrt((x ** 2).mean()) * 100),
        "MAPE_level_pct": g.ape.mean() * 100,
        "direction_acc": g.hit.mean(),
        "n": g.size(),
    })
    rw = out.xs("random_walk", level="model")["RMSE_pct"]
    out["rel_RMSE_vs_RW"] = out["RMSE_pct"] / out.index.get_level_values("h").map(rw).values
    return out.round(3)


@dataclass
class Forecast:
    table: pd.DataFrame  # index: target quarter; columns: point, lo80, hi80, lo95, hi95


def forecast(s: pd.Series, model: str = "ensemble", horizons=range(1, 9),
             bt: pd.DataFrame | None = None) -> Forecast:
    """Forecast future index levels. Intervals come from empirical backtest residuals."""
    t = s.index[-1]
    rows = []
    for h in horizons:
        Xtr, ytr, Xall = _train_frame(s, h, t)
        pred = float(MODELS[model](Xtr, ytr, Xall.loc[[t]], s, h)[0])
        if bt is not None and ((bt.model == model) & (bt.h == h)).any():
            res = (bt.query("model == @model and h == @h").eval("actual - pred")).values
        else:  # fall back to in-sample residual scale
            res = ytr.values - ytr.mean()
        q = np.quantile(res, [0.025, 0.1, 0.9, 0.975])
        lvl = lambda x: s[t] * np.exp(pred + x)
        rows.append({"quarter": t + h, "h": h, "growth_pct": (np.exp(pred) - 1) * 100,
                     "point": lvl(0), "lo95": lvl(q[0]), "lo80": lvl(q[1]),
                     "hi80": lvl(q[2]), "hi95": lvl(q[3])})
    return Forecast(pd.DataFrame(rows).set_index("quarter").round(2))


def feature_importance(s: pd.Series, h: int = 4) -> pd.Series:
    Xtr, ytr, _ = _train_frame(s, h, s.index[-1])
    m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=0, importance_type="gain").fit(Xtr, ytr)
    return pd.Series(m.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
