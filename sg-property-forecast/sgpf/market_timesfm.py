"""TimesFM 3.0 market path, averaged with the current path (adopted after backtesting).

The unit forecast's market path is the average, in log terms, of
  * the current path: short-term index model for years 1-2, then income growth, and
  * Google TimesFM 3.0's forecast of the URA Non-Landed index (pretrained, zero-shot).
In the 1997-2025 unit backtest this cut 10-year error from 28.0% to 19.4% and passed the
pre-set adoption rule (run_timesfm_test.py). TimesFM forecasts are cached by series, quarter
and latest index value.

Optional dependency: pip install torch "git+https://github.com/google-research/timesfm.git"
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import DATA_DIR, fetch_price_index

H = 80  # quarters (20 years)
MODEL_ID = "google/timesfm-3.0-pytorch"
_model = None


def _load():
    global _model
    if _model is None:
        try:
            import timesfm
        except ImportError as e:
            raise ImportError("TimesFM is required for the market path: pip install torch "
                              '"git+https://github.com/google-research/timesfm.git"') from e
        _model = timesfm.TimesFM3Forecaster.from_pretrained(MODEL_ID, device="cpu")
    return _model


def timesfm_log_path(asof: pd.Period, series: str = "Non-Landed") -> np.ndarray:
    """Forecast log growth of the index at 0..80 quarters after `asof`, using data to `asof`."""
    s = fetch_price_index()[series].dropna()
    hist = s[s.index <= asof]
    key = f"{series}|{hist.index[-1]}|{hist.iloc[-1]:.2f}"
    cache = DATA_DIR / "timesfm_paths.json"
    c = json.loads(cache.read_text()) if cache.exists() else {}
    if key not in c:
        out = np.asarray(_load().predict(context=hist.values.astype(np.float32),
                                         horizon=H).forecast).ravel()
        c[key] = np.r_[0.0, np.log(np.maximum(out[:H], 1e-6) / hist.iloc[-1])].tolist()
        cache.write_text(json.dumps(c))
    return np.array(c[key])


def timesfm_growth(path: np.ndarray, years) -> np.ndarray:
    """Growth multiplier from a TimesFM log path at fractional years ahead."""
    return np.exp(np.interp(np.asarray(years, dtype=float) * 4, np.arange(H + 1), path))


def averaged(current_fn, path: np.ndarray):
    """Market growth function: geometric mean of the current path and TimesFM's path."""
    return lambda x: np.sqrt(np.asarray(current_fn(x)) * timesfm_growth(path, x))
