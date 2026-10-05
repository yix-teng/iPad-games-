"""Out-of-sample test of the forecast bands: 'likely' (50%: 25th-75th percentile) and
'plausible' (80%: 10th-90th), with the adopted market path (current path averaged with TimesFM).

Test A, walk-forward: for each start year T, bands come only from earlier start years whose
outcomes were known by T (start year + horizon <= T, at least 3 start years). Coverage is the
share of start year T's actual resale prices inside those bands.
Test B, unseen projects: at start years 2008, 2013 and 2018, projects are split into 5 groups;
for each group LightGBM is refitted on sales up to T WITHOUT those projects, their units are
valued as unknown projects, grown with the market path known at T, and compared with actual
later resales. Bands are the ones the tool shows (from the full backtest).
Target: ~50% inside the likely band, ~80% inside the plausible band.
"""
import pickle
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from sgpf.market_timesfm import averaged, timesfm_log_path
from sgpf.unit_backtest import _market_fn
from sgpf.unit_forecast import ARTIFACTS
from sgpf.unit_model import AGE_BINS, AGE_LABELS, fit_gbm, predict_gbm

warnings.filterwarnings("ignore")
OUT = Path("outputs")
d = pickle.loads(ARTIFACTS.read_bytes())["data"]
bt = pd.read_pickle("data/propertynoob/unit_backtest.pkl")
paths = {}


def avg_fn(y):
    if y not in paths:
        cur, _, _ = _market_fn(pd.Timestamp(f"{y}-12-31"))
        paths[y] = averaged(cur, timesfm_log_path(pd.Period(f"{y}Q4", "Q")))
    return paths[y]


bt["pred"] = np.nan
for y in sorted(bt.origin.unique()):
    k = (bt.origin == y).values
    bt.loc[k, "pred"] = bt.flat_gbm.values[k] + np.log(avg_fn(y)(bt.years.values[k]))
bt["r"] = bt.actual - bt.pred  # log(actual / forecast)

# Published bands (what the tool shows): quantiles over the full backtest.
pub = bt.groupby("h").r.quantile([0.1, 0.25, 0.75, 0.9]).unstack()

# ---------------------------------------------------------------- Test A: walk-forward
rows = []
for T in sorted(bt.origin.unique()):
    for h in range(1, 21):
        test = bt[(bt.origin == T) & (bt.h == h)]
        past = bt[(bt.h == h) & (bt.origin + h <= T)]
        if test.empty or past.origin.nunique() < 3:
            continue
        q = past.r.quantile([0.1, 0.25, 0.75, 0.9]).values
        rows.append({"origin": T, "h": h, "n": len(test),
                     "in50": test.r.between(q[1], q[2]).mean(),
                     "in80": test.r.between(q[0], q[3]).mean(),
                     "width50": (np.exp(q[2]) - np.exp(q[1])) * 100,
                     "width80": (np.exp(q[3]) - np.exp(q[0])) * 100})
A = pd.DataFrame(rows)
ta = A.groupby("h").apply(lambda g: pd.Series({
    "start_years_tested": len(g), "resales": g.n.sum(),
    "in_likely_50": np.average(g.in50, weights=g.n),
    "in_likely_by_start_year": g.in50.mean(),
    "in_plausible_80": np.average(g.in80, weights=g.n),
    "in_plausible_by_start_year": g.in80.mean(),
    "start_years_likely_25_75pct": ((g.in50 >= .25) & (g.in50 <= .75)).mean(),
    "likely_width_pts": g.width50.median(), "plausible_width_pts": g.width80.median()}))
ta.round(3).to_csv(OUT / "band_test_walkforward.csv")
print("Test A (walk-forward) done", flush=True)

# ---------------------------------------------------------------- Test B: unseen projects
rows = []
for y in (2008, 2013, 2018):
    T = pd.Timestamp(f"{y}-12-31")
    hist = d[d.date <= T]
    later = d[(d.date > T) & (d.sale_type == "resale") & d.slug.isin(hist.slug.unique())].copy()
    fold = later.slug.map(lambda s: zlib.crc32(s.encode()) % 5)
    for f in range(5):
        units = later[fold == f].copy()
        train = hist[~hist.slug.isin(units.slug.unique())]  # projects never seen
        vcut = train.month.max() - 6
        B = fit_gbm(train[train.month <= vcut], train[train.month > vcut])
        yrs = (units.date - T).dt.days / 365.25
        at_T = units.assign(age=units.age - yrs, lease_left=units.lease_left - yrs,
                            slug="__unseen__")
        b = pd.cut(at_T.age, AGE_BINS, labels=AGE_LABELS).astype(object).fillna("unknown")
        at_T["ec_age_bin"] = np.where(at_T.is_ec == 1, b, "not EC")
        val = predict_gbm(B, at_T, at_month=str(train.month.max()))
        x = yrs.values
        pred = val + np.log(avg_fn(y)(x))
        h = np.clip(np.round(x).astype(int), 1, 20)
        rows.append(pd.DataFrame({"origin": y, "fold": f, "slug": units.slug.values, "h": h,
                                  "r": units.log_psf.values - pred}))
        print(f"  {y} fold {f}: {len(units):,} resales of {units.slug.nunique()} unseen projects",
              flush=True)
Bt = pd.concat(rows)
Bt = Bt.join(pub, on="h")
Bt["in50"] = (Bt.r >= Bt[0.25]) & (Bt.r <= Bt[0.75])
Bt["in80"] = (Bt.r >= Bt[0.1]) & (Bt.r <= Bt[0.9])
Bt["below"] = Bt.r < Bt[0.1]
Bt["above"] = Bt.r > Bt[0.9]
tb = Bt.groupby("h").agg(resales=("r", "size"), projects=("slug", "nunique"),
                         in_likely_50=("in50", "mean"), in_plausible_80=("in80", "mean"),
                         below_plausible=("below", "mean"), above_plausible=("above", "mean"),
                         typical_err_pct=("r", lambda r: np.median(np.abs(np.expm1(-r))) * 100))
tb.round(3).to_csv(OUT / "band_test_unseen_projects.csv")

pd.set_option("display.width", 220)
keep = [h for h in (1, 2, 3, 5, 7, 10, 15) if h in ta.index]
print("\nTest A, walk-forward (bands from earlier start years only):")
print(ta.loc[keep].round(2).to_string())
keep = [h for h in (1, 2, 3, 5, 7, 10, 15) if h in tb.index]
print("\nTest B, unseen projects (bands shown by the tool):")
print(tb.loc[keep].round(2).to_string())
