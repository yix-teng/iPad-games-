"""Recalibrate the forecast bands and add bands for unknown completed projects.

1. Recency-weighted bands. Band quantiles (likely: 25th-75th, plausible: 10th-90th of
   log(actual/forecast)) are computed with start years weighted by 0.5^(age / half-life).
   Half-lives compared in a walk-forward test (bands from earlier start years only):
   none (equal weights), 10 years, 5 years. The one with coverage closest to 50% / 80% is used
   to rebuild the bands of all four pricing paths from all start years.
   Run with --unknown-only to redo just part 2.
2. Unknown completed projects. At 2008, 2013 and 2018, 5 project groups are valued by LightGBM
   refitted without them (as in run_band_test.py); residuals at the start (sales within 6
   months) and 1-20 years later give the bands for this path.
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
from sgpf.unit_model import AGE_BINS, AGE_LABELS, TransparentModel, fit_gbm, predict_gbm

warnings.filterwarnings("ignore")
OUT, DP = Path("outputs"), Path("data/propertynoob")
d = pickle.loads(ARTIFACTS.read_bytes())["data"]
LATEST = 2025
_paths = {}


def avg_fn(q: pd.Period):
    if q not in _paths:
        cur, _, _ = _market_fn(q.end_time.normalize())
        _paths[q] = (cur, averaged(cur, timesfm_log_path(q)))
    return _paths[q]


def wquant(x, w, qs):
    o = np.argsort(x); x, w = np.asarray(x)[o], np.asarray(w)[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(qs, c, x)


def weights(start_years, ref, half_life):
    a = np.asarray(ref - start_years, dtype=float)
    return np.ones_like(a) if half_life is None else 0.5 ** (a / half_life)


def band_table(df, group_col, half_life, start_col="origin"):
    rows = []
    for h, g in df.groupby("h"):
        w = weights(g[start_col].values, LATEST, half_life)
        q = wquant(g.r.values, w, [0.1, 0.25, 0.5, 0.75, 0.9])
        e = np.abs(np.expm1(-g.r))  # |forecast / actual - 1|
        rows.append({"years": h, "sales": len(g), "groups": g[group_col].nunique(),
                     "typical_err": np.median(e) * 100, "within_10": (e < .1).mean(),
                     "range80_lo": np.expm1(q[0]) * 100, "range50_lo": np.expm1(q[1]) * 100,
                     "range50_hi": np.expm1(q[3]) * 100, "range80_hi": np.expm1(q[4]) * 100,
                     "bias": np.median(np.expm1(-g.r)) * 100})
    return pd.DataFrame(rows).set_index("years")


import sys

UNKNOWN_ONLY = "--unknown-only" in sys.argv  # skip part 1 (e.g. to resume after part 1 ran)
if not UNKNOWN_ONLY:
    # ------------------------------------------------ residuals for each pricing path
    bt = pd.read_pickle(DP / "unit_backtest.pkl")
    top = d.groupby("slug").agg(top_year=("top_year", "first"), is_ec=("is_ec", "max"))
    bt = bt.join(top, on="slug")
    bt["avg"] = np.nan
    for y in sorted(bt.origin.unique()):
        k = (bt.origin == y).values
        _, avg = avg_fn(pd.Period(f"{y}Q4", "Q"))
        bt.loc[k, "avg"] = bt.flat_gbm.values[k] + np.log(avg(bt.years.values[k]))
    bt["r"] = bt.actual - bt.avg
    resale = bt                                             # tool's resale table: all units
    launch_phase = bt[(bt.top_year > bt.origin) & (bt.is_ec == 0)]
    ec = bt[bt.is_ec == 1].copy()
    ec_adj = np.zeros(len(ec))
    for o in sorted(ec.origin.unique()):
        m = (ec.origin == o).values
        hist = d[d.date <= pd.Timestamp(f"{o}-12-31")]
        if hist.is_ec.sum() < 200:
            continue
        Ao = TransparentModel().fit(hist)
        age0 = (o + 1) - ec.top_year.values[m]
        b0 = pd.cut(age0, AGE_BINS, labels=AGE_LABELS).astype(object)
        b1 = pd.cut(age0 + ec.years.values[m], AGE_BINS, labels=AGE_LABELS).astype(object)
        f = np.vectorize(lambda l: Ao.effect("ec_age", l))
        ec_adj[m] = f(b1) - f(b0)
    ec["r"] = ec.actual - (ec.avg + np.nan_to_num(ec_adj))
    print("EC residuals done", flush=True)
    lb = pd.read_pickle(DP / "launch_backtest.pkl")
    lb = lb[lb.is_ec == 0].copy()
    launch = d[d.sale_type == "new"].groupby("slug").date.min()
    lb["launch_year"] = lb.group.map(launch).dt.year
    lb["r"] = np.nan
    for slug, g in lb.groupby("group"):
        q = pd.Period(launch[slug], "Q") - 1
        cur, avg = avg_fn(q)
        k = (lb.group == slug).values
        lb.loc[k, "r"] = lb.actual.values[k] - (lb.launch_plain.values[k]
                                                - np.log(cur(lb.x.values[k]))
                                                + np.log(avg(lb.x.values[k])))
    pickle.dump({"resale": resale[["origin", "h", "slug", "r"]],
                 "launch_phase": launch_phase[["origin", "h", "slug", "r"]],
                 "ec": ec[["origin", "h", "slug", "r"]],
                 "brand_new": lb[["launch_year", "h", "group", "r"]]},
                open(DP / "band_residuals.pkl", "wb"))

    # ------------------------------------------------ 1. choose the half-life (walk-forward)
    HL = {"equal weights": None, "10-yr half-life": 10, "5-yr half-life": 5}
    rows = []
    for name, hl in HL.items():
        for T in sorted(resale.origin.unique()):
            for h in range(1, 21):
                test = resale[(resale.origin == T) & (resale.h == h)]
                past = resale[(resale.h == h) & (resale.origin + h <= T)]
                if test.empty or past.origin.nunique() < 3:
                    continue
                q = wquant(past.r.values, weights(past.origin.values, T, hl), [.1, .25, .75, .9])
                rows.append({"weighting": name, "h": h, "n": len(test),
                             "in50": test.r.between(q[1], q[2]).mean(),
                             "in80": test.r.between(q[0], q[3]).mean()})
    W = pd.DataFrame(rows)
    cal = W.groupby(["weighting", "h"]).apply(lambda g: pd.Series({
        "in_likely_50": np.average(g.in50, weights=g.n),
        "in_plausible_80": np.average(g.in80, weights=g.n)})).reset_index()
    hs = [1, 2, 3, 5, 7, 10]
    score = cal[cal.h.isin(hs)].groupby("weighting").apply(
        lambda g: (np.abs(g.in_likely_50 - .5) + np.abs(g.in_plausible_80 - .8)).mean() / 2)
    chosen = score.idxmin()
    cal.round(3).to_csv(OUT / "band_calibration_walkforward.csv", index=False)
    print("\nWalk-forward coverage by weighting (likely / plausible):")
    print(cal[cal.h.isin(hs)].pivot(index="h", columns="weighting",
          values=["in_likely_50", "in_plausible_80"]).round(2).to_string())
    print("mean |coverage - target|:", score.round(3).to_dict(), "-> chosen:", chosen)
    hl = HL[chosen]

    # ------------------------------------------------ rebuild the bands with the chosen weighting
    band_table(resale, "origin", hl).round(2).to_csv(OUT / "bands_resale.csv")
    band_table(launch_phase, "origin", hl).round(2).to_csv(OUT / "bands_launch_phase.csv")
    band_table(ec, "origin", hl).round(2).to_csv(OUT / "bands_ec.csv")
    bn = band_table(lb.rename(columns={"group": "slug"}), "slug", hl, start_col="launch_year")
    bn.round(2).to_csv(OUT / "bands_brand_new_launch.csv")
    print("band tables written", flush=True)

# ------------------------------------------------ 2. unknown completed projects
rows = []
for y in (2008, 2013, 2018):
    T = pd.Timestamp(f"{y}-12-31")
    hist = d[d.date <= T]
    known = hist.slug.unique()
    after = d[(d.date > T) & (d.sale_type == "resale") & d.slug.isin(known)
              & (d.top_year <= y)].copy()
    fold = after.slug.map(lambda s: zlib.crc32(s.encode()) % 5)
    for f in range(5):
        units = after[fold == f].copy()
        train = hist[~hist.slug.isin(units.slug.unique())]
        vcut = train.month.max() - 6
        B = fit_gbm(train[train.month <= vcut], train[train.month > vcut])
        yrs = (units.date - T).dt.days / 365.25
        at_T = units.assign(age=units.age - yrs, lease_left=units.lease_left - yrs,
                            slug="__unseen__")
        val = predict_gbm(B, at_T, at_month=str(train.month.max()))
        x = yrs.values
        _, avg = avg_fn(pd.Period(f"{y}Q4", "Q"))
        r_now = units.log_psf.values - val  # vs value at T, before any market growth
        r_fut = units.log_psf.values - (val + np.log(avg(x)))
        rows.append(pd.DataFrame({"origin": y, "slug": units.slug.values, "x": x,
                                  "h": np.clip(np.round(x).astype(int), 0, 20),
                                  "r_now": r_now, "r": r_fut}))
        print(f"  unknown completed projects {y} fold {f}: {units.slug.nunique()} projects",
              flush=True)
U = pd.concat(rows)
U.to_pickle(DP / "unseen_completed_residuals.pkl")
now = U[U.x <= 0.5]
q = np.quantile(now.r_now, [.1, .25, .5, .75, .9])
e = np.abs(np.expm1(-now.r_now))
today = pd.DataFrame([{"years": 0, "sales": len(now), "groups": now.slug.nunique(),
                       "typical_err": np.median(e) * 100, "within_10": (e < .1).mean(),
                       "range80_lo": np.expm1(q[0]) * 100, "range50_lo": np.expm1(q[1]) * 100,
                       "range50_hi": np.expm1(q[3]) * 100, "range80_hi": np.expm1(q[4]) * 100,
                       "bias": np.median(np.expm1(-now.r_now)) * 100}]).set_index("years")
fut = band_table(U[U.h >= 1], "slug", None)  # only 3 start years: equal weights
pd.concat([today, fut]).round(2).to_csv(OUT / "bands_unknown_completed.csv")

pd.set_option("display.width", 220)
for f in ("bands_resale", "bands_launch_phase", "bands_ec", "bands_brand_new_launch",
          "bands_unknown_completed"):
    t = pd.read_csv(OUT / f"{f}.csv", index_col="years")
    print(f"\n{f}:")
    print(t.loc[[h for h in (0, 1, 2, 3, 5, 10, 20) if h in t.index],
                ["groups", "typical_err", "range80_lo", "range50_lo", "range50_hi",
                 "range80_hi"]].round(1).to_string())
