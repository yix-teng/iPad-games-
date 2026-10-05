"""Out-of-sample test on projects the models have never seen.

For each project: remove it from the data entirely, rebuild the models with sales up to a
2016 start date, price its units as an unseen project (comparable launches within 3 km, as
the forecast tool does for projects with no transactions; LightGBM shown for comparison),
grow them with the market path known at the previous quarter end, and compare with the
actual resale prices of the same units in 2025-26.

    python run_holdout_test.py                       # default projects below
    python run_holdout_test.py lake-grande 2016-07-21

Unit-level results go to data/propertynoob/ (not committed); the summary to outputs/.
"""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sgpf.launch import base_log_psf, comparables, effects
from sgpf.unit_backtest import _market_fn
from sgpf.unit_forecast import ARTIFACTS
from sgpf.unit_model import TransparentModel, fit_gbm, predict_gbm

warnings.filterwarnings("ignore")
CASES = {"lake-grande": "2016-07-21", "oue-twin-peaks": "2016-06-30"}
if len(sys.argv) > 2:
    CASES = {sys.argv[1]: sys.argv[2]}

d = pickle.loads(ARTIFACTS.read_bytes())["data"]
d["uid"] = d.slug + "|" + d.unit.astype(str)
lt = pd.read_csv("outputs/launch_forecast_backtest.csv")
lt = lt[(lt.method == "launch, condo/apartment") & (lt.groups >= 200)].set_index("years")

summary = []
for P, Ts in CASES.items():
    T = pd.Timestamp(Ts)
    hist = d[(d.date <= T) & (d.slug != P)]  # the project is never seen
    A = TransparentModel().fit(hist)
    vcut = hist.month.max() - 6
    B = fit_gbm(hist[hist.month <= vcut], hist[hist.month > vcut])
    proj = d[d.slug == P]
    p0 = proj.iloc[0]
    q = pd.Period(T, "Q") - 1  # last quarter published by T
    market, g1, ginc = _market_fn(q.end_time.normalize())
    Tq = q.end_time.normalize()
    early = proj[proj.date.dt.year == T.year].sort_values("date").groupby("uid").first()
    late = (proj[(proj.date >= "2025-07-01") & (proj.sale_type == "resale")]
            .sort_values("date").groupby("uid").last())
    units = early.join(late[["date", "psf_sgd", "area_sqft"]], rsuffix="_late", how="inner")
    comps = comparables(A, hist, p0.lat, p0.lon, p0.district, p0.tenure_type, p0.is_ec, T)
    base = base_log_psf(comps)
    rows = []
    for uid, u in units.iterrows():
        f = pd.DataFrame([u]).assign(ec_age_bin="not EC")
        start = base + effects(A, f)[0]  # price at T for the same kind of deal (new/resale)
        resale_eq = base + effects(A, f.assign(sale_type="resale"))[0]
        x = (u.date_late - Tq).days / 365.25
        pred_late = resale_eq + np.log(market(x))
        b = lt.loc[lt.index[np.argmin(np.abs(lt.index - round(x)))]]
        gbm = predict_gbm(B, f.assign(slug="__unseen__"), at_month=str(hist.month.max()))[0]
        rows.append({"project": P, "unit": uid.split("|")[1], "sqft": u.area_sqft,
                     "start_sale_type": u.sale_type, "start_date": u.date,
                     "start_actual_psf": u.psf_sgd, "start_pred_psf": np.exp(start),
                     "start_lgbm_psf": np.exp(gbm), "late_date": u.date_late, "years": x,
                     "late_actual_psf": u.psf_sgd_late, "late_pred_psf": np.exp(pred_late),
                     "range_lo_psf": np.exp(pred_late) * (1 + b.range80_lo / 100),
                     "range_hi_psf": np.exp(pred_late) * (1 + b.range80_hi / 100)})
    r = pd.DataFrame(rows)
    r["err_start"] = r.start_pred_psf / r.start_actual_psf - 1
    r["err_start_lgbm"] = r.start_lgbm_psf / r.start_actual_psf - 1
    r["err_late"] = r.late_pred_psf / r.late_actual_psf - 1
    r["in_range"] = r.late_actual_psf.between(r.range_lo_psf, r.range_hi_psf)
    r.to_csv(f"data/propertynoob/holdout_{P}.csv", index=False)
    summary.append({
        "project": P, "start": Ts, "tenure": p0.tenure_type, "top": int(p0.top_year),
        "region": p0.region, "district": p0.district, "units": len(r),
        "comparables": f"{len(comps)} ({comps.how.iloc[0]})",
        "market_1y_pct": g1 * 100, "income_growth_pa_pct": ginc * 100,
        "start_typical_err_pct": r.err_start.abs().median() * 100,
        "start_bias_pct": r.err_start.median() * 100,
        "start_lgbm_typical_err_pct": r.err_start_lgbm.abs().median() * 100,
        "years_ahead": r.years.median(),
        "late_typical_err_pct": r.err_late.abs().median() * 100,
        "late_bias_pct": r.err_late.median() * 100,
        "late_err_p10_pct": r.err_late.quantile(0.1) * 100,
        "late_err_p90_pct": r.err_late.quantile(0.9) * 100,
        "in_80pct_range": r.in_range.mean(),
        "actual_change_pct": ((r.late_actual_psf / r.start_actual_psf) - 1).median() * 100})
s = pd.DataFrame(summary).set_index("project")
out = Path("outputs/holdout_test_summary.csv")
if out.exists() and len(CASES) == 1:  # add or replace a single project's row
    s = pd.concat([pd.read_csv(out, index_col="project").drop(s.index, errors="ignore"), s])
s.round(2).to_csv(out)
pd.set_option("display.width", 200)
print(s.round(1).T.to_string())
