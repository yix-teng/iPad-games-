"""Unit-level price model on propertynoob transactions, built to be explainable.

Valuation, "transparent" model A (log price per sqft):

    log psf = project premium (from its recent sales) + market level (region x quarter)
              + floor + size + EC age + sale type

Every term is a lookup table of % effects, so a valuation reads as a list of adjustments.

Ageing and lease run-down are NOT terms in this model. Within one building, age, remaining
lease and calendar time all advance one year per year, so with project premiums and market
levels in the model their linear effects are not identified (the age-period-cohort problem).
They are measured separately by `relative_performance`: how projects of a given age and
remaining lease grew compared with the median project in their region over the following
years. That is exactly the quantity a unit forecast needs on top of a market forecast.

Model B, accuracy check: LightGBM on the same information plus location (district, distance
to MRT and CBD), with per-unit contributions from LightGBM's built-in SHAP values.

Both are tested out-of-time: trained on sales up to a cutoff, scored on the following months
at the last known market level.
"""
from __future__ import annotations

import re

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

FLOOR_BINS = [-99, 2, 5, 10, 15, 20, 25, 30, 40, 200]
FLOOR_LABELS = ["1-2", "3-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-40", "41+"]
AREA_BINS = [0, 500, 650, 800, 1000, 1250, 1500, 2000, 3000, 1e9]
AREA_LABELS = ["<500", "500-650", "650-800", "800-1k", "1k-1.25k", "1.25k-1.5k", "1.5k-2k",
               "2k-3k", "3k+"]
AGE_BINS = [-99, 0, 3, 6, 10, 15, 20, 25, 30, 40, 200]
AGE_LABELS = ["pre-TOP", "0-3", "3-6", "6-10", "10-15", "15-20", "20-25", "25-30", "30-40",
              "40+"]
LEASE_BINS = [-1, 50, 60, 70, 75, 80, 85, 90, 95, 1000]
LEASE_LABELS = ["<50", "50-60", "60-70", "70-75", "75-80", "80-85", "85-90", "90-95", "95+"]
REF = {"floor": "6-10", "area": "800-1k", "sale_type": "resale"}
# Levels with no column (effect fixed at 0): non-ECs, and ECs aged 3-6 as the EC reference.
NO_COLUMN = {"ec_age": {"not EC", "3-6"}}


# Approximate URA market segments by postal district (URA's exact boundaries follow planning
# areas, which split a few districts).
CCR_DISTRICTS = {"D1", "D2", "D6", "D9", "D10", "D11"}
RCR_DISTRICTS = {"D3", "D4", "D5", "D7", "D8", "D12", "D13", "D14", "D15", "D20"}


def region_of(district) -> str:
    return "CCR" if district in CCR_DISTRICTS else "RCR" if district in RCR_DISTRICTS else "OCR"


# ----------------------------------------------------------------------------- preparation

def parse_tenure(s) -> tuple[str, float, pd.Timestamp | None]:
    s = str(s)
    m = re.match(r"(\d+)y", s)
    if "freehold" in s.lower() or (m and int(m.group(1)) >= 900):
        return "freehold", np.nan, None
    if not m:
        return "unknown", np.nan, None
    d = re.search(r"from (\d{1,2} \w{3} \d{4})", s)
    return "leasehold", float(m.group(1)), (pd.to_datetime(d.group(1)) if d else None)


def parse_floor(unit) -> float:
    m = re.match(r"#?(B)?(\d+)-", str(unit))
    if not m:
        return np.nan
    return -float(m.group(2)) if m.group(1) else float(m.group(2))


def prepare(info: pd.DataFrame, tx: pd.DataFrame, locations: pd.DataFrame | None = None):
    """Join project info onto transactions and derive features."""
    p = info.copy()
    ten = p.Tenure.map(parse_tenure)
    p["tenure_type"] = ten.str[0]
    p["lease_years"] = ten.str[1]
    p["lease_start"] = pd.to_datetime(ten.str[2])
    top = pd.to_numeric(p.TOP, errors="coerce")
    p["top_year"] = top.where(top > 1900)  # site shows 0 when unknown
    p["district"] = p.District
    p["postal"] = p.Address.str.extract(r"Singapore\s+(\d{6})")[0]
    cols = ["slug", "tenure_type", "lease_years", "lease_start", "top_year", "district", "postal"]
    d = tx.merge(p[cols], on="slug", how="left")
    if locations is not None:
        d = d.merge(locations[["postal", "lat", "lon", "mrt_km", "cbd_km"]], on="postal",
                    how="left")
    d["date"] = pd.to_datetime(d.date)
    d = d[(d.units_sold == 1) & d.psf_sgd.gt(0) & d.area_sqft.gt(0)]
    d = d[d.tenure_type.isin(["freehold", "leasehold"])]
    d["floor"] = d.unit.map(parse_floor)
    d["stack"] = d.unit.astype(str).str.extract(r"-(\w+)")[0]
    yrs = d.date.dt.year + (d.date.dt.dayofyear - 1) / 365.25
    d["age"] = yrs - d.top_year
    d["lease_left"] = np.where(d.tenure_type == "leasehold",
                               d.lease_years - (d.date - d.lease_start).dt.days / 365.25, np.nan)
    d["month"] = d.date.dt.to_period("M")
    d["quarter"] = d.date.dt.to_period("Q")
    d["region"] = d.district.map(region_of)
    # Stack (#xx-05) = same layout, facing and view within a project.
    d["slug_stack"] = d.slug + "|" + d["stack"].fillna("?").astype(str)
    d["market"] = d.region + "|" + d.quarter.astype(str)
    d["log_psf"] = np.log(d.psf_sgd)
    d["is_ec"] = (d.property_type == "EC").astype(int)
    d["sale_type"] = d.sale_type.replace({"sub": "subsale"}).fillna("unknown")
    d["floor_bin"] = pd.cut(d.floor, FLOOR_BINS, labels=FLOOR_LABELS).astype(object).fillna("unknown")
    d["area_bin"] = pd.cut(d.area_sqft, AREA_BINS, labels=AREA_LABELS).astype(object)
    d["age_bin"] = pd.cut(d.age, AGE_BINS, labels=AGE_LABELS).astype(object).fillna("unknown")
    # Executive condos: resale opens to private buyers at 5 years and they become fully private
    # at 10, so they get an extra age table on top of the general one.
    d["ec_age_bin"] = np.where(d.is_ec == 1, d.age_bin, "not EC")
    lease_bin = pd.cut(d.lease_left, LEASE_BINS, labels=LEASE_LABELS).astype(object)
    d["lease_bin"] = np.where(d.tenure_type == "leasehold", lease_bin.fillna("unknown"),
                              "freehold")
    # Drop extreme psf outliers within project (data errors, bulk/odd deals).
    z = d.groupby("slug").log_psf.transform(lambda x: (x - x.median()).abs())
    return d[z < np.log(2)].reset_index(drop=True)


# -------------------------------------------------------------------- Model A: transparent

class TransparentModel:
    """log psf = sum of lookup-table effects, in two stages.

    Stage 1 (all history, sparse least squares): project premium, market level per region and
    quarter, floor, size, EC age, sale type.

    Stage 2 (recent sales): each project's CURRENT premium = weighted average of its sales'
    residuals after removing all stage-1 effects except the project's own, weighted toward
    recent sales (half-life `half_life_years`) and shrunk toward the stage-1 premium by
    `prior_sales` pseudo-sales. A valuation therefore reads: "recent sales in this project,
    adjusted for market movement, floor, size and sale type".
    """

    TERMS = [("project", "slug"), ("market", "market"), ("stack", "slug_stack"),
             ("floor", "floor_bin"), ("area", "area_bin"), ("ec_age", "ec_age_bin"),
             ("sale_type", "sale_type")]

    def _design(self, d: pd.DataFrame, fit: bool, skip=()):
        blocks, names = [], []
        for term, col in self.TERMS:
            if term in skip:
                continue
            vals = d[col].astype(object).map(str)  # NaN -> "nan" (pandas 3 keeps NaN)
            if fit:
                levels = sorted(vals.unique())
                drop = NO_COLUMN.get(term, {REF.get(term)})
                if term == "market":  # first quarter of each region is that region's base
                    first = pd.Series(levels).groupby(pd.Series(levels).str[:3]).first()
                    drop = set(first.values)
                    self.base_market_ = first.to_dict()
                self.levels_[term] = [l for l in levels if l not in drop]
            idx = {l: i for i, l in enumerate(self.levels_[term])}
            rows = np.arange(len(d))
            cols = vals.map(idx)
            ok = cols.notna().values
            m = sp.csr_matrix((np.ones(ok.sum()), (rows[ok], cols[ok].astype(int).values)),
                              shape=(len(d), len(idx)))
            blocks.append(m)
            names += [(term, l) for l in self.levels_[term]]
        return sp.hstack(blocks).tocsr(), names

    def fit(self, d: pd.DataFrame, ridge: float = 0.3, half_life_years: float = 1.0,
            prior_sales: float = 3.0):
        self.levels_ = {}
        X, names = self._design(d, fit=True)
        self.mean_ = d.log_psf.mean()
        beta = lsqr(X, d.log_psf.values - self.mean_, damp=ridge, atol=1e-10, btol=1e-10,
                    iter_lim=50000)[0]
        self.coef_ = pd.Series(beta, index=pd.MultiIndex.from_tuples(names))
        self.last_quarter_ = d.quarter.max()
        self.region_ = d.groupby("slug").region.first()
        # Stage 2: current project premium from recent, fully-adjusted sales.
        resid = d.log_psf.values - self._log_pred(d, skip_project=True)
        age_yrs = (d.date.max() - d.date).dt.days.values / 365.25
        w = 0.5 ** (age_yrs / half_life_years)
        g = pd.DataFrame({"slug": d.slug.values, "w": w, "wr": w * resid})
        g = g.groupby("slug")[["w", "wr"]].sum()
        stage1 = self.coef_.loc["project"].reindex(g.index).fillna(0.0)
        self.premium_ = (g.wr + prior_sales * stage1) / (g.w + prior_sales)
        self.recent_weight_ = g.w
        return self

    def _log_pred(self, d, skip_project=False):
        X, _ = self._design(d, fit=False, skip=("project",))
        rest = self.coef_.drop("project", level=0)
        out = self.mean_ + X @ rest.values
        if not skip_project:
            out = out + d.slug.map(self.premium_).fillna(0.0).values
        return out

    def effect(self, term: str, level) -> float:
        try:
            return float(self.coef_.loc[(term, str(level))])
        except KeyError:
            return 0.0  # reference level (or unseen: treated as reference)

    def table(self, term: str) -> pd.Series:
        """% effect of each level vs the reference level."""
        s = self.coef_.loc[term]
        for ref in NO_COLUMN.get(term, {REF.get(term)}):
            if ref:
                s = pd.concat([s, pd.Series({ref: 0.0})])
        return (np.exp(s) - 1).mul(100).round(1)

    def market_index(self, region: str) -> pd.Series:
        """Quality-adjusted market level by quarter for a region (its base quarter = 1)."""
        s = self.coef_.loc["market"]
        s = s[s.index.str.startswith(region)]
        s = pd.concat([pd.Series({self.base_market_[region]: 0.0}), s])
        s.index = pd.PeriodIndex(s.index.str[4:], freq="Q")
        return np.exp(s.sort_index())

    def predict(self, d: pd.DataFrame, at_quarter=None):
        """(predicted log psf, project-seen mask); market level set to `at_quarter`
        (default: last training quarter) in the unit's region."""
        d = d.copy()
        d["quarter"] = pd.Period(str(at_quarter or self.last_quarter_), "Q")
        d["market"] = d.region + "|" + d.quarter.astype(str)
        return self._log_pred(d), d.slug.isin(self.premium_.index).values

    def explain(self, row: pd.Series, at_quarter=None) -> pd.DataFrame:
        at = str(at_quarter or self.last_quarter_)
        market = f"{row.region}|{at}"
        parts = [(f"project premium (recent sales) + {row.region} market level {at}",
                  self.mean_ + float(self.premium_.get(row.slug, 0.0))
                  + self.effect("market", market), None)]
        for term, col in self.TERMS[2:]:
            if term == "ec_age" and row[col] == "not EC":
                continue
            label = (f"stack #xx-{row['stack']} vs project average" if term == "stack"
                     else f"{term}: {row[col]}")
            parts.append((label, self.effect(term, row[col]), REF.get(term)))
        out = pd.DataFrame(parts, columns=["item", "log_effect", "vs"])
        out["pct"] = (np.exp(out.log_effect) - 1) * 100
        out.loc[0, "pct"] = np.nan
        out["psf_after"] = np.exp(out.log_effect.cumsum())
        return out


# ------------------------------------------------- ageing & lease: relative performance

REL_AGE_BINS = [0, 5, 10, 15, 20, 30, 200]
REL_AGE_LABELS = ["0-5", "5-10", "10-15", "15-20", "20-30", "30+"]
REL_LEASE_BINS = [0, 60, 70, 80, 90, 1000]
REL_LEASE_LABELS = ["<60", "60-70", "70-80", "80-90", "90+"]


def project_year_levels(d: pd.DataFrame, model: TransparentModel, min_sales: int = 3):
    """Median price level per project and year, after removing floor, size, EC-age and
    sale-type effects (so a year with more high floors doesn't look like growth)."""
    adj = d.log_psf.copy()
    for term, col in model.TERMS[2:]:
        adj -= d[col].astype(object).map(str).map(lambda l, t=term: model.effect(t, l))
    g = d.assign(adj=adj, year=d.date.dt.year).groupby(["slug", "year"])
    lv = g.agg(level=("adj", "median"), n=("adj", "size"), age=("age", "median"),
               lease_left=("lease_left", "median"), region=("region", "first"),
               tenure=("tenure_type", "first"), is_ec=("is_ec", "first")).reset_index()
    return lv[lv.n >= min_sales]


def relative_performance(d: pd.DataFrame, model: TransparentModel, span: int = 5):
    """Per project: growth over `span` years minus the median project's growth in the same
    region and years, in % per year. Returns (pairs, table by tenure x age x lease)."""
    lv = project_year_levels(d, model)
    nxt = lv[["slug", "year", "level"]].assign(year=lambda x: x.year - span)
    pairs = lv.merge(nxt, on=["slug", "year"], suffixes=("", "_end"))
    pairs["growth"] = pairs.level_end - pairs.level
    bench = pairs.groupby(["region", "year"]).growth.transform("median")
    pairs["rel_pa"] = (np.exp((pairs.growth - bench) / span) - 1) * 100
    pairs = pairs[(pairs.age >= 0) & (pairs.is_ec == 0)]  # completed, non-EC projects
    pairs["age_bin"] = pd.cut(pairs.age, REL_AGE_BINS, labels=REL_AGE_LABELS)
    pairs["lease_bin"] = np.where(
        pairs.tenure == "leasehold",
        pd.cut(pairs.lease_left, REL_LEASE_BINS, labels=REL_LEASE_LABELS).astype(object),
        "freehold")
    tab = (pairs.groupby(["tenure", "age_bin", "lease_bin"], observed=True).rel_pa
           .agg(["mean", "median", "count", "std"]))
    tab["projects"] = pairs.groupby(["tenure", "age_bin", "lease_bin"], observed=True) \
        .slug.nunique()
    tab["se"] = tab["std"] / np.sqrt(tab.projects.clip(lower=1))  # projects, not pairs
    return pairs, tab.round(2)


# ---------------------------------------------------------------- Model B: accuracy check

GBM_CAT = ["slug", "district", "region", "tenure_type", "sale_type"]
GBM_NUM = ["t", "floor", "area_sqft", "age", "lease_left", "is_ec", "lat", "lon", "mrt_km",
           "cbd_km"]


def _gbm_frame(d, cats=None):
    X = pd.DataFrame(index=d.index)
    X["t"] = (d.month.dt.year - 1990) * 12 + d.month.dt.month
    for c in GBM_NUM:
        if c != "t":
            X[c] = d[c] if c in d else np.nan
    for c in GBM_CAT:
        X[c] = pd.Categorical(d[c].astype(str), categories=None if cats is None else cats[c])
    return X


def fit_gbm(train: pd.DataFrame, valid: pd.DataFrame, seed: int = 0):
    Xtr = _gbm_frame(train)
    cats = {c: Xtr[c].cat.categories for c in GBM_CAT}
    Xva = _gbm_frame(valid, cats)
    m = lgb.LGBMRegressor(n_estimators=4000, learning_rate=0.05, num_leaves=127,
                          min_child_samples=20, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, cat_smooth=10, random_state=seed, verbose=-1)
    m.fit(Xtr, train.log_psf, eval_set=[(Xva, valid.log_psf)],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    m.cats_ = cats
    return m


def predict_gbm(m, d: pd.DataFrame, at_month=None):
    d = d.copy()
    if at_month is not None:
        d["month"] = pd.Period(at_month, "M")
    return m.predict(_gbm_frame(d, m.cats_))


# --------------------------------------------------------------------------- evaluation

def errors(log_pred, log_act) -> dict:
    ape = np.abs(np.exp(log_pred - log_act) - 1)
    return {"n": int(len(ape)), "median_err_pct": round(float(np.median(ape)) * 100, 2),
            "mean_err_pct": round(float(ape.mean()) * 100, 2),
            "within_5pct": round(float((ape < 0.05).mean()), 3),
            "within_10pct": round(float((ape < 0.10).mean()), 3),
            "p90_err_pct": round(float(np.quantile(ape, 0.9)) * 100, 2)}


RULES = {
    "transparent": lambda f: f.pred_A,
    "lightgbm": lambda f: f.pred_B,
    "average": lambda f: (f.pred_A + f.pred_B) / 2,
    "transparent for new sales & ECs, lightgbm for resales": lambda f: np.where(
        (f.sale_type == "new") | (f.is_ec == 1), f.pred_A, f.pred_B),
}


def _fit_predict(train: pd.DataFrame, target: pd.DataFrame, valid_months: int = 6):
    cut = train.month.max()
    A = TransparentModel().fit(train)
    B = fit_gbm(train[train.month <= cut - valid_months], train[train.month > cut - valid_months])
    pa, seen = A.predict(target)
    pb = predict_gbm(B, target, at_month=str(cut))
    return A, B, target.assign(pred_A=pa, pred_B=pb, project_seen=seen)


def evaluate(d: pd.DataFrame, test_months: int = 6, select_months: int = 6):
    """Out-of-time test with honest model selection.

    1. Selection window: train up to (cutoff - select_months), score the next select_months,
       pick the rule (transparent / lightgbm / average / split by sale type) with the lowest
       median error on resales.
    2. Test window: retrain up to the cutoff, score the last `test_months` (never used for any
       choice) and report every rule, flagging the selected one.
    """
    last = d.month.max()
    cut = last - test_months
    scut = cut - select_months
    _, _, sel = _fit_predict(d[d.month <= scut], d[(d.month > scut) & (d.month <= cut)])
    sel = sel[sel.project_seen & (sel.sale_type == "resale")]
    sel_scores = {k: errors(f(sel), sel.log_psf.values)["median_err_pct"]
                  for k, f in RULES.items()}
    chosen = min(sel_scores, key=sel_scores.get)
    A, B, test = _fit_predict(d[d.month <= cut], d[d.month > cut])
    test = test[test.project_seen].copy()
    test["pred"] = RULES[chosen](test)
    res = {"cutoff": str(cut), "test_from": str(cut + 1), "test_to": str(last),
           "selection_window": f"{scut + 1}..{cut}",
           "selection_median_err_resale": sel_scores, "chosen_rule": chosen}
    for name, f in RULES.items():
        p = f(test)
        res[name] = {"all": errors(p, test.log_psf.values),
                     **{k: errors(np.asarray(p)[(test.sale_type == k).values],
                                  g.log_psf.values) for k, g in test.groupby("sale_type")}}
    resale = test[test.sale_type == "resale"]
    q = np.quantile(resale.log_psf - resale.pred, [0.1, 0.9])
    res["chosen_resale_log_resid_q10"], res["chosen_resale_log_resid_q90"] = map(float, q)
    return res, A, B, test


GBM_GROUPS = {"project & location": ["slug", "district", "region", "lat", "lon", "mrt_km",
                                     "cbd_km"],
              "market level now vs 1995-2026 average": ["t"], "floor": ["floor"], "size": ["area_sqft"],
              "building age": ["age"], "lease left": ["lease_left"],
              "tenure": ["tenure_type"], "EC status": ["is_ec"], "sale type": ["sale_type"]}


def explain_gbm(m, row: pd.Series) -> pd.DataFrame:
    """LightGBM's per-feature contributions (SHAP) for one unit, grouped and in % terms.
    The baseline is the model's average log psf; each group multiplies it."""
    X = _gbm_frame(pd.DataFrame([row]), m.cats_)
    contrib = m.predict(X, pred_contrib=True)[0]
    names = list(X.columns)
    base = contrib[-1]
    rows = [("average condo in training data", base)]
    for label, cols in GBM_GROUPS.items():
        rows.append((label, sum(contrib[names.index(c)] for c in cols if c in names)))
    out = pd.DataFrame(rows, columns=["item", "log_effect"])
    out["pct"] = (np.exp(out.log_effect) - 1) * 100
    out.loc[0, "pct"] = np.nan
    out["psf_after"] = np.exp(out.log_effect.cumsum())
    return out
