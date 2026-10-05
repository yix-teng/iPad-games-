import numpy as np
import pandas as pd

from sgpf import hedonic
from sgpf.index_model import make_features, make_target


def synthetic_ura_projects(n_projects=60, n_tx=40, seed=0):
    """Projects shaped like the URA PMI_Resi_Transaction API response."""
    rng = np.random.default_rng(seed)
    projects = []
    for i in range(n_projects):
        seg = rng.choice(["CCR", "RCR", "OCR"])
        freehold = rng.random() < 0.4
        start = int(rng.integers(1995, 2022))
        tenure = "Freehold" if freehold else f"99 yrs lease commencing from {start}"
        base = {"CCR": 30000, "RCR": 22000, "OCR": 16000}[seg] * (1.15 if freehold else 1)
        txs = []
        for _ in range(n_tx):
            m = int(rng.integers(0, 60))  # months since Jan 2021
            yy, mm = 21 + m // 12, m % 12 + 1
            lo = int(rng.integers(1, 30))
            area = float(rng.uniform(45, 200))
            psm = base * (1.004 ** m) * (1 + 0.004 * lo) * rng.lognormal(0, 0.05)
            txs.append({"area": str(round(area, 1)), "floorRange": f"{lo:02d}-{lo + 4:02d}",
                        "noOfUnits": "1", "contractDate": f"{mm:02d}{yy:02d}",
                        "typeOfSale": str(rng.choice(["1", "3"])), "price": str(round(psm * area)),
                        "propertyType": "Condominium", "district": f"{rng.integers(1, 28):02d}",
                        "typeOfArea": "Strata", "tenure": tenure})
        projects.append({"project": f"P{i}", "street": "X", "marketSegment": seg,
                         "x": str(rng.uniform(1e4, 4e4)), "y": str(rng.uniform(3e4, 5e4)),
                         "transaction": txs})
    return projects


def test_tenure_and_floor_parsing():
    assert hedonic._tenure("Freehold", 2024)[0] == "freehold"
    kind, rem, start = hedonic._tenure("99 yrs lease commencing from 2015", 2024.0)
    assert (kind, rem, start) == ("99yr", 90.0, 2015.0)
    assert hedonic._floor_mid("06-10") == 8
    assert hedonic._floor_mid("B1-B5") == -3
    assert np.isnan(hedonic._floor_mid("-"))


def test_hedonic_model_learns_synthetic_prices():
    from sgpf.data import flatten_ura_projects
    df = flatten_ura_projects(synthetic_ura_projects())
    d = hedonic.prepare(df)
    assert d.sale_date.min() >= pd.Timestamp("2021-01-01")
    _, metrics, te = hedonic.train_eval(df)
    assert metrics["n_test"] > 0
    assert metrics["MAPE_pct"] < 10, metrics


def test_index_features_have_no_lookahead():
    idx = pd.period_range("2000Q1", periods=40, freq="Q")
    s = pd.Series(np.exp(np.linspace(0, 1, 40)), index=idx)
    f1 = make_features(s)
    s2 = s.copy(); s2.iloc[-1] *= 2  # perturb the last point only
    f2 = make_features(s2)
    pd.testing.assert_frame_equal(f1.iloc[:-1], f2.iloc[:-1])
    assert np.isclose(make_target(s, 4).iloc[0], np.log(s.iloc[4] / s.iloc[0]))


def test_long_run_central_path_is_income_growth():
    """Uses the cached data in data/ (committed), so runs offline."""
    from sgpf.data import fetch_price_index
    from sgpf.long_run import scenarios
    p0 = fetch_price_index()["All Residential"].dropna().iloc[-1]
    sc = scenarios({"base": 0.03}, inflation=0.02, years=10)
    ten = sc[sc.years == 10].iloc[0]
    assert np.isclose(ten.central, p0 * (1.03 * 1.02) ** 10)  # nominal: real growth + CPI
    assert np.isclose(ten.real_central, p0 * 1.03 ** 10)  # today's dollars: real growth only
    assert (sc.real_likely_lo <= sc.real_central).all()
    assert (sc.real_central <= sc.real_likely_hi).all()
    assert (sc.wide_lo <= sc.likely_lo).all() and (sc.likely_lo <= sc.central).all()
    assert (sc.central <= sc.likely_hi).all() and (sc.likely_hi <= sc.wide_hi).all()


def test_long_run_beats_flat_when_income_known():
    from sgpf.long_run import evaluate
    ev = evaluate(40)
    assert ev["income_known_MAE"] < ev["flat_MAE"]


def _synthetic_condo_sales(seed=0):
    """Projects with known effects: +2% per floor band step, -10% for 1.25k-1.5k sqft."""
    rng = np.random.default_rng(seed)
    info, tx = [], []
    floor_eff = {3: -0.02, 8: 0.0, 13: 0.02, 18: 0.04}
    for i in range(40):
        slug = f"p{i}"
        district = rng.choice(["D9", "D15", "D19"])
        info.append({"slug": slug, "Address": f"1 Road, Singapore {100000 + i}",
                     "District": district, "Tenure": "Freehold", "TOP": "2010"})
        prem = rng.normal(0, 0.2)
        for _ in range(120):
            day = pd.Timestamp("2015-01-01") + pd.Timedelta(days=int(rng.integers(0, 3600)))
            fl = int(rng.choice(list(floor_eff)))
            big = rng.random() < 0.3
            area = 1300.0 if big else 900.0
            lp = np.log(1500) + prem + 0.00012 * (day - pd.Timestamp("2015-01-01")).days \
                + floor_eff[fl] + (np.log(0.9) if big else 0) + rng.normal(0, 0.02)
            tx.append({"slug": slug, "date": day.strftime("%Y-%m-%d"), "unit": f"#{fl:02d}-0{1 + i % 5}",
                       "area_sqft": area, "psf_sgd": float(np.exp(lp)), "property_type": "Condo",
                       "sale_type": "resale", "units_sold": 1})
    return pd.DataFrame(info), pd.DataFrame(tx)


def test_unit_parsers():
    from sgpf.unit_model import parse_floor, parse_tenure, region_of
    assert parse_floor("#12-05") == 12 and parse_floor("#B1-02") == -1
    assert np.isnan(parse_floor("-"))
    kind, yrs, start = parse_tenure("99y leasehold from 15 Feb 2016")
    assert (kind, yrs, start.year) == ("leasehold", 99.0, 2016)
    assert parse_tenure("999y")[0] == "freehold" and parse_tenure("Freehold")[0] == "freehold"
    assert region_of("D10") == "CCR" and region_of("D15") == "RCR" and region_of("D19") == "OCR"


def test_transparent_model_recovers_known_effects():
    from sgpf.unit_model import TransparentModel, prepare
    info, tx = _synthetic_condo_sales()
    d = prepare(info, tx)
    A = TransparentModel().fit(d, ridge=0.01)
    floor = A.table("floor")
    assert abs(floor["16-20"] - 4.0) < 1.5 and abs(floor["11-15"] - 2.0) < 1.5
    assert abs(A.table("area")["1.25k-1.5k"] - (-10.0)) < 1.5
    pred, seen = A.predict(d.tail(200))
    assert seen.all()
