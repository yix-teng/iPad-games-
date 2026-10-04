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
