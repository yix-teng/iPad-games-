"""Data loading: URA private residential price index (data.gov.sg) and URA transactions."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# URA Private Residential Property Price Index (1975Q1 = ... , 2009Q1 = 100), quarterly.
PPI_RESOURCE_ID = "d_97f8a2e995022d311c6c68cfda6d034c"
DATAGOV_URL = "https://data.gov.sg/api/action/datastore_search"

URA_TOKEN_URL = "https://eservice.ura.gov.sg/uraDataService/insertNewToken/v1"
URA_DATA_URL = "https://eservice.ura.gov.sg/uraDataService/invokeUraDS/v1"

# Major property cooling measures (announcement month). Used as regime features.
COOLING_MEASURES = pd.to_datetime([
    "2009-09-14", "2010-02-19", "2010-08-30", "2011-01-13", "2011-12-07",
    "2012-10-05", "2013-01-11", "2013-06-28", "2018-07-05", "2021-12-15",
    "2022-09-29", "2023-04-26",
])


def fetch_price_index(refresh: bool = False) -> pd.DataFrame:
    """Return wide quarterly index: columns All Residential / Landed / Non-Landed, PeriodIndex."""
    cache = DATA_DIR / "ura_ppi.csv"
    if cache.exists() and not refresh:
        raw = pd.read_csv(cache)
    else:
        records, offset = [], 0
        while True:
            r = requests.get(DATAGOV_URL, params={
                "resource_id": PPI_RESOURCE_ID, "limit": 1000, "offset": offset}, timeout=30)
            r.raise_for_status()
            res = r.json()["result"]
            records += res["records"]
            offset += len(res["records"])
            if not res["records"] or offset >= res["total"]:
                break
        raw = pd.DataFrame(records)[["quarter", "property_type", "index"]]
        DATA_DIR.mkdir(exist_ok=True)
        raw.to_csv(cache, index=False)
    raw["index"] = pd.to_numeric(raw["index"], errors="coerce")
    wide = raw.pivot(index="quarter", columns="property_type", values="index")
    wide.index = pd.PeriodIndex(wide.index.str.replace("-", ""), freq="Q")
    return wide.sort_index().dropna(how="all")


def fetch_ura_transactions(access_key: str, batches=(1, 2, 3, 4)) -> pd.DataFrame:
    """Download the last ~5 years of private residential transactions from the URA API.

    Register for a free access key at https://eservice.ura.gov.sg/maps/api/reg.html
    """
    tok = requests.get(URA_TOKEN_URL, headers={"AccessKey": access_key}, timeout=30).json()
    headers = {"AccessKey": access_key, "Token": tok["Result"], "User-Agent": "Mozilla/5.0"}
    projects = []
    for b in batches:
        r = requests.get(URA_DATA_URL, params={"service": "PMI_Resi_Transaction", "batch": b},
                         headers=headers, timeout=120)
        r.raise_for_status()
        projects += r.json()["Result"]
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "ura_transactions.json").write_text(json.dumps(projects))
    return flatten_ura_projects(projects)


def flatten_ura_projects(projects: list[dict]) -> pd.DataFrame:
    """URA API returns one record per project with a nested list of transactions."""
    rows = []
    for p in projects:
        for t in p.get("transaction", []):
            rows.append({
                "project": p.get("project"), "street": p.get("street"),
                "market_segment": p.get("marketSegment"),
                "x": p.get("x"), "y": p.get("y"), **t,
            })
    df = pd.DataFrame(rows)
    return df.rename(columns={
        "contractDate": "contract_date", "floorRange": "floor_range",
        "propertyType": "property_type", "typeOfSale": "type_of_sale",
        "typeOfArea": "type_of_area", "noOfUnits": "no_of_units",
    })


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load a saved URA API JSON dump or a CSV with the same (flattened) columns."""
    path = Path(path)
    if path.suffix == ".json":
        return flatten_ura_projects(json.loads(path.read_text()))
    return pd.read_csv(path)
