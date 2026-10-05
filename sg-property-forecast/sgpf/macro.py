"""Macro / housing-market driver data from data.gov.sg (SingStat & URA tables, no key needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from .data import DATA_DIR, DATAGOV_URL

TABLES = {
    # name: (resource_id, {series label in table: column name})
    "gdp": ("d_08707057660b06abaf120dd0c6c520fb", {"GDP At Current Market Prices": "gdp_nominal"}),
    "rent": ("d_647dabde09e726b9eeb75e4d9cd96699",
             {"Rental Index Of Residential Properties": "rent_index"}),
    "stock": ("d_01e3556fb916ca19a7e29fc39520fa78", {
        "All Types Private Residential Properties Available": "stock_units",
        "All Types Private Residential Properties Vacant": "vacant_units"}),
    "pipeline": ("d_055b6549444dedb341c50805d9682a41", {
        "Total Landed Properties": "pipeline_landed",
        "Total Non-Landed Properties": "pipeline_nonlanded"}),
    "population": ("d_6c26f6181f2e5dd62bf3210fa1029074", {"Total Population": "population"}),
    "cpi": ("d_bdaff844e3ef89d39fceb962ff8f0791", {"All Items": "cpi"}),
    "rates": ("d_5fe5a4bb4a1ecc4d8a56a095832e2b24", {
        "Government Securities - 10-Year Bond Yield": "sgs_10y"}),
}


def _fetch_wide(resource_id: str) -> pd.DataFrame:
    r = requests.get(DATAGOV_URL, params={"resource_id": resource_id, "limit": 500}, timeout=60)
    r.raise_for_status()
    return pd.DataFrame(r.json()["result"]["records"])


def _to_series(row: pd.Series) -> pd.Series:
    """SingStat wide rows have period columns like '20262Q', '2025' or '2026Jul'."""
    vals = {}
    for col, v in row.items():
        if col in ("DataSeries", "_id"):
            continue
        num = pd.to_numeric(v, errors="coerce")
        if pd.isna(num):
            continue
        if col.endswith("Q"):
            p = pd.Period(f"{col[:4]}Q{col[4]}", "Q")
        elif len(col) == 4:
            p = pd.Period(f"{col}Q2", "Q")  # annual end-June figures -> Q2
        else:
            p = pd.Period(pd.to_datetime(col, format="%Y%b"), "M").asfreq("Q")
        vals[p] = num
    return pd.Series(vals).sort_index()


def fetch_macro(refresh: bool = False) -> pd.DataFrame:
    """Quarterly frame of driver series (PeriodIndex)."""
    cache = DATA_DIR / "macro.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, index_col=0)
        df.index = pd.PeriodIndex(df.index, freq="Q")
        return df
    cols = {}
    for rid, labels in TABLES.values():
        wide = _fetch_wide(rid)
        for label, name in labels.items():
            # Several tables repeat labels for sub-groups; the first match is the headline.
            row = wide[wide.DataSeries.str.strip() == label].iloc[0]
            s = _to_series(row)
            cols[name] = s.groupby(level=0).last()  # monthly -> end of quarter
    df = pd.DataFrame(cols).sort_index()
    df.index = pd.PeriodIndex(df.index, freq="Q")
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(cache)
    return df
