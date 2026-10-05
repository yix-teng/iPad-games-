"""Step 2: project-level features known at a start date, and a walk-forward test of whether
they predict how units deviate from the market path.

Features (all as of the start date T, per project):
* supply_new_1km_2y   new-launch sales in OTHER projects within 1 km in the 2 years to T
* pipeline_units_1km  units in other projects within 1 km that had launched but not completed
* mrt_km_asof         distance to the nearest MRT/LRT exit open at T
* new_mrt_800m_5y     1 if a station within 800 m opens in the 5 years after T
* old_freehold_lowrise  en-bloc proxy: freehold, 25+ years old, at most 10 storeys
* age, lease_left, units (development size), cbd_km
* momentum_3y         project's price change over the 3 years to T minus its region's median

Station opening years are hand-coded below for stations opened from 1999 (earliest line at
interchanges); all others are treated as open before the first backtest start (1997).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .geo import haversine_km, mrt_exits

_OPENED = {
    1999: "BUKIT PANJANG LRT,CHOA CHU KANG LRT,SOUTH VIEW LRT,KEAT HONG LRT,TECK WHYE LRT,"
          "PHOENIX LRT,BANGKIT LRT,FAJAR LRT,SEGAR LRT,JELAPANG LRT,SENJA LRT,PETIR LRT,"
          "PENDING LRT",
    2001: "EXPO MRT,DOVER MRT",
    2002: "CHANGI AIRPORT MRT",
    2003: "HARBOURFRONT MRT,CHINATOWN MRT,CLARKE QUAY MRT,LITTLE INDIA MRT,FARRER PARK MRT,"
          "BOON KENG MRT,POTONG PASIR MRT,SERANGOON MRT,KOVAN MRT,HOUGANG MRT,SENGKANG MRT,"
          "PUNGGOL MRT,COMPASSVALE LRT,RUMBIA LRT,BAKAU LRT,KANGKAR LRT,RANGGUNG LRT",
    2005: "CHENG LIM LRT,FARMWAY LRT,KUPANG LRT,THANGGAM LRT,FERNVALE LRT,LAYAR LRT,"
          "TONGKANG LRT,RENJONG LRT,COVE LRT,MERIDIAN LRT,CORAL EDGE LRT,RIVIERA LRT,"
          "KADALOOR LRT,OASIS LRT,DAMAI LRT",
    2006: "BUANGKOK MRT",
    2007: "SOO TECK LRT",
    2009: "BARTLEY MRT,LORONG CHUAN MRT,MARYMOUNT MRT,PIONEER MRT,JOO KOON MRT",
    2010: "BRAS BASAH MRT,ESPLANADE MRT,PROMENADE MRT,NICOLL HIGHWAY MRT,STADIUM MRT,"
          "MOUNTBATTEN MRT,DAKOTA MRT,TAI SENG MRT,CC9",
    2011: "CALDECOTT MRT,BOTANIC GARDENS MRT,FARRER ROAD MRT,HOLLAND VILLAGE MRT,"
          "ONE-NORTH MRT,KENT RIDGE MRT,HAW PAR VILLA MRT,PASIR PANJANG MRT,"
          "LABRADOR PARK MRT,TELOK BLANGAH MRT,WOODLEIGH MRT",
    2012: "BAYFRONT MRT",
    2013: "DOWNTOWN MRT,TELOK AYER MRT,DT18",
    2014: "MARINA SOUTH PIER MRT,SAM KEE LRT,PUNGGOL POINT LRT,SAMUDERA LRT",
    2015: "CASHEW MRT,HILLVIEW MRT,BEAUTY WORLD MRT,KING ALBERT PARK MRT,SIXTH AVENUE MRT,"
          "TAN KAH KEE MRT,STEVENS MRT,ROCHOR MRT,BUKIT PANJANG MRT",
    2016: "NIBONG LRT,SUMANG LRT",
    2017: "FORT CANNING MRT,BENCOOLEN MRT,JALAN BESAR MRT,BENDEMEER MRT,GEYLANG BAHRU MRT,"
          "MATTAR MRT,UBI MRT,KAKI BUKIT MRT,BEDOK NORTH MRT,BEDOK RESERVOIR MRT,"
          "TAMPINES WEST MRT,TAMPINES EAST MRT,UPPER CHANGI MRT,GUL CIRCLE MRT,"
          "TUAS CRESCENT MRT,TUAS WEST ROAD MRT,TUAS LINK MRT",
    2019: "CANBERRA MRT",
    2020: "WOODLANDS NORTH MRT,WOODLANDS SOUTH MRT",
    2021: "SPRINGLEAF MRT,LENTOR MRT,MAYFLOWER MRT,BRIGHT HILL MRT,UPPER THOMSON MRT",
    2022: "NAPIER MRT,ORCHARD BOULEVARD MRT,GREAT WORLD MRT,HAVELOCK MRT,MAXWELL MRT,"
          "SHENTON WAY MRT,GARDENS BY THE BAY MRT",
    2024: "TANJONG RHU MRT,KATONG PARK MRT,TANJONG KATONG MRT,MARINE PARADE MRT,"
          "MARINE TERRACE MRT,SIGLAP MRT,BAYSHORE MRT,NE18",
    2025: "DT4",
    2026: "CC30,CC31,CC32",
}
STATION_OPENED = {n.strip(): y for y, names in _OPENED.items() for n in names.split(",")}


def station_open_year(name: str) -> int:
    key = re.sub(r" STATION$", "", str(name))
    return STATION_OPENED.get(key, 1990)


def project_table(info: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """One row per project with static attributes and first-sale date."""
    g = d.groupby("slug").agg(lat=("lat", "first"), lon=("lon", "first"),
                              cbd_km=("cbd_km", "first"), region=("region", "first"),
                              tenure=("tenure_type", "first"), top_year=("top_year", "first"),
                              first_sale=("date", "min"))
    i = info.set_index("slug")
    g["units"] = pd.to_numeric(i["Number of Units"].astype(str).str.replace(",", ""),
                               errors="coerce").reindex(g.index)
    g["max_floor"] = pd.to_numeric(i["Highest floor"], errors="coerce").reindex(g.index)
    return g


def features_asof(proj: pd.DataFrame, d: pd.DataFrame, year: int,
                  levels: pd.DataFrame) -> pd.DataFrame:
    """Feature table (index slug) as known at the end of `year`."""
    T = pd.Timestamp(f"{year}-12-31")
    p = proj[proj.first_sale <= T].copy()
    ok = p.lat.notna()
    f = pd.DataFrame(index=p.index)
    # Pairwise project distances (km); a project is not its own neighbour.
    lat, lon = p.lat.fillna(0).values, p.lon.fillna(0).values
    dist = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    near = (dist <= 1.0) & ok.values[:, None] & ok.values[None, :]
    np.fill_diagonal(near, False)
    new_sales = d[(d.sale_type == "new") & (d.date > T - pd.DateOffset(years=2)) & (d.date <= T)]
    cnt = new_sales.groupby("slug").size().reindex(p.index).fillna(0).values
    f["supply_new_1km_2y"] = near @ cnt
    under_constr = ((p.top_year > year) | p.top_year.isna()).values * p.units.fillna(0).values
    f["pipeline_units_1km"] = near @ under_constr
    ex = mrt_exits()
    ex["opened"] = ex.station.map(station_open_year)
    open_now, open_soon = ex[ex.opened <= year], ex[(ex.opened > year) & (ex.opened <= year + 5)]
    def nearest(e):
        if e.empty:
            return np.full(len(p), np.inf)
        return haversine_km(lat[:, None], lon[:, None], e.lat.values[None, :],
                            e.lon.values[None, :]).min(axis=1)
    f["mrt_km_asof"] = np.where(ok, nearest(open_now), np.nan)
    f["new_mrt_800m_5y"] = np.where(ok, (nearest(open_soon) <= 0.8).astype(float), np.nan)
    age = year - p.top_year
    f["age"] = age
    f["old_freehold_lowrise"] = ((p.tenure == "freehold") & (age >= 25)
                                 & (p.max_floor <= 10)).astype(float)
    f["log_units"] = np.log1p(p.units)
    f["cbd_km"] = p.cbd_km
    f["freehold"] = (p.tenure == "freehold").astype(float)
    # Momentum: project level change over the 3 years to T minus the region median.
    lv = levels[(levels.year <= year) & (levels.year >= year - 3)]
    a = lv[lv.year == year - 3].set_index("slug").level
    b = lv[lv.year == year].set_index("slug").level
    chg = (b - a).dropna()
    reg = p.region.reindex(chg.index)
    f["momentum_3y"] = (chg - chg.groupby(reg).transform("median")).reindex(p.index)
    return f


FEATURES = ["supply_new_1km_2y", "pipeline_units_1km", "mrt_km_asof", "new_mrt_800m_5y",
            "age", "old_freehold_lowrise", "log_units", "cbd_km", "freehold", "momentum_3y"]


def design(df: pd.DataFrame, stats=None, per_year: bool = True):
    """Standardised features, plus (if `per_year`) their interaction with years ahead."""
    X = df[FEATURES].astype(float).copy()
    X["supply_new_1km_2y"] = np.log1p(X.supply_new_1km_2y)
    X["pipeline_units_1km"] = np.log1p(X.pipeline_units_1km)
    X["mrt_km_asof"] = np.minimum(X.mrt_km_asof, 3.0)
    if stats is None:
        stats = (X.mean(), X.std().replace(0, 1))
    Z = ((X - stats[0]) / stats[1]).fillna(0.0)  # missing -> average
    names = [f"{c} (level)" for c in FEATURES]
    if not per_year:
        return Z.values, names, stats
    yrs = df.years.values[:, None]
    M = np.hstack([Z.values, Z.values * yrs])
    return M, names + [f"{c} (per year)" for c in FEATURES], stats
