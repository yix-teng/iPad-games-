"""Location features: geocode postal codes (OneMap) and distance to nearest MRT/LRT exit (LTA)."""
from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd
import requests

from .data import DATA_DIR

ONEMAP = "https://www.onemap.gov.sg/api/common/elastic/search"
MRT_EXITS_ID = "d_b39d3a0871985372d7e1637193335da5"  # LTA MRT Station Exit (GeoJSON)
# Downtown core reference point (Raffles Place MRT) for distance-to-CBD.
CBD_LATLON = (1.2840, 103.8514)


def postal_from_address(addr: str | None) -> str | None:
    m = re.search(r"Singapore\s+(\d{6})", str(addr))
    return m.group(1) if m else None


def _onemap(query: str):
    r = requests.get(ONEMAP, params={"searchVal": query, "returnGeom": "Y",
                                     "getAddrDetails": "Y", "pageNum": 1}, timeout=30)
    res = r.json().get("results", [])
    return [float(res[0]["LATITUDE"]), float(res[0]["LONGITUDE"])] if res else None


def geocode_postals(postals, delay_s: float = 0.25, addresses: dict | None = None,
                    retry_missing: bool = False) -> pd.DataFrame:
    """Cached OneMap lookups -> DataFrame(postal, lat, lon). Postals that OneMap does not
    know (e.g. new projects) are retried by street address when `addresses` is given."""
    cache = DATA_DIR / "geocode.json"
    known = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [p for p in sorted(set(postals)) if p and (p not in known or
                                                    (retry_missing and known[p] is None))]
    for i, p in enumerate(todo, 1):
        try:
            hit = _onemap(p)
            if hit is None and addresses and addresses.get(p):
                street = re.sub(r",?\s*Singapore\s+\d{6}", "", addresses[p]).strip()
                hit = _onemap(street)
            known[p] = hit
        except (requests.RequestException, ValueError, KeyError):
            continue
        time.sleep(delay_s)
        if i % 200 == 0:
            cache.write_text(json.dumps(known))
    DATA_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps(known))
    rows = [(p, *v) for p, v in known.items() if v]
    return pd.DataFrame(rows, columns=["postal", "lat", "lon"])


def mrt_exits() -> pd.DataFrame:
    cache = DATA_DIR / "mrt_exits.geojson"
    if not cache.exists():
        url = requests.get(f"https://api-open.data.gov.sg/v1/public/api/datasets/{MRT_EXITS_ID}"
                           "/poll-download", timeout=30).json()["data"]["url"]
        cache.write_text(requests.get(url, timeout=60).text)
    gj = json.loads(cache.read_text())
    rows = []
    for f in gj["features"]:
        lon, lat = f["geometry"]["coordinates"][:2]
        name = re.search(r"<th>STATION_NA</th>\s*<td>([^<]+)", str(f["properties"]))
        rows.append({"station": (name.group(1) if name else
                                 f["properties"].get("STATION_NA")), "lat": lat, "lon": lon})
    return pd.DataFrame(rows)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def location_features(points: pd.DataFrame) -> pd.DataFrame:
    """points: columns lat, lon -> adds mrt_km (nearest exit) and cbd_km."""
    ex = mrt_exits()
    d = haversine_km(points.lat.values[:, None], points.lon.values[:, None],
                     ex.lat.values[None, :], ex.lon.values[None, :])
    out = points.copy()
    out["mrt_km"] = d.min(axis=1)
    out["cbd_km"] = haversine_km(points.lat, points.lon, *CBD_LATLON)
    return out
