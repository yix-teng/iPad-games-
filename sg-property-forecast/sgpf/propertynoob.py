"""Collect condo sale transactions and project details from propertynoob.com.

Per condo, two requests: the project page (address, district, tenure, TOP, units) and the
sales JSON feed the site's own pages load (/data/v1/condos/<slug>/sales.json).

The feed is labelled `"source": "huttons"`; the data is for personal analysis and is kept
out of git (see .gitignore). Requests are spaced out to be gentle on the site.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from .data import DATA_DIR

BASE = "https://propertynoob.com"
OUT = DATA_DIR / "propertynoob"
UA = {"User-Agent": "Mozilla/5.0 (personal property research; low request rate)"}
DELAY_S = 1.0
INFO_FIELDS = ["Address", "District", "Developer", "Tenure", "TOP", "Number of Units",
               "Highest floor", "Area (sqm)"]


def condo_slugs() -> list[str]:
    xml = requests.get(f"{BASE}/sitemap.xml", headers=UA, timeout=60).text
    return sorted(set(re.findall(r"<loc>https://propertynoob\.com/condo/([^/<]+)/sales/?</loc>", xml)))


def parse_info(html: str) -> dict:
    body = re.sub(r"<script.*?</script>|<style.*?</style>", "", html, flags=re.S)
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)).replace("&amp;", "&")
    seg = text[text.find("Condo Info"):]
    seg = seg[:seg.find("Transactions")] if "Transactions" in seg else seg[:800]
    info = {}
    for i, f in enumerate(INFO_FIELDS):
        nxt = "|".join(re.escape(x) for x in INFO_FIELDS[i + 1:]) or "$"
        m = re.search(re.escape(f) + r"\s+(.*?)\s*(?=" + nxt + r"|$)", seg)
        info[f] = m.group(1).strip() if m else None
    return info


def _get(session, url):
    for attempt in range(4):
        try:
            r = session.get(url, headers=UA, timeout=60)
            if r.status_code in (200, 404):
                return r
        except requests.RequestException:
            pass
        time.sleep(2 ** (attempt + 1))
    return None


def collect(limit: int | None = None) -> None:
    """Resumable: skips condos already saved."""
    (OUT / "sales").mkdir(parents=True, exist_ok=True)
    info_path = OUT / "info.jsonl"
    done = set()
    if info_path.exists():
        done = {json.loads(l)["slug"] for l in info_path.read_text().splitlines() if l.strip()}
    slugs = [s for s in condo_slugs() if s not in done][:limit]
    print(f"{len(done)} done, {len(slugs)} to fetch", flush=True)
    s = requests.Session()
    for i, slug in enumerate(slugs, 1):
        page = _get(s, f"{BASE}/condo/{slug}/")
        time.sleep(DELAY_S)
        sales = _get(s, f"{BASE}/data/v1/condos/{slug}/sales.json")
        time.sleep(DELAY_S)
        if page is None or sales is None:
            print(f"  failed {slug}", flush=True)
            continue
        info = parse_info(page.text) if page.status_code == 200 else {}
        n = 0
        if sales.status_code == 200:
            (OUT / "sales" / f"{slug}.json").write_text(sales.text)
            n = sales.json().get("row_count", 0)
        with info_path.open("a") as f:
            f.write(json.dumps({"slug": slug, "sales_status": sales.status_code,
                                "n_sales": n, **info}) + "\n")
        if i % 50 == 0 or i == len(slugs):
            print(f"  {i}/{len(slugs)} {slug} ({n} sales)", flush=True)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (projects, transactions) frames from the collected files."""
    info = pd.read_json(OUT / "info.jsonl", lines=True)
    rows = []
    for p in sorted((OUT / "sales").glob("*.json")):
        d = json.loads(p.read_text())
        for t in d.get("transactions", []):
            rows.append({"slug": d["project_slug"], **t})
    return info, pd.DataFrame(rows)


if __name__ == "__main__":
    collect(int(sys.argv[1]) if len(sys.argv) > 1 else None)
