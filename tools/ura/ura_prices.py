#!/usr/bin/env python3
"""
URA private residential transaction fetcher + price benchmarker.

Reads the API key from the URA_ACCESS_KEY environment variable.

Usage:
  export URA_ACCESS_KEY=...
  python3 ura_prices.py                       # download + cache all 4 batches
  python3 ura_prices.py --project "THE ARDEN"
  python3 ura_prices.py --price 1988000 --sqft-min 700 --sqft-max 850
"""
import argparse, json, os, sys, time, statistics, urllib.request, urllib.error
from collections import defaultdict

BASE = "https://eservice.ura.gov.sg/uraDataService"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ura_cache.json")
SQM_TO_SQFT = 10.7639104


def _get(url, headers, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read()
                # URA occasionally emits Latin-1 bytes (accented project names)
                for enc in ("utf-8", "cp1252", "latin-1"):
                    try:
                        return json.loads(raw.decode(enc))
                    except UnicodeDecodeError:
                        continue
                return json.loads(raw.decode("utf-8", "replace"))
        except Exception as e:                       # noqa: BLE001
            if i == retries - 1:
                raise
            wait = 2 ** (i + 1)
            print(f"  retry {i+1} after {wait}s ({e})", file=sys.stderr)
            time.sleep(wait)


def get_token(key):
    d = _get(f"{BASE}/insertNewToken/v1", {"AccessKey": key, "User-Agent": UA})
    if d.get("Status") != "Success":
        raise RuntimeError(f"token failed: {d}")
    return d["Result"]


def get_batch(key, token, batch):
    url = f"{BASE}/invokeUraDS/v1?service=PMI_Resi_Transaction&batch={batch}"
    d = _get(url, {"AccessKey": key, "Token": token, "User-Agent": UA})
    if d.get("Status") != "Success":
        raise RuntimeError(f"batch {batch} failed: {d.get('Message')}")
    return d.get("Result", [])


def download(key):
    token = get_token(key)
    print(f"Token acquired ({len(token)} chars)")
    projects = []
    for b in (1, 2, 3, 4):
        print(f"Fetching batch {b} ...", flush=True)
        r = get_batch(key, token, b)
        print(f"  batch {b}: {len(r)} projects")
        projects.extend(r)
        time.sleep(1)
    with open(CACHE, "w") as f:
        json.dump(projects, f)
    print(f"Cached {len(projects)} project records -> {CACHE}")
    return projects


def load(key=None):
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)
    if not key:
        sys.exit("No cache and no URA_ACCESS_KEY set.")
    return download(key)


def flatten(projects):
    """One row per transaction, with sqft + psf derived."""
    rows = []
    for p in projects:
        proj, street, mkt = p.get("project"), p.get("street"), p.get("marketSegment")
        for t in p.get("transaction", []):
            try:
                price = float(t["price"])
                area = float(t["area"])            # sqm of the unit
            except (KeyError, TypeError, ValueError):
                continue
            sqft = area * SQM_TO_SQFT
            rows.append({
                "project": proj, "street": street, "segment": mkt,
                "price": price, "sqm": area, "sqft": sqft,
                "psf": price / sqft if sqft else None,
                "contract": t.get("contractDate"),      # MMYY
                "type": t.get("propertyType"),
                "tenure": t.get("tenure"),
                "sale_type": t.get("typeOfSale"),       # 1=new,2=subsale,3=resale
                "floor": t.get("floorRange"),
                "district": t.get("district"),
                "no_units": int(t.get("noOfUnits", 1) or 1),
            })
    return rows


def yyq(c):
    """MMYY contract date -> sortable YYYY-MM."""
    if not c or len(c) != 4:
        return "?"
    mm, yy = c[:2], c[2:]
    return f"20{yy}-{mm}"


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def summarise(rows, label):
    if not rows:
        print(f"\n{label}: no matching transactions")
        return None
    prices = sorted(r["price"] for r in rows)
    psfs = sorted(r["psf"] for r in rows if r["psf"])
    sqfts = sorted(r["sqft"] for r in rows)
    print(f"\n=== {label} — {len(rows)} transactions ===")
    print(f"  size   : {sqfts[0]:,.0f} – {sqfts[-1]:,.0f} sqft (median {pct(sqfts,.5):,.0f})")
    print(f"  price  : {prices[0]:,.0f} – {prices[-1]:,.0f} "
          f"(p25 {pct(prices,.25):,.0f} | median {pct(prices,.5):,.0f} | p75 {pct(prices,.75):,.0f})")
    print(f"  psf    : {psfs[0]:,.0f} – {psfs[-1]:,.0f} "
          f"(p25 {pct(psfs,.25):,.0f} | median {pct(psfs,.5):,.0f} | p75 {pct(psfs,.75):,.0f})")
    return {"prices": prices, "psfs": psfs, "sqfts": sqfts}


def by_quarter(rows):
    g = defaultdict(list)
    for r in rows:
        g[yyq(r["contract"])].append(r["psf"])
    print("\n  psf trend by month of contract:")
    for k in sorted(g):
        v = sorted(x for x in g[k] if x)
        if v:
            print(f"    {k}  n={len(v):4d}  median psf {pct(v,.5):,.0f}")


def rank(value, sorted_vals):
    below = sum(1 for v in sorted_vals if v < value)
    return 100.0 * below / len(sorted_vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--project")
    ap.add_argument("--district")
    ap.add_argument("--segment", help="CCR / RCR / OCR")
    ap.add_argument("--type", dest="ptype", help="Apartment, Condominium, ...")
    ap.add_argument("--sqft-min", type=float)
    ap.add_argument("--sqft-max", type=float)
    ap.add_argument("--price", type=float, help="benchmark this price")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    key = os.environ.get("URA_ACCESS_KEY")
    projects = download(key) if a.refresh else load(key)
    rows = flatten(projects)
    print(f"\nLoaded {len(rows):,} transactions across {len(projects):,} projects")
    if rows:
        cds = sorted({yyq(r['contract']) for r in rows})
        print(f"Period covered: {cds[0]} .. {cds[-1]}")

    sel = rows
    if a.project:
        sel = [r for r in sel if r["project"] and a.project.upper() in r["project"].upper()]
    if a.district:
        sel = [r for r in sel if r["district"] == a.district]
    if a.segment:
        sel = [r for r in sel if (r["segment"] or "").upper() == a.segment.upper()]
    if a.ptype:
        sel = [r for r in sel if (r["type"] or "").upper() == a.ptype.upper()]
    if a.sqft_min:
        sel = [r for r in sel if r["sqft"] >= a.sqft_min]
    if a.sqft_max:
        sel = [r for r in sel if r["sqft"] <= a.sqft_max]

    label = " / ".join(filter(None, [
        a.project, f"D{a.district}" if a.district else None, a.segment, a.ptype,
        f"{a.sqft_min or 0:.0f}-{a.sqft_max or 99999:.0f} sqft"
        if (a.sqft_min or a.sqft_max) else None])) or "ALL TRANSACTIONS"

    s = summarise(sel, label)
    if s and len(sel) <= 4000:
        by_quarter(sel)

    if a.price and s:
        p = a.price
        print(f"\n>>> Benchmarking {p:,.0f} against {len(sel)} comparable transactions")
        print(f"    percentile by absolute price: {rank(p, s['prices']):.0f}th")
        print(f"    median comparable price     : {pct(s['prices'],.5):,.0f} "
              f"({(p/pct(s['prices'],.5)-1)*100:+.1f}% vs median)")
        if a.sqft_min and a.sqft_max:
            mid = (a.sqft_min + a.sqft_max) / 2
            implied = p / mid
            print(f"    implied psf at {mid:,.0f} sqft   : {implied:,.0f}")
            print(f"    percentile by psf           : {rank(implied, s['psfs']):.0f}th")
            print(f"    median comparable psf       : {pct(s['psfs'],.5):,.0f} "
                  f"({(implied/pct(s['psfs'],.5)-1)*100:+.1f}% vs median)")

        print(f"\n    closest {a.top} comparables by price:")
        for r in sorted(sel, key=lambda r: abs(r["price"] - p))[:a.top]:
            print(f"      {yyq(r['contract'])}  {r['project'][:28]:28s} "
                  f"{r['sqft']:6,.0f}sqft  {r['price']:>11,.0f}  {r['psf']:>6,.0f}psf  "
                  f"fl{r['floor']:>7s}  D{r['district']}  {r['segment']}")


if __name__ == "__main__":
    main()
