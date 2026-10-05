"""Forecast one condo unit with an explainable breakdown and backtested ranges.

    python run_unit_forecast.py artra "#12-05"                    # existing project
    python run_unit_forecast.py artra "#12-05" --area 829
    python run_unit_forecast.py new "#15-03" --area 850 --postal 579837 \\
        --tenure leasehold --top 2030                             # project with no sales yet
    (add --ec for an executive condominium, --district D20 to override the district)
"""
import argparse

import numpy as np
import pandas as pd

from sgpf.unit_forecast import forecast_unit

p = argparse.ArgumentParser()
p.add_argument("project", help="propertynoob slug (e.g. artra), or 'new' for a project with "
                                "no sales yet")
p.add_argument("unit", help='unit number, e.g. "#12-05"')
p.add_argument("--area", type=float, help="sqft (default: median of same stack)")
p.add_argument("--postal", help="new project: postal code")
p.add_argument("--tenure", choices=["freehold", "leasehold"], help="new project: tenure")
p.add_argument("--ec", action="store_true", help="new project: executive condominium")
p.add_argument("--top", type=int, help="new project: expected TOP year")
p.add_argument("--district", help="new project: district, e.g. D20 (default: nearest project)")
a = p.parse_args()

new = None
if a.project == "new" or a.postal:
    if not (a.postal and a.tenure and a.top):
        p.error("a new project needs --postal, --tenure and --top")
    new = {"postal": a.postal, "tenure": a.tenure, "is_ec": a.ec, "top_year": a.top,
           "district": a.district}
r = forecast_unit(None if a.project == "new" else a.project, a.unit, a.area, new_project=new)
row = r["row"]
lease = (f" ({row.lease_left:.0f} yrs left)" if row.tenure_type == "leasehold"
         and pd.notna(row.lease_left) else "")
kind = "EC" if row.is_ec else "condo"
age = f"building age {row.age:.0f} yrs" if row.age >= 0 else f"TOP in {-row.age:.0f} yrs"
print(f"{a.project} {a.unit}: {row.area_sqft:,.0f} sqft, floor {row.floor:.0f}, {kind}, "
      f"{row.tenure_type}{lease}, {age}, {row.region}, {row.district}")
print(f"Pricing path: {r['mode']}")


def show(title, ex):
    print(f"\n{title}")
    for _, e in ex.iterrows():
        if "vs" in e and isinstance(e.vs, str) and abs(e.pct) < 0.05:
            continue
        pct = "" if pd.isna(e.pct) else f"{e.pct:+.1f}%" + (
            f" vs {e.vs}" if "vs" in e and isinstance(e.vs, str) else "")
        print(f"  {e['item']:<60} {pct:>18}   -> ${e.psf_after:,.0f} psf")


if r["comparables"] is not None:
    c = r["comparables"]
    print(f"\nComparable launches ({c.how.iloc[0]}, last 12 months, same tenure and EC "
          f"status), base price after removing floor/size effects:")
    for s, x in c.head(10).iterrows():
        print(f"  {s:<36} {x.km:5.1f} km  {x.sales:4.0f} new sales  ${np.exp(x.base):,.0f} psf")
    if len(c) > 10:
        print(f"  ... and {len(c) - 10} more")
else:
    show("Transparent model (recent sales in this project + lookup-table adjustments):",
         r["explain_A"])
    show("LightGBM model (contribution of each factor):", r["explain_B"])

label = {"brand-new launch": "Launch price today (comparable launches)",
         "launch phase": "New-sale price today (this project's recent launch prices)",
         "resale": "Resale value today (LightGBM)"}[r["mode"]]
print(f"\n{label}: ${r['psf']:,.0f} psf x {row.area_sqft:,.0f} sqft = S${r['price_now']:,.0f}")
print(f"  Likely range (80%): S${r['price_now_low']:,.0f} - S${r['price_now_high']:,.0f}"
      f"   [typical error {r['now_err']:.1f}% on {r['now_source']}]")
if r["mode"] != "resale":
    print(f"  Resale-equivalent value today (no new-launch premium): "
          f"S${r['resale_value_now']:,.0f}; forecasts below are resale values.")

todays = "today's $"
print("\nForecast (likely range = where 80% of actual resales landed in the backtest for this "
      "pricing path):")
print(f"  {'':>7} {'future dollars':>14}  {'80% range':>27}  {todays:>11}  {'market':>7}"
      + ("  EC adj" if row.is_ec else "") + "  backtest: typical error, within 10%")
for _, x in r["forecast"].iterrows():
    ec = f"  {x.ec_adjust_pct:+5.1f}%" if row.is_ec else ""
    note = f" [from {x.backtest_years:.0f}-yr]" if x.backtest_years != x.years else ""
    print(f"  +{x.years:>2.0f} yr  S${x.value:>12,.0f}  (S${x.low:>10,.0f} - S${x.high:>10,.0f})"
          f"  S${x.value_todays_dollars:>9,.0f}  {x.market_growth_pct:+6.1f}%{ec}"
          f"   {x.backtest_err_pct:4.1f}%, {x.backtest_within_10pct:.0%}"
          f" ({x.backtest_groups:.0f} {'launches' if r['mode'] == 'brand-new launch' else 'start years'})"
          f"{note}")
print("\n  Market path: short-term index model for years 1-2, then base-case income growth"
      " (URA Non-Landed index).")
