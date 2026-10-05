"""Forecast one condo unit with an explainable breakdown.

    python run_unit_forecast.py artra "#12-05"            # area from the stack's past sales
    python run_unit_forecast.py artra "#12-05" --area 829
"""
import argparse

import pandas as pd

from sgpf.unit_forecast import forecast_unit

p = argparse.ArgumentParser()
p.add_argument("project", help="propertynoob slug, e.g. artra")
p.add_argument("unit", help='unit number, e.g. "#12-05"')
p.add_argument("--area", type=float, help="sqft (default: median of same stack)")
a = p.parse_args()

r = forecast_unit(a.project, a.unit, a.area)
row, acc = r["row"], r["accuracy"]
lease = f" ({row.lease_left:.0f} yrs left)" if row.tenure_type == "leasehold" else ""
print(f"{a.project} {a.unit}: {row.area_sqft:,.0f} sqft, floor {row.floor:.0f}, "
      f"{row.tenure_type}{lease}, building age {row.age:.0f} yrs, {row.region}, "
      f"{row.district}, {row.mrt_km * 1000:,.0f} m to nearest MRT exit")


def show(title, ex):
    print(f"\n{title}")
    for _, e in ex.iterrows():
        if "vs" in e and isinstance(e.vs, str) and abs(e.pct) < 0.05:
            continue  # the unit is at the reference level: no adjustment
        pct = "" if pd.isna(e.pct) else f"{e.pct:+.1f}%" + (
            f" vs {e.vs}" if "vs" in e and isinstance(e.vs, str) else "")
        print(f"  {e['item']:<60} {pct:>18}   -> ${e.psf_after:,.0f} psf")


show("Transparent model (recent sales in this project + lookup-table adjustments):",
     r["explain_A"])
show("LightGBM model (contribution of each factor):", r["explain_B"])
res = acc[acc["chosen_rule"]]["resale"]
print(f"\nEstimate today ({r['rule']}): ${r['psf']:,.0f} psf x {row.area_sqft:,.0f} sqft"
      f" = S${r['value_now']:,.0f}")
print(f"  Likely range (80%): S${r['value_now_low']:,.0f} - S${r['value_now_high']:,.0f}")
print(f"  Tested on {res['n']} resales in {acc['test_from']}..{acc['test_to']}: typical error"
      f" {res['median_err_pct']}%, {res['within_10pct']:.0%} within 10%")

print("\nForecast (likely range = where 80% of actual resales landed in the 1997-2025"
      " backtest):")
print(f"  {'':>7} {'future dollars':>14}  {'80% range':>27}  {"today's $":>11}  "
      f"{'market':>7}  backtest: typical error, within 10%")
for _, x in r["forecast"].iterrows():
    print(f"  +{x.years:>2.0f} yr  S${x.value:>12,.0f}  (S${x.low:>10,.0f} - S${x.high:>10,.0f})"
          f"  S${x.value_todays_dollars:>9,.0f}  {x.market_growth_pct:+6.1f}%"
          f"   {x.backtest_median_err_pct:4.1f}%, {x.backtest_within_10pct:.0%}"
          f" ({x.backtest_origins} start years)")
print("\n  Market path: short-term index model for years 1-2, then base-case income growth"
      " (URA Non-Landed index).")
