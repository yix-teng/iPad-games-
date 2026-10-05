# Singapore private property price forecast

Four complementary models:

| | Question | Data | Model |
|---|---|---|---|
| **Index model** (`run_index_forecast.py`) | Where is the market going over the next 1–8 quarters? | URA Private Residential Property Price Index, 1975Q1–present (data.gov.sg, no key needed) | ARIMA, ridge, LightGBM, and an ensemble, with direct multi-horizon targets. Compared against random-walk and drift baselines |
| **Long-run scenarios** (`run_long_run.py`) | Where could prices be in 3–10 years? | Same index + SingStat nominal GDP, population and CPI (data.gov.sg) | Rule: prices grow with incomes. The scenarios are real income growth plus inflation, results come in future dollars and today's dollars, and the range comes from how far prices deviated from incomes in past decades |
| **Unit forecast** (`run_unit_model.py`, `run_unit_forecast.py`) | What is this condo unit worth now, and in 1–20 years? | ~600k condo sales since 1995 from propertynoob.com (not committed), OneMap geocoding, LTA MRT exits | Valuation from a transparent model and LightGBM, times the market path. Ranges come from a 1997–2025 backtest against actual resale prices |
| **Hedonic model** (`run_hedonic.py`) | What is a specific unit worth? | URA private residential transactions (last ~5 yrs, [free URA API key](https://eservice.ura.gov.sg/maps/api/reg.html)) | LightGBM on log price per sqm. Features: project, segment (CCR/RCR/OCR), district, area, floor, tenure and remaining lease, location, sale date |

To combine them, the hedonic model prices a unit at today's market level and the index
forecast rolls that price forward (`hedonic.predict_future(model, units, index_growth=...)`).

## Quick start

```bash
pip install -r requirements.txt
python run_index_forecast.py                       # All Residential; also --series Landed / Non-Landed
python run_long_run.py                             # 3/5/10-year scenarios; --base 0.03 --inflation 0.02 etc.
python -m sgpf.propertynoob                        # collect condo sales (~2.5 h, resumable)
python run_unit_model.py                           # fit + test unit models
python run_unit_backtest.py                        # 1-20 yr backtest (~25 min)
python run_unit_forecast.py artra "#12-05"         # one unit
python run_hedonic.py --ura-key YOUR_ACCESS_KEY    # needs a URA key
python -m pytest -q tests
```

Outputs are written to `outputs/`: backtest scores and predictions, the forecast table, and `forecast.png`.

## Results: index model, All Residential (data to 2026Q2)

Expanding-window backtest. Each quarter from 2010Q1 onward, the models are refit using only
the data available at that time. RMSE is in % log-growth; lower is better.

| Horizon | Random walk | ARIMA | Ridge | LightGBM | **Ensemble** |
|---|---|---|---|---|---|
| 1Q | 1.76 | 1.47 | 1.42 | 1.87 | **1.35** |
| 2Q | 3.08 | 2.35 | 2.29 | 3.64 | **2.22** |
| 4Q | 5.51 | **4.21** | 4.71 | 8.85 | 4.28 |
| 8Q | 9.74 | **8.24** | 22.61 | 16.59 | **8.24** |

![forecast](outputs/forecast.png)

## What the backtest shows

* **On ~200 quarterly points, a simple time-series model beats the ML models.** LightGBM
  never beats the random walk. Ridge helps only up to about a year. An 8-quarter direct
  target has only ~17 non-overlapping observations, so the ML models overfit. The ensemble
  therefore averages ARIMA and ridge up to h=4 and uses ARIMA alone beyond that. This split
  was chosen from the same backtest, so its scores are slightly optimistic. Because the
  model mix changes at h=5, the forecast path has a small kink between 2027Q2 and 2027Q3.
* **Policy regime features are risky for linear models.** Cooling measures began in 2009.
  In early backtest windows, ridge saw almost no variation in those features and
  extrapolated them to +50% 2-year forecasts. Ridge now drops them, and the tree models
  keep them.
* **Prediction intervals** come from the actual backtest errors at each horizon, not from
  model assumptions. They do not cover regime breaks such as new cooling measures, rate
  shocks or a GFC-style crash.

## Long-run scenarios (3–10 years)

### How it works

```
price in N years = price today × (income growth over N years) × historical deviation
income growth    = real income growth + inflation (CPI)
```

* **Income** is nominal GDP per resident (SingStat GDP divided by population). It is the
  strongest single driver of 10-year price growth since 1975 (correlation +0.86).
* **The central path has no fitted parameters:** prices grow at the same rate as incomes.
  Each scenario is a real income growth assumption plus an inflation assumption, both set
  to real decades since 1990:

  | Scenario | Real income growth per year | Inflation (CPI) per year | Nominal income growth | Where it comes from |
  |---|---|---|---|---|
  | Bear | 1.6% | 1.6% | 3.2% | Weakest real-income decade since 1990 |
  | Base | 2.9% | 1.6% | 4.6% | Median decade since 1990 |
  | Bull | 4.5% | 1.6% | 6.2% | Strongest decade since 1990 |

  The inflation default is the median decade since 1990 (range 0.6–2.6%). Use `--inflation`
  to override it.

* **Two units.** "Future dollars" is the price tag in that year. "Today's dollars" divides
  it by cumulative CPI inflation, i.e. purchasing power. In today's dollars the rule becomes
  *prices grow with real incomes*, so the inflation assumption changes the price tag but not
  the real result.
* **The range** comes from how far prices ran ahead of or behind incomes over the same
  horizon in every window since 1985. It covers everything the rule leaves out: rate cycles,
  cooling measures, supply gluts and crises. The same range applies in both units.

### Results (index today 219.4, data to 2026Q2)

| 10 years (2036Q2) | Future dollars | Today's dollars | Likely range, today's dollars |
|---|---|---|---|
| Bear | 301 (+37%) | 257 (+17%) | 159–348 (−27% to +59%) |
| **Base** | **343 (+56%)** | **293 (+33%)** | **182–397 (−17% to +81%)** |
| Bull | 399 (+82%) | 340 (+55%) | 211–461 (−4% to +110%) |

Base case at 10 years under different inflation: future dollars +42% (0.6%/yr), +56% (1.6%),
+73% (2.6%). In today's dollars it is +33% in every case.

**Context:** since 1990, real home prices grew about 1.7% a year in a typical decade, slower
than real incomes (2.9%). Real prices fell over the decades starting 1994–1996 and were flat
over the decade starting 2008. The central path assumes prices keep pace with incomes; the
historical range includes those weaker decades.

### Accuracy (out-of-sample, start dates 1985Q4 onward)

Average miss in percentage points of total price growth:

| Horizon | Flat forecast | Income growth known | Income guessed (last decade's pace) | "Likely" range held | "Wide" range held |
|---|---|---|---|---|---|
| 3 yrs | 28 | **19** | 23 | 76% (nominal 80%) | 84% (nominal 90%) |
| 5 yrs | 45 | **29** | 40 | 72% | 76% |
| 10 yrs | 74 | **43** | 61 | 68% | 69% |

The same test in today's dollars. Here the baseline is that prices just keep pace with
inflation, a tougher comparison:

| Horizon | Prices keep pace with inflation | Real income growth known |
|---|---|---|
| 3 yrs | 22 | **18** |
| 5 yrs | 34 | **27** |
| 10 yrs | 48 | **36** |

There are only about 3 non-overlapping 10-year windows since 1985, so the 10-year ranges
are rough. In testing they held about 7 times in 10, not the nominal 8 or 9 in 10. For the
same reason, the range edges move unevenly from year to year.

### Tested and rejected (they made out-of-sample error worse)

| Variant | 10-yr average miss |
|---|---|
| Income elasticity fitted freely (fitted value 2.2, driven by the 1975–1990 boom) | 47 (start dates from 1976); in a real-time test it undershot recent decades by 34 pts on average |
| Income growth plus a valuation correction (price/income vs its history) | 66 |
| Valuation correction plus supply pipeline (pipeline sign flipped, a sign of overfitting) | unstable; only ~2 decades of data |
| **Prices = incomes (chosen)** | **43** |

The inflation assumption itself has no measured error: it is an input. Interest rates are not modelled. data.gov.sg only has rate history from 1998 (bond yields)
and 2005 (other bank rates), which is too short to fit a 10-year relationship. Their effect is
inside the historical range.

![long-run scenarios](outputs/long_run_scenarios.png)

## Unit forecast (individual condos)

```
value in N years = value today                 (valuation model)
                 × market growth to year N     (URA Non-Landed index: short-term model for
                                                years 1–2, then base-case income growth)
```

The range at each horizon is where 80% of actual resale prices landed relative to forecasts
made this way in the backtest below.

### Data

603,137 single-unit condo, apartment and EC sales in 2,736 projects, 1995-01 to 2026-09, from
propertynoob.com. The feed is labelled `"source": "huttons"`. It is collected for personal
analysis and kept out of git. Project details come from the same site (tenure and lease
start, TOP, district, address). Location comes from OneMap and LTA MRT exits (91% of
postcodes geocoded).

### Valuation: two models, both explainable

* **Transparent model:** the project's recent sales (weighted toward the last ~6 months),
  then lookup-table adjustments for the region's market level, stack, floor, size, EC age
  and sale type. Each line in its breakdown is a % effect, for example:

  | Floor | 1–2 | 3–5 | 6–10 | 11–15 | 16–20 | 21–25 | 26–30 | 31–40 | 41+ |
  |---|---|---|---|---|---|---|---|---|---|
  | Effect | −4.7% | −2.6% | ref | +2.1% | +4.2% | +7.0% | +9.9% | +13.3% | +20.3% |

  | Size (sqft) | <500 | 500–650 | 650–800 | 800–1k | 1k–1.25k | 1.25k–1.5k | 1.5k–2k | 2k–3k | 3k+ |
  |---|---|---|---|---|---|---|---|---|---|
  | Effect on $psf | +24.5% | +12.6% | +5.5% | ref | −5.4% | −9.4% | −14.9% | −21.7% | −27.5% |

  New-launch sales carry a +5.6% premium over resales. ECs sell 18% below comparable condos
  before TOP and 13% below in their first 3 years.
* **LightGBM:** the same information plus location (district, coordinates, distance to MRT
  and CBD). Each valuation comes with the contribution of each factor (SHAP values).

**Accuracy.** The models were trained on sales up to 2026-03 and tested on 6,183 sales from
2026-04 to 2026-09. The way the two models are combined was chosen on a separate window
(2025-10 to 2026-03), not on the test window.

| Typical error (median), test window | New sales | Resales | Resales within 10% | 1 in 10 resales off by more than |
|---|---|---|---|---|
| Transparent | **2.2%** | 5.8% | 74% | 15.2% |
| LightGBM | 8.4% | **4.0%** | **88%** | 10.5% |

A unit forecast values the unit as a resale, so it uses LightGBM. The transparent breakdown
is shown alongside it as the explanation in plain lookup terms.

### Ageing and lease: did condos keep up with their region?

The model compares each project's growth over 5 years with the median project in its
region over the same years. Ageing and lease run-down cannot be separated from market
movement inside a valuation model, because age, lease and time all advance together; this
comparison sidesteps that.

| Group (% per year vs region, 5-year spans) | Effect | Projects |
|---|---|---|
| Freehold, 0–5 yrs old | −0.7% | 771 |
| Leasehold, 0–5 yrs old, 90+ yrs left | −0.9% | 335 |
| Leasehold, 5–15 yrs old, 80–90 yrs left | 0.0% to −0.3% | 181–269 |
| Freehold, 15–30 yrs old | +1.2% to +1.3% | 153–218 |
| Leasehold, 20–30 yrs old, 70–80 yrs left | +1.2% | 92 |
| Leasehold, 30+ yrs old, under 60 yrs left | +0.2% | 18 |

New condos lag their region as the launch premium fades. In this data, older leaseholds
have **not** clearly lagged. Two caveats: the groups with short leases are small, and condos
already sold en bloc are missing from the site. Survivors, and old condos that buyers expect
to go en bloc, may make old condos look better than a typical old condo really did.

### Accuracy 1–20 years ahead (backtest)

For each year-end from 1997 to 2025 (the "start year"), only data available at that time was
used:

1. Units were valued with LightGBM (and, for comparison, the transparent model) fitted on
   sales up to then.
2. Values were grown with the short-term index forecast for years 1–2, then with the
   trailing 10-year income growth.
3. The forecasts were compared with the actual resale prices of the same units (same
   project and unit number) 1–20 years later.

| Years ahead | Start years | Resales tested | Typical error | Within 10% | Within 20% | 80% of actual prices landed within | Forecast bias | No-growth baseline error |
|---|---|---|---|---|---|---|---|---|
| 1 | 29 | 365k | **8.0%** | 58% | 85% | -11% to +26% | -3.3% | 8.7% |
| 2 | 28 | 239k | **11.2%** | 46% | 75% | -16% to +33% | -2.1% | 13.0% |
| 3 | 27 | 232k | **13.6%** | 38% | 67% | -20% to +38% | -2.7% | 17.0% |
| 4 | 26 | 225k | **17.1%** | 31% | 57% | -24% to +46% | -3.1% | 19.9% |
| 5 | 25 | 214k | **19.3%** | 27% | 52% | -27% to +51% | -2.2% | 21.8% |
| 7 | 23 | 185k | **23.0%** | 24% | 44% | -34% to +56% | +4.3% | 24.1% |
| 10 | 20 | 127k | **28.0%** | 17% | 35% | -40% to +44% | +17.0% | 29.1% |
| 15 | 15 | 55k | **22.8%** | 22% | 44% | -43% to +46% | -3.5% | 49.3% |
| 20 | 10 | 26k | **38.3%** | 13% | 26% | -68% to +55% | +17.5% | 53.3% |

**Start years 2016–2025 only** (a calmer period with richer data; few start years at
longer horizons, so treat these as the good-times case, not the expected case):

| Years ahead | Start years | Typical error | Within 10% | 80% of actual prices landed within |
|---|---|---|---|---|
| 1 | 10 | 5.2% | 79% | -7% to +13% |
| 2 | 9 | 7.1% | 66% | -10% to +18% |
| 3 | 8 | 7.8% | 62% | -10% to +20% |
| 5 | 6 | 9.9% | 50% | -12% to +27% |
| 7 | 4 | 12.6% | 40% | -16% to +31% |

Forecast bias is the median of forecast vs actual (+ = forecast too high). Valuations are LightGBM, refitted at each start year on data up to then; the unit forecast uses these ranges.

Full tables for every year: `outputs/unit_backtest_by_horizon.csv` (all start years) and
`outputs/unit_backtest_by_horizon_since2016.csv`.

**How to read this:**

* **Start year matters most.** Units in the same start year share one market path, so their
  errors move together. Typical 10-year error by start year ranged from −36% (forecasts made
  in 2003, too low) to +106% (forecasts made in 1997, too high: 9%/yr income growth was
  extrapolated just before the Asian crisis).
* **Long horizons rest on few independent periods.** The 20-year figures come from only 10
  start years (1997–2006), which is one market cycle.
* **At 9–10 years the no-growth baseline did as well as the forecast.** Starts in 2007 and
  2010–2013 were followed by cooling measures and a slump, and the income-growth path
  over-forecast them by 30–44%.
* **The age/lease adjustment was dropped.** "Valuation × market path × age/lease relative
  performance" had a higher error than "valuation × market path" at every horizon from 3
  years (for example 32.9% vs 29.0% at 10 years). The unit forecast therefore does not apply
  it; the table above is kept for information.

## Pricing paths in the unit forecast

`run_unit_forecast.py` picks one of three pricing paths. ECs also get the EC age adjustment.
Each path's 80% ranges come from its own backtest.

| Path | When | Price today | Today: typical error, 80% range |
|---|---|---|---|
| **Resale** | Project has resales | LightGBM resale value | 4.0%, −6% to +10% (4,675 resales, Apr–Sep 2026) |
| **Launch phase** | Project is selling new units and has no resales yet | Transparent model's new-sale price (the project's own recent launch prices) | 2.2%, −9% to +5% (1,266 new sales, Apr–Sep 2026) |
| **Brand-new launch** | Project has no transactions. Run `python run_unit_forecast.py new "#15-03" --area 850 --postal 579837 --tenure leasehold --top 2030 [--ec]` | Comparable launches within 3 km in the last 12 months, same tenure and EC status, adjusted for floor and size (`sgpf/launch.py`). The comparables are listed in the output | 11.4%, −18% to +29% (1,551 launches since 2000) |

For the launch paths, forecasts are of the **resale** value (no new-launch premium), because
the backtests score against later resales.

**EC adjustment.** An EC forecast is multiplied by the change in the EC age effect between now
and the target year, using the transparent model's EC age table. This captures the jumps when
the 5-year minimum occupation period ends and at 10-year privatisation. In the backtest the
table was re-estimated at each start year from data up to then. Without the adjustment, EC
forecasts were too low by 12–17% at 2–5 years.

| EC typical error | 1 yr | 2 yr | 3 yr | 5 yr | 7 yr | 10 yr | 15 yr | 20 yr |
|---|---|---|---|---|---|---|---|---|
| Without adjustment | 8.6% | 13.4% | 16.6% | 20.6% | 22.2% | 20.9% | 22.2% | 41.6% |
| **With adjustment (used)** | 8.7% | 12.4% | 14.0% | 16.8% | 20.1% | 21.3% | 23.5% | 40.3% |

### Forecast error and range by path

**Resale (condo/apartment)**, LightGBM valuation (start years 1997–2025):

| Years ahead | Typical error | Within 10% | 80% of actual within | Start years |
|---|---|---|---|---|
| 1 | 8.0% | 58% | -11% to +26% | 29 |
| 2 | 11.2% | 46% | -16% to +33% | 28 |
| 3 | 13.6% | 38% | -20% to +38% | 27 |
| 5 | 19.3% | 27% | -27% to +51% | 25 |
| 10 | 28.0% | 17% | -40% to +44% | 20 |
| 15 | 22.8% | 22% | -43% to +46% | 15 |
| 20 | 38.3% | 13% | -68% to +55% | 10 |

**Launch phase** (bought before completion, non-EC):

| Years ahead | Typical error | Within 10% | 80% of actual within | Start years |
|---|---|---|---|---|
| 1 | 10.3% | 49% | -16% to +35% | 29 |
| 2 | 10.7% | 48% | -19% to +34% | 28 |
| 3 | 12.8% | 41% | -21% to +37% | 27 |
| 5 | 18.9% | 29% | -29% to +45% | 25 |
| 10 | 31.9% | 14% | -44% to +33% | 20 |
| 15 | 24.8% | 21% | -51% to +36% | 15 |
| 20 | 49.8% | 14% | -72% to +35% | 10 |

**EC** (resale or launch phase, with EC adjustment):

| Years ahead | Typical error | Within 10% | 80% of actual within | Start years |
|---|---|---|---|---|
| 1 | 8.7% | 57% | -6% to +22% | 27 |
| 2 | 12.4% | 40% | -10% to +30% | 27 |
| 3 | 14.0% | 33% | -12% to +34% | 27 |
| 5 | 16.8% | 26% | -21% to +41% | 25 |
| 10 | 21.3% | 26% | -41% to +44% | 20 |
| 15 | 23.5% | 19% | -40% to +45% | 15 |
| 20 | 40.3% | 17% | -64% to +53% | 10 |

**Brand-new launch** (priced from comparables; horizons backed by fewer than 200 launches use the
nearest that has enough, because few units resell before completion):

| Years ahead | Typical error | Within 10% | 80% of actual within | Launches |
|---|---|---|---|---|
| 1 (uses 3-yr) | 20.3% | 26% | -32% to +83% | 371 |
| 2 (uses 3-yr) | 20.3% | 26% | -32% to +83% | 371 |
| 3 | 20.3% | 26% | -32% to +83% | 371 |
| 5 | 21.6% | 24% | -29% to +71% | 980 |
| 10 | 26.7% | 18% | -40% to +33% | 1027 |
| 15 | 23.3% | 23% | -41% to +44% | 767 |
| 20 | 38.8% | 11% | -52% to +53% | 263 |

EC brand-new launches use the condo launch ranges: only 5–40 EC launches per horizon could be
tested (`outputs/launch_forecast_backtest.csv`).

Worked examples of all four paths: `outputs/unit_forecast_example.txt`.

### Accuracy by segment

**Valuation today** (train to 2026-03, test 2026-04..09; `outputs/unit_valuation_by_segment.csv`):

| Segment | Test sales | LightGBM typical error | 80% of actual within | Transparent typical error |
|---|---|---|---|---|
| Resales | 4,675 | **4.0%** | −6% to +10% | 5.8% |
| New sales in projects already selling | 1,266 | — | — | **2.2%** |
| Brand-new projects (no prior sales) | 1,796 | 32.3% (always too low) | +28% to +89% | cannot price |
| Freehold | 1,539 | 4.5% | −9% to +10% | 5.8% |
| Leasehold | 6,440 | 6.9% | −4% to +57% | 4.7% |
| Projects with 1–9 prior sales | 20 | 9.0% | −7% to +22% | 13.4% |
| Projects with 50+ prior sales | 5,989 | 4.2% | −6% to +14% | 4.9% |

The leasehold and OCR figures for LightGBM include the brand-new launches, most of which were
leasehold OCR projects.

**New launches priced from comparable launches** (`run_launch_backtest.py`,
`outputs/launch_pricing_backtest.csv`). Every project launched from 2000 was priced from new
sales of other projects within 3 km (fallback: same district) in the 12 months before its
launch, matched on tenure and EC status and adjusted for floor and size. The result was then
compared with its first 3 months of actual sales:

| Launches | Number | Typical error | Within 10% | 80% of actual within |
|---|---|---|---|---|
| All, 2000–2026 | 1,551 | **11.4%** | 45% | −18% to +29% |
| Launched 2020–2026 | 154 | **9.2%** | 50% | −12% to +25% |
| Leasehold | 416 | 10.3% | 50% | −17% to +26% |
| Freehold | 1,135 | 15.0% | 40% | −23% to +46% |
| CCR | 436 | 19.6% | 30% | −20% to +81% |
| OCR | 370 | 9.8% | 50% | −16% to +22% |
| EC | 55 | 7.2% | 60% | −6% to +20% |

The forecast tool uses this method for projects with no transactions (see Pricing paths).

**Multi-year forecasts** (`outputs/unit_backtest_by_segment.csv`; scored against later resales):

| Typical error | 1 yr | 3 yr | 5 yr | 10 yr |
|---|---|---|---|---|
| Completed at start | 8.0% | 13.7% | 19.4% | 27.1% |
| New launch / under construction at start | 10.3% | 12.8% | 18.7% | 30.3% |
| Freehold | 9.0% | 14.2% | 20.2% | 28.1% |
| Leasehold | 7.4% | 13.2% | 18.7% | 27.8% |
| No resales before start | 9.8% | 13.2% | 18.6% | 30.3% |
| 1–9 resales before start | 10.3% | 16.1% | 21.4% | 31.2% |
| 50+ resales before start | 7.5% | 13.1% | 19.2% | 25.2% |
| CCR | 9.0% | 15.2% | 22.7% | 32.4% |
| OCR | 7.7% | 13.6% | 18.8% | 26.3% |
| EC | 8.6% | 16.6% | 20.6% | 20.9% |

ECs were forecast too low from 3 years (bias about −15% to −17%). The forecasts did not
capture the jump in EC prices when the 5-year minimum occupation period ends and again at
full privatisation after 10 years.

### Attempts to improve accuracy

The backtest was split into two error sources. Valuation and unit-level noise account for an
error of about 7–14% even if the market path had been known exactly. Everything above that is
market-path error. Three improvements were then tested in sequence, each against the same
backtest:

| Step | What was tested | Result | Files |
|---|---|---|---|
| 1 | LightGBM valuation refitted at every start year | Same overall: 1 yr 8.0% vs 7.9%, 10 yr 28.0% vs 29.0%. Slightly better for start years 2016+ (1 yr 5.2% vs 5.8%). **Adopted for the forecast ranges**, since the tool values units with LightGBM | `outputs/unit_backtest_step1_lightgbm.csv` |
| 2 | Project features known at the start year: nearby new-launch supply, construction pipeline, MRT distance and upcoming stations (hand-coded opening years), en-bloc proxies, age, size, location, 3-yr momentum. Walk-forward ridge on de-meaned errors | Worse at every horizon from 2 yrs, with or without per-year effects. Each effect is under 3% per standard deviation and unstable over time | `outputs/unit_backtest_step2_*.csv`, `outputs/unit_step2_feature_effects*.csv` |
| 3 | Leading indicators in the short-term index model: 10-yr SGS yield and its change, pipeline/stock, vacancy, plus a variant with cooling measures | Index error worse at 2–4 quarters ahead (e.g. 4 quarters: 5.7 vs 4.7 pts RMSE). Unit error 1 yr 7.8% vs 7.9% (better in 11 of 29 start years); unchanged from 3 yrs | `outputs/index_backtest_step3.csv`, `outputs/unit_backtest_step3.csv` |

The adoption rule for step 3 was set before running it: adopt only if both the index backtest
and the unit backtest at 1–3 years improved. It did not meet that rule.

### Example: Artra #12-05 (786 sqft, 12th floor, 99-yr lease from 2016, RCR)

See `outputs/unit_forecast_example.txt`. Value today is **S$1.83M** (80% range S$1.73M–2.01M).
Same-size units on floors 25 and 33 sold for S$1.945M–1.949M in mid-2026.

| Horizon | Future dollars | 80% range (backtest) | Today's dollars |
|---|---|---|---|
| 1 yr | S$1.90M | 1.68–2.38M | S$1.87M |
| 3 yr | S$2.06M | 1.66–2.85M | S$1.97M |
| 5 yr | S$2.25M | 1.65–3.40M | S$2.08M |
| 10 yr | S$2.82M | 1.70–4.07M | S$2.41M |
| 20 yr | S$4.40M | 1.39–6.81M | S$3.21M |

## Possible next steps

* Add macro drivers to the short-term model (3M SORA or mortgage rates, unemployment,
  new-launch supply and the unsold pipeline, HDB resale index). This is likely the biggest
  gain, but the drivers must be forecast or lagged to avoid look-ahead.
* Add en-bloc history (demolished projects) to remove survivorship bias from the ageing table.
* Forecast the region sub-indices (CCR/RCR/OCR) or a monthly index built from transactions,
  which gives about 3× more data points.
* Add amenity features to the hedonic model (distance to MRT stations and good schools).
  The URA `x`/`y` coordinates are SVY21 and can be joined to OneMap or LTA data.

Data: URA via data.gov.sg (Singapore Open Data Licence). This repository does not include
any URA transaction data.
