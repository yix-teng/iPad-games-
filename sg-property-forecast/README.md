# Singapore private property price forecast

Three complementary models:

| | Question | Data | Model |
|---|---|---|---|
| **Index model** (`run_index_forecast.py`) | Where is the market going over the next 1–8 quarters? | URA Private Residential Property Price Index, 1975Q1–present (data.gov.sg, no key needed) | ARIMA, ridge, LightGBM, and an ensemble, with direct multi-horizon targets. Compared against random-walk and drift baselines |
| **Long-run scenarios** (`run_long_run.py`) | Where could prices be in 3–10 years? | Same index + SingStat nominal GDP and population (data.gov.sg) | Rule: prices grow with incomes. The scenarios are income-growth assumptions, and the range comes from how far prices deviated from incomes in past decades |
| **Hedonic model** (`run_hedonic.py`) | What is a specific unit worth? | URA private residential transactions (last ~5 yrs, [free URA API key](https://eservice.ura.gov.sg/maps/api/reg.html)) | LightGBM on log price per sqm. Features: project, segment (CCR/RCR/OCR), district, area, floor, tenure and remaining lease, location, sale date |

To combine them, the hedonic model prices a unit at today's market level and the index
forecast rolls that price forward (`hedonic.predict_future(model, units, index_growth=...)`).

## Quick start

```bash
pip install -r requirements.txt
python run_index_forecast.py                       # All Residential; also --series Landed / Non-Landed
python run_long_run.py                             # 3/5/10-year scenarios; --base 0.05 etc.
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
```

* **Income** is nominal GDP per resident (SingStat GDP divided by population). It is the
  strongest single driver of 10-year price growth since 1975 (correlation +0.86).
* **The central path has no fitted parameters:** prices grow at the same rate as incomes.
  Each scenario is one income-growth assumption, set to real decades since 1990:

  | Scenario | Income growth per year | Where it comes from |
  |---|---|---|
  | Bear | 3.5% | Weakest decade since 1990 |
  | Base | 4.6% | Median decade since 1990 |
  | Bull | 5.9% | Strongest decade since 1990 (≈ the last 10 years, 5.8%) |

* **The range** comes from how far prices ran ahead of or behind incomes over the same
  horizon in every window since 1985. It covers everything the rule leaves out: rate cycles,
  cooling measures, supply gluts and crises.

### Accuracy (out-of-sample, start dates 1985Q4 onward)

Average miss in percentage points of total price growth:

| Horizon | Flat forecast | Income growth known | Income guessed (last decade's pace) | "Likely" range held | "Wide" range held |
|---|---|---|---|---|---|
| 3 yrs | 28 | **19** | 23 | 76% (nominal 80%) | 84% (nominal 90%) |
| 5 yrs | 45 | **29** | 40 | 72% | 76% |
| 10 yrs | 74 | **43** | 61 | 68% | 69% |

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

Interest rates are not modelled. data.gov.sg only has rate history from 1998 (bond yields)
and 2005 (other bank rates), which is too short to fit a 10-year relationship. Their effect is
inside the historical range.

![long-run scenarios](outputs/long_run_scenarios.png)

## Possible next steps

* Add macro drivers to the short-term model (3M SORA or mortgage rates, unemployment,
  new-launch supply and the unsold pipeline, HDB resale index). This is likely the biggest
  gain, but the drivers must be forecast or lagged to avoid look-ahead.
* Estimate lease decay for individual units from transaction data (needs a URA key) so that
  long-run scenarios can be applied per unit.
* Forecast the region sub-indices (CCR/RCR/OCR) or a monthly index built from transactions,
  which gives about 3× more data points.
* Add amenity features to the hedonic model (distance to MRT stations and good schools).
  The URA `x`/`y` coordinates are SVY21 and can be joined to OneMap or LTA data.

Data: URA via data.gov.sg (Singapore Open Data Licence). This repository does not include
any URA transaction data.
