# Singapore private property price forecast

Two complementary models:

| | Question | Data | Model |
|---|---|---|---|
| **Index model** (`run_index_forecast.py`) | Where is the market going over the next 1–8 quarters? | URA Private Residential Property Price Index, 1975Q1–present (data.gov.sg, no key needed) | ARIMA, ridge, LightGBM, and an ensemble, with direct multi-horizon targets. Compared against random-walk and drift baselines |
| **Hedonic model** (`run_hedonic.py`) | What is a specific unit worth? | URA private residential transactions (last ~5 yrs, [free URA API key](https://eservice.ura.gov.sg/maps/api/reg.html)) | LightGBM on log price per sqm. Features: project, segment (CCR/RCR/OCR), district, area, floor, tenure and remaining lease, location, sale date |

To combine them, the hedonic model prices a unit at today's market level and the index
forecast rolls that price forward (`hedonic.predict_future(model, units, index_growth=...)`).

## Quick start

```bash
pip install -r requirements.txt
python run_index_forecast.py                       # All Residential; also --series Landed / Non-Landed
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

## Possible next steps

* Add macro drivers (3M SORA or mortgage rates, GDP, unemployment, new-launch supply and
  the unsold pipeline, HDB resale index). This is likely the biggest gain, but the drivers
  must be forecast or lagged to avoid look-ahead.
* Forecast the region sub-indices (CCR/RCR/OCR) or a monthly index built from transactions,
  which gives about 3× more data points.
* Add amenity features to the hedonic model (distance to MRT stations and good schools).
  The URA `x`/`y` coordinates are SVY21 and can be joined to OneMap or LTA data.

Data: URA via data.gov.sg (Singapore Open Data Licence). This repository does not include
any URA transaction data.
