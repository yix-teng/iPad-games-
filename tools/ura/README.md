# `uradata` — URA private residential transaction analysis

A zero-dependency Python package for pulling Singapore private residential
transaction data from the [URA Data Service](https://eservice.ura.gov.sg/maps/api/)
and benchmarking a price against comparable evidence.

Stdlib only — no `requests`, no `numpy`, no `pandas`. The least-squares fit is
hand-rolled in `stats.ols`.

## Quick start

```bash
export URA_ACCESS_KEY=<your-ura-api-key>
cd tools/ura

python3 -m uradata fetch                       # download + cache ~135k transactions
python3 -m uradata benchmark "RIVERCOVE RESIDENCES"
python3 -m uradata offer "RIVERCOVE RESIDENCES" \
    --price 1988000 --sqft 1163 --floor 12 --since 2025-10
python3 -m uradata eventstudy                  # test the EC privatisation premium
```

Run the tests with:

```bash
cd tools/ura && python3 -m unittest discover -s . -t .
```

## Commands

| Command | What it does |
|---|---|
| `fetch` | Downloads all four `PMI_Resi_Transaction` batches and caches them |
| `benchmark PROJECT` | Lists a project's transactions and its median psf by month |
| `offer PROJECT --price --sqft [--floor]` | Scores an offer against the project's own history; `--floor` enables the hedonic model |
| `eventstudy` | Tests whether EC psf steps up at the 10-year privatisation mark |

Both `benchmark` and `offer` accept `--since YYYY-MM` to restrict the window.

## Notes on the URA API

These are the things that cost time when writing against the live service:

* **Token**: `GET /uraDataService/insertNewToken/v1` with an `AccessKey` header.
  A browser-like `User-Agent` is **required** — without it the endpoint returns
  403. Tokens last for the day and are memoised by `URAClient`.
* **Data**: `GET /uraDataService/invokeUraDS/v1?service=<name>&batch=<n>` with
  both `AccessKey` and `Token` headers.
* **Encoding**: responses are *mostly* UTF-8, but some batches carry Latin-1
  bytes in project names. A plain `raw.decode("utf-8")` raises
  `UnicodeDecodeError` roughly 4.7 MB into batch 2. `client.decode_payload`
  falls back through `cp1252` and `latin-1`.
* **Errors are HTTP 200**: a failed call returns `{"Status": "Error", ...}` with
  a 200 status, so the status field must be checked explicitly.
* **Services**: `PMI_Resi_Transaction` (batches 1–4), `PMI_Resi_Rental_Median`
  and `PMI_Resi_Pipeline` work on a standard key. `PMI_Resi_Rental_Contract`
  and `PMI_Resi_Developer_Sales` return `Invalid service`.
* **Executive Condominiums have no rental data.** ECs under 10 years old sit
  under HDB rules, and no EC appears in `PMI_Resi_Rental_Median`.

### Field semantics

| Field | Meaning |
|---|---|
| `area` | Strata area of the unit, in **square metres** |
| `price` | Transacted price in SGD |
| `contractDate` | `MMYY` — so `0126` is January 2026 |
| `typeOfSale` | `1` new sale, `2` sub sale, `3` resale |
| `floorRange` | Storey band such as `11-15`; landed rows carry `-` |
| `tenure` | e.g. `99 yrs lease commencing from 2016` |
| `marketSegment` | `CCR` / `RCR` / `OCR` |

The data window is a rolling five years, so `fetch` on a later date returns a
later window. Transactions are stamped by **contract date**, and caveats take
weeks to lodge, so the most recent one or two months are always undercounted.

## The hedonic model

`HedonicModel` fits, within a single project:

```
log(price) ~ 1 + log(sqft) + floor + months_since_start
```

Fitted on Rivercove Residences (117 post-MOP resales) this gives R² = 0.959 and
a residual sd of 2.4%, which is tight enough to value a unit that has not
itself transacted.

Floor enters linearly rather than as band dummies: within one development the
storey premium is small and close to linear, and dummies would burn degrees of
freedom a 100-odd transaction sample cannot spare.

### The gap trap

**Fitting a time trend across a dormant period is the main way to get a wrong
answer here**, and it bit during development.

Rivercove's five-year window contains six stale transactions from 2022–2025
(the pre-MOP era, when only a handful of units could change hands) followed by
117 post-MOP resales from October 2025. Fitting all 123 together produced a
drift of **+0.73%/month — about +9% a year** — because the trend line was doing
the work of explaining the level shift between the two regimes. The same model
fitted on the post-October-2025 window gives **+0.21%/month**, and that matches
both the raw same-size repeat evidence and URA's own OCR index.

The difference is not academic: it moved the fair-value estimate for a
1,163 sqft unit by **$53,000**.

`analysis.largest_gap` detects this, and the `offer` command warns and suggests
a `--since` value whenever a gap of more than three months is still in the
sample. Always check that warning before quoting a drift figure.

## The EC privatisation premium

An Executive Condominium bought under the pre-May-2026 rules clears its MOP at
5 years and fully privatises at 10, at which point foreigners and companies may
buy it. Property marketing routinely treats that as a step-change in value.

`eventstudy` tests it. For each EC project, the median psf in the 24 months
after its estimated privatisation is compared against the 24 months before,
after dividing out the EC market's own monthly median psf — without that
detrending, a project that privatised into a rising market looks like it earned
a premium it did not.

URA publishes tenure but not TOP, so TOP is estimated as lease commencement +
4 years (validated against Rivercove: lease 2016, TOP October 2020). Because
that estimate is the dominant source of error, the command sweeps it:

| TOP proxy | min post-obs | events | median lift | mean | positive |
|---|---|---|---|---|---|
| lease+3 | 8 | 29 | +1.1% | +0.8% | 19/29 |
| lease+3 | 20 | 23 | +2.2% | +1.3% | 17/23 |
| lease+3 | 30 | 20 | +1.7% | +1.4% | 15/20 |
| lease+4 | 20 | 13 | +1.6% | +2.0% | 10/13 |
| lease+5 | 20 | 7 | −0.7% | −0.2% | 2/7 |

**The premium is real but small — roughly +1% to +2%, not the double digits the
marketing implies — and it is not robust to the TOP estimate.** A cross-sectional
check agrees: post-MOP ECs aged 5–10 years traded at a median 1,509 psf over the
last 12 months, while privatised 10–15 year ECs traded at 1,447 psf. Lease decay
cancels the privatisation gain.

Caveats: 20–29 events is a small sample; the window is a single five-year slice;
and projects that privatised near the window edges have truncated post-periods.

## Layout

```
tools/ura/
  uradata/
    client.py     URAClient — token, retries, encoding fallback, injectable opener
    model.py      Transaction record + flatten() + date/tenure/floor parsing
    stats.py      percentile, rank, OLS via normal equations, R², residual sd
    hedonic.py    HedonicModel — per-project log-price fit and valuation
    analysis.py   benchmarks, offer scoring, gap detection, event study
    cli.py        argparse entry point
  tests/          78 unit tests, no network access required
```

`URAClient` takes an injectable `opener`, so `tests/test_client.py` exercises
retry/backoff, the encoding fallback, token memoisation and error handling
against a scripted fake rather than the live service.
