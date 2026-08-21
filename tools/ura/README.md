# URA private residential transaction price checker

Pulls the last 5 years of private residential transactions from the URA Data
Service (`PMI_Resi_Transaction`, batches 1–4) and benchmarks a given price
against comparable transactions.

## Usage

```bash
export URA_ACCESS_KEY=<your-ura-api-key>

# download + cache all four batches (~135k transactions)
python3 ura_prices.py --refresh

# benchmark a price against a size band in one project
python3 ura_prices.py --project "THE ARDEN" --price 1988000
python3 ura_prices.py --district 19 --sqft-min 700 --sqft-max 850 --price 1988000
```

The API key is read from `URA_ACCESS_KEY` and never written to disk.
`ura_cache.json` (the downloaded data) is gitignored.

## Notes on the API

* Token: `GET /uraDataService/insertNewToken/v1` with an `AccessKey` header —
  a browser-like `User-Agent` header is required or the request is rejected.
* Data: `GET /uraDataService/invokeUraDS/v1?service=PMI_Resi_Transaction&batch=N`
  with both `AccessKey` and `Token` headers. Tokens are valid for the day.
* Responses are mostly UTF-8 but some batches contain Latin-1 bytes in project
  names, so the client falls back through `cp1252`/`latin-1`.
* `area` is the unit's strata area in **sqm**; `price` is the transacted price.
* `typeOfSale`: 1 = new sale, 2 = sub sale, 3 = resale.
* `contractDate` is `MMYY`.
