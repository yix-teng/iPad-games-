"""HTTP client for the URA Data Service.

Two quirks of the live API that this client exists to paper over:

1. The token endpoint returns 403 unless a browser-like ``User-Agent`` is sent.
2. Some batches contain Latin-1 bytes in project names, so a plain
   ``bytes.decode("utf-8")`` raises ``UnicodeDecodeError`` partway through an
   otherwise valid 30 MB response.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

BASE_URL = "https://eservice.ura.gov.sg/uraDataService"

# The endpoint rejects requests that look automated.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Tried in order; URA is mostly UTF-8 but not exclusively.
ENCODINGS = ("utf-8", "cp1252", "latin-1")

# PMI_Resi_Transaction is served in four batches covering the last five years.
TRANSACTION_BATCHES = (1, 2, 3, 4)


class URAError(RuntimeError):
    """The URA service returned a non-success payload."""


def decode_payload(raw: bytes) -> dict:
    """Decode a URA response body, tolerating mixed encodings."""
    for encoding in ENCODINGS:
        try:
            return json.loads(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
    # Every candidate failed: keep the JSON rather than the exact bytes.
    return json.loads(raw.decode("utf-8", "replace"))


def _default_opener(url: str, headers: dict) -> bytes:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


class URAClient:
    """Minimal client for the URA Data Service.

    ``opener`` is injectable so the tests can exercise retry, decoding and
    error handling without touching the network.
    """

    def __init__(self, access_key: str | None = None, opener=_default_opener,
                 retries: int = 4, sleep=time.sleep, base_url: str = BASE_URL):
        self.access_key = access_key or os.environ.get("URA_ACCESS_KEY")
        if not self.access_key:
            raise URAError("No access key: pass access_key or set URA_ACCESS_KEY.")
        self._opener = opener
        self._retries = retries
        self._sleep = sleep
        self._base_url = base_url
        self._token: str | None = None

    def _get(self, url: str, headers: dict) -> dict:
        last: Exception | None = None
        for attempt in range(self._retries):
            try:
                return decode_payload(self._opener(url, headers))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last = exc
                if attempt < self._retries - 1:
                    self._sleep(2 ** (attempt + 1))   # 2s, 4s, 8s
        raise URAError(f"GET {url} failed after {self._retries} attempts: {last}")

    @property
    def token(self) -> str:
        """Fetch (and memoise) a daily access token."""
        if self._token is None:
            payload = self._get(
                f"{self._base_url}/insertNewToken/v1",
                {"AccessKey": self.access_key, "User-Agent": BROWSER_UA},
            )
            if payload.get("Status") != "Success" or not payload.get("Result"):
                raise URAError(f"Token request rejected: {payload.get('Message') or payload}")
            self._token = payload["Result"]
        return self._token

    def service(self, name: str, **params) -> list:
        """Call ``invokeUraDS`` for one service and return its Result list."""
        query = "".join(f"&{k}={v}" for k, v in params.items())
        payload = self._get(
            f"{self._base_url}/invokeUraDS/v1?service={name}{query}",
            {"AccessKey": self.access_key, "Token": self.token, "User-Agent": BROWSER_UA},
        )
        if payload.get("Status") != "Success":
            raise URAError(f"{name} failed: {payload.get('Message') or payload}")
        return payload.get("Result") or []

    def transactions(self, batches=TRANSACTION_BATCHES, progress=None) -> list:
        """Fetch every batch of PMI_Resi_Transaction (~5 years of sales)."""
        projects: list = []
        for batch in batches:
            result = self.service("PMI_Resi_Transaction", batch=batch)
            if progress:
                progress(batch, len(result))
            projects.extend(result)
        return projects

    def rental_medians(self) -> list:
        """Fetch PMI_Resi_Rental_Median (quarterly median rents by project)."""
        return self.service("PMI_Resi_Rental_Median")
