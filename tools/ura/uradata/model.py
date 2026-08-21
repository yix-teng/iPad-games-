"""Parsing URA's nested project/transaction payload into flat records."""

from __future__ import annotations

import re
from dataclasses import dataclass

SQM_TO_SQFT = 10.7639104

# URA encodes the sale channel as a single digit.
SALE_TYPE_NEW = "1"
SALE_TYPE_SUBSALE = "2"
SALE_TYPE_RESALE = "3"
SALE_TYPES = {SALE_TYPE_NEW: "New Sale", SALE_TYPE_SUBSALE: "Sub Sale", SALE_TYPE_RESALE: "Resale"}

_LEASE_YEAR = re.compile(r"commencing from (\d{4})")
_FLOOR_BAND = re.compile(r"^(\d{2})-(\d{2})$")


def parse_contract_date(contract: str | None) -> tuple[int, int] | None:
    """URA's ``contractDate`` is ``MMYY``. Return ``(year, month)`` or None."""
    if not contract or len(contract) != 4 or not contract.isdigit():
        return None
    month, year = int(contract[:2]), int(contract[2:])
    if not 1 <= month <= 12:
        return None
    return 2000 + year, month


def month_index(contract: str | None) -> int | None:
    """Months since Jan 2000 — a sortable, subtractable time axis."""
    parsed = parse_contract_date(contract)
    if parsed is None:
        return None
    year, month = parsed
    return year * 12 + month - 1


def year_month(contract: str | None) -> str:
    """``MMYY`` -> ``YYYY-MM`` for display and grouping."""
    parsed = parse_contract_date(contract)
    return "?" if parsed is None else f"{parsed[0]:04d}-{parsed[1]:02d}"


def lease_commencement_year(tenure: str | None) -> int | None:
    """Pull the lease start year out of ``99 yrs lease commencing from 2016``."""
    if not tenure:
        return None
    match = _LEASE_YEAR.search(tenure)
    return int(match.group(1)) if match else None


def floor_midpoint(floor_range: str | None) -> float | None:
    """``11-15`` -> 13.0. Returns None for '-' (landed) and malformed values."""
    if not floor_range:
        return None
    match = _FLOOR_BAND.match(floor_range.strip())
    if not match:
        return None
    return (int(match.group(1)) + int(match.group(2))) / 2


@dataclass(frozen=True)
class Transaction:
    project: str | None
    street: str | None
    segment: str | None          # CCR / RCR / OCR
    district: str | None
    property_type: str | None
    tenure: str | None
    sale_type: str | None
    floor_range: str | None
    contract: str | None         # MMYY
    price: float
    sqm: float
    units: int

    @property
    def sqft(self) -> float:
        return self.sqm * SQM_TO_SQFT

    @property
    def psf(self) -> float | None:
        return self.price / self.sqft if self.sqm else None

    @property
    def month(self) -> int | None:
        return month_index(self.contract)

    @property
    def year_month(self) -> str:
        return year_month(self.contract)

    @property
    def floor(self) -> float | None:
        return floor_midpoint(self.floor_range)

    @property
    def lease_year(self) -> int | None:
        return lease_commencement_year(self.tenure)

    @property
    def sale_type_name(self) -> str:
        return SALE_TYPES.get(self.sale_type or "", self.sale_type or "?")

    def is_resale(self) -> bool:
        return self.sale_type == SALE_TYPE_RESALE


def flatten(projects) -> list[Transaction]:
    """Explode URA's project -> transaction[] nesting into flat records.

    Rows with an unparseable price or area are dropped rather than guessed at.
    """
    out: list[Transaction] = []
    for project in projects:
        name = project.get("project")
        street = project.get("street")
        segment = project.get("marketSegment")
        for row in project.get("transaction") or []:
            try:
                price = float(row["price"])
                sqm = float(row["area"])
            except (KeyError, TypeError, ValueError):
                continue
            if price <= 0 or sqm <= 0:
                continue
            try:
                units = int(row.get("noOfUnits") or 1)
            except (TypeError, ValueError):
                units = 1
            out.append(Transaction(
                project=name, street=street, segment=segment,
                district=row.get("district"), property_type=row.get("propertyType"),
                tenure=row.get("tenure"), sale_type=row.get("typeOfSale"),
                floor_range=row.get("floorRange"), contract=row.get("contractDate"),
                price=price, sqm=sqm, units=units,
            ))
    return out
