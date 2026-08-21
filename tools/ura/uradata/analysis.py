"""Benchmarks built on top of the flattened transaction records."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .model import Transaction, month_index
from .stats import median, percentile, rank_of

# Years from lease commencement to TOP for an EC. Validated against Rivercove
# Residences: lease commenced 2016, TOP 2 Oct 2020.
EC_LEASE_TO_TOP_YEARS = 4
EC_PRIVATISATION_YEARS = 10       # after TOP, for ECs launched before 8 May 2026


def project_benchmark(transactions, project: str, resale_only: bool = True):
    """Every transaction for one project, newest last."""
    needle = project.upper()
    rows = [t for t in transactions if t.project and needle in t.project.upper()]
    if resale_only:
        rows = [t for t in rows if t.is_resale()]
    return sorted(rows, key=lambda t: (t.month or 0, t.sqft))


def size_cohort(transactions, sqft: float, tolerance: float = 0.03):
    """Transactions within ``tolerance`` of a target size (default +/-3%)."""
    return [t for t in transactions if abs(t.sqft - sqft) / sqft <= tolerance]


@dataclass
class OfferAssessment:
    offer: float
    sqft: float
    offer_psf: float = field(init=False)
    cohort_n: int = 0
    cohort_median_price: float | None = None
    cohort_median_psf: float | None = None
    project_median_psf: float | None = None
    psf_percentile: float | None = None
    price_percentile: float | None = None
    fair_value: float | None = None
    verdict: str = ""

    def __post_init__(self):
        self.offer_psf = self.offer / self.sqft


def assess_offer(project_rows, offer: float, sqft: float,
                 tolerance: float = 0.03, fair_value: float | None = None) -> OfferAssessment:
    """Score an offer against a project's own transaction history."""
    assessment = OfferAssessment(offer=offer, sqft=sqft)
    cohort = size_cohort(project_rows, sqft, tolerance)
    assessment.cohort_n = len(cohort)
    if cohort:
        assessment.cohort_median_price = median([t.price for t in cohort])
        assessment.cohort_median_psf = median([t.psf for t in cohort])
    all_psf = [t.psf for t in project_rows if t.psf]
    all_price = [t.price for t in project_rows]
    assessment.project_median_psf = median(all_psf)
    assessment.psf_percentile = rank_of(assessment.offer_psf, all_psf)
    assessment.price_percentile = rank_of(offer, all_price)
    assessment.fair_value = fair_value

    reference = fair_value or assessment.cohort_median_price
    if reference:
        gap = assessment.offer / reference - 1
        if gap >= 0.04:
            assessment.verdict = "strong — materially above comparable evidence"
        elif gap >= -0.015:
            assessment.verdict = "fair — at comparable evidence"
        elif gap >= -0.05:
            assessment.verdict = "slightly low — worth one counter"
        else:
            assessment.verdict = "low — well below comparable evidence"
    return assessment


def monthly_medians(transactions, min_count: int = 1):
    """``{month_index: median psf}``, for detrending or plotting."""
    buckets = defaultdict(list)
    for t in transactions:
        if t.month is not None and t.psf:
            buckets[t.month].append(t.psf)
    return {m: median(v) for m, v in buckets.items() if len(v) >= min_count}


def largest_gap(project_rows):
    """Find the biggest hole in a project's transaction history.

    A project that sat dormant and then reopened — an EC clearing its MOP, a
    development finishing its launch phase — leaves a long gap with a handful of
    stale trades on the far side. Fitting a linear time trend across that gap
    extrapolates the old price level into the new regime and badly overstates
    drift. Callers should use this to pick a sensible ``since`` window.

    Returns ``(gap_months, resume_month)`` or ``None`` if there is no history.
    """
    months = sorted({t.month for t in project_rows if t.month is not None})
    if len(months) < 2:
        return None
    gaps = [(b - a, b) for a, b in zip(months, months[1:])]
    return max(gaps)


def since_month(project_rows, year_month_str: str):
    """Filter to transactions on or after a ``YYYY-MM`` string."""
    year, month = year_month_str.split("-")
    cutoff = int(year) * 12 + int(month) - 1
    return [t for t in project_rows if t.month is not None and t.month >= cutoff]


def absorption(project_rows, months: int = 12, total_units: int | None = None):
    """Resale turnover over the trailing ``months``."""
    if not project_rows:
        return {"sales": 0, "per_month": 0.0, "turnover_pct": None}
    latest = max(t.month for t in project_rows if t.month is not None)
    window = [t for t in project_rows if t.month is not None and latest - t.month < months]
    per_month = len(window) / months
    return {
        "sales": len(window),
        "per_month": per_month,
        "turnover_pct": (100.0 * len(window) / total_units) if total_units else None,
        "latest_month": latest,
    }


def privatisation_event_study(transactions, lease_to_top: int = EC_LEASE_TO_TOP_YEARS,
                              window_months: int = 24, min_pre: int = 8, min_post: int = 20,
                              control_min_count: int = 15):
    """Does EC psf step up when a project privatises at TOP + 10 years?

    Each project is compared against itself either side of its estimated
    privatisation month, after dividing out the EC market's own monthly median
    psf. That detrending matters: without it a project that privatised into a
    rising market looks like it earned a privatisation premium.

    TOP is estimated as lease commencement + ``lease_to_top`` years, because URA
    publishes tenure but not TOP. The estimate is the main source of error, so
    callers should sweep ``lease_to_top`` to check the result is stable.

    ``control_min_count`` is the number of sales a month needs before it is
    trusted as a market reference; months thinner than that are dropped from
    the control series rather than allowed to swing it.
    """
    ec_resales = [
        t for t in transactions
        if t.property_type == "Executive Condominium" and t.is_resale() and t.psf and t.month
    ]
    control = monthly_medians(ec_resales, min_count=control_min_count)

    by_project = defaultdict(list)
    for t in ec_resales:
        by_project[t.project].append(t)

    events = []
    for name, rows in by_project.items():
        lease = rows[0].lease_year
        if not lease:
            continue
        privatisation = (lease + lease_to_top + EC_PRIVATISATION_YEARS) * 12
        pre = [t for t in rows if -window_months <= t.month - privatisation < 0]
        post = [t for t in rows if 0 <= t.month - privatisation < window_months]
        if len(pre) < min_pre or len(post) < min_post:
            continue
        relative = lambda rs: [t.psf / control[t.month] for t in rs if t.month in control]
        before, after = relative(pre), relative(post)
        if not before or not after:
            continue
        before_median, after_median = median(before), median(after)
        events.append({
            "project": name,
            "lease_year": lease,
            "privatisation_year": lease + lease_to_top + EC_PRIVATISATION_YEARS,
            "n_pre": len(pre), "n_post": len(post),
            "relative_psf_before": before_median,
            "relative_psf_after": after_median,
            "lift": after_median / before_median - 1,
        })

    lifts = [e["lift"] for e in events]
    return {
        "events": sorted(events, key=lambda e: -e["lift"]),
        "n_events": len(events),
        "median_lift": median(lifts),
        "mean_lift": (sum(lifts) / len(lifts)) if lifts else None,
        "positive": sum(1 for x in lifts if x > 0),
    }
