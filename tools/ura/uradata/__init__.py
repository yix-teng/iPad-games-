"""Tools for pulling and analysing URA private residential transaction data."""

from .client import URAClient, URAError
from .model import Transaction, flatten, parse_contract_date, month_index, lease_commencement_year
from .stats import percentile, median, ols, r_squared
from .hedonic import HedonicModel
from .analysis import (
    project_benchmark,
    assess_offer,
    OfferAssessment,
    privatisation_event_study,
    absorption,
    largest_gap,
    since_month,
)

__all__ = [
    "URAClient", "URAError",
    "Transaction", "flatten", "parse_contract_date", "month_index", "lease_commencement_year",
    "percentile", "median", "ols", "r_squared",
    "HedonicModel",
    "project_benchmark", "assess_offer", "OfferAssessment",
    "privatisation_event_study", "absorption", "largest_gap", "since_month",
]
