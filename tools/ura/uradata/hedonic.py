"""A small hedonic price model for a single project.

Within one development, the things that move price are size, floor and when the
deal was struck. Fitting ``log(price)`` on those three gives a fair value for a
unit that has not itself transacted — which is exactly the position a seller
weighing an offer is in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .stats import ols, predict, r_squared, residual_sd


@dataclass(frozen=True)
class Valuation:
    price: float
    psf: float
    low: float          # -1 residual sd
    high: float         # +1 residual sd
    n: int
    r2: float
    residual_sd: float

    def assess(self, offer: float) -> str:
        if offer >= self.high:
            return "above the model's range"
        if offer >= self.price:
            return "above fair value, inside the range"
        if offer >= self.low:
            return "below fair value, inside the range"
        return "below the model's range"


class HedonicModel:
    """``log(price) ~ 1 + log(sqft) + floor + months``, fitted per project.

    Floor enters linearly: within a single development the storey premium is
    small and close to linear, and a band-dummy specification would burn degrees
    of freedom that a 100-odd transaction sample cannot spare.
    """

    FEATURES = ("intercept", "log_sqft", "floor", "months")

    def __init__(self, transactions):
        usable = [
            t for t in transactions
            if t.psf and t.floor is not None and t.month is not None and t.sqft > 0
        ]
        if len(usable) < len(self.FEATURES) + 2:
            raise ValueError(
                f"need at least {len(self.FEATURES) + 2} usable transactions, got {len(usable)}"
            )
        self.transactions = usable
        self.base_month = min(t.month for t in usable)
        design = [self._row(t.sqft, t.floor, t.month) for t in usable]
        target = [math.log(t.price) for t in usable]
        self.coefficients = ols(design, target)
        self.r2 = r_squared(design, target, self.coefficients)
        self.residual_sd = residual_sd(design, target, self.coefficients)
        self.n = len(usable)

    def _row(self, sqft: float, floor: float, month: int) -> list[float]:
        return [1.0, math.log(sqft), float(floor), float(month - self.base_month)]

    @property
    def monthly_drift(self) -> float:
        """Fitted price drift per month, as a fraction (0.002 == +0.2%/month)."""
        return math.exp(self.coefficients[3]) - 1

    @property
    def floor_premium(self) -> float:
        """Fitted price premium per storey, as a fraction."""
        return math.exp(self.coefficients[2]) - 1

    @property
    def size_elasticity(self) -> float:
        """% change in price per 1% change in area. Below 1 means bigger units
        carry a lower psf."""
        return self.coefficients[1]

    def value(self, sqft: float, floor: float, month: int) -> Valuation:
        log_price = predict(self.coefficients, self._row(sqft, floor, month))
        price = math.exp(log_price)
        return Valuation(
            price=price,
            psf=price / sqft,
            low=math.exp(log_price - self.residual_sd),
            high=math.exp(log_price + self.residual_sd),
            n=self.n,
            r2=self.r2,
            residual_sd=self.residual_sd,
        )
