"""Small statistics helpers — stdlib only, no numpy."""

from __future__ import annotations

import math


def percentile(values, p: float) -> float | None:
    """Linear-interpolated percentile. ``p`` in [0, 1]. Input need not be sorted."""
    if not values:
        return None
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"percentile p must be in [0, 1], got {p}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * p
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def median(values) -> float | None:
    return percentile(values, 0.5)


def rank_of(value: float, values) -> float | None:
    """Percentage of ``values`` strictly below ``value`` (0-100)."""
    if not values:
        return None
    return 100.0 * sum(1 for v in values if v < value) / len(values)


def ols(design, target):
    """Least squares via the normal equations, solved with Gaussian elimination.

    ``design`` is a list of feature rows (already including any intercept column),
    ``target`` the response vector. Returns the coefficient list.
    Raises ValueError on a singular / rank-deficient system.
    """
    if not design:
        raise ValueError("empty design matrix")
    n_features = len(design[0])
    if len(design) != len(target):
        raise ValueError("design and target lengths differ")
    if len(design) < n_features:
        raise ValueError("fewer observations than features")

    # Build the augmented normal-equation matrix [X'X | X'y].
    matrix = [[0.0] * (n_features + 1) for _ in range(n_features)]
    for row, y in zip(design, target):
        for i in range(n_features):
            for j in range(n_features):
                matrix[i][j] += row[i] * row[j]
            matrix[i][n_features] += row[i] * y

    # Gaussian elimination with partial pivoting.
    for col in range(n_features):
        pivot = max(range(col, n_features), key=lambda r: abs(matrix[r][col]))
        if abs(matrix[pivot][col]) < 1e-12:
            raise ValueError("singular design matrix (collinear features?)")
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        pivot_value = matrix[col][col]
        for j in range(col, n_features + 1):
            matrix[col][j] /= pivot_value
        for r in range(n_features):
            if r == col:
                continue
            factor = matrix[r][col]
            if factor:
                for j in range(col, n_features + 1):
                    matrix[r][j] -= factor * matrix[col][j]
    return [matrix[i][n_features] for i in range(n_features)]


def predict(coefficients, row) -> float:
    return sum(c * x for c, x in zip(coefficients, row))


def r_squared(design, target, coefficients) -> float:
    mean = sum(target) / len(target)
    ss_total = sum((y - mean) ** 2 for y in target)
    ss_residual = sum((y - predict(coefficients, row)) ** 2 for row, y in zip(design, target))
    if ss_total == 0:
        return 1.0
    return 1.0 - ss_residual / ss_total


def residual_sd(design, target, coefficients) -> float:
    """Residual standard deviation, with the usual n - k correction."""
    dof = len(target) - len(coefficients)
    if dof <= 0:
        return 0.0
    ss_residual = sum((y - predict(coefficients, row)) ** 2 for row, y in zip(design, target))
    return math.sqrt(ss_residual / dof)
