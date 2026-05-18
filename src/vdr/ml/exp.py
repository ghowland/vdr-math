"""
vdr.ml.exp — Exact exponential function for ML pipelines.

    from vdr.ml.exp import exp_series, exp_range_reduced, exp_neg

    e_val = exp_series(VDR(1), depth=20)   # e to ~18 digits
    safe = exp_neg(VDR(5), depth=20)       # exp(-5), always positive

Taylor series with exact VDR rational arithmetic.
Range reduction for large arguments.
"""

from __future__ import annotations
from fractions import Fraction

from vdr.core import VDR

__all__ = [
    "exp_series",
    "exp_range_reduced",
    "exp_neg",
]


def exp_series(x, depth=16):
    """
    Taylor series for exp(x) = sum_{k=0}^{depth} x^k / k!

    I: x (VDR or int), depth (number of terms beyond constant)
    O: exact rational partial sum

    Super-geometric convergence: ~35 terms for 100 digits at x=1/2.

        exp_series(VDR(1), 20)        # e ≈ 2.71828...
        exp_series(VDR(1, 2), 16)     # exp(0.5)
        exp_series(VDR(-1), 20)       # exp(-1) ≈ 0.36788...
    """
    if isinstance(x, int):
        x = VDR(x)
    total = VDR(1)
    term = VDR(1)
    for k in range(1, depth + 1):
        term = term * x / VDR(k)
        total = total + term
    return total


def exp_range_reduced(x, depth=16):
    """
    Range-reduced exponential for large |x|.

    Uses exp(x) = exp(x - n*ln2) * 2^n where n = round(x / ln2).
    The reduced argument is in [-ln2/2, ln2/2] for fast convergence.

    For exact work: n is chosen so that the reduced argument is small,
    then exp of the small part is computed by Taylor, and the 2^n factor
    is an exact power of two (just changes D in VDR frame).

    I: x (VDR), depth
    O: exact rational approximation

        exp_range_reduced(VDR(10), 20)
    """
    if isinstance(x, int):
        x = VDR(x)

    # approximate x / ln(2) to find reduction count
    # ln(2) ~ 693/1000 for reduction planning
    frac = x.to_fraction()
    ln2_approx = Fraction(693, 1000)
    n = int(round(float(frac / ln2_approx)))

    if n == 0:
        return exp_series(x, depth)

    # reduced argument: x - n * ln(2)
    # use exact ln(2) from transcendental if available
    try:
        from vdr.math.transcendental import LN2
        ln2_exact = LN2
    except ImportError:
        # fallback: compute ln(2) via series
        ln2_exact = _ln2_series(depth)

    reduced = x - VDR(n) * ln2_exact
    exp_reduced = exp_series(reduced, depth)

    # multiply by 2^n
    if n >= 0:
        return exp_reduced * VDR(2 ** n)
    else:
        return exp_reduced / VDR(2 ** (-n))


def exp_neg(x, depth=16):
    """
    Compute exp(-|x|), ensuring positive result.

    Convenience for ML contexts where we need exp of negative values
    (softmax, sigmoid, etc.).

    I: x (VDR, non-negative), depth
    O: exp(-x) as VDR, exact rational, positive

        exp_neg(VDR(5), 20)
    """
    if isinstance(x, int):
        x = VDR(x)
    neg_x = -abs(x)
    return exp_range_reduced(neg_x, depth)


def _ln2_series(depth):
    """Fallback ln(2) computation via arctanh."""
    # ln(2) = 2 * arctanh(1/3) = 2 * sum_{k=0}^{depth} (1/3)^(2k+1) / (2k+1)
    u = VDR(1, 3)
    total = VDR(0)
    u_power = u
    for k in range(depth):
        n = 2 * k + 1
        total = total + u_power / VDR(n)
        u_power = u_power * u * u
    return VDR(2) * total
