"""
vdr.ml.logarithm — Exact logarithm for ML pipelines.

    from vdr.ml.logarithm import log_series, log1p_series

    ln_val = log_series(VDR(2), depth=20)        # ln(2)
    ln1p = log1p_series(VDR(1, 10), depth=20)    # ln(1.1)

Arctanh-based series for general positive arguments.
Direct series for arguments near 1.
"""

from __future__ import annotations

from vdr.core import VDR

__all__ = [
    "log1p_series",
    "log_series",
    "log_ratio_near_one",
]


def log1p_series(x, depth=16):
    """
    ln(1 + x) via Taylor series: x - x^2/2 + x^3/3 - ...

    Converges for -1 < x <= 1.

    I: x (VDR), depth (number of terms)
    O: exact rational partial sum

        log1p_series(VDR(1, 10), 20)  # ln(1.1)
    """
    if isinstance(x, int):
        x = VDR(x)

    total = VDR(0)
    x_power = x
    for k in range(1, depth + 1):
        if k % 2 == 1:
            total = total + x_power / VDR(k)
        else:
            total = total - x_power / VDR(k)
        x_power = x_power * x
    return total


def log_series(x, depth=16):
    """
    Natural logarithm via arctanh series.

    ln(x) = 2 * arctanh((x-1)/(x+1)) = 2 * sum u^(2k+1)/(2k+1)
    where u = (x-1)/(x+1).

    Converges for all x > 0. Fastest near x = 1.

    I: x (VDR, positive), depth
    O: exact rational partial sum of ln(x)

        log_series(VDR(2), 20)     # ln(2) ≈ 0.693...
        log_series(VDR(10), 30)    # ln(10)
    """
    if isinstance(x, int):
        x = VDR(x)

    frac = x.to_fraction()
    if frac <= 0:
        raise ValueError("log requires positive argument, got %s" % x)

    u = (x - VDR(1)) / (x + VDR(1))
    total = VDR(0)
    u_power = u
    for k in range(depth):
        n = 2 * k + 1
        total = total + u_power / VDR(n)
        u_power = u_power * u * u
    return VDR(2) * total


def log_ratio_near_one(num, den, depth=16):
    """
    ln(num/den) when num ≈ den, avoiding catastrophic cancellation.

    Uses ln(1 + (num-den)/den) = log1p((num-den)/den).
    Exact VDR subtraction — no cancellation.

    I: numerator num (VDR), denominator den (VDR), depth
    O: ln(num/den) as VDR

        log_ratio_near_one(VDR(101, 100), VDR(1), 20)  # ln(1.01)
    """
    if isinstance(num, int):
        num = VDR(num)
    if isinstance(den, int):
        den = VDR(den)

    diff = num - den
    ratio = diff / den
    return log1p_series(ratio, depth)
