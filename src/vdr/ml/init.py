"""
vdr.ml.init — Exact rational weight initialization for neural networks.

    from vdr.ml.init import rational_uniform_mat, xavier_like_mat, zero_bias

    W = xavier_like_mat(4, 3, denom=100, seed=42)
    b = zero_bias(3)

All initial weights are exact VDR rationals. Deterministic: same seed
produces identical weights on any platform.
"""

from __future__ import annotations
from typing import Optional
from math import isqrt

from vdr.core import VDR
from vdr.linalg import Vec, Mat
from vdr.ml.rng import VDRRandom

__all__ = [
    "rational_uniform_vec",
    "rational_uniform_mat",
    "xavier_like_mat",
    "zero_bias",
    "constant_vec",
    "constant_mat",
]


def rational_uniform_vec(dim, denom=100, seed=1, lo=-1, hi=1):
    """
    Random Vec with entries uniformly distributed in [lo, hi).

    Each entry is an exact rational with the given denominator.

    I: dimension, denominator, seed, range [lo, hi)
    O: Vec of exact VDR rationals

        v = rational_uniform_vec(4, denom=100, seed=42)
    """
    rng = VDRRandom(seed)
    span = hi - lo
    data = []
    for _ in range(dim):
        raw = rng.randbelow(span * denom)
        val = VDR(lo * denom + raw, denom)
        data.append(val)
    return Vec(data)


def rational_uniform_mat(nrows, ncols, denom=100, seed=1, lo=-1, hi=1):
    """
    Random Mat with entries uniformly distributed in [lo, hi).

    I: dimensions, denominator, seed, range
    O: Mat of exact VDR rationals

        W = rational_uniform_mat(3, 4, denom=100, seed=42)
    """
    rng = VDRRandom(seed)
    span = hi - lo
    rows = []
    for _ in range(nrows):
        row = []
        for _ in range(ncols):
            raw = rng.randbelow(span * denom)
            val = VDR(lo * denom + raw, denom)
            row.append(val)
        rows.append(row)
    return Mat(rows)


def xavier_like_mat(nrows, ncols, denom=100, seed=1):
    """
    Xavier-like initialization with rational bounds.

    Standard Xavier uniform: U[-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]

    We approximate the bound as a rational: isqrt(6*denom^2 / (fan_in+fan_out)) / denom.
    This gives a rational approximation of the Xavier bound.

    I: dimensions nrows (fan_out) x ncols (fan_in), denominator, seed
    O: Mat of exact VDR rationals in approximately Xavier range

        W = xavier_like_mat(4, 3, denom=100, seed=42)
    """
    fan_in = ncols
    fan_out = nrows
    fan_sum = fan_in + fan_out

    # bound ≈ sqrt(6 / fan_sum)
    # as rational: isqrt(6 * denom^2 / fan_sum) / denom
    # ensure integer under sqrt
    under_sqrt = 6 * denom * denom // fan_sum
    bound_num = isqrt(under_sqrt) if under_sqrt > 0 else 1
    # bound = bound_num / denom

    rng = VDRRandom(seed)
    rows = []
    for _ in range(nrows):
        row = []
        for _ in range(ncols):
            # uniform in [-bound_num, bound_num) / denom
            raw = rng.randbelow(2 * bound_num) - bound_num
            row.append(VDR(raw, denom))
        rows.append(row)
    return Mat(rows)


def zero_bias(dim):
    """
    Zero bias vector.

    I: dimension
    O: Vec of zeros

        b = zero_bias(4)  # Vec([VDR(0), VDR(0), VDR(0), VDR(0)])
    """
    return Vec.zero(dim)


def constant_vec(dim, value=None):
    """
    Constant-value vector.

    I: dimension, value (VDR or int, default 0)
    O: Vec with all entries equal to value
    """
    if value is None:
        value = VDR(0)
    if isinstance(value, int):
        value = VDR(value)
    return Vec([value] * dim)


def constant_mat(nrows, ncols, value=None):
    """
    Constant-value matrix.

    I: dimensions, value (VDR or int, default 0)
    O: Mat with all entries equal to value
    """
    if value is None:
        value = VDR(0)
    if isinstance(value, int):
        value = VDR(value)
    return Mat([[value] * ncols for _ in range(nrows)])
