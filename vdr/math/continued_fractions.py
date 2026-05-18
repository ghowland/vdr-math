"""
vdr.math.continued_fractions — Exact continued fraction operations.

    from vdr.math.continued_fractions import to_cf, from_cf, sqrt_cf_period

    cf = to_cf(355, 113)              # [3, 7, 16]
    val = from_cf([3, 7, 16])         # VDR(355, 113)
    period = sqrt_cf_period(2)        # (1, [2])

All operations exact. Convergents are exact VDR rationals.
"""

from __future__ import annotations
from math import isqrt
from typing import List, Tuple, Optional

from vdr.core import VDR
from vdr.math.number_theory import convergents as _convergents

__all__ = [
    "to_cf",
    "from_cf",
    "convergents_from_cf",
    "sb_path",
    "sqrt_cf_period",
    "best_rational",
]


def to_cf(v, d):
    """
    Convert fraction v/d to continued fraction coefficients.

    I: numerator v, denominator d (integers)
    O: list of CF coefficients [a0; a1, a2, ...]

        to_cf(355, 113) -> [3, 7, 16]
    """
    if d == 0:
        raise ValueError("Denominator must be nonzero")
    coeffs = []
    while d != 0:
        q, r = divmod(v, d)
        coeffs.append(q)
        v, d = d, r
    return coeffs


def from_cf(coeffs):
    """
    Convert continued fraction coefficients to VDR rational.

    I: list of CF coefficients [a0, a1, ...]
    O: exact VDR rational

        from_cf([3, 7, 16]) -> VDR(355, 113)
    """
    if not coeffs:
        return VDR(0)

    # build from the bottom up
    # start with last coefficient
    result = VDR(coeffs[-1])
    for i in range(len(coeffs) - 2, -1, -1):
        # result = coeffs[i] + 1/result
        result = VDR(coeffs[i]) + VDR(1) / result
    return result


def convergents_from_cf(coeffs):
    """
    Compute all convergents from CF coefficients.

    I: list of CF coefficients
    O: list of VDR convergent fractions

        convergents_from_cf([3, 7, 16]) -> [VDR(3,1), VDR(22,7), VDR(355,113)]
    """
    return _convergents(coeffs)


def sb_path(p, q):
    """
    Stern-Brocot path for fraction p/q.

    I: numerator p, denominator q (positive, coprime)
    O: path string of L and R moves

        sb_path(3, 5) -> "LLRL"
    """
    if p <= 0 or q <= 0:
        raise ValueError("p and q must be positive")

    path = []
    # mediant tree walk
    lo_n, lo_d = 0, 1
    hi_n, hi_d = 1, 0  # represents infinity

    while True:
        med_n = lo_n + hi_n
        med_d = lo_d + hi_d

        # compare p/q with med_n/med_d via cross multiply
        lhs = p * med_d
        rhs = med_n * q

        if lhs == rhs:
            break
        elif lhs < rhs:
            path.append('L')
            hi_n, hi_d = med_n, med_d
        else:
            path.append('R')
            lo_n, lo_d = med_n, med_d

    return ''.join(path)


def sqrt_cf_period(n):
    """
    Continued fraction expansion of sqrt(n) for non-square n.

    I: positive non-square integer
    O: (integer_part, periodic_block) where sqrt(n) = [a0; a1, a2, ..., a_k, ...]
       and [a1, ..., a_k] is the repeating period.

        sqrt_cf_period(2) -> (1, [2])
        sqrt_cf_period(3) -> (1, [1, 2])
        sqrt_cf_period(7) -> (2, [1, 1, 1, 4])
    """
    s = isqrt(n)
    if s * s == n:
        raise ValueError("%d is a perfect square" % n)

    a0 = s
    period = []

    # standard algorithm for periodic CF of sqrt(n)
    m, d, a = 0, 1, a0

    seen = {}
    while True:
        m = d * a - m
        d = (n - m * m) // d
        a = (a0 + m) // d

        state = (m, d)
        if state in seen:
            break
        seen[state] = len(period)
        period.append(a)

    return (a0, period)


def best_rational(x, max_denom):
    """
    Best rational approximation to x with denominator <= max_denom.

    I: target VDR value, maximum denominator
    O: best approximation as VDR

    Uses continued fraction convergents.
    """
    frac = x.to_fraction()
    v, d = frac.numerator, frac.denominator

    # if denominator already small enough, return as-is
    if abs(d) <= max_denom:
        return x

    coeffs = to_cf(v, d)
    convs = convergents_from_cf(coeffs)

    # find last convergent with denominator <= max_denom
    best = VDR(0)
    for c in convs:
        if abs(c.d) <= max_denom:
            best = c
        else:
            break

    return best
