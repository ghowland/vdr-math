"""
vdr.math.symbolic — Exact symbolic algebra over VDR.

    from vdr.math.symbolic import partial_fractions_simple, power_sum

    pf = partial_fractions_simple([VDR(1)], [VDR(1), VDR(2)])
    # 1/((x-1)(x-2)) = -1/(x-1) + 1/(x-2)

    s = power_sum(3, 100)  # 1^3 + 2^3 + ... + 100^3 exact

Rational function arithmetic, partial fractions, power sums.
All exact VDR arithmetic.
"""

from __future__ import annotations
from typing import List, Tuple

from vdr.core import VDR
from vdr.math.polynomial import (
    poly_eval, poly_mul, poly_add, poly_sub, poly_scale,
    poly_trim, poly_degree, poly_divmod, _ensure_vdr,
    poly_derivative, poly_integral, definite_integral,
)

__all__ = [
    "partial_fractions_simple",
    "ratfun_add",
    "ratfun_mul",
    "ratfun_eval",
    "power_sum",
]


def partial_fractions_simple(numerator, roots):
    """
    Partial fraction decomposition for simple (non-repeated) linear roots.

    Decomposes N(x) / ((x-r1)(x-r2)...(x-rn)) into
        c1/(x-r1) + c2/(x-r2) + ... + cn/(x-rn)

    I: numerator polynomial coefficients, list of root VDR values
    O: list of (coefficient, root) tuples [(c1, r1), (c2, r2), ...]

    Uses the cover-up method: ci = N(ri) / product_{j!=i}(ri - rj)

        partial_fractions_simple([VDR(1)], [VDR(1), VDR(2)])
        -> [(VDR(-1), VDR(1)), (VDR(1), VDR(2))]
        # meaning -1/(x-1) + 1/(x-2)
    """
    roots = [_ensure_vdr(r) for r in roots]
    n = len(roots)
    result = []

    for i in range(n):
        ri = roots[i]
        # N(ri)
        num_val = poly_eval(numerator, ri)
        # product_{j!=i} (ri - rj)
        denom_val = VDR(1)
        for j in range(n):
            if j != i:
                denom_val = denom_val * (ri - roots[j])
        ci = num_val / denom_val
        result.append((ci, ri))

    return result


def ratfun_add(pq1, pq2):
    """
    Add two rational functions.

    Each rational function is (numerator_poly, denominator_poly).

    I: two rational functions as (num, den) tuples of coefficient lists
    O: sum as (num, den) tuple

        ratfun_add(([VDR(1)], [VDR(-1), VDR(1)]),
                   ([VDR(1)], [VDR(-2), VDR(1)]))
        # 1/(x-1) + 1/(x-2)
    """
    n1, d1 = pq1
    n2, d2 = pq2
    # n1/d1 + n2/d2 = (n1*d2 + n2*d1) / (d1*d2)
    new_num = poly_add(poly_mul(n1, d2), poly_mul(n2, d1))
    new_den = poly_mul(d1, d2)
    return (new_num, new_den)


def ratfun_mul(pq1, pq2):
    """
    Multiply two rational functions.

    I: two rational functions as (num, den) tuples
    O: product as (num, den) tuple
    """
    n1, d1 = pq1
    n2, d2 = pq2
    return (poly_mul(n1, n2), poly_mul(d1, d2))


def ratfun_eval(pq, x):
    """
    Evaluate rational function at x.

    I: rational function as (num, den), evaluation point x
    O: N(x)/D(x) as VDR
    """
    n, d = pq
    x = _ensure_vdr(x)
    num_val = poly_eval(n, x)
    den_val = poly_eval(d, x)
    return num_val / den_val


def power_sum(k, n):
    """
    Power sum: 1^k + 2^k + ... + n^k.

    I: exponent k (non-negative int), upper limit n (positive int)
    O: exact VDR value

        power_sum(2, 100)  -> VDR(338350)
        power_sum(3, 100)  -> VDR(25502500)

    Computed directly by summation. For large n, Faulhaber's formula
    could be used but direct summation is exact and simple.
    """
    total = VDR(0)
    for i in range(1, n + 1):
        # i^k as integer
        term = VDR(i ** k)
        total = total + term
    return total
