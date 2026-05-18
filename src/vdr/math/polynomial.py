"""
vdr.math.polynomial — Exact polynomial arithmetic over VDR.

Polynomials are represented as coefficient lists, constant term first:
    [a0, a1, a2, ..., an] represents a0 + a1*x + a2*x^2 + ... + an*x^n

    from vdr.math.polynomial import poly_eval, poly_mul, lagrange_interpolate

    p = [VDR(1), VDR(1), VDR(1)]  # 1 + x + x^2
    poly_eval(p, VDR(2))          # VDR(7)

All operations exact. Zero-testing is structural: coefficient is VDR(0) or not.
"""

from __future__ import annotations
from typing import List, Tuple, Optional

from vdr.core import VDR
from vdr.linalg import Mat

__all__ = [
    "poly_eval",
    "poly_add",
    "poly_sub",
    "poly_mul",
    "poly_scale",
    "poly_neg",
    "poly_degree",
    "poly_divmod",
    "poly_gcd",
    "rational_roots",
    "lagrange_interpolate",
    "char_poly_2x2",
    "poly_derivative",
    "poly_integral",
    "definite_integral",
    "poly_str",
    "poly_trim",
]


def _ensure_vdr(x):
    if isinstance(x, VDR):
        return x
    if isinstance(x, int):
        return VDR(x)
    raise TypeError("Expected VDR or int, got %s" % type(x).__name__)


def poly_trim(p):
    """Remove trailing zero coefficients."""
    result = list(p)
    while len(result) > 1 and result[-1] == VDR(0):
        result.pop()
    return result


def poly_degree(p):
    """Degree of polynomial. Zero polynomial has degree -1."""
    trimmed = poly_trim(p)
    if len(trimmed) == 1 and trimmed[0] == VDR(0):
        return -1
    return len(trimmed) - 1


def poly_eval(p, x):
    """
    Evaluate polynomial at x using Horner's method.

    I: coefficient list [a0, a1, ...], evaluation point x (VDR)
    O: p(x) as VDR, exact

        poly_eval([VDR(1), VDR(1), VDR(1)], VDR(2)) -> VDR(7)
    """
    x = _ensure_vdr(x)
    if not p:
        return VDR(0)
    result = _ensure_vdr(p[-1])
    for i in range(len(p) - 2, -1, -1):
        result = result * x + _ensure_vdr(p[i])
    return result


def poly_add(p, q):
    """
    Add two polynomials.

    I: two coefficient lists
    O: sum coefficient list, trimmed
    """
    n = max(len(p), len(q))
    result = []
    for i in range(n):
        a = _ensure_vdr(p[i]) if i < len(p) else VDR(0)
        b = _ensure_vdr(q[i]) if i < len(q) else VDR(0)
        result.append(a + b)
    return poly_trim(result)


def poly_sub(p, q):
    """Subtract two polynomials."""
    n = max(len(p), len(q))
    result = []
    for i in range(n):
        a = _ensure_vdr(p[i]) if i < len(p) else VDR(0)
        b = _ensure_vdr(q[i]) if i < len(q) else VDR(0)
        result.append(a - b)
    return poly_trim(result)


def poly_mul(p, q):
    """
    Multiply two polynomials.

    I: two coefficient lists
    O: product coefficient list
    """
    if not p or not q:
        return [VDR(0)]
    n = len(p) + len(q) - 1
    result = [VDR(0)] * n
    for i in range(len(p)):
        for j in range(len(q)):
            result[i + j] = result[i + j] + _ensure_vdr(p[i]) * _ensure_vdr(q[j])
    return poly_trim(result)


def poly_scale(p, s):
    """Multiply polynomial by scalar."""
    s = _ensure_vdr(s)
    return poly_trim([_ensure_vdr(c) * s for c in p])


def poly_neg(p):
    """Negate polynomial."""
    return [-_ensure_vdr(c) for c in p]


def poly_divmod(p, q):
    """
    Polynomial division with remainder.

    I: dividend p, divisor q (coefficient lists)
    O: (quotient, remainder) as coefficient lists

    Exact zero-testing at each step.
    """
    p = [_ensure_vdr(c) for c in p]
    q = [_ensure_vdr(c) for c in q]
    q = poly_trim(q)

    if poly_degree(q) < 0:
        raise ValueError("Division by zero polynomial")

    if poly_degree(p) < poly_degree(q):
        return [VDR(0)], list(p)

    # work with mutable copy
    rem = list(p)
    deg_q = poly_degree(q)
    lead_q = q[deg_q]

    quot_len = len(rem) - deg_q
    quot = [VDR(0)] * quot_len

    for i in range(quot_len - 1, -1, -1):
        if len(rem) - 1 < i + deg_q:
            continue
        coeff = rem[i + deg_q] / lead_q
        quot[i] = coeff
        for j in range(deg_q + 1):
            rem[i + j] = rem[i + j] - coeff * q[j]

    return poly_trim(quot), poly_trim(rem)


def poly_gcd(p, q):
    """
    Polynomial GCD via Euclidean algorithm.

    I: two polynomials as coefficient lists
    O: GCD polynomial (monic), exact

    Zero-testing is structural: VDR(0) or not.

        poly_gcd([VDR(-1),VDR(0),VDR(1)], [VDR(1),VDR(2),VDR(1)])
        -> represents (x+1)
    """
    a = [_ensure_vdr(c) for c in p]
    b = [_ensure_vdr(c) for c in q]

    while poly_degree(b) >= 0:
        _, r = poly_divmod(a, b)
        a = b
        b = r

    # make monic
    a = poly_trim(a)
    if not a or a[-1] == VDR(0):
        return [VDR(0)]
    lead = a[-1]
    return poly_trim([c / lead for c in a])


def rational_roots(p):
    """
    Find all rational roots of polynomial via rational root theorem.

    Tests +/-(factors of a0) / (factors of an).

    I: coefficient list
    O: list of VDR roots
    """
    p = poly_trim([_ensure_vdr(c) for c in p])
    if poly_degree(p) <= 0:
        return []

    a0 = p[0]
    an = p[-1]

    if a0 == VDR(0):
        # 0 is a root
        roots = [VDR(0)]
        # factor out x and recurse
        _, reduced = poly_divmod(p, [VDR(0), VDR(1)])
        roots.extend(rational_roots(reduced))
        return roots

    a0_abs = abs(a0.v)
    an_abs = abs(an.v)

    # factors
    def factors(n):
        if n == 0:
            return [1]
        n = abs(n)
        result = []
        for i in range(1, n + 1):
            if n % i == 0:
                result.append(i)
        return result

    num_factors = factors(a0_abs)
    den_factors = factors(an_abs)

    roots = []
    tested = set()
    for nf in num_factors:
        for df in den_factors:
            for sign in [1, -1]:
                candidate = VDR(sign * nf, df)
                key = (sign * nf, df)
                if key in tested:
                    continue
                tested.add(key)
                val = poly_eval(p, candidate)
                if val == VDR(0):
                    roots.append(candidate)

    return roots


def lagrange_interpolate(points):
    """
    Lagrange interpolation through given points.

    I: list of (x, y) pairs as (VDR, VDR) tuples
    O: polynomial coefficient list [a0, a1, ..., an]

        lagrange_interpolate([(VDR(0),VDR(1)), (VDR(1),VDR(3)), (VDR(2),VDR(7))])
        -> [VDR(1), VDR(1), VDR(1)]  # 1 + x + x^2
    """
    n = len(points)
    if n == 0:
        return [VDR(0)]

    # result accumulator
    result = [VDR(0)] * n

    for i in range(n):
        xi, yi = points[i]
        xi = _ensure_vdr(xi)
        yi = _ensure_vdr(yi)

        # build basis polynomial L_i(x) = product_{j!=i} (x - xj) / (xi - xj)
        basis = [VDR(1)]
        denom = VDR(1)
        for j in range(n):
            if j == i:
                continue
            xj = _ensure_vdr(points[j][0])
            # multiply basis by (x - xj)
            basis = poly_mul(basis, [-xj, VDR(1)])
            denom = denom * (xi - xj)

        # scale basis by yi / denom
        scale = yi / denom
        basis = poly_scale(basis, scale)

        # add to result
        result = poly_add(result, basis)

    return poly_trim(result)


def char_poly_2x2(m):
    """
    Characteristic polynomial of a 2x2 matrix.

    p(lambda) = lambda^2 - trace*lambda + det

    I: 2x2 Mat
    O: coefficient list [det, -trace, 1]
    """
    if m.nrows != 2 or m.ncols != 2:
        raise ValueError("char_poly_2x2 requires 2x2 matrix")

    tr = m.trace()
    d = m.det()

    return [d, -tr, VDR(1)]


def poly_derivative(p):
    """
    Symbolic differentiation of polynomial.

    I: coefficient list [a0, a1, a2, ..., an]
    O: derivative [a1, 2*a2, 3*a3, ..., n*an]
    """
    if len(p) <= 1:
        return [VDR(0)]
    result = []
    for i in range(1, len(p)):
        result.append(_ensure_vdr(p[i]) * VDR(i))
    return poly_trim(result)


def poly_integral(p, c=None):
    """
    Symbolic antiderivative of polynomial.

    I: coefficient list, optional constant of integration c
    O: antiderivative [c, a0, a1/2, a2/3, ...]
    """
    if c is None:
        c = VDR(0)
    c = _ensure_vdr(c)

    result = [c]
    for i in range(len(p)):
        result.append(_ensure_vdr(p[i]) / VDR(i + 1))
    return poly_trim(result)


def definite_integral(p, a, b):
    """
    Exact definite integral of polynomial from a to b.

    I: coefficient list, bounds a and b (VDR)
    O: integral value as VDR, exact

        definite_integral([VDR(0), VDR(0), VDR(1)], VDR(0), VDR(1))
        -> VDR(1, 3)  # integral of x^2 from 0 to 1
    """
    a = _ensure_vdr(a)
    b = _ensure_vdr(b)
    antideriv = poly_integral(p, VDR(0))
    return poly_eval(antideriv, b) - poly_eval(antideriv, a)


def poly_str(p):
    """
    Pretty-print polynomial as string.

        poly_str([VDR(1), VDR(-2), VDR(3)]) -> "1 - 2x + 3x^2"
    """
    p = [_ensure_vdr(c) for c in p]
    if not p:
        return "0"

    parts = []
    for i, c in enumerate(p):
        if c == VDR(0):
            continue
        # coefficient string
        if i == 0:
            parts.append(str(c))
        else:
            c_str = str(c)
            if c == VDR(1):
                c_str = ""
            elif c == VDR(-1):
                c_str = "-"
            # variable part
            if i == 1:
                var = "x"
            else:
                var = "x^%d" % i
            parts.append("%s%s" % (c_str, var))

    if not parts:
        return "0"

    # join with appropriate signs
    result = parts[0]
    for part in parts[1:]:
        if part.startswith("-"):
            result += " - " + part[1:]
        else:
            result += " + " + part
    return result
