"""
vdr.math.sequences — Exact recursive sequences.

    from vdr.math.sequences import fibonacci, bernoulli, lucas

    f30 = fibonacci(30)       # VDR(832040)
    b12 = bernoulli(12)       # VDR(-691, 2730)
    l10 = lucas(10)           # VDR(123)

All values exact. Bernoulli numbers cached on demand.
"""

from __future__ import annotations
from typing import List, Callable

from vdr.core import VDR

__all__ = [
    "fibonacci",
    "fibonacci_seq",
    "lucas",
    "lucas_seq",
    "catalan_seq",
    "bernoulli",
    "bernoulli_seq",
    "tribonacci",
    "tribonacci_seq",
    "rational_recurrence",
]


# ---------------------------------------------------------------------------
# Bernoulli cache
# ---------------------------------------------------------------------------

_bernoulli_cache = {}


# ---------------------------------------------------------------------------
# Fibonacci
# ---------------------------------------------------------------------------

def fibonacci(n):
    """
    Fibonacci number F(n).

    I: non-negative integer n
    O: F(n) as VDR

        fibonacci(30) -> VDR(832040)

    Uses iterative computation.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return VDR(n)

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return VDR(b)


def fibonacci_seq(n):
    """
    Fibonacci sequence F(0) through F(n).

    I: non-negative integer n
    O: list of VDR [F(0), F(1), ..., F(n)]
    """
    if n < 0:
        return []
    seq = [VDR(0)]
    if n == 0:
        return seq
    seq.append(VDR(1))
    for i in range(2, n + 1):
        seq.append(seq[i - 1] + seq[i - 2])
    return seq


# ---------------------------------------------------------------------------
# Lucas
# ---------------------------------------------------------------------------

def lucas(n):
    """
    Lucas number L(n).
    L(0)=2, L(1)=1, L(n)=L(n-1)+L(n-2).

    I: non-negative integer n
    O: L(n) as VDR

        lucas(10) -> VDR(123)
    """
    if n == 0:
        return VDR(2)
    if n == 1:
        return VDR(1)
    a, b = 2, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return VDR(b)


def lucas_seq(n):
    """Lucas sequence L(0) through L(n)."""
    if n < 0:
        return []
    seq = [VDR(2)]
    if n == 0:
        return seq
    seq.append(VDR(1))
    for i in range(2, n + 1):
        seq.append(seq[i - 1] + seq[i - 2])
    return seq


# ---------------------------------------------------------------------------
# Catalan sequence
# ---------------------------------------------------------------------------

def catalan_seq(n):
    """
    Catalan numbers C(0) through C(n).

    C(n) = C(2n, n) / (n+1)
    """
    from vdr.math.combinatorics import catalan
    return [catalan(k) for k in range(n + 1)]


# ---------------------------------------------------------------------------
# Bernoulli
# ---------------------------------------------------------------------------

def bernoulli(n):
    """
    Bernoulli number B(n).

    I: non-negative integer n
    O: B(n) as exact VDR

        bernoulli(0) -> VDR(1)
        bernoulli(1) -> VDR(-1, 2)
        bernoulli(12) -> VDR(-691, 2730)

    Uses the recurrence:
        B(0) = 1
        sum_{k=0}^{n} C(n+1, k) * B(k) = 0  for n >= 1

    Results cached for repeated access.
    """
    if n in _bernoulli_cache:
        return _bernoulli_cache[n]

    if n == 0:
        _bernoulli_cache[0] = VDR(1)
        return VDR(1)

    # odd Bernoulli numbers > 1 are zero
    if n > 1 and n % 2 == 1:
        _bernoulli_cache[n] = VDR(0)
        return VDR(0)

    # ensure all lower Bernoulli numbers are computed
    for k in range(n):
        if k not in _bernoulli_cache:
            bernoulli(k)

    # recurrence: sum_{k=0}^{n-1} C(n+1, k) * B(k) = -(n+1) * B(n)
    # so B(n) = -1/(n+1) * sum_{k=0}^{n-1} C(n+1, k) * B(k)
    from vdr.math.combinatorics import binom
    total = VDR(0)
    for k in range(n):
        total = total + binom(n + 1, k) * _bernoulli_cache[k]
    result = -total / VDR(n + 1)

    _bernoulli_cache[n] = result
    return result


def bernoulli_seq(n):
    """Bernoulli numbers B(0) through B(n)."""
    return [bernoulli(k) for k in range(n + 1)]


# ---------------------------------------------------------------------------
# Tribonacci
# ---------------------------------------------------------------------------

def tribonacci(n):
    """
    Tribonacci number T(n).
    T(0)=0, T(1)=0, T(2)=1, T(n)=T(n-1)+T(n-2)+T(n-3).

    I: non-negative integer n
    O: T(n) as VDR
    """
    if n <= 1:
        return VDR(0)
    if n == 2:
        return VDR(1)
    a, b, c = 0, 0, 1
    for _ in range(3, n + 1):
        a, b, c = b, c, a + b + c
    return VDR(c)


def tribonacci_seq(n):
    """Tribonacci sequence T(0) through T(n)."""
    if n < 0:
        return []
    seq = []
    a, b, c = 0, 0, 1
    for i in range(n + 1):
        if i == 0:
            seq.append(VDR(0))
        elif i == 1:
            seq.append(VDR(0))
        elif i == 2:
            seq.append(VDR(1))
        else:
            a, b, c = b, c, a + b + c
            seq.append(VDR(c))
    return seq


# ---------------------------------------------------------------------------
# General rational recurrence
# ---------------------------------------------------------------------------

def rational_recurrence(coeffs, initial, n):
    """
    General linear recurrence with VDR coefficients.

    x(k) = c0*x(k-1) + c1*x(k-2) + ... + c_{m-1}*x(k-m)

    I: coeffs = [c0, c1, ..., c_{m-1}], initial = [x(0), x(1), ..., x(m-1)],
       n = number of terms to generate
    O: list of VDR values [x(0), x(1), ..., x(n-1)]

        # a(n) = 3 - 2^(1-n)
        # via recurrence a(n) = (3/2)*a(n-1) - (1/2)*a(n-2)
        rational_recurrence(
            [VDR(3,2), VDR(-1,2)],
            [VDR(1), VDR(2)],
            14
        )
    """
    m = len(coeffs)
    if len(initial) != m:
        raise ValueError(
            "Need %d initial values, got %d" % (m, len(initial))
        )

    seq = list(initial)
    while len(seq) < n:
        val = VDR(0)
        for j, c in enumerate(coeffs):
            val = val + c * seq[len(seq) - 1 - j]
        seq.append(val)

    return seq[:n]
