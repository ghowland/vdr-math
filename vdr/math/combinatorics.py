"""
vdr.math.combinatorics — Exact combinatorial functions.

    from vdr.math.combinatorics import binom, bell, multinomial

    c = binom(20, 10)        # VDR(184756)
    b = bell(5)              # VDR(52)
    m = multinomial(10, [3,3,4])  # VDR(4200)

All results exact VDR integers or rationals.
"""

from __future__ import annotations
from typing import List

from vdr.core import VDR

__all__ = [
    "factorial",
    "falling_factorial",
    "binom",
    "stirling2",
    "bell",
    "derangement",
    "catalan",
    "catalan_gf",
    "multinomial",
]


def factorial(n):
    """n! as VDR."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return VDR(result)


def falling_factorial(n, k):
    """
    Falling factorial: n * (n-1) * ... * (n-k+1).

    I: VDR n, int k
    O: exact VDR
    """
    result = VDR(1)
    for i in range(k):
        result = result * (n - VDR(i))
    return result


def binom(n, k):
    """
    Binomial coefficient C(n, k).

    I: non-negative integers n, k
    O: C(n,k) as VDR

        binom(20, 10) -> VDR(184756)
    """
    if k < 0 or k > n:
        return VDR(0)
    if k == 0 or k == n:
        return VDR(1)
    # use smaller k for efficiency
    if k > n - k:
        k = n - k
    result = 1
    for i in range(k):
        result = result * (n - i)
        result = result // (i + 1)
    return VDR(result)


def stirling2(n, k):
    """
    Stirling number of the second kind S(n, k).
    Number of ways to partition n elements into k non-empty subsets.

    I: non-negative integers n, k
    O: S(n,k) as VDR

    Uses the explicit formula:
        S(n,k) = (1/k!) * sum_{j=0}^{k} (-1)^(k-j) * C(k,j) * j^n
    """
    if n == 0 and k == 0:
        return VDR(1)
    if n == 0 or k == 0:
        return VDR(0)
    if k > n:
        return VDR(0)
    if k == 1:
        return VDR(1)
    if k == n:
        return VDR(1)

    total = VDR(0)
    for j in range(k + 1):
        term = binom(k, j) * VDR(j ** n)
        if (k - j) % 2 == 0:
            total = total + term
        else:
            total = total - term

    # divide by k!
    kfact = 1
    for i in range(2, k + 1):
        kfact *= i
    return total / VDR(kfact)


def bell(n):
    """
    Bell number B(n): total number of partitions of n elements.

    I: non-negative integer n
    O: B(n) as VDR

        bell(5) -> VDR(52)

    Uses Bell triangle.
    """
    if n == 0:
        return VDR(1)

    # Bell triangle method
    row = [VDR(1)]
    for i in range(1, n + 1):
        new_row = [row[-1]]
        for j in range(1, i + 1):
            new_row.append(new_row[j - 1] + row[j - 1])
        row = new_row

    return row[0]


def derangement(n):
    """
    Derangement D(n): number of permutations with no fixed points.

    I: non-negative integer n
    O: D(n) as VDR

        derangement(7) -> VDR(1854)

    Uses recurrence: D(n) = (n-1)(D(n-1) + D(n-2))
    """
    if n == 0:
        return VDR(1)
    if n == 1:
        return VDR(0)

    d_prev2 = VDR(1)  # D(0)
    d_prev1 = VDR(0)  # D(1)
    for k in range(2, n + 1):
        d_curr = VDR(k - 1) * (d_prev1 + d_prev2)
        d_prev2 = d_prev1
        d_prev1 = d_curr
    return d_prev1


def catalan(n):
    """
    Catalan number C_n = C(2n,n) / (n+1).

    I: non-negative integer n
    O: C_n as VDR

        catalan(5) -> VDR(42)
    """
    return binom(2 * n, n) / VDR(n + 1)


def catalan_gf(x, terms):
    """
    Catalan generating function partial sum:
        sum_{n=0}^{terms} C_n * x^n

    I: VDR x, number of terms
    O: partial sum as VDR
    """
    total = VDR(0)
    x_power = VDR(1)
    for n in range(terms + 1):
        total = total + catalan(n) * x_power
        x_power = x_power * x
    return total


def multinomial(n, ks):
    """
    Multinomial coefficient: n! / (k1! * k2! * ... * km!)

    I: total n, list of group sizes [k1, k2, ..., km]
    O: multinomial coefficient as VDR

        multinomial(10, [3, 3, 4]) -> VDR(4200)
    """
    if sum(ks) != n:
        raise ValueError(
            "Group sizes %s do not sum to %d" % (ks, n)
        )
    # compute as product of binomials for numerical stability
    result = VDR(1)
    remaining = n
    for k in ks:
        result = result * binom(remaining, k)
        remaining -= k
    return result
