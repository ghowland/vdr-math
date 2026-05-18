"""
vdr.math.number_theory — Exact integer and rational number theory.

    from vdr.math.number_theory import harmonic, euler_totient, farey

    h10 = harmonic(10)           # VDR(7381, 2520) exact
    phi = euler_totient(100)     # 40
    f5 = farey(5)                # all fractions p/q with q <= 5

All operations exact. No float. No truncation.
"""

from __future__ import annotations
from math import gcd as _math_gcd
from typing import List, Tuple

from vdr.core import VDR

__all__ = [
    "vdr_gcd",
    "vdr_lcm",
    "egyptian_fractions",
    "stern_brocot",
    "farey",
    "euler_totient",
    "harmonic",
    "vdr_pow",
    "vdr_mod",
    "convergents",
    "e_cf",
    "factorial",
]


def vdr_gcd(a, b):
    """
    GCD of two integer VDR values.

    I: two closed VDR with D=1
    O: GCD as VDR
    """
    return VDR(_math_gcd(abs(a.v), abs(b.v)))


def vdr_lcm(a, b):
    """LCM of two integer VDR values."""
    g = _math_gcd(abs(a.v), abs(b.v))
    if g == 0:
        return VDR(0)
    return VDR(abs(a.v * b.v) // g)


def factorial(n):
    """n! as VDR. Exact."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return VDR(result)


def egyptian_fractions(v, d, max_terms=20):
    """
    Decompose v/d into sum of unit fractions (greedy algorithm).

    I: numerator v, denominator d, max terms
    O: list of VDR unit fractions [1/a1, 1/a2, ...] summing to v/d

        egyptian_fractions(3, 7) -> [VDR(1,3), VDR(1,11), VDR(1,231)]
    """
    from fractions import Fraction
    frac = Fraction(v, d)
    result = []
    for _ in range(max_terms):
        if frac.numerator == 0:
            break
        # ceiling of denominator/numerator
        ceil_d = -(-frac.denominator // frac.numerator)
        result.append(VDR(1, ceil_d))
        frac = frac - Fraction(1, ceil_d)
    return result


def stern_brocot(depth):
    """
    Generate Stern-Brocot tree fractions at given depth.

    I: tree depth (0 = just 1/1)
    O: sorted list of VDR fractions

        stern_brocot(2) -> [VDR(1,3), VDR(1,2), VDR(2,3), VDR(1,1), ...]
    """
    # iterative mediant insertion
    fracs = [(0, 1), (1, 1)]  # 0/1 and 1/1 as boundaries
    for _ in range(depth):
        new_fracs = [fracs[0]]
        for i in range(len(fracs) - 1):
            a_n, a_d = fracs[i]
            b_n, b_d = fracs[i + 1]
            mediant = (a_n + b_n, a_d + b_d)
            new_fracs.append(mediant)
            new_fracs.append(fracs[i + 1])
        fracs = new_fracs

    # convert to VDR, skip 0/1 boundary
    result = []
    for n, d in fracs:
        if n > 0:
            result.append(VDR(n, d))
    return result


def farey(n):
    """
    Farey sequence F_n: all fractions p/q with 0 <= p/q <= 1, q <= n,
    in ascending order.

    I: order n
    O: list of VDR fractions

        farey(5) -> [VDR(0,1), VDR(1,5), VDR(1,4), ..., VDR(1,1)]
    """
    result = []
    a, b = 0, 1
    c, d = 1, n
    result.append(VDR(a, b))
    while c <= n:
        result.append(VDR(c, d))
        k = (n + b) // d
        a, b, c, d = c, d, k * c - a, k * d - b
    return result


def euler_totient(n):
    """
    Euler's totient function phi(n).

    I: positive integer
    O: int (count of integers 1..n coprime to n)

        euler_totient(100) -> 40
    """
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def harmonic(n):
    """
    Harmonic number H_n = 1 + 1/2 + 1/3 + ... + 1/n.

    I: positive integer n
    O: exact VDR rational

        harmonic(10) -> VDR(7381, 2520)
    """
    total = VDR(0)
    for k in range(1, n + 1):
        total = total + VDR(1, k)
    return total


def vdr_pow(base, exp):
    """
    Exact integer power: base^exp.

    I: VDR base, non-negative int exponent
    O: base^exp as VDR

    Uses repeated squaring for efficiency.
    """
    if exp == 0:
        return VDR(1)
    if exp < 0:
        # negative exponent: 1 / base^|exp|
        return VDR(1) / vdr_pow(base, -exp)

    result = VDR(1)
    b = base
    e = exp
    while e > 0:
        if e % 2 == 1:
            result = result * b
        b = b * b
        e //= 2
    return result


def vdr_mod(a, m):
    """
    Modular reduction: a mod m.

    I: integer VDR a (D=1), int modulus m
    O: VDR(a.v % m)
    """
    return VDR(a.v % m)


def convergents(cf_coeffs):
    """
    Compute convergents from continued fraction coefficients.

    I: list of CF coefficients [a0, a1, a2, ...]
    O: list of VDR convergents [p0/q0, p1/q1, ...]

    Uses the recurrence:
        p_n = a_n * p_{n-1} + p_{n-2}
        q_n = a_n * q_{n-1} + q_{n-2}
    """
    if not cf_coeffs:
        return []

    result = []
    p_prev2, p_prev1 = 0, 1
    q_prev2, q_prev1 = 1, 0

    for a in cf_coeffs:
        p = a * p_prev1 + p_prev2
        q = a * q_prev1 + q_prev2
        result.append(VDR(p, q))
        p_prev2, p_prev1 = p_prev1, p
        q_prev2, q_prev1 = q_prev1, q

    return result


def e_cf(n):
    """
    First n continued fraction coefficients of e.

    e = [2; 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8, ...]
    Pattern after a0=2: repeating (1, 2k, 1) for k=1,2,3,...

    I: number of coefficients
    O: list of ints
    """
    if n <= 0:
        return []
    coeffs = [2]
    k = 1
    while len(coeffs) < n:
        coeffs.append(1)
        if len(coeffs) < n:
            coeffs.append(2 * k)
        if len(coeffs) < n:
            coeffs.append(1)
        k += 1
    return coeffs[:n]
