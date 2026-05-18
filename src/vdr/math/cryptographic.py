"""
vdr.math.cryptographic — Exact modular arithmetic and cryptographic primitives.

    from vdr.math.cryptographic import rsa_keygen, rsa_encrypt, rsa_decrypt

    n, e, d = rsa_keygen(61, 53, 17)
    c = rsa_encrypt(42, e, n)
    m = rsa_decrypt(c, d, n)  # 42 exactly

All operations exact integer modular arithmetic.
Float categorically excluded from this domain.
"""

from __future__ import annotations
from math import gcd as _math_gcd
from typing import List, Tuple

__all__ = [
    "mod_exp",
    "extended_gcd",
    "mod_inverse",
    "chinese_remainder",
    "rsa_keygen",
    "rsa_encrypt",
    "rsa_decrypt",
    "baby_giant",
    "euler_totient",
    "is_prime",
    "miller_rabin_deterministic",
]


def mod_exp(base, exp, mod):
    """
    Fast modular exponentiation: base^exp mod mod.

    I: integers base, exp (non-negative), mod (positive)
    O: integer result

    Uses repeated squaring. O(log exp) multiplications.

        mod_exp(2, 10, 1000) -> 24
    """
    if mod == 1:
        return 0
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp //= 2
        base = (base * base) % mod
    return result


def extended_gcd(a, b):
    """
    Extended Euclidean algorithm.

    I: integers a, b
    O: (gcd, x, y) where a*x + b*y = gcd

        extended_gcd(35, 15) -> (5, 1, -2)
    """
    if a == 0:
        return (b, 0, 1)

    g, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1

    return (g, x, y)


def mod_inverse(a, m):
    """
    Modular multiplicative inverse: a^(-1) mod m.

    I: integer a, modulus m
    O: integer x such that (a * x) % m == 1

        mod_inverse(17, 3120) -> 2753
    """
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(
            "Modular inverse does not exist: gcd(%d, %d) = %d" % (a, m, g)
        )
    return x % m


def chinese_remainder(remainders, moduli):
    """
    Chinese Remainder Theorem reconstruction.

    I: list of remainders [r1, r2, ...], list of pairwise coprime moduli [m1, m2, ...]
    O: integer x such that x = ri (mod mi) for all i, 0 <= x < product(mi)

        chinese_remainder([2, 3, 2], [3, 5, 7]) -> 23
    """
    if len(remainders) != len(moduli):
        raise ValueError("remainders and moduli must have same length")

    # product of all moduli
    M = 1
    for m in moduli:
        M *= m

    result = 0
    for ri, mi in zip(remainders, moduli):
        Mi = M // mi
        yi = mod_inverse(Mi, mi)
        result = (result + ri * Mi * yi) % M

    return result


def euler_totient(n):
    """
    Euler's totient function phi(n).

    I: positive integer n
    O: integer count of values 1..n coprime to n
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


def rsa_keygen(p, q, e):
    """
    RSA key generation.

    I: primes p, q, public exponent e
    O: (n, e, d) where n = p*q, d = e^(-1) mod phi(n)

        rsa_keygen(61, 53, 17) -> (3233, 17, 2753)
    """
    n = p * q
    phi_n = (p - 1) * (q - 1)

    if _math_gcd(e, phi_n) != 1:
        raise ValueError(
            "e=%d is not coprime with phi(n)=%d" % (e, phi_n)
        )

    d = mod_inverse(e, phi_n)

    return (n, e, d)


def rsa_encrypt(m, e, n):
    """
    RSA encryption: c = m^e mod n.

    I: plaintext integer m, public exponent e, modulus n
    O: ciphertext integer c

        rsa_encrypt(42, 17, 3233) -> some ciphertext
    """
    if m < 0 or m >= n:
        raise ValueError("Message must be in range [0, n)")
    return mod_exp(m, e, n)


def rsa_decrypt(c, d, n):
    """
    RSA decryption: m = c^d mod n.

    I: ciphertext integer c, private exponent d, modulus n
    O: plaintext integer m

        rsa_decrypt(ciphertext, 2753, 3233) -> 42
    """
    return mod_exp(c, d, n)


def baby_giant(g, h, p):
    """
    Baby-step giant-step discrete logarithm.

    Find x such that g^x = h (mod p).

    I: generator g, target h, prime modulus p
    O: integer x, or raises ValueError if not found

        baby_giant(2, 8, 13) -> 3  (because 2^3 = 8 mod 13)
    """
    from math import isqrt

    n = isqrt(p) + 1

    # baby step: compute g^j mod p for j = 0..n-1
    baby = {}
    val = 1
    for j in range(n):
        baby[val] = j
        val = (val * g) % p

    # giant step factor: g^(-n) mod p
    g_inv_n = mod_exp(mod_inverse(g, p), n, p)

    # giant step
    gamma = h
    for i in range(n):
        if gamma in baby:
            return i * n + baby[gamma]
        gamma = (gamma * g_inv_n) % p

    raise ValueError(
        "Discrete log not found (g=%d, h=%d, p=%d)" % (g, h, p)
    )


def is_prime(n):
    """
    Simple primality test.

    I: integer n
    O: bool
    """
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def miller_rabin_deterministic(n):
    """
    Deterministic Miller-Rabin for n < 3,317,044,064,679,887,385,961,981.

    Uses the first 13 primes as witnesses, which is sufficient for
    all numbers below the bound above.

    I: integer n >= 2
    O: bool (True if prime)
    """
    if n < 2:
        return False
    if n < 4:
        return True

    # small primes check
    small = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    for p in small:
        if n == p:
            return True
        if n % p == 0:
            return False

    # write n-1 = d * 2^r
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    for a in small:
        if a >= n:
            continue
        x = mod_exp(a, d, n)
        if x == 1 or x == n - 1:
            continue
        found = False
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                found = True
                break
        if not found:
            return False

    return True
