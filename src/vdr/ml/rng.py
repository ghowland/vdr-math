"""
vdr.ml.rng — Deterministic exact rational random number generator.

    from vdr.ml.rng import VDRRandom

    rng = VDRRandom(seed=42)
    val = rng.rand_fraction()    # exact rational in [0, 1)
    idx = rng.randbelow(10)      # integer in [0, 10)

Linear congruential generator with exact integer arithmetic.
Deterministic: same seed = same sequence on any platform.
"""

from __future__ import annotations
from typing import List

from vdr.core import VDR

__all__ = ["VDRRandom"]


class VDRRandom:
    """
    Deterministic pseudo-random number generator for VDR.

    Uses linear congruential generator (LCG):
        state_{n+1} = (a * state_n + c) mod m

    Constants from Numerical Recipes (period 2^32):
        a = 1664525
        c = 1013904223
        m = 2^32

    All operations exact integer arithmetic.
    Platform-independent: same seed, same sequence, always.
    """

    __slots__ = ("_state", "_a", "_c", "_m")

    def __init__(self, seed=1):
        """
        I: integer seed (determines entire sequence)
        """
        self._a = 1664525
        self._c = 1013904223
        self._m = 2 ** 32
        self._state = seed % self._m

    def next_int(self):
        """
        Advance state and return raw integer in [0, m).

        O: integer
        """
        self._state = (self._a * self._state + self._c) % self._m
        return self._state

    def randbelow(self, n):
        """
        Random integer in [0, n).

        I: upper bound n (positive integer)
        O: integer in [0, n)
        """
        if n <= 0:
            raise ValueError("n must be positive, got %d" % n)
        return self.next_int() % n

    def rand_fraction(self):
        """
        Random VDR rational in [0, 1).

        Returns exact rational with denominator 2^32.

        O: VDR in [0, 1)
        """
        return VDR(self.next_int(), self._m)

    def randint(self, lo, hi):
        """
        Random integer in [lo, hi] inclusive.

        I: lower bound lo, upper bound hi
        O: integer
        """
        if lo > hi:
            raise ValueError("lo must be <= hi")
        return lo + self.randbelow(hi - lo + 1)

    def shuffle_in_place(self, xs):
        """
        Fisher-Yates shuffle of a list.

        I: mutable list
        S: shuffles list in place
        """
        n = len(xs)
        for i in range(n - 1, 0, -1):
            j = self.randbelow(i + 1)
            xs[i], xs[j] = xs[j], xs[i]

    def permutation(self, n):
        """
        Random permutation of [0, 1, ..., n-1].

        I: size n
        O: list of integers
        """
        perm = list(range(n))
        self.shuffle_in_place(perm)
        return perm

    def rand_vec(self, dim, denom=100):
        """
        Random VDR Vec with entries in [-1, 1) with given denominator.

        I: dimension, denominator for precision
        O: Vec of exact VDR rationals
        """
        from vdr.linalg import Vec
        data = []
        for _ in range(dim):
            # value in [-denom, denom) / denom
            raw = self.randbelow(2 * denom) - denom
            data.append(VDR(raw, denom))
        return Vec(data)
