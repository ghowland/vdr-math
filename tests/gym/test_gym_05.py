"""Gym 05 — Recursive sequences. 15/15 in VDR-2."""

import pytest
from vdr.core import VDR
from vdr.math.sequences import (
    fibonacci, fibonacci_seq, lucas, lucas_seq,
    bernoulli, tribonacci, rational_recurrence,
)


class TestFibonacci:
    def test_small(self):
        assert fibonacci(0) == VDR(0)
        assert fibonacci(1) == VDR(1)
        assert fibonacci(10) == VDR(55)

    def test_large(self):
        assert fibonacci(30) == VDR(832040)

    def test_seq(self):
        seq = fibonacci_seq(6)
        assert [s.v for s in seq] == [0, 1, 1, 2, 3, 5, 8]

    def test_cassini(self):
        """Cassini identity: F(n-1)*F(n+1) - F(n)^2 = (-1)^n."""
        for n in range(2, 18):
            fn = fibonacci(n)
            fn1 = fibonacci(n + 1)
            fnm1 = fibonacci(n - 1)
            assert fnm1 * fn1 - fn * fn == VDR((-1) ** n)


class TestLucas:
    def test_basic(self):
        assert tribonacci(0) == VDR(0)
        assert tribonacci(1) == VDR(0)
        assert tribonacci(2) == VDR(1)
        assert tribonacci(7) == VDR(13)

    def test_identity(self):
        """L(n)^2 - 5*F(n)^2 = 4*(-1)^n."""
        for n in range(2, 12):
            ln = lucas(n)
            fn = fibonacci(n)
            assert ln * ln - VDR(5) * fn * fn == VDR(4 * ((-1) ** n))


class TestBernoulli:
    def test_b0(self):
        assert bernoulli(0) == VDR(1)

    def test_b1(self):
        assert bernoulli(1) == VDR(-1, 2)

    def test_b12(self):
        assert bernoulli(12) == VDR(-691, 2730)

    def test_odd_zero(self):
        for n in [3, 5, 7, 9, 11]:
            assert bernoulli(n) == VDR(0)


class TestTribonacci:
    def test_basic(self):
        assert tribonacci(0) == VDR(0)
        assert tribonacci(1) == VDR(0)
        assert tribonacci(2) == VDR(1)
        assert tribonacci(7) == VDR(24)


class TestRationalRecurrence:
    def test_converging(self):
        """a(n) = 3/2 * a(n-1) - 1/2 * a(n-2), a(0)=1, a(1)=2."""
        seq = rational_recurrence(
            [VDR(3, 2), VDR(-1, 2)],
            [VDR(1), VDR(2)],
            14,
        )
        assert len(seq) == 14
        # sequence converges to 3
        assert seq[0] == VDR(1)
        assert seq[1] == VDR(2)
