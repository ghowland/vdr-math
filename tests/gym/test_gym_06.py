"""Gym 06 — Combinatorics. 31/31 in VDR-2."""

import pytest
from vdr.core import VDR
from vdr.math.combinatorics import (
    binom, stirling2, bell, derangement, catalan, multinomial, factorial,
)


class TestBinom:
    def test_c20_10(self):
        assert binom(20, 10) == VDR(184756)

    def test_symmetry(self):
        assert binom(10, 3) == binom(10, 7)

    def test_edges(self):
        assert binom(10, 0) == VDR(1)
        assert binom(10, 10) == VDR(1)

    def test_pascal(self):
        """C(n,k) = C(n-1,k-1) + C(n-1,k)."""
        for n in range(2, 15):
            for k in range(1, n):
                assert binom(n, k) == binom(n - 1, k - 1) + binom(n - 1, k)


class TestStirling:
    def test_known(self):
        assert stirling2(4, 2) == VDR(7)
        assert stirling2(5, 3) == VDR(25)

    def test_edges(self):
        assert stirling2(5, 1) == VDR(1)
        assert stirling2(5, 5) == VDR(1)


class TestBell:
    def test_small(self):
        assert bell(0) == VDR(1)
        assert bell(1) == VDR(1)
        assert bell(5) == VDR(52)

    def test_sum_stirling(self):
        """B(n) = sum S(n,k) for k=0..n."""
        for n in range(1, 7):
            total = VDR(0)
            for k in range(n + 1):
                total = total + stirling2(n, k)
            assert total == bell(n)


class TestDerangement:
    def test_small(self):
        assert derangement(0) == VDR(1)
        assert derangement(1) == VDR(0)
        assert derangement(2) == VDR(1)
        assert derangement(7) == VDR(1854)


class TestCatalan:
    def test_small(self):
        assert catalan(0) == VDR(1)
        assert catalan(1) == VDR(1)
        assert catalan(5) == VDR(42)


class TestMultinomial:
    def test_basic(self):
        assert multinomial(10, [3, 3, 4]) == VDR(4200)

    def test_binomial_case(self):
        assert multinomial(5, [2, 3]) == binom(5, 2)

    def test_sum_mismatch_raises(self):
        with pytest.raises(ValueError):
            multinomial(10, [3, 3, 3])
