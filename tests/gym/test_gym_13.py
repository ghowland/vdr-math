"""Gym 13 — Symbolic algebra. 20/20 in VDR-2."""

import pytest
from fractions import Fraction
from vdr.core import VDR
from vdr.math.symbolic import (
    partial_fractions_simple, ratfun_add, ratfun_mul, ratfun_eval, power_sum,
)
from vdr.math.polynomial import (
    poly_eval, poly_derivative, poly_integral, definite_integral,
)


class TestPartialFractions:
    def test_simple(self):
        """1 / ((x-1)(x-2)) = -1/(x-1) + 1/(x-2)."""
        pf = partial_fractions_simple([VDR(1)], [VDR(1), VDR(2)])
        # coefficients
        c1 = pf[0][0]
        c2 = pf[1][0]
        assert c1 == VDR(-1)
        assert c2 == VDR(1)

    def test_verify_at_point(self):
        """Evaluate original and decomposition at x=5."""
        pf = partial_fractions_simple([VDR(1)], [VDR(1), VDR(2)])
        # original: 1 / ((5-1)(5-2)) = 1/12
        original = VDR(1) / (VDR(4) * VDR(3))
        # decomposition: -1/(5-1) + 1/(5-2) = -1/4 + 1/3 = 1/12
        decomp = pf[0][0] / (VDR(5) - pf[0][1]) + pf[1][0] / (VDR(5) - pf[1][1])
        assert original == decomp


class TestRatFun:
    def test_add(self):
        """1/(x-1) + 1/(x-2) via ratfun_add."""
        pq1 = ([VDR(1)], [VDR(-1), VDR(1)])   # 1/(x-1)
        pq2 = ([VDR(1)], [VDR(-2), VDR(1)])   # 1/(x-2)
        num, den = ratfun_add(pq1, pq2)
        # at x=5: 1/4 + 1/3 = 7/12
        val = ratfun_eval((num, den), VDR(5))
        assert val == VDR(7, 12)

    def test_mul(self):
        pq1 = ([VDR(1)], [VDR(-1), VDR(1)])
        pq2 = ([VDR(1)], [VDR(-2), VDR(1)])
        num, den = ratfun_mul(pq1, pq2)
        # 1/((x-1)(x-2)) at x=5 = 1/12
        val = ratfun_eval((num, den), VDR(5))
        assert val == VDR(1, 12)


class TestPowerSum:
    def test_s1(self):
        """1 + 2 + ... + 100 = 5050."""
        assert power_sum(1, 100) == VDR(5050)

    def test_s2(self):
        """1^2 + 2^2 + ... + 100^2 = 338350."""
        assert power_sum(2, 100) == VDR(338350)

    def test_s3(self):
        """1^3 + ... + 100^3 = 25502500."""
        assert power_sum(3, 100) == VDR(25502500)

    def test_s1_squared_equals_s3(self):
        """S_1(n)^2 == S_3(n) (Nicomachus' theorem)."""
        for n in [10, 20, 50]:
            s1 = power_sum(1, n)
            s3 = power_sum(3, n)
            assert s1 * s1 == s3


class TestExactIntegrals:
    def test_x2_0_to_1(self):
        p = [VDR(0), VDR(0), VDR(1)]
        assert definite_integral(p, VDR(0), VDR(1)) == VDR(1, 3)

    def test_x3_0_to_2(self):
        p = [VDR(0), VDR(0), VDR(0), VDR(1)]
        assert definite_integral(p, VDR(0), VDR(2)) == VDR(4)

    def test_rational_bounds(self):
        p = [VDR(0), VDR(0), VDR(1)]
        result = definite_integral(p, VDR(1, 3), VDR(2, 3))
        expected = Fraction(2, 3) ** 3 / 3 - Fraction(1, 3) ** 3 / 3
        assert result.to_fraction() == expected
