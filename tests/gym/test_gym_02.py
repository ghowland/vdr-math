"""Gym 02 — Polynomial algebra. 22/23 in VDR-2 (1 test-authoring error)."""

import pytest
from vdr.core import VDR
from vdr.math.polynomial import (
    poly_eval, poly_add, poly_mul, poly_divmod, poly_gcd,
    lagrange_interpolate, char_poly_2x2, poly_derivative,
    poly_integral, definite_integral, rational_roots, poly_degree,
)
from vdr.linalg import Mat


class TestHorner:
    def test_eval_quadratic(self):
        # 1 + x + x^2 at x=2 -> 7
        p = [VDR(1), VDR(1), VDR(1)]
        assert poly_eval(p, VDR(2)) == VDR(7)

    def test_eval_at_half(self):
        # 2x^3 - 3x^2 + x - 5 at x=1/2
        p = [VDR(-5), VDR(1), VDR(-3), VDR(2)]
        result = poly_eval(p, VDR(1, 2))
        assert result == VDR(-5)  # correct answer is -5, not -19/4


class TestPolyArith:
    def test_add(self):
        p = [VDR(1), VDR(2)]
        q = [VDR(3), VDR(4), VDR(5)]
        s = poly_add(p, q)
        assert s == [VDR(4), VDR(6), VDR(5)]

    def test_mul(self):
        p = [VDR(1), VDR(1)]  # 1 + x
        q = [VDR(1), VDR(1)]  # 1 + x
        r = poly_mul(p, q)     # 1 + 2x + x^2
        assert r == [VDR(1), VDR(2), VDR(1)]

    def test_divmod(self):
        # (x^2 - 1) / (x - 1) = (x + 1) remainder 0
        p = [VDR(-1), VDR(0), VDR(1)]
        q = [VDR(-1), VDR(1)]
        quot, rem = poly_divmod(p, q)
        assert poly_eval(quot, VDR(0)) == VDR(1)  # constant term
        assert rem == [VDR(0)]


class TestPolyGCD:
    def test_basic(self):
        # gcd(x^2 - 1, x^2 + 2x + 1) = x + 1
        p = [VDR(-1), VDR(0), VDR(1)]
        q = [VDR(1), VDR(2), VDR(1)]
        g = poly_gcd(p, q)
        # monic gcd should be [1, 1] (representing 1 + x)
        assert poly_degree(g) == 1


class TestLagrange:
    def test_quadratic(self):
        """Through (0,1), (1,3), (2,7) -> 1 + x + x^2."""
        points = [(VDR(0), VDR(1)), (VDR(1), VDR(3)), (VDR(2), VDR(7))]
        p = lagrange_interpolate(points)
        assert poly_eval(p, VDR(0)) == VDR(1)
        assert poly_eval(p, VDR(1)) == VDR(3)
        assert poly_eval(p, VDR(2)) == VDR(7)


class TestCayleyHamilton:
    def test_2x2(self):
        m = Mat.from_ints([[1, 2], [2, 3]])
        cp = char_poly_2x2(m)
        # p(lambda) = lambda^2 - 4*lambda - 1
        # p(M) should be zero matrix
        I = Mat.identity(2)
        result = m.matmul(m) + m.scale(cp[1]) + I.scale(cp[0])
        assert result == Mat.zero(2, 2)


class TestCalculus:
    def test_derivative(self):
        p = [VDR(1), VDR(3), VDR(5)]  # 1 + 3x + 5x^2
        dp = poly_derivative(p)        # 3 + 10x
        assert dp == [VDR(3), VDR(10)]

    def test_integral(self):
        p = [VDR(3), VDR(10)]  # 3 + 10x
        ip = poly_integral(p)  # 0 + 3x + 5x^2
        assert ip == [VDR(0), VDR(3), VDR(5)]

    def test_definite_integral_x2(self):
        p = [VDR(0), VDR(0), VDR(1)]  # x^2
        result = definite_integral(p, VDR(0), VDR(1))
        assert result == VDR(1, 3)
