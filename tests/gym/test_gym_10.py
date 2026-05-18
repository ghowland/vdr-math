"""Gym 10 — Optimization. 8/8 in VDR-2."""

import pytest
from vdr.core import VDR
from vdr.math.optimization import newton_optimize, simplex_2d, bisection


class TestNewtonOptimize:
    def test_quadratic(self):
        """Minimize x^2 - 4x: f'=2x-4, f''=2. Min at x=2."""
        x = newton_optimize(
            lambda x: VDR(2) * x - VDR(4),
            lambda x: VDR(2),
            VDR(0), 1,
        )
        assert x == VDR(2)


class TestSimplex:
    def test_basic(self):
        """Minimize x + y subject to x+y<=4, x>=0, y>=0."""
        # min [1, 1]^T x subject to [[1,1]] x <= [4]
        result = simplex_2d(
            [VDR(1), VDR(1)],
            [[VDR(1), VDR(1)]],
            [VDR(4)],
        )
        # minimum at origin
        assert result[0] == VDR(0)
        assert result[1] == VDR(0)


class TestBisection:
    def test_sqrt2(self):
        """Bisection for x^2 - 2 = 0 in [1, 2]."""
        result = bisection(lambda x: x * x - VDR(2), VDR(1), VDR(2), 30)
        residual = result * result - VDR(2)
        assert abs(float(residual.to_fraction())) < 1e-8
