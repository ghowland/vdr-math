"""Gym 14 — Fixed-point iteration (partial in VDR-2, killed by chaos cost)."""

import pytest
from fractions import Fraction
from vdr.core import VDR
from vdr.fn import make_newton_fn, make_iterative_fn


class TestNewtonSqrt2:
    def test_8_steps(self):
        fn = make_newton_fn("sqrt2", lambda x: (x + VDR(2) / x) / VDR(2))
        result = fn.expand(8)
        residual = result * result - VDR(2)
        assert abs(residual.to_fraction()) < Fraction(1, 10 ** 50)


class TestContraction:
    def test_converges_to_2(self):
        """f(x) = x/2 + 1 converges to fixed point 2."""
        fn = make_iterative_fn(
            "contract",
            lambda x: x / VDR(2) + VDR(1),
            VDR(100),
        )
        result = fn.expand(20)
        # x_n = 2 + 98/2^n
        expected = Fraction(2) + Fraction(98, 2 ** 20)
        assert result.to_fraction() == expected


class TestCollatz:
    def test_27_reaches_1(self):
        """Collatz(27) reaches 1 in 111 steps."""
        def collatz_step(x):
            frac = x.to_fraction()
            n = int(frac)
            if n % 2 == 0:
                return VDR(n // 2)
            else:
                return VDR(3 * n + 1)

        x = VDR(27)
        steps = 0
        while x != VDR(1) and steps < 200:
            x = collatz_step(x)
            steps += 1
        assert x == VDR(1)
        assert steps == 111
