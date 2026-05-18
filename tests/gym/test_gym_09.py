"""Gym 09 — Differential equations. 10/10 in VDR-2."""

import pytest
from fractions import Fraction
from vdr.core import VDR
from vdr.math.differential_eq import (
    euler_solve, rk4_solve, picard_exp_coefficients, lotka_volterra_solve,
)


class TestEuler:
    def test_exponential(self):
        """dy/dx = y, y(0) = 1, h = 1/10, 10 steps -> y(1) = (11/10)^10."""
        traj = euler_solve(lambda x, y: y, VDR(1), VDR(0), VDR(1, 10), 10)
        y_final = traj[-1][1]
        expected = Fraction(11, 10) ** 10
        assert y_final.to_fraction() == expected


class TestRK4:
    def test_more_accurate(self):
        """RK4 should be much more accurate than Euler for same step count."""
        f = lambda x, y: y
        euler = euler_solve(f, VDR(1), VDR(0), VDR(1, 10), 10)
        rk4 = rk4_solve(f, VDR(1), VDR(0), VDR(1, 10), 10)

        # true e ≈ 2.71828...
        euler_err = abs(float(euler[-1][1].to_fraction()) - 2.718281828)
        rk4_err = abs(float(rk4[-1][1].to_fraction()) - 2.718281828)
        assert rk4_err < euler_err


class TestPicard:
    def test_exp_coefficients(self):
        """Picard for dy/dx=y gives 1/k! coefficients."""
        coeffs = picard_exp_coefficients(8)
        for k in range(9):
            fk = 1
            for i in range(2, k + 1):
                fk *= i
            assert coeffs[k] == VDR(1, fk)


class TestLotkaVolterra:
    def test_runs(self):
        """LV should run without error, all exact."""
        traj = lotka_volterra_solve(
            (VDR(10), VDR(5)), VDR(1, 100),
            VDR(1, 10), VDR(1, 100), VDR(1, 10), VDR(1, 100),
            3,
        )
        assert len(traj) == 4
        x, y = traj[-1]
        assert isinstance(x.to_fraction(), Fraction)
        