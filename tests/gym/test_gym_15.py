"""Gym 15 — Chaos and sensitivity (partial in VDR-2, killed by chaos cost)."""

import pytest
from vdr.core import VDR
from vdr.math.chaos import (
    tent_map, bernoulli_shift, arnold_cat, logistic_map,
    iterate_map, iterate_map_2d, detect_period, detect_period_2d,
    lyapunov_product,
)


class TestTentMap:
    def test_period_3_on_1_7(self):
        """Tent map on 1/7 has period 3, exact forever."""
        orbit = iterate_map(tent_map, VDR(1, 7), 20)
        period = detect_period(orbit)
        assert period == 3

    def test_orbit_values(self):
        x = VDR(1, 7)
        x = tent_map(x)
        assert x == VDR(2, 7)
        x = tent_map(x)
        assert x == VDR(4, 7)
        x = tent_map(x)
        assert x == VDR(6, 7)
        x = tent_map(x)
        assert x == VDR(2, 7)  # back to period start


class TestBernoulliShift:
    def test_period_on_1_3(self):
        orbit = iterate_map(bernoulli_shift, VDR(1, 3), 10)
        period = detect_period(orbit)
        assert period == 2

    def test_orbit(self):
        x = bernoulli_shift(VDR(1, 3))
        assert x == VDR(2, 3)
        x = bernoulli_shift(x)
        assert x == VDR(1, 3)


class TestArnoldCat:
    def test_period_on_rationals(self):
        orbit = iterate_map_2d(arnold_cat, VDR(1, 7), VDR(3, 11), 50)
        period = detect_period_2d(orbit)
        assert period is not None
        assert period <= 42  # known to be period 40 for these values


class TestLogistic:
    def test_fixed_point(self):
        """r=2: fixed point at 1/2."""
        x = VDR(1, 4)
        for _ in range(20):
            x = logistic_map(x, VDR(2))
        # should converge to 1/2
        assert abs(float(x.to_fraction()) - 0.5) < 1e-10

    def test_exact(self):
        """Each step is exact rational."""
        x = logistic_map(VDR(1, 3), VDR(4))
        assert x == VDR(8, 9)


class TestLyapunov:
    def test_tent_lyapunov(self):
        """Tent map: |f'| = 2 everywhere. Product = 2^n."""
        derivs = [VDR(2)] * 20
        product = lyapunov_product(derivs)
        assert product == VDR(1048576)  # 2^20
