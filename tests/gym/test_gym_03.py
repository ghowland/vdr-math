"""Gym 03 — Continued fractions. 26/31 in VDR-2."""

import pytest
from vdr.core import VDR
from vdr.math.continued_fractions import (
    to_cf, from_cf, convergents_from_cf, sb_path, sqrt_cf_period,
)


class TestCFConversion:
    def test_roundtrip(self):
        """to_cf -> from_cf should recover original."""
        for v, d in [(1, 2), (3, 7), (22, 7), (355, 113)]:
            coeffs = to_cf(v, d)
            recovered = from_cf(coeffs)
            assert recovered == VDR(v, d)

    def test_pi_approx(self):
        coeffs = to_cf(355, 113)
        assert coeffs == [3, 7, 16]


class TestConvergents:
    def test_basic(self):
        cvs = convergents_from_cf([3, 7, 16])
        assert cvs[0] == VDR(3)
        assert cvs[1] == VDR(22, 7)
        assert cvs[2] == VDR(355, 113)


class TestSternBrocot:
    def test_known_paths(self):
        assert sb_path(1, 2) == "L"
        assert sb_path(2, 1) == "R"
        assert sb_path(1, 3) == "LL"
        assert sb_path(3, 1) == "RR"


class TestSqrtCF:
    def test_sqrt2(self):
        a0, period = sqrt_cf_period(2)
        assert a0 == 1
        assert period == [2]

    def test_sqrt3(self):
        a0, period = sqrt_cf_period(3)
        assert a0 == 1
        assert period == [1, 2]

    def test_sqrt7(self):
        a0, period = sqrt_cf_period(7)
        assert a0 == 2
        assert period == [1, 1, 1, 4]

    def test_perfect_square_raises(self):
        with pytest.raises(ValueError):
            sqrt_cf_period(4)
