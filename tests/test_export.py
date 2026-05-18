"""
Tests for vdr.export — lossy projection boundary.
"""

import pytest
from fractions import Fraction

from vdr.core import VDR
from vdr.export import to_fraction, to_float, to_decimal


class TestToFraction:
    def test_closed(self):
        assert to_fraction(VDR(3, 7)) == Fraction(3, 7)

    def test_integer(self):
        assert to_fraction(VDR(5)) == Fraction(5)

    def test_zero(self):
        assert to_fraction(VDR(0)) == Fraction(0)


class TestToFloat:
    def test_half(self):
        assert to_float(VDR(1, 2)) == 0.5

    def test_third(self):
        assert abs(to_float(VDR(1, 3)) - 1 / 3) < 1e-15

    def test_zero(self):
        assert to_float(VDR(0)) == 0.0

    def test_negative(self):
        assert to_float(VDR(-1, 2)) == -0.5


class TestToDecimal:
    def test_simple(self):
        s = to_decimal(VDR(1, 2), digits=5)
        assert "0.5" in s

    def test_repeating(self):
        s = to_decimal(VDR(1, 3), digits=10)
        assert "3333" in s

    def test_integer(self):
        s = to_decimal(VDR(42), digits=5)
        assert "42" in s

    def test_negative(self):
        s = to_decimal(VDR(-1, 7), digits=10)
        assert s.startswith("-")

    def test_many_digits(self):
        s = to_decimal(VDR(1, 7), digits=50)
        assert len(s) >= 50

    def test_exact_terminating(self):
        s = to_decimal(VDR(1, 8), digits=10)
        assert "125" in s
