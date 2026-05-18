"""Gym 01 — Number theory. 37/37 in VDR-2."""

import pytest
from fractions import Fraction
from vdr.core import VDR
from vdr.math.number_theory import (
    vdr_gcd, vdr_lcm, egyptian_fractions, stern_brocot, farey,
    euler_totient, harmonic, vdr_pow, vdr_mod, convergents, e_cf,
)


class TestGCD:
    def test_basic(self):
        assert vdr_gcd(VDR(12), VDR(8)) == VDR(4)

    def test_coprime(self):
        assert vdr_gcd(VDR(7), VDR(13)) == VDR(1)

    def test_zero(self):
        assert vdr_gcd(VDR(0), VDR(5)) == VDR(5)


class TestLCM:
    def test_basic(self):
        assert vdr_lcm(VDR(4), VDR(6)) == VDR(12)

    def test_coprime(self):
        assert vdr_lcm(VDR(7), VDR(13)) == VDR(91)


class TestEgyptianFractions:
    def test_3_7(self):
        ef = egyptian_fractions(3, 7)
        total = sum((f.to_fraction() for f in ef), Fraction(0))
        assert total == Fraction(3, 7)

    def test_unit(self):
        ef = egyptian_fractions(1, 5)
        assert len(ef) == 1
        assert ef[0] == VDR(1, 5)


class TestFarey:
    def test_farey_5(self):
        f5 = farey(5)
        assert f5[0] == VDR(0, 1)
        assert f5[-1] == VDR(1, 1)
        # F5 has 11 elements
        assert len(f5) == 11

    def test_mediant_property(self):
        """For adjacent Farey fractions a/b, c/d: |ad - bc| = 1."""
        f5 = farey(5)
        for i in range(len(f5) - 1):
            a, b = f5[i].v, f5[i].d
            c, d = f5[i + 1].v, f5[i + 1].d
            assert abs(a * d - b * c) == 1


class TestTotient:
    def test_prime(self):
        assert euler_totient(7) == 6

    def test_100(self):
        assert euler_totient(100) == 40

    def test_1(self):
        assert euler_totient(1) == 1


class TestHarmonic:
    def test_h1(self):
        assert harmonic(1) == VDR(1)

    def test_h10(self):
        assert harmonic(10) == VDR(7381, 2520)

    def test_h5(self):
        expected = Fraction(1) + Fraction(1, 2) + Fraction(1, 3) + Fraction(1, 4) + Fraction(1, 5)
        assert harmonic(5).to_fraction() == expected


class TestPow:
    def test_square(self):
        assert vdr_pow(VDR(3), 2) == VDR(9)

    def test_zero_exp(self):
        assert vdr_pow(VDR(7), 0) == VDR(1)

    def test_rational_base(self):
        assert vdr_pow(VDR(1, 2), 3) == VDR(1, 8)

    def test_negative_exp(self):
        assert vdr_pow(VDR(2), -1) == VDR(1, 2)


class TestMod:
    def test_basic(self):
        assert vdr_mod(VDR(17), 5) == VDR(2)


class TestConvergents:
    def test_pi_approx(self):
        """[3; 7, 15, 1] -> convergents include 22/7, 333/106, 355/113."""
        cvs = convergents([3, 7, 15, 1])
        assert cvs[0] == VDR(3, 1)
        assert cvs[1] == VDR(22, 7)
        assert cvs[3] == VDR(355, 113)

    def test_e_convergents(self):
        coeffs = e_cf(6)
        cvs = convergents(coeffs)
        # 5th convergent is 87/32
        assert cvs[4] == VDR(87, 32)


class TestECF:
    def test_length(self):
        assert len(e_cf(10)) == 10

    def test_starts_with_2(self):
        assert e_cf(1) == [2]

    def test_pattern(self):
        coeffs = e_cf(7)
        assert coeffs[0] == 2
        assert coeffs[1] == 1
        assert coeffs[2] == 2  # 2*1
        assert coeffs[3] == 1
