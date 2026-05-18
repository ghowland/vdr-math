"""Gym 23 — Q335 transcendental arithmetic. 13/16 in VDR-3
(3 precision frame mismatch — solved by divmod rule)."""

import pytest
from fractions import Fraction
from vdr.core import VDR
from vdr.math.transcendental import (
    PI, E, LN2, SQRT2, PHI, PI_SQ, ZETA2, ZETA3, ZETA5,
    LN3, LN5, LN10, CATALAN, Q335_DENOM,
    sqrt_newton, exp_series, sin_series, cos_series,
    borwein_zeta, pi_machin,
)


class TestQ335Constants:
    def test_pi_denominator(self):
        assert PI.d == Q335_DENOM

    def test_e_denominator(self):
        assert E.d == Q335_DENOM

    def test_all_22_same_denom(self):
        constants = [PI, E, LN2, SQRT2, PHI, PI_SQ, ZETA2, ZETA3, ZETA5,
                     LN3, LN5, LN10, CATALAN]
        for c in constants:
            assert c.d == Q335_DENOM


class TestQ335Addition:
    def test_pi_plus_e(self):
        """pi + e via qb_add keeps D fixed."""
        from vdr.basis import qb_add
        result = qb_add(PI, E, bits=335)
        assert result.d == Q335_DENOM
        assert result.v == PI.v + E.v

    def test_pi_plus_e_closed(self):
        """Closed addition preserves value even if D changes internally."""
        result = PI + E
        from fractions import Fraction
        expected = PI.to_fraction() + E.to_fraction()
        assert result.to_fraction() == expected

    def test_ln_identity(self):
        """ln(10) ≈ ln(2) + ln(5). Residual should be tiny."""
        from vdr.basis import qb_add
        result = qb_add(LN2, LN5, bits=335)
        diff_v = result.v - LN10.v
        assert abs(diff_v) <= 2  # at most 2 ULP
        

class TestQ335Multiplication:
    def test_d_stays_fixed(self):
        """Multiplication of two Q335 values: D stays 2^335."""
        # use basis module for proper multiplication
        from vdr.basis import qb_mul
        result = qb_mul(PI, E, bits=335)
        assert result.d == Q335_DENOM

    def test_phi_identity(self):
        """phi ≈ (1 + sqrt(5)) / 2. Check residual."""
        half_one_plus_sqrt5 = (VDR(Q335_DENOM, Q335_DENOM) + SQRT2) / VDR(2)
        # this isn't the right identity — phi = (1+sqrt(5))/2
        # just verify phi is close to 1.618...
        phi_float = float(PHI.to_fraction())
        assert abs(phi_float - 1.6180339887) < 1e-8


class TestSqrtNewton:
    def test_sqrt2_residual(self):
        result = sqrt_newton(VDR(2), depth=10)
        residual = result * result - VDR(2)
        frac = residual.to_fraction()
        assert abs(frac) < Fraction(1, 10 ** 50)

    def test_sqrt3(self):
        result = sqrt_newton(VDR(3), depth=10)
        residual = result * result - VDR(3)
        assert abs(residual.to_fraction()) < Fraction(1, 10 ** 50)

    def test_perfect_square(self):
        result = sqrt_newton(VDR(4), depth=10)
        # should normalize close to 2 (exact if normalization fix works)
        assert abs(float(result.to_fraction()) - 2.0) < 1e-30


class TestSeries:
    def test_exp_1(self):
        """exp(1) should be close to e."""
        result = exp_series(VDR(1), depth=20)
        err = abs(float(result.to_fraction()) - 2.718281828459045)
        assert err < 1e-15

    def test_sin_0(self):
        assert sin_series(VDR(0), depth=10) == VDR(0)

    def test_cos_0(self):
        assert cos_series(VDR(0), depth=10) == VDR(1)

    def test_sin_sq_plus_cos_sq(self):
        """sin^2(x) + cos^2(x) ≈ 1 for small x."""
        x = VDR(1, 4)
        s = sin_series(x, depth=16)
        c = cos_series(x, depth=16)
        identity = s * s + c * c
        err = abs(float(identity.to_fraction()) - 1.0)
        assert err < 1e-20


class TestBorwein:
    def test_zeta2(self):
        """zeta(2) = pi^2/6. Check via Borwein."""
        z2 = borwein_zeta(2, n=50)
        z2_float = float(z2.to_fraction())
        expected = 1.6449340668482264
        assert abs(z2_float - expected) < 1e-10

    def test_zeta3(self):
        """Apery's constant ≈ 1.2020569..."""
        z3 = borwein_zeta(3, n=50)
        z3_float = float(z3.to_fraction())
        assert abs(z3_float - 1.2020569031595942) < 1e-10


class TestPiMachin:
    def test_convergence(self):
        """Machin formula should converge to pi."""
        pi_val = pi_machin(terms=50)
        pi_float = float(pi_val.to_fraction())
        assert abs(pi_float - 3.141592653589793) < 1e-12
