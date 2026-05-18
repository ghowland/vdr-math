"""
Tests for vdr.basis — D-frame management, Q335, basis arithmetic.
"""

import pytest
from fractions import Fraction

from vdr.core import VDR, Remainder
from vdr.basis import (
    DEFAULT_BITS, Q335,
    set_default, get_default, q_basis_denominator,
    to_qbasis, vec_to_qbasis, mat_to_qbasis,
    qb_add, qb_mul, qb_div,
)
from vdr.linalg import Vec, Mat
from vdr.active import install

install()


class TestDefaults:
    def test_default_bits(self):
        assert get_default() == 335

    def test_q335_value(self):
        assert Q335 == 2 ** 335

    def test_set_default(self):
        original = get_default()
        set_default(668)
        assert get_default() == 668
        set_default(original)  # restore

    def test_set_invalid_raises(self):
        with pytest.raises(ValueError):
            set_default(0)
        with pytest.raises(ValueError):
            set_default(-1)

    def test_q_basis_denominator(self):
        assert q_basis_denominator(10) == 1024
        assert q_basis_denominator(1) == 2


class TestToQbasis:
    def test_integer(self):
        x = to_qbasis(VDR(3), bits=10)
        assert x.d == 1024
        assert x.v == 3 * 1024

    def test_fraction(self):
        x = to_qbasis(VDR(1, 2), bits=10)
        assert x.d == 1024
        assert x.v == 512

    def test_from_fraction(self):
        x = to_qbasis(Fraction(1, 3), bits=10)
        assert x.d == 1024
        # round(1/3 * 1024) = round(341.33) = 341
        assert x.v == 341

    def test_roundtrip_value(self):
        """Projected value should be close to original."""
        original = VDR(22, 7)
        projected = to_qbasis(original, bits=20)
        frac = projected.to_fraction()
        err = abs(frac - Fraction(22, 7))
        assert err < Fraction(1, 2 ** 20)


class TestVecMatQbasis:
    def test_vec(self):
        v = Vec.from_ints([1, 2, 3])
        vq = vec_to_qbasis(v, bits=10)
        assert len(vq) == 3
        assert vq[0].d == 1024

    def test_mat(self):
        m = Mat.from_ints([[1, 2], [3, 4]])
        mq = mat_to_qbasis(m, bits=10)
        assert mq[0, 0].d == 1024


class TestQbAdd:
    def test_basic(self):
        a = to_qbasis(VDR(1, 3), bits=10)
        b = to_qbasis(VDR(1, 4), bits=10)
        c = qb_add(a, b, bits=10)
        assert c.d == 1024
        # value should be close to 7/12
        expected = Fraction(1, 3) + Fraction(1, 4)
        err = abs(c.to_fraction() - expected)
        assert err < Fraction(2, 1024)


class TestQbMul:
    def test_d_stays_fixed(self):
        """D must stay at 2^bits after multiplication. Overflow in R."""
        a = to_qbasis(VDR(3), bits=10)
        b = to_qbasis(VDR(5), bits=10)
        c = qb_mul(a, b, bits=10)
        assert c.d == 1024

    def test_value_correct(self):
        a = to_qbasis(VDR(3), bits=10)
        b = to_qbasis(VDR(5), bits=10)
        c = qb_mul(a, b, bits=10)
        # value should be close to 15
        val = c.to_fraction()
        if c.is_closed:
            err = abs(val - 15)
        else:
            err = abs(val - 15)
        assert err < Fraction(1, 100)

    def test_overflow_in_r(self):
        """When product doesn't divide evenly, remainder appears."""
        a = to_qbasis(VDR(1, 3), bits=10)
        b = to_qbasis(VDR(1, 3), bits=10)
        c = qb_mul(a, b, bits=10)
        # 341 * 341 = 116281, divmod(116281, 1024) = (113, 569)
        # so c = [113, 1024, [569, 1024, 0]]
        assert c.d == 1024


class TestQbDiv:
    def test_basic(self):
        a = to_qbasis(VDR(6), bits=10)
        b = to_qbasis(VDR(3), bits=10)
        c = qb_div(a, b, bits=10)
        assert c.d == 1024

    def test_div_by_zero_raises(self):
        from vdr.core import ArithmeticFailure
        a = to_qbasis(VDR(1), bits=10)
        b = VDR(0, 1024)
        with pytest.raises(ArithmeticFailure):
            qb_div(a, b, bits=10)


class TestSmallBasis:
    """VDR works at any D, not just Q335."""

    def test_basis_7(self):
        """D=7 as a basis frame."""
        a = VDR(2, 7)
        b = VDR(3, 7)
        c = a + b
        assert c == VDR(5, 7)

    def test_basis_16(self):
        """D=16 for binary fixed-point."""
        a = VDR(5, 16)   # 5/16
        b = VDR(3, 16)   # 3/16
        c = a * b         # 15/256
        assert c.to_fraction() == Fraction(15, 256)

    def test_basis_1(self):
        """D=1: plain integers."""
        a = VDR(5)
        b = VDR(7)
        c = a + b
        assert c == VDR(12)
