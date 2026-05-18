"""
Tests for normalization fix — large Q numbers reducing to simple form,
Newton perfect-square problem, value-zero remainders collapsing.
"""

import pytest
from fractions import Fraction

from vdr.core import VDR, Remainder
from vdr.active import install

install()


class TestGCDReduction:
    def test_basic_reduction(self):
        x = VDR(4, 6).normalize()
        assert x.v == 2
        assert x.d == 3

    def test_large_reduction(self):
        x = VDR(1000000, 2000000).normalize()
        assert x.v == 1
        assert x.d == 2

    def test_already_reduced(self):
        x = VDR(3, 7).normalize()
        assert x.v == 3
        assert x.d == 7

    def test_negative_sign(self):
        x = VDR(-6, 10).normalize()
        assert x.v == -3
        assert x.d == 5

    def test_negative_d(self):
        x = VDR(3, -5).normalize()
        assert x.d == 5
        assert x.v == -3


class TestSignConvention:
    def test_positive_d(self):
        x = VDR(1, -3).normalize()
        assert x.d > 0

    def test_double_negative(self):
        x = VDR(-1, -3).normalize()
        assert x.d > 0
        assert x.v > 0


class TestClosedFormPreference:
    """N7: if remainder normalizes to zero, collapse to closed."""

    def test_zero_remainder_child(self):
        r = Remainder(0, [VDR(0)])
        x = VDR(3, 7, r).normalize()
        assert x.is_closed
        assert x == VDR(3, 7)

    def test_zero_base_no_children(self):
        r = Remainder(0)
        x = VDR(3, 7, r).normalize()
        assert x.is_closed

    def test_nonzero_does_not_collapse(self):
        r = Remainder(1)
        x = VDR(3, 7, r).normalize()
        assert x.is_active


class TestNewtonPerfectSquare:
    """
    Newton sqrt on perfect squares should normalize to simple form.
    sqrt(4) should give [2, 1, 0], not [2k, k, 0] for large k.
    """

    def test_manual_unreduced(self):
        """Simulate the Newton problem: 2k/k should reduce to 2."""
        k = 2 ** 50
        x = VDR(2 * k, k).normalize()
        assert x.v == 2
        assert x.d == 1
        assert x.is_closed

    def test_large_unreduced_half(self):
        k = 10 ** 30
        x = VDR(k, 2 * k).normalize()
        assert x.v == 1
        assert x.d == 2

    def test_large_unreduced_third(self):
        k = 7 ** 20
        x = VDR(k, 3 * k).normalize()
        assert x.v == 1
        assert x.d == 3

    def test_trivially_one(self):
        k = 2 ** 100
        x = VDR(k, k).normalize()
        assert x.v == 1
        assert x.d == 1

    def test_trivially_zero(self):
        k = 3 ** 50
        x = VDR(0, k).normalize()
        assert x.v == 0
        assert x.d == 1


class TestValueZeroRemainder:
    """
    Remainders that are structurally nonzero but value-equivalent to zero
    should collapse during normalization.
    """

    def test_child_cancellation(self):
        """Two children that cancel: [1,2,0] + [-1,2,0] = 0."""
        r = Remainder(0, [VDR(1, 2), VDR(-1, 2)])
        x = VDR(5, 7, r).normalize()
        assert x.is_closed
        assert x == VDR(5, 7)

    def test_base_cancels_child(self):
        """Base 1 + child [-1, 1, 0] = 0."""
        r = Remainder(1, [VDR(-1)])
        x = VDR(3, 5, r).normalize()
        # after normalization: base absorbs child (D=1), 1 + (-1) = 0
        assert x.is_closed
        assert x == VDR(3, 5)

    def test_atomic_zero(self):
        """Atomic remainder of 0 is already zero."""
        r = Remainder(0)
        x = VDR(3, 5, r).normalize()
        assert x.is_closed

    def test_nonzero_remains_active(self):
        """Non-cancelling remainder stays active."""
        r = Remainder(0, [VDR(1, 2), VDR(1, 3)])
        x = VDR(1, 1, r).normalize()
        assert x.is_active


class TestSameDenomMerge:
    """Children sharing a denominator should merge."""

    def test_merge_same_d(self):
        r = Remainder(0, [VDR(1, 5), VDR(2, 5)])
        x = VDR(0, 1, r).normalize()
        # 1/5 + 2/5 = 3/5, should merge
        assert x.to_fraction() == Fraction(3, 5)

    def test_merge_cancels(self):
        r = Remainder(0, [VDR(1, 5), VDR(-1, 5)])
        x = VDR(3, 7, r).normalize()
        assert x.is_closed
        assert x == VDR(3, 7)


class TestCanonicalOrdering:
    """Children should be sorted by |D|, then V."""

    def test_ordering(self):
        r = Remainder(0, [VDR(1, 7), VDR(1, 3), VDR(1, 5)])
        x = VDR(0, 1, r).normalize()
        if x.r.children:
            denoms = [abs(c.d) for c in x.r.children]
            assert denoms == sorted(denoms)


class TestNormalizationIdempotent:
    def test_closed_idempotent(self):
        x = VDR(6, 10).normalize()
        y = x.normalize()
        assert x.structural_eq(y)

    def test_active_idempotent(self):
        r = Remainder(3, [VDR(1, 5)])
        x = VDR(1, 2, r).normalize()
        y = x.normalize()
        assert x.structural_eq(y)
