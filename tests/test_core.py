"""
Tests for vdr.core — VDR triple, Remainder, closed arithmetic,
normalization, rebase, lift, equality, comparison, coercion.
"""

import pytest
from fractions import Fraction

from vdr.core import (
    VDR, Remainder,
    VDRError, ZeroDenominatorError, InvalidStructureError,
    RebaseError, ArithmeticFailure,
    ZERO, ONE, NEG_ONE,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_integer(self):
        x = VDR(3)
        assert x.v == 3
        assert x.d == 1
        assert x.r.is_zero

    def test_rational(self):
        x = VDR(1, 2)
        assert x.v == 1
        assert x.d == 2

    def test_active(self):
        x = VDR(1, 3, Remainder(1))
        assert x.v == 1
        assert x.d == 3
        assert x.r.base == 1

    def test_remainder_int_shorthand(self):
        x = VDR(1, 3, 5)
        assert x.r.base == 5
        assert x.r.is_atomic

    def test_from_fraction(self):
        x = VDR.from_fraction(Fraction(5, 6))
        assert x.v == 5
        assert x.d == 6

    def test_from_int(self):
        x = VDR.from_int(42)
        assert x.v == 42
        assert x.d == 1

    def test_zero_denominator_raises(self):
        with pytest.raises(ZeroDenominatorError):
            VDR(1, 0)

    def test_v_must_be_int(self):
        with pytest.raises(InvalidStructureError):
            VDR(1.5, 1)

    def test_d_must_be_int(self):
        with pytest.raises(InvalidStructureError):
            VDR(1, 1.5)

    def test_invalid_remainder_type(self):
        with pytest.raises(InvalidStructureError):
            VDR(1, 1, "bad")


class TestRemainder:
    def test_zero(self):
        r = Remainder(0)
        assert r.is_zero
        assert r.is_atomic

    def test_atomic_nonzero(self):
        r = Remainder(5)
        assert not r.is_zero
        assert r.is_atomic

    def test_composite(self):
        r = Remainder(1, [VDR(1, 2)])
        assert not r.is_zero
        assert not r.is_atomic
        assert len(r.children) == 1

    def test_child_must_be_vdr(self):
        with pytest.raises(InvalidStructureError):
            Remainder(0, [42])

    def test_base_must_be_int(self):
        with pytest.raises(InvalidStructureError):
            Remainder(1.5)

    def test_negate(self):
        r = Remainder(3, [VDR(1, 2)])
        neg = r.negate()
        assert neg.base == -3
        assert neg.children[0].v == -1

    def test_combine_add(self):
        r1 = Remainder(2)
        r2 = Remainder(3)
        combined = r1.combine(r2, sign=1)
        assert combined.base == 5

    def test_combine_sub(self):
        r1 = Remainder(5)
        r2 = Remainder(3)
        combined = r1.combine(r2, sign=-1)
        assert combined.base == 2

    def test_lift(self):
        r = Remainder(3, [VDR(1, 7)])
        lifted = r.lift(5)
        assert lifted.base == 15
        assert lifted.children[0].v == 5
        assert lifted.children[0].d == 7

    def test_lift_zero_raises(self):
        r = Remainder(1)
        with pytest.raises(VDRError):
            r.lift(0)

    def test_legacy_value_atomic(self):
        r = Remainder(7)
        assert r.legacy_value() == Fraction(7)

    def test_legacy_value_composite(self):
        r = Remainder(1, [VDR(1, 2)])
        assert r.legacy_value() == Fraction(3, 2)

    def test_structural_eq(self):
        r1 = Remainder(3, [VDR(1, 2)])
        r2 = Remainder(3, [VDR(1, 2)])
        assert r1.structural_eq(r2)

    def test_structural_ne(self):
        r1 = Remainder(3)
        r2 = Remainder(4)
        assert not r1.structural_eq(r2)


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

class TestPredicates:
    def test_closed(self):
        assert VDR(1, 2).is_closed
        assert not VDR(1, 2, 3).is_closed

    def test_active(self):
        assert VDR(1, 2, 3).is_active
        assert not VDR(1, 2).is_active

    def test_globally_closed(self):
        assert VDR(1, 2).is_globally_closed
        assert not VDR(1, 2, Remainder(0, [VDR(1, 3)])).is_globally_closed


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------

class TestEquality:
    def test_structural_eq(self):
        a = VDR(1, 2)
        b = VDR(1, 2)
        assert a.structural_eq(b)

    def test_structural_ne(self):
        a = VDR(1, 2)
        b = VDR(2, 4)
        assert not a.structural_eq(b)

    def test_value_eq(self):
        a = VDR(1, 2)
        b = VDR(2, 4)
        assert a.value_eq(b)
        assert a == b

    def test_eq_int(self):
        assert VDR(5) == 5
        assert VDR(0) == 0

    def test_eq_fraction(self):
        assert VDR(1, 2) == Fraction(1, 2)

    def test_ne(self):
        assert VDR(1, 2) != VDR(1, 3)
        assert VDR(1) != 2

    def test_hash_consistent(self):
        a = VDR(1, 2)
        b = VDR(2, 4)
        assert hash(a) == hash(b)

    def test_hash_as_dict_key(self):
        d = {VDR(1, 2): "half"}
        assert d[VDR(2, 4)] == "half"


# ---------------------------------------------------------------------------
# Closed Arithmetic
# ---------------------------------------------------------------------------

class TestClosedArithmetic:
    def test_add(self):
        assert VDR(1, 2) + VDR(1, 3) == VDR(5, 6)

    def test_add_int(self):
        assert VDR(1, 2) + 1 == VDR(3, 2)

    def test_radd_int(self):
        assert 1 + VDR(1, 2) == VDR(3, 2)

    def test_sub(self):
        assert VDR(1, 2) - VDR(1, 3) == VDR(1, 6)

    def test_rsub_int(self):
        assert 1 - VDR(1, 3) == VDR(2, 3)

    def test_mul(self):
        assert VDR(2, 3) * VDR(3, 4) == VDR(1, 2)

    def test_mul_int(self):
        assert VDR(1, 3) * 3 == VDR(1)

    def test_rmul_int(self):
        assert 5 * VDR(1, 5) == VDR(1)

    def test_div(self):
        assert VDR(1, 2) / VDR(1, 3) == VDR(3, 2)

    def test_div_int(self):
        assert VDR(1) / 3 == VDR(1, 3)

    def test_rdiv_int(self):
        assert 1 / VDR(3) == VDR(1, 3)

    def test_div_by_zero_raises(self):
        with pytest.raises(ArithmeticFailure):
            VDR(1) / VDR(0)

    def test_neg(self):
        assert -VDR(1, 2) == VDR(-1, 2)

    def test_abs_positive(self):
        assert abs(VDR(1, 2)) == VDR(1, 2)

    def test_abs_negative(self):
        assert abs(VDR(-1, 2)) == VDR(1, 2)

    def test_additive_identity(self):
        x = VDR(7, 13)
        assert x + ZERO == x

    def test_multiplicative_identity(self):
        x = VDR(7, 13)
        assert x * ONE == x

    def test_additive_inverse(self):
        x = VDR(7, 13)
        assert x + (-x) == ZERO

    def test_return_to_origin(self):
        """200-op return to origin: VDR error = 0."""
        x = VDR(1, 7)
        step = VDR(1, 13)
        for _ in range(100):
            x = x + step
        for _ in range(100):
            x = x - step
        assert x == VDR(1, 7)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_gcd_reduction(self):
        x = VDR(2, 4).normalize()
        assert x.v == 1
        assert x.d == 2

    def test_sign_convention(self):
        x = VDR(1, -2).normalize()
        assert x.d > 0
        assert x.v == -1
        assert x.d == 2

    def test_zero_normalize(self):
        x = VDR(0, 5).normalize()
        assert x.v == 0
        assert x.d == 1

    def test_idempotent(self):
        x = VDR(6, 10).normalize()
        y = x.normalize()
        assert x.structural_eq(y)

    def test_closed_form_preference(self):
        """N7: if remainder settles to zero, close it."""
        r = Remainder(0, [VDR(0, 1)])
        x = VDR(3, 7, r).normalize()
        assert x.is_closed
        assert x == VDR(3, 7)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

class TestComparison:
    def test_lt(self):
        assert VDR(1, 3) < VDR(1, 2)

    def test_le(self):
        assert VDR(1, 2) <= VDR(1, 2)
        assert VDR(1, 3) <= VDR(1, 2)

    def test_gt(self):
        assert VDR(1, 2) > VDR(1, 3)

    def test_ge(self):
        assert VDR(1, 2) >= VDR(1, 2)

    def test_lt_int(self):
        assert VDR(1, 2) < 1

    def test_gt_int(self):
        assert VDR(3, 2) > 1


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

class TestProjection:
    def test_to_fraction_closed(self):
        assert VDR(3, 7).to_fraction() == Fraction(3, 7)

    def test_to_fraction_active(self):
        x = VDR(1, 3, Remainder(1))
        assert x.to_fraction() == Fraction(2, 3)

    def test_to_float(self):
        assert abs(VDR(1, 3).to_float() - 1/3) < 1e-15

    def test_float_builtin(self):
        assert abs(float(VDR(1, 2)) - 0.5) < 1e-15


# ---------------------------------------------------------------------------
# Rebase
# ---------------------------------------------------------------------------

class TestRebase:
    def test_same_d(self):
        x = VDR(3, 7)
        y = x.rebase(7)
        assert y == x

    def test_closed_rebase(self):
        x = VDR(1, 2)
        y = x.rebase(6)
        assert y == VDR(3, 6)

    def test_active_rebase(self):
        x = VDR(1, 3)
        y = x.rebase(7)
        # value preserved
        assert y.to_fraction() == Fraction(1, 3)

    def test_rebase_preserves_value(self):
        x = VDR(5, 13)
        for target in [7, 11, 100, 1]:
            y = x.rebase(target)
            assert y.to_fraction() == x.to_fraction()

    def test_rebase_zero_raises(self):
        with pytest.raises(RebaseError):
            VDR(1, 2).rebase(0)


# ---------------------------------------------------------------------------
# Lift
# ---------------------------------------------------------------------------

class TestLift:
    def test_lift_identity(self):
        x = VDR(3, 7)
        y = x._lift_vdr(1)
        assert y.v == 3
        assert y.d == 7

    def test_lift_scale(self):
        x = VDR(3, 7)
        y = x._lift_vdr(5)
        assert y.v == 15
        assert y.d == 7

    def test_lift_compose(self):
        """lift(lift(R, a), b) = lift(R, a*b)."""
        r = Remainder(3, [VDR(1, 5)])
        r_ab = r.lift(2).lift(3)
        r_6 = r.lift(6)
        assert r_ab.base == r_6.base
        assert r_ab.children[0].v == r_6.children[0].v


# ---------------------------------------------------------------------------
# Active Addition
# ---------------------------------------------------------------------------

class TestActiveAddition:
    def test_same_d(self):
        a = VDR(1, 3, Remainder(1))
        b = VDR(2, 3, Remainder(2))
        c = a + b
        assert c.to_fraction() == Fraction(1 + 1, 3) + Fraction(2 + 2, 3)

    def test_different_d(self):
        a = VDR(1, 2, Remainder(1))
        b = VDR(1, 3, Remainder(1))
        c = a + b
        expected = a.to_fraction() + b.to_fraction()
        assert c.to_fraction() == expected

    def test_active_sub(self):
        a = VDR(1, 2, Remainder(1))
        b = VDR(1, 3)
        c = a - b
        expected = a.to_fraction() - b.to_fraction()
        assert c.to_fraction() == expected


# ---------------------------------------------------------------------------
# Structural Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_depth_closed(self):
        assert VDR(1, 2).depth() == 0

    def test_depth_one_level(self):
        x = VDR(1, 2, Remainder(0, [VDR(1, 3)]))
        assert x.depth() == 1

    def test_size_closed(self):
        assert VDR(1, 2).size() == 2  # 1 for VDR + 1 for atomic remainder

    def test_den_complexity(self):
        x = VDR(1, 6)
        u, s, c = x.den_complexity()
        assert u == 1
        assert s == 6
        assert c == 1


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

class TestDisplay:
    def test_repr(self):
        x = VDR(1, 2)
        assert "VDR" in repr(x)
        assert "1" in repr(x)
        assert "2" in repr(x)

    def test_str(self):
        x = VDR(1, 2)
        assert str(x) == "[1, 2, 0]"

    def test_bracket(self):
        x = VDR(1, 2)
        assert x.bracket() == "[1, 2, 0]"


# ---------------------------------------------------------------------------
# Module Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_zero(self):
        assert ZERO == VDR(0)
        assert ZERO.is_closed

    def test_one(self):
        assert ONE == VDR(1)

    def test_neg_one(self):
        assert NEG_ONE == VDR(-1)
        assert NEG_ONE == -ONE
