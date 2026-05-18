"""
Tests for vdr.active — active multiplication and division.
"""

import pytest
from fractions import Fraction

from vdr.core import VDR, Remainder, ArithmeticFailure
from vdr.active import active_mul, active_div, install, uninstall


# Ensure active arithmetic is installed
@pytest.fixture(autouse=True)
def ensure_installed():
    install()
    yield
    # leave installed — other tests need it


# ---------------------------------------------------------------------------
# Closed multiplication (should still work via active path)
# ---------------------------------------------------------------------------

class TestClosedMul:
    def test_basic(self):
        assert active_mul(VDR(2, 3), VDR(3, 4)) == VDR(1, 2)

    def test_identity(self):
        x = VDR(7, 13)
        assert active_mul(x, VDR(1)) == x

    def test_zero(self):
        assert active_mul(VDR(7, 13), VDR(0)) == VDR(0)

    def test_negation(self):
        x = VDR(3, 5)
        assert active_mul(x, VDR(-1)) == -x


# ---------------------------------------------------------------------------
# Active multiplication
# ---------------------------------------------------------------------------

class TestActiveMul:
    def test_active_times_closed(self):
        a = VDR(1, 2, Remainder(1))  # value = (1+1)/2 = 1
        b = VDR(3)
        c = active_mul(a, b)
        assert c.to_fraction() == a.to_fraction() * b.to_fraction()

    def test_closed_times_active(self):
        a = VDR(5)
        b = VDR(1, 3, Remainder(2))  # value = 3/3 = 1
        c = active_mul(a, b)
        assert c.to_fraction() == a.to_fraction() * b.to_fraction()

    def test_active_times_active(self):
        a = VDR(1, 2, Remainder(1))
        b = VDR(1, 3, Remainder(1))
        c = active_mul(a, b)
        expected = a.to_fraction() * b.to_fraction()
        assert c.to_fraction() == expected

    def test_value_preservation(self):
        """Product value must equal product of projected values."""
        a = VDR(3, 7, Remainder(2))
        b = VDR(5, 11, Remainder(3))
        c = active_mul(a, b)
        expected = a.to_fraction() * b.to_fraction()
        assert c.to_fraction() == expected

    def test_operator_mul(self):
        """Test that * operator dispatches to active_mul."""
        a = VDR(1, 2, Remainder(1))
        b = VDR(3)
        c = a * b
        assert c.to_fraction() == a.to_fraction() * Fraction(3)

    def test_operator_rmul(self):
        a = VDR(1, 2, Remainder(1))
        c = 3 * a
        assert c.to_fraction() == 3 * a.to_fraction()

    def test_mul_chain(self):
        """Chain of multiplications preserves value."""
        x = VDR(1, 7)
        y = VDR(1, 13)
        z = x * y * VDR(7) * VDR(13)
        assert z == VDR(1)


# ---------------------------------------------------------------------------
# Active division
# ---------------------------------------------------------------------------

class TestActiveDiv:
    def test_closed_by_closed(self):
        assert active_div(VDR(1, 2), VDR(1, 3)) == VDR(3, 2)

    def test_active_by_closed(self):
        a = VDR(1, 2, Remainder(1))
        b = VDR(3)
        c = active_div(a, b)
        assert c.to_fraction() == a.to_fraction() / Fraction(3)

    def test_active_by_active(self):
        """v1 compromise: divisor projected, structure lost."""
        a = VDR(3, 7, Remainder(1))
        b = VDR(2, 5, Remainder(1))
        c = active_div(a, b)
        expected = a.to_fraction() / b.to_fraction()
        assert c.to_fraction() == expected

    def test_div_by_zero_closed(self):
        with pytest.raises(ArithmeticFailure):
            active_div(VDR(1), VDR(0))

    def test_div_by_zero_active(self):
        """Active object projecting to zero."""
        # VDR(0, 1, Remainder(0)) projects to 0
        with pytest.raises(ArithmeticFailure):
            active_div(VDR(1), VDR(0, 1, Remainder(0)))

    def test_operator_div(self):
        a = VDR(1, 2, Remainder(1))
        b = VDR(3)
        c = a / b
        assert c.to_fraction() == a.to_fraction() / Fraction(3)

    def test_operator_rdiv(self):
        c = 1 / VDR(3)
        assert c == VDR(1, 3)

    def test_div_inverse(self):
        """x * (1/x) == 1 for closed."""
        x = VDR(7, 13)
        assert x * (VDR(1) / x) == VDR(1)


# ---------------------------------------------------------------------------
# Cross-term structure
# ---------------------------------------------------------------------------

class TestCrossTerms:
    def test_active_mul_produces_remainder(self):
        """When at least one operand is active, result may have remainder."""
        a = VDR(1, 2, Remainder(3))
        b = VDR(1, 3, Remainder(5))
        c = active_mul(a, b)
        # c must have correct value regardless of structure
        expected = a.to_fraction() * b.to_fraction()
        assert c.to_fraction() == expected

    def test_cross_terms_add_up(self):
        """V1*R2 + V2*R1 + R1*R2 must produce correct total."""
        for v1, d1, r1, v2, d2, r2 in [
            (1, 2, 1, 1, 3, 1),
            (3, 5, 2, 7, 11, 3),
            (0, 1, 5, 1, 1, 7),
            (10, 1, 0, 1, 7, 3),
        ]:
            a = VDR(v1, d1, Remainder(r1))
            b = VDR(v2, d2, Remainder(r2))
            c = active_mul(a, b)
            expected = a.to_fraction() * b.to_fraction()
            assert c.to_fraction() == expected, (
                "Failed for %s * %s" % (a, b)
            )


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------

class TestInstall:
    def test_install_idempotent(self):
        install()
        install()  # should not raise

    def test_uninstall_restores(self):
        uninstall()
        with pytest.raises(ArithmeticFailure):
            VDR(1, 2, Remainder(1)) * VDR(3)
        install()  # restore for other tests

    def test_reinstall_works(self):
        uninstall()
        install()
        a = VDR(1, 2, Remainder(1))
        b = VDR(3)
        c = a * b  # should work
        assert c.to_fraction() == a.to_fraction() * Fraction(3)
