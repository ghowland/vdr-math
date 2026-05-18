"""
vdr.active — Active multiplication and division for VDR objects.

Extends VDR arithmetic beyond the closed subclass.

For [V1,D1,R1] * [V2,D2,R2]:
    Frame: D1*D2
    Closed part: V1*V2
    Remainder: V1*R2 + V2*R1 + R1*R2 (cross-terms)

All cross-terms captured as exact remainder structure. No approximation.

    import vdr.active
    vdr.active.install()   # patches VDR.__mul__ and __truediv__
"""

from __future__ import annotations
from fractions import Fraction
from typing import Union

from vdr.core import VDR, Remainder, ArithmeticFailure

__all__ = ["active_mul", "active_div", "install", "uninstall"]


def active_mul(a, b):
    """
    Exact multiplication of two VDR objects, including active.

    Both closed: direct formula V1*V2 / D1*D2.
    At least one active: construct product with cross-term remainder.
    """
    if a.is_closed and b.is_closed:
        return VDR(a.v * b.v, a.d * b.d).normalize()

    new_d = a.d * b.d
    new_v = a.v * b.v

    r_parts = _mul_cross_terms(a, b)

    return VDR(new_v, new_d, r_parts).normalize()


def active_div(a, b):
    """
    Exact division of two VDR objects.

    By closed: multiply by reciprocal.
    By active: project divisor to exact rational, invert, multiply.
               Divisor remainder structure lost (v1 compromise).
    """
    if b.is_closed:
        if b.v == 0:
            raise ArithmeticFailure("Division by zero")
        if a.is_closed:
            return VDR(a.v * b.d, a.d * b.v).normalize()
        return active_mul(a, VDR(b.d, b.v))

    b_frac = b.to_fraction()
    if b_frac == 0:
        raise ArithmeticFailure(
            "Division by zero (active object projects to 0)"
        )

    b_inv = VDR(b_frac.denominator, b_frac.numerator)
    return active_mul(a, b_inv)


def _mul_cross_terms(a, b):
    """
    Construct remainder from multiplying two VDR objects.

    Cross-term contributions:
      1. V1 * R2  (first value times second remainder)
      2. V2 * R1  (second value times first remainder)
      3. R1 * R2  (remainder times remainder)
    """
    left = _scale_remainder(b.r, a.v)
    right = _scale_remainder(a.r, b.v)
    cross = _mul_remainders(a.r, b.r)

    return _combine_three(left, right, cross)


def _scale_remainder(r, k):
    """
    Multiply a remainder by an integer scalar.
    k * (base + X1 + ... + Xn) = k*base + k*X1 + ... + k*Xn
    """
    if k == 0:
        return Remainder(0)
    return Remainder(
        r.base * k,
        [VDR(k * c.v, c.d, _scale_remainder(c.r, k)) for c in r.children],
    )


def _mul_remainders(r1, r2):
    """
    Multiply two remainders: R1 * R2.

    Computes exact rational value via legacy projection and expresses
    the result as remainder structure.
    """
    val1 = r1.legacy_value()
    val2 = r2.legacy_value()

    product = val1 * val2

    if product == 0:
        return Remainder(0)

    if product.denominator == 1:
        return Remainder(int(product))

    return Remainder(0, [
        VDR(int(product.numerator), int(product.denominator))
    ])


def _combine_three(r1, r2, r3):
    """Combine three remainder contributions into one."""
    base = r1.base + r2.base + r3.base
    children = list(r1.children) + list(r2.children) + list(r3.children)
    return Remainder(base, children)


# ---------------------------------------------------------------------------
# Installation: patch VDR operators
# ---------------------------------------------------------------------------

_original_mul = None
_original_rmul = None
_original_div = None
_original_rdiv = None
_installed = False


def _patched_mul(self, other):
    from vdr._compat import _coerce
    other = _coerce(other)
    return active_mul(self, other)


def _patched_rmul(self, other):
    from vdr._compat import _coerce
    return active_mul(self, _coerce(other))


def _patched_div(self, other):
    from vdr._compat import _coerce
    other = _coerce(other)
    return active_div(self, other)


def _patched_rdiv(self, other):
    from vdr._compat import _coerce
    return active_div(_coerce(other), self)


def install():
    """
    Patch VDR with active multiplication and division.

    After this call, VDR * and / operators handle active objects.
    Called automatically by vdr.__init__.
    """
    global _original_mul, _original_rmul, _original_div, _original_rdiv
    global _installed

    if _installed:
        return

    _original_mul = VDR.__mul__
    _original_rmul = VDR.__rmul__
    _original_div = VDR.__truediv__
    _original_rdiv = VDR.__rtruediv__

    VDR.__mul__ = _patched_mul
    VDR.__rmul__ = _patched_rmul
    VDR.__truediv__ = _patched_div
    VDR.__rtruediv__ = _patched_rdiv

    _installed = True


def uninstall():
    """Restore original VDR operators (active mul/div raises)."""
    global _installed

    if not _installed:
        return

    VDR.__mul__ = _original_mul
    VDR.__rmul__ = _original_rmul
    VDR.__truediv__ = _original_div
    VDR.__rtruediv__ = _original_rdiv

    _installed = False
