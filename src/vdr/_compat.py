"""
vdr._compat — Internal coercion helpers shared across modules.
Not public API.
"""

from __future__ import annotations
from fractions import Fraction
from typing import Union


def _coerce(other):
    """Coerce int or Fraction into VDR for arithmetic."""
    from vdr.core import VDR
    if isinstance(other, VDR):
        return other
    if isinstance(other, int):
        return VDR(other)
    if isinstance(other, Fraction):
        return VDR.from_fraction(other)
    raise TypeError("Cannot coerce %s to VDR" % type(other).__name__)


def _to_vdr(x):
    """Convert int to VDR, pass VDR through."""
    from vdr.core import VDR
    if isinstance(x, VDR):
        return x
    if isinstance(x, int):
        return VDR(x)
    raise TypeError("Expected VDR or int, got %s" % type(x).__name__)

