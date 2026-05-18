"""
vdr-math — Exact arithmetic with Value, Denominator, Remainder triples.

    from vdr import VDR, Remainder
    x = VDR(1, 3)   # exact 1/3
    y = VDR(2, 7)   # exact 2/7
    z = x + y       # exact, normalized
    z = x * y       # active mul, auto-installed

Remainder is first-class. Never error. Never residue.
"""

from vdr.core import (
    VDR,
    Remainder,
    VDRError,
    ZeroDenominatorError,
    InvalidStructureError,
    RebaseError,
    ArithmeticFailure,
    ZERO,
    ONE,
    NEG_ONE,
)
from vdr.linalg import Vec, Mat

from vdr import active as _active
from vdr import fn as _fn

_active.install()
_fn.install()

__version__ = "0.1.0"

__all__ = [
    "VDR",
    "Remainder",
    "Vec",
    "Mat",
    "VDRError",
    "ZeroDenominatorError",
    "InvalidStructureError",
    "RebaseError",
    "ArithmeticFailure",
    "ZERO",
    "ONE",
    "NEG_ONE",
]
