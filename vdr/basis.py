"""
vdr.basis — D-frame management and Q335 basis.

The D-frame is the denominator you keep. divmod keeps it stable,
overflow goes to R. D never explodes.

Q335 (D = 2^335) is the default — proven across ~1000 tests in physics,
transcendental, and ML domains. But any D is valid. If you're doing
pure rational arithmetic with small denominators, D might be 1. If
you're doing binary fixed-point, it might be 2^16.

    import vdr.basis
    vdr.basis.set_default(bits=335)   # Q335, the default

    from vdr.basis import to_qbasis, qb_mul
    a = to_qbasis(3.14159, bits=335)  # project onto 2^335 grid
    b = to_qbasis(2.71828, bits=335)
    c = qb_mul(a, b)                  # D stays 2^335, overflow in R
"""

from __future__ import annotations
from fractions import Fraction
from typing import Optional, Union, TYPE_CHECKING

from vdr.core import VDR, Remainder, VDRError

if TYPE_CHECKING:
    from vdr.linalg import Vec, Mat

__all__ = [
    "DEFAULT_BITS",
    "Q335",
    "set_default",
    "get_default",
    "q_basis_denominator",
    "to_qbasis",
    "vec_to_qbasis",
    "mat_to_qbasis",
    "qb_add",
    "qb_mul",
    "qb_div",
]


# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------

DEFAULT_BITS = 335
Q335 = 2 ** 335

def get_default():
    """Return current default bit width."""
    return DEFAULT_BITS

def set_default(bits):
    """
    Set the default basis bit width.

        vdr.basis.set_default(668)  # 200-digit precision
    """
    global DEFAULT_BITS
    if not isinstance(bits, int) or bits < 1:
        raise ValueError("bits must be a positive integer, got %r" % bits)
    DEFAULT_BITS = bits


# ---------------------------------------------------------------------------
# Denominator construction
# ---------------------------------------------------------------------------

def q_basis_denominator(bits=None):
    """
    Return the basis denominator 2^bits.

    If bits is None, uses DEFAULT_BITS.

        q_basis_denominator()      # 2^335 by default
        q_basis_denominator(668)   # 2^668 for 200 digits
    """
    if bits is None:
        bits = DEFAULT_BITS
    return 2 ** bits


# ---------------------------------------------------------------------------
# Projection onto basis
# ---------------------------------------------------------------------------

def to_qbasis(x, bits=None):
    """
    Project a value onto the 2^bits grid as [round(x * 2^bits), 2^bits, 0].

    Accepts VDR, int, float, or Fraction.

        to_qbasis(VDR(22, 7))         # 22/7 projected onto Q335
        to_qbasis(3.14159)            # float projected (one-time boundary loss)
        to_qbasis(Fraction(1, 3))     # exact rational projected
    """
    if bits is None:
        bits = DEFAULT_BITS
    denom = 2 ** bits

    if isinstance(x, VDR):
        frac = x.to_fraction()
    elif isinstance(x, int):
        return VDR(x * denom, denom)
    elif isinstance(x, float):
        frac = Fraction(x).limit_denominator(10 ** 20)
    elif isinstance(x, Fraction):
        frac = x
    else:
        raise TypeError("Cannot project %s to Q basis" % type(x).__name__)

    # round(frac * denom) — exact integer rounding
    num = frac.numerator * denom
    den = frac.denominator
    # integer division with rounding to nearest
    q, s = divmod(num, den)
    if 2 * s >= den:
        q += 1
    return VDR(q, denom)


def vec_to_qbasis(v, bits=None):
    """Project each element of a Vec onto the basis."""
    from vdr.linalg import Vec
    if bits is None:
        bits = DEFAULT_BITS
    return Vec([to_qbasis(x, bits) for x in v])


def mat_to_qbasis(m, bits=None):
    """Project each element of a Mat onto the basis."""
    from vdr.linalg import Mat
    if bits is None:
        bits = DEFAULT_BITS
    rows = []
    for i in range(m.nrows):
        row = [to_qbasis(m[i, j], bits) for j in range(m.ncols)]
        rows.append(row)
    return Mat(rows)


# ---------------------------------------------------------------------------
# Basis-frame arithmetic
# ---------------------------------------------------------------------------

def qb_add(a, b, bits=None):
    """
    Addition staying in basis frame.

    Both operands should share the same D = 2^bits.
    Result: [(p1 + p2), 2^bits, 0] — one integer add.

    If operands have different D, rebases both first.
    """
    if bits is None:
        bits = DEFAULT_BITS
    denom = 2 ** bits

    # If both already in the right frame, do direct integer add
    if isinstance(a, VDR) and a.d == denom and a.is_closed and \
       isinstance(b, VDR) and b.d == denom and b.is_closed:
        return VDR(a.v + b.v, denom)

    a = _ensure_basis(a, denom, bits)
    b = _ensure_basis(b, denom, bits)

    return VDR(a.v + b.v, denom)


def qb_mul(a, b, bits=None):
    """
    Multiplication with divmod back to basis frame.

    D stays 2^bits. Overflow in R. Zero loss.

    Product p1*p2 is a big integer. divmod by 2^bits:
        Q, S = divmod(p1 * p2, 2^bits)
        Result: [Q, 2^bits, [S, 2^bits, 0]]

    D never changed. R caught what V couldn't absorb — exactly.
    """
    if bits is None:
        bits = DEFAULT_BITS
    denom = 2 ** bits

    a = _ensure_basis(a, denom, bits)
    b = _ensure_basis(b, denom, bits)

    product = a.v * b.v
    q, s = divmod(product, denom)

    if s == 0:
        return VDR(q, denom)

    return VDR(q, denom, Remainder(0, [VDR(s, denom)]))


def qb_div(a, b, bits=None):
    """
    Division with divmod back to basis frame.

    a / b where both are in 2^bits frame.
    Multiply a.v by 2^bits, divmod by b.v.
    Odd factors go into R.
    """
    if bits is None:
        bits = DEFAULT_BITS
    denom = 2 ** bits

    a = _ensure_basis(a, denom, bits)
    b = _ensure_basis(b, denom, bits)

    if b.v == 0:
        from vdr.core import ArithmeticFailure
        raise ArithmeticFailure("Division by zero in Q basis")

    # a/b = (a.v / 2^bits) / (b.v / 2^bits) = a.v / b.v
    # We want this as [Q, 2^bits, remainder]
    # Multiply numerator by denom to keep in frame:
    # (a.v * 2^bits) / b.v = Q remainder S
    numerator = a.v * denom
    q, s = divmod(numerator, b.v)

    if s == 0:
        return VDR(q, denom)

    # Remainder: [S, b.v, 0] represents the exact unresolved part.
    # But we want it in the 2^bits frame. The mismatch witness
    # carries the odd factor from b.v.
    return VDR(q, denom, Remainder(0, [VDR(s, b.v)]))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_basis(x, denom, bits):
    """Ensure x is in the target basis frame."""
    if isinstance(x, VDR):
        if x.d == denom and x.is_closed:
            return x
        # rebase or project
        return to_qbasis(x, bits)
    return to_qbasis(x, bits)
