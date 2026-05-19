"""
VDR — Value, Denominator, Remainder.
Exact finite discrete representation in irreducible triple form.

A VDR object is [V, D, R] where:
    V in Z           — value slot (settled numerator)
    D in Z \\ {0}     — denominator slot (frame)
    R in Remainder    — remainder slot (exact unresolved structure)

Remainder is first-class. Never error. Never residue.

R is either:
    an integer (atomic remainder), or
    an integer base plus a finite list of child VDR objects (composite remainder)

Recursion exists only in R. Every valid object is a finite tree.
No limits, no approximation, no infinity.

Closed object:  [V, D, 0]  — projects to V/D exactly
Active object:  [V, D, R]  with R != 0 — carries exact remainder state
"""

from __future__ import annotations
from fractions import Fraction
from math import gcd
from typing import List, Optional, Tuple, Union

__all__ = [
    "VDR",
    "Remainder",
    "VDRError",
    "ZeroDenominatorError",
    "InvalidStructureError",
    "RebaseError",
    "ArithmeticFailure",
    "ZERO",
    "ONE",
    "NEG_ONE",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class VDRError(Exception):
    """Base error for all VDR operations."""
    pass


class ZeroDenominatorError(VDRError):
    pass


class InvalidStructureError(VDRError):
    pass


class RebaseError(VDRError):
    pass


class ArithmeticFailure(VDRError):
    pass


# ---------------------------------------------------------------------------
# Basis frame detection
# ---------------------------------------------------------------------------

def _get_basis_denom():
    """
    Return the current default basis denominator, or None if basis module
    is not loaded. Lazy import to avoid circular dependency.
    """
    try:
        from vdr.basis import get_default
        return 2 ** get_default()
    except ImportError:
        return None


def _both_in_basis(a, b):
    """
    True when both operands share the same D and that D equals the
    current default basis denominator.
    """
    if a.d != b.d:
        return False
    denom = _get_basis_denom()
    if denom is None:
        return False
    return a.d == denom


def _one_in_basis(a, b):
    """
    Check if exactly one operand is in the basis frame.
    Returns (basis_operand, other_operand) or (None, None).
    """
    denom = _get_basis_denom()
    if denom is None:
        return None, None
    if a.d == denom and b.d != denom:
        return a, b
    if b.d == denom and a.d != denom:
        return b, a
    return None, None


def _rebase_to_basis(x):
    """
    Project a non-basis VDR onto the current basis frame.
    Returns a closed VDR with D = basis denominator.
    """
    from vdr.basis import to_qbasis
    return to_qbasis(x)


# ---------------------------------------------------------------------------
# Remainder
# ---------------------------------------------------------------------------

class Remainder:
    """
    The remainder slot of a VDR triple.

    Atomic form:   Remainder(base=r, children=[])
    Composite form: Remainder(base=r, children=[X1, X2, ...])

    R = r + X1 + X2 + ... + Xn
    where r is an integer and each Xi is a valid VDR object.
    """

    __slots__ = ("base", "children")

    def __init__(self, base=0, children=None):
        if not isinstance(base, int):
            raise InvalidStructureError(
                "Remainder base must be int, got %s" % type(base).__name__
            )
        self.base = base
        self.children = list(children) if children else []
        for c in self.children:
            if not isinstance(c, VDR):
                raise InvalidStructureError(
                    "Remainder child must be VDR, got %s" % type(c).__name__
                )

    # -- predicates --------------------------------------------------------

    @property
    def is_zero(self):
        """True when base is 0 and no children."""
        return self.base == 0 and len(self.children) == 0

    @property
    def is_atomic(self):
        return len(self.children) == 0

    @property
    def is_functional(self):
        return False

    @property
    def is_globally_zero(self):
        """True when base is 0 and all children are globally closed with value 0."""
        if self.base != 0:
            return False
        for c in self.children:
            if not c.is_globally_closed:
                return True
            if c.v != 0:
                return True
        return True

    # -- equality ----------------------------------------------------------

    def structural_eq(self, other):
        if not isinstance(other, Remainder):
            return False
        if self.base != other.base:
            return False
        if len(self.children) != len(other.children):
            return False
        return all(
            a.structural_eq(b)
            for a, b in zip(self.children, other.children)
        )

    # -- operations --------------------------------------------------------

    def negate(self):
        """-(r + X1 + ... + Xn) = -r + (-X1) + ... + (-Xn)"""
        return Remainder(
            -self.base,
            [c.negate() for c in self.children],
        )

    def combine(self, other, sign=1):
        """
        Same-frame remainder combination.
        sign=1  for addition:    R1 + R2
        sign=-1 for subtraction: R1 - R2
        """
        if sign == 1:
            new_base = self.base + other.base
            new_children = list(self.children) + list(other.children)
        else:
            new_base = self.base - other.base
            new_children = list(self.children) + [
                c.negate() for c in other.children
            ]
        return Remainder(new_base, new_children)

    def lift(self, k):
        """
        Transport remainder into a scaled denominator frame.

        lift(r, k) = k * r
        lift(r + X1+...+Xn, k) = k*r + lift(X1,k) + ... + lift(Xn,k)

        Child VDR lift: lift([V, D, R], k) = [k*V, D, lift(R, k)]
        """
        if k == 0:
            raise VDRError("lift by zero is invalid")
        return Remainder(
            self.base * k,
            [c._lift_vdr(k) for c in self.children],
        )

    # -- projection --------------------------------------------------------

    def legacy_value(self):
        """
        Additive flattening for scalar comparison.
        legacy(r) = r
        legacy(r + X1+...+Xn) = r + Fraction(X1) + ... + Fraction(Xn)
        """
        total = Fraction(self.base)
        for c in self.children:
            total += c.to_fraction()
        return total

    # -- normalization -----------------------------------------------------

    def normalize(self):
        """
        Normalize remainder structure:
        1. Normalize all children recursively
        2. Remove zero-value closed children
        3. Merge closed children sharing a denominator
        4. Absorb D=1 closed children into base
        5. Sort children canonically
        """
        normed = [c.normalize() for c in self.children]

        live = []
        absorbed_base = self.base
        for c in normed:
            if c.is_globally_closed and c.v == 0:
                continue
            live.append(c)

        merged = _merge_same_denom_children(live)

        final_children = []
        for c in merged:
            if c.is_globally_closed and c.d == 1:
                absorbed_base += c.v
            else:
                final_children.append(c)

        final_children.sort(key=_child_sort_key)

        return Remainder(absorbed_base, final_children)

    # -- display -----------------------------------------------------------

    def __repr__(self):
        if self.is_atomic:
            return str(self.base)
        parts = [str(self.base)]
        for c in self.children:
            parts.append(repr(c))
        return " + ".join(parts)

    def __eq__(self, other):
        if isinstance(other, int) and self.is_atomic:
            return self.base == other
        if isinstance(other, Remainder):
            return self.structural_eq(other)
        return NotImplemented

    def __hash__(self):
        return hash((self.base, tuple(self.children)))


# ---------------------------------------------------------------------------
# VDR
# ---------------------------------------------------------------------------

class VDR:
    """
    Exact finite discrete triple: [V, D, R]

    Construction:
        VDR(3)              -> [3, 1, 0]   integer
        VDR(1, 2)           -> [1, 2, 0]   rational 1/2
        VDR(1, 3, Remainder(1))  -> [1, 3, 1]   active
        VDR.from_fraction(Fraction(5, 6))  -> [5, 6, 0]
    """

    __slots__ = ("v", "d", "r")

    def __init__(self, v, d=1, r=None):
        if not isinstance(v, int):
            raise InvalidStructureError(
                "V must be int, got %s" % type(v).__name__
            )
        if not isinstance(d, int):
            raise InvalidStructureError(
                "D must be int, got %s" % type(d).__name__
            )
        if d == 0:
            raise ZeroDenominatorError("D must not be zero")

        self.v = v
        self.d = d

        if r is None:
            self.r = Remainder(0)
        elif isinstance(r, int):
            self.r = Remainder(r)
        elif isinstance(r, Remainder):
            self.r = r
        else:
            raise InvalidStructureError(
                "R must be int, Remainder, or None, got %s" % type(r).__name__
            )

    # -- class constructors ------------------------------------------------

    @classmethod
    def from_fraction(cls, frac):
        """Exact construction from fractions.Fraction."""
        return cls(frac.numerator, frac.denominator)

    @classmethod
    def from_int(cls, n):
        """Exact construction from integer."""
        return cls(n, 1)

    # -- predicates --------------------------------------------------------

    @property
    def is_closed(self):
        """Top-level remainder is zero."""
        return self.r.is_zero

    @property
    def is_globally_closed(self):
        """All remainders in the entire tree are zero."""
        return self.r.is_zero and self.r.is_globally_zero

    @property
    def is_active(self):
        return not self.is_closed

    # -- equality ----------------------------------------------------------

    def structural_eq(self, other):
        """X =s Y iff every slot matches exactly, recursively."""
        if not isinstance(other, VDR):
            return False
        return (
            self.v == other.v
            and self.d == other.d
            and self.r.structural_eq(other.r)
        )

    def value_eq(self, other):
        """X =n Y iff norm(X) =s norm(Y)."""
        if not isinstance(other, VDR):
            return False
        return self.normalize().structural_eq(other.normalize())

    def __eq__(self, other):
        """Python == uses value equality."""
        if isinstance(other, VDR):
            return self.value_eq(other)
        if isinstance(other, int):
            return self.value_eq(VDR(other))
        if isinstance(other, Fraction):
            return self.value_eq(VDR.from_fraction(other))
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self):
        n = self.normalize()
        return hash((n.v, n.d, n.r))

    # -- normalization -----------------------------------------------------

    def normalize(self):
        """
        Produce canonical form:
        1. Normalize remainder recursively
        2. Sign convention: D > 0
        3. Check if remainder is value-equivalent to zero -> collapse to closed
        4. GCD reduce (V, D) for closed nodes
        5. For active nodes, reduce if remainder is cleanly divisible
        """
        nr = self.r.normalize()
        v, d = self.v, self.d

        # sign convention: positive denominator
        if d < 0:
            v, d = -v, -d

        # Check if remainder is value-equivalent to zero.
        # This fixes the Newton perfect-square problem: sqrt(4) producing
        # [2k, k, 0] instead of [2, 1, 0], and value-zero remainders
        # that are structurally nonzero.
        remainder_is_zero = nr.is_zero
        if not remainder_is_zero:
            # Check: is the legacy value of this remainder exactly 0?
            # This catches structurally nonzero but value-zero remainders.
            try:
                if nr.legacy_value() == 0:
                    remainder_is_zero = True
            except Exception:
                pass

        if remainder_is_zero:
            # Absorb any remainder base into v if needed
            if not nr.is_zero and nr.base != 0:
                v = v + nr.base
            g = gcd(abs(v), abs(d))
            if g > 0:
                v, d = v // g, d // g
            return VDR(v, d, Remainder(0))

        # Roll over atomic remainder when |base| >= D
        if nr.is_atomic and abs(nr.base) >= d:
            q, s = divmod(nr.base, d)
            v = v + q
            nr = Remainder(s)
            if nr.is_zero:
                g = gcd(abs(v), abs(d))
                if g > 0:
                    v, d = v // g, d // g
                return VDR(v, d, Remainder(0))

        # Active node: try to reduce V, D if remainder is divisible
        g = gcd(abs(v), abs(d))
        if g > 1:
            if _remainder_divisible_by(nr, g):
                v, d = v // g, d // g
                nr = _remainder_divide(nr, g)

        return VDR(v, d, nr)

    # -- projection --------------------------------------------------------

    def to_fraction(self):
        """
        Exact projection to fractions.Fraction.

        Closed: V/D
        Active: (V + legacy(R)) / D
        """
        return Fraction(self.v, self.d) + Fraction(
            self.r.legacy_value(), self.d
        )

    def to_float(self):
        """Lossy export to float. Loss belongs to float format."""
        return float(self.to_fraction())

    def __float__(self):
        return self.to_float()

    # -- basis-aware arithmetic --------------------------------------------

    def _basis_add_or_fallback(self, other, sign):
        """
        Addition/subtraction with basis frame awareness.

        Both in basis, same D: integer add/sub in frame.
        One in basis, other not: rebase other into basis, then integer op.
        Neither in basis: cross-multiply as before.
        """
        if _both_in_basis(self, other):
            if sign == 1:
                return VDR(self.v + other.v, self.d)
            else:
                return VDR(self.v - other.v, self.d)

        basis_op, non_basis_op = _one_in_basis(self, other)
        if basis_op is not None:
            rebased = _rebase_to_basis(non_basis_op)
            # figure out which was self and which was other
            if self.d == basis_op.d and self.v == basis_op.v:
                if sign == 1:
                    return VDR(self.v + rebased.v, self.d)
                else:
                    return VDR(self.v - rebased.v, self.d)
            else:
                # other was the basis operand, self was rebased
                if sign == 1:
                    return VDR(rebased.v + other.v, other.d)
                else:
                    return VDR(rebased.v - other.v, other.d)

        return None  # signal: no basis path, use fallback

    def __add__(self, other):
        from vdr._compat import _coerce
        other = _coerce(other)
        if self.is_closed and other.is_closed:
            result = self._basis_add_or_fallback(other, sign=1)
            if result is not None:
                return result
            return VDR(
                self.v * other.d + other.v * self.d,
                self.d * other.d,
            ).normalize()
        return _active_add(self, other, sign=1)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        from vdr._compat import _coerce
        other = _coerce(other)
        if self.is_closed and other.is_closed:
            result = self._basis_add_or_fallback(other, sign=-1)
            if result is not None:
                return result
            return VDR(
                self.v * other.d - other.v * self.d,
                self.d * other.d,
            ).normalize()
        return _active_add(self, other, sign=-1)

    def __rsub__(self, other):
        from vdr._compat import _coerce
        return _coerce(other).__sub__(self)

    def __mul__(self, other):
        from vdr._compat import _coerce
        other = _coerce(other)
        if self.is_closed and other.is_closed:
            if _both_in_basis(self, other):
                return _basis_mul(self, other)
            return VDR(
                self.v * other.v,
                self.d * other.d,
            ).normalize()
        raise ArithmeticFailure(
            "Active multiplication requires vdr.active.install()"
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        from vdr._compat import _coerce
        other = _coerce(other)
        if other.is_closed and other.v == 0:
            raise ArithmeticFailure("Division by zero")
        if self.is_closed and other.is_closed:
            if _both_in_basis(self, other):
                return _basis_div(self, other)
            return VDR(
                self.v * other.d,
                self.d * other.v,
            ).normalize()
        raise ArithmeticFailure(
            "Active division requires vdr.active.install()"
        )

    def __rtruediv__(self, other):
        from vdr._compat import _coerce
        return _coerce(other).__truediv__(self)

    def __neg__(self):
        """-[V, D, R] = [-V, D, -R]"""
        return VDR(-self.v, self.d, self.r.negate())

    def __pos__(self):
        return self

    def __abs__(self):
        if self.to_fraction() < 0:
            return -self
        return VDR(self.v, self.d, self.r)

    # -- rebase ------------------------------------------------------------

    def rebase(self, target_d):
        """
        Change top-level denominator to target_d, preserving exact value.

        Closed rebase: succeeds when V*B/D is integer
        Active rebase: [Q, B, [S,D,0] + lift(R, B)]
            where V*B = Q*D + S
        """
        if not isinstance(target_d, int) or target_d == 0:
            raise RebaseError("Target denominator must be nonzero integer")

        if target_d == self.d:
            return VDR(self.v, self.d, self.r)

        n = self.v * target_d
        q, s = divmod(n, self.d)

        if s == 0 and self.r.is_zero:
            return VDR(q, target_d).normalize()

        # active rebase
        mismatch = VDR(s, self.d)
        lifted_r = self.r.lift(target_d)

        if mismatch.v == 0:
            new_r = lifted_r
        else:
            new_children = [mismatch]
            if not lifted_r.is_zero:
                new_children.extend(lifted_r.children)
            new_r = Remainder(lifted_r.base, new_children)

        return VDR(q, target_d, new_r).normalize()

    # -- lift (on VDR object) ----------------------------------------------

    def _lift_vdr(self, k):
        """lift([V, D, R], k) = [k*V, D, lift(R, k)]"""
        return VDR(k * self.v, self.d, self.r.lift(k))

    # -- negation helper ---------------------------------------------------

    def negate(self):
        return self.__neg__()

    # -- structural metrics ------------------------------------------------

    def depth(self):
        """Recursive depth of the VDR tree."""
        if self.r.is_zero:
            return 0
        if not self.r.children:
            return 0
        return 1 + max(c.depth() for c in self.r.children)

    def size(self):
        """Total node count."""
        return 1 + _remainder_size(self.r)

    def den_complexity(self):
        """Denominator complexity: (distinct_count, magnitude_sum, node_count)."""
        denoms = []
        _collect_denoms(self, denoms)
        abs_denoms = [abs(dd) for dd in denoms]
        unique = len(set(abs_denoms))
        total = sum(abs_denoms)
        count = len(abs_denoms)
        return (unique, total, count)

    # -- display -----------------------------------------------------------

    def __repr__(self):
        return "VDR(%s, %s, %s)" % (self.v, self.d, repr(self.r))

    def __str__(self):
        return "[%s, %s, %s]" % (self.v, self.d, self.r)

    def bracket(self):
        """Native bracket notation."""
        return str(self)

    # -- comparison (ordering via projection) ------------------------------

    def __lt__(self, other):
        from vdr._compat import _coerce
        other = _coerce(other)
        return self.to_fraction() < other.to_fraction()

    def __le__(self, other):
        from vdr._compat import _coerce
        other = _coerce(other)
        return self.to_fraction() <= other.to_fraction()

    def __gt__(self, other):
        from vdr._compat import _coerce
        other = _coerce(other)
        return self.to_fraction() > other.to_fraction()

    def __ge__(self, other):
        from vdr._compat import _coerce
        other = _coerce(other)
        return self.to_fraction() >= other.to_fraction()


# ---------------------------------------------------------------------------
# Basis-aware arithmetic helpers
# ---------------------------------------------------------------------------

def _basis_mul(a, b):
    """
    Multiplication with divmod back to basis frame.
    D stays fixed. Overflow goes to R. Zero loss.
    """
    denom = a.d
    product = a.v * b.v
    q, s = divmod(product, denom)
    if s == 0:
        return VDR(q, denom)
    return VDR(q, denom, Remainder(0, [VDR(s, denom)]))


def _basis_div(a, b):
    """
    Division with divmod back to basis frame.
    D stays fixed. Mismatch witness in R.
    """
    denom = a.d
    if b.v == 0:
        raise ArithmeticFailure("Division by zero in basis frame")
    numerator = a.v * denom
    q, s = divmod(numerator, b.v)
    if s == 0:
        return VDR(q, denom)
    return VDR(q, denom, Remainder(0, [VDR(s, b.v)]))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _active_add(a, b, sign=1):
    """
    General addition (sign=1) or subtraction (sign=-1).

    Same denominator:
        [V1,D,R1] + [V2,D,R2] = [V1+V2, D, R1+R2]

    Different denominator:
        Basis-aware: rebase non-basis operand if one is in basis.
        Otherwise: cross-scale to D1*D2, lift remainders.
    """
    if a.d == b.d:
        if sign == 1:
            new_v = a.v + b.v
        else:
            new_v = a.v - b.v
        new_r = a.r.combine(b.r, sign=sign)
        return VDR(new_v, a.d, new_r).normalize()

    # different D — check if we should rebase into basis frame
    basis_op, non_basis_op = _one_in_basis(a, b)
    if basis_op is not None:
        rebased = _rebase_to_basis(non_basis_op)
        denom = basis_op.d
        # determine operand order for subtraction
        if a.d == denom:
            a_v, b_v = a.v, rebased.v
            a_r, b_r = a.r, rebased.r
        else:
            a_v, b_v = rebased.v, b.v
            a_r, b_r = rebased.r, b.r
        if sign == 1:
            new_v = a_v + b_v
        else:
            new_v = a_v - b_v
        new_r = a_r.combine(b_r, sign=sign)
        return VDR(new_v, denom, new_r).normalize()

    new_d = a.d * b.d
    a_lifted_r = a.r.lift(b.d)
    b_lifted_r = b.r.lift(a.d)

    if sign == 1:
        new_v = a.v * b.d + b.v * a.d
    else:
        new_v = a.v * b.d - b.v * a.d

    new_r = a_lifted_r.combine(b_lifted_r, sign=sign)
    return VDR(new_v, new_d, new_r).normalize()


def _remainder_divisible_by(r, g):
    """Check if a remainder can be cleanly divided by g."""
    if r.base % g != 0:
        return False
    for c in r.children:
        if c.v % g != 0:
            return False
        if not _remainder_divisible_by(c.r, g):
            return False
    return True


def _remainder_divide(r, g):
    """Divide remainder structure by g (must be pre-checked)."""
    new_children = []
    for c in r.children:
        new_children.append(VDR(c.v // g, c.d, _remainder_divide(c.r, g)))
    return Remainder(r.base // g, new_children)


def _merge_same_denom_children(children):
    """Merge closed children sharing a denominator."""
    if not children:
        return []

    by_denom = {}
    non_closed = []
    for c in children:
        if c.is_globally_closed:
            key = abs(c.d)
            if key not in by_denom:
                by_denom[key] = []
            by_denom[key].append(c)
        else:
            non_closed.append(c)

    merged = []
    for key in sorted(by_denom.keys()):
        group = by_denom[key]
        if len(group) == 1:
            merged.append(group[0])
        else:
            total = group[0]
            for g in group[1:]:
                total = total + g
            if total.v != 0 or not total.is_globally_closed:
                merged.append(total.normalize())

    return merged + non_closed


def _child_sort_key(c):
    """Canonical sort key for remainder children."""
    return (abs(c.d), c.d, c.v, c.r.base)


def _remainder_size(r):
    """size(R): 1 for atomic base + sum of child sizes."""
    total = 1
    for c in r.children:
        total += c.size()
    return total


def _collect_denoms(x, acc):
    """Collect all denominators in the tree."""
    acc.append(x.d)
    for c in x.r.children:
        _collect_denoms(c, acc)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

ZERO = VDR(0, 1)
ONE = VDR(1, 1)
NEG_ONE = VDR(-1, 1)
