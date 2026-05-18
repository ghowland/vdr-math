"""
vdr.linalg — Exact rational linear algebra over VDR objects.

    from vdr.linalg import Vec, Mat

    v = Vec([VDR(1,2), VDR(1,3), VDR(1,7)])
    m = Mat.identity(3)
    d = m.det()           # exact
    m_inv = m.inv()       # exact

All operations use exact VDR arithmetic. Zero drift.

Determinant, inverse, and solve dispatch automatically:
    n <= 4: cofactor / adjugate / Cramer (simple, exact)
    n >= 5: Gaussian elimination O(n^3) (practical, exact)
"""

from __future__ import annotations
from fractions import Fraction
from typing import List, Union, Optional, Tuple

from vdr.core import VDR, Remainder, VDRError, InvalidStructureError

__all__ = [
    "Vec",
    "Mat",
    "LinAlgError",
    "parse_vdr",
    "vdr_to_dict",
    "vdr_from_dict",
    "vdr_to_latex",
]


class LinAlgError(VDRError):
    """Linear algebra specific errors."""
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_vdr(x):
    if isinstance(x, VDR):
        return x
    if isinstance(x, int):
        return VDR(x)
    raise TypeError("Expected VDR or int, got %s" % type(x).__name__)


def _check_same_dim(a, b, op):
    if len(a) != len(b):
        raise LinAlgError(
            "Vec %s requires same dimension: %d vs %d" % (op, len(a), len(b))
        )


def _check_same_shape(a, b, op):
    if a.shape != b.shape:
        raise LinAlgError(
            "Mat %s requires same shape: %s vs %s" % (op, a.shape, b.shape)
        )


# ---------------------------------------------------------------------------
# Vec
# ---------------------------------------------------------------------------

class Vec:
    """
    Exact VDR vector — an ordered list of VDR objects.

        v = Vec([VDR(1,2), VDR(1,3)])
        w = Vec.from_ints([1, 2, 3])
        v + w, v - w, v * VDR(2), v.dot(w)
    """

    __slots__ = ("_data",)

    def __init__(self, data):
        self._data = [_to_vdr(x) for x in data]

    @classmethod
    def from_ints(cls, ns):
        return cls([VDR(n) for n in ns])

    @classmethod
    def from_fracs(cls, pairs):
        """Vec.from_fracs([(1,2), (3,4)]) -> Vec([VDR(1,2), VDR(3,4)])"""
        return cls([VDR(a, b) for a, b in pairs])

    @classmethod
    def zero(cls, n):
        return cls([VDR(0)] * n)

    # -- access ------------------------------------------------------------

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

    def __iter__(self):
        return iter(self._data)

    @property
    def dim(self):
        return len(self._data)

    # -- arithmetic --------------------------------------------------------

    def __add__(self, other):
        _check_same_dim(self, other, "+")
        return Vec([a + b for a, b in zip(self._data, other._data)])

    def __sub__(self, other):
        _check_same_dim(self, other, "-")
        return Vec([a - b for a, b in zip(self._data, other._data)])

    def __neg__(self):
        return Vec([-x for x in self._data])

    def scale(self, s):
        """Scalar multiplication: s * v"""
        s = _to_vdr(s)
        return Vec([x * s for x in self._data])

    def __mul__(self, other):
        return self.scale(other)

    def __rmul__(self, other):
        return self.scale(other)

    def dot(self, other):
        """Exact dot product: v . w = sum(vi * wi)"""
        _check_same_dim(self, other, "dot")
        total = VDR(0)
        for a, b in zip(self._data, other._data):
            total = total + a * b
        return total

    def norm_sq(self):
        """Squared norm: v . v (no sqrt, stays exact)."""
        return self.dot(self)

    # -- comparison --------------------------------------------------------

    def __eq__(self, other):
        if not isinstance(other, Vec):
            return NotImplemented
        if len(self) != len(other):
            return False
        return all(a == b for a, b in zip(self._data, other._data))

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    # -- display -----------------------------------------------------------

    def __repr__(self):
        return "Vec([%s])" % ", ".join(str(x) for x in self._data)

    def to_fractions(self):
        return [x.to_fraction() for x in self._data]


# ---------------------------------------------------------------------------
# Mat
# ---------------------------------------------------------------------------

class Mat:
    """
    Exact VDR matrix — list of row Vecs, all same dimension.

        m = Mat.from_ints([[1,2],[3,4]])
        m.det(), m.inv(), m.solve(b), m.rank()
    """

    __slots__ = ("_rows",)

    def __init__(self, rows):
        if not rows:
            raise LinAlgError("Matrix must have at least one row")
        parsed = []
        for row in rows:
            if isinstance(row, Vec):
                parsed.append(row)
            else:
                parsed.append(Vec(row))
        ncols = len(parsed[0])
        for i, row in enumerate(parsed):
            if len(row) != ncols:
                raise LinAlgError(
                    "Row %d has %d cols, expected %d" % (i, len(row), ncols)
                )
        self._rows = parsed

    @classmethod
    def from_ints(cls, data):
        return cls([[VDR(x) for x in row] for row in data])

    @classmethod
    def from_fracs(cls, data):
        """Mat.from_fracs([[(1,2),(3,4)],[(5,6),(7,8)]])"""
        return cls([[VDR(a, b) for a, b in row] for row in data])

    @classmethod
    def identity(cls, n):
        rows = []
        for i in range(n):
            row = [VDR(0)] * n
            row[i] = VDR(1)
            rows.append(row)
        return cls(rows)

    @classmethod
    def zero(cls, nrows, ncols):
        return cls([[VDR(0)] * ncols for _ in range(nrows)])

    # -- access ------------------------------------------------------------

    @property
    def nrows(self):
        return len(self._rows)

    @property
    def ncols(self):
        return len(self._rows[0])

    @property
    def shape(self):
        return (self.nrows, self.ncols)

    @property
    def is_square(self):
        return self.nrows == self.ncols

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            r, c = idx
            return self._rows[r][c]
        return self._rows[idx]

    def row(self, i):
        return self._rows[i]

    def col(self, j):
        return Vec([self._rows[i][j] for i in range(self.nrows)])

    # -- arithmetic --------------------------------------------------------

    def __add__(self, other):
        _check_same_shape(self, other, "+")
        return Mat([a + b for a, b in zip(self._rows, other._rows)])

    def __sub__(self, other):
        _check_same_shape(self, other, "-")
        return Mat([a - b for a, b in zip(self._rows, other._rows)])

    def __neg__(self):
        return Mat([-r for r in self._rows])

    def scale(self, s):
        """Scalar multiplication."""
        return Mat([r.scale(s) for r in self._rows])

    def __mul__(self, other):
        if isinstance(other, Mat):
            return self.matmul(other)
        if isinstance(other, Vec):
            return self.matvec(other)
        return self.scale(other)

    def __rmul__(self, other):
        return self.scale(other)

    def matmul(self, other):
        """Exact matrix multiplication: C[i,j] = sum A[i,k]*B[k,j]"""
        if self.ncols != other.nrows:
            raise LinAlgError(
                "Cannot multiply %dx%d by %dx%d" % (
                    self.nrows, self.ncols, other.nrows, other.ncols
                )
            )
        result = []
        for i in range(self.nrows):
            row = []
            for j in range(other.ncols):
                total = VDR(0)
                for k in range(self.ncols):
                    total = total + self[i, k] * other[k, j]
                row.append(total)
            result.append(row)
        return Mat(result)

    def matvec(self, v):
        """Matrix-vector product: Ax"""
        if self.ncols != len(v):
            raise LinAlgError(
                "Cannot multiply %dx%d matrix by %d-vector" % (
                    self.nrows, self.ncols, len(v)
                )
            )
        return Vec([self.row(i).dot(v) for i in range(self.nrows)])

    # -- transpose ---------------------------------------------------------

    @property
    def T(self):
        return Mat([
            [self[i, j] for i in range(self.nrows)]
            for j in range(self.ncols)
        ])

    # -- trace -------------------------------------------------------------

    def trace(self):
        """Sum of diagonal elements."""
        if not self.is_square:
            raise LinAlgError("Trace requires square matrix")
        total = VDR(0)
        for i in range(self.nrows):
            total = total + self[i, i]
        return total

    # -- determinant -------------------------------------------------------

    def det(self):
        """
        Exact determinant. Dispatches:
            n <= 4: cofactor expansion
            n >= 5: Gaussian elimination
        """
        if not self.is_square:
            raise LinAlgError("Determinant requires square matrix")
        if self.nrows <= 4:
            return self.det_cofactor()
        return self.det_gauss()

    def det_cofactor(self):
        """Exact determinant by cofactor expansion. O(n!)."""
        if not self.is_square:
            raise LinAlgError("Determinant requires square matrix")
        n = self.nrows
        if n == 1:
            return self[0, 0]
        if n == 2:
            return self[0, 0] * self[1, 1] - self[0, 1] * self[1, 0]
        total = VDR(0)
        for j in range(n):
            cofactor = self._minor(0, j).det_cofactor()
            term = self[0, j] * cofactor
            if j % 2 == 0:
                total = total + term
            else:
                total = total - term
        return total

    def det_gauss(self):
        """
        Exact determinant by Gaussian elimination. O(n^3).

        Uses exact VDR division for pivot operations.
        Tracks sign from row swaps.
        """
        if not self.is_square:
            raise LinAlgError("Determinant requires square matrix")
        n = self.nrows
        # work on a mutable copy
        rows = [[self[i, j] for j in range(n)] for i in range(n)]
        sign = VDR(1)

        for col in range(n):
            # find pivot
            pivot_row = None
            for row in range(col, n):
                if rows[row][col] != VDR(0):
                    pivot_row = row
                    break
            if pivot_row is None:
                return VDR(0)  # singular

            # swap if needed
            if pivot_row != col:
                rows[col], rows[pivot_row] = rows[pivot_row], rows[col]
                sign = -sign

            pivot_val = rows[col][col]

            # eliminate below
            for row in range(col + 1, n):
                if rows[row][col] != VDR(0):
                    factor = rows[row][col] / pivot_val
                    for k in range(col, n):
                        rows[row][k] = rows[row][k] - factor * rows[col][k]

        # determinant is product of diagonal times sign
        result = sign
        for i in range(n):
            result = result * rows[i][i]
        return result

    def _minor(self, row_skip, col_skip):
        """Matrix with row row_skip and column col_skip removed."""
        rows = []
        for i in range(self.nrows):
            if i == row_skip:
                continue
            row = []
            for j in range(self.ncols):
                if j == col_skip:
                    continue
                row.append(self[i, j])
            rows.append(row)
        return Mat(rows)

    # -- inverse -----------------------------------------------------------

    def inv(self):
        """
        Exact matrix inverse. Dispatches:
            n <= 4: adjugate method
            n >= 5: Gaussian elimination
        """
        if not self.is_square:
            raise LinAlgError("Inverse requires square matrix")
        if self.nrows <= 4:
            return self.inv_adjugate()
        return self.inv_gauss()

    def inv_adjugate(self):
        """Exact inverse via adjugate: A^-1 = adj(A) / det(A)."""
        if not self.is_square:
            raise LinAlgError("Inverse requires square matrix")
        d = self.det_cofactor()
        if d == VDR(0):
            raise LinAlgError("Matrix is singular (det = 0)")
        adj = self._adjugate()
        return adj.scale(VDR(1) / d)

    def _adjugate(self):
        """Transpose of cofactor matrix."""
        n = self.nrows
        cof = []
        for i in range(n):
            row = []
            for j in range(n):
                minor_det = self._minor(i, j).det_cofactor()
                if (i + j) % 2 == 1:
                    minor_det = -minor_det
                row.append(minor_det)
            cof.append(row)
        return Mat(cof).T

    def inv_gauss(self):
        """
        Exact inverse via Gaussian elimination with augmented matrix.
        [A | I] -> [I | A^-1]
        """
        if not self.is_square:
            raise LinAlgError("Inverse requires square matrix")
        n = self.nrows

        # build augmented matrix [A | I]
        aug = []
        for i in range(n):
            row = [self[i, j] for j in range(n)]
            for j in range(n):
                row.append(VDR(1) if i == j else VDR(0))
            aug.append(row)

        # forward elimination with partial pivoting
        for col in range(n):
            # find pivot
            pivot_row = None
            for row in range(col, n):
                if aug[row][col] != VDR(0):
                    pivot_row = row
                    break
            if pivot_row is None:
                raise LinAlgError("Matrix is singular (det = 0)")

            if pivot_row != col:
                aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

            pivot_val = aug[col][col]

            # scale pivot row
            for k in range(2 * n):
                aug[col][k] = aug[col][k] / pivot_val

            # eliminate all other rows
            for row in range(n):
                if row == col:
                    continue
                if aug[row][col] != VDR(0):
                    factor = aug[row][col]
                    for k in range(2 * n):
                        aug[row][k] = aug[row][k] - factor * aug[col][k]

        # extract right half
        result = []
        for i in range(n):
            row = [aug[i][j + n] for j in range(n)]
            result.append(row)
        return Mat(result)

    # -- solve -------------------------------------------------------------

    def solve(self, b):
        """
        Solve Ax = b. Dispatches:
            n <= 4: Cramer's rule
            n >= 5: Gaussian elimination
        """
        if not self.is_square:
            raise LinAlgError("Solve requires square matrix")
        if self.nrows != len(b):
            raise LinAlgError("Dimension mismatch")
        if self.nrows <= 4:
            return self.solve_cramer(b)
        return self.solve_gauss(b)

    def solve_cramer(self, b):
        """Solve Ax = b by Cramer's rule."""
        if not self.is_square:
            raise LinAlgError("Solve requires square matrix")
        if self.nrows != len(b):
            raise LinAlgError("Dimension mismatch")
        d = self.det()
        if d == VDR(0):
            raise LinAlgError("System is singular")
        n = self.nrows
        result = []
        for j in range(n):
            cols = []
            for jj in range(n):
                if jj == j:
                    cols.append([b[i] for i in range(n)])
                else:
                    cols.append([self[i, jj] for i in range(n)])
            replaced = Mat([
                [cols[jj][i] for jj in range(n)]
                for i in range(n)
            ])
            result.append(replaced.det() / d)
        return Vec(result)

    def solve_gauss(self, b):
        """
        Solve Ax = b by Gaussian elimination with back-substitution.
        """
        if not self.is_square:
            raise LinAlgError("Solve requires square matrix")
        n = self.nrows
        if n != len(b):
            raise LinAlgError("Dimension mismatch")

        # build augmented matrix [A | b]
        aug = []
        for i in range(n):
            row = [self[i, j] for j in range(n)]
            row.append(b[i])
            aug.append(row)

        # forward elimination
        for col in range(n):
            pivot_row = None
            for row in range(col, n):
                if aug[row][col] != VDR(0):
                    pivot_row = row
                    break
            if pivot_row is None:
                raise LinAlgError("System is singular")

            if pivot_row != col:
                aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

            pivot_val = aug[col][col]

            for row in range(col + 1, n):
                if aug[row][col] != VDR(0):
                    factor = aug[row][col] / pivot_val
                    for k in range(n + 1):
                        aug[row][k] = aug[row][k] - factor * aug[col][k]

        # back-substitution
        x = [VDR(0)] * n
        for i in range(n - 1, -1, -1):
            total = aug[i][n]
            for j in range(i + 1, n):
                total = total - aug[i][j] * x[j]
            x[i] = total / aug[i][i]

        return Vec(x)

    # -- rank --------------------------------------------------------------

    def rank(self):
        """Exact rank via Gaussian elimination."""
        rows = [[self[i, j] for j in range(self.ncols)]
                for i in range(self.nrows)]
        m, n = self.nrows, self.ncols
        pivot_row = 0
        for col in range(n):
            found = None
            for row in range(pivot_row, m):
                if rows[row][col] != VDR(0):
                    found = row
                    break
            if found is None:
                continue
            rows[pivot_row], rows[found] = rows[found], rows[pivot_row]
            pivot_val = rows[pivot_row][col]
            for row in range(pivot_row + 1, m):
                if rows[row][col] != VDR(0):
                    factor = rows[row][col] / pivot_val
                    for k in range(n):
                        rows[row][k] = rows[row][k] - factor * rows[pivot_row][k]
            pivot_row += 1
        return pivot_row

    # -- rref --------------------------------------------------------------

    def rref(self):
        """Reduced row echelon form via Gaussian elimination."""
        m, n = self.nrows, self.ncols
        rows = [[self[i, j] for j in range(n)] for i in range(m)]
        pivot_row = 0

        for col in range(n):
            # find pivot
            found = None
            for row in range(pivot_row, m):
                if rows[row][col] != VDR(0):
                    found = row
                    break
            if found is None:
                continue

            # swap
            rows[pivot_row], rows[found] = rows[found], rows[pivot_row]

            # scale pivot row
            pivot_val = rows[pivot_row][col]
            for k in range(n):
                rows[pivot_row][k] = rows[pivot_row][k] / pivot_val

            # eliminate all other rows
            for row in range(m):
                if row == pivot_row:
                    continue
                if rows[row][col] != VDR(0):
                    factor = rows[row][col]
                    for k in range(n):
                        rows[row][k] = rows[row][k] - factor * rows[pivot_row][k]

            pivot_row += 1

        return Mat(rows)

    # -- comparison --------------------------------------------------------

    def __eq__(self, other):
        if not isinstance(other, Mat):
            return NotImplemented
        if self.shape != other.shape:
            return False
        return all(a == b for a, b in zip(self._rows, other._rows))

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    # -- display -----------------------------------------------------------

    def __repr__(self):
        row_strs = []
        for r in self._rows:
            row_strs.append("[%s]" % ", ".join(str(x) for x in r))
        return "Mat([%s])" % ", ".join(row_strs)

    def pretty(self):
        """Human-readable matrix display."""
        lines = []
        strs = [[str(self[i, j]) for j in range(self.ncols)]
                for i in range(self.nrows)]
        widths = [max(len(strs[i][j]) for i in range(self.nrows))
                  for j in range(self.ncols)]
        for i in range(self.nrows):
            cells = [strs[i][j].rjust(widths[j]) for j in range(self.ncols)]
            lines.append("| %s |" % "  ".join(cells))
        return "\n".join(lines)

    def to_fractions(self):
        """Export as list of lists of Fraction."""
        return [[self[i, j].to_fraction()
                 for j in range(self.ncols)]
                for i in range(self.nrows)]


# ---------------------------------------------------------------------------
# VDR Bracket Notation Parser
# ---------------------------------------------------------------------------

def parse_vdr(text):
    """
    Parse bracket notation into a VDR object.

        parse_vdr("[1, 2, 0]")       -> VDR(1, 2)
        parse_vdr("[1, 3, [1, 6, 0]]") -> VDR(1, 3, Remainder(0, [VDR(1, 6)]))
    """
    text = text.strip()
    result, pos = _parse_vdr(text, 0)
    if pos != len(text):
        raise InvalidStructureError(
            "Unexpected trailing content at position %d" % pos
        )
    return result


def _parse_vdr(text, pos):
    pos = _skip_ws(text, pos)
    if pos >= len(text) or text[pos] != '[':
        raise InvalidStructureError("Expected '[' at position %d" % pos)
    pos += 1

    pos = _skip_ws(text, pos)
    v, pos = _parse_int(text, pos)

    pos = _skip_ws(text, pos)
    if pos >= len(text) or text[pos] != ',':
        raise InvalidStructureError("Expected ',' after V at position %d" % pos)
    pos += 1

    pos = _skip_ws(text, pos)
    d, pos = _parse_int(text, pos)

    pos = _skip_ws(text, pos)
    if pos >= len(text) or text[pos] != ',':
        raise InvalidStructureError("Expected ',' after D at position %d" % pos)
    pos += 1

    pos = _skip_ws(text, pos)
    r, pos = _parse_remainder(text, pos)

    pos = _skip_ws(text, pos)
    if pos >= len(text) or text[pos] != ']':
        raise InvalidStructureError("Expected ']' at position %d" % pos)
    pos += 1

    return VDR(v, d, r), pos


def _parse_remainder(text, pos):
    pos = _skip_ws(text, pos)

    if pos < len(text) and text[pos] == '[':
        child, pos = _parse_vdr(text, pos)
        children = [child]
        while pos < len(text):
            pos = _skip_ws(text, pos)
            if pos < len(text) and text[pos] == '+':
                pos += 1
                pos = _skip_ws(text, pos)
                if pos < len(text) and text[pos] == '[':
                    child, pos = _parse_vdr(text, pos)
                    children.append(child)
                else:
                    n, pos = _parse_int(text, pos)
                    return Remainder(n, children), pos
            else:
                break
        return Remainder(0, children), pos

    base, pos = _parse_int(text, pos)

    children = []
    while pos < len(text):
        pos = _skip_ws(text, pos)
        if pos < len(text) and text[pos] == '+':
            saved = pos
            pos += 1
            pos = _skip_ws(text, pos)
            if pos < len(text) and text[pos] == '[':
                child, pos = _parse_vdr(text, pos)
                children.append(child)
            else:
                pos = saved
                break
        else:
            break

    return Remainder(base, children), pos


def _parse_int(text, pos):
    pos = _skip_ws(text, pos)
    start = pos
    if pos < len(text) and text[pos] in '+-':
        pos += 1
    if pos >= len(text) or not text[pos].isdigit():
        raise InvalidStructureError(
            "Expected integer at position %d" % start
        )
    while pos < len(text) and text[pos].isdigit():
        pos += 1
    return int(text[start:pos]), pos


def _skip_ws(text, pos):
    while pos < len(text) and text[pos] in ' \t\n\r':
        pos += 1
    return pos


# ---------------------------------------------------------------------------
# JSON Serialization
# ---------------------------------------------------------------------------

def vdr_to_dict(x):
    """Serialize VDR to a JSON-compatible dict."""
    return {
        "v": x.v,
        "d": x.d,
        "r": _remainder_to_dict(x.r),
    }


def vdr_from_dict(d):
    """Deserialize VDR from a JSON-compatible dict."""
    return VDR(d["v"], d["d"], _remainder_from_dict(d["r"]))


def _remainder_to_dict(r):
    return {
        "base": r.base,
        "children": [vdr_to_dict(c) for c in r.children],
    }


def _remainder_from_dict(d):
    children = [vdr_from_dict(c) for c in d.get("children", [])]
    return Remainder(d["base"], children)


# ---------------------------------------------------------------------------
# LaTeX Export
# ---------------------------------------------------------------------------

def vdr_to_latex(x):
    """
    Export VDR to LaTeX notation.

        [1, 2, 0]          -> \\frac{1}{2}
        [1, 3, [1, 6, 0]]  -> \\frac{1}{3}\\left\\{\\frac{1}{6}\\right\\}
    """
    if x.is_closed:
        if x.d == 1:
            return str(x.v)
        return "\\frac{%d}{%d}" % (x.v, x.d)
    base = "\\frac{%d}{%d}" % (x.v, x.d) if x.d != 1 else str(x.v)
    rem = _remainder_to_latex(x.r)
    return "%s\\left\\{%s\\right\\}" % (base, rem)


def _remainder_to_latex(r):
    parts = []
    if r.base != 0 or not r.children:
        parts.append(str(r.base))
    for c in r.children:
        parts.append(vdr_to_latex(c))
    return " + ".join(parts)

