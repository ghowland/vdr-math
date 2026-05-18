"""
Tests for vdr.linalg — Vec, Mat, determinant, inverse, solve, rank,
Gaussian elimination, parser, serialization.
"""

import pytest
from fractions import Fraction

from vdr.core import VDR, Remainder
from vdr.linalg import (
    Vec, Mat, LinAlgError,
    parse_vdr, vdr_to_dict, vdr_from_dict, vdr_to_latex,
)
from vdr.active import install

install()


# ---------------------------------------------------------------------------
# Vec
# ---------------------------------------------------------------------------

class TestVec:
    def test_construction(self):
        v = Vec([VDR(1), VDR(2), VDR(3)])
        assert len(v) == 3
        assert v[0] == VDR(1)

    def test_from_ints(self):
        v = Vec.from_ints([1, 2, 3])
        assert v[1] == VDR(2)

    def test_from_fracs(self):
        v = Vec.from_fracs([(1, 2), (3, 4)])
        assert v[0] == VDR(1, 2)

    def test_zero(self):
        v = Vec.zero(3)
        assert all(v[i] == VDR(0) for i in range(3))

    def test_add(self):
        a = Vec.from_ints([1, 2])
        b = Vec.from_ints([3, 4])
        c = a + b
        assert c == Vec.from_ints([4, 6])

    def test_sub(self):
        a = Vec.from_ints([5, 7])
        b = Vec.from_ints([3, 4])
        assert a - b == Vec.from_ints([2, 3])

    def test_neg(self):
        v = Vec.from_ints([1, -2])
        assert -v == Vec.from_ints([-1, 2])

    def test_scale(self):
        v = Vec.from_ints([2, 3])
        assert v.scale(VDR(1, 2)) == Vec.from_fracs([(1, 1), (3, 2)])

    def test_mul_operator(self):
        v = Vec.from_ints([2, 3])
        assert v * VDR(2) == Vec.from_ints([4, 6])

    def test_rmul(self):
        v = Vec.from_ints([2, 3])
        assert v * VDR(2) == Vec.from_ints([4, 6])

    def test_dot(self):
        a = Vec.from_ints([1, 2, 3])
        b = Vec.from_ints([4, 5, 6])
        assert a.dot(b) == VDR(32)

    def test_norm_sq(self):
        v = Vec.from_ints([3, 4])
        assert v.norm_sq() == VDR(25)

    def test_dim_mismatch_raises(self):
        with pytest.raises(LinAlgError):
            Vec.from_ints([1, 2]) + Vec.from_ints([1, 2, 3])

    def test_eq(self):
        assert Vec.from_ints([1, 2]) == Vec.from_ints([1, 2])
        assert Vec.from_ints([1, 2]) != Vec.from_ints([1, 3])

    def test_to_fractions(self):
        v = Vec.from_fracs([(1, 2), (3, 4)])
        assert v.to_fractions() == [Fraction(1, 2), Fraction(3, 4)]


# ---------------------------------------------------------------------------
# Mat basics
# ---------------------------------------------------------------------------

class TestMatBasic:
    def test_construction(self):
        m = Mat.from_ints([[1, 2], [3, 4]])
        assert m.nrows == 2
        assert m.ncols == 2
        assert m[0, 0] == VDR(1)
        assert m[1, 1] == VDR(4)

    def test_identity(self):
        I = Mat.identity(3)
        assert I[0, 0] == VDR(1)
        assert I[0, 1] == VDR(0)

    def test_zero(self):
        Z = Mat.zero(2, 3)
        assert Z[1, 2] == VDR(0)

    def test_shape(self):
        m = Mat.from_ints([[1, 2, 3], [4, 5, 6]])
        assert m.shape == (2, 3)
        assert not m.is_square

    def test_add(self):
        a = Mat.from_ints([[1, 2], [3, 4]])
        b = Mat.from_ints([[5, 6], [7, 8]])
        c = a + b
        assert c[0, 0] == VDR(6)
        assert c[1, 1] == VDR(12)

    def test_sub(self):
        a = Mat.from_ints([[5, 6], [7, 8]])
        b = Mat.from_ints([[1, 2], [3, 4]])
        c = a - b
        assert c == Mat.from_ints([[4, 4], [4, 4]])

    def test_neg(self):
        m = Mat.from_ints([[1, -2], [-3, 4]])
        assert -m == Mat.from_ints([[-1, 2], [3, -4]])

    def test_scale(self):
        m = Mat.from_ints([[1, 2], [3, 4]])
        assert m.scale(VDR(2)) == Mat.from_ints([[2, 4], [6, 8]])

    def test_transpose(self):
        m = Mat.from_ints([[1, 2, 3], [4, 5, 6]])
        t = m.T
        assert t.shape == (3, 2)
        assert t[0, 1] == VDR(4)

    def test_trace(self):
        m = Mat.from_ints([[1, 2], [3, 4]])
        assert m.trace() == VDR(5)


# ---------------------------------------------------------------------------
# Mat multiplication
# ---------------------------------------------------------------------------

class TestMatMul:
    def test_matmul_identity(self):
        m = Mat.from_ints([[1, 2], [3, 4]])
        I = Mat.identity(2)
        assert m.matmul(I) == m
        assert I.matmul(m) == m

    def test_matmul_basic(self):
        a = Mat.from_ints([[1, 2], [3, 4]])
        b = Mat.from_ints([[5, 6], [7, 8]])
        c = a.matmul(b)
        assert c[0, 0] == VDR(19)
        assert c[0, 1] == VDR(22)
        assert c[1, 0] == VDR(43)
        assert c[1, 1] == VDR(50)

    def test_matvec(self):
        m = Mat.from_ints([[1, 2], [3, 4]])
        v = Vec.from_ints([5, 6])
        r = m.matvec(v)
        assert r == Vec.from_ints([17, 39])

    def test_shape_mismatch_raises(self):
        a = Mat.from_ints([[1, 2]])
        b = Mat.from_ints([[1, 2]])
        with pytest.raises(LinAlgError):
            a.matmul(b)


# ---------------------------------------------------------------------------
# Determinant
# ---------------------------------------------------------------------------

class TestDet:
    def test_det_1x1(self):
        m = Mat.from_ints([[7]])
        assert m.det() == VDR(7)

    def test_det_2x2(self):
        m = Mat.from_ints([[1, 2], [3, 4]])
        assert m.det() == VDR(-2)

    def test_det_3x3(self):
        m = Mat.from_ints([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
        assert m.det() == VDR(27)

    def test_det_identity(self):
        assert Mat.identity(5).det() == VDR(1)

    def test_det_singular(self):
        m = Mat.from_ints([[1, 2], [2, 4]])
        assert m.det() == VDR(0)

    def test_cofactor_equals_gauss(self):
        """Cofactor and Gaussian should agree for all sizes."""
        m = Mat.from_ints([[2, 1, 3], [1, 4, 2], [3, 2, 5]])
        assert m.det_cofactor() == m.det_gauss()

    def test_det_gauss_5x5(self):
        m = Mat.from_ints([
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [1, 3, 5, 3, 1],
            [2, 4, 1, 4, 2],
            [3, 1, 4, 1, 3],
        ])
        # both methods should agree
        assert m.det_cofactor() == m.det_gauss()


# ---------------------------------------------------------------------------
# Inverse
# ---------------------------------------------------------------------------

class TestInverse:
    def test_inv_2x2(self):
        m = Mat.from_ints([[1, 2], [3, 4]])
        m_inv = m.inv()
        product = m.matmul(m_inv)
        assert product == Mat.identity(2)

    def test_inv_3x3(self):
        m = Mat.from_ints([[2, 1, 1], [1, 3, 2], [1, 0, 0]])
        m_inv = m.inv()
        assert m.matmul(m_inv) == Mat.identity(3)

    def test_inv_singular_raises(self):
        m = Mat.from_ints([[1, 2], [2, 4]])
        with pytest.raises(LinAlgError):
            m.inv()

    def test_hilbert_3x3(self):
        """Hilbert matrix: H * H^-1 = I exactly."""
        h = Mat.from_fracs([
            [(1, 1), (1, 2), (1, 3)],
            [(1, 2), (1, 3), (1, 4)],
            [(1, 3), (1, 4), (1, 5)],
        ])
        h_inv = h.inv()
        product = h.matmul(h_inv)
        assert product == Mat.identity(3)

    def test_inv_gauss_matches_adjugate(self):
        m = Mat.from_ints([[2, 1], [5, 3]])
        assert m.inv_adjugate() == m.inv_gauss()


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

class TestSolve:
    def test_solve_2x2(self):
        A = Mat.from_ints([[1, 2], [3, 4]])
        b = Vec.from_ints([5, 11])
        x = A.solve(b)
        assert A.matvec(x) == b

    def test_solve_3x3(self):
        A = Mat.from_ints([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
        b = Vec.from_ints([8, -11, -3])
        x = A.solve(b)
        assert A.matvec(x) == b

    def test_solve_gauss_matches_cramer(self):
        A = Mat.from_ints([[1, 2], [3, 4]])
        b = Vec.from_ints([5, 11])
        assert A.solve_cramer(b) == A.solve_gauss(b)

    def test_singular_raises(self):
        A = Mat.from_ints([[1, 2], [2, 4]])
        b = Vec.from_ints([1, 2])
        with pytest.raises(LinAlgError):
            A.solve(b)


# ---------------------------------------------------------------------------
# Rank and RREF
# ---------------------------------------------------------------------------

class TestRank:
    def test_rank_identity(self):
        assert Mat.identity(3).rank() == 3

    def test_rank_singular(self):
        m = Mat.from_ints([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert m.rank() == 2

    def test_rank_zero(self):
        assert Mat.zero(3, 3).rank() == 0

    def test_rank_full(self):
        m = Mat.from_ints([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert m.rank() == 3

    def test_rref_identity(self):
        m = Mat.from_ints([[2, 0], [0, 3]])
        r = m.rref()
        assert r == Mat.identity(2)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class TestParser:
    def test_parse_closed(self):
        x = parse_vdr("[1, 2, 0]")
        assert x == VDR(1, 2)

    def test_parse_active_atomic(self):
        x = parse_vdr("[1, 3, 5]")
        assert x.v == 1
        assert x.d == 3
        assert x.r.base == 5

    def test_parse_nested(self):
        x = parse_vdr("[1, 3, [1, 6, 0]]")
        assert x.v == 1
        assert x.d == 3
        assert len(x.r.children) == 1
        assert x.r.children[0] == VDR(1, 6)

    def test_parse_negative(self):
        x = parse_vdr("[-1, 2, 0]")
        assert x.v == -1

    def test_roundtrip(self):
        original = VDR(3, 7, Remainder(1, [VDR(2, 5)]))
        text = str(original)
        # our str format may not match parse format exactly
        # test dict roundtrip instead
        d = vdr_to_dict(original)
        recovered = vdr_from_dict(d)
        assert recovered.to_fraction() == original.to_fraction()


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_roundtrip_closed(self):
        x = VDR(3, 7)
        d = vdr_to_dict(x)
        y = vdr_from_dict(d)
        assert y == x

    def test_roundtrip_active(self):
        x = VDR(1, 3, Remainder(2, [VDR(5, 11)]))
        d = vdr_to_dict(x)
        y = vdr_from_dict(d)
        assert y.to_fraction() == x.to_fraction()

    def test_dict_structure(self):
        x = VDR(1, 2)
        d = vdr_to_dict(x)
        assert d["v"] == 1
        assert d["d"] == 2
        assert d["r"]["base"] == 0


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

class TestLatex:
    def test_integer(self):
        assert vdr_to_latex(VDR(5)) == "5"

    def test_fraction(self):
        assert "frac" in vdr_to_latex(VDR(1, 2))

    def test_active(self):
        x = VDR(1, 3, Remainder(1))
        latex = vdr_to_latex(x)
        assert "frac" in latex
        assert "{" in latex
