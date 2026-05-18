"""Gym 04 — Matrix decomposition. 13/13 in VDR-2."""

import pytest
from vdr.core import VDR
from vdr.linalg import Vec, Mat
from vdr.math.control import mat_pow


class TestLU:
    def test_lu_roundtrip(self):
        """For a well-conditioned matrix, solve via Gaussian and verify."""
        A = Mat.from_ints([[2, 1], [5, 3]])
        b = Vec.from_ints([4, 7])
        x = A.solve(b)
        assert A.matvec(x) == b

    def test_lu_3x3(self):
        A = Mat.from_ints([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        b = Vec.from_ints([6, 15, 25])
        x = A.solve(b)
        assert A.matvec(x) == b


class TestMatrixPower:
    def test_fibonacci(self):
        """Fibonacci via [[1,1],[1,0]]^n."""
        F = Mat.from_ints([[1, 1], [1, 0]])
        F10 = mat_pow(F, 10)
        assert F10[0, 0] == VDR(89)   # F(11)
        assert F10[0, 1] == VDR(55)   # F(10)
        assert F10[1, 0] == VDR(55)   # F(10)
        assert F10[1, 1] == VDR(34)   # F(9)

    def test_identity_power(self):
        I = Mat.identity(3)
        assert mat_pow(I, 100) == I


class TestGramSchmidt:
    def test_orthogonal_2d(self):
        from vdr.math.tropical import gram_schmidt_exact
        v1 = Vec.from_ints([1, 1])
        v2 = Vec.from_ints([1, 0])
        ortho, mu = gram_schmidt_exact([v1, v2])
        # b*_0 . b*_1 should be exactly 0
        assert ortho[0].dot(ortho[1]) == VDR(0)

    def test_orthogonal_3d(self):
        from vdr.math.tropical import gram_schmidt_exact
        v1 = Vec.from_ints([1, 1, 0])
        v2 = Vec.from_ints([1, 0, 1])
        v3 = Vec.from_ints([0, 1, 1])
        ortho, mu = gram_schmidt_exact([v1, v2, v3])
        for i in range(3):
            for j in range(i + 1, 3):
                assert ortho[i].dot(ortho[j]) == VDR(0)
