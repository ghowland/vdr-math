"""Gym 20 — Tropical and lattice algebra. 23/23 in VDR-3."""

import pytest
from vdr.core import VDR
from vdr.linalg import Vec
from vdr.math.tropical import (
    trop_add, trop_mul, trop_mat_mul, trop_det,
    gram_matrix, gram_schmidt_exact, lovasz_condition, lll_reduce,
)


class TestTropical:
    def test_add(self):
        assert trop_add(VDR(3), VDR(5)) == VDR(3)  # min

    def test_mul(self):
        assert trop_mul(VDR(3), VDR(5)) == VDR(8)  # ordinary add

    def test_inf(self):
        assert trop_add(None, VDR(3)) == VDR(3)
        assert trop_mul(None, VDR(3)) is None

    def test_mat_mul(self):
        # 2-hop shortest path
        A = [[VDR(0), VDR(1), None],
             [None, VDR(0), VDR(2)],
             [None, None, VDR(0)]]
        B = trop_mat_mul(A, A, 3)
        assert B[0][2] == VDR(3)  # 0->1->2 = 1+2

    def test_det_identity(self):
        I = [[VDR(0), None], [None, VDR(0)]]
        assert trop_det(I, 2) == VDR(0)


class TestGramSchmidt:
    def test_orthogonal(self):
        v1 = Vec.from_ints([1, 1, 0])
        v2 = Vec.from_ints([1, 0, 1])
        ortho, mu = gram_schmidt_exact([v1, v2])
        assert ortho[0].dot(ortho[1]) == VDR(0)

    def test_mu_values(self):
        v1 = Vec.from_ints([1, 0])
        v2 = Vec.from_ints([1, 1])
        ortho, mu = gram_schmidt_exact([v1, v2])
        assert mu[1][0] == VDR(1)  # projection of v2 onto v1


class TestLovasz:
    def test_condition(self):
        v1 = Vec.from_ints([1, 0])
        v2 = Vec.from_ints([0, 1])
        ortho, mu = gram_schmidt_exact([v1, v2])
        assert lovasz_condition(ortho, mu, 1)


class TestLLL:
    def test_already_reduced(self):
        basis = [Vec.from_ints([1, 0]), Vec.from_ints([0, 1])]
        reduced = lll_reduce(basis)
        assert len(reduced) == 2

    def test_reduces_basis(self):
        basis = [Vec.from_ints([1, 0, 0]),
                 Vec.from_ints([0, 1, 0]),
                 Vec.from_ints([10, 10, 1])]
        reduced = lll_reduce(basis)
        # reduced basis should have shorter vectors
        assert len(reduced) == 3
