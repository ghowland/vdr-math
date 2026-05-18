"""Gym 19 — Algebraic topology. 16/16 in VDR-3."""

import pytest
from vdr.core import VDR
from vdr.linalg import Mat
from vdr.math.topology import (
    boundary_matrix, verify_d_squared_zero, betti_numbers,
    euler_characteristic, simplicial_complex_boundaries,
)


class TestBoundary:
    def test_d_squared_zero(self):
        """d∘d = 0 for triangle complex."""
        simplices = {
            0: [(0,), (1,), (2,)],
            1: [(0, 1), (0, 2), (1, 2)],
            2: [(0, 1, 2)],
        }
        boundaries = simplicial_complex_boundaries(simplices)
        # d1 (edges -> vertices) and d2 (face -> edges)
        d1 = boundaries[0]
        d2 = boundaries[1]
        assert verify_d_squared_zero(d2, d1)


class TestBetti:
    def test_filled_triangle(self):
        simplices = {
            0: [(0,), (1,), (2,)],
            1: [(0, 1), (0, 2), (1, 2)],
            2: [(0, 1, 2)],
        }
        boundaries = simplicial_complex_boundaries(simplices)
        betti = betti_numbers(boundaries)
        assert betti[0] == 1  # connected
        assert betti[1] == 0  # no holes

    def test_hollow_triangle(self):
        simplices = {
            0: [(0,), (1,), (2,)],
            1: [(0, 1), (0, 2), (1, 2)],
        }
        boundaries = simplicial_complex_boundaries(simplices)
        betti = betti_numbers(boundaries)
        assert betti[0] == 1
        assert betti[1] == 1  # one 1-cycle (the triangle boundary)

    def test_disconnected(self):
        simplices = {
            0: [(0,), (1,), (2,), (3,)],
            1: [(0, 1), (2, 3)],
        }
        boundaries = simplicial_complex_boundaries(simplices)
        betti = betti_numbers(boundaries)
        assert betti[0] == 2  # two components


class TestEuler:
    def test_sphere(self):
        """Tetrahedron surface: chi = 2 (sphere)."""
        assert euler_characteristic([1, 0, 1]) == 2

    def test_torus(self):
        """Torus: chi = 0."""
        assert euler_characteristic([1, 2, 1]) == 0
