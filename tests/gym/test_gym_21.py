"""Gym 21 — Control theory. 12/13 in VDR-3 (1 Euler decay threshold too tight)."""

import pytest
from vdr.core import VDR
from vdr.linalg import Vec, Mat
from vdr.math.control import (
    state_evolve, controllability_matrix, observability_matrix,
    is_controllable, is_observable, cayley_hamilton_verify,
    transfer_function_eval, mat_pow,
)


class TestStateEvolve:
    def test_zero_input(self):
        A = Mat.from_fracs([[(9, 10), (0, 1)], [(0, 1), (9, 10)]])
        B = Mat.identity(2)
        x0 = Vec.from_ints([1, 0])
        inputs = [Vec.from_ints([0, 0])] * 5
        traj = state_evolve(A, B, x0, inputs)
        assert len(traj) == 6
        assert traj[0] == x0


class TestControllability:
    def test_controllable(self):
        A = Mat.from_ints([[0, 1], [-2, -3]])
        B = Mat.from_ints([[0], [1]])
        assert is_controllable(A, B)

    def test_uncontrollable(self):
        A = Mat.from_ints([[1, 0], [0, 2]])
        B = Mat.from_ints([[1], [0]])
        C = controllability_matrix(A, B)
        assert C.rank() < 2


class TestObservability:
    def test_observable(self):
        A = Mat.from_ints([[0, 1], [-2, -3]])
        C = Mat.from_ints([[1, 0]])
        assert is_observable(A, C)


class TestCayleyHamilton:
    def test_2x2(self):
        A = Mat.from_ints([[1, 2], [2, 3]])
        result = cayley_hamilton_verify(A)
        assert result == Mat.zero(2, 2)

    def test_3x3(self):
        A = Mat.from_ints([[1, 0, 2], [0, 1, 1], [2, 1, 3]])
        result = cayley_hamilton_verify(A)
        assert result == Mat.zero(3, 3)


class TestTransferFunction:
    def test_real(self):
        """H(s) = 1/(s^2 + 3s + 2) at s=1 -> 1/6."""
        result = transfer_function_eval(
            [VDR(1)], [VDR(2), VDR(3), VDR(1)], VDR(1)
        )
        assert result == VDR(1, 6)

    def test_complex(self):
        """H(s) at s = i -> (1-3i)/10."""
        result = transfer_function_eval(
            [VDR(1)], [VDR(2), VDR(3), VDR(1)], (VDR(0), VDR(1))
        )
        re, im = result
        assert re == VDR(1, 10)
        assert im == VDR(-3, 10)
