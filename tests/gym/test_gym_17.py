"""Gym 17 — Game theory. 24/24 in VDR-3."""

import pytest
from vdr.core import VDR
from vdr.linalg import Vec, Mat
from vdr.math.game_theory import (
    minimax_2x2, nash_2x2, shapley_values, cournot_duopoly,
)


class TestMinimax:
    def test_basic(self):
        payoff = Mat.from_ints([[3, -1], [-2, 4]])
        p, q, val = minimax_2x2(payoff)
        # verify: p*=3/5, q*=1/2, val=1
        assert p == VDR(3, 5)
        assert q == VDR(1, 2)
        assert val == VDR(1)


class TestNash:
    def test_bos(self):
        """Battle of the Sexes."""
        A = Mat.from_ints([[3, 0], [0, 2]])
        B = Mat.from_ints([[2, 0], [0, 3]])
        p, q, ea, eb = nash_2x2(A, B)
        assert p == VDR(3, 5)
        assert q == VDR(2, 5)
        assert ea == VDR(6, 5)


class TestShapley:
    def test_sum_to_grand(self):
        """Shapley values must sum to v(grand coalition)."""
        def v(s):
            if len(s) == 3:
                return VDR(1)
            if len(s) == 2:
                return VDR(1, 2)
            if len(s) == 1:
                return VDR(0)
            return VDR(0)

        phi = shapley_values(v, 3)
        total = phi[0] + phi[1] + phi[2]
        assert total == VDR(1)

    def test_symmetric(self):
        """Symmetric game: all players get equal share."""
        def v(s):
            return VDR(len(s))

        phi = shapley_values(v, 3)
        assert phi[0] == phi[1]
        assert phi[1] == phi[2]


class TestCournot:
    def test_symmetric(self):
        q1, q2, pi1, pi2 = cournot_duopoly(VDR(100), VDR(1), VDR(10), VDR(10))
        assert q1 == q2  # symmetric
        assert q1 == VDR(30)  # (100 - 20 + 10) / 3 = 30

    def test_asymmetric(self):
        q1, q2, pi1, pi2 = cournot_duopoly(VDR(100), VDR(1), VDR(10), VDR(20))
        # q1 = (100 - 2*10 + 20) / (3*1) = 100/3
        # q2 = (100 - 2*20 + 10) / (3*1) = 70/3
        assert q1 == VDR(100, 3)
        assert q2 == VDR(70, 3)
