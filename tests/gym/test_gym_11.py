"""Gym 11 — Probability. 13/13 in VDR-2."""

import pytest
from vdr.core import VDR
from vdr.linalg import Vec, Mat
from vdr.math.probability import (
    binom_pmf, binom_pmf_full, bayes_update, bayes_sequential,
    markov_steady_state, gamblers_ruin, expected_value, variance,
)


class TestBinomPMF:
    def test_sum_to_one(self):
        pmf = binom_pmf_full(10, VDR(1, 3))
        total = VDR(0)
        for p in pmf:
            total = total + p
        assert total == VDR(1)

    def test_specific(self):
        p = binom_pmf(10, 0, VDR(1, 3))
        # (2/3)^10
        expected = VDR(1024, 59049)
        assert p == expected


class TestBayes:
    def test_basic(self):
        post = bayes_update(VDR(1, 2), VDR(3))
        assert post == VDR(3, 4)

    def test_sequential(self):
        posts = bayes_sequential(VDR(1, 2), [VDR(3), VDR(2)])
        assert posts[0] == VDR(3, 4)
        # second update
        expected = bayes_update(VDR(3, 4), VDR(2))
        assert posts[1] == expected

    def test_posterior_exact(self):
        """Known result: prior 1/3, LR=2 -> posterior 2/4 = 1/2."""
        post = bayes_update(VDR(1, 3), VDR(2))
        assert post == VDR(1, 2)


class TestMarkov:
    def test_steady_state(self):
        P = Mat.from_fracs([
            [(1, 2), (1, 2)],
            [(1, 4), (3, 4)],
        ])
        ss = markov_steady_state(P)
        assert ss[0] + ss[1] == VDR(1)
        # steady state: [1/3, 2/3]
        assert ss[0] == VDR(1, 3)
        assert ss[1] == VDR(2, 3)


class TestGamblersRuin:
    def test_basic(self):
        assert gamblers_ruin(3, 10) == VDR(7, 10)

    def test_certain_ruin(self):
        assert gamblers_ruin(0, 10) == VDR(1)


class TestExpectedValue:
    def test_fair_die(self):
        values = [VDR(i) for i in range(1, 7)]
        probs = [VDR(1, 6)] * 6
        ev = expected_value(values, probs)
        assert ev == VDR(7, 2)

    def test_variance(self):
        values = [VDR(0), VDR(1)]
        probs = [VDR(1, 2), VDR(1, 2)]
        v = variance(values, probs)
        assert v == VDR(1, 4)
