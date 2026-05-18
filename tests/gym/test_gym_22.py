"""Gym 22 — Wavelets. 18/18 in VDR-3."""

import pytest
from vdr.core import VDR
from vdr.math.wavelets import (
    haar_forward, haar_inverse, haar_multilevel,
    haar_reconstruct_multilevel, energy, parseval_verify, threshold_hard,
)


class TestHaarRoundtrip:
    def test_basic(self):
        signal = [VDR(1), VDR(3), VDR(5), VDR(7)]
        avgs, dets = haar_forward(signal)
        recovered = haar_inverse(avgs, dets)
        assert recovered == signal

    def test_rational(self):
        signal = [VDR(1, 3), VDR(1, 7), VDR(2, 5), VDR(3, 11)]
        avgs, dets = haar_forward(signal)
        recovered = haar_inverse(avgs, dets)
        assert recovered == signal


class TestHaarValues:
    def test_averages(self):
        signal = [VDR(1), VDR(3), VDR(5), VDR(7)]
        avgs, dets = haar_forward(signal)
        assert avgs == [VDR(2), VDR(6)]
        assert dets == [VDR(-1), VDR(-1)]


class TestMultilevel:
    def test_roundtrip(self):
        signal = [VDR(1), VDR(3), VDR(5), VDR(7)]
        decomp = haar_multilevel(signal, 2)
        recovered = haar_reconstruct_multilevel(decomp)
        assert recovered == signal

    def test_64_sample(self):
        signal = [VDR(i) for i in range(64)]
        decomp = haar_multilevel(signal, 6)
        recovered = haar_reconstruct_multilevel(decomp)
        assert recovered == signal


class TestEnergy:
    def test_basic(self):
        signal = [VDR(1), VDR(3), VDR(5), VDR(7)]
        assert energy(signal) == VDR(84)

    def test_parseval(self):
        signal = [VDR(1), VDR(2), VDR(3), VDR(4)]
        decomp = haar_multilevel(signal, 2)
        assert parseval_verify(signal, decomp)


class TestThreshold:
    def test_basic(self):
        dets = [VDR(1, 10), VDR(3), VDR(-1, 5)]
        result = threshold_hard(dets, VDR(1, 2))
        assert result[0] == VDR(0)
        assert result[1] == VDR(3)
        assert result[2] == VDR(0)
