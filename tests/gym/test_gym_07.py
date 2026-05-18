"""Gym 07 — Signal processing. 11/11 in VDR-2."""

import pytest
from vdr.core import VDR
from vdr.signal.convolution import convolve, correlate
from vdr.signal.filters import iir_filter, moving_average, z_transform


class TestConvolution:
    def test_basic(self):
        a = [VDR(1), VDR(2), VDR(3)]
        b = [VDR(1), VDR(1)]
        c = convolve(a, b)
        assert c == [VDR(1), VDR(3), VDR(5), VDR(3)]

    def test_identity(self):
        a = [VDR(5), VDR(3)]
        b = [VDR(1)]
        assert convolve(a, b) == a

    def test_commutative(self):
        a = [VDR(1), VDR(2)]
        b = [VDR(3), VDR(4)]
        assert convolve(a, b) == convolve(b, a)


class TestIIR:
    def test_impulse(self):
        """IIR with a=1/2: y[n] = (1/2)^n for impulse input."""
        x = [VDR(1)] + [VDR(0)] * 5
        y = iir_filter(x, VDR(1, 2))
        assert y[0] == VDR(1)
        assert y[1] == VDR(1, 2)
        assert y[2] == VDR(1, 4)
        assert y[3] == VDR(1, 8)

    def test_step(self):
        """Step input: y converges to 1/(1-a)."""
        x = [VDR(1)] * 10
        y = iir_filter(x, VDR(1, 2))
        # y[n] approaches 2


class TestMovingAverage:
    def test_basic(self):
        signal = [VDR(1), VDR(2), VDR(3), VDR(4)]
        result = moving_average(signal, 2)
        assert result[1] == VDR(3, 2)  # (1+2)/2


class TestZTransform:
    def test_basic(self):
        """H(z) = 1 + z^{-1} at z=2 -> 1 + 1/2 = 3/2."""
        h = [VDR(1), VDR(1)]
        assert z_transform(h, VDR(2)) == VDR(3, 2)
