"""Gym 18 — Coding theory. 27/27 in VDR-3."""

import pytest
from vdr.math.coding_theory import (
    gf_add, gf_mul, gf_inv, gf_poly_eval,
    hamming74_encode, hamming74_syndrome, hamming74_correct,
    hamming_distance, hamming_weight, min_distance,
)


class TestGaloisField:
    def test_gf7_inverses(self):
        """Every nonzero element in GF(7) has an inverse."""
        for a in range(1, 7):
            inv = gf_inv(a, 7)
            assert gf_mul(a, inv, 7) == 1

    def test_gf11_poly(self):
        coeffs = [1, 2, 3]
        result = gf_poly_eval(coeffs, 4, 11)
        # 1 + 2*4 + 3*16 = 1 + 8 + 48 = 57 = 57 mod 11 = 2
        assert result == 2


class TestHamming74:
    def test_all_codewords_syndrome_zero(self):
        """All 16 valid codewords have syndrome 0."""
        for bits in range(16):
            data = [(bits >> i) & 1 for i in range(4)]
            cw = hamming74_encode(data)
            assert hamming74_syndrome(cw) == 0

    def test_single_error_correction(self):
        """All 7 single-bit errors corrected."""
        data = [1, 0, 1, 1]
        cw = hamming74_encode(data)
        for pos in range(7):
            received = list(cw)
            received[pos] ^= 1
            corrected = hamming74_correct(received)
            assert corrected == cw

    def test_min_distance(self):
        """Hamming(7,4) has minimum distance 3."""
        codewords = []
        for bits in range(16):
            data = [(bits >> i) & 1 for i in range(4)]
            codewords.append(hamming74_encode(data))
        md = min_distance(codewords)
        assert md == 3


class TestHammingMetrics:
    def test_distance(self):
        assert hamming_distance([0, 1, 1, 0], [1, 1, 0, 0]) == 2

    def test_weight(self):
        assert hamming_weight([1, 0, 1, 1, 0]) == 3
