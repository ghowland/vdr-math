"""
vdr.math.coding_theory — Exact finite field and error correction.

    from vdr.math.coding_theory import hamming74_encode, hamming74_correct, gf_inv

    codeword = hamming74_encode([1, 0, 1, 1])
    corrected = hamming74_correct([1, 0, 1, 1, 0, 1, 1])  # single-bit fix

All field arithmetic exact modular integer. Float categorically excluded.
"""

from __future__ import annotations
from typing import List, Tuple

__all__ = [
    "gf_add",
    "gf_sub",
    "gf_mul",
    "gf_inv",
    "gf_div",
    "gf_pow",
    "gf_poly_eval",
    "hamming74_encode",
    "hamming74_syndrome",
    "hamming74_correct",
    "hamming_distance",
    "hamming_weight",
    "repetition_decode",
    "compute_checksum",
    "min_distance",
]


def gf_add(a, b, p):
    """Addition in GF(p): (a + b) mod p."""
    return (a + b) % p


def gf_sub(a, b, p):
    """Subtraction in GF(p): (a - b) mod p."""
    return (a - b) % p


def gf_mul(a, b, p):
    """Multiplication in GF(p): (a * b) mod p."""
    return (a * b) % p


def gf_inv(a, p):
    """
    Multiplicative inverse in GF(p): a^(-1) mod p.

    I: element a (nonzero), prime p
    O: integer x such that (a * x) % p == 1

        gf_inv(3, 7) -> 5  (because 3*5 = 15 = 1 mod 7)
    """
    if a == 0:
        raise ValueError("Cannot invert zero in GF(%d)" % p)
    # Fermat's little theorem: a^(-1) = a^(p-2) mod p
    return pow(a, p - 2, p)


def gf_div(a, b, p):
    """Division in GF(p): a * b^(-1) mod p."""
    return gf_mul(a, gf_inv(b, p), p)


def gf_pow(a, n, p):
    """Exponentiation in GF(p): a^n mod p."""
    return pow(a, n, p)


def gf_poly_eval(coeffs, x, p):
    """
    Evaluate polynomial over GF(p) using Horner's method.

    I: coefficient list [a0, a1, ...] (constant first), point x, prime p
    O: p(x) mod p

        gf_poly_eval([1, 2, 3], 4, 11) -> (1 + 2*4 + 3*16) mod 11
    """
    if not coeffs:
        return 0
    result = coeffs[-1] % p
    for i in range(len(coeffs) - 2, -1, -1):
        result = (result * x + coeffs[i]) % p
    return result


# ---------------------------------------------------------------------------
# Hamming(7,4) code
# ---------------------------------------------------------------------------

# Generator matrix G (4x7): data bits -> codeword
# Systematic form: [I4 | P]
_G = [
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
]

# Parity check matrix H (3x7)
_H = [
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1],
]


def hamming74_encode(data):
    """
    Encode 4-bit data word to 7-bit Hamming(7,4) codeword.

    I: list of 4 bits [d0, d1, d2, d3]
    O: list of 7 bits [d0, d1, d2, d3, p0, p1, p2]

        hamming74_encode([1, 0, 1, 1]) -> [1, 0, 1, 1, 0, 0, 0]
    """
    if len(data) != 4:
        raise ValueError("Data must be 4 bits, got %d" % len(data))
    codeword = [0] * 7
    for j in range(7):
        total = 0
        for i in range(4):
            total += data[i] * _G[i][j]
        codeword[j] = total % 2
    return codeword


def hamming74_syndrome(received):
    """
    Compute syndrome of received 7-bit word.

    I: list of 7 bits
    O: integer syndrome (0-7), 0 means no error

        hamming74_syndrome([1, 0, 1, 1, 0, 0, 0]) -> 0
    """
    if len(received) != 7:
        raise ValueError("Received word must be 7 bits, got %d" % len(received))
    syndrome = 0
    for i in range(3):
        bit = 0
        for j in range(7):
            bit += _H[i][j] * received[j]
        bit = bit % 2
        syndrome += bit * (1 << i)
    return syndrome


def hamming74_correct(received):
    """
    Correct single-bit error in received 7-bit Hamming(7,4) codeword.

    I: list of 7 bits (possibly with one error)
    O: corrected list of 7 bits

    Syndrome maps to error position:
        0 -> no error
        1-7 -> flip bit at position (syndrome - 1)

        hamming74_correct([1, 1, 1, 1, 0, 0, 0])
        # if original was [1, 0, 1, 1, 0, 0, 0], corrects bit 1
    """
    syn = hamming74_syndrome(received)
    corrected = list(received)
    if syn != 0:
        # syndrome gives 1-indexed position in H column order
        # map syndrome to bit position
        for j in range(7):
            col_val = 0
            for i in range(3):
                col_val += _H[i][j] * (1 << i)
            if col_val == syn:
                corrected[j] ^= 1
                break
    return corrected


def hamming_distance(a, b):
    """
    Hamming distance between two bit vectors.

    I: two lists of bits (same length)
    O: integer count of differing positions
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have same length")
    return sum(1 for x, y in zip(a, b) if x != y)


def hamming_weight(a):
    """
    Hamming weight of a bit vector.

    I: list of bits
    O: integer count of 1s
    """
    return sum(1 for x in a if x != 0)


def min_distance(codewords):
    """
    Minimum Hamming distance of a code.

    I: list of codewords (each a list of bits)
    O: minimum distance between any two distinct codewords

        codewords = [hamming74_encode(d) for d in all_4bit_words]
        min_distance(codewords) -> 3
    """
    n = len(codewords)
    if n < 2:
        raise ValueError("Need at least 2 codewords")
    best = len(codewords[0]) + 1
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_distance(codewords[i], codewords[j])
            if d < best:
                best = d
    return best


def repetition_decode(bits, n):
    """
    Majority-vote decoding for repetition code.

    I: received bits (list of int), repetition factor n
    O: decoded bit (0 or 1)

        repetition_decode([1, 0, 1], 3) -> 1
    """
    if len(bits) != n:
        raise ValueError("Expected %d bits, got %d" % (n, len(bits)))
    ones = sum(bits)
    return 1 if ones > n // 2 else 0


def compute_checksum(data, p):
    """
    Simple polynomial checksum over GF(p).

    I: list of integer data values, prime p
    O: checksum integer in GF(p)

    Evaluates data as polynomial at x=2 in GF(p).
    """
    return gf_poly_eval(data, 2, p)
