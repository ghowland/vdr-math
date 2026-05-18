"""
vdr.signal.convolution — Exact discrete convolution and correlation.

    from vdr.signal.convolution import convolve, correlate

    y = convolve([VDR(1), VDR(2), VDR(3)], [VDR(1), VDR(1)])
    # [VDR(1), VDR(3), VDR(5), VDR(3)] exact

Each product exact. Each sum exact. No accumulation error.
"""

from __future__ import annotations
from typing import List

from vdr.core import VDR
from vdr.linalg import Vec, Mat

__all__ = [
    "convolve",
    "correlate",
    "toeplitz_mat",
    "convolve_via_toeplitz",
    "deconvolve",
]


def convolve(a, b):
    """
    Discrete linear convolution.

    (a * b)[n] = sum_k a[k] * b[n-k]

    I: two lists of VDR
    O: convolution result, length len(a) + len(b) - 1

        convolve([VDR(1), VDR(2), VDR(3)], [VDR(1), VDR(1)])
        -> [VDR(1), VDR(3), VDR(5), VDR(3)]
    """
    if not a or not b:
        return [VDR(0)]

    na = len(a)
    nb = len(b)
    nc = na + nb - 1
    result = [VDR(0)] * nc

    for i in range(na):
        ai = a[i] if isinstance(a[i], VDR) else VDR(a[i])
        for j in range(nb):
            bj = b[j] if isinstance(b[j], VDR) else VDR(b[j])
            result[i + j] = result[i + j] + ai * bj

    return result


def correlate(a, b):
    """
    Discrete cross-correlation.

    (a ⋆ b)[n] = sum_k a[k] * b[k+n]

    Equivalent to convolve(a, reversed(b)).

    I: two lists of VDR
    O: cross-correlation result

        correlate([VDR(1), VDR(2), VDR(3)], [VDR(1), VDR(1)])
    """
    b_rev = list(reversed(b))
    return convolve(a, b_rev)


def toeplitz_mat(h, n):
    """
    Build Toeplitz convolution matrix from impulse response h.

    For h of length m and output length n, builds an n x n matrix
    where row i has h shifted by i positions.

    I: impulse response h (list of VDR), output dimension n
    O: Toeplitz Mat (n x n)

    Used to verify that matrix-vector product equals direct convolution.

        T = toeplitz_mat([VDR(1), VDR(2), VDR(3)], 4)
    """
    m = len(h)
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            idx = i - j
            if 0 <= idx < m:
                row.append(h[idx] if isinstance(h[idx], VDR) else VDR(h[idx]))
            else:
                row.append(VDR(0))
        rows.append(row)
    return Mat(rows)


def convolve_via_toeplitz(a, b):
    """
    Convolution via Toeplitz matrix-vector product.

    Verifies that direct convolution equals matrix form.

    I: two lists of VDR
    O: convolution result via matrix multiply

        direct = convolve(a, b)
        matrix = convolve_via_toeplitz(a, b)
        assert direct == matrix  # exact
    """
    na = len(a)
    nb = len(b)
    nc = na + nb - 1

    # build Toeplitz matrix for a, size nc x nb
    rows = []
    for i in range(nc):
        row = []
        for j in range(nb):
            idx = i - j
            if 0 <= idx < na:
                row.append(a[idx] if isinstance(a[idx], VDR) else VDR(a[idx]))
            else:
                row.append(VDR(0))
        rows.append(row)

    T = Mat(rows)
    b_vec = Vec([x if isinstance(x, VDR) else VDR(x) for x in b])
    result_vec = T.matvec(b_vec)

    return [result_vec[i] for i in range(nc)]


def deconvolve(y, h):
    """
    Deconvolution: given y = h * x, recover x.

    Uses polynomial division: x = y / h in polynomial ring.

    I: output signal y, impulse response h (lists of VDR)
    O: input signal x (list of VDR), exact

    Only works when h divides y exactly (no remainder).

        x = deconvolve(convolve(a, b), b)
        assert x == a  # exact recovery
    """
    from vdr.math.polynomial import poly_divmod
    quot, rem = poly_divmod(y, h)
    return quot
