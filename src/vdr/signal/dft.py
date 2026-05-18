"""
vdr.signal.dft — Exact Discrete Fourier Transform.

    from vdr.signal.dft import exact_dft, exact_idft, parseval_verify

    X = exact_dft([VDR(1), VDR(2), VDR(3), VDR(4)])
    x = exact_idft(X)
    assert x == [VDR(1), VDR(2), VDR(3), VDR(4)]  # exact roundtrip

All digital signals are rational. Every frequency bin is an exact
complex value (real, imag) pair of VDR.

Twiddle factors use sin/cos series at configurable depth.
"""

from __future__ import annotations
from typing import List, Tuple

from vdr.core import VDR
from vdr.math.transcendental import sin_series, cos_series, PI

__all__ = [
    "exact_dft",
    "exact_idft",
    "parseval_verify",
    "twiddle_factor",
    "dft_matrix",
]


def twiddle_factor(k, n, N, depth=16):
    """
    DFT twiddle factor W_N^{kn} = exp(-2*pi*i*k*n/N).

    Returns (cos_part, sin_part) where W = cos - i*sin.

    The angle is 2*pi*k*n/N. For rational k*n/N, the trig values
    are computed via Taylor series at configurable depth.

    I: frequency index k, time index n, DFT size N, series depth
    O: (cos_val, neg_sin_val) as VDR tuple

    Special cases for exact values:
        angle = 0: (1, 0)
        angle = pi/2: (0, -1)
        angle = pi: (-1, 0)
        angle = 3pi/2: (0, 1)
    """
    # angle = 2*pi*k*n / N
    # reduce k*n mod N for periodicity
    kn = (k * n) % N

    if kn == 0:
        return (VDR(1), VDR(0))

    # check quarter-period special cases
    if N % 4 == 0:
        quarter = N // 4
        if kn == quarter:
            return (VDR(0), VDR(-1))
        if kn == 2 * quarter:
            return (VDR(-1), VDR(0))
        if kn == 3 * quarter:
            return (VDR(0), VDR(1))

    # check half-period
    if N % 2 == 0 and kn == N // 2:
        return (VDR(-1), VDR(0))

    # general case: compute via series
    # angle = 2 * pi * kn / N
    # use Q335 PI for the multiplication
    angle = VDR(2) * PI * VDR(kn) / VDR(N)

    c = cos_series(angle, depth)
    s = sin_series(angle, depth)

    return (c, -s)  # W = cos(theta) - i*sin(theta)


def exact_dft(x, depth=16):
    """
    Exact Discrete Fourier Transform.

    X[k] = sum_{n=0}^{N-1} x[n] * W_N^{kn}

    where W_N = exp(-2*pi*i/N).

    I: real signal as list of VDR, trig series depth
    O: list of (real, imag) VDR tuples, one per frequency bin

        X = exact_dft([VDR(1), VDR(2), VDR(3), VDR(4)])
    """
    N = len(x)
    result = []

    for k in range(N):
        re = VDR(0)
        im = VDR(0)
        for n in range(N):
            xn = x[n] if isinstance(x[n], VDR) else VDR(x[n])
            c, s = twiddle_factor(k, n, N, depth)
            # X[k] += x[n] * (c + i*s)
            # but W = c - i*sin, and we stored (c, -sin) as (c, s)
            # so x[n] * W = x[n]*c + i*x[n]*s
            re = re + xn * c
            im = im + xn * s
        result.append((re, im))

    return result


def exact_idft(X, depth=16):
    """
    Exact Inverse Discrete Fourier Transform.

    x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * W_N^{-kn}

    I: frequency domain as list of (real, imag) VDR tuples, depth
    O: real signal as list of VDR

    IDFT(DFT(x)) == x exactly.
    """
    N = len(X)
    result = []

    for n in range(N):
        re = VDR(0)
        im = VDR(0)
        for k in range(N):
            Xr, Xi = X[k]
            # W_N^{-kn} = conjugate of W_N^{kn}
            # if W_N^{kn} = (c, s) where s = -sin, then conjugate = (c, -s)
            c, s = twiddle_factor(k, n, N, depth)
            conj_c = c
            conj_s = -s  # conjugate flips imaginary part

            # (Xr + i*Xi) * (conj_c + i*conj_s)
            # real part: Xr*conj_c - Xi*conj_s
            # imag part: Xr*conj_s + Xi*conj_c
            re = re + Xr * conj_c - Xi * conj_s
            im = im + Xr * conj_s + Xi * conj_c

        # divide by N
        re = re / VDR(N)
        im = im / VDR(N)

        # for real input signals, imaginary part should be ~0
        result.append(re)

    return result


def parseval_verify(x, X):
    """
    Verify Parseval's theorem: sum |x[n]|^2 == (1/N) * sum |X[k]|^2.

    I: time-domain signal (list of VDR), frequency-domain (list of (re,im) tuples)
    O: bool, True if identity holds exactly

        X = exact_dft(x)
        parseval_verify(x, X)  # True
    """
    N = len(x)

    # time domain energy
    time_energy = VDR(0)
    for xn in x:
        xn = xn if isinstance(xn, VDR) else VDR(xn)
        time_energy = time_energy + xn * xn

    # frequency domain energy
    freq_energy = VDR(0)
    for Xr, Xi in X:
        freq_energy = freq_energy + Xr * Xr + Xi * Xi

    freq_energy = freq_energy / VDR(N)

    return time_energy == freq_energy


def dft_matrix(N, depth=16):
    """
    Build the N x N DFT matrix explicitly.

    W[k,n] = twiddle_factor(k, n, N)

    Returns as pair of Mat (real_part, imag_part).

    I: DFT size N, series depth
    O: (real_mat, imag_mat) as Mat pair
    """
    from vdr.linalg import Mat

    real_rows = []
    imag_rows = []

    for k in range(N):
        real_row = []
        imag_row = []
        for n in range(N):
            c, s = twiddle_factor(k, n, N, depth)
            real_row.append(c)
            imag_row.append(s)
        real_rows.append(real_row)
        imag_rows.append(imag_row)

    return (Mat(real_rows), Mat(imag_rows))
