"""
vdr.signal.filters — Exact digital filters.

    from vdr.signal.filters import iir_filter, moving_average, z_transform

    y = iir_filter([VDR(1),VDR(0),VDR(0)], VDR(1,2))
    # y[n] = (1/2)*y[n-1] + x[n], exact at every step

IIR chains: each step nests one R level. When rational powers collapse
(e.g. (1/sqrt(2))^20 = 1/1024), normalization rule N7 fires.
Year-long operation same precision as second 1.
"""

from __future__ import annotations
from typing import List, Tuple

from vdr.core import VDR

__all__ = [
    "iir_filter",
    "iir_filter_general",
    "moving_average",
    "z_transform",
    "frequency_response",
]


def iir_filter(x, a_coeff):
    """
    First-order IIR filter: y[n] = a * y[n-1] + x[n].

    I: input signal x (list of VDR), feedback coefficient a (VDR)
    O: output signal y (list of VDR), exact at every step

    Each step is one VDR multiplication and one addition.
    No drift regardless of signal length.

        iir_filter([VDR(1), VDR(0), VDR(0), VDR(0)], VDR(1, 2))
        -> [VDR(1), VDR(1,2), VDR(1,4), VDR(1,8)]
    """
    if not x:
        return []

    y = [VDR(0)] * len(x)
    y[0] = x[0] if isinstance(x[0], VDR) else VDR(x[0])

    for n in range(1, len(x)):
        xn = x[n] if isinstance(x[n], VDR) else VDR(x[n])
        y[n] = a_coeff * y[n - 1] + xn

    return y


def iir_filter_general(x, b_coeffs, a_coeffs):
    """
    General IIR filter (direct form I).

    y[n] = sum_{k=0}^{M} b[k]*x[n-k] - sum_{k=1}^{N} a[k]*y[n-k]

    Convention: a[0] = 1 (normalized). If a[0] != 1, all coefficients
    are divided by a[0].

    I: input x (list of VDR), feedforward b (list of VDR),
       feedback a (list of VDR, a[0]=1)
    O: output y (list of VDR)

        # Second-order: y[n] = x[n] + 0.5*x[n-1] - 0.25*y[n-1]
        iir_filter_general(x, [VDR(1), VDR(1,2)], [VDR(1), VDR(1,4)])
    """
    if not x:
        return []

    M = len(b_coeffs)
    N = len(a_coeffs)

    # normalize by a[0]
    a0 = a_coeffs[0] if isinstance(a_coeffs[0], VDR) else VDR(a_coeffs[0])
    b = [(_v(b_coeffs[i]) / a0) if i < M else VDR(0) for i in range(M)]
    a = [(_v(a_coeffs[i]) / a0) if i < N else VDR(0) for i in range(N)]

    L = len(x)
    y = [VDR(0)] * L

    for n in range(L):
        total = VDR(0)
        # feedforward
        for k in range(M):
            if n - k >= 0:
                xnk = x[n - k] if isinstance(x[n - k], VDR) else VDR(x[n - k])
                total = total + b[k] * xnk
        # feedback (skip k=0 since a[0]=1 after normalization)
        for k in range(1, N):
            if n - k >= 0:
                total = total - a[k] * y[n - k]

        y[n] = total

    return y


def _v(x):
    if isinstance(x, VDR):
        return x
    return VDR(x)


def moving_average(signal, window):
    """
    Moving average filter.

    y[n] = (1/window) * sum_{k=0}^{window-1} x[n-k]

    I: input signal (list of VDR), window size (int)
    O: filtered signal (list of VDR), exact

    Note: division by window introduces 1/window in denominator.
    If window has odd factors, this is where the decimal trap bites float
    but VDR handles exactly.

        moving_average([VDR(1), VDR(2), VDR(3), VDR(4)], 3)
    """
    if not signal or window <= 0:
        return []

    n = len(signal)
    result = []
    w = VDR(window)

    for i in range(n):
        total = VDR(0)
        count = 0
        for k in range(window):
            idx = i - k
            if idx >= 0:
                xk = signal[idx] if isinstance(signal[idx], VDR) else VDR(signal[idx])
                total = total + xk
                count += 1
        if count > 0:
            result.append(total / VDR(count))
        else:
            result.append(VDR(0))

    return result


def z_transform(h, z):
    """
    Evaluate z-transform H(z) = sum_{n=0}^{N-1} h[n] * z^{-n}.

    I: impulse response h (list of VDR), evaluation point z (VDR)
    O: H(z) as VDR, exact

        z_transform([VDR(1), VDR(1, 2), VDR(1, 4)], VDR(2))
        # H(2) = 1 + 1/2 * 1/2 + 1/4 * 1/4 = 1 + 1/4 + 1/16
    """
    if not h:
        return VDR(0)
    if z == VDR(0):
        return h[0] if isinstance(h[0], VDR) else VDR(h[0])

    z_inv = VDR(1) / z
    total = VDR(0)
    z_inv_n = VDR(1)

    for n in range(len(h)):
        hn = h[n] if isinstance(h[n], VDR) else VDR(h[n])
        total = total + hn * z_inv_n
        z_inv_n = z_inv_n * z_inv

    return total


def frequency_response(b_coeffs, a_coeffs, omega, depth=16):
    """
    Frequency response H(e^{j*omega}) of a digital filter.

    H(z) = B(z) / A(z) evaluated at z = e^{j*omega}.

    I: feedforward b, feedback a coefficients (lists of VDR),
       frequency omega (VDR), trig series depth
    O: (magnitude_sq, real, imag) as VDR tuple

    Uses exact sin/cos series for the complex exponential.

        frequency_response([VDR(1)], [VDR(1), VDR(-1,2)], VDR(1,4))
    """
    from vdr.math.transcendental import sin_series, cos_series

    # evaluate B(e^{jw}) and A(e^{jw})
    # e^{-jnw} = cos(nw) - j*sin(nw)
    b_re = VDR(0)
    b_im = VDR(0)
    for n in range(len(b_coeffs)):
        bn = b_coeffs[n] if isinstance(b_coeffs[n], VDR) else VDR(b_coeffs[n])
        angle = VDR(n) * omega
        c = cos_series(angle, depth)
        s = sin_series(angle, depth)
        b_re = b_re + bn * c
        b_im = b_im - bn * s

    a_re = VDR(0)
    a_im = VDR(0)
    for n in range(len(a_coeffs)):
        an = a_coeffs[n] if isinstance(a_coeffs[n], VDR) else VDR(a_coeffs[n])
        angle = VDR(n) * omega
        c = cos_series(angle, depth)
        s = sin_series(angle, depth)
        a_re = a_re + an * c
        a_im = a_im - an * s

    # H = B / A as complex division
    # (b_re + j*b_im) / (a_re + j*a_im)
    denom = a_re * a_re + a_im * a_im
    h_re = (b_re * a_re + b_im * a_im) / denom
    h_im = (b_im * a_re - b_re * a_im) / denom

    mag_sq = h_re * h_re + h_im * h_im

    return (mag_sq, h_re, h_im)
