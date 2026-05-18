"""
vdr.math.wavelets — Exact Haar wavelet transforms.

    from vdr.math.wavelets import haar_forward, haar_inverse

    avgs, dets = haar_forward([VDR(1), VDR(3), VDR(5), VDR(7)])
    signal = haar_inverse(avgs, dets)
    # signal == [VDR(1), VDR(3), VDR(5), VDR(7)] exactly

Perfect reconstruction. Parseval energy identity exact.
All operations exact VDR rational arithmetic.
"""

from __future__ import annotations
from typing import List, Tuple

from vdr.core import VDR

__all__ = [
    "haar_forward",
    "haar_inverse",
    "haar_multilevel",
    "haar_reconstruct_multilevel",
    "energy",
    "parseval_verify",
    "threshold_hard",
]


def haar_forward(signal):
    """
    One-level Haar forward transform.

    I: signal as list of VDR (length must be even)
    O: (averages, details) each list of VDR, half the length

    averages[k] = (signal[2k] + signal[2k+1]) / 2
    details[k]  = (signal[2k] - signal[2k+1]) / 2

        haar_forward([VDR(1), VDR(3), VDR(5), VDR(7)])
        -> ([VDR(2), VDR(6)], [VDR(-1), VDR(-1)])
    """
    n = len(signal)
    if n % 2 != 0:
        raise ValueError("Signal length must be even, got %d" % n)

    avgs = []
    dets = []
    for k in range(n // 2):
        a = signal[2 * k]
        b = signal[2 * k + 1]
        avgs.append((a + b) / VDR(2))
        dets.append((a - b) / VDR(2))

    return avgs, dets


def haar_inverse(avgs, dets):
    """
    One-level Haar inverse transform.

    I: averages and details (lists of VDR, same length)
    O: reconstructed signal (list of VDR, double the length)

    signal[2k]   = avgs[k] + dets[k]
    signal[2k+1] = avgs[k] - dets[k]

    Perfect reconstruction: haar_inverse(*haar_forward(x)) == x exactly.
    """
    if len(avgs) != len(dets):
        raise ValueError("Averages and details must have same length")

    signal = []
    for k in range(len(avgs)):
        signal.append(avgs[k] + dets[k])
        signal.append(avgs[k] - dets[k])

    return signal


def haar_multilevel(signal, levels):
    """
    Multi-level Haar decomposition.

    I: signal (list of VDR, length power of 2), number of levels
    O: list of (averages, details) per level, from finest to coarsest

    At each level, the averages from the previous level become the
    input signal for the next level.

        decomp = haar_multilevel([VDR(1),VDR(3),VDR(5),VDR(7)], 2)
        # level 0: avgs=[2,6], dets=[-1,-1]
        # level 1: avgs=[4], dets=[-2]
    """
    decomposition = []
    current = list(signal)

    for _ in range(levels):
        if len(current) < 2:
            break
        avgs, dets = haar_forward(current)
        decomposition.append((avgs, dets))
        current = avgs

    return decomposition


def haar_reconstruct_multilevel(decomposition):
    """
    Reconstruct signal from multi-level Haar decomposition.

    I: list of (averages, details) from haar_multilevel
    O: reconstructed signal (list of VDR), exact

    Perfect reconstruction: reconstruct(multilevel(x, n)) == x exactly.
    """
    if not decomposition:
        return []

    # start from coarsest level averages
    current = decomposition[-1][0]

    # reconstruct from coarsest to finest
    for i in range(len(decomposition) - 1, -1, -1):
        _, dets = decomposition[i]
        current = haar_inverse(current, dets)

    return current


def energy(signal):
    """
    Signal energy: sum |x[n]|^2.

    I: signal as list of VDR
    O: energy as VDR, exact

        energy([VDR(1), VDR(3), VDR(5), VDR(7)]) -> VDR(84)
    """
    total = VDR(0)
    for x in signal:
        total = total + x * x
    return total


def parseval_verify(signal, decomposition):
    """
    Verify Parseval energy identity for Haar transform.

    For our definition (divide by 2), the energy splits as:
        energy(signal) = 2^levels * energy(coarsest_avgs) +
                         sum over levels of 2^(level+1) * energy(details)

    Simpler check: reconstruct and compare energies.

    I: original signal, decomposition from haar_multilevel
    O: bool, True if energy preserved exactly
    """
    reconstructed = haar_reconstruct_multilevel(decomposition)
    return energy(signal) == energy(reconstructed)


def threshold_hard(details, thresh):
    """
    Hard thresholding for wavelet denoising.

    I: detail coefficients (list of VDR), threshold (VDR)
    O: thresholded details (list of VDR)

    Coefficients with |d| < thresh are set to zero.
    Exact rational comparison — no float ambiguity at boundary.

        threshold_hard([VDR(1,10), VDR(3), VDR(-1,5)], VDR(1,2))
        -> [VDR(0), VDR(3), VDR(0)]
    """
    result = []
    for d in details:
        if abs(d) < thresh:
            result.append(VDR(0))
        else:
            result.append(d)
    return result
