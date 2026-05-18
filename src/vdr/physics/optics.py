"""
vdr.physics.optics — Exact paraxial optics via ABCD matrices.

    from vdr.physics.optics import free_space, thin_lens, system_matrix
    from vdr.physics.optics import verify_symplecticity, resonator_stability

    M = system_matrix([free_space(VDR(1)), thin_lens(VDR(2)), free_space(VDR(1))])
    verify_symplecticity(M)  # True, det(M) == 1 exactly

det(M) = 1 (symplecticity) exact after any number of elements.
M^1000 via repeated squaring: exact. Float accumulates ~1e-12.
"""

from __future__ import annotations
from typing import List

from vdr.core import VDR
from vdr.linalg import Mat

__all__ = [
    "free_space",
    "thin_lens",
    "flat_mirror",
    "curved_mirror",
    "thick_lens",
    "interface",
    "system_matrix",
    "verify_symplecticity",
    "resonator_stability",
    "matrix_power",
    "beam_parameter",
]


def free_space(d):
    """
    ABCD matrix for free-space propagation of distance d.

    [[1, d], [0, 1]]

    I: distance d (VDR)
    O: 2x2 Mat

        free_space(VDR(1, 2))  # half-unit propagation
    """
    d = d if isinstance(d, VDR) else VDR(d)
    return Mat([[VDR(1), d], [VDR(0), VDR(1)]])


def thin_lens(f):
    """
    ABCD matrix for thin lens of focal length f.

    [[1, 0], [-1/f, 1]]

    I: focal length f (VDR, nonzero)
    O: 2x2 Mat

        thin_lens(VDR(10))  # f = 10
    """
    f = f if isinstance(f, VDR) else VDR(f)
    return Mat([[VDR(1), VDR(0)], [-VDR(1) / f, VDR(1)]])


def flat_mirror():
    """
    ABCD matrix for flat mirror (reflection).

    [[1, 0], [0, 1]]  (identity — flat mirror just reverses direction)

    O: 2x2 identity Mat
    """
    return Mat.identity(2)


def curved_mirror(R):
    """
    ABCD matrix for curved mirror with radius of curvature R.

    [[1, 0], [-2/R, 1]]

    Equivalent to thin lens with f = R/2.

    I: radius of curvature R (VDR)
    O: 2x2 Mat
    """
    R = R if isinstance(R, VDR) else VDR(R)
    return Mat([[VDR(1), VDR(0)], [-VDR(2) / R, VDR(1)]])


def thick_lens(n, R1, R2, d):
    """
    ABCD matrix for thick lens.

    Composed of: interface(1, n, R1) * free_space(d/n) * interface(n, 1, R2)

    I: refractive index n, front radius R1, back radius R2, thickness d (all VDR)
    O: 2x2 Mat
    """
    M1 = interface(VDR(1), n, R1)
    M2 = free_space(d / n)
    M3 = interface(n, VDR(1), R2)
    return M3.matmul(M2.matmul(M1))


def interface(n1, n2, R):
    """
    ABCD matrix for refraction at a curved interface.

    [[1, 0], [(n1-n2)/(n2*R), n1/n2]]

    I: refractive indices n1, n2, radius of curvature R (all VDR)
    O: 2x2 Mat
    """
    n1 = n1 if isinstance(n1, VDR) else VDR(n1)
    n2 = n2 if isinstance(n2, VDR) else VDR(n2)
    R = R if isinstance(R, VDR) else VDR(R)

    return Mat([
        [VDR(1), VDR(0)],
        [(n1 - n2) / (n2 * R), n1 / n2]
    ])


def system_matrix(elements):
    """
    System ABCD matrix from sequence of optical elements.

    Product of element matrices, right-to-left (first element is rightmost
    in the product since light passes through it first).

    I: list of 2x2 Mat [M1, M2, M3, ...] in order light encounters them
    O: system Mat = Mn * ... * M2 * M1

        M = system_matrix([free_space(VDR(1)), thin_lens(VDR(2)), free_space(VDR(1))])
    """
    if not elements:
        return Mat.identity(2)

    result = elements[-1]
    for i in range(len(elements) - 2, -1, -1):
        result = result.matmul(elements[i])

    return result


def verify_symplecticity(M):
    """
    Verify det(M) == 1 exactly (symplecticity / energy conservation).

    I: 2x2 Mat
    O: bool, True if det is exactly 1

    Float gives 1.0 +/- ~1e-12 after 1000 elements.
    VDR gives exactly 1.

        verify_symplecticity(system_matrix(elements))  # True
    """
    return M.det() == VDR(1)


def resonator_stability(M):
    """
    Resonator stability criterion.

    A resonator is stable when |(A+D)/2| < 1.

    I: round-trip system Mat (2x2)
    O: (is_stable, half_trace) where is_stable is bool (exact comparison)
       and half_trace is VDR

    Exact rational comparison — no borderline float ambiguity.

        stable, ht = resonator_stability(roundtrip_matrix)
    """
    A = M[0, 0]
    D = M[1, 1]
    half_trace = (A + D) / VDR(2)

    is_stable = abs(half_trace) < VDR(1)

    return (is_stable, half_trace)


def matrix_power(M, n):
    """
    Exact matrix power via repeated squaring.

    M^1000 in 10 multiplications, exact.

    I: 2x2 Mat, non-negative integer n
    O: M^n as Mat

    Float accumulates ~1000 butterfly-equivalent roundings.
    VDR stays exact.

        M1000 = matrix_power(system_matrix(elements), 1000)
        verify_symplecticity(M1000)  # still True
    """
    if n == 0:
        return Mat.identity(M.nrows)
    if n == 1:
        return M

    result = Mat.identity(M.nrows)
    base = M
    exp = n

    while exp > 0:
        if exp % 2 == 1:
            result = result.matmul(base)
        base = base.matmul(base)
        exp //= 2

    return result


def beam_parameter(M, q_in):
    """
    Transform complex beam parameter through ABCD matrix.

    q_out = (A*q_in + B) / (C*q_in + D)

    For real beam parameters (geometric optics):

    I: 2x2 Mat, input beam parameter q (VDR)
    O: output beam parameter as VDR

        q_out = beam_parameter(M, VDR(10))
    """
    A = M[0, 0]
    B = M[0, 1]
    C = M[1, 0]
    D = M[1, 1]

    num = A * q_in + B
    den = C * q_in + D

    return num / den
