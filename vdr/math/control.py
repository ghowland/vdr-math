"""
vdr.math.control — Exact control theory computations.

    from vdr.math.control import state_evolve, is_controllable, cayley_hamilton_verify

    trajectory = state_evolve(A, B, x0, inputs)  # zero drift
    ctrl = is_controllable(A, B)                  # exact rank
    zero_mat = cayley_hamilton_verify(A)           # exact zero matrix

State-space evolution, controllability, observability, transfer functions.
All exact VDR arithmetic. Cayley-Hamilton verified as structural equality.
"""

from __future__ import annotations
from typing import List, Tuple, Callable

from vdr.core import VDR
from vdr.linalg import Vec, Mat
from vdr.math.polynomial import poly_eval, char_poly_2x2

__all__ = [
    "state_evolve",
    "controllability_matrix",
    "observability_matrix",
    "is_controllable",
    "is_observable",
    "transfer_function_eval",
    "cayley_hamilton_verify",
    "char_poly",
    "controllability_gramian",
    "mat_pow",
]


def mat_pow(M, n):
    """
    Exact matrix power via repeated squaring.

    I: square Mat M, non-negative integer n
    O: M^n as Mat, exact

        mat_pow(Mat.from_ints([[1,1],[1,0]]), 10)  # Fibonacci matrix
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


def state_evolve(A, B, x0, inputs):
    """
    Discrete-time state-space evolution.

    x[n+1] = A * x[n] + B * u[n]

    I: system matrix A (Mat), input matrix B (Mat), initial state x0 (Vec),
       input sequence (list of Vec)
    O: list of state Vec [x0, x1, x2, ...], all exact, zero drift

        A = Mat.from_fracs([[(9,10), (1,10)], [(-1,10), (9,10)]])
        B = Mat.from_ints([[1, 0], [0, 1]])
        x0 = Vec.from_ints([1, 0])
        inputs = [Vec.from_ints([0, 0])] * 100
        trajectory = state_evolve(A, B, x0, inputs)
    """
    trajectory = [x0]
    x = x0

    for u in inputs:
        x = A.matvec(x) + B.matvec(u)
        trajectory.append(x)

    return trajectory


def controllability_matrix(A, B):
    """
    Controllability matrix: [B, AB, A^2B, ..., A^(n-1)B].

    I: A (n x n Mat), B (n x m Mat)
    O: (n x n*m) Mat

        C = controllability_matrix(A, B)
        is_controllable = (C.rank() == A.nrows)
    """
    n = A.nrows
    cols_per_block = B.ncols

    blocks = [B]
    power = B
    for _ in range(1, n):
        power = A.matmul(power)
        blocks.append(power)

    # assemble into single matrix
    rows = []
    for i in range(n):
        row = []
        for block in blocks:
            for j in range(cols_per_block):
                row.append(block[i, j])
        rows.append(row)

    return Mat(rows)


def observability_matrix(A, C):
    """
    Observability matrix: [C; CA; CA^2; ...; CA^(n-1)].

    I: A (n x n Mat), C (p x n Mat)
    O: (p*n x n) Mat
    """
    n = A.nrows
    rows_per_block = C.nrows

    blocks = [C]
    power = C
    for _ in range(1, n):
        power = power.matmul(A)
        blocks.append(power)

    # stack vertically
    rows = []
    for block in blocks:
        for i in range(rows_per_block):
            row = [block[i, j] for j in range(block.ncols)]
            rows.append(row)

    return Mat(rows)


def is_controllable(A, B):
    """
    Test controllability: controllability matrix has full rank.

    I: A (n x n Mat), B (n x m Mat)
    O: bool, exact (rank via Gaussian)
    """
    C = controllability_matrix(A, B)
    return C.rank() == A.nrows


def is_observable(A, C):
    """
    Test observability: observability matrix has full rank.

    I: A (n x n Mat), C (p x n Mat)
    O: bool, exact
    """
    O = observability_matrix(A, C)
    return O.rank() == A.nrows


def char_poly(A):
    """
    Characteristic polynomial of a square matrix.

    For 2x2: lambda^2 - trace*lambda + det
    For general n: uses Faddeev-LeVerrier algorithm.

    I: square Mat A
    O: coefficient list [c0, c1, ..., cn] (constant first, leading = 1)
       where p(lambda) = c0 + c1*lambda + ... + cn*lambda^n
    """
    n = A.nrows

    if n == 1:
        return [-A[0, 0], VDR(1)]

    if n == 2:
        return char_poly_2x2(A)

    # Faddeev-LeVerrier algorithm
    # c_n = 1
    # M_k = A * M_{k-1} + c_{n-k+1} * I
    # c_{n-k} = -trace(A * M_k) / k

    coeffs = [VDR(0)] * (n + 1)
    coeffs[n] = VDR(1)

    M = Mat.identity(n)
    for k in range(1, n + 1):
        AM = A.matmul(M)
        c = -AM.trace() / VDR(k)
        coeffs[n - k] = c
        if k < n:
            M = AM + Mat.identity(n).scale(c)

    return coeffs


def cayley_hamilton_verify(A):
    """
    Verify Cayley-Hamilton theorem: p(A) = 0 where p is characteristic polynomial.

    I: square Mat A
    O: p(A) as Mat — should be exact zero matrix

        result = cayley_hamilton_verify(A)
        assert result == Mat.zero(A.nrows, A.ncols)
    """
    n = A.nrows
    cp = char_poly(A)

    # evaluate polynomial at matrix A
    # p(A) = c0*I + c1*A + c2*A^2 + ... + cn*A^n
    result = Mat.zero(n, n)
    A_power = Mat.identity(n)

    for i, c in enumerate(cp):
        if c != VDR(0):
            result = result + A_power.scale(c)
        if i < len(cp) - 1:
            A_power = A_power.matmul(A)

    return result


def transfer_function_eval(num_coeffs, den_coeffs, s):
    """
    Evaluate transfer function H(s) = N(s) / D(s).

    I: numerator polynomial coefficients, denominator polynomial coefficients,
       complex frequency s (VDR for real, or tuple (real, imag) for complex)
    O: H(s) as VDR (real) or tuple (real, imag)

        # H(s) = 1 / (s^2 + 3s + 2)
        transfer_function_eval([VDR(1)], [VDR(2), VDR(3), VDR(1)], VDR(1))
    """
    if isinstance(s, tuple):
        # complex evaluation
        sr, si = s
        nr = _complex_poly_eval_real(num_coeffs, sr, si)
        ni = _complex_poly_eval_imag(num_coeffs, sr, si)
        dr = _complex_poly_eval_real(den_coeffs, sr, si)
        di = _complex_poly_eval_imag(den_coeffs, sr, si)

        # (nr + ni*i) / (dr + di*i) = ((nr*dr + ni*di) + (ni*dr - nr*di)*i) / (dr^2 + di^2)
        denom = dr * dr + di * di
        real_part = (nr * dr + ni * di) / denom
        imag_part = (ni * dr - nr * di) / denom
        return (real_part, imag_part)
    else:
        num_val = poly_eval(num_coeffs, s)
        den_val = poly_eval(den_coeffs, s)
        return num_val / den_val


def _complex_poly_eval_real(coeffs, sr, si):
    """Real part of polynomial evaluated at s = sr + si*i."""
    # Horner-like for complex: track real and imaginary parts
    if not coeffs:
        return VDR(0)
    rr = VDR(0)
    ri = VDR(0)
    for c in reversed(coeffs):
        # (rr + ri*i) * (sr + si*i) + c
        new_rr = rr * sr - ri * si + c
        new_ri = rr * si + ri * sr
        rr, ri = new_rr, new_ri
    return rr


def _complex_poly_eval_imag(coeffs, sr, si):
    """Imaginary part of polynomial evaluated at s = sr + si*i."""
    if not coeffs:
        return VDR(0)
    rr = VDR(0)
    ri = VDR(0)
    for c in reversed(coeffs):
        new_rr = rr * sr - ri * si + c
        new_ri = rr * si + ri * sr
        rr, ri = new_rr, new_ri
    return ri


def controllability_gramian(A, B, steps):
    """
    Discrete-time controllability Gramian.

    W = sum_{k=0}^{steps-1} A^k * B * B^T * (A^T)^k

    I: A (n x n), B (n x m), number of steps
    O: Gramian Mat (n x n), symmetric, exact

        W = controllability_gramian(A, B, 10)
        # det(W) > 0 means controllable within horizon
    """
    n = A.nrows
    W = Mat.zero(n, n)
    BT = B.T
    AT = A.T

    A_pow = Mat.identity(n)
    AT_pow = Mat.identity(n)

    for _ in range(steps):
        # A^k * B * B^T * (A^T)^k
        AB = A_pow.matmul(B)
        ABT = AB.matmul(BT)
        term = ABT.matmul(AT_pow)
        W = W + term

        A_pow = A_pow.matmul(A)
        AT_pow = AT_pow.matmul(AT)

    return W
