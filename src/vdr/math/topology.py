"""
vdr.math.topology — Exact algebraic topology computations.

    from vdr.math.topology import betti_numbers, euler_characteristic

    betti = betti_numbers([d1, d0])   # exact via rank
    chi = euler_characteristic(betti) # alternating sum

All boundary operators, ranks, and Betti numbers computed with
exact VDR arithmetic. d∘d = 0 verified as exact zero matrix.
"""

from __future__ import annotations
from typing import List, Tuple, Set, FrozenSet

from vdr.core import VDR
from vdr.linalg import Vec, Mat

__all__ = [
    "boundary_matrix",
    "verify_d_squared_zero",
    "betti_numbers",
    "euler_characteristic",
    "simplicial_complex_boundaries",
]


def _simplex_sign(face, simplex):
    """
    Sign of a face in the boundary of a simplex.

    The boundary of [v0, v1, ..., vk] is
        sum (-1)^i [v0, ..., v_hat_i, ..., vk]

    Returns (-1)^i where face is obtained by removing vertex at index i.
    """
    simplex_list = sorted(simplex)
    face_list = sorted(face)

    # find which vertex was removed
    for i, v in enumerate(simplex_list):
        candidate = simplex_list[:i] + simplex_list[i + 1:]
        if candidate == face_list:
            return (-1) ** i

    return 0


def boundary_matrix(simplices_k, simplices_k_minus_1):
    """
    Construct boundary operator matrix d_k.

    d_k maps k-chains to (k-1)-chains.
    Entry [i, j] = coefficient of (k-1)-simplex i in the boundary of k-simplex j.

    I: list of k-simplices (as sorted tuples), list of (k-1)-simplices
    O: Mat with entries in {-1, 0, 1}

        # Triangle [0,1,2]: edges [0,1], [0,2], [1,2]
        d1 = boundary_matrix(
            [(0,1,2)],          # 2-simplices
            [(0,1), (0,2), (1,2)]  # 1-simplices
        )
    """
    n_rows = len(simplices_k_minus_1)
    n_cols = len(simplices_k)

    # index lookup for (k-1)-simplices
    face_index = {}
    for i, s in enumerate(simplices_k_minus_1):
        face_index[tuple(sorted(s))] = i

    rows = []
    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            sigma = simplices_k[j]
            face = simplices_k_minus_1[i]
            sign = _simplex_sign(face, sigma)
            row.append(VDR(sign))
        rows.append(row)

    return Mat(rows)


def verify_d_squared_zero(d_high, d_low):
    """
    Verify d∘d = 0 (fundamental property of boundary operators).

    I: d_{k+1} (Mat) and d_k (Mat)
    O: bool, True if d_low * d_high is the exact zero matrix

        verify_d_squared_zero(d2, d1) -> True
    """
    product = d_low.matmul(d_high)
    zero = Mat.zero(product.nrows, product.ncols)
    return product == zero


def betti_numbers(boundary_matrices):
    """
    Compute Betti numbers from a sequence of boundary matrices.

    beta_k = dim(ker(d_k)) - dim(im(d_{k+1}))
           = (n_k - rank(d_k)) - rank(d_{k+1})

    I: list of boundary matrices [d_1, d_2, ...] where d_k maps
       k-chains to (k-1)-chains. d_0 is omitted (maps to 0).
    O: list of Betti numbers [beta_0, beta_1, ...]

    Ranks computed via exact Gaussian elimination.

        betti = betti_numbers([d1, d2])
        # For hollow triangle: [1, 1]
        # For filled triangle: [1, 0]
    """
    if not boundary_matrices:
        return []

    # infer chain dimensions from matrix shapes
    # d_k has shape (n_{k-1}, n_k)
    # so n_k = d_k.ncols, n_{k-1} = d_k.nrows

    # number of chain groups = len(boundary_matrices) + 1
    # d_1 is boundary_matrices[0], maps 1-chains to 0-chains
    # n_0 = d_1.nrows, n_1 = d_1.ncols
    # if d_2 exists: n_1 = d_2.nrows, n_2 = d_2.ncols

    num_groups = len(boundary_matrices) + 1
    dims = [0] * num_groups
    ranks = [0] * (num_groups + 1)  # rank of d_k, padded

    # d_k is boundary_matrices[k-1] for k >= 1
    # dims[0] = d_1.nrows
    dims[0] = boundary_matrices[0].nrows
    for k in range(1, num_groups):
        if k - 1 < len(boundary_matrices):
            dims[k] = boundary_matrices[k - 1].ncols

    # ranks: rank of d_k
    # rank of d_0 = 0 (maps to 0)
    ranks[0] = 0
    for k in range(1, num_groups):
        if k - 1 < len(boundary_matrices):
            ranks[k] = boundary_matrices[k - 1].rank()

    # rank of d_{num_groups} = 0 (no higher boundary)
    ranks[num_groups] = 0

    # beta_k = dim(ker(d_k)) - dim(im(d_{k+1}))
    #        = (n_k - rank(d_k)) - rank(d_{k+1})
    betti = []
    for k in range(num_groups):
        ker_dim = dims[k] - ranks[k]
        im_dim = ranks[k + 1] if k + 1 <= num_groups else 0
        betti.append(ker_dim - im_dim)

    return betti


def euler_characteristic(betti):
    """
    Euler characteristic from Betti numbers.

    chi = sum (-1)^k * beta_k

    I: list of Betti numbers
    O: integer

        euler_characteristic([1, 0, 1]) -> 2  (sphere)
    """
    total = 0
    for k, b in enumerate(betti):
        if k % 2 == 0:
            total += b
        else:
            total -= b
    return total


def simplicial_complex_boundaries(simplices):
    """
    Build all boundary matrices for a simplicial complex.

    I: dict mapping dimension -> list of simplices (as sorted tuples)
       e.g. {0: [(0,),(1,),(2,)], 1: [(0,1),(0,2),(1,2)], 2: [(0,1,2)]}
    O: list of boundary matrices [d_1, d_2, ...]

        simplices = {
            0: [(0,), (1,), (2,)],
            1: [(0,1), (0,2), (1,2)],
            2: [(0,1,2)]
        }
        boundaries = simplicial_complex_boundaries(simplices)
    """
    max_dim = max(simplices.keys())
    boundaries = []

    for k in range(1, max_dim + 1):
        if k in simplices and (k - 1) in simplices:
            d = boundary_matrix(simplices[k], simplices[k - 1])
            boundaries.append(d)

    return boundaries
