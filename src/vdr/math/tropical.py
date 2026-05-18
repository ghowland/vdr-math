"""
vdr.math.tropical — Exact tropical algebra and lattice computations.

    from vdr.math.tropical import trop_mat_mul, gram_schmidt_exact, lll_reduce

    shortest = trop_mat_mul(A, B, n)          # min-plus matrix multiply
    ortho, mu = gram_schmidt_exact(basis)     # exact Gram-Schmidt
    reduced = lll_reduce(basis)               # exact LLL, no float in threshold

Tropical: min-plus semiring. Lattice: exact Gram-Schmidt and Lovasz condition.
"""

from __future__ import annotations
from typing import List, Tuple, Optional

from vdr.core import VDR
from vdr.linalg import Vec, Mat

__all__ = [
    "trop_add",
    "trop_mul",
    "trop_mat_mul",
    "trop_det",
    "gram_matrix",
    "gram_schmidt_exact",
    "lovasz_condition",
    "lll_reduce",
    "size_reduce",
]

# Tropical infinity sentinel
TROP_INF = None


def trop_add(a, b):
    """
    Tropical addition: min(a, b).

    None represents +infinity.

        trop_add(VDR(3), VDR(5)) -> VDR(3)
        trop_add(VDR(3), None) -> VDR(3)
    """
    if a is None:
        return b
    if b is None:
        return a
    return a if a <= b else b


def trop_mul(a, b):
    """
    Tropical multiplication: a + b (ordinary addition).

    None represents +infinity. inf + anything = inf.

        trop_mul(VDR(3), VDR(5)) -> VDR(8)
    """
    if a is None or b is None:
        return None
    return a + b


def trop_mat_mul(A, B, n):
    """
    Tropical matrix multiplication (min-plus).

    C[i][j] = min_k (A[i][k] + B[k][j])

    I: A, B as list-of-lists (n x n), entries VDR or None
    O: product as list-of-lists

        # 2-hop shortest paths
        result = trop_mat_mul(dist, dist, n)
    """
    C = []
    for i in range(n):
        row = []
        for j in range(n):
            val = None
            for k in range(n):
                product = trop_mul(A[i][k], B[k][j])
                val = trop_add(val, product)
            row.append(val)
        C.append(row)
    return C


def trop_det(M, n):
    """
    Tropical determinant: minimum weight perfect matching.

    det_trop(A) = min over permutations sigma of sum A[i][sigma(i)]

    I: matrix as list-of-lists (n x n), entries VDR or None
    O: tropical determinant as VDR or None

        trop_det([[VDR(0), VDR(1)], [VDR(1), VDR(0)]], 2) -> VDR(0)
    """
    from itertools import permutations

    best = None
    for perm in permutations(range(n)):
        weight = VDR(0)
        valid = True
        for i in range(n):
            entry = M[i][perm[i]]
            if entry is None:
                valid = False
                break
            weight = weight + entry
        if valid:
            best = trop_add(best, weight)

    return best


# ---------------------------------------------------------------------------
# Lattice computations
# ---------------------------------------------------------------------------

def gram_matrix(vectors):
    """
    Gram matrix: G[i][j] = vectors[i] . vectors[j].

    I: list of Vec
    O: Mat, exact

        gram_matrix([Vec.from_ints([1,0]), Vec.from_ints([1,1])])
    """
    n = len(vectors)
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(vectors[i].dot(vectors[j]))
        rows.append(row)
    return Mat(rows)


def gram_schmidt_exact(vectors):
    """
    Exact Gram-Schmidt orthogonalization.

    I: list of Vec (basis vectors)
    O: (orthogonalized, mu) where:
       - orthogonalized is list of Vec (Gram-Schmidt vectors b*_i)
       - mu is list-of-lists of VDR (GSO coefficients mu[i][j] for j < i)

    Cross-dot products are exactly 0.
    mu[i][j] = b_i . b*_j / (b*_j . b*_j), exact rational.

        ortho, mu = gram_schmidt_exact([Vec.from_ints([1,1,0]), ...])
    """
    n = len(vectors)
    b_star = []
    mu = [[VDR(0)] * n for _ in range(n)]

    for i in range(n):
        # start with original vector
        v = vectors[i]

        for j in range(i):
            # mu[i][j] = b_i . b*_j / (b*_j . b*_j)
            bsj_norm_sq = b_star[j].dot(b_star[j])
            if bsj_norm_sq == VDR(0):
                mu[i][j] = VDR(0)
                continue
            mu[i][j] = vectors[i].dot(b_star[j]) / bsj_norm_sq

        # b*_i = b_i - sum mu[i][j] * b*_j
        result = vectors[i]
        for j in range(i):
            result = result - b_star[j].scale(mu[i][j])

        b_star.append(result)

    return b_star, mu


def lovasz_condition(b_star, mu, i, delta=None):
    """
    Check Lovasz condition at index i:
        ||b*_i||^2 >= (delta - mu[i][i-1]^2) * ||b*_{i-1}||^2

    I: orthogonalized vectors b_star, mu coefficients, index i,
       delta parameter (VDR, default 3/4)
    O: bool, exact rational comparison

    No float rounding in this comparison. This is where float LLL
    makes wrong decisions on borderline cases.
    """
    if delta is None:
        delta = VDR(3, 4)

    if i < 1 or i >= len(b_star):
        return True

    norm_sq_i = b_star[i].dot(b_star[i])
    norm_sq_prev = b_star[i - 1].dot(b_star[i - 1])
    mu_val = mu[i][i - 1]

    rhs = (delta - mu_val * mu_val) * norm_sq_prev

    return norm_sq_i >= rhs


def size_reduce(vectors, b_star, mu, i, j):
    """
    Size reduction step: ensure |mu[i][j]| <= 1/2.

    If |mu[i][j]| > 1/2, subtract round(mu[i][j]) * b_j from b_i.

    I: basis vectors, GSO vectors, mu coefficients, indices i, j
    O: updated (vectors, mu) — modifies in place and returns
    """
    mu_val = mu[i][j]

    # round to nearest integer
    frac = mu_val.to_fraction()
    # Python's round uses banker's rounding; we want nearest integer
    from fractions import Fraction
    r_int = int(Fraction(frac.numerator + frac.denominator // 2, frac.denominator))
    # more careful: floor(frac + 1/2)
    shifted = frac + Fraction(1, 2)
    r_int = int(shifted.numerator // shifted.denominator)

    if r_int == 0:
        return vectors, mu

    r = VDR(r_int)

    # b_i = b_i - r * b_j
    dim = len(vectors[i])
    new_data = []
    for d in range(dim):
        new_data.append(vectors[i][d] - r * vectors[j][d])
    vectors[i] = Vec(new_data)

    # update mu[i][j]
    mu[i][j] = mu[i][j] - r

    # update mu[i][k] for k < j
    for k in range(j):
        mu[i][k] = mu[i][k] - r * mu[j][k]

    return vectors, mu


def lll_reduce(basis, delta=None):
    """
    LLL lattice basis reduction with exact VDR arithmetic.

    I: list of Vec (basis vectors), delta parameter (VDR, default 3/4)
    O: LLL-reduced list of Vec

    All Gram-Schmidt coefficients exact. Lovasz condition checked
    with exact rational comparison — no float rounding in threshold
    decisions.

        basis = [Vec.from_ints([1, 1, 1]),
                 Vec.from_ints([-1, 0, 2]),
                 Vec.from_ints([3, 5, 6])]
        reduced = lll_reduce(basis)
    """
    if delta is None:
        delta = VDR(3, 4)

    vectors = list(basis)
    n = len(vectors)

    # compute initial GSO
    b_star, mu = gram_schmidt_exact(vectors)

    k = 1
    while k < n:
        # size reduce b_k with respect to b_{k-1}, ..., b_0
        for j in range(k - 1, -1, -1):
            vectors, mu = size_reduce(vectors, b_star, mu, k, j)

        # check Lovasz condition
        if lovasz_condition(b_star, mu, k, delta):
            k += 1
        else:
            # swap b_k and b_{k-1}
            vectors[k], vectors[k - 1] = vectors[k - 1], vectors[k]

            # recompute GSO from scratch (simple but correct)
            b_star, mu = gram_schmidt_exact(vectors)

            k = max(k - 1, 1)

    return vectors
