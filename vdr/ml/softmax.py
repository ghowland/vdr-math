"""
vdr.ml.softmax — Exact softmax and logsumexp.

    from vdr.ml.softmax import softmax

    probs = softmax(Vec.from_ints([1, 2, 3]))
    # sums to exactly 1

Uses max-subtraction for stability (exact, no overflow).
Exp via Taylor series at configurable depth.
All constants projected to basis frame to avoid D mixing.
"""

from __future__ import annotations
from typing import Optional, List

from vdr.core import VDR
from vdr.linalg import Vec, Mat
from vdr.basis import to_qbasis

__all__ = [
    "softmax",
    "logsumexp",
    "softmax_matrix_rows",
    "softmax_surrogate_square",
]


def _to_vdr(x):
    if isinstance(x, VDR):
        return x
    if isinstance(x, int):
        return VDR(x)
    raise TypeError("Expected VDR or int, got %s" % type(x).__name__)


def _build_inv_table(depth):
    """
    Precompute 1/k in basis frame for k=1..depth.
    Called once per exp evaluation, not per element.
    """
    from vdr.basis import to_qbasis
    table = [None]  # index 0 unused
    for k in range(1, depth + 1):
        table.append(to_qbasis(VDR(1, k)))
    return table


def _exp(x, depth, inv_table=None):
    """
    Taylor exp: sum_{k=0}^{depth} x^k / k!

    If inv_table is provided, uses precomputed basis-frame 1/k
    to avoid per-iteration D mixing from VDR(k) division.
    """
    total = VDR(1)
    term = VDR(1)
    if inv_table is not None:
        for k in range(1, depth + 1):
            term = term * x * inv_table[k]
            total = total + term
    else:
        for k in range(1, depth + 1):
            term = term * x / VDR(k)
            total = total + term
    return total


def _vec_max(v):
    """Maximum element of a Vec."""
    best = v[0]
    for i in range(1, len(v)):
        if v[i] > best:
            best = v[i]
    return best


def softmax(logits, exp_depth=16):
    """
    Exact softmax with max-subtraction for numerical stability.

    softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))

    I: logits as Vec, exp Taylor series depth
    O: probability Vec, sums to exactly 1

    Max-subtraction is exact (VDR comparison and subtraction).
    Each exp is exact rational at given depth.
    The sum is exact. Division is exact.

        softmax(Vec.from_ints([1, 2, 3]))
    """
    inv_table = _build_inv_table(exp_depth)

    m = _vec_max(logits)
    shifted = Vec([x - m for x in logits])

    exps = [_exp(shifted[i], exp_depth, inv_table) for i in range(len(shifted))]

    total = VDR(0)
    for e in exps:
        total = total + e

    probs = [e / total for e in exps]
    return Vec(probs)


def logsumexp(logits, exp_depth=16, log_depth=16):
    """
    Exact log-sum-exp: log(sum(exp(x_i))).

    Uses max-subtraction: lse(x) = max(x) + log(sum(exp(x_i - max(x))))

    I: logits as Vec, exp and log depths
    O: VDR scalar

        logsumexp(Vec.from_ints([1, 2, 3]))
    """
    from vdr.math.transcendental import ln_series

    inv_table = _build_inv_table(exp_depth)

    m = _vec_max(logits)
    shifted = Vec([x - m for x in logits])

    total = VDR(0)
    for i in range(len(shifted)):
        total = total + _exp(shifted[i], exp_depth, inv_table)

    return m + ln_series(total, log_depth)


def softmax_matrix_rows(M, exp_depth=16):
    """
    Apply softmax independently to each row of a matrix.

    I: Mat of logits, exp depth
    O: Mat where each row is a probability distribution summing to exactly 1

        weights = softmax_matrix_rows(score_matrix)
    """
    rows = []
    for i in range(M.nrows):
        row_vec = M.row(i)
        prob_vec = softmax(row_vec, exp_depth)
        rows.append(prob_vec)
    return Mat([list(r) for r in rows])


def softmax_surrogate_square(logits, shift=None):
    """
    Polynomial softmax surrogate using squared values.

    p_i = (x_i - shift)^2 / sum((x_j - shift)^2)

    Avoids exp entirely. Still sums to exactly 1.
    Shift parameter centers logits to keep V slots small.
    Default shift is the minimum logit.

    I: logits as Vec, optional shift as VDR
    O: probability Vec, sums to exactly 1

        softmax_surrogate_square(Vec.from_ints([1, 2, 3]))
        softmax_surrogate_square(logits, shift=VDR(0))
    """
    if shift is None:
        shift = logits[0]
        for i in range(1, len(logits)):
            if logits[i] < shift:
                shift = logits[i]

    shifted = [x - shift for x in logits]
    squares = [x * x for x in shifted]

    total = VDR(0)
    for s in squares:
        total = total + s

    if total == VDR(0):
        n = len(logits)
        return Vec([VDR(1, n)] * n)

    probs = []
    running = VDR(0)
    one = to_qbasis(VDR(1))
    for i in range(len(squares) - 1):
        p = squares[i] / total
        probs.append(p)
        running = running + p
    probs.append(one - running)
    return Vec(probs)
