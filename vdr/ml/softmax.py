"""
vdr.ml.softmax — Exact softmax and logsumexp.

    from vdr.ml.softmax import softmax

    probs = softmax(Vec.from_ints([1, 2, 3]))
    # sums to exactly 1

Uses max-subtraction for stability (exact, no overflow).
Exp via Taylor series at configurable depth.
"""

from __future__ import annotations
from typing import Optional, List

from vdr.core import VDR
from vdr.linalg import Vec, Mat

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


def _exp(x, depth):
    """Taylor exp inlined to avoid circular imports."""
    total = VDR(1)
    term = VDR(1)
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
    m = _vec_max(logits)
    shifted = Vec([x - m for x in logits])

    exps = [_exp(shifted[i], exp_depth) for i in range(len(shifted))]

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

    m = _vec_max(logits)
    shifted = Vec([x - m for x in logits])

    total = VDR(0)
    for i in range(len(shifted)):
        total = total + _exp(shifted[i], exp_depth)

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


def softmax_surrogate_square(logits):
    """
    Polynomial softmax surrogate using squared values.

    p_i = x_i^2 / sum(x_j^2)

    Avoids exp entirely. Still sums to exactly 1.
    Less accurate approximation to true softmax but useful when
    exp is too expensive.

    I: logits as Vec
    O: probability Vec, sums to exactly 1

        softmax_surrogate_square(Vec.from_ints([1, 2, 3]))
    """
    squares = [x * x for x in logits]
    total = VDR(0)
    for s in squares:
        total = total + s

    if total == VDR(0):
        n = len(logits)
        return Vec([VDR(1, n)] * n)

    return Vec([s / total for s in squares])
