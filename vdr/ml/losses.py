"""
vdr.ml.losses — Exact loss functions.

    from vdr.ml.losses import mse, l1, mse_grad

    loss = mse(pred, target)      # exact MSE as VDR
    grad = mse_grad(pred, target) # exact gradient as Vec

All losses exact VDR rationals. Gradients exact via closed-form expressions.
"""

from __future__ import annotations

from vdr.core import VDR
from vdr.linalg import Vec

__all__ = [
    "mse",
    "l1",
    "hinge_binary",
    "mse_grad",
    "l1_grad",
    "cross_entropy_binary",
]


def _to_vdr(x):
    if isinstance(x, VDR):
        return x
    if isinstance(x, int):
        return VDR(x)
    raise TypeError("Expected VDR or int")


def mse(pred, target):
    """
    Mean squared error: (1/n) * sum (pred_i - target_i)^2.

    I: pred Vec, target Vec (same dimension)
    O: MSE as VDR, exact

        mse(Vec.from_ints([1, 2, 3]), Vec.from_ints([1, 2, 4]))
        -> VDR(1, 3)
    """
    if len(pred) != len(target):
        raise ValueError("Dimension mismatch: %d vs %d" % (len(pred), len(target)))
    n = len(pred)
    total = VDR(0)
    for i in range(n):
        diff = pred[i] - target[i]
        total = total + diff * diff
    return total / VDR(n)


def l1(pred, target):
    """
    Mean absolute error: (1/n) * sum |pred_i - target_i|.

    I: pred Vec, target Vec
    O: L1 loss as VDR, exact
    """
    if len(pred) != len(target):
        raise ValueError("Dimension mismatch")
    n = len(pred)
    total = VDR(0)
    for i in range(n):
        total = total + abs(pred[i] - target[i])
    return total / VDR(n)


def hinge_binary(score, label):
    """
    Binary hinge loss: max(0, 1 - label * score).

    I: score (VDR), label (int, +1 or -1)
    O: hinge loss as VDR

        hinge_binary(VDR(3, 2), 1) -> VDR(0)  # margin > 1
        hinge_binary(VDR(1, 2), 1) -> VDR(1, 2)
    """
    score = _to_vdr(score)
    margin = VDR(1) - VDR(label) * score
    if margin > VDR(0):
        return margin
    return VDR(0)


def mse_grad(pred, target):
    """
    Gradient of MSE with respect to pred.

    d/d(pred_i) [(1/n) sum (pred_j - target_j)^2] = (2/n) * (pred_i - target_i)

    I: pred Vec, target Vec
    O: gradient Vec, exact

        grad = mse_grad(Vec.from_ints([1, 2, 3]), Vec.from_ints([1, 2, 4]))
    """
    if len(pred) != len(target):
        raise ValueError("Dimension mismatch")
    n = len(pred)
    two_over_n = VDR(2, n)
    result = []
    for i in range(n):
        result.append(two_over_n * (pred[i] - target[i]))
    return Vec(result)


def l1_grad(pred, target):
    """
    Subgradient of L1 loss with respect to pred.

    d/d(pred_i) |pred_i - target_i| = sign(pred_i - target_i) / n

    I: pred Vec, target Vec
    O: subgradient Vec

    At pred_i == target_i, returns 0 (subgradient choice).
    """
    if len(pred) != len(target):
        raise ValueError("Dimension mismatch")
    n = len(pred)
    result = []
    for i in range(n):
        diff = pred[i] - target[i]
        if diff > VDR(0):
            result.append(VDR(1, n))
        elif diff < VDR(0):
            result.append(VDR(-1, n))
        else:
            result.append(VDR(0))
    return Vec(result)


def cross_entropy_binary(pred_prob, label, log_depth=16):
    """
    Binary cross-entropy loss: -[y*ln(p) + (1-y)*ln(1-p)].

    I: predicted probability pred_prob (VDR in (0,1)),
       true label (VDR, 0 or 1), log depth
    O: loss as VDR

    Uses exact log series.
    """
    from vdr.ml.logarithm import log_series

    pred_prob = _to_vdr(pred_prob)
    label = _to_vdr(label)

    log_p = log_series(pred_prob, log_depth)
    log_1_minus_p = log_series(VDR(1) - pred_prob, log_depth)

    return -(label * log_p + (VDR(1) - label) * log_1_minus_p)
