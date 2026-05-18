"""
vdr.ml.losses — Exact loss functions.

    from vdr.ml.losses import mse, l1, mse_grad

    loss = mse(pred, target)      # exact MSE as VDR
    grad = mse_grad(pred, target) # exact gradient as Vec

All losses exact VDR rationals. Gradients exact via closed-form expressions.
Constants projected to basis frame to avoid D mixing in hot paths.
"""

from __future__ import annotations

from vdr.core import VDR
from vdr.linalg import Vec

from vdr.basis import to_qbasis

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


def _basis_const(v, d=1):
    """Project a rational constant to basis frame."""
    from vdr.basis import to_qbasis
    return to_qbasis(VDR(v, d))


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
    inv_n = _basis_const(1, n)
    total = to_qbasis(0)
    for i in range(n):
        diff = pred[i] - target[i]
        total = total + diff * diff
    return total * inv_n


def l1(pred, target):
    """
    Mean absolute error: (1/n) * sum |pred_i - target_i|.

    I: pred Vec, target Vec
    O: L1 loss as VDR, exact
    """
    if len(pred) != len(target):
        raise ValueError("Dimension mismatch")
    n = len(pred)
    inv_n = _basis_const(1, n)
    total = to_qbasis(0)
    for i in range(n):
        total = total + abs(pred[i] - target[i])
    return total * inv_n


def hinge_binary(score, label):
    """
    Binary hinge loss: max(0, 1 - label * score).

    I: score (VDR), label (int, +1 or -1)
    O: hinge loss as VDR

        hinge_binary(VDR(3, 2), 1) -> to_qbasis(0)
        hinge_binary(VDR(1, 2), 1) -> VDR(1, 2)
    """
    score = _to_vdr(score)
    one = _basis_const(1)
    lab = _basis_const(label)
    margin = one - lab * score
    if margin > to_qbasis(0):
        return margin
    return to_qbasis(0)


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
    two_over_n = _basis_const(2, n)
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
    pos = _basis_const(1, n)
    neg = _basis_const(-1, n)
    zero = to_qbasis(0)
    result = []
    for i in range(n):
        diff = pred[i] - target[i]
        if diff > zero:
            result.append(pos)
        elif diff < zero:
            result.append(neg)
        else:
            result.append(zero)
    return Vec(result)


def cross_entropy_binary(pred_prob, label, log_depth=16):
    """
    Binary cross-entropy loss: -[y*ln(p) + (1-y)*ln(1-p)].

    I: predicted probability pred_prob (VDR in (0,1)),
       true label (VDR, 0 or 1), log depth
    O: loss as VDR

    Uses exact log series.
    """
    from vdr.math.transcendental import ln_series

    pred_prob = _to_vdr(pred_prob)
    label = _to_vdr(label)
    one = _basis_const(1)

    log_p = ln_series(pred_prob, log_depth)
    log_1_minus_p = ln_series(one - pred_prob, log_depth)

    return -(label * log_p + (one - label) * log_1_minus_p)
