"""
vdr.ml.metrics — Exact evaluation metrics.

    from vdr.ml.metrics import exact_accuracy, denominator_complexity_vec

    acc = exact_accuracy([0, 1, 2], [0, 1, 1])
    # VDR(2, 3) = 66.7% accuracy, exact rational

    complexity = denominator_complexity_vec(v)
    # (distinct_denoms, sum_denoms, node_count)

Accuracy as exact fraction. Denominator complexity as structural metric.
"""

from __future__ import annotations
from typing import Sequence, Tuple, List

from vdr.core import VDR
from vdr.linalg import Vec, Mat

__all__ = [
    "exact_accuracy",
    "argmax_vec",
    "denominator_complexity_vec",
    "denominator_complexity_mat",
    "parameter_denominator_complexity",
    "total_parameters",
    "mean_loss",
]


def exact_accuracy(pred_ids, target_ids):
    """
    Classification accuracy as exact rational.

    I: list of predicted class ids, list of target class ids
    O: accuracy as VDR (correct / total)

        exact_accuracy([0, 1, 2, 1], [0, 1, 1, 1])
        -> VDR(3, 4)
    """
    if len(pred_ids) != len(target_ids):
        raise ValueError("Length mismatch")
    if not pred_ids:
        return VDR(0)

    correct = sum(1 for p, t in zip(pred_ids, target_ids) if p == t)
    return VDR(correct, len(pred_ids))


def argmax_vec(v):
    """
    Index of maximum element in Vec.

    I: Vec
    O: integer index

        argmax_vec(Vec([VDR(1,3), VDR(1,2), VDR(1,4)])) -> 1
    """
    best = 0
    for i in range(1, len(v)):
        if v[i] > v[best]:
            best = i
    return best


def denominator_complexity_vec(v):
    """
    Denominator complexity of a Vec.

    I: Vec
    O: (distinct_denoms, sum_denoms, node_count) tuple

    Measures structural complexity of the VDR representation.
    Lower is simpler. Used to track model complexity during training.

        dc = denominator_complexity_vec(Vec([VDR(1,2), VDR(1,3)]))
        # (2, 5, 2)
    """
    distinct = set()
    total = 0
    count = 0
    for i in range(len(v)):
        dc = v[i].den_complexity()
        # dc is (unique, sum, count) for that element
        distinct.add(abs(v[i].d))
        total += abs(v[i].d)
        count += 1
    return (len(distinct), total, count)


def denominator_complexity_mat(m):
    """
    Denominator complexity of a Mat.

    I: Mat
    O: (distinct_denoms, sum_denoms, node_count) tuple
    """
    distinct = set()
    total = 0
    count = 0
    for i in range(m.nrows):
        for j in range(m.ncols):
            distinct.add(abs(m[i, j].d))
            total += abs(m[i, j].d)
            count += 1
    return (len(distinct), total, count)


def parameter_denominator_complexity(params):
    """
    Aggregate denominator complexity across all model parameters.

    I: list of VecParam or MatParam
    O: (total_distinct, total_sum, total_count) tuple

        complexity = parameter_denominator_complexity(model.parameters())
    """
    all_denoms = set()
    total_sum = 0
    total_count = 0

    for p in params:
        if hasattr(p.value, '_rows'):
            # Mat
            for i in range(p.value.nrows):
                for j in range(p.value.ncols):
                    val = p.value[i, j]
                    all_denoms.add(abs(val.d))
                    total_sum += abs(val.d)
                    total_count += 1
        else:
            # Vec
            for i in range(len(p.value)):
                val = p.value[i]
                all_denoms.add(abs(val.d))
                total_sum += abs(val.d)
                total_count += 1

    return (len(all_denoms), total_sum, total_count)


def total_parameters(params):
    """
    Count total number of scalar parameters.

    I: list of VecParam or MatParam
    O: integer count

        n = total_parameters(model.parameters())
    """
    count = 0
    for p in params:
        if hasattr(p.value, '_rows'):
            count += p.value.nrows * p.value.ncols
        else:
            count += len(p.value)
    return count


def mean_loss(losses):
    """
    Mean of a list of loss values.

    I: list of VDR loss values
    O: mean as VDR, exact rational

        avg = mean_loss([VDR(1,3), VDR(1,2), VDR(1,4)])
    """
    if not losses:
        return VDR(0)
    total = VDR(0)
    for l in losses:
        total = total + l
    return total / VDR(len(losses))
