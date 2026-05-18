"""
vdr.ml.sampling — Exact sampling from probability distributions.

    from vdr.ml.sampling import categorical_sample, top_k_probs, nucleus_probs

    rng = VDRRandom(seed=42)
    probs = softmax(logits)
    idx = categorical_sample(probs, rng)  # exact rational CDF comparison

All CDF values exact. Sampling decisions are exact comparisons,
not float threshold tests.
"""

from __future__ import annotations
from typing import List

from vdr.core import VDR
from vdr.linalg import Vec

__all__ = [
    "cdf_from_probs",
    "categorical_sample",
    "top_k_probs",
    "nucleus_probs",
    "argmax_sample",
    "temperature_scale",
]


def _to_vdr(x):
    if isinstance(x, VDR):
        return x
    if isinstance(x, int):
        return VDR(x)
    raise TypeError("Expected VDR or int")


def cdf_from_probs(probs):
    """
    Cumulative distribution function from probability vector.

    I: probability Vec (should sum to exactly 1)
    O: CDF Vec where cdf[i] = sum_{j=0}^{i} probs[j]

    Final entry is exactly 1 when input sums to exactly 1.

        probs = Vec([VDR(1,4), VDR(1,2), VDR(1,4)])
        cdf = cdf_from_probs(probs)
        # Vec([VDR(1,4), VDR(3,4), VDR(1)])
    """
    data = []
    total = VDR(0)
    for i in range(len(probs)):
        total = total + probs[i]
        data.append(total)
    return Vec(data)


def categorical_sample(probs, rng):
    """
    Sample from a categorical distribution using exact CDF comparison.

    Generates a uniform random rational in [0, 1) and finds the first
    CDF bin that exceeds it.

    I: probability Vec (sums to exactly 1), VDRRandom instance
    O: integer index of sampled category

    The comparison is exact rational — no float threshold ambiguity.

        from vdr.ml.rng import VDRRandom
        rng = VDRRandom(seed=42)
        idx = categorical_sample(probs, rng)
    """
    cdf = cdf_from_probs(probs)
    u = rng.rand_fraction()

    for i in range(len(cdf)):
        if u < cdf[i]:
            return i

    # fallback: return last index (should not happen if probs sum to 1)
    return len(probs) - 1


def argmax_sample(probs):
    """
    Deterministic argmax sampling (greedy decoding).

    I: probability or logit Vec
    O: integer index of maximum element

        argmax_sample(Vec([VDR(1,4), VDR(1,2), VDR(1,4)])) -> 1
    """
    best_idx = 0
    best_val = probs[0]
    for i in range(1, len(probs)):
        if probs[i] > best_val:
            best_val = probs[i]
            best_idx = i
    return best_idx


def top_k_probs(probs, k):
    """
    Top-k filtering: keep only the k largest probabilities, renormalize.

    I: probability Vec, k (number of top entries to keep)
    O: filtered probability Vec (sums to exactly 1)

    Non-top-k entries set to zero. Remaining entries renormalized
    by exact division by their sum.

        filtered = top_k_probs(Vec([VDR(1,10), VDR(6,10), VDR(3,10)]), 2)
        # VDR(1,10) -> VDR(0), renormalize [VDR(6,10), VDR(3,10)]
    """
    n = len(probs)
    if k >= n:
        return probs

    # find top-k indices
    indexed = [(probs[i], i) for i in range(n)]
    indexed.sort(key=lambda x: x[0].to_fraction(), reverse=True)

    top_indices = set()
    for i in range(k):
        top_indices.add(indexed[i][1])

    # zero out non-top-k and renormalize
    filtered = []
    total = VDR(0)
    for i in range(n):
        if i in top_indices:
            filtered.append(probs[i])
            total = total + probs[i]
        else:
            filtered.append(VDR(0))

    if total == VDR(0):
        return Vec(filtered)

    return Vec([f / total if f != VDR(0) else VDR(0) for f in filtered])


def nucleus_probs(probs, threshold):
    """
    Nucleus (top-p) sampling: keep smallest set of tokens whose
    cumulative probability exceeds threshold.

    I: probability Vec, threshold (VDR, e.g. VDR(9, 10) for top-90%)
    O: filtered probability Vec (sums to exactly 1)

    The threshold comparison is exact rational.

        filtered = nucleus_probs(probs, VDR(9, 10))
    """
    threshold = _to_vdr(threshold)
    n = len(probs)

    # sort by probability descending
    indexed = [(probs[i], i) for i in range(n)]
    indexed.sort(key=lambda x: x[0].to_fraction(), reverse=True)

    cumsum = VDR(0)
    keep_indices = set()

    for prob, idx in indexed:
        keep_indices.add(idx)
        cumsum = cumsum + prob
        if cumsum >= threshold:
            break

    # zero out excluded, renormalize
    filtered = []
    total = VDR(0)
    for i in range(n):
        if i in keep_indices:
            filtered.append(probs[i])
            total = total + probs[i]
        else:
            filtered.append(VDR(0))

    if total == VDR(0):
        return Vec(filtered)

    return Vec([f / total if f != VDR(0) else VDR(0) for f in filtered])


def temperature_scale(logits, temperature):
    """
    Scale logits by temperature before softmax.

    I: logits Vec, temperature (VDR, > 0)
    O: scaled logits Vec

    temperature < 1: sharper distribution (more greedy)
    temperature > 1: flatter distribution (more random)
    temperature = 1: unchanged

        scaled = temperature_scale(logits, VDR(1, 2))  # temperature 0.5
    """
    temperature = _to_vdr(temperature)
    return Vec([x / temperature for x in logits])
