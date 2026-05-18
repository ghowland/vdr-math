"""
vdr.ml.attention — Exact self-attention mechanism.

    from vdr.ml.attention import self_attention

    outputs = self_attention(Q, K, V, causal=True)
    # every weight row sums to exactly 1

Attention scores exact. Softmax weights sum to exactly 1.
Weighted sums exact. No float drift in long sequences.
All mask fill values projected to basis frame to avoid D mixing.
"""

from __future__ import annotations
from typing import List, Sequence, Optional

from vdr.core import VDR
from vdr.linalg import Vec, Mat
from vdr.ml.softmax import softmax

__all__ = [
    "attention_scores",
    "causal_mask",
    "apply_boolean_mask",
    "attention_weights",
    "weighted_sum",
    "attention_mix",
    "self_attention",
    "multi_head_split",
    "multi_head_concat",
]


def _basis_fill_value(fill=None):
    """
    Return a fill value in basis frame for masking.
    Default is -1000 projected to basis.
    """
    from vdr.basis import to_qbasis
    if fill is None:
        return to_qbasis(VDR(-1000))
    if isinstance(fill, VDR):
        return to_qbasis(fill)
    return to_qbasis(VDR(fill))


def attention_scores(Q, K):
    """
    Compute raw attention score matrix: S[i,j] = Q[i] . K[j].

    I: Q, K as lists of Vec (sequence of query/key vectors)
    O: list of Vec (score rows), forming the score matrix

        Q = [Vec.from_ints([1, 0]), Vec.from_ints([0, 1])]
        K = [Vec.from_ints([1, 1]), Vec.from_ints([1, -1])]
        scores = attention_scores(Q, K)
    """
    n_q = len(Q)
    n_k = len(K)
    rows = []
    for i in range(n_q):
        row = []
        for j in range(n_k):
            row.append(Q[i].dot(K[j]))
        rows.append(Vec(row))
    return rows


def causal_mask(n):
    """
    Lower-triangular boolean mask for causal (autoregressive) attention.

    mask[i][j] = True if j <= i, False otherwise.

    I: sequence length n
    O: list of lists of bool

        causal_mask(3) -> [[True,False,False],[True,True,False],[True,True,True]]
    """
    mask = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(j <= i)
        mask.append(row)
    return mask


def apply_boolean_mask(scores, mask, fill=None):
    """
    Apply boolean mask to attention scores.

    Where mask[i][j] is False, replace score with fill value
    (very negative for softmax to produce ~0 weight).
    Fill value is projected to basis frame once.

    I: score rows (list of Vec), mask (list of lists of bool),
       fill value (VDR, default -1000 in basis)
    O: masked score rows (list of Vec)

        masked = apply_boolean_mask(scores, causal_mask(n))
    """
    basis_fill = _basis_fill_value(fill)

    result = []
    for i in range(len(scores)):
        row = []
        for j in range(len(scores[i])):
            if mask[i][j]:
                row.append(scores[i][j])
            else:
                row.append(basis_fill)
        result.append(Vec(row))
    return result


def attention_weights(scores, mask=None, exp_depth=16):
    """
    Convert scores to attention weights via softmax.

    Optionally applies causal mask before softmax.

    I: score rows (list of Vec), optional mask, exp depth
    O: weight rows (list of Vec), each row sums to exactly 1

        weights = attention_weights(scores, mask=causal_mask(n))
    """
    if mask is not None:
        scores = apply_boolean_mask(scores, mask)

    return [softmax(row, exp_depth) for row in scores]


def weighted_sum(weights, values):
    """
    Weighted sum of value vectors: out = sum_j w_j * V[j].

    I: weights as Vec, values as list of Vec
    O: Vec, exact

        ws = Vec([VDR(1,2), VDR(1,2)])
        vs = [Vec.from_ints([1, 0]), Vec.from_ints([0, 1])]
        weighted_sum(ws, vs)  # Vec([VDR(1,2), VDR(1,2)])
    """
    if len(weights) != len(values):
        raise ValueError("weights length %d != values length %d" % (
            len(weights), len(values)))

    dim = len(values[0])
    result = [VDR(0)] * dim

    for j in range(len(values)):
        w = weights[j]
        for d in range(dim):
            result[d] = result[d] + w * values[j][d]

    return Vec(result)


def attention_mix(weight_rows, V):
    """
    Apply attention weights to value vectors.

    output[i] = sum_j weight_rows[i][j] * V[j]

    I: weight_rows (list of Vec), V (list of Vec)
    O: list of Vec (attended outputs)

        outputs = attention_mix(weights, V)
    """
    return [weighted_sum(w_row, V) for w_row in weight_rows]


def self_attention(Q, K, V, causal=False, exp_depth=16):
    """
    Full self-attention pipeline.

    1. Compute scores: S[i,j] = Q[i] . K[j]
    2. Optionally apply causal mask
    3. Softmax each row (weights sum to exactly 1)
    4. Weighted sum of values

    I: Q, K, V as lists of Vec, causal flag, exp depth
    O: list of Vec (attended outputs)

        Q = [Vec.from_ints([1, 0]), Vec.from_ints([0, 1])]
        K = [Vec.from_ints([1, 1]), Vec.from_ints([1, -1])]
        V = [Vec.from_ints([10, 0]), Vec.from_ints([0, 10])]
        out = self_attention(Q, K, V, causal=True)
    """
    scores = attention_scores(Q, K)

    mask = None
    if causal:
        mask = causal_mask(len(Q))

    weights = attention_weights(scores, mask=mask, exp_depth=exp_depth)

    return attention_mix(weights, V)


def multi_head_split(vecs, n_heads):
    """
    Split vectors into multiple attention heads.

    Each Vec of dimension d is split into n_heads Vecs of dimension d // n_heads.

    I: list of Vec, number of heads
    O: list of lists — outer list is heads, inner is sequence

        heads = multi_head_split([Vec.from_ints([1,2,3,4])], 2)
    """
    if not vecs:
        return [[] for _ in range(n_heads)]

    dim = len(vecs[0])
    if dim % n_heads != 0:
        raise ValueError("dim %d not divisible by n_heads %d" % (dim, n_heads))

    head_dim = dim // n_heads
    heads = [[] for _ in range(n_heads)]

    for v in vecs:
        for h in range(n_heads):
            start = h * head_dim
            end = start + head_dim
            head_vec = Vec([v[i] for i in range(start, end)])
            heads[h].append(head_vec)

    return heads


def multi_head_concat(heads):
    """
    Concatenate attention head outputs back into full vectors.

    I: list of lists — outer is heads, inner is sequence of Vec
    O: list of Vec with concatenated dimensions

        full = multi_head_concat(heads)
    """
    if not heads or not heads[0]:
        return []

    seq_len = len(heads[0])
    result = []

    for pos in range(seq_len):
        parts = []
        for h in range(len(heads)):
            for x in heads[h][pos]:
                parts.append(x)
        result.append(Vec(parts))

    return result
