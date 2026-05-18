"""
examples/toy_llm/attention_backward.py

Backward through self-attention with surrogate softmax.
Uses raw VDR arithmetic (not qb_*) since backward pass
produces gradients that get rebased in train.py before optimizer step.
"""

from vdr.core import VDR
from vdr.linalg import Vec
from frames import zero_vec


def surrogate_softmax_backward(grad_probs, probs, shifted):
    """
    Backward through surrogate softmax.

    Forward: shifted[i] = z[i]-m+c, sq[i] = shifted[i]^2,
    total = sum(sq), probs[i] = sq[i]/total.

    Backward:
      grad_sq[j] = (grad_probs[j] - dot(grad_probs, probs)) / total
      grad_shifted[j] = grad_sq[j] * 2 * shifted[j]
      grad_z[j] = grad_shifted[j]

    I: grad_probs Vec, probs Vec, shifted Vec
    O: grad_scores Vec
    """
    n = len(probs)

    total = VDR(0)
    for i in range(n):
        total = total + shifted[i] * shifted[i]

    if total == VDR(0):
        return Vec([VDR(0)] * n)

    dot_gp = VDR(0)
    for i in range(n):
        dot_gp = dot_gp + grad_probs[i] * probs[i]

    result = []
    for j in range(n):
        grad_sq_j = (grad_probs[j] - dot_gp) / total
        grad_shifted_j = grad_sq_j * shifted[j] * VDR(2)
        result.append(grad_shifted_j)

    return Vec(result)


def attention_mix_backward(grad_out, weights, V):
    """
    Backward through weighted sum: output[i] = sum_j w[i][j]*V[j]

    I: grad_out list[Vec], weights list[Vec], V list[Vec]
    O: (grad_weights list[Vec], grad_V list[Vec])
    """
    n_q = len(grad_out)
    n_k = len(V)
    dim = len(V[0])

    grad_weights = []
    for i in range(n_q):
        row = []
        for j in range(n_k):
            row.append(grad_out[i].dot(V[j]))
        grad_weights.append(Vec(row))

    grad_V = [zero_vec(dim) for _ in range(n_k)]
    for j in range(n_k):
        for i in range(n_q):
            w_ij = weights[i][j]
            if w_ij != VDR(0):
                contribution = grad_out[i].scale(w_ij)
                grad_V[j] = grad_V[j] + contribution

    return grad_weights, grad_V


def score_backward(grad_scores, Q, K):
    """
    Backward through S[i][j] = Q[i].K[j]

    I: grad_scores list[Vec], Q list[Vec], K list[Vec]
    O: (grad_Q list[Vec], grad_K list[Vec])
    """
    n_q = len(Q)
    n_k = len(K)
    dim = len(Q[0])

    grad_Q = [zero_vec(dim) for _ in range(n_q)]
    grad_K = [zero_vec(dim) for _ in range(n_k)]

    for i in range(n_q):
        for j in range(n_k):
            g = grad_scores[i][j]
            if g != VDR(0):
                grad_Q[i] = grad_Q[i] + K[j].scale(g)
                grad_K[j] = grad_K[j] + Q[i].scale(g)

    return grad_Q, grad_K


def attention_backward(grad_projected, Q, K, V, weights, shifted,
                       Wq, Wk, Wv, Wo, xs_input):
    """
    Full backward through single-head causal self-attention.

    I: grad_projected list[Vec], cached Q/K/V/weights/shifted,
       Linear layers Wq/Wk/Wv/Wo, original input xs_input
    O: grad_input list[Vec]
    S: accumulates into layer parameter gradients
    """
    n = len(grad_projected)
    dim = len(V[0])

    # backward through Wo
    grad_attn_out = []
    for i in range(n):
        attn_out_i = zero_vec(dim)
        for j in range(len(V)):
            w_ij = weights[i][j]
            if w_ij != VDR(0):
                attn_out_i = attn_out_i + V[j].scale(w_ij)
        Wo._last_input = attn_out_i
        g = Wo.backward(grad_projected[i])
        grad_attn_out.append(g)

    # backward through mix
    grad_weights, grad_V = attention_mix_backward(grad_attn_out, weights, V)

    # backward through softmax
    grad_scores = []
    for i in range(n):
        gs = surrogate_softmax_backward(grad_weights[i], weights[i], shifted[i])
        grad_scores.append(gs)

    # backward through scores
    grad_Q, grad_K = score_backward(grad_scores, Q, K)

    # backward through projections
    grad_input_from_V = [zero_vec(len(xs_input[0])) for _ in range(n)]
    grad_input_from_Q = [zero_vec(len(xs_input[0])) for _ in range(n)]
    grad_input_from_K = [zero_vec(len(xs_input[0])) for _ in range(n)]

    for i in range(n):
        Wv._last_input = xs_input[i]
        grad_input_from_V[i] = Wv.backward(grad_V[i])

        Wq._last_input = xs_input[i]
        grad_input_from_Q[i] = Wq.backward(grad_Q[i])

        Wk._last_input = xs_input[i]
        grad_input_from_K[i] = Wk.backward(grad_K[i])

    grad_input = [
        grad_input_from_Q[i] + grad_input_from_K[i] + grad_input_from_V[i]
        for i in range(n)
    ]
    return grad_input
