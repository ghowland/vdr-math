"""
examples/toy_llm/attention_backward.py

Backward pass through self-attention with surrogate softmax.

Surrogate softmax: s_i = (z_i - m + c)^2 / sum((z_j - m + c)^2)
Backward through square-and-normalize is simpler than Taylor exp.
Every gradient is exact VDR rational.
"""

from vdr.core import VDR
from vdr.linalg import Vec

from frames import zero_vec


def surrogate_softmax_backward(grad_probs, probs, shifted):
    """
    Backward through surrogate softmax.

    Forward was:
      shifted[i] = z[i] - m + c
      sq[i] = shifted[i]^2
      total = sum(sq)
      probs[i] = sq[i] / total

    This is f(z) = g(z)^2 / sum(g(z)^2) where g(z) = z - m + c.

    Using quotient rule on the normalization:
      d(probs[i])/d(z[j]):
        if i == j: (2*shifted[i]*total - sq[i]*2*shifted[i]) / total^2
                 = 2*shifted[i] * (total - sq[i]) / total^2
                 = 2*shifted[i] * probs[i] * (1 - probs[i]) / sq[i]  ... complex
        if i != j: (0 - sq[i]*2*shifted[j]) / total^2
                 = -2*shifted[j] * probs[i] / total

    Simpler formulation via chain rule through two stages:
      Stage 1: probs = sq / total  (normalization)
      Stage 2: sq = shifted^2      (squaring)
      Stage 3: shifted = z - m + c (shift, but m depends on z)

    For normalization backward (same structure as softmax Jacobian):
      grad_sq[i] = (grad_probs[i] - dot(grad_probs, probs)) * probs[i] / sq[i]
    But sq[i] = probs[i] * total, so probs[i]/sq[i] = 1/total:
      grad_sq[i] = (grad_probs[i] - dot(grad_probs, probs)) / total

    Actually, for f_i = sq_i / total:
      df_i/dsq_j = (delta_ij * total - sq_i) / total^2
                  = (delta_ij - f_i) / total

    So: grad_sq[j] = sum_i grad_probs[i] * (delta_ij - probs[i]) / total
                    = (grad_probs[j] - sum_i grad_probs[i] * probs[i]) / total
                    = (grad_probs[j] - dot(grad_probs, probs)) / total

    For squaring backward: sq = shifted^2, so dsq/dshifted = 2*shifted:
      grad_shifted[j] = grad_sq[j] * 2 * shifted[j]

    For shift backward: shifted = z - m + c where m = max(z).
    The max function has gradient 1 for the argmax, 0 elsewhere.
    For simplicity (and because the max-shift is for stability, not
    mathematically essential), we treat shifted as linear in z:
      grad_z[j] = grad_shifted[j]  (subgradient: ignore max derivative)

    I: grad_probs Vec (gradient w.r.t. softmax output)
       probs Vec (softmax output, sums to exactly 1)
       shifted Vec (z - m + c values, cached from forward)
    O: grad_scores Vec (gradient w.r.t. softmax input z)
    """
    n = len(probs)

    # compute total = sum of squared shifted values
    total = VDR(0)
    for i in range(n):
        total = total + shifted[i] * shifted[i]

    if total == VDR(0):
        return Vec([VDR(0)] * n)

    # dot(grad_probs, probs)
    dot_gp = VDR(0)
    for i in range(n):
        dot_gp = dot_gp + grad_probs[i] * probs[i]

    # grad_sq[j] = (grad_probs[j] - dot_gp) / total
    # grad_shifted[j] = grad_sq[j] * 2 * shifted[j]
    # grad_z[j] = grad_shifted[j]
    result = []
    for j in range(n):
        grad_sq_j = (grad_probs[j] - dot_gp) / total
        grad_shifted_j = grad_sq_j * shifted[j] * VDR(2)
        result.append(grad_shifted_j)

    return Vec(result)


def attention_mix_backward(grad_out, weights, V):
    """
    Backward through attention weighted sum.

    Forward: output[i] = sum_j weights[i][j] * V[j]

    Backward:
        grad_weights[i][j] = dot(grad_out[i], V[j])
        grad_V[j] = sum_i weights[i][j] * grad_out[i]

    I: grad_out (list of Vec), weights (list of Vec), V (list of Vec)
    O: (grad_weights, grad_V)
       grad_weights: list of Vec (same shape as weights)
       grad_V: list of Vec (same shape as V)
    """
    n_q = len(grad_out)
    n_k = len(V)
    dim = len(V[0])

    # grad_weights[i][j] = grad_out[i] . V[j]
    grad_weights = []
    for i in range(n_q):
        row = []
        for j in range(n_k):
            row.append(grad_out[i].dot(V[j]))
        grad_weights.append(Vec(row))

    # grad_V[j] = sum_i weights[i][j] * grad_out[i]
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
    Backward through score computation: S[i][j] = Q[i] . K[j]

    Backward:
        grad_Q[i] = sum_j grad_scores[i][j] * K[j]
        grad_K[j] = sum_i grad_scores[i][j] * Q[i]

    I: grad_scores (list of Vec), Q (list of Vec), K (list of Vec)
    O: (grad_Q, grad_K) each list of Vec
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

    I: grad_projected — list of Vec, gradient w.r.t. Wo output
       Q, K, V — cached query/key/value vectors
       weights — cached attention weight rows (each sums to exactly 1)
       shifted — cached shifted values from surrogate softmax (per row)
       Wq, Wk, Wv, Wo — Linear layers
       xs_input — original input to attention block

    O: grad_input — list of Vec, gradient w.r.t. attention block input
    S: accumulates gradients into Wq, Wk, Wv, Wo parameters

    All gradients exact VDR rationals.
    """
    n = len(grad_projected)
    dim = len(V[0])

    # Step 1: backward through Wo
    grad_attn_out = []
    for i in range(n):
        # reconstruct attn_out[i] = sum_j weights[i][j] * V[j]
        attn_out_i = zero_vec(dim)
        for j in range(len(V)):
            w_ij = weights[i][j]
            if w_ij != VDR(0):
                attn_out_i = attn_out_i + V[j].scale(w_ij)

        Wo._last_input = attn_out_i
        g = Wo.backward(grad_projected[i])
        grad_attn_out.append(g)

    # Step 2: backward through attention mix
    grad_weights, grad_V = attention_mix_backward(grad_attn_out, weights, V)

    # Step 3: backward through surrogate softmax (per row)
    grad_scores = []
    for i in range(n):
        gs = surrogate_softmax_backward(grad_weights[i], weights[i], shifted[i])
        grad_scores.append(gs)

    # Step 4: backward through score computation
    grad_Q, grad_K = score_backward(grad_scores, Q, K)

    # Step 5: backward through Wv projections
    grad_input_from_V = [zero_vec(len(xs_input[0])) for _ in range(n)]
    for i in range(n):
        Wv._last_input = xs_input[i]
        g = Wv.backward(grad_V[i])
        grad_input_from_V[i] = g

    # Step 6: backward through Wq projections
    grad_input_from_Q = [zero_vec(len(xs_input[0])) for _ in range(n)]
    for i in range(n):
        Wq._last_input = xs_input[i]
        g = Wq.backward(grad_Q[i])
        grad_input_from_Q[i] = g

    # Step 7: backward through Wk projections
    grad_input_from_K = [zero_vec(len(xs_input[0])) for _ in range(n)]
    for i in range(n):
        Wk._last_input = xs_input[i]
        g = Wk.backward(grad_K[i])
        grad_input_from_K[i] = g

    # Step 8: combine all input gradients
    grad_input = []
    for i in range(n):
        g = grad_input_from_Q[i] + grad_input_from_K[i] + grad_input_from_V[i]
        grad_input.append(g)

    return grad_input

