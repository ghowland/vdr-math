"""
Attention backward pass for toy transformer.
Surrogate quadratic softmax backward — no Taylor exp derivatives.
All arithmetic through basis-safe VDR operators.
"""

from vdr.core import VDR
from vdr.linalg import Vec, Mat
from vdr.basis import to_qbasis


def _outer(a, b):
    """Outer product: M[i,j] = a[i] * b[j]."""
    rows = []
    for i in range(len(a)):
        row = []
        for j in range(len(b)):
            row.append(a[i] * b[j])
        rows.append(row)
    return Mat(rows)


def softmax_surrogate_backward(grad_probs, probs, shifted):
    """
    Backward through quadratic softmax surrogate.

    Forward: p_i = s_i^2 / sum(s_j^2)  where s = shifted input
    Backward: dp/ds_i = 2*s_i * (sum(s^2) - s_i^2) / sum(s^2)^2
                       = 2*s_i * (1 - p_i) / sum(s^2)

    I: grad_probs Vec (dL/dp), probs Vec (p), shifted Vec (s)
    O: grad_scores Vec (dL/ds)
    """
    n = len(probs)

    # reconstruct sum of squares from probs and shifted
    sum_sq = VDR(0)
    for i in range(n):
        sum_sq = sum_sq + shifted[i] * shifted[i]

    if sum_sq == VDR(0):
        return Vec([VDR(0)] * n)

    result = []
    for i in range(n):
        # dp_i/ds_i = 2*s_i*(1 - p_i) / sum_sq
        # dp_j/ds_i = -2*s_i*p_j / sum_sq  for j != i
        # dL/ds_i = sum_j (dL/dp_j * dp_j/ds_i)
        #         = dL/dp_i * 2*s_i*(1-p_i)/sum_sq
        #           + sum_{j!=i} dL/dp_j * (-2*s_i*p_j/sum_sq)
        #         = 2*s_i/sum_sq * (dL/dp_i*(1-p_i) - sum_{j!=i} dL/dp_j*p_j)
        #         = 2*s_i/sum_sq * (dL/dp_i - sum_j dL/dp_j*p_j)

        dot_gp = VDR(0)
        for j in range(n):
            dot_gp = dot_gp + grad_probs[j] * probs[j]

        grad_s_i = VDR(2) * shifted[i] / sum_sq * (grad_probs[i] - dot_gp)
        result.append(grad_s_i)

    return Vec(result)


def attention_mix_backward(grad_out, weights, V):
    """
    Backward through weighted sum: out[i] = sum_j w[i][j] * V[j].

    I: grad_out (list of Vec), weights (list of Vec), V (list of Vec)
    O: (grad_weights as list of Vec, grad_V as list of Vec)
    """
    n = len(grad_out)
    n_v = len(V)
    dim = len(V[0])

    # grad_weights[i][j] = grad_out[i] . V[j]
    grad_weights = []
    for i in range(n):
        row = []
        for j in range(n_v):
            row.append(grad_out[i].dot(V[j]))
        grad_weights.append(Vec(row))

    # grad_V[j] = sum_i w[i][j] * grad_out[i]
    grad_V = [Vec([VDR(0)] * dim) for _ in range(n_v)]
    for i in range(n):
        for j in range(n_v):
            scaled = Vec([weights[i][j] * grad_out[i][d] for d in range(dim)])
            grad_V[j] = grad_V[j] + scaled

    return grad_weights, grad_V


def score_backward(grad_scores, Q, K):
    """
    Backward through score computation: S[i][j] = Q[i] . K[j].

    I: grad_scores (list of Vec), Q (list of Vec), K (list of Vec)
    O: (grad_Q as list of Vec, grad_K as list of Vec)
    """
    n_q = len(Q)
    n_k = len(K)
    dim = len(Q[0])

    # grad_Q[i] = sum_j grad_scores[i][j] * K[j]
    grad_Q = []
    for i in range(n_q):
        g = Vec([VDR(0)] * dim)
        for j in range(n_k):
            scaled = Vec([grad_scores[i][j] * K[j][d] for d in range(dim)])
            g = g + scaled
        grad_Q.append(g)

    # grad_K[j] = sum_i grad_scores[i][j] * Q[i]
    grad_K = []
    for j in range(n_k):
        g = Vec([VDR(0)] * dim)
        for i in range(n_q):
            scaled = Vec([grad_scores[i][j] * Q[i][d] for d in range(dim)])
            g = g + scaled
        grad_K.append(g)

    return grad_Q, grad_K


def attention_backward(grad_projected, Q, K, V, weights, shifted,
                       Wq, Wk, Wv, Wo, xs_input):
    """
    Full attention block backward.

    I: grad_projected (list of Vec — grad w.r.t. Wo output),
       Q, K, V (cached projections),
       weights (cached softmax output),
       shifted (cached shifted scores for surrogate backward),
       Wq, Wk, Wv, Wo (Linear layers — grads accumulated),
       xs_input (list of Vec — attention block input)
    O: list of Vec — grad w.r.t. attention block input
    """
    n = len(grad_projected)
    dim = len(Q[0])

    # backward through Wo projection
    grad_attn_out = []
    for i in range(n):
        # Wo.forward was called with attn_out[i], backward gives grad w.r.t. input
        # we need to set Wo._last_input to attn_out[i] for backward
        # but Wo.backward was already called for last position
        # we do manual backward here
        g_in = Wo.weight.value.T.matvec(grad_projected[i])
        grad_attn_out.append(g_in)
        # accumulate Wo grads
        Wo.weight.grad = Wo.weight.grad + _outer(grad_projected[i], V[0])  # placeholder
        Wo.bias.grad = Wo.bias.grad + grad_projected[i]

    # Fix Wo grad: use actual attn_out from cache
    # Reset and recompute properly
    Wo.weight.grad = Mat.zero(Wo.weight.value.nrows, Wo.weight.value.ncols)
    Wo.bias.grad = Vec.zero(Wo.weight.value.nrows)
    attn_out_cache = [None] * n
    for i in range(n):
        # reconstruct attn_out from weights and V
        attn_out_i = Vec([VDR(0)] * dim)
        for j in range(n):
            scaled = Vec([weights[i][j] * V[j][d] for d in range(dim)])
            attn_out_i = attn_out_i + scaled
        attn_out_cache[i] = attn_out_i

        g_in = Wo.weight.value.T.matvec(grad_projected[i])
        grad_attn_out[i] = g_in
        Wo.weight.grad = Wo.weight.grad + _outer(grad_projected[i], attn_out_i)
        Wo.bias.grad = Wo.bias.grad + grad_projected[i]

    # backward through attention mix
    grad_weights, grad_V = attention_mix_backward(grad_attn_out, weights, V)

    # backward through softmax surrogate per row
    grad_scores = []
    for i in range(n):
        g = softmax_surrogate_backward(grad_weights[i], weights[i], shifted[i])
        grad_scores.append(g)

    # backward through score computation
    grad_Q, grad_K = score_backward(grad_scores, Q, K)

    # backward through Wq, Wk, Wv projections
    grad_xs = [Vec([VDR(0)] * dim) for _ in range(n)]

    for i in range(n):
        # Wq backward: grad_Q[i] -> grad w.r.t. xs_input[i]
        g_x_q = Wq.weight.value.T.matvec(grad_Q[i])
        Wq.weight.grad = Wq.weight.grad + _outer(grad_Q[i], xs_input[i])
        Wq.bias.grad = Wq.bias.grad + grad_Q[i]

        # Wk backward
        g_x_k = Wk.weight.value.T.matvec(grad_K[i])
        Wk.weight.grad = Wk.weight.grad + _outer(grad_K[i], xs_input[i])
        Wk.bias.grad = Wk.bias.grad + grad_K[i]

        # Wv backward
        g_x_v = Wv.weight.value.T.matvec(grad_V[i])
        Wv.weight.grad = Wv.weight.grad + _outer(grad_V[i], xs_input[i])
        Wv.bias.grad = Wv.bias.grad + grad_V[i]

        grad_xs[i] = g_x_q + g_x_k + g_x_v

    return grad_xs
