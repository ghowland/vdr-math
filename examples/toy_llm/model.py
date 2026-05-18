"""
examples/toy_llm/model.py

Toy LLM — 1-layer decoder-only transformer.
All arithmetic via qb_* functions. D never compounds.

    from model import ToyTransformer
    model = ToyTransformer()
    logits = model.forward([0, 1, 2, 3])
"""

from vdr.core import VDR
from vdr.linalg import Vec, Mat
from vdr.ml.nn import Linear, ReLU

from frames import (
    qb_dot, qb_matvec, qb_vec_add, qb_vec_sub, qb_vec_scale,
    qb_linear_forward, qb_mul, qb_add,
    rebase_vec, rebase_params, rebase_grads,
    init_weight_mat, init_bias_vec, zero_vec, zero_vec_in_frame,
)
import config as cfg


# -- surrogate softmax ----------------------------------------------------

def surrogate_softmax(scores, c=None):
    """
    Rational surrogate: s_i = (z_i - m + c)^2 / sum((z_j - m + c)^2)

    No transcendentals. Pure integer arithmetic. Sums to exactly 1.
    All intermediate arithmetic in BITS_PROB frame.

    I: scores Vec, c shift constant
    O: (probs Vec summing to exactly 1, shifted Vec for backward)
    """
    if c is None:
        c = cfg.SURR_C
    bits = cfg.BITS_PROB
    n = len(scores)

    # find max (exact comparison)
    m = scores[0]
    for i in range(1, n):
        if scores[i] > m:
            m = scores[i]

    # shift: z_i - m + c
    shifted = []
    for i in range(n):
        s = qb_add(scores[i], VDR(-m.v, m.d, m.r), bits=bits)
        s = qb_add(s, c, bits=bits)
        shifted.append(s)
    shifted = Vec(shifted)

    # square each and sum
    squared = [qb_mul(shifted[i], shifted[i], bits=bits) for i in range(n)]
    total = VDR(0, 2 ** bits)
    for s in squared:
        total = qb_add(total, s, bits=bits)

    # normalize
    if total == VDR(0):
        probs = Vec([VDR(1, n) for _ in range(n)])
        return probs, shifted

    # probs[i] = squared[i] / total — this division may not stay in frame,
    # but the result is a proper probability so we just compute it
    probs_list = []
    for i in range(n):
        if squared[i] == VDR(0):
            probs_list.append(VDR(0))
        else:
            p = squared[i] / total
            probs_list.append(p)
    probs = Vec(probs_list)

    return probs, shifted


def surrogate_softmax_probs(scores, c=None):
    """Surrogate softmax returning only probabilities."""
    probs, _ = surrogate_softmax(scores, c)
    return probs


# -- causal mask ----------------------------------------------------------

def causal_mask_value():
    """Large negative for masking future positions."""
    return VDR(-1000)


def apply_causal_mask(scores_rows, seq_len):
    """
    Apply causal mask to score rows.
    scores_rows[i] is list of VDR for row i.

    I: list of list of VDR, seq_len
    O: list of list of VDR (masked)
    """
    mask_val = causal_mask_value()
    result = []
    for i in range(seq_len):
        row = []
        for j in range(seq_len):
            if j > i:
                row.append(mask_val)
            else:
                row.append(scores_rows[i][j])
        result.append(row)
    return result


# -- model ----------------------------------------------------------------

class ToyTransformer:
    """
    1-layer decoder-only transformer. All arithmetic in fixed D-frames.
    D never compounds. 226 parameters, all small-integer rationals.
    """

    def __init__(self, seed=None):
        if seed is None:
            seed = cfg.SEED

        dim = cfg.DIM
        ffn_dim = cfg.FFN_DIM
        vocab_size = cfg.VOCAB_SIZE
        seq_len = cfg.SEQ_LEN
        d_w = cfg.D_WEIGHT

        # embeddings in D_WEIGHT frame
        emb_mat = init_weight_mat(vocab_size, dim, d_w, seed, 0)
        self.token_emb = [
            Vec([emb_mat[i, j] for j in range(dim)])
            for i in range(vocab_size)
        ]

        pos_mat = init_weight_mat(seq_len, dim, d_w, seed, 100)
        self.pos_emb = [
            Vec([pos_mat[i, j] for j in range(dim)])
            for i in range(seq_len)
        ]

        # Linear layers — we store them for backward/parameters,
        # but forward uses qb_linear_forward with .weight.value and .bias.value
        self.Wq = Linear(
            init_weight_mat(dim, dim, d_w, seed, 200),
            init_bias_vec(dim, d_w), name="Wq"
        )
        self.Wk = Linear(
            init_weight_mat(dim, dim, d_w, seed, 300),
            init_bias_vec(dim, d_w), name="Wk"
        )
        self.Wv = Linear(
            init_weight_mat(dim, dim, d_w, seed, 400),
            init_bias_vec(dim, d_w), name="Wv"
        )
        self.Wo = Linear(
            init_weight_mat(dim, dim, d_w, seed, 500),
            init_bias_vec(dim, d_w), name="Wo"
        )
        self.ffn_l1 = Linear(
            init_weight_mat(ffn_dim, dim, d_w, seed, 600),
            init_bias_vec(ffn_dim, d_w), name="ffn_l1"
        )
        self.ffn_l2 = Linear(
            init_weight_mat(dim, ffn_dim, d_w, seed, 700),
            init_bias_vec(dim, d_w), name="ffn_l2"
        )
        self.relu = ReLU()
        self.output_proj = Linear(
            init_weight_mat(vocab_size, dim, d_w, seed, 800),
            init_bias_vec(vocab_size, d_w), name="output"
        )

        self._cache = {}

    # -- helpers for accessing raw weight/bias ----------------------------

    def _linear_fwd(self, layer, x, bits):
        """Linear forward via qb_linear_forward."""
        return qb_linear_forward(layer.weight.value, layer.bias.value, x, bits)

    # -- embedding --------------------------------------------------------

    def embed(self, token_ids):
        """Embed + positional in D_ACT frame."""
        bits = cfg.BITS_ACT
        xs = []
        for i, tid in enumerate(token_ids):
            combined = qb_vec_add(self.token_emb[tid], self.pos_emb[i], bits)
            xs.append(combined)
        return xs

    # -- attention --------------------------------------------------------

    def attention_block(self, xs):
        """
        Single-head causal self-attention. All qb_* arithmetic.
        D never leaves target frames.
        """
        n = len(xs)
        b_s = cfg.BITS_SCORE
        b_a = cfg.BITS_ACT
        b_p = cfg.BITS_PROB

        Q = [self._linear_fwd(self.Wq, x, b_s) for x in xs]
        K = [self._linear_fwd(self.Wk, x, b_s) for x in xs]
        V = [self._linear_fwd(self.Wv, x, b_a) for x in xs]

        # scores
        scores = []
        for i in range(n):
            row = [qb_dot(Q[i], K[j], b_s) for j in range(n)]
            scores.append(row)

        masked = apply_causal_mask(scores, n)

        # softmax per row
        weights = []
        for i in range(n):
            probs, _ = surrogate_softmax(Vec(masked[i]))
            weights.append(probs)

        # value mix
        attn_out = []
        for i in range(n):
            acc = zero_vec_in_frame(cfg.DIM, b_a)
            for j in range(n):
                if weights[i][j] != VDR(0):
                    scaled = qb_vec_scale(V[j], weights[i][j], b_a)
                    acc = qb_vec_add(acc, scaled, b_a)
            attn_out.append(acc)

        projected = [self._linear_fwd(self.Wo, a, b_a) for a in attn_out]
        return projected

    def attention_block_with_cache(self, xs):
        """Attention block caching intermediates for backward."""
        n = len(xs)
        b_s = cfg.BITS_SCORE
        b_a = cfg.BITS_ACT

        Q = [self._linear_fwd(self.Wq, x, b_s) for x in xs]
        K = [self._linear_fwd(self.Wk, x, b_s) for x in xs]
        V = [self._linear_fwd(self.Wv, x, b_a) for x in xs]

        scores = []
        for i in range(n):
            row = [qb_dot(Q[i], K[j], b_s) for j in range(n)]
            scores.append(row)

        masked = apply_causal_mask(scores, n)

        weights = []
        shifted_cache = []
        for i in range(n):
            probs, shifted = surrogate_softmax(Vec(masked[i]))
            weights.append(probs)
            shifted_cache.append(shifted)

        attn_out = []
        for i in range(n):
            acc = zero_vec_in_frame(cfg.DIM, b_a)
            for j in range(n):
                if weights[i][j] != VDR(0):
                    scaled = qb_vec_scale(V[j], weights[i][j], b_a)
                    acc = qb_vec_add(acc, scaled, b_a)
            attn_out.append(acc)

        projected = [self._linear_fwd(self.Wo, a, b_a) for a in attn_out]

        cache = {
            "Q": Q, "K": K, "V": V,
            "scores_masked": masked,
            "weights": weights,
            "shifted": shifted_cache,
            "attn_out": attn_out,
            "input": xs,
        }
        return projected, cache

    # -- FFN --------------------------------------------------------------

    def ffn_block(self, x):
        """FFN: linear -> relu -> linear. All qb_* arithmetic."""
        b_a = cfg.BITS_ACT
        h = self._linear_fwd(self.ffn_l1, x, b_a)
        h = self.relu.forward(h)
        h = self._linear_fwd(self.ffn_l2, h, b_a)
        return h

    # -- forward ----------------------------------------------------------

    def forward(self, token_ids):
        """Full forward pass. D never compounds."""
        b_a = cfg.BITS_ACT
        xs = self.embed(token_ids)

        attn = self.attention_block(xs)
        xs = [qb_vec_add(attn[i], xs[i], b_a) for i in range(len(xs))]

        ffn_out = [self.ffn_block(x) for x in xs]
        xs = [qb_vec_add(ffn_out[i], xs[i], b_a) for i in range(len(xs))]

        logits = [self._linear_fwd(self.output_proj, x, b_a) for x in xs]
        return logits

    def forward_with_cache(self, token_ids):
        """Forward caching all intermediates."""
        b_a = cfg.BITS_ACT
        xs_embed = self.embed(token_ids)

        attn_out, attn_cache = self.attention_block_with_cache(xs_embed)
        xs_post_attn = [qb_vec_add(attn_out[i], xs_embed[i], b_a)
                        for i in range(len(xs_embed))]

        ffn_pre = [self._linear_fwd(self.ffn_l1, x, b_a) for x in xs_post_attn]
        ffn_relu = [self.relu.forward(h) for h in ffn_pre]
        ffn_out = [self._linear_fwd(self.ffn_l2, h, b_a) for h in ffn_relu]
        xs_post_ffn = [qb_vec_add(ffn_out[i], xs_post_attn[i], b_a)
                       for i in range(len(xs_post_attn))]

        logits = [self._linear_fwd(self.output_proj, x, b_a) for x in xs_post_ffn]

        self._cache = {
            "token_ids": token_ids,
            "xs_embed": xs_embed,
            "attn_cache": attn_cache,
            "attn_out": attn_out,
            "xs_post_attn": xs_post_attn,
            "ffn_pre": ffn_pre,
            "ffn_relu": ffn_relu,
            "ffn_out": ffn_out,
            "xs_post_ffn": xs_post_ffn,
            "logits": logits,
        }
        return logits, self._cache

    def forward_last_logits(self, token_ids):
        return self.forward(token_ids)[-1]

    def forward_last_logits_with_cache(self, token_ids):
        logits, cache = self.forward_with_cache(token_ids)
        return logits[-1], cache

    # -- parameters -------------------------------------------------------

    def parameters(self):
        params = []
        params.extend(self.Wq.parameters())
        params.extend(self.Wk.parameters())
        params.extend(self.Wv.parameters())
        params.extend(self.Wo.parameters())
        params.extend(self.ffn_l1.parameters())
        params.extend(self.ffn_l2.parameters())
        params.extend(self.output_proj.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    # -- backward ---------------------------------------------------------

    def backward_from_output(self, grad_logits):
        """
        Backprop from logit gradients through entire model.
        Uses Linear.backward() for gradient accumulation into parameters.
        Requires forward_with_cache called first.
        """
        from attention_backward import attention_backward

        cache = self._cache
        n = len(grad_logits)

        # backward through output projection
        grad_post_ffn = []
        for i in range(n):
            self.output_proj._last_input = cache["xs_post_ffn"][i]
            g = self.output_proj.backward(grad_logits[i])
            grad_post_ffn.append(g)

        # residual split
        grad_ffn_out = list(grad_post_ffn)
        grad_xs_post_attn_from_res2 = list(grad_post_ffn)

        # backward through FFN
        grad_xs_post_attn_from_ffn = []
        for i in range(n):
            self.ffn_l2._last_input = cache["ffn_relu"][i]
            g = self.ffn_l2.backward(grad_ffn_out[i])

            self.relu._last_input = cache["ffn_pre"][i]
            g = self.relu.backward(g)

            self.ffn_l1._last_input = cache["xs_post_attn"][i]
            g = self.ffn_l1.backward(g)

            grad_xs_post_attn_from_ffn.append(g)

        grad_xs_post_attn = [
            grad_xs_post_attn_from_res2[i] + grad_xs_post_attn_from_ffn[i]
            for i in range(n)
        ]

        # residual split
        grad_attn_out = list(grad_xs_post_attn)
        grad_xs_embed_from_res1 = list(grad_xs_post_attn)

        # backward through attention
        attn_cache = cache["attn_cache"]
        grad_xs_embed_from_attn = attention_backward(
            grad_attn_out,
            attn_cache["Q"], attn_cache["K"], attn_cache["V"],
            attn_cache["weights"], attn_cache["shifted"],
            self.Wq, self.Wk, self.Wv, self.Wo,
            attn_cache["input"],
        )

        grad_xs_embed = [
            grad_xs_embed_from_res1[i] + grad_xs_embed_from_attn[i]
            for i in range(n)
        ]

    def backward_from_last(self, grad):
        cache = self._cache
        n = len(cache["logits"])
        grad_logits = [zero_vec(len(grad))] * n
        grad_logits[n - 1] = grad
        self.backward_from_output(grad_logits)
        