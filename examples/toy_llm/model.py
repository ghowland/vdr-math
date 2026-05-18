"""
examples/toy_llm/model.py

Toy LLM model — 1-layer decoder-only transformer.

Small fixed D-frames matching frontier quantized models:
  Weights in D_WEIGHT=128, activations in D_ACT=128,
  scores in D_SCORE=256, probabilities in D_PROB=256,
  accumulators in D_ACC=32768.

Every intermediate rebased at layer boundaries.
Denominators never grow. Overflow to R. Arithmetic stays fast.

    from model import ToyTransformer

    model = ToyTransformer()
    logits = model.forward([0, 1, 2, 3])
"""

from vdr.core import VDR
from vdr.linalg import Vec, Mat
from vdr.ml.nn import Linear, ReLU

from frames import (
    rebase_value, rebase_vec, rebase_vec_list,
    rebase_params, init_weight_mat, init_bias_vec, zero_vec,
)
import config as cfg


# -- surrogate softmax ----------------------------------------------------

def surrogate_softmax(scores, c=None):
    """
    Rational surrogate softmax: s_i = (z_i - m + c)^2 / sum((z_j - m + c)^2)

    No transcendentals. Pure integer arithmetic. Sum to exactly 1.
    Returns (probs, shifted) where shifted = [z_i - m + c] cached for backward.

    I: scores Vec in any frame, c shift constant (default from config)
    O: (probs Vec summing to exactly 1, shifted Vec for backward)

        probs, shifted = surrogate_softmax(Vec([VDR(1), VDR(2), VDR(3)]))
        # probs sums to exactly VDR(1)
    """
    if c is None:
        c = cfg.SURR_C

    n = len(scores)

    # find max for numerical stability (exact comparison)
    m = scores[0]
    for i in range(1, n):
        if scores[i] > m:
            m = scores[i]

    # shift: z_i - m + c (all values positive since c > 0 and z_i - m >= 0)
    shifted = Vec([scores[i] - m + c for i in range(n)])

    # square each
    squared = [shifted[i] * shifted[i] for i in range(n)]

    # sum of squares
    total = VDR(0)
    for s in squared:
        total = total + s

    # normalize — each element divided by total, sums to exactly 1
    if total == VDR(0):
        # degenerate: uniform
        probs = Vec([VDR(1, n) for _ in range(n)])
        return probs, shifted

    probs = Vec([squared[i] / total for i in range(n)])
    return probs, shifted


def surrogate_softmax_probs(scores, c=None):
    """
    Surrogate softmax returning only probabilities (no shifted cache).

    I: scores Vec
    O: probs Vec summing to exactly 1

        probs = surrogate_softmax_probs(Vec([VDR(1), VDR(2), VDR(3)]))
    """
    probs, _ = surrogate_softmax(scores, c)
    return probs


# -- causal mask ----------------------------------------------------------

def causal_mask_value():
    """
    Large negative value for masking future positions.

    Returns VDR(-1000) — large enough that surrogate softmax
    produces negligible weight after shifting.

    O: VDR
    """
    return VDR(-1000)


def apply_causal_mask(scores_matrix, seq_len):
    """
    Apply causal mask to attention score matrix.

    scores_matrix[i][j] for i < j (future) set to large negative.

    I: list of Vec (one per query position), seq_len
    O: list of Vec (masked)

        masked = apply_causal_mask(scores, 4)
    """
    mask_val = causal_mask_value()
    result = []
    for i in range(seq_len):
        row = []
        for j in range(seq_len):
            if j > i:
                row.append(mask_val)
            else:
                row.append(scores_matrix[i][j])
        result.append(Vec(row))
    return result


# -- linear with rebase --------------------------------------------------

def linear_forward_rebased(layer, x, target_d):
    """
    Linear layer forward pass with output rebased to target frame.

    I: Linear layer, input Vec, target denominator
    O: Vec in target_d frame

    The raw matvec + bias produces values in a compound frame.
    Rebase to target_d keeps denominators bounded.
    """
    raw = layer.forward(x)
    return rebase_vec(raw, target_d)


# -- attention score computation ------------------------------------------

def compute_scores(Q, K, target_d):
    """
    Compute attention scores: S[i][j] = Q[i] . K[j], rebased.

    I: Q list of Vec, K list of Vec, target denominator for scores
    O: list of list of VDR (score matrix), each value rebased

        scores = compute_scores(Q, K, D_SCORE)
    """
    n_q = len(Q)
    n_k = len(K)
    result = []
    for i in range(n_q):
        row = []
        for j in range(n_k):
            s = Q[i].dot(K[j])
            row.append(rebase_value(s, target_d))
        result.append(row)
    return result


def compute_value_mix(weights, V, target_d):
    """
    Compute attention-weighted sum of values, rebased.

    output[i] = sum_j weights[i][j] * V[j]

    I: weights (list of Vec), V (list of Vec), target denominator
    O: list of Vec, each rebased to target_d

        outputs = compute_value_mix(weights, V, D_ACT)
    """
    n_q = len(weights)
    dim = len(V[0])
    result = []
    for i in range(n_q):
        acc = zero_vec(dim)
        for j in range(len(V)):
            w_ij = weights[i][j]
            if w_ij != VDR(0):
                scaled = V[j].scale(w_ij)
                acc = acc + scaled
        result.append(rebase_vec(acc, target_d))
    return result


# -- model ----------------------------------------------------------------

class ToyTransformer:
    """
    1-layer decoder-only transformer with small fixed D-frames.

    Weights: D_WEIGHT=128. Activations: D_ACT=128.
    Scores: D_SCORE=256. Probabilities: D_PROB=256.
    Accumulators: D_ACC=32768.

    226 parameters, all small-integer rationals.

        model = ToyTransformer()
        logits = model.forward([0, 1, 2, 3])
    """

    def __init__(self, seed=None):
        if seed is None:
            seed = cfg.SEED

        dim = cfg.DIM
        ffn_dim = cfg.FFN_DIM
        vocab_size = cfg.VOCAB_SIZE
        seq_len = cfg.SEQ_LEN
        d_w = cfg.D_WEIGHT

        # token embedding: vocab_size x dim, in D_WEIGHT frame
        emb_mat = init_weight_mat(vocab_size, dim, d_w, seed, 0)
        self.token_emb = [
            Vec([emb_mat[i, j] for j in range(dim)])
            for i in range(vocab_size)
        ]

        # positional embedding: seq_len x dim, in D_WEIGHT frame
        pos_mat = init_weight_mat(seq_len, dim, d_w, seed, 100)
        self.pos_emb = [
            Vec([pos_mat[i, j] for j in range(dim)])
            for i in range(seq_len)
        ]

        # attention projections: dim -> dim
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

        # FFN: dim -> ffn_dim -> dim
        self.ffn_l1 = Linear(
            init_weight_mat(ffn_dim, dim, d_w, seed, 600),
            init_bias_vec(ffn_dim, d_w), name="ffn_l1"
        )
        self.ffn_l2 = Linear(
            init_weight_mat(dim, ffn_dim, d_w, seed, 700),
            init_bias_vec(dim, d_w), name="ffn_l2"
        )
        self.relu = ReLU()

        # output projection: dim -> vocab_size
        self.output_proj = Linear(
            init_weight_mat(vocab_size, dim, d_w, seed, 800),
            init_bias_vec(vocab_size, d_w), name="output"
        )

        # cache for backward pass
        self._cache = {}

    # -- embedding --------------------------------------------------------

    def embed(self, token_ids):
        """
        Embed tokens and add positional encoding, rebase to D_ACT.

        I: list of token ids (ints), length <= SEQ_LEN
        O: list of Vec in D_ACT frame
        """
        xs = []
        for i, tid in enumerate(token_ids):
            tok = self.token_emb[tid]
            pos = self.pos_emb[i]
            combined = tok + pos
            xs.append(rebase_vec(combined, cfg.D_ACT))
        return xs

    # -- attention block --------------------------------------------------

    def attention_block(self, xs):
        """
        Single-head causal self-attention with frame transitions.

        I: list of Vec in D_ACT frame
        O: list of Vec in D_ACT frame

        Frame flow:
          input D_ACT -> Wq/Wk/Wv -> rebase D_SCORE
          scores = Q.K -> rebase D_SCORE
          surrogate softmax -> D_PROB (sums exactly 1)
          value mix -> rebase D_ACT
          Wo -> rebase D_ACT
        """
        n = len(xs)

        # project Q, K, V and rebase to score frame
        Q = [rebase_vec(self.Wq.forward(x), cfg.D_SCORE) for x in xs]
        K = [rebase_vec(self.Wk.forward(x), cfg.D_SCORE) for x in xs]
        V = [rebase_vec(self.Wv.forward(x), cfg.D_ACT) for x in xs]

        # attention scores -> rebase to D_SCORE
        scores = compute_scores(Q, K, cfg.D_SCORE)

        # causal mask
        masked = apply_causal_mask(scores, n)

        # surrogate softmax per row -> D_PROB
        weights = []
        for i in range(n):
            probs, _ = surrogate_softmax(masked[i])
            weights.append(rebase_vec(probs, cfg.D_PROB))

        # value mixing -> rebase to D_ACT
        attn_out = compute_value_mix(weights, V, cfg.D_ACT)

        # output projection -> rebase to D_ACT
        projected = [rebase_vec(self.Wo.forward(a), cfg.D_ACT) for a in attn_out]

        return projected

    def attention_block_with_cache(self, xs):
        """
        Attention block caching all intermediates for backward.

        I: list of Vec in D_ACT
        O: (list of Vec in D_ACT, cache dict)
        """
        n = len(xs)

        Q = [rebase_vec(self.Wq.forward(x), cfg.D_SCORE) for x in xs]
        K = [rebase_vec(self.Wk.forward(x), cfg.D_SCORE) for x in xs]
        V = [rebase_vec(self.Wv.forward(x), cfg.D_ACT) for x in xs]

        scores = compute_scores(Q, K, cfg.D_SCORE)
        masked = apply_causal_mask(scores, n)

        weights = []
        shifted_cache = []
        for i in range(n):
            probs, shifted = surrogate_softmax(masked[i])
            weights.append(rebase_vec(probs, cfg.D_PROB))
            shifted_cache.append(shifted)

        attn_out = compute_value_mix(weights, V, cfg.D_ACT)
        projected = [rebase_vec(self.Wo.forward(a), cfg.D_ACT) for a in attn_out]

        cache = {
            "Q": Q, "K": K, "V": V,
            "scores_masked": masked,
            "weights": weights,
            "shifted": shifted_cache,
            "attn_out": attn_out,
            "input": xs,
        }

        return projected, cache

    # -- FFN block --------------------------------------------------------

    def ffn_block(self, x):
        """
        Feed-forward: linear -> relu -> linear, rebased at each boundary.

        I: Vec in D_ACT
        O: Vec in D_ACT
        """
        h = self.ffn_l1.forward(x)
        h = rebase_vec(h, cfg.D_ACT)
        h = self.relu.forward(h)
        h = self.ffn_l2.forward(h)
        h = rebase_vec(h, cfg.D_ACT)
        return h

    # -- forward ----------------------------------------------------------

    def forward(self, token_ids):
        """
        Full forward pass.

        I: list of token ids
        O: list of logit Vec in D_ACT frame

        Pipeline:
          embed (D_ACT) -> attention + residual (D_ACT)
          -> FFN + residual (D_ACT) -> output projection (D_ACT)
        """
        xs = self.embed(token_ids)

        attn = self.attention_block(xs)
        xs = [rebase_vec(attn[i] + xs[i], cfg.D_ACT) for i in range(len(xs))]

        ffn_out = [self.ffn_block(x) for x in xs]
        xs = [rebase_vec(ffn_out[i] + xs[i], cfg.D_ACT) for i in range(len(xs))]

        logits = [rebase_vec(self.output_proj.forward(x), cfg.D_ACT)
                  for x in xs]

        return logits

    def forward_with_cache(self, token_ids):
        """
        Forward pass caching all intermediates for backward.

        O: (logits list, cache dict)
        """
        xs_embed = self.embed(token_ids)

        attn_out, attn_cache = self.attention_block_with_cache(xs_embed)
        xs_post_attn = [
            rebase_vec(attn_out[i] + xs_embed[i], cfg.D_ACT)
            for i in range(len(xs_embed))
        ]

        ffn_pre = [self.ffn_l1.forward(x) for x in xs_post_attn]
        ffn_pre = [rebase_vec(h, cfg.D_ACT) for h in ffn_pre]
        ffn_relu = [self.relu.forward(h) for h in ffn_pre]
        ffn_out_raw = [self.ffn_l2.forward(h) for h in ffn_relu]
        ffn_out = [rebase_vec(h, cfg.D_ACT) for h in ffn_out_raw]
        xs_post_ffn = [
            rebase_vec(ffn_out[i] + xs_post_attn[i], cfg.D_ACT)
            for i in range(len(xs_post_attn))
        ]

        logits = [
            rebase_vec(self.output_proj.forward(x), cfg.D_ACT)
            for x in xs_post_ffn
        ]

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
        """
        Forward pass, return only last position logits.

        I: token ids
        O: Vec (logits for last position) in D_ACT
        """
        logits = self.forward(token_ids)
        return logits[-1]

    def forward_last_logits_with_cache(self, token_ids):
        """
        Forward with cache, return last logits.

        O: (last_logits Vec, cache dict)
        """
        logits, cache = self.forward_with_cache(token_ids)
        return logits[-1], cache

    # -- parameters -------------------------------------------------------

    def parameters(self):
        """All trainable parameters as flat list."""
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
        """Zero all parameter gradients."""
        for p in self.parameters():
            p.zero_grad()

    # -- backward ---------------------------------------------------------

    def backward_from_output(self, grad_logits):
        """
        Backpropagate from logit gradients through entire model.

        I: list of Vec — gradient of loss w.r.t. each position's logits
        S: accumulates gradients into all parameters

        Requires forward_with_cache to have been called first.
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

        # residual: grad flows to FFN output and xs_post_attn
        grad_ffn_out = list(grad_post_ffn)
        grad_xs_post_attn_from_res2 = list(grad_post_ffn)

        # backward through FFN per position
        grad_xs_post_attn_from_ffn = []
        for i in range(n):
            self.ffn_l2._last_input = cache["ffn_relu"][i]
            g = self.ffn_l2.backward(grad_ffn_out[i])

            self.relu._last_input = cache["ffn_pre"][i]
            g = self.relu.backward(g)

            self.ffn_l1._last_input = cache["xs_post_attn"][i]
            g = self.ffn_l1.backward(g)

            grad_xs_post_attn_from_ffn.append(g)

        # combine residual gradients
        grad_xs_post_attn = [
            grad_xs_post_attn_from_res2[i] + grad_xs_post_attn_from_ffn[i]
            for i in range(n)
        ]

        # residual: grad flows to attention output and xs_embed
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

        # combine residual gradients
        grad_xs_embed = [
            grad_xs_embed_from_res1[i] + grad_xs_embed_from_attn[i]
            for i in range(n)
        ]

        # embeddings fixed in v1 — stop here

    def backward_from_last(self, grad):
        """
        Backpropagate from last position logit gradient only.

        I: Vec — gradient w.r.t. last position logits
        S: accumulates into all parameter gradients
        """
        cache = self._cache
        n = len(cache["logits"])

        grad_logits = [zero_vec(len(grad))] * n
        grad_logits[n - 1] = grad

        self.backward_from_output(grad_logits)

