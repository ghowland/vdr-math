"""
Toy transformer LLM — single-block, single-head, exact VDR arithmetic.

All weights initialized as small integers in basis frame via to_qbasis.
All operations go through basis-safe operators — D stays at 2^32.
"""

from __future__ import annotations

from vdr.core import VDR
from vdr.linalg import Vec, Mat
from vdr.basis import to_qbasis, vec_to_qbasis, mat_to_qbasis
from vdr.ml.nn import Linear, ReLU, MatParam, VecParam
from vdr.ml.softmax import softmax_surrogate_square
from vdr.ml.attention import attention_scores, causal_mask, apply_boolean_mask, weighted_sum

from config import DIM, FFN_DIM, VOCAB_SIZE, SEQ_LEN, SEED


# ---------------------------------------------------------------------------
# Deterministic weight init
# ---------------------------------------------------------------------------

def _lcg(seed):
    """Simple LCG RNG returning values in sequence."""
    state = seed
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state


def _init_vec(dim, rng, scale=4):
    """
    Init a Vec with small-integer values in basis frame.
    Values in [-scale, scale].
    """
    data = []
    for _ in range(dim):
        raw = next(rng)
        val = (raw % (2 * scale + 1)) - scale
        data.append(to_qbasis(VDR(val, scale)))
    return Vec(data)


def _init_mat(rows, cols, rng, scale=4):
    """
    Init a Mat with small-integer values in basis frame.
    Values in [-scale, scale].
    """
    mat_rows = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            raw = next(rng)
            val = (raw % (2 * scale + 1)) - scale
            row.append(to_qbasis(VDR(val, scale)))
        mat_rows.append(row)
    return Mat(mat_rows)


def _init_linear(out_dim, in_dim, rng, scale=4, name=None):
    """Init a Linear layer with small weights and zero bias in basis frame."""
    w = _init_mat(out_dim, in_dim, rng, scale)
    b = Vec([to_qbasis(VDR(0))] * out_dim)
    return Linear(w, b, name=name)


def _zero_vec(dim):
    """Zero Vec in basis frame."""
    return Vec([to_qbasis(VDR(0))] * dim)


# ---------------------------------------------------------------------------
# Outer product for backward
# ---------------------------------------------------------------------------

def _outer(a, b):
    """Outer product: M[i,j] = a[i] * b[j]."""
    rows = []
    for i in range(len(a)):
        row = []
        for j in range(len(b)):
            row.append(a[i] * b[j])
        rows.append(row)
    return Mat(rows)


# ---------------------------------------------------------------------------
# ToyTransformer
# ---------------------------------------------------------------------------

class ToyTransformer:
    """
    Single-block, single-head transformer language model.

    Architecture:
        token_emb + pos_emb -> attention -> residual -> FFN -> residual -> output_proj

    All parameters in basis frame (D = 2^32).
    Surrogate quadratic softmax — no transcendentals.
    """

    def __init__(self, seed=None):
        if seed is None:
            seed = SEED
        rng = _lcg(seed)

        # Embeddings: small integer vectors in basis frame
        self.token_emb = [_init_vec(DIM, rng) for _ in range(VOCAB_SIZE)]
        self.pos_emb = [_init_vec(DIM, rng) for _ in range(SEQ_LEN)]

        # Attention projections
        self.Wq = _init_linear(DIM, DIM, rng, name="Wq")
        self.Wk = _init_linear(DIM, DIM, rng, name="Wk")
        self.Wv = _init_linear(DIM, DIM, rng, name="Wv")
        self.Wo = _init_linear(DIM, DIM, rng, name="Wo")

        # FFN
        self.ffn_l1 = _init_linear(FFN_DIM, DIM, rng, name="ffn_l1")
        self.ffn_l2 = _init_linear(DIM, FFN_DIM, rng, name="ffn_l2")
        self.relu = ReLU()

        # Output projection
        self.output_proj = _init_linear(VOCAB_SIZE, DIM, rng, name="out")

        # Cache for backward
        self._cache = {}

    # -- embed -------------------------------------------------------------

    def embed(self, token_ids):
        """
        Look up token + positional embeddings.

        I: list of int token ids
        O: list of Vec in basis frame
        """
        result = []
        for pos, tid in enumerate(token_ids):
            result.append(self.token_emb[tid] + self.pos_emb[pos])
        return result

    # -- attention ---------------------------------------------------------

    def attention_block(self, xs):
        """
        Single-head self-attention with quadratic softmax surrogate.

        I: list of Vec (sequence of hidden states)
        O: list of Vec (attended + projected)
        """
        out, _ = self.attention_block_with_cache(xs)
        return out

    def attention_block_with_cache(self, xs):
        """
        Attention block with cached intermediates for backward.

        I: list of Vec
        O: (list of Vec, cache dict)
        """
        Q = [self.Wq.forward(x) for x in xs]
        K = [self.Wk.forward(x) for x in xs]
        V = [self.Wv.forward(x) for x in xs]

        # scores: S[i,j] = Q[i] . K[j]
        n = len(xs)
        score_rows = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(Q[i].dot(K[j]))
            score_rows.append(Vec(row))

        # causal mask
        mask = causal_mask(n)
        masked_scores = apply_boolean_mask(score_rows, mask)

        # quadratic softmax surrogate per row
        weight_rows = []
        shifted_rows = []
        for row in masked_scores:
            # compute shift (min of row)
            shift = row[0]
            for k in range(1, len(row)):
                if row[k] < shift:
                    shift = row[k]
            shifted = Vec([x - shift for x in row])
            shifted_rows.append(shifted)
            weight_rows.append(softmax_surrogate_square(row, shift=shift))

        # weighted sum of values
        attn_out = [weighted_sum(w_row, V) for w_row in weight_rows]

        # output projection
        projected = [self.Wo.forward(a) for a in attn_out]

        cache = {
            "xs_input": xs,
            "Q": Q, "K": K, "V": V,
            "scores": masked_scores,
            "shifted": shifted_rows,
            "weights": weight_rows,
            "attn_out": attn_out,
        }

        return projected, cache

    # -- FFN ---------------------------------------------------------------

    def ffn_block(self, x):
        """
        Feed-forward network: linear -> relu -> linear.

        I: Vec
        O: Vec
        """
        h = self.ffn_l1.forward(x)
        h = self.relu.forward(h)
        return self.ffn_l2.forward(h)

    # -- forward -----------------------------------------------------------

    def forward(self, token_ids):
        """
        Full forward pass.

        I: list of int token ids
        O: list of Vec (logit vectors, one per position)
        """
        logits, _ = self.forward_with_cache(token_ids)
        return logits

    def forward_with_cache(self, token_ids):
        """
        Forward pass caching all intermediates for backward.

        I: list of int token ids
        O: (list of Vec logits, cache dict)
        """
        # embed
        xs = self.embed(token_ids)
        pre_attn = xs

        # attention + residual
        attn_out, attn_cache = self.attention_block_with_cache(xs)
        post_attn = [a + x for a, x in zip(attn_out, xs)]

        # FFN + residual (cache per position)
        pre_ffn = post_attn
        ffn_out = [self.ffn_block(x) for x in post_attn]
        post_ffn = [f + x for f, x in zip(ffn_out, post_attn)]

        # output projection
        logits = [self.output_proj.forward(h) for h in post_ffn]

        self._cache = {
            "token_ids": token_ids,
            "pre_attn": pre_attn,
            "attn_cache": attn_cache,
            "attn_out": attn_out,
            "post_attn": post_attn,
            "pre_ffn": pre_ffn,
            "ffn_out": ffn_out,
            "post_ffn": post_ffn,
            "logits": logits,
        }

        return logits, self._cache

    def forward_last_logits(self, token_ids):
        """
        Forward pass returning only last position logits.

        I: list of int token ids
        O: Vec (logits for last position)
        """
        logits = self.forward(token_ids)
        return logits[-1]

    def forward_last_logits_with_cache(self, token_ids):
        """
        Forward returning last logits + full cache.

        I: list of int token ids
        O: (Vec, cache dict)
        """
        logits, cache = self.forward_with_cache(token_ids)
        return logits[-1], cache

    # -- parameters --------------------------------------------------------

    def parameters(self):
        """Return flat list of all trainable parameter objects."""
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

    # -- backward ----------------------------------------------------------

    def backward_from_output(self, grad_logits):
        """
        Full backward pass from logit gradients.

        I: list of Vec — gradient per position logits
        S: accumulates gradients into all parameters
        Requires forward_with_cache called first.
        """
        cache = self._cache
        n = len(grad_logits)

        # backward through output_proj
        grad_post_ffn = []
        for i in range(n):
            g = self.output_proj.backward(grad_logits[i])
            grad_post_ffn.append(g)

        # backward through FFN residual: grad splits to FFN and skip
        grad_pre_ffn = []
        for i in range(n):
            # FFN backward
            g_ffn = self.ffn_l2.backward(grad_post_ffn[i])
            g_ffn = self.relu.backward(g_ffn)
            g_ffn = self.ffn_l1.backward(g_ffn)
            # residual: gradient flows through both paths
            grad_pre_ffn.append(g_ffn + grad_post_ffn[i])

        # backward through attention residual
        grad_attn_projected = grad_pre_ffn  # same as grad into attention output

        # backward through attention block
        from attention_backward import attention_backward
        grad_pre_attn = attention_backward(
            grad_attn_projected,
            cache["attn_cache"]["Q"],
            cache["attn_cache"]["K"],
            cache["attn_cache"]["V"],
            cache["attn_cache"]["weights"],
            cache["attn_cache"]["shifted"],
            self.Wq, self.Wk, self.Wv, self.Wo,
            cache["attn_cache"]["xs_input"],
        )

        # attention residual: grad splits
        # grad_pre_attn already includes the skip connection contribution
        # from attention_backward, plus we add the residual skip
        for i in range(n):
            grad_pre_attn[i] = grad_pre_attn[i] + grad_pre_ffn[i]

    def backward_from_last(self, grad):
        """
        Backward from last position gradient only.

        I: Vec — gradient for last position
        S: wraps backward_from_output with zeros for other positions
        """
        n = len(self._cache["logits"])
        grad_logits = [_zero_vec(len(grad))] * n
        grad_logits[-1] = grad
        self.backward_from_output(grad_logits)
