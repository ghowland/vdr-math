"""
vdr.ml.transformer — Exact transformer architecture.

    from vdr.ml.transformer import TransformerLM

    model = TransformerLM(embedding, [block], output_proj)
    logits = model.forward_logits([0, 1, 2])

Every attention weight sums to exactly 1. Every gradient exact.
No float drift across layers or sequence positions.
"""

from __future__ import annotations
from typing import List, Sequence, Optional, Tuple

from vdr.core import VDR
from vdr.linalg import Vec, Mat
from vdr.ml.nn import Linear, ReLU, FFN, Module
from vdr.ml.attention import self_attention

__all__ = [
    "Embedding",
    "FFNBlock",
    "TransformerBlock",
    "TransformerLM",
]


def _to_vdr(x):
    if isinstance(x, VDR):
        return x
    if isinstance(x, int):
        return VDR(x)
    raise TypeError("Expected VDR or int")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class Embedding:
    """
    Lookup table embedding: integer token id -> Vec.

        emb = Embedding.from_ints([[1,0,0],[0,1,0],[0,0,1]])
        v = emb.lookup(1)  # Vec([VDR(0), VDR(1), VDR(0)])
    """

    def __init__(self, table, name=None):
        """
        I: table as list of Vec
        """
        self.table = list(table)
        self.name = name

    @classmethod
    def from_ints(cls, rows, name=None):
        """Construct from integer lists."""
        return cls([Vec([VDR(x) for x in row]) for row in rows], name)

    @classmethod
    def from_fracs(cls, rows, name=None):
        """Construct from fraction tuples."""
        return cls([Vec([VDR(a, b) for a, b in row]) for row in rows], name)

    @property
    def vocab_size(self):
        return len(self.table)

    @property
    def dim(self):
        if not self.table:
            return 0
        return len(self.table[0])

    def lookup(self, idx):
        """
        Look up embedding vector for token id.

        I: integer token id
        O: Vec
        """
        if idx < 0 or idx >= len(self.table):
            raise ValueError("Token id %d out of range [0, %d)" % (idx, len(self.table)))
        return self.table[idx]

    def lookup_many(self, ids):
        """
        Look up embedding vectors for sequence of token ids.

        I: list of integer token ids
        O: list of Vec
        """
        return [self.lookup(i) for i in ids]

    def to_qbasis(self, bits):
        """Project all embedding vectors onto Q basis."""
        from vdr.basis import vec_to_qbasis
        return Embedding(
            [vec_to_qbasis(v, bits) for v in self.table],
            name=self.name,
        )


# ---------------------------------------------------------------------------
# FFN Block
# ---------------------------------------------------------------------------

class FFNBlock(Module):
    """
    Feed-forward network block: linear -> relu -> linear.

    Common in transformer architectures.

        ffn = FFNBlock(
            Linear.from_ints([[1,2],[3,4],[5,6]], [0,0,0]),
            Linear.from_ints([[1,0],[0,1],[1,1]], [0,0]),
        )
    """

    def __init__(self, l1, l2):
        self.l1 = l1
        self.act = ReLU()
        self.l2 = l2

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

    def zero_grad(self):
        self.l1.zero_grad()
        self.l2.zero_grad()

    def forward(self, x):
        """x -> linear -> relu -> linear."""
        h = self.l1.forward(x)
        h = self.act.forward(h)
        return self.l2.forward(h)

    def backward(self, grad_out):
        g = self.l2.backward(grad_out)
        g = self.act.backward(g)
        return self.l1.backward(g)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(Module):
    """
    Single transformer block: self-attention + FFN.

    For each position:
        h = self_attention(Q, K, V)  (with optional causal mask)
        out = ffn(h)

    Simplified: no layer norm, no residual connection in v1.
    These can be added as the identity + output pattern.

        block = TransformerBlock(Wq, Wk, Wv, Wo, ffn)
        outputs = block.forward(token_vecs)
    """

    def __init__(self, Wq, Wk, Wv, Wo, ffn, causal=True, exp_depth=16):
        """
        I: projection matrices Wq, Wk, Wv, Wo as Mat,
           ffn as FFNBlock, causal flag, exp depth
        """
        self.Wq = Wq
        self.Wk = Wk
        self.Wv = Wv
        self.Wo = Wo
        self.ffn = ffn
        self.causal = causal
        self.exp_depth = exp_depth

    def parameters(self):
        # Wo could be wrapped in MatParam for training
        return self.ffn.parameters()

    def zero_grad(self):
        self.ffn.zero_grad()

    def forward(self, xs):
        """
        Forward pass through transformer block.

        I: xs as list of Vec (sequence of hidden states)
        O: list of Vec (transformed hidden states)
        """
        # project to Q, K, V
        Q = [self.Wq.matvec(x) for x in xs]
        K = [self.Wk.matvec(x) for x in xs]
        V = [self.Wv.matvec(x) for x in xs]

        # self-attention
        attn_out = self_attention(Q, K, V,
                                  causal=self.causal,
                                  exp_depth=self.exp_depth)

        # output projection
        projected = [self.Wo.matvec(a) for a in attn_out]

        # FFN on each position
        result = [self.ffn.forward(p) for p in projected]

        return result

    def forward_with_cache(self, xs):
        """
        Forward pass returning attention intermediate values for inspection.

        O: (outputs, Q, K, V, attn_outputs)
        """
        Q = [self.Wq.matvec(x) for x in xs]
        K = [self.Wk.matvec(x) for x in xs]
        V = [self.Wv.matvec(x) for x in xs]

        attn_out = self_attention(Q, K, V,
                                  causal=self.causal,
                                  exp_depth=self.exp_depth)

        projected = [self.Wo.matvec(a) for a in attn_out]
        result = [self.ffn.forward(p) for p in projected]

        return result, Q, K, V, attn_out


# ---------------------------------------------------------------------------
# Transformer Language Model
# ---------------------------------------------------------------------------

class TransformerLM(Module):
    """
    Complete transformer language model.

    embedding -> transformer blocks -> output projection -> logits

        model = TransformerLM(embedding, [block1, block2], output_linear)
        logits = model.forward_logits([0, 1, 2, 3])
    """

    def __init__(self, embedding, blocks, output_proj):
        """
        I: Embedding, list of TransformerBlock, Linear output projection
        """
        self.embedding = embedding
        self.blocks = list(blocks)
        self.output_proj = output_proj

    def parameters(self):
        params = []
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.output_proj.parameters())
        return params

    def zero_grad(self):
        for block in self.blocks:
            block.zero_grad()
        self.output_proj.zero_grad()

    def embed(self, token_ids):
        """
        Embed token ids into vectors.

        I: list of integer token ids
        O: list of Vec
        """
        return self.embedding.lookup_many(token_ids)

    def forward_hidden(self, token_ids):
        """
        Forward pass through embedding and transformer blocks.

        I: list of integer token ids
        O: list of Vec (final hidden states)
        """
        xs = self.embed(token_ids)
        for block in self.blocks:
            xs = block.forward(xs)
        return xs

    def forward_logits(self, token_ids):
        """
        Full forward pass producing logits.

        I: list of integer token ids
        O: list of Vec (logit vectors, one per position)
        """
        hidden = self.forward_hidden(token_ids)
        return [self.output_proj.forward(h) for h in hidden]

    def forward_logits_with_cache(self, token_ids):
        """
        Forward pass with cached intermediates for inspection.

        O: (logits, hidden_states)
        """
        hidden = self.forward_hidden(token_ids)
        logits = [self.output_proj.forward(h) for h in hidden]
        return logits, hidden

    def to_qbasis(self, bits):
        """Project embedding to Q basis. Blocks unchanged."""
        return TransformerLM(
            self.embedding.to_qbasis(bits),
            self.blocks,
            self.output_proj,
        )
