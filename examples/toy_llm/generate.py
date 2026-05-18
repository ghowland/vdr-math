"""
Text generation for toy transformer LLM.
Greedy, sampled, top-k, and nucleus decoding.
All sampling uses exact rational CDF comparison.
"""

from vdr.core import VDR
from vdr.linalg import Vec
from vdr.basis import to_qbasis
from vdr.ml.softmax import softmax_surrogate_square

from config import VOCAB_SIZE, SEQ_LEN


# ---------------------------------------------------------------------------
# Simple deterministic RNG
# ---------------------------------------------------------------------------

class SimpleRNG:
    """
    Deterministic LCG RNG producing exact rational values in [0, 1).
    """

    def __init__(self, seed=42):
        self._state = seed

    def _next_raw(self):
        self._state = (1103515245 * self._state + 12345) & 0x7FFFFFFF
        return self._state

    def rand_fraction(self):
        """Return a VDR in [0, 1) with denominator 2^31."""
        raw = self._next_raw()
        return VDR(raw, 0x80000000)


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------

def sample_greedy(probs):
    """
    Greedy (argmax) sampling.

    I: probs Vec
    O: int index of maximum element
    """
    best_idx = 0
    best_val = probs[0]
    for i in range(1, len(probs)):
        if probs[i] > best_val:
            best_val = probs[i]
            best_idx = i
    return best_idx


def sample_categorical(probs, rng):
    """
    Categorical sampling via exact CDF comparison.

    I: probs Vec (sums to exactly 1), RNG
    O: int sampled index
    """
    u = rng.rand_fraction()
    cumsum = VDR(0)
    for i in range(len(probs)):
        cumsum = cumsum + probs[i]
        if u < cumsum:
            return i
    return len(probs) - 1


def sample_top_k(probs, k, rng):
    """
    Top-k sampling: keep k largest, renormalize, sample.

    I: probs Vec, k int, RNG
    O: int sampled index
    """
    n = len(probs)
    if k >= n:
        return sample_categorical(probs, rng)

    # find top-k indices
    indexed = [(probs[i], i) for i in range(n)]
    indexed.sort(key=lambda x: x[0].to_fraction(), reverse=True)
    top_indices = set(indexed[i][1] for i in range(k))

    # renormalize
    total = VDR(0)
    for i in top_indices:
        total = total + probs[i]

    if total == VDR(0):
        return indexed[0][1]

    filtered = []
    for i in range(n):
        if i in top_indices:
            filtered.append(probs[i] / total)
        else:
            filtered.append(VDR(0))

    return sample_categorical(Vec(filtered), rng)


def sample_nucleus(probs, p_threshold, rng):
    """
    Nucleus (top-p) sampling: smallest set exceeding threshold.

    I: probs Vec, threshold VDR, RNG
    O: int sampled index
    """
    n = len(probs)
    indexed = [(probs[i], i) for i in range(n)]
    indexed.sort(key=lambda x: x[0].to_fraction(), reverse=True)

    cumsum = VDR(0)
    keep = set()
    for prob, idx in indexed:
        keep.add(idx)
        cumsum = cumsum + prob
        if cumsum >= p_threshold:
            break

    total = VDR(0)
    for i in keep:
        total = total + probs[i]

    if total == VDR(0):
        return indexed[0][1]

    filtered = []
    for i in range(n):
        if i in keep:
            filtered.append(probs[i] / total)
        else:
            filtered.append(VDR(0))

    return sample_categorical(Vec(filtered), rng)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_ids(model, prompt_ids, max_tokens, strategy="greedy",
                 seed=42, top_k=3, top_p=None):
    """
    Autoregressive token generation.

    I: model, prompt token ids, max new tokens, strategy config
    O: list of int (prompt + generated ids)
    """
    if top_p is None:
        top_p = VDR(9, 10)
    elif not isinstance(top_p, VDR):
        top_p = VDR(top_p * 10, 10) if isinstance(top_p, float) else VDR(top_p, 10)

    rng = SimpleRNG(seed)
    ids = list(prompt_ids)

    for _ in range(max_tokens):
        # use last SEQ_LEN tokens as context
        context = ids[-SEQ_LEN:] if len(ids) >= SEQ_LEN else ids
        # pad if shorter than SEQ_LEN
        if len(context) < SEQ_LEN:
            context = [0] * (SEQ_LEN - len(context)) + context

        logits = model.forward_last_logits(context)

        # surrogate softmax
        shift = logits[0]
        for k in range(1, len(logits)):
            if logits[k] < shift:
                shift = logits[k]
        probs = softmax_surrogate_square(logits, shift=shift)

        # sample
        if strategy == "greedy":
            next_id = sample_greedy(probs)
        elif strategy == "sample":
            next_id = sample_categorical(probs, rng)
        elif strategy == "top_k":
            next_id = sample_top_k(probs, top_k, rng)
        elif strategy == "nucleus":
            next_id = sample_nucleus(probs, top_p, rng)
        else:
            raise ValueError("Unknown strategy: %s" % strategy)

        ids.append(next_id)

    return ids


def generate_text(model, prompt, max_tokens, vocab, inv_vocab,
                  strategy="greedy", seed=42, top_k=3, top_p=None):
    """
    Generate text from a text prompt.

    I: model, prompt string, max tokens, vocab dicts, strategy config
    O: full generated text string
    """
    prompt_ids = tokenize_safe(prompt, vocab)
    ids = generate_ids(model, prompt_ids, max_tokens, strategy, seed, top_k, top_p)
    return " ".join(inv_vocab[i] for i in ids)


def tokenize_safe(text, vocab):
    """
    Tokenize with error handling for unknown tokens.

    I: text string, vocab dict
    O: list of int ids
    """
    ids = []
    for word in text.split():
        if word not in vocab:
            raise ValueError("Unknown token: '%s'" % word)
        ids.append(vocab[word])
    return ids


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def generate_greedy(model, prompt, max_tokens, vocab, inv_vocab):
    """Greedy generation."""
    return generate_text(model, prompt, max_tokens, vocab, inv_vocab, strategy="greedy")


def generate_sampled(model, prompt, max_tokens, vocab, inv_vocab, seed=42):
    """Sampled generation."""
    return generate_text(model, prompt, max_tokens, vocab, inv_vocab,
                         strategy="sample", seed=seed)


def generate_top_k_text(model, prompt, max_tokens, vocab, inv_vocab, k=3, seed=42):
    """Top-k generation."""
    return generate_text(model, prompt, max_tokens, vocab, inv_vocab,
                         strategy="top_k", seed=seed, top_k=k)


def generate_nucleus_text(model, prompt, max_tokens, vocab, inv_vocab, p=None, seed=42):
    """Nucleus generation."""
    return generate_text(model, prompt, max_tokens, vocab, inv_vocab,
                         strategy="nucleus", seed=seed, top_p=p)


def show_generation(model, prompt, max_tokens, vocab, inv_vocab,
                    strategy="greedy", seed=42, top_k=3, top_p=None):
    """
    Generate with step-by-step printing.

    I: model, prompt, config
    O: generated text
    S: prints each step with probabilities
    """
    if top_p is None:
        top_p = VDR(9, 10)

    rng = SimpleRNG(seed)
    prompt_ids = tokenize_safe(prompt, vocab)
    ids = list(prompt_ids)

    print("prompt: %s" % prompt)
    print("strategy: %s" % strategy)
    print()

    for step in range(max_tokens):
        context = ids[-SEQ_LEN:] if len(ids) >= SEQ_LEN else ids
        if len(context) < SEQ_LEN:
            context = [0] * (SEQ_LEN - len(context)) + context

        logits = model.forward_last_logits(context)

        shift = logits[0]
        for k in range(1, len(logits)):
            if logits[k] < shift:
                shift = logits[k]
        probs = softmax_surrogate_square(logits, shift=shift)

        if strategy == "greedy":
            next_id = sample_greedy(probs)
        elif strategy == "sample":
            next_id = sample_categorical(probs, rng)
        elif strategy == "top_k":
            next_id = sample_top_k(probs, top_k, rng)
        elif strategy == "nucleus":
            next_id = sample_nucleus(probs, top_p, rng)
        else:
            next_id = sample_greedy(probs)

        prob_strs = ["%.4f" % probs[i].to_float() for i in range(len(probs))]
        print("  step %d: [%s] -> %s (%.4f)" % (
            step + 1,
            ", ".join(prob_strs),
            inv_vocab[next_id],
            probs[next_id].to_float(),
        ))

        ids.append(next_id)

    result = " ".join(inv_vocab[i] for i in ids)
    print()
    print("result: %s" % result)
    return result
