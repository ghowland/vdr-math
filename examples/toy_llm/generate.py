"""
examples/toy_llm/generate.py

Toy LLM text generation — exact VDR arithmetic.

Uses surrogate softmax for probability computation.
Every sampling decision uses exact rational CDF comparison.
Same seed = same output on any platform.

    from generate import generate_text

    generate_text(model, "the cat", max_tokens=6)
"""

from vdr.core import VDR
from vdr.linalg import Vec
from vdr.ml.rng import VDRRandom

from model import surrogate_softmax_probs
from data import build_vocab, invert_vocab, detokenize
import config as cfg


# -- sampling strategies --------------------------------------------------

def sample_categorical(probs, rng):
    """
    Sample from exact probability distribution via CDF comparison.

    Probabilities are exact VDR rationals summing to exactly 1.
    Random threshold is exact rational from deterministic PRNG.
    Comparison is exact — no float rounding at decision boundary.

    I: probs Vec (sums to exactly 1), rng VDRRandom
    O: int — sampled index
    """
    threshold = rng.random()
    cumulative = VDR(0)
    for i in range(len(probs)):
        cumulative = cumulative + probs[i]
        if cumulative > threshold:
            return i
    return len(probs) - 1


def sample_greedy(probs):
    """
    Greedy decoding: return index of highest probability.

    Ties broken by lowest index. Comparison is exact.

    I: probs Vec
    O: int — argmax index
    """
    best_idx = 0
    best_val = probs[0]
    for i in range(1, len(probs)):
        if probs[i] > best_val:
            best_val = probs[i]
            best_idx = i
    return best_idx


def sample_top_k(probs, k, rng):
    """
    Top-k sampling: zero all but k highest, renormalize, sample.

    Renormalized probabilities sum to exactly 1.

    I: probs Vec, k int, rng VDRRandom
    O: int — sampled index
    """
    n = len(probs)
    if k >= n:
        return sample_categorical(probs, rng)

    # find top-k indices by exact comparison
    indexed = [(probs[i], i) for i in range(n)]
    indexed.sort(key=lambda pair: pair[0], reverse=True)
    top_indices = set(indexed[j][1] for j in range(k))

    # zero out non-top-k
    filtered = []
    for i in range(n):
        if i in top_indices:
            filtered.append(probs[i])
        else:
            filtered.append(VDR(0))

    # renormalize
    total = VDR(0)
    for i in range(n):
        total = total + filtered[i]

    if total == VDR(0):
        # degenerate: uniform over top-k
        for j in range(k):
            idx = indexed[j][1]
            filtered[idx] = VDR(1, k)
        return sample_categorical(Vec(filtered), rng)

    renormed = [filtered[i] / total for i in range(n)]
    return sample_categorical(Vec(renormed), rng)


def sample_nucleus(probs, p_threshold, rng):
    """
    Nucleus (top-p) sampling: keep smallest set exceeding threshold,
    renormalize, sample.

    I: probs Vec, p_threshold VDR, rng VDRRandom
    O: int — sampled index
    """
    n = len(probs)

    # sort descending by exact comparison
    indexed = [(probs[i], i) for i in range(n)]
    indexed.sort(key=lambda pair: pair[0], reverse=True)

    # accumulate until exceeding threshold
    cumulative = VDR(0)
    keep_indices = set()
    for prob_val, idx in indexed:
        keep_indices.add(idx)
        cumulative = cumulative + prob_val
        if cumulative >= p_threshold:
            break

    # zero out non-kept
    filtered = []
    for i in range(n):
        if i in keep_indices:
            filtered.append(probs[i])
        else:
            filtered.append(VDR(0))

    # renormalize
    total = VDR(0)
    for i in range(n):
        total = total + filtered[i]

    if total == VDR(0):
        return sample_categorical(probs, rng)

    renormed = [filtered[i] / total for i in range(n)]
    return sample_categorical(Vec(renormed), rng)


# -- generation core ------------------------------------------------------

def generate_ids(model, prompt_ids, max_tokens=10, strategy="greedy",
                 seed=1, top_k=3, top_p=None):
    """
    Autoregressive token generation.

    Every step:
      1. Forward pass on last SEQ_LEN tokens
      2. Surrogate softmax -> exact probabilities summing to 1
      3. Sample or select next token
      4. Append and repeat

    I: model, prompt_ids list[int], generation config
    O: list of int — prompt + generated ids
    """
    rng = VDRRandom(seed=seed)
    ids = list(prompt_ids)

    for _ in range(max_tokens):
        context = ids[-cfg.SEQ_LEN:]

        # pad if shorter than SEQ_LEN
        while len(context) < cfg.SEQ_LEN:
            context = [context[0]] + context

        # forward + surrogate softmax
        logits = model.forward_last_logits(context)
        probs = surrogate_softmax_probs(logits)

        # select next token
        if strategy == "greedy":
            next_id = sample_greedy(probs)
        elif strategy == "sample":
            next_id = sample_categorical(probs, rng)
        elif strategy == "top_k":
            next_id = sample_top_k(probs, top_k, rng)
        elif strategy == "nucleus":
            if top_p is None:
                top_p = VDR(9, 10)
            next_id = sample_nucleus(probs, top_p, rng)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        ids.append(next_id)

    return ids


# -- text-level interface -------------------------------------------------

def tokenize_safe(text, vocab):
    """
    Tokenize with error handling for unknown tokens.

    I: text string, vocab dict
    O: list of int
    Raises ValueError for unknown tokens.
    """
    ids = []
    for word in text.split():
        if word not in vocab:
            raise ValueError(
                f"Token '{word}' not in vocabulary. "
                f"Known: {list(vocab.keys())}"
            )
        ids.append(vocab[word])
    return ids


def generate_text(model, prompt="the cat", max_tokens=6, strategy="greedy",
                  seed=1, top_k=3, top_p=None, corpus=None):
    """
    Generate text from a prompt string.

    I: model, prompt string, generation config
    O: string — full text including prompt
    """
    if corpus is None:
        corpus = cfg.CORPUS

    vocab = build_vocab(corpus)
    inv_vocab = invert_vocab(vocab)

    prompt_ids = tokenize_safe(prompt, vocab)
    all_ids = generate_ids(
        model, prompt_ids, max_tokens=max_tokens,
        strategy=strategy, seed=seed, top_k=top_k, top_p=top_p,
    )
    return detokenize(all_ids, inv_vocab)


# -- convenience wrappers -------------------------------------------------

def generate_greedy(model, prompt="the cat", max_tokens=6, corpus=None):
    """Greedy generation."""
    return generate_text(
        model, prompt, max_tokens=max_tokens,
        strategy="greedy", corpus=corpus,
    )


def generate_sampled(model, prompt="the cat", max_tokens=6, seed=1,
                     corpus=None):
    """Sampled generation."""
    return generate_text(
        model, prompt, max_tokens=max_tokens,
        strategy="sample", seed=seed, corpus=corpus,
    )


def generate_top_k_text(model, prompt="the cat", max_tokens=6, k=3,
                        seed=1, corpus=None):
    """Top-k generation."""
    return generate_text(
        model, prompt, max_tokens=max_tokens,
        strategy="top_k", top_k=k, seed=seed, corpus=corpus,
    )


def generate_nucleus_text(model, prompt="the cat", max_tokens=6,
                          p=None, seed=1, corpus=None):
    """Nucleus (top-p) generation."""
    if p is None:
        p = VDR(9, 10)
    return generate_text(
        model, prompt, max_tokens=max_tokens,
        strategy="nucleus", top_p=p, seed=seed, corpus=corpus,
    )


# -- display helper -------------------------------------------------------

def show_generation(model, prompt="the cat", max_tokens=6, strategy="greedy",
                    seed=1, top_k=3, top_p=None, corpus=None):
    """
    Generate and display with probability details at each step.

    Prints context, full distribution (exact fractions), and selection.
    """
    if corpus is None:
        corpus = cfg.CORPUS

    vocab = build_vocab(corpus)
    inv_vocab = invert_vocab(vocab)

    prompt_ids = tokenize_safe(prompt, vocab)
    rng = VDRRandom(seed=seed)
    ids = list(prompt_ids)

    print(f"Prompt: '{prompt}'")
    print(f"Strategy: {strategy}")
    print(f"Prompt ids: {prompt_ids}")
    print()

    for step in range(max_tokens):
        context = ids[-cfg.SEQ_LEN:]
        while len(context) < cfg.SEQ_LEN:
            context = [context[0]] + context

        logits = model.forward_last_logits(context)
        probs = surrogate_softmax_probs(logits)

        # display distribution
        print(f"Step {step + 1}:")
        print(f"  Context: {[inv_vocab[i] for i in context]}")
        print(f"  Distribution:")
        for i in range(len(probs)):
            token_str = inv_vocab.get(i, f"<{i}>")
            p_frac = probs[i].to_fraction()
            p_approx = float(probs[i])
            print(f"    {token_str:8s}: {p_frac}  (~{p_approx:.4f})")

        # select
        if strategy == "greedy":
            next_id = sample_greedy(probs)
        elif strategy == "sample":
            next_id = sample_categorical(probs, rng)
        elif strategy == "top_k":
            next_id = sample_top_k(probs, top_k, rng)
        elif strategy == "nucleus":
            if top_p is None:
                top_p = VDR(9, 10)
            next_id = sample_nucleus(probs, top_p, rng)
        else:
            next_id = sample_greedy(probs)

        print(f"  Selected: {inv_vocab[next_id]} (id={next_id})")
        print()

        ids.append(next_id)

    result = detokenize(ids, inv_vocab)
    print(f"Full output: '{result}'")
    return result

