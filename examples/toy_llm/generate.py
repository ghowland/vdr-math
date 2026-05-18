"""
examples/toy_llm/generate.py

Text generation with surrogate softmax.
"""

from vdr.core import VDR
from vdr.linalg import Vec
from vdr.ml.rng import VDRRandom

from model import surrogate_softmax_probs
from data import build_vocab, invert_vocab, detokenize
import config as cfg


def sample_categorical(probs, rng):
    """Sample via exact CDF comparison."""
    threshold = rng.random()
    cumulative = VDR(0)
    for i in range(len(probs)):
        cumulative = cumulative + probs[i]
        if cumulative > threshold:
            return i
    return len(probs) - 1


def sample_greedy(probs):
    """Argmax, ties to lowest index."""
    best_idx = 0
    best_val = probs[0]
    for i in range(1, len(probs)):
        if probs[i] > best_val:
            best_val = probs[i]
            best_idx = i
    return best_idx


def sample_top_k(probs, k, rng):
    """Top-k with exact renormalization."""
    n = len(probs)
    if k >= n:
        return sample_categorical(probs, rng)

    indexed = [(probs[i], i) for i in range(n)]
    indexed.sort(key=lambda pair: pair[0], reverse=True)
    top_indices = set(indexed[j][1] for j in range(k))

    filtered = [probs[i] if i in top_indices else VDR(0) for i in range(n)]
    total = VDR(0)
    for f in filtered:
        total = total + f

    if total == VDR(0):
        for j in range(k):
            filtered[indexed[j][1]] = VDR(1, k)
        return sample_categorical(Vec(filtered), rng)

    renormed = [filtered[i] / total for i in range(n)]
    return sample_categorical(Vec(renormed), rng)


def sample_nucleus(probs, p_threshold, rng):
    """Top-p with exact renormalization."""
    n = len(probs)
    indexed = [(probs[i], i) for i in range(n)]
    indexed.sort(key=lambda pair: pair[0], reverse=True)

    cumulative = VDR(0)
    keep = set()
    for prob_val, idx in indexed:
        keep.add(idx)
        cumulative = cumulative + prob_val
        if cumulative >= p_threshold:
            break

    filtered = [probs[i] if i in keep else VDR(0) for i in range(n)]
    total = VDR(0)
    for f in filtered:
        total = total + f

    if total == VDR(0):
        return sample_categorical(probs, rng)

    renormed = [filtered[i] / total for i in range(n)]
    return sample_categorical(Vec(renormed), rng)


def generate_ids(model, prompt_ids, max_tokens=10, strategy="greedy",
                 seed=1, top_k=3, top_p=None):
    """Autoregressive generation."""
    rng = VDRRandom(seed=seed)
    ids = list(prompt_ids)

    for _ in range(max_tokens):
        context = ids[-cfg.SEQ_LEN:]
        while len(context) < cfg.SEQ_LEN:
            context = [context[0]] + context

        logits = model.forward_last_logits(context)
        probs = surrogate_softmax_probs(logits)

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


def tokenize_safe(text, vocab):
    ids = []
    for word in text.split():
        if word not in vocab:
            raise ValueError(f"Token '{word}' not in vocabulary. Known: {list(vocab.keys())}")
        ids.append(vocab[word])
    return ids


def generate_text(model, prompt="the cat", max_tokens=6, strategy="greedy",
                  seed=1, top_k=3, top_p=None, corpus=None):
    if corpus is None:
        corpus = cfg.CORPUS
    vocab = build_vocab(corpus)
    inv_vocab = invert_vocab(vocab)
    prompt_ids = tokenize_safe(prompt, vocab)
    all_ids = generate_ids(model, prompt_ids, max_tokens=max_tokens,
                           strategy=strategy, seed=seed, top_k=top_k, top_p=top_p)
    return detokenize(all_ids, inv_vocab)


def generate_greedy(model, prompt="the cat", max_tokens=6, corpus=None):
    return generate_text(model, prompt, max_tokens=max_tokens, strategy="greedy", corpus=corpus)


def generate_sampled(model, prompt="the cat", max_tokens=6, seed=1, corpus=None):
    return generate_text(model, prompt, max_tokens=max_tokens, strategy="sample", seed=seed, corpus=corpus)


def generate_top_k_text(model, prompt="the cat", max_tokens=6, k=3, seed=1, corpus=None):
    return generate_text(model, prompt, max_tokens=max_tokens, strategy="top_k", top_k=k, seed=seed, corpus=corpus)


def generate_nucleus_text(model, prompt="the cat", max_tokens=6, p=None, seed=1, corpus=None):
    if p is None:
        p = VDR(9, 10)
    return generate_text(model, prompt, max_tokens=max_tokens, strategy="nucleus", top_p=p, seed=seed, corpus=corpus)


def show_generation(model, prompt="the cat", max_tokens=6, strategy="greedy",
                    seed=1, top_k=3, top_p=None, corpus=None):
    """Generate with probability display at each step."""
    if corpus is None:
        corpus = cfg.CORPUS
    vocab = build_vocab(corpus)
    inv_vocab = invert_vocab(vocab)
    prompt_ids = tokenize_safe(prompt, vocab)
    rng = VDRRandom(seed=seed)
    ids = list(prompt_ids)

    print(f"Prompt: '{prompt}'")
    print(f"Strategy: {strategy}")
    print()

    for step in range(max_tokens):
        context = ids[-cfg.SEQ_LEN:]
        while len(context) < cfg.SEQ_LEN:
            context = [context[0]] + context

        logits = model.forward_last_logits(context)
        probs = surrogate_softmax_probs(logits)

        print(f"Step {step + 1}:")
        print(f"  Context: {[inv_vocab[i] for i in context]}")
        for i in range(len(probs)):
            tok = inv_vocab.get(i, f"<{i}>")
            p_approx = float(probs[i])
            print(f"    {tok:8s}: ~{p_approx:.4f}")

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

        print(f"  -> {inv_vocab[next_id]}")
        print()
        ids.append(next_id)

    result = detokenize(ids, inv_vocab)
    print(f"Output: '{result}'")
    return result
