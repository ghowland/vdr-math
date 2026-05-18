"""
examples/toy_llm/data.py

Toy LLM data preparation. Unchanged from previous version.
"""

from vdr.core import VDR
from vdr.linalg import Vec


def build_vocab(corpus):
    vocab = {}
    for word in corpus.split():
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def invert_vocab(vocab):
    return {v: k for k, v in vocab.items()}


def tokenize(text, vocab):
    return [vocab[w] for w in text.split()]


def detokenize(ids, inv_vocab):
    return " ".join(inv_vocab[i] for i in ids)


def make_windows(ids, seq_len):
    windows = []
    for i in range(len(ids) - seq_len):
        context = ids[i:i + seq_len]
        target = ids[i + seq_len]
        windows.append((context, target))
    return windows


def one_hot_target(target_id, vocab_size):
    data = [VDR(0)] * vocab_size
    data[target_id] = VDR(1)
    return Vec(data)


def build_dataset(corpus, seq_len):
    vocab = build_vocab(corpus)
    inv_vocab = invert_vocab(vocab)
    ids = tokenize(corpus, vocab)
    windows = make_windows(ids, seq_len)
    return windows, vocab, inv_vocab, len(vocab)
