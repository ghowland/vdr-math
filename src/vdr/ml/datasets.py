"""
vdr.ml.datasets — Tokenization and dataset utilities.

    from vdr.ml.datasets import tiny_text_dataset, one_hot

    windows, vocab, inv_vocab = tiny_text_dataset("hello world", seq_len=3)
    vec = one_hot(2, 10)  # Vec with 1 at index 2, 0 elsewhere

All token ids are integers. One-hot vectors are exact VDR.
"""

from __future__ import annotations
from typing import Dict, List, Sequence, Tuple

from vdr.core import VDR
from vdr.linalg import Vec

__all__ = [
    "build_vocab",
    "encode_tokens",
    "decode_tokens",
    "invert_vocab",
    "sliding_windows",
    "one_hot",
    "batchify_windows",
    "tiny_text_dataset",
]


def build_vocab(tokens):
    """
    Build vocabulary mapping from unique tokens.

    I: list of string tokens
    O: dict {token: id} sorted by first appearance

        vocab = build_vocab(["hello", "world", "hello"])
        # {"hello": 0, "world": 1}
    """
    vocab = {}
    for t in tokens:
        if t not in vocab:
            vocab[t] = len(vocab)
    return vocab


def encode_tokens(tokens, vocab):
    """
    Encode token strings to integer ids.

    I: list of strings, vocab dict
    O: list of ints

        encode_tokens(["hello", "world"], vocab) -> [0, 1]
    """
    return [vocab[t] for t in tokens]


def decode_tokens(ids, inv_vocab):
    """
    Decode integer ids to token strings.

    I: list of ints, inverse vocab dict
    O: list of strings
    """
    return [inv_vocab[i] for i in ids]


def invert_vocab(vocab):
    """
    Invert vocabulary: {token: id} -> {id: token}.

    I: vocab dict
    O: inverse dict

        inv = invert_vocab({"hello": 0, "world": 1})
        # {0: "hello", 1: "world"}
    """
    return {v: k for k, v in vocab.items()}


def sliding_windows(ids, seq_len):
    """
    Create sliding window training examples from token id sequence.

    Each window is (context, target) where context has seq_len tokens
    and target is the next token.

    I: list of token ids, sequence length
    O: list of (context_list, target_int) tuples

        sliding_windows([0, 1, 2, 3, 4], 3)
        -> [([0,1,2], 3), ([1,2,3], 4)]
    """
    windows = []
    for i in range(len(ids) - seq_len):
        context = ids[i:i + seq_len]
        target = ids[i + seq_len]
        windows.append((context, target))
    return windows


def one_hot(index, size):
    """
    One-hot vector: 1 at index, 0 elsewhere.

    I: index (int), vector size (int)
    O: Vec with VDR(1) at index and VDR(0) elsewhere

        one_hot(2, 5) -> Vec([VDR(0), VDR(0), VDR(1), VDR(0), VDR(0)])
    """
    data = [VDR(0)] * size
    data[index] = VDR(1)
    return Vec(data)


def batchify_windows(windows, vocab_size):
    """
    Convert sliding windows to training pairs with one-hot targets.

    I: list of (context, target) tuples, vocabulary size
    O: list of (context_ids, target_vec) tuples

        batches = batchify_windows(windows, vocab_size=10)
    """
    result = []
    for context, target in windows:
        target_vec = one_hot(target, vocab_size)
        result.append((context, target_vec))
    return result


def tiny_text_dataset(text, seq_len):
    """
    Build a tiny text dataset from a string.

    Splits on whitespace, builds vocab, creates sliding windows.

    I: text string, sequence length
    O: (windows, vocab, inv_vocab) tuple

        windows, vocab, inv_vocab = tiny_text_dataset(
            "the cat sat on the mat", seq_len=2
        )
    """
    tokens = text.split()
    vocab = build_vocab(tokens)
    inv_vocab = invert_vocab(vocab)
    ids = encode_tokens(tokens, vocab)
    windows = sliding_windows(ids, seq_len)
    return windows, vocab, inv_vocab
