"""
examples/toy_llm/data.py

Toy LLM data preparation.

    from data import build_dataset

    windows, vocab, inv_vocab, vocab_size = build_dataset("the cat sat on the mat", 4)
"""

from vdr.core import VDR
from vdr.linalg import Vec


def build_vocab(corpus):
    """
    Build vocabulary from corpus text.

    I: text string (space-separated tokens)
    O: dict {token: id} ordered by first appearance

        build_vocab("the cat sat on the mat")
        -> {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
    """
    vocab = {}
    for word in corpus.split():
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def invert_vocab(vocab):
    """
    Reverse vocabulary: {id: token}.

    I: vocab dict
    O: inverse dict
    """
    return {v: k for k, v in vocab.items()}


def tokenize(text, vocab):
    """
    Convert text to token ids.

    I: text string, vocab dict
    O: list of ints

        tokenize("the cat sat", vocab) -> [0, 1, 2]
    """
    return [vocab[w] for w in text.split()]


def detokenize(ids, inv_vocab):
    """
    Convert token ids back to text.

    I: list of ints, inverse vocab
    O: space-joined string
    """
    return " ".join(inv_vocab[i] for i in ids)


def make_windows(ids, seq_len):
    """
    Create sliding window training examples.

    Each window: (context of seq_len tokens, next token as target).

    I: full token id sequence, context length
    O: list of (context_ids, target_id)

        make_windows([0,1,2,3,0,4], 4)
        -> [([0,1,2,3], 0), ([1,2,3,0], 4)]
    """
    windows = []
    for i in range(len(ids) - seq_len):
        context = ids[i:i + seq_len]
        target = ids[i + seq_len]
        windows.append((context, target))
    return windows


def one_hot_target(target_id, vocab_size):
    """
    Exact one-hot vector.

    I: target index, vocabulary size
    O: Vec with VDR(1) at target, VDR(0) elsewhere

        one_hot_target(2, 5) -> Vec([0, 0, 1, 0, 0])
    """
    data = [VDR(0)] * vocab_size
    data[target_id] = VDR(1)
    return Vec(data)


def build_dataset(corpus, seq_len):
    """
    Build complete dataset from corpus text.

    I: corpus string, sequence length
    O: (windows, vocab, inv_vocab, vocab_size)

        windows, vocab, inv_vocab, vs = build_dataset("the cat sat on the mat", 4)
    """
    vocab = build_vocab(corpus)
    inv_vocab = invert_vocab(vocab)
    ids = tokenize(corpus, vocab)
    windows = make_windows(ids, seq_len)
    return windows, vocab, inv_vocab, len(vocab)

