"""
Toy LLM data pipeline — tokenization, windowing, one-hot targets.
"""

from vdr.core import VDR
from vdr.linalg import Vec
from vdr.basis import to_qbasis


def build_vocab(corpus):
    """
    Build vocabulary from corpus, ordered by first appearance.

    I: text string
    O: {token: id} dict
    """
    vocab = {}
    for word in corpus.split():
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def invert_vocab(vocab):
    """
    Invert vocabulary mapping.

    I: {token: id} dict
    O: {id: token} dict
    """
    return {v: k for k, v in vocab.items()}


def tokenize(text, vocab):
    """
    Convert text to token ids.

    I: text string, vocab dict
    O: list of int token ids
    """
    return [vocab[w] for w in text.split()]


def detokenize(ids, inv_vocab):
    """
    Convert token ids back to text.

    I: list of int ids, inverse vocab dict
    O: space-joined text string
    """
    return " ".join(inv_vocab[i] for i in ids)


def make_windows(ids, seq_len):
    """
    Create sliding context windows with targets.

    I: full token id sequence, context length
    O: list of (context_ids, target_id) tuples

    Each window is seq_len input tokens predicting the next token.
    """
    windows = []
    for i in range(len(ids) - seq_len):
        context = ids[i:i + seq_len]
        target = ids[i + seq_len]
        windows.append((context, target))
    return windows


def one_hot_target(target_id, vocab_size):
    """
    Create one-hot target vector in basis frame.

    I: target index, vocab size
    O: Vec with basis-frame VDR(1) at target, VDR(0) elsewhere
    """
    data = []
    one = to_qbasis(VDR(1))
    zero = to_qbasis(VDR(0))
    for i in range(vocab_size):
        if i == target_id:
            data.append(one)
        else:
            data.append(zero)
    return Vec(data)


def build_dataset(corpus, seq_len):
    """
    Build complete dataset from corpus.

    I: corpus string, sequence length
    O: (windows, vocab, inv_vocab, vocab_size)
    """
    vocab = build_vocab(corpus)
    inv_vocab = invert_vocab(vocab)
    ids = tokenize(corpus, vocab)
    windows = make_windows(ids, seq_len)
    return windows, vocab, inv_vocab, len(vocab)
