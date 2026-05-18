"""
vdr.ml.tensor — Batch tensor operations for VDR.

    from vdr.ml.tensor import Tensor3D, batched_matvec

    t = Tensor3D([[row1_vecs], [row2_vecs]])
    results = batched_matvec(matrices, vectors)

Tensor3D is batch x sequence x dimension.
All operations exact VDR arithmetic across batches.
"""

from __future__ import annotations
from typing import List, Sequence

from vdr.core import VDR
from vdr.linalg import Vec, Mat

__all__ = [
    "Tensor3D",
    "batched_matvec",
    "rowwise_add_bias",
    "masked_fill_rows",
    "reduce_sum_rows",
    "stack_vecs",
    "unstack_vecs",
]


def _to_vdr(x):
    if isinstance(x, VDR):
        return x
    if isinstance(x, int):
        return VDR(x)
    raise TypeError("Expected VDR or int")


class Tensor3D:
    """
    3D tensor: batch x sequence x dimension.

    Stored as list of list of Vec.

        data = [
            [Vec.from_ints([1,2]), Vec.from_ints([3,4])],   # batch 0
            [Vec.from_ints([5,6]), Vec.from_ints([7,8])],   # batch 1
        ]
        t = Tensor3D(data)
        t.shape  # (2, 2, 2)
    """

    __slots__ = ("_data",)

    def __init__(self, data):
        """
        I: list of list of Vec (batch x sequence)
        """
        self._data = data

    @classmethod
    def zero(cls, b, n, d):
        """Zero tensor of shape (b, n, d)."""
        return cls([
            [Vec.zero(d) for _ in range(n)]
            for _ in range(b)
        ])

    @property
    def shape(self):
        """(batch, sequence, dimension)."""
        b = len(self._data)
        if b == 0:
            return (0, 0, 0)
        n = len(self._data[0])
        if n == 0:
            return (b, 0, 0)
        d = len(self._data[0][0])
        return (b, n, d)

    @property
    def batch(self):
        return len(self._data)

    @property
    def nrows(self):
        """Sequence length."""
        if not self._data:
            return 0
        return len(self._data[0])

    @property
    def dim(self):
        """Feature dimension."""
        if not self._data or not self._data[0]:
            return 0
        return len(self._data[0][0])

    def __getitem__(self, idx):
        """
        Index by batch: t[i] returns list of Vec (sequence).
        Index by (batch, seq): t[i, j] returns Vec.
        Index by (batch, seq, dim): t[i, j, k] returns VDR.
        """
        if isinstance(idx, tuple):
            if len(idx) == 2:
                return self._data[idx[0]][idx[1]]
            if len(idx) == 3:
                return self._data[idx[0]][idx[1]][idx[2]]
        return self._data[idx]

    def __len__(self):
        return len(self._data)

    def to_fractions(self):
        """Export as nested list of Fraction."""
        return [
            [v.to_fractions() for v in batch]
            for batch in self._data
        ]

    def __repr__(self):
        return "Tensor3D(shape=%s)" % (self.shape,)


def batched_matvec(mats, vecs):
    """
    Batched matrix-vector product.

    I: list of Mat, list of Vec (same length)
    O: list of Vec where result[i] = mats[i] @ vecs[i]

        results = batched_matvec([M1, M2], [v1, v2])
    """
    if len(mats) != len(vecs):
        raise ValueError("mats and vecs must have same length")
    return [m.matvec(v) for m, v in zip(mats, vecs)]


def rowwise_add_bias(rows, bias):
    """
    Add bias vector to each row.

    I: list of Vec (rows), bias Vec
    O: list of Vec where result[i] = rows[i] + bias

        biased = rowwise_add_bias([v1, v2, v3], bias)
    """
    return [row + bias for row in rows]


def masked_fill_rows(rows, mask, fill=None):
    """
    Apply element-wise mask to rows of Vec.

    I: list of Vec, mask as list of list of bool,
       fill value (VDR, default 0)
    O: list of Vec with masked positions replaced by fill

        masked = masked_fill_rows(rows, mask, VDR(-1000))
    """
    if fill is None:
        fill = VDR(0)

    result = []
    for i in range(len(rows)):
        row = []
        for j in range(len(rows[i])):
            if mask[i][j]:
                row.append(rows[i][j])
            else:
                row.append(fill)
        result.append(Vec(row))
    return result


def reduce_sum_rows(rows):
    """
    Element-wise sum across a list of Vec.

    I: list of Vec (all same dimension)
    O: Vec where result[j] = sum_i rows[i][j]

        total = reduce_sum_rows([v1, v2, v3])
    """
    if not rows:
        return Vec([])
    dim = len(rows[0])
    result = [VDR(0)] * dim
    for row in rows:
        for j in range(dim):
            result[j] = result[j] + row[j]
    return Vec(result)


def stack_vecs(vecs):
    """
    Stack list of Vec into a Mat (each Vec becomes a row).

    I: list of Vec (all same dimension)
    O: Mat

        m = stack_vecs([Vec.from_ints([1,2]), Vec.from_ints([3,4])])
    """
    return Mat([list(v) for v in vecs])


def unstack_vecs(m):
    """
    Unstack Mat into list of row Vec.

    I: Mat
    O: list of Vec

        vecs = unstack_vecs(m)
    """
    return [m.row(i) for i in range(m.nrows)]
