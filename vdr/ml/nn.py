"""
vdr.ml.nn — Exact neural network layers and parameters.

    from vdr.ml.nn import Linear, ReLU, Sequential

    model = Sequential([
        Linear.from_ints([[1,2],[3,4]], [0,0]),
        ReLU(),
        Linear.from_ints([[1,0],[0,1]], [1,1]),
    ])
    output = model.forward(Vec.from_ints([1, 2]))
    model.backward(mse_grad(output, target))

All forward passes exact. All gradients exact via chain rule.
No float drift in training loops.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Union

from vdr.core import VDR
from vdr.linalg import Vec, Mat

__all__ = [
    "VecParam",
    "MatParam",
    "Module",
    "Linear",
    "ReLU",
    "Sequential",
    "FFN",
    "relu_scalar",
    "relu_prime_scalar",
]


def _to_vdr(x):
    if isinstance(x, VDR):
        return x
    if isinstance(x, int):
        return VDR(x)
    raise TypeError("Expected VDR or int")


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------

def relu_scalar(x):
    """
    ReLU on a single VDR: max(0, x).

    I: VDR
    O: VDR (0 if x < 0, x otherwise)
    """
    if x > VDR(0):
        return x
    return VDR(0)


def relu_prime_scalar(x):
    """
    ReLU derivative on a single VDR.

    I: VDR
    O: VDR(1) if x > 0, VDR(0) if x <= 0
    """
    if x > VDR(0):
        return VDR(1)
    return VDR(0)


# ---------------------------------------------------------------------------
# Trainable parameters
# ---------------------------------------------------------------------------

class VecParam:
    """
    Trainable vector parameter with gradient accumulator.

        p = VecParam(Vec.from_ints([1, 2, 3]), name="bias")
        p.zero_grad()
        p.step(lr=VDR(1, 100))
    """

    __slots__ = ("value", "grad", "name")

    def __init__(self, value, name=None):
        self.value = value
        self.grad = Vec.zero(len(value))
        self.name = name

    def zero_grad(self):
        """Reset gradient to zero."""
        self.grad = Vec.zero(len(self.value))

    def grad_like(self):
        """Return zero Vec with same shape as value."""
        return Vec.zero(len(self.value))

    def zeros_like(self):
        """Return zero Vec with same shape."""
        return Vec.zero(len(self.value))

    def step(self, lr):
        """SGD update: value = value - lr * grad."""
        lr = _to_vdr(lr)
        self.value = self.value - self.grad.scale(lr)

    def combine_scaled(self, a, a_scale, b, b_scale):
        """Weighted combination: a_scale * a + b_scale * b."""
        return Vec([
            _to_vdr(a_scale) * a[i] + _to_vdr(b_scale) * b[i]
            for i in range(len(a))
        ])

    def apply_update(self, update, lr):
        """Apply an arbitrary update vector: value = value - lr * update."""
        lr = _to_vdr(lr)
        self.value = self.value - update.scale(lr)

    def to_qbasis(self, bits=None):
        """Project value and reset grad in basis frame."""
        from vdr.basis import vec_to_qbasis
        self.value = vec_to_qbasis(self.value, bits)
        self.grad = Vec.zero(len(self.value))
        return self


class MatParam:
    """
    Trainable matrix parameter with gradient accumulator.

        p = MatParam(Mat.from_ints([[1,2],[3,4]]), name="weight")
        p.zero_grad()
        p.step(lr=VDR(1, 100))
    """

    __slots__ = ("value", "grad", "name")

    def __init__(self, value, name=None):
        self.value = value
        self.grad = Mat.zero(value.nrows, value.ncols)
        self.name = name

    def zero_grad(self):
        """Reset gradient to zero."""
        self.grad = Mat.zero(self.value.nrows, self.value.ncols)

    def grad_like(self):
        """Return zero Mat with same shape."""
        return Mat.zero(self.value.nrows, self.value.ncols)

    def zeros_like(self):
        """Return zero Mat with same shape."""
        return Mat.zero(self.value.nrows, self.value.ncols)

    def step(self, lr):
        """SGD update: value = value - lr * grad."""
        lr = _to_vdr(lr)
        self.value = self.value - self.grad.scale(lr)

    def combine_scaled(self, a, a_scale, b, b_scale):
        """Weighted combination of two Mat values."""
        return a.scale(a_scale) + b.scale(b_scale)

    def apply_update(self, update, lr):
        """Apply an arbitrary update matrix: value = value - lr * update."""
        lr = _to_vdr(lr)
        self.value = self.value - update.scale(lr)

    def to_qbasis(self, bits=None):
        """Project value and reset grad in basis frame."""
        from vdr.basis import mat_to_qbasis
        self.value = mat_to_qbasis(self.value, bits)
        self.grad = Mat.zero(self.value.nrows, self.value.ncols)
        return self


# ---------------------------------------------------------------------------
# Outer product helper
# ---------------------------------------------------------------------------

def _outer(a, b):
    """Outer product of two Vecs: M[i,j] = a[i] * b[j]."""
    rows = []
    for i in range(len(a)):
        row = []
        for j in range(len(b)):
            row.append(a[i] * b[j])
        rows.append(row)
    return Mat(rows)


# ---------------------------------------------------------------------------
# Module base
# ---------------------------------------------------------------------------

class Module:
    """Base class for neural network modules."""

    def parameters(self):
        """Return list of trainable parameters (VecParam or MatParam)."""
        return []

    def zero_grad(self):
        """Zero all parameter gradients."""
        for p in self.parameters():
            p.zero_grad()

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_out):
        raise NotImplementedError

    def to_qbasis(self, bits=None):
        """Project all parameters to basis frame. Override in subclasses."""
        for p in self.parameters():
            p.to_qbasis(bits)
        return self


# ---------------------------------------------------------------------------
# Linear layer
# ---------------------------------------------------------------------------

class Linear(Module):
    """
    Exact linear layer: y = W @ x + b.

    Forward: matrix-vector multiply + bias add.
    Backward: exact gradients for W, b, and input.

        layer = Linear.from_ints([[1,2],[3,4]], [0,0])
        y = layer.forward(Vec.from_ints([1, 1]))
    """

    def __init__(self, weight, bias, name=None):
        """
        I: weight as Mat, bias as Vec, optional name
        """
        self.weight = MatParam(weight, name=("%s.W" % name) if name else "W")
        self.bias = VecParam(bias, name=("%s.b" % name) if name else "b")
        self._last_input = None

    @classmethod
    def from_ints(cls, weight, bias, name=None):
        """Construct from integer lists."""
        return cls(Mat.from_ints(weight), Vec.from_ints(bias), name)

    @classmethod
    def from_fracs(cls, weight, bias, name=None):
        """Construct from fraction tuples."""
        return cls(Mat.from_fracs(weight), Vec.from_fracs(bias), name)

    def parameters(self):
        return [self.weight, self.bias]

    def forward(self, x):
        """
        y = W @ x + b.

        I: input Vec x
        O: output Vec y, exact
        S: caches input for backward
        """
        self._last_input = x
        return self.weight.value.matvec(x) + self.bias.value

    def backward(self, grad_out):
        """
        Backpropagate gradient through linear layer.

        Given dL/dy (grad_out as Vec):
            dL/dW = grad_out (outer) input
            dL/db = grad_out
            dL/dx = W^T @ grad_out

        I: grad_out Vec (gradient of loss w.r.t. output)
        O: grad_input Vec (gradient of loss w.r.t. input)
        S: accumulates into weight.grad and bias.grad
        """
        x = self._last_input

        # dL/dW += grad_out outer x
        grad_W = _outer(grad_out, x)
        self.weight.grad = self.weight.grad + grad_W

        # dL/db += grad_out
        self.bias.grad = self.bias.grad + grad_out

        # dL/dx = W^T @ grad_out
        grad_input = self.weight.value.T.matvec(grad_out)
        return grad_input


# ---------------------------------------------------------------------------
# ReLU activation
# ---------------------------------------------------------------------------

class ReLU(Module):
    """
    Exact ReLU activation: y_i = max(0, x_i).

    Forward: piecewise linear, exact.
    Backward: 0 or 1 per element, exact.
    """

    def __init__(self, name=None):
        self.name = name
        self._last_input = None

    def parameters(self):
        return []

    def forward(self, x):
        """
        y_i = max(0, x_i).

        I: input Vec
        O: output Vec, exact piecewise linear
        """
        self._last_input = x
        return Vec([relu_scalar(xi) for xi in x])

    def backward(self, grad_out):
        """
        dL/dx_i = dL/dy_i * relu'(x_i).

        relu'(x) = 1 if x > 0, 0 otherwise.

        I: grad_out Vec
        O: grad_input Vec, exact
        """
        x = self._last_input
        return Vec([
            grad_out[i] * relu_prime_scalar(x[i])
            for i in range(len(x))
        ])

    def to_qbasis(self, bits=None):
        """ReLU has no parameters."""
        return self


# ---------------------------------------------------------------------------
# Sequential container
# ---------------------------------------------------------------------------

class Sequential(Module):
    """
    Sequential composition of layers.

    Forward: passes through layers in order.
    Backward: passes gradient through layers in reverse order.

        model = Sequential([
            Linear.from_ints([[1,2],[3,4]], [0,0]),
            ReLU(),
            Linear.from_ints([[1,0],[0,1]], [1,1]),
        ])
    """

    def __init__(self, layers):
        self.layers = list(layers)

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def forward(self, x):
        """Pass x through all layers in order."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_out):
        """Pass gradient through all layers in reverse."""
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out

    def to_qbasis(self, bits=None):
        """Project all layers to basis frame."""
        for layer in self.layers:
            layer.to_qbasis(bits)
        return self


# ---------------------------------------------------------------------------
# FFN (Feed-Forward Network block)
# ---------------------------------------------------------------------------

class FFN(Module):
    """
    Two-layer feed-forward network with ReLU activation.

    y = W2 @ relu(W1 @ x + b1) + b2

    Common building block for transformers and MLPs.

        ffn = FFN(
            Linear.from_ints([[1,2],[3,4]], [0,0]),
            ReLU(),
            Linear.from_ints([[1,0],[0,1]], [1,1]),
        )
    """

    def __init__(self, l1, act, l2):
        self.l1 = l1
        self.act = act
        self.l2 = l2

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

    def zero_grad(self):
        self.l1.zero_grad()
        self.l2.zero_grad()

    def forward(self, x):
        h = self.l1.forward(x)
        h = self.act.forward(h)
        return self.l2.forward(h)

    def backward(self, grad_out):
        g = self.l2.backward(grad_out)
        g = self.act.backward(g)
        return self.l1.backward(g)

    def to_qbasis(self, bits=None):
        """Project both linear layers to basis frame."""
        self.l1.to_qbasis(bits)
        self.l2.to_qbasis(bits)
        return self
    