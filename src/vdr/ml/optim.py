"""
vdr.ml.optim — Exact optimizers for VDR neural networks.

    from vdr.ml.optim import SGD, Momentum

    opt = SGD(model.parameters(), lr=VDR(1, 100))
    opt.zero_grad()
    loss = ...
    model.backward(grad)
    opt.step()  # w = w - lr * grad, exact

All parameter updates exact VDR arithmetic.
No float accumulation in optimizer state.
"""

from __future__ import annotations
from typing import List, Union

from vdr.core import VDR
from vdr.linalg import Vec, Mat

__all__ = [
    "SGD",
    "Momentum",
]


def _to_vdr(x):
    if isinstance(x, VDR):
        return x
    if isinstance(x, int):
        return VDR(x)
    raise TypeError("Expected VDR or int")


class SGD:
    """
    Stochastic Gradient Descent optimizer.

    update: w = w - lr * grad

    I: list of parameters (VecParam or MatParam), learning rate (VDR)

        opt = SGD(model.parameters(), lr=VDR(1, 100))
        opt.step()
    """

    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = _to_vdr(lr)

    def zero_grad(self):
        """Zero all parameter gradients."""
        for p in self.params:
            p.zero_grad()

    def step(self):
        """Apply one SGD update to all parameters."""
        for p in self.params:
            p.step(self.lr)


class Momentum:
    """
    SGD with momentum optimizer.

    v = beta * v + grad
    w = w - lr * v

    I: params list, learning rate (VDR), momentum beta (VDR, default 9/10)

        opt = Momentum(model.parameters(), lr=VDR(1, 100), beta=VDR(9, 10))
        opt.step()
    """

    def __init__(self, params, lr, beta=None):
        self.params = list(params)
        self.lr = _to_vdr(lr)
        self.beta = _to_vdr(beta) if beta is not None else VDR(9, 10)

        # initialize velocity buffers
        self._velocities = []
        for p in self.params:
            self._velocities.append(p.zeros_like())

    def zero_grad(self):
        """Zero all parameter gradients."""
        for p in self.params:
            p.zero_grad()

    def step(self):
        """Apply one momentum update to all parameters."""
        for i, p in enumerate(self.params):
            v = self._velocities[i]

            # v = beta * v + grad
            v = p.combine_scaled(v, self.beta, p.grad, VDR(1))
            self._velocities[i] = v

            # w = w - lr * v
            p.apply_update(v, self.lr)
