"""
vdr.ml.autodiff — Exact automatic differentiation via computation graph.

    from vdr.ml.autodiff import Node, ensure_node, relu, mse_loss

    a = Node(VDR(3))
    b = Node(VDR(4))
    c = a * b + a
    c.backward()
    print(a.grad)  # VDR(5) = d/da(a*b + a) = b + 1

All gradients exact VDR rationals via reverse-mode AD.
Chain rule applied symbolically — no numerical differentiation.
Node values and gradients projected to basis frame to avoid D mixing.
"""

from __future__ import annotations
from typing import Callable, Optional, Sequence, Set, List, Union

from vdr.core import VDR
from vdr.linalg import Vec

from vdr.basis import to_qbasis

__all__ = [
    "Node",
    "ensure_node",
    "relu",
    "sum_nodes",
    "mean_nodes",
    "mse_loss",
    "dot_nodes",
    "linear_node",
    "zero_grads",
    "value_of_vec",
    "grad_of_vec",
]

ScalarLike = Union[VDR, int]


def _to_vdr(x):
    if isinstance(x, VDR):
        return x
    if isinstance(x, int):
        return VDR(x)
    raise TypeError("Expected VDR or int")


def _basis_vdr(x):
    """Project a VDR or int to basis frame."""
    from vdr.basis import to_qbasis
    return to_qbasis(_to_vdr(x))


def _basis_zero():
    """Return VDR(0) in basis frame."""
    from vdr.basis import to_qbasis
    return to_qbasis(to_qbasis(0))


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    """
    Computation graph node for reverse-mode automatic differentiation.

    Each node holds:
        - value: exact VDR result of the forward computation
        - grad: accumulated gradient (VDR), initialized to basis-frame zero
        - _backward: closure that propagates gradient to children
        - _children: set of parent nodes for topological sort

        a = Node(VDR(3))
        b = Node(VDR(4))
        c = a * b
        c.backward()
        # a.grad == VDR(4), b.grad == VDR(3)
    """

    __slots__ = ("value", "grad", "_backward", "_children")

    def __init__(self, value, children=None, backward_fn=None):
        self.value = _to_vdr(value)
        self.grad = _basis_zero()
        self._children = set(children) if children else set()
        self._backward = backward_fn if backward_fn else lambda: None

    def zero_grad(self):
        """Reset gradient to basis-frame zero."""
        self.grad = _basis_zero()

    # -- arithmetic --------------------------------------------------------

    def __add__(self, other):
        other = ensure_node(other)
        out = Node(self.value + other.value, children=[self, other])

        s = self
        o = other

        def _backward():
            s.grad = s.grad + out.grad
            o.grad = o.grad + out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = ensure_node(other)
        out = Node(self.value - other.value, children=[self, other])

        s = self
        o = other

        def _backward():
            s.grad = s.grad + out.grad
            o.grad = o.grad - out.grad

        out._backward = _backward
        return out

    def __rsub__(self, other):
        other = ensure_node(other)
        return other.__sub__(self)

    def __mul__(self, other):
        other = ensure_node(other)
        out = Node(self.value * other.value, children=[self, other])

        s = self
        o = other

        def _backward():
            s.grad = s.grad + out.grad * o.value
            o.grad = o.grad + out.grad * s.value

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = ensure_node(other)
        out = Node(self.value / other.value, children=[self, other])

        s = self
        o = other

        def _backward():
            # d/da (a/b) = 1/b
            # d/db (a/b) = -a/b^2
            s.grad = s.grad + out.grad / o.value
            o.grad = o.grad - out.grad * s.value / (o.value * o.value)

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        other = ensure_node(other)
        return other.__truediv__(self)

    def __neg__(self):
        out = Node(-self.value, children=[self])

        s = self

        def _backward():
            s.grad = s.grad - out.grad

        out._backward = _backward
        return out

    def __pow__(self, n):
        """Power by positive integer n."""
        if not isinstance(n, int) or n < 0:
            raise ValueError("Only non-negative integer powers supported")
        if n == 0:
            return Node(VDR(1))

        # compute value
        val = VDR(1)
        for _ in range(n):
            val = val * self.value

        out = Node(val, children=[self])

        s = self
        exp = n

        def _backward():
            # d/dx (x^n) = n * x^(n-1)
            if exp == 0:
                return
            deriv = _basis_vdr(VDR(exp))
            x_pow = VDR(1)
            for _ in range(exp - 1):
                x_pow = x_pow * s.value
            s.grad = s.grad + out.grad * deriv * x_pow

        out._backward = _backward
        return out

    # -- backward ----------------------------------------------------------

    def backward(self):
        """
        Reverse-mode automatic differentiation.

        Sets self.grad = 1 (in basis frame), then propagates through
        the graph in reverse topological order.
        """
        topo = []
        visited = set()
        _build_topo(self, visited, topo)

        self.grad = _basis_vdr(VDR(1))
        for node in reversed(topo):
            node._backward()

    # -- display -----------------------------------------------------------

    def __repr__(self):
        return "Node(%s, grad=%s)" % (self.value, self.grad)


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------

def _build_topo(node, visited, topo):
    """Build topological ordering for backward pass."""
    nid = id(node)
    if nid in visited:
        return
    visited.add(nid)
    for child in node._children:
        _build_topo(child, visited, topo)
    topo.append(node)


# ---------------------------------------------------------------------------
# Node construction helpers
# ---------------------------------------------------------------------------

def ensure_node(x):
    """Convert scalar to Node if needed. Projects to basis frame."""
    if isinstance(x, Node):
        return x
    if isinstance(x, VDR):
        return Node(x)
    if isinstance(x, int):
        return Node(_basis_vdr(VDR(x)))
    raise TypeError("Cannot convert %s to Node" % type(x).__name__)


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def relu(x):
    """
    ReLU as autodiff Node.

    I: Node
    O: Node with relu applied and backward wired

        a = Node(VDR(-3))
        b = relu(a)
        b.backward()
        # a.grad == VDR(0)
    """
    x = ensure_node(x)
    val = x.value if x.value > to_qbasis(0) else to_qbasis(0)
    out = Node(val, children=[x])

    inp = x

    def _backward():
        if inp.value > to_qbasis(0):
            inp.grad = inp.grad + out.grad
        # else: gradient is 0, nothing to add

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# Reduction functions
# ---------------------------------------------------------------------------

def sum_nodes(xs):
    """
    Sum a sequence of Nodes.

    I: list of Nodes
    O: Node representing the sum
    """
    if not xs:
        return Node(_basis_zero())
    result = xs[0]
    for x in xs[1:]:
        result = result + x
    return result


def mean_nodes(xs):
    """
    Mean of a sequence of Nodes.

    I: list of Nodes
    O: Node representing the mean
    """
    n = len(xs)
    if n == 0:
        return Node(_basis_zero())
    s = sum_nodes(xs)
    return s * Node(_basis_vdr(VDR(1, n)))


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def mse_loss(pred, target):
    """
    Mean squared error loss as autodiff graph.

    I: pred (list of Nodes), target (list of VDR or int)
    O: Node representing (1/n) * sum (pred_i - target_i)^2

        pred = [Node(VDR(1)), Node(VDR(2))]
        loss = mse_loss(pred, [VDR(0), VDR(3)])
        loss.backward()
    """
    n = len(pred)
    if n != len(target):
        raise ValueError("Dimension mismatch")

    terms = []
    for i in range(n):
        diff = pred[i] - ensure_node(target[i])
        terms.append(diff ** 2)

    return sum_nodes(terms) * Node(_basis_vdr(VDR(1, n)))


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def dot_nodes(a, b):
    """
    Dot product of two Node sequences.

    I: two lists of Nodes
    O: Node representing sum a_i * b_i
    """
    if len(a) != len(b):
        raise ValueError("Dimension mismatch")
    terms = [a[i] * b[i] for i in range(len(a))]
    return sum_nodes(terms)


def linear_node(weights, xs, bias):
    """
    Linear combination: sum(w_i * x_i) + bias.

    I: weights (list of VDR/int), xs (list of Nodes), bias (VDR/int)
    O: Node

        out = linear_node([VDR(1,2), VDR(3,4)], [x1, x2], VDR(1))
    """
    terms = []
    for w, x in zip(weights, xs):
        terms.append(ensure_node(w) * x)
    result = sum_nodes(terms)
    return result + ensure_node(bias)


def zero_grads(nodes):
    """Reset gradients of all nodes to basis-frame zero."""
    for n in nodes:
        n.zero_grad()


def value_of_vec(nodes):
    """Extract values from Node list as Vec."""
    return Vec([n.value for n in nodes])


def grad_of_vec(nodes):
    """Extract gradients from Node list as Vec."""
    return Vec([n.grad for n in nodes])
