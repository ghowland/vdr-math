"""
vdr.fn — Functional remainder for VDR objects.

Extends the remainder slot to hold a callable that produces VDR structure
on demand. This is how VDR handles square roots, trig, exp, log — every
depth gives a complete exact rational, not an approximation of a limit.

A functional remainder is:
    - a Python callable: f(depth: int) -> VDR
    - a name string for inspectability
    - optional metadata dict

The function is finite. The structure it produces at any depth is finite
and exact. Expansion is on demand via resolve(obj, depth).

    from vdr.fn import FnRemainder, resolve, make_newton_fn
    from vdr.core import VDR

    sqrt2_fn = make_newton_fn("sqrt2", lambda x: (x + VDR(2)/x) / VDR(2))
    obj = VDR(0, 1, sqrt2_fn)
    val = resolve(obj, depth=10)  # exact rational, >100 correct digits

Each depth is a complete exact value, not an approximation.
The depth parameter controls how many recursive expansion steps are taken.
"""

from __future__ import annotations
from fractions import Fraction
from typing import Callable, Optional, Dict, Any, Union, Sequence

from vdr.core import VDR, Remainder, VDRError, InvalidStructureError

__all__ = [
    "FnRemainder",
    "vdr_fn",
    "resolve",
    "resolve_recursive",
    "is_functional",
    "make_constant_fn",
    "make_series_fn",
    "make_newton_fn",
    "make_iterative_fn",
    "discrete_derivative",
    "discrete_derivative_nth",
    "discrete_integral",
    "discrete_integral_trapz",
    "install",
    "uninstall",
]


# ---------------------------------------------------------------------------
# FnRemainder
# ---------------------------------------------------------------------------

class FnRemainder(Remainder):
    """
    A remainder slot holding a callable instead of concrete structure.

    The callable signature is: f(depth: int) -> VDR

    At any given depth, the function produces an exact finite VDR object.
    There is no limit process. Each depth is a complete exact answer.

        fr = FnRemainder(my_func, name="sqrt2")
        result = fr.expand(5)   # calls my_func(5), returns VDR
    """

    __slots__ = ("func", "name", "meta")

    def __init__(self, func, name=None, meta=None):
        super(FnRemainder, self).__init__(0, [])
        if not callable(func):
            raise InvalidStructureError(
                "FnRemainder requires a callable, got %s" % type(func).__name__
            )
        self.func = func
        self.name = name or getattr(func, "__name__", "<fn>")
        self.meta = meta or {}

    # -- predicates --------------------------------------------------------

    @property
    def is_zero(self):
        return False  # functional remainder is never zero

    @property
    def is_atomic(self):
        return False

    @property
    def is_functional(self):
        return True

    @property
    def is_globally_zero(self):
        return False

    # -- expansion ---------------------------------------------------------

    def expand(self, depth):
        """
        Expand the function at the given depth.
        Returns an exact finite VDR object.
        """
        if depth < 0:
            raise VDRError("Expansion depth must be non-negative")
        result = self.func(depth)
        if not isinstance(result, VDR):
            raise VDRError(
                "FnRemainder '%s' returned %s, expected VDR" % (
                    self.name, type(result).__name__
                )
            )
        return result

    # -- projection --------------------------------------------------------

    def legacy_value(self):
        """
        Functional remainder has no default scalar projection.
        Expand first, then project.
        """
        raise VDRError(
            "Cannot project functional remainder '%s' without expansion. "
            "Call resolve(obj, depth) first." % self.name
        )

    # -- operations --------------------------------------------------------

    def negate(self):
        fn = self.func
        def negated(depth):
            return -fn(depth)
        return FnRemainder(negated, name="-%s" % self.name, meta=self.meta)

    def lift(self, k):
        fn = self.func
        nm = self.name
        def lifted(depth):
            result = fn(depth)
            return result._lift_vdr(k)
        return FnRemainder(
            lifted, name="lift(%s,%d)" % (nm, k), meta=self.meta
        )

    def structural_eq(self, other):
        if not isinstance(other, FnRemainder):
            return False
        return self.func is other.func and self.name == other.name

    def normalize(self):
        # functional remainders don't normalize — expand first
        return self

    def combine(self, other, sign=1):
        """
        Combining a functional remainder with another remainder.
        If both functional, compose them.
        If one is concrete, wrap in a hybrid.
        """
        if isinstance(other, FnRemainder):
            fn_a = self.func
            fn_b = other.func
            s = sign
            name_a = self.name
            name_b = other.name
            def combined(depth):
                a_val = fn_a(depth)
                b_val = fn_b(depth)
                if s == 1:
                    return a_val + b_val
                else:
                    return a_val - b_val
            op = "+" if sign == 1 else "-"
            return FnRemainder(
                combined,
                name="(%s %s %s)" % (name_a, op, name_b),
            )

        # other is concrete Remainder
        fn_a = self.func
        concrete = other
        s = sign
        nm = self.name
        def hybrid(depth):
            a_val = fn_a(depth)
            concrete_vdr = VDR(0, 1, concrete)
            if s == 1:
                return a_val + concrete_vdr
            else:
                return a_val - concrete_vdr
        return FnRemainder(
            hybrid,
            name="(%s + concrete)" % nm,
        )

    # -- display -----------------------------------------------------------

    def __repr__(self):
        return "fn:%s" % self.name

    def __eq__(self, other):
        if isinstance(other, FnRemainder):
            return self.structural_eq(other)
        if isinstance(other, int) and other == 0:
            return False  # functional remainder is never zero
        if isinstance(other, Remainder):
            return False
        return NotImplemented

    def __hash__(self):
        return hash(("fn", self.name, id(self.func)))


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def vdr_fn(name=None):
    """
    Decorator for creating named VDR remainder functions.

        @vdr_fn("sqrt2")
        def sqrt2(depth):
            x = VDR(1)
            for _ in range(depth):
                x = (x + VDR(2)/x) / VDR(2)
            return x
    """
    def decorator(func):
        func._vdr_fn_name = name or func.__name__
        return func
    return decorator


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------

def resolve(x, depth=1):
    """
    Resolve a VDR object by expanding any functional remainder.

    If the remainder is functional, expand it at the given depth
    and construct a concrete VDR result.

    If the remainder is already concrete, return unchanged.

        resolved = resolve(obj, depth=10)
    """
    if not is_functional(x):
        return x

    fn_r = x.r
    expanded = fn_r.expand(depth)

    if x.v == 0 and x.d == 1:
        # trivial frame: the function IS the value
        return expanded

    # non-trivial frame: expanded value goes into remainder
    if expanded.is_closed and expanded.v == 0:
        return VDR(x.v, x.d).normalize()

    new_r = Remainder(0, [expanded])
    return VDR(x.v, x.d, new_r).normalize()


def is_functional(x):
    """Check if a VDR object has a functional remainder."""
    return isinstance(x.r, FnRemainder)


def resolve_recursive(x, depth=1):
    """
    Resolve all functional remainders in a VDR tree, recursively.
    """
    if is_functional(x):
        x = resolve(x, depth)

    if x.r.is_zero or x.r.is_atomic:
        return x

    new_children = []
    for c in x.r.children:
        new_children.append(resolve_recursive(c, depth))

    return VDR(x.v, x.d, Remainder(x.r.base, new_children)).normalize()


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_constant_fn(name, value_func):
    """
    Functional remainder that always returns the same value.

        pi_approx = make_constant_fn("pi_22_7", lambda: VDR(22, 7))
    """
    def const(depth):
        return value_func()
    return FnRemainder(const, name=name)


def make_series_fn(name, term_func, initial=None):
    """
    Functional remainder from a series.

    term_func(n) returns the nth exact rational term.
    At depth N, result is sum of terms 0..N plus initial.

        # Leibniz series for pi/4
        def leibniz_term(n):
            sign = 1 if n % 2 == 0 else -1
            return VDR(sign, 2*n + 1)

        pi_fn = make_series_fn("leibniz_pi4", leibniz_term)
    """
    init = initial if initial is not None else VDR(0)
    start = init

    def series(depth):
        total = start
        for n in range(depth + 1):
            total = total + term_func(n)
        return total

    return FnRemainder(series, name=name)


def make_newton_fn(name, step_fn, start=None):
    """
    Functional remainder from Newton-Raphson iteration.

    step_fn(x) takes current VDR and returns next iterate.
    Each step is exact rational arithmetic.
    Quadratic convergence: digits double per step.

        # sqrt(2): x_{n+1} = (x + 2/x) / 2
        sqrt2_fn = make_newton_fn("sqrt2", lambda x: (x + VDR(2)/x) / VDR(2))

    At depth N, applies step_fn N times starting from start (default VDR(1)).
    """
    initial = start if start is not None else VDR(1)
    init = initial

    def newton(depth):
        x = init
        for _ in range(depth):
            x = step_fn(x)
        return x

    return FnRemainder(newton, name=name)


def make_iterative_fn(name, step, start):
    """
    General iterative function: apply step N times to start.

        fn = make_iterative_fn("collatz", collatz_step, VDR(27))
    """
    init = start

    def iterate(depth):
        x = init
        for _ in range(depth):
            x = step(x)
        return x

    return FnRemainder(iterate, name=name, meta={"start": str(start)})


# ---------------------------------------------------------------------------
# Discrete calculus operators
# ---------------------------------------------------------------------------

def discrete_derivative(f, h):
    """
    Discrete derivative operator.

    Given f: VDR -> VDR and step size h (VDR rational),
    returns Df where:
        Df(x) = (f(x + h) - f(x)) / h

    Every evaluation is exact VDR arithmetic. No limits.

        f = lambda x: x * x
        df = discrete_derivative(f, VDR(1, 1000))
        print(df(VDR(3)))  # exact 6001/1000
    """
    step = h

    def deriv(x):
        return (f(x + step) - f(x)) / step

    return deriv


def discrete_derivative_nth(f, h, order=1):
    """
    Nth-order discrete derivative by repeated application.
    D^n f = D(D^(n-1) f)
    """
    result = f
    for _ in range(order):
        result = discrete_derivative(result, h)
    return result


def discrete_integral(f, a, b, n):
    """
    Discrete integral (left Riemann sum).

    Computes the exact sum:
        sum f(a + k*h) * h   for k = 0, 1, ..., n-1

    where h = (b - a) / n.

    Every term is exact VDR arithmetic. No limits.
    Each n gives an exact answer.

        result = discrete_integral(lambda x: x*x, VDR(0), VDR(1), 100)
    """
    if n <= 0:
        raise VDRError("Number of steps must be positive")
    h = (b - a) / VDR(n)
    total = VDR(0)
    for k in range(n):
        x_k = a + VDR(k) * h
        total = total + f(x_k) * h
    return total


def discrete_integral_trapz(f, a, b, n):
    """
    Discrete integral (trapezoidal rule).

    Computes:
        h/2 * (f(a) + 2*f(a+h) + ... + 2*f(a+(n-1)*h) + f(b))

    More accurate than left Riemann for the same n.
    Still exact VDR arithmetic at every step.
    """
    if n <= 0:
        raise VDRError("Number of steps must be positive")
    h = (b - a) / VDR(n)
    total = f(a) + f(b)
    for k in range(1, n):
        x_k = a + VDR(k) * h
        total = total + VDR(2) * f(x_k)
    return total * h / VDR(2)


# ---------------------------------------------------------------------------
# Installation: make VDR aware of FnRemainder
# ---------------------------------------------------------------------------

_original_to_fraction = None
_original_is_closed = None
_original_is_active = None
_installed = False


def _patched_to_fraction(self):
    if isinstance(self.r, FnRemainder):
        raise VDRError(
            "Cannot project VDR with functional remainder '%s'. "
            "Call resolve(obj, depth) first." % self.r.name
        )
    return _original_to_fraction(self)


def _patched_is_closed(self):
    if isinstance(self.r, FnRemainder):
        return False
    return _original_is_closed(self)


def _patched_is_active(self):
    if isinstance(self.r, FnRemainder):
        return True
    return _original_is_active(self)


def install():
    """
    Patch VDR to be aware of functional remainders.

    After this, to_fraction() raises on functional remainders
    (forcing explicit resolve), and is_closed/is_active work correctly.

    Called automatically by vdr.__init__.
    """
    global _original_to_fraction, _original_is_closed, _original_is_active
    global _installed

    if _installed:
        return

    _original_to_fraction = VDR.to_fraction
    _original_is_closed = VDR.is_closed.fget
    _original_is_active = VDR.is_active.fget

    VDR.to_fraction = _patched_to_fraction
    VDR.is_closed = property(_patched_is_closed)
    VDR.is_active = property(_patched_is_active)

    _installed = True


def uninstall():
    """Restore original VDR behavior."""
    global _installed

    if not _installed:
        return

    VDR.to_fraction = _original_to_fraction
    VDR.is_closed = property(_original_is_closed)
    VDR.is_active = property(_original_is_active)

    _installed = False
