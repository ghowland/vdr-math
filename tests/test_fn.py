"""
Tests for vdr.fn — FnRemainder, resolve, factories, discrete calculus.
"""

import pytest
from fractions import Fraction

from vdr.core import VDR, Remainder, VDRError
from vdr.fn import (
    FnRemainder, resolve, resolve_recursive, is_functional,
    make_newton_fn, make_series_fn, make_iterative_fn, make_constant_fn,
    discrete_derivative, discrete_derivative_nth,
    discrete_integral, discrete_integral_trapz,
    install,
)
from vdr.active import install as install_active

install_active()
install()


# ---------------------------------------------------------------------------
# FnRemainder basics
# ---------------------------------------------------------------------------

class TestFnRemainder:
    def test_construction(self):
        fn = FnRemainder(lambda d: VDR(d), name="test")
        assert fn.name == "test"
        assert fn.is_functional
        assert not fn.is_zero
        assert not fn.is_atomic

    def test_expand(self):
        fn = FnRemainder(lambda d: VDR(d * d), name="square")
        assert fn.expand(5) == VDR(25)

    def test_expand_negative_depth_raises(self):
        fn = FnRemainder(lambda d: VDR(d), name="test")
        with pytest.raises(VDRError):
            fn.expand(-1)

    def test_expand_must_return_vdr(self):
        fn = FnRemainder(lambda d: 42, name="bad")
        with pytest.raises(VDRError):
            fn.expand(1)

    def test_not_callable_raises(self):
        with pytest.raises(Exception):
            FnRemainder(42)

    def test_negate(self):
        fn = FnRemainder(lambda d: VDR(d + 1), name="inc")
        neg = fn.negate()
        assert neg.expand(5) == VDR(-6)

    def test_lift(self):
        fn = FnRemainder(lambda d: VDR(1, 2), name="half")
        lifted = fn.lift(3)
        result = lifted.expand(0)
        assert result.v == 3
        assert result.d == 2

    def test_combine_two_fn(self):
        fn1 = FnRemainder(lambda d: VDR(d), name="id")
        fn2 = FnRemainder(lambda d: VDR(d * 2), name="double")
        combined = fn1.combine(fn2, sign=1)
        assert isinstance(combined, FnRemainder)
        # at depth 5: 5 + 10 = 15
        assert combined.expand(5) == VDR(15)

    def test_combine_sub(self):
        fn1 = FnRemainder(lambda d: VDR(10), name="ten")
        fn2 = FnRemainder(lambda d: VDR(3), name="three")
        combined = fn1.combine(fn2, sign=-1)
        assert combined.expand(0) == VDR(7)

    def test_structural_eq_same(self):
        f = lambda d: VDR(d)
        fn1 = FnRemainder(f, name="f")
        fn2 = FnRemainder(f, name="f")
        assert fn1.structural_eq(fn2)

    def test_structural_eq_different(self):
        fn1 = FnRemainder(lambda d: VDR(d), name="a")
        fn2 = FnRemainder(lambda d: VDR(d), name="b")
        assert not fn1.structural_eq(fn2)

    def test_legacy_value_raises(self):
        fn = FnRemainder(lambda d: VDR(1), name="test")
        with pytest.raises(VDRError):
            fn.legacy_value()

    def test_repr(self):
        fn = FnRemainder(lambda d: VDR(1), name="myfn")
        assert "myfn" in repr(fn)


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------

class TestResolve:
    def test_resolve_trivial_frame(self):
        fn = FnRemainder(lambda d: VDR(42), name="const")
        obj = VDR(0, 1, fn)
        result = resolve(obj, depth=0)
        assert result == VDR(42)

    def test_resolve_nontrivial_frame(self):
        fn = FnRemainder(lambda d: VDR(1), name="one")
        obj = VDR(3, 7, fn)
        result = resolve(obj, depth=0)
        # value = (3 + 1) / 7 = 4/7... but structurally
        # result is VDR(3, 7, Remainder(0, [VDR(1)]))
        assert result.to_fraction() == Fraction(3, 7) + Fraction(1, 7)

    def test_resolve_nonfunctional_passthrough(self):
        obj = VDR(1, 2)
        assert resolve(obj) is obj

    def test_is_functional(self):
        fn = FnRemainder(lambda d: VDR(1), name="test")
        assert is_functional(VDR(0, 1, fn))
        assert not is_functional(VDR(1, 2))


class TestResolveRecursive:
    def test_nested_functional(self):
        inner_fn = FnRemainder(lambda d: VDR(7), name="inner")
        inner_vdr = VDR(0, 1, inner_fn)
        outer = VDR(1, 1, Remainder(0, [inner_vdr]))
        # can't resolve outer directly (inner child is functional)
        # resolve_recursive handles it
        result = resolve_recursive(outer, depth=1)
        # inner resolves to 7, so outer = 1 + 7 = 8
        assert result.to_fraction() == Fraction(8)


# ---------------------------------------------------------------------------
# Newton factory
# ---------------------------------------------------------------------------

class TestNewtonFn:
    def test_sqrt2(self):
        fn = make_newton_fn("sqrt2", lambda x: (x + VDR(2) / x) / VDR(2))
        result = fn.expand(10)
        # verify: result^2 should be very close to 2
        residual = result * result - VDR(2)
        frac = residual.to_fraction()
        assert abs(frac) < Fraction(1, 10 ** 50)

    def test_sqrt2_depth_0(self):
        fn = make_newton_fn("sqrt2", lambda x: (x + VDR(2) / x) / VDR(2))
        result = fn.expand(0)
        # depth 0 = starting value VDR(1)
        assert result == VDR(1)

    def test_sqrt3(self):
        fn = make_newton_fn("sqrt3", lambda x: (x + VDR(3) / x) / VDR(2))
        result = fn.expand(10)
        residual = result * result - VDR(3)
        frac = residual.to_fraction()
        assert abs(frac) < Fraction(1, 10 ** 50)

    def test_custom_start(self):
        fn = make_newton_fn(
            "sqrt2_start2",
            lambda x: (x + VDR(2) / x) / VDR(2),
            start=VDR(2),
        )
        result = fn.expand(8)
        residual = result * result - VDR(2)
        assert abs(residual.to_fraction()) < Fraction(1, 10 ** 40)

    def test_quadratic_convergence(self):
        """Digits should roughly double per step."""
        fn = make_newton_fn("sqrt2", lambda x: (x + VDR(2) / x) / VDR(2))
        prev_digits = 0
        for depth in range(1, 8):
            result = fn.expand(depth)
            residual = abs((result * result - VDR(2)).to_fraction())
            if residual > 0:
                digits = -float(residual.numerator) / float(residual.denominator)
                # just verify convergence improves
                import math
                current_digits = -math.log10(float(residual)) if float(residual) > 0 else 999
                assert current_digits > prev_digits
                prev_digits = current_digits


# ---------------------------------------------------------------------------
# Series factory
# ---------------------------------------------------------------------------

class TestSeriesFn:
    def test_leibniz_pi4(self):
        """Leibniz series: pi/4 = 1 - 1/3 + 1/5 - 1/7 + ..."""
        def term(n):
            sign = 1 if n % 2 == 0 else -1
            return VDR(sign, 2 * n + 1)

        fn = make_series_fn("leibniz", term)
        result = fn.expand(100)
        # should be close to pi/4 ~ 0.785
        val = float(result.to_fraction())
        assert abs(val - 0.7853981633974483) < 0.02

    def test_series_depth_0(self):
        fn = make_series_fn("test", lambda n: VDR(1))
        assert fn.expand(0) == VDR(1)  # just term 0


# ---------------------------------------------------------------------------
# Iterative factory
# ---------------------------------------------------------------------------

class TestIterativeFn:
    def test_doubling(self):
        fn = make_iterative_fn("double", lambda x: x * VDR(2), VDR(1))
        assert fn.expand(0) == VDR(1)
        assert fn.expand(3) == VDR(8)
        assert fn.expand(10) == VDR(1024)

    def test_contraction(self):
        fn = make_iterative_fn(
            "contract",
            lambda x: x / VDR(2) + VDR(1),
            VDR(100),
        )
        result = fn.expand(20)
        # converges to 2
        assert abs(float(result.to_fraction()) - 2.0) < 0.001


# ---------------------------------------------------------------------------
# Constant factory
# ---------------------------------------------------------------------------

class TestConstantFn:
    def test_constant(self):
        fn = make_constant_fn("pi_approx", lambda: VDR(22, 7))
        assert fn.expand(0) == VDR(22, 7)
        assert fn.expand(100) == VDR(22, 7)


# ---------------------------------------------------------------------------
# Discrete calculus
# ---------------------------------------------------------------------------

class TestDiscreteDerivative:
    def test_x_squared(self):
        """D_h(x^2) at x=3, h=1/1000 should be ~6.001."""
        f = lambda x: x * x
        df = discrete_derivative(f, VDR(1, 1000))
        result = df(VDR(3))
        assert result == VDR(6001, 1000)

    def test_constant_function(self):
        """Derivative of constant = 0."""
        f = lambda x: VDR(5)
        df = discrete_derivative(f, VDR(1, 100))
        result = df(VDR(7))
        assert result == VDR(0)

    def test_linear_function(self):
        """Derivative of 3x = 3 (exact for any h)."""
        f = lambda x: VDR(3) * x
        df = discrete_derivative(f, VDR(1, 7))
        result = df(VDR(10))
        assert result == VDR(3)


class TestDiscreteDerivativeNth:
    def test_second_derivative_x_squared(self):
        """D^2(x^2) = 2 (for small enough h, close to 2)."""
        f = lambda x: x * x
        ddf = discrete_derivative_nth(f, VDR(1, 100), order=2)
        result = ddf(VDR(5))
        assert abs(float(result.to_fraction()) - 2.0) < 0.03


class TestDiscreteIntegral:
    def test_x_squared_0_to_1(self):
        """Left Riemann integral of x^2 from 0 to 1 with n=10."""
        f = lambda x: x * x
        result = discrete_integral(f, VDR(0), VDR(1), 10)
        assert result == VDR(57, 200)

    def test_constant_function(self):
        """Integral of 1 from 0 to 3 = 3."""
        f = lambda x: VDR(1)
        result = discrete_integral(f, VDR(0), VDR(3), 100)
        assert result == VDR(3)

    def test_zero_steps_raises(self):
        with pytest.raises(VDRError):
            discrete_integral(lambda x: x, VDR(0), VDR(1), 0)


class TestTrapezoidalIntegral:
    def test_x_squared_0_to_1(self):
        """Trapezoidal should be more accurate than left Riemann."""
        f = lambda x: x * x
        trap = discrete_integral_trapz(f, VDR(0), VDR(1), 100)
        left = discrete_integral(f, VDR(0), VDR(1), 100)
        exact = Fraction(1, 3)
        trap_err = abs(trap.to_fraction() - exact)
        left_err = abs(left.to_fraction() - exact)
        assert trap_err < left_err

    def test_linear_exact(self):
        """Trapezoidal is exact for linear functions."""
        f = lambda x: VDR(3) * x + VDR(1)
        result = discrete_integral_trapz(f, VDR(0), VDR(2), 5)
        # integral of 3x+1 from 0 to 2 = [3x^2/2 + x]_0^2 = 6+2 = 8
        assert result == VDR(8)


# ---------------------------------------------------------------------------
# Finite difference tables
# ---------------------------------------------------------------------------

class TestFiniteDifferences:
    def test_delta3_x3(self):
        """Delta^3(x^3) at integer points should be constant 6."""
        f = lambda x: x * x * x
        h = VDR(1)
        d3f = discrete_derivative_nth(f, h, order=3)
        vals = [d3f(VDR(k)) for k in range(3)]
        assert vals[0] == VDR(6)
        assert vals[1] == VDR(6)

    def test_delta4_x3(self):
        """Delta^4(x^3) should be 0."""
        f = lambda x: x * x * x
        h = VDR(1)
        d4f = discrete_derivative_nth(f, h, order=4)
        assert d4f(VDR(0)) == VDR(0)
