# Part 1: Migration Guide

## Porting Existing Code to VDR

This guide walks through converting code that uses `float`, `Fraction`, `mpmath`, `numpy`, or `SymPy` to use VDR exact arithmetic. Each section shows the old pattern and the VDR equivalent.

---

## From float

### The Problem with float

```python
# Float: silent information loss
x = 1/3                    # 0.3333333333333333 (16 digits, then wrong)
y = 1/7                    # 0.14285714285714285
z = x + y                  # 0.47619047619047616 (last digit already wrong)

# Accumulation over chains
x = 1/7
for _ in range(100):
    x += 1/13
for _ in range(100):
    x -= 1/13
print(x == 1/7)            # False — 2.78e-16 error
```

### The VDR Equivalent

```python
from vdr import VDR

x = VDR(1, 3)              # exact 1/3 — three integers, zero truncation
y = VDR(1, 7)              # exact 1/7
z = x + y                  # exact 10/21

# Accumulation: zero error regardless of chain length
x = VDR(1, 7)
for _ in range(100):
    x = x + VDR(1, 13)
for _ in range(100):
    x = x - VDR(1, 13)
assert x == VDR(1, 7)      # True — exactly zero error
```

### Migration Pattern

| float | VDR |
|---|---|
| `x = 0.5` | `x = VDR(1, 2)` |
| `x = 1/3` | `x = VDR(1, 3)` |
| `x = 3.14159` | `x = VDR(314159, 100000)` or use `PI` from transcendental |
| `x + y` | `x + y` (same syntax) |
| `x * y` | `x * y` (same syntax) |
| `abs(x - y) < 1e-10` | `x == y` (exact equality) |
| `math.sqrt(2)` | `sqrt_newton(VDR(2), depth=10)` |
| `math.sin(x)` | `sin_series(x, depth=16)` |
| `float(result)` | `to_float(result)` (explicit lossy boundary) |

### Converting Float Constants

```python
# BAD: float constant loses precision on entry
x = VDR(int(0.1 * 1000), 1000)   # 0.1 as float is already wrong

# GOOD: construct from exact integers
x = VDR(1, 10)                    # exactly 0.1

# For measured values with known precision:
# "3.14159 ± 0.00001" means you know 5 decimal digits
x = VDR(314159, 100000)           # exact rational representation

# For transcendental constants: use the library
from vdr.math.transcendental import PI, E, SQRT2
```

### Converting Float Comparisons

```python
# BEFORE (float): tolerance-based
def is_symmetric_float(M):
    for i in range(n):
        for j in range(n):
            if abs(M[i][j] - M[j][i]) > 1e-12:
                return False
    return True

# AFTER (VDR): exact equality
def is_symmetric_vdr(M):
    return M == M.T
```

### Converting Float Convergence Checks

```python
# BEFORE: "close enough" loop
while abs(x_new - x_old) > 1e-15:
    x_old = x_new
    x_new = step(x_old)

# AFTER: run for fixed depth, inspect residual exactly
from vdr.fn import make_newton_fn
fn = make_newton_fn("my_iter", step)
result = fn.expand(depth=10)
residual = check_function(result)
print("Exact residual:", residual.to_fraction())
```

---

## From fractions.Fraction

### Why Migrate

`Fraction` gives exact rational arithmetic. For closed operations (add, subtract, multiply, divide of rationals), VDR's closed subclass gives identical results. The reasons to migrate:

1. **Denominator explosion.** Fraction lets D grow without bound. After 1000 multiplications, denominators can have thousands of digits. VDR with a fixed frame keeps D stable via divmod.

2. **No remainder slot.** When you need structure beyond a single rational — nested values, functional remainders for sqrt/trig/exp — Fraction has no mechanism. VDR's R slot handles this natively.

3. **No framework.** Fraction gives you arithmetic. VDR gives you arithmetic plus linear algebra, discrete calculus, 38 math domains, and a complete ML/physics/signal processing stack.

### Migration Pattern

```python
# BEFORE
from fractions import Fraction
x = Fraction(1, 3)
y = Fraction(2, 7)
z = x + y               # Fraction(13, 21)

# AFTER
from vdr import VDR
x = VDR(1, 3)
y = VDR(2, 7)
z = x + y               # VDR(13, 21) — same value

# Converting between them
from fractions import Fraction
frac = z.to_fraction()                    # VDR -> Fraction
vdr = VDR.from_fraction(Fraction(5, 6))   # Fraction -> VDR
```

### Denominator Growth Comparison

```python
from fractions import Fraction
from vdr import VDR
from vdr.basis import to_qbasis, qb_mul

# Fraction: denominator grows with every multiply
f = Fraction(1, 7)
for _ in range(20):
    f = f * Fraction(3, 11)
print("Fraction denom after 20 muls:", len(str(f.denominator)), "digits")

# VDR with Q335 frame: D stays fixed
a = to_qbasis(VDR(1, 7), bits=335)
b = to_qbasis(VDR(3, 11), bits=335)
result = a
for _ in range(20):
    result = qb_mul(result, b, bits=335)
print("VDR D after 20 muls: 2^335 (fixed)")
print("Overflow captured in R, zero loss")
```

---

## From mpmath

### Why Migrate

mpmath provides arbitrary-precision decimal arithmetic. It's excellent for what it does, but it still truncates — at 50 digits, or 1000 digits, it's still truncation. Each operation discards information.

VDR is categorically different. `[1, 3, 0]` is exact. No digits to configure. No truncation at any precision.

### Migration Pattern

```python
# BEFORE
import mpmath
mpmath.mp.dps = 50
x = mpmath.mpf('1') / mpmath.mpf('3')
y = mpmath.mpf('1') / mpmath.mpf('7')
z = x + y
print(z)   # 0.47619... (50 digits, then truncated)

# AFTER
from vdr import VDR
x = VDR(1, 3)
y = VDR(1, 7)
z = x + y   # [10, 21, 0] — exact

# For decimal output at the end:
from vdr.export import to_decimal
print(to_decimal(z, digits=50))  # 50-digit decimal, loss is in the rendering
```

### Migrating mpmath Special Functions

```python
# BEFORE
import mpmath
mpmath.mp.dps = 100
pi_val = mpmath.pi
sqrt2_val = mpmath.sqrt(2)
zeta3_val = mpmath.zeta(3)

# AFTER
from vdr.math.transcendental import PI, SQRT2, ZETA3, sqrt_newton, borwein_zeta

# Pre-computed at 100-digit precision:
PI          # ready to use
SQRT2       # ready to use
ZETA3       # ready to use

# Or compute at arbitrary precision:
sqrt2_val = sqrt_newton(VDR(2), depth=10)        # >100 digits
zeta5_val = borwein_zeta(5, n=210)               # 100+ digits

# Key difference: mpmath gives you 100 truncated digits.
# VDR gives you an exact rational. The residual is inspectable.
residual = sqrt2_val * sqrt2_val - VDR(2)
print("Exact residual:", residual.to_fraction())
# mpmath cannot tell you this. Its error is hidden in the truncation.
```

---

## From numpy

### Why Migrate

numpy operates on float64 arrays. Fast, vectorized, but every element has ~15.7 digits of precision with silent rounding at every operation.

VDR is not a numpy replacement for large-scale computation. VDR is for cases where exactness matters more than throughput: verification, small-to-medium exact computations, ground truth generation.

### Migration Pattern

```python
# BEFORE
import numpy as np
A = np.array([[1, 2], [3, 4]], dtype=float)
b = np.array([5, 11], dtype=float)
x = np.linalg.solve(A, b)
print(np.allclose(A @ x, b))   # True (within tolerance)

# AFTER
from vdr.linalg import Mat, Vec
A = Mat.from_ints([[1, 2], [3, 4]])
b = Vec.from_ints([5, 11])
x = A.solve(b)
assert A.matvec(x) == b        # True (exact, not "close")
```

### When to Keep numpy

Keep numpy for:
- Large matrices (n > 50)
- Real-time processing
- GPU computation
- Approximate is good enough

Use VDR for:
- Verification: compute exact answer on small instance, verify numpy on same instance
- Ill-conditioned systems where float fails
- Chains where drift accumulates
- Reproducibility across platforms

### Hybrid Pattern

```python
import numpy as np
from vdr.linalg import Mat, Vec

# Compute exact ground truth with VDR
A_vdr = Mat.from_ints([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
b_vdr = Vec.from_ints([6, 15, 25])
x_exact = A_vdr.solve(b_vdr)

# Compute with numpy
A_np = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)
b_np = np.array([6, 15, 25], dtype=float)
x_np = np.linalg.solve(A_np, b_np)

# Compare: how much error does numpy introduce?
for i in range(3):
    exact_val = float(x_exact[i].to_fraction())
    numpy_val = x_np[i]
    print("x[%d]: exact=%s, numpy=%s, error=%e" % (
        i, exact_val, numpy_val, abs(exact_val - numpy_val)))
```

---

## From SymPy

### Why Migrate

SymPy is a computer algebra system with symbolic expressions, rewrite rules, and pattern matching. It's far more general than VDR for symbolic manipulation.

VDR is not a CAS. It doesn't simplify expressions, solve equations symbolically, or manipulate algebraic forms. It does exact rational arithmetic with a fixed three-slot structure.

### When VDR Wins Over SymPy

- **Speed on pure rational computation.** VDR is simpler and faster for chains of rational operations.
- **Structural identity.** VDR objects have a canonical normalized form. SymPy expressions can be equivalent but structurally different.
- **No expression swell.** VDR denominators stay bounded (with Q335 frame). SymPy expressions can grow without bound.

### Migration Pattern

```python
# BEFORE (SymPy)
from sympy import Rational, Matrix, sqrt
x = Rational(1, 3)
y = Rational(2, 7)
z = x + y
M = Matrix([[1, 2], [3, 4]])
det = M.det()

# AFTER (VDR)
from vdr import VDR
from vdr.linalg import Mat
x = VDR(1, 3)
y = VDR(2, 7)
z = x + y
M = Mat.from_ints([[1, 2], [3, 4]])
det = M.det()

# SymPy sqrt: symbolic
# VDR sqrt: exact rational at chosen depth
from sympy import sqrt as sympy_sqrt
s = sympy_sqrt(2)   # symbolic object, not a number

from vdr.math.transcendental import sqrt_newton
s = sqrt_newton(VDR(2), depth=10)   # exact rational, >100 digits
# You can inspect the residual:
print((s * s - VDR(2)).to_fraction())
```

---

## General Migration Checklist

1. **Replace float literals with VDR construction from integers.**
   `0.5` → `VDR(1, 2)`. `1/3` → `VDR(1, 3)`. Never construct VDR from float.

2. **Replace tolerance checks with exact equality.**
   `abs(a - b) < 1e-10` → `a == b`.

3. **Replace math.sqrt/sin/cos with VDR functional equivalents.**
   `math.sqrt(2)` → `sqrt_newton(VDR(2), depth=10)`.

4. **Replace numpy linalg with VDR linalg.**
   `np.linalg.solve` → `Mat.solve`. `np.linalg.det` → `Mat.det`.

5. **Keep float/numpy for final output if needed.**
   Use `to_float()` or `to_decimal()` at the very end.

6. **Don't mix VDR and float in the same computation chain.**
   Float contaminates exactness. Convert at boundaries only.

7. **For transcendental constants, use the library.**
   Don't convert `3.14159` to VDR. Use `PI` from `vdr.math.transcendental`.

8. **For long chains, consider Q335 frame.**
   Import `qb_mul` from `vdr.basis` to keep denominators bounded.

---

# Part 2: Domain Extension Guide

## Building New Domain Modules for VDR

This guide shows how to create a new mathematical domain as a library module, following the patterns established by the 38 existing domains.

---

## Architecture

Every domain module follows the same structure:

```
src/vdr/math/my_domain.py     # or src/vdr/physics/, src/vdr/signal/, etc.
tests/gym/test_gym_my.py      # thin test layer calling library functions
```

The module contains functions and constants. Tests call the functions and assert exact results. No tolerance. No approximation hedging.

---

## Step 1: Module Skeleton

```python
"""
vdr.math.my_domain — Exact my-domain computations.

    from vdr.math.my_domain import my_function, MY_CONSTANT

    result = my_function(VDR(1, 3))   # exact

All operations exact VDR rational. Zero drift. Zero truncation.
"""

from __future__ import annotations
from typing import List, Tuple

from vdr.core import VDR
from vdr.linalg import Vec, Mat

__all__ = [
    "my_function",
    "MY_CONSTANT",
]
```

The docstring shows the import and a usage example. `__all__` lists the public API.

---

## Step 2: Choose Your Inputs and Outputs

Every function takes VDR objects (or ints, which coerce to VDR) and returns VDR objects. The signature pattern:

```python
def my_function(x, n=10):
    """
    Compute something exact.

    I: x as VDR, optional parameter n
    O: result as VDR, exact

        my_function(VDR(1, 3))  -> VDR(...)
    """
```

Document Input, Output, and show an example in the docstring.

---

## Step 3: Use Only VDR Operations

Inside your function, use only VDR arithmetic. This guarantees exactness.

```python
def triangle_area(ax, ay, bx, by, cx, cy):
    """Exact triangle area via Shoelace."""
    return abs(
        ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)
    ) / VDR(2)
```

If you need a loop, each iteration uses VDR:

```python
def power_sum(k, n):
    """1^k + 2^k + ... + n^k, exact."""
    total = VDR(0)
    for i in range(1, n + 1):
        total = total + VDR(i ** k)
    return total
```

If you need matrix operations:

```python
def my_matrix_check(A):
    """Verify A^2 = I."""
    return A.matmul(A) == Mat.identity(A.nrows)
```

---

## Step 4: Handle Irrational-Like Quantities

When your domain needs sqrt, trig, exp, log — use the transcendental module.

```python
from vdr.math.transcendental import sqrt_newton, sin_series, cos_series, exp_series

def my_rotation(angle_over_pi, depth=16):
    """Rotation matrix for rational multiple of pi."""
    from vdr.math.transcendental import PI
    theta = angle_over_pi * PI
    c = cos_series(theta, depth)
    s = sin_series(theta, depth)
    return Mat([[c, -s], [s, c]])
```

The `depth` parameter is the precision knob. Document what depth gives what precision. For Newton iteration, each depth roughly doubles correct digits. For Taylor series, it depends on the argument.

---

## Step 5: Module-Level Constants

If your domain has important constants, define them at module level:

```python
# Named constants available on import
GOLDEN_RATIO = VDR(1618034, 1000000)   # if rational approximation suffices

# Or use Q335 for full precision
from vdr.math.transcendental import PHI as GOLDEN_RATIO

# Or define as functional remainder for arbitrary depth
from vdr.fn import make_newton_fn
GOLDEN_RATIO_FN = make_newton_fn(
    "phi",
    lambda x: VDR(1) + VDR(1) / x,
    start=VDR(2),
)
```

---

## Step 6: Write Tests

Tests go in `tests/gym/test_gym_XX.py`. They call library functions and assert exact results.

```python
"""Gym XX — My Domain. N/N tests."""

import pytest
from vdr import VDR
from vdr.linalg import Vec, Mat
from vdr.math.my_domain import my_function, MY_CONSTANT


class TestMyFunction:
    def test_basic(self):
        result = my_function(VDR(1, 3))
        assert result == VDR(...)  # exact expected value

    def test_identity(self):
        """Some identity that should hold exactly."""
        a = my_function(VDR(1, 2))
        b = my_function(VDR(1, 3))
        assert a + b == my_function(VDR(5, 6))

    def test_zero_case(self):
        assert my_function(VDR(0)) == VDR(0)

    def test_conservation(self):
        """Something that should be exactly conserved."""
        result = my_conservation_check(...)
        assert result == VDR(1)   # not abs(result - 1) < 1e-10
```

### Test Patterns

**Pattern 1: Exact value.**

```python
def test_known_value(self):
    assert harmonic(10) == VDR(7381, 2520)
```

**Pattern 2: Identity verification.**

```python
def test_identity(self):
    # Cassini: F(n-1)*F(n+1) - F(n)^2 = (-1)^n
    for n in range(2, 20):
        assert fibonacci(n-1) * fibonacci(n+1) - fibonacci(n)**2 == VDR((-1)**n)
```

**Pattern 3: Conservation law.**

```python
def test_sums_to_one(self):
    probs = binom_pmf_full(10, VDR(1, 3))
    total = VDR(0)
    for p in probs:
        total = total + p
    assert total == VDR(1)
```

**Pattern 4: Roundtrip.**

```python
def test_roundtrip(self):
    original = VDR(355, 113)
    cf = to_cf(355, 113)
    recovered = from_cf(cf)
    assert recovered == original
```

**Pattern 5: Residual inspection.**

```python
def test_sqrt_residual(self):
    s = sqrt_newton(VDR(2), depth=10)
    residual = s * s - VDR(2)
    assert abs(residual.to_fraction()) < Fraction(1, 10**50)
```

**Pattern 6: Matrix equation.**

```python
def test_inverse(self):
    M = Mat.from_ints([[1, 2], [3, 4]])
    assert M.matmul(M.inv()) == Mat.identity(2)
```

**Pattern 7: Exact zero.**

```python
def test_cayley_hamilton(self):
    result = cayley_hamilton_verify(A)
    assert result == Mat.zero(n, n)   # every entry [0, 1, 0]
```

---

## Step 7: Register in Package

Add your module to the appropriate `__init__.py`:

```python
# In src/vdr/math/__init__.py docstring, add to the domain list:
#     my_domain
```

No import is needed in `__init__.py` — users import directly:

```python
from vdr.math.my_domain import my_function
```

---

## Complete Example: Rational Bezier Curves

Here's a full domain module from scratch.

### Module: `src/vdr/math/bezier.py`

```python
"""
vdr.math.bezier — Exact rational Bezier curve computation.

    from vdr.math.bezier import bezier_point, bezier_derivative, de_casteljau

    # Cubic Bezier through exact rational control points
    pts = [(VDR(0), VDR(0)), (VDR(1,3), VDR(1)),
           (VDR(2,3), VDR(1)), (VDR(1), VDR(0))]
    x, y = bezier_point(pts, VDR(1, 2))

Every evaluation exact VDR rational. No Runge phenomenon from arithmetic.
Subdivision is exact. De Casteljau is exact.
"""

from __future__ import annotations
from typing import List, Tuple

from vdr.core import VDR
from vdr.math.combinatorics import binom

__all__ = [
    "bezier_point",
    "bezier_derivative",
    "de_casteljau",
    "bezier_split",
    "bezier_length_squared_segments",
]

Point = Tuple[VDR, VDR]


def bezier_point(control_points, t):
    """
    Evaluate degree-n Bezier curve at parameter t.

    B(t) = sum_{i=0}^{n} C(n,i) * (1-t)^(n-i) * t^i * P_i

    I: list of (VDR, VDR) control points, parameter t (VDR in [0,1])
    O: (x, y) as VDR tuple, exact

        pts = [(VDR(0), VDR(0)), (VDR(1), VDR(1)), (VDR(2), VDR(0))]
        bezier_point(pts, VDR(1, 2))  # midpoint of quadratic
    """
    n = len(control_points) - 1
    one_minus_t = VDR(1) - t

    rx = VDR(0)
    ry = VDR(0)

    # precompute powers
    t_powers = [VDR(1)]
    for _ in range(n):
        t_powers.append(t_powers[-1] * t)

    omt_powers = [VDR(1)]
    for _ in range(n):
        omt_powers.append(omt_powers[-1] * one_minus_t)

    for i in range(n + 1):
        coeff = binom(n, i) * omt_powers[n - i] * t_powers[i]
        px, py = control_points[i]
        rx = rx + coeff * px
        ry = ry + coeff * py

    return (rx, ry)


def de_casteljau(control_points, t):
    """
    De Casteljau algorithm — numerically stable Bezier evaluation.

    Recursive linear interpolation. Each step is one VDR multiply and add.
    Exact at every level.

    I: control points, parameter t
    O: (x, y) point on curve
    """
    pts = list(control_points)
    one_minus_t = VDR(1) - t

    while len(pts) > 1:
        new_pts = []
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            nx = one_minus_t * x0 + t * x1
            ny = one_minus_t * y0 + t * y1
            new_pts.append((nx, ny))
        pts = new_pts

    return pts[0]


def bezier_derivative(control_points, t):
    """
    Derivative of Bezier curve at parameter t.

    B'(t) = n * sum_{i=0}^{n-1} C(n-1,i) * (1-t)^(n-1-i) * t^i * (P_{i+1} - P_i)

    I: control points, parameter t
    O: (dx, dy) tangent vector, exact
    """
    n = len(control_points) - 1
    if n < 1:
        return (VDR(0), VDR(0))

    # difference points
    diff_pts = []
    for i in range(n):
        x0, y0 = control_points[i]
        x1, y1 = control_points[i + 1]
        diff_pts.append((x1 - x0, y1 - y0))

    # evaluate degree-(n-1) Bezier on difference points
    dx, dy = bezier_point(diff_pts, t)

    return (VDR(n) * dx, VDR(n) * dy)


def bezier_split(control_points, t):
    """
    Split Bezier curve at parameter t into two curves.

    Uses de Casteljau pyramid: left curve gets the left edge,
    right curve gets the right edge.

    I: control points, parameter t
    O: (left_points, right_points) each a list of control points

    Both halves are exact. Rejoining reproduces the original exactly.
    """
    n = len(control_points) - 1
    levels = [list(control_points)]
    one_minus_t = VDR(1) - t

    for _ in range(n):
        prev = levels[-1]
        new_level = []
        for i in range(len(prev) - 1):
            x0, y0 = prev[i]
            x1, y1 = prev[i + 1]
            new_level.append((one_minus_t * x0 + t * x1,
                              one_minus_t * y0 + t * y1))
        levels.append(new_level)

    left = [levels[k][0] for k in range(n + 1)]
    right = [levels[k][-1] for k in range(n + 1)]

    return left, right


def bezier_length_squared_segments(control_points, n_segments):
    """
    Approximate curve length² by summing squared chord lengths.

    Each chord is exact. Sum is exact. This is a lower bound on
    actual arc length², getting tighter with more segments.

    I: control points, number of segments
    O: sum of squared chord lengths as VDR
    """
    total = VDR(0)
    for i in range(n_segments):
        t0 = VDR(i, n_segments)
        t1 = VDR(i + 1, n_segments)
        x0, y0 = bezier_point(control_points, t0)
        x1, y1 = bezier_point(control_points, t1)
        dx = x1 - x0
        dy = y1 - y0
        total = total + dx * dx + dy * dy
    return total
```

### Tests: `tests/gym/test_gym_bezier.py`

```python
"""Gym — Rational Bezier curves."""

import pytest
from vdr import VDR
from vdr.math.bezier import (
    bezier_point, de_casteljau, bezier_derivative, bezier_split,
)


class TestBezierPoint:
    def test_endpoints(self):
        """Curve passes through first and last control point."""
        pts = [(VDR(0), VDR(0)), (VDR(1), VDR(1)), (VDR(2), VDR(0))]
        assert bezier_point(pts, VDR(0)) == (VDR(0), VDR(0))
        assert bezier_point(pts, VDR(1)) == (VDR(2), VDR(0))

    def test_midpoint_quadratic(self):
        """Quadratic Bezier at t=1/2."""
        pts = [(VDR(0), VDR(0)), (VDR(1), VDR(2)), (VDR(2), VDR(0))]
        x, y = bezier_point(pts, VDR(1, 2))
        assert x == VDR(1)
        assert y == VDR(1)

    def test_linear(self):
        """Linear Bezier = straight line."""
        pts = [(VDR(0), VDR(0)), (VDR(6), VDR(9))]
        x, y = bezier_point(pts, VDR(1, 3))
        assert x == VDR(2)
        assert y == VDR(3)


class TestDeCasteljau:
    def test_agrees_with_direct(self):
        """De Casteljau and direct evaluation must agree."""
        pts = [(VDR(0), VDR(0)), (VDR(1, 3), VDR(1)),
               (VDR(2, 3), VDR(1)), (VDR(1), VDR(0))]
        for t_num in range(11):
            t = VDR(t_num, 10)
            direct = bezier_point(pts, t)
            casteljau = de_casteljau(pts, t)
            assert direct == casteljau


class TestDerivative:
    def test_linear_derivative(self):
        """Derivative of linear Bezier = constant direction."""
        pts = [(VDR(0), VDR(0)), (VDR(3), VDR(4))]
        dx, dy = bezier_derivative(pts, VDR(1, 2))
        assert dx == VDR(3)
        assert dy == VDR(4)


class TestSplit:
    def test_split_rejoin(self):
        """Splitting at t and evaluating halves gives same point."""
        pts = [(VDR(0), VDR(0)), (VDR(1), VDR(2)), (VDR(3), VDR(1))]
        t_split = VDR(1, 3)

        left, right = bezier_split(pts, t_split)

        # left curve at t=1 should equal original at t=1/3
        original = bezier_point(pts, t_split)
        left_end = bezier_point(left, VDR(1))
        assert left_end == original

    def test_endpoints_preserved(self):
        pts = [(VDR(0), VDR(0)), (VDR(1), VDR(1)), (VDR(2), VDR(0))]
        left, right = bezier_split(pts, VDR(1, 2))
        assert left[0] == pts[0]
        assert right[-1] == pts[-1]
```

---

## Domain Extension Checklist

1. **Create module file** in the appropriate subdirectory (`math/`, `physics/`, `signal/`, `ml/`).

2. **Write module docstring** with import example and one-liner usage.

3. **Define `__all__`** listing every public function and constant.

4. **Import only from VDR core.** `vdr.core`, `vdr.linalg`, `vdr.fn`, `vdr.math.transcendental` as needed. No float. No numpy. No external dependencies.

5. **Every function documents I/O/E** (Input, Output, Errors) and shows an example in the docstring.

6. **Every function returns VDR** (or Vec, Mat, bool, int — never float).

7. **Use `==` for verification**, not tolerance.

8. **Write tests** following the gym pattern: known values, identities, conservation laws, roundtrips, residual inspection, exact zero.

9. **Add to `__init__.py` docstring** domain list.

10. **Run the full test suite** to verify no regressions.

---

## Patterns to Avoid

**Never construct VDR from float:**

```python
# BAD
x = VDR(int(0.1 * 100), 100)   # 0.1 as float is 0.1000000000000000055511...

# GOOD
x = VDR(1, 10)
```

**Never use tolerance in assertions:**

```python
# BAD
assert abs(float(result.to_fraction()) - expected) < 1e-10

# GOOD
assert result == VDR(expected_num, expected_den)
```

**Never let D explode:**

```python
# BAD: unbounded chain of multiplications without frame
result = VDR(1)
for _ in range(1000):
    result = result * VDR(3, 7)   # denominator grows as 7^1000

# GOOD: use Q335 frame
from vdr.basis import to_qbasis, qb_mul
result = to_qbasis(VDR(1), bits=335)
factor = to_qbasis(VDR(3, 7), bits=335)
for _ in range(1000):
    result = qb_mul(result, factor, bits=335)  # D stays 2^335
```

**Never mix float and VDR in a computation chain:**

```python
# BAD
x = VDR(1, 3)
y = float(x) * 2.0         # lost exactness
z = VDR(int(y * 1000), 1000)  # garbage in, garbage out

# GOOD
x = VDR(1, 3)
y = x * VDR(2)              # stays exact
```

**Never ignore the remainder:**

```python
# BAD: treating active object as if it's closed
x = VDR(1, 3, Remainder(5))
print(x.v, "/", x.d)        # prints 1/3, but value is (1+5)/3 = 2

# GOOD: use to_fraction() or work with the full object
print(x.to_fraction())       # 2
```
