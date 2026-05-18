#!/usr/bin/env python3
"""
vdr-math quickstart — core concepts in 5 minutes.

Run: python examples/quickstart.py
"""

from vdr import VDR, Remainder, Vec, Mat

print("=" * 60)
print("VDR-MATH QUICKSTART")
print("=" * 60)

# --- 1. Basic rational arithmetic ---

print("\n--- Exact Rational Arithmetic ---")

x = VDR(1, 3)       # exact 1/3
y = VDR(2, 7)       # exact 2/7
z = x + y
print("1/3 + 2/7 =", z, "=", z.to_fraction())

z = x * y
print("1/3 * 2/7 =", z, "=", z.to_fraction())

z = x / y
print("1/3 / 2/7 =", z, "=", z.to_fraction())

# --- 2. Zero drift demonstration ---

print("\n--- Zero Drift ---")

x = VDR(1, 7)
step = VDR(1, 13)
for _ in range(100):
    x = x + step
for _ in range(100):
    x = x - step
print("200-op return to origin:", x, "== 1/7?", x == VDR(1, 7))

# --- 3. Active objects with remainder ---

print("\n--- Active Objects ---")

a = VDR(1, 2, Remainder(1))
print("Active object:", a, "  value =", a.to_fraction())
print("  is_closed:", a.is_closed)
print("  is_active:", a.is_active)

b = VDR(3)
c = a * b
print("Active * closed:", a, "*", b, "=", c, "  value =", c.to_fraction())

# --- 4. Linear algebra ---

print("\n--- Exact Linear Algebra ---")

m = Mat.from_ints([[1, 2], [3, 4]])
print("Matrix:")
print(m.pretty())

print("det =", m.det())

m_inv = m.inv()
print("Inverse:")
print(m_inv.pretty())

product = m.matmul(m_inv)
print("M * M^-1 == I?", product == Mat.identity(2))

# --- 5. Solve Ax = b ---

print("\n--- Exact Solve ---")

A = Mat.from_ints([[2, 1], [5, 3]])
b = Vec.from_ints([4, 7])
x = A.solve(b)
print("A =")
print(A.pretty())
print("b =", b)
print("x =", x)
print("A*x == b?", A.matvec(x) == b)

# --- 6. Functional remainder ---

print("\n--- Functional Remainder (sqrt 2) ---")

from vdr.fn import make_newton_fn, resolve

sqrt2_fn = make_newton_fn("sqrt2", lambda x: (x + VDR(2) / x) / VDR(2))
obj = VDR(0, 1, sqrt2_fn)

for depth in [1, 3, 5, 8]:
    val = resolve(obj, depth=depth)
    residual = val * val - VDR(2)
    frac = residual.to_fraction()
    print("  depth %d: residual x^2-2 = %s" % (depth, frac))

# --- 7. Export boundary ---

print("\n--- Export (Lossy Boundary) ---")

from vdr.export import to_decimal, to_float

x = VDR(1, 7)
print("VDR(1, 7) as decimal (20 digits):", to_decimal(x, 20))
print("VDR(1, 7) as float:", to_float(x))
print("(Loss belongs to the target format, not to VDR)")

# --- 8. Comparison ---

print("\n--- Exact Comparison ---")

a = VDR(1, 3)
b = VDR(1, 2)
print("1/3 < 1/2?", a < b)
print("1/3 == 2/6?", VDR(1, 3) == VDR(2, 6))

print("\n" + "=" * 60)
print("All operations exact. Zero drift. Zero truncation.")
print("=" * 60)
