#!/usr/bin/env python3
"""
vdr-math Newton sqrt(2) — quadratic convergence, exact at every step.

Each Newton step doubles the number of correct digits.
The residual x^2 - 2 is an exact inspectable rational.

Run: python examples/newton_sqrt2.py
"""

from fractions import Fraction
from vdr import VDR
from vdr.fn import make_newton_fn, resolve
from vdr.export import to_decimal

print("=" * 60)
print("NEWTON SQRT(2) — EXACT RATIONAL AT EVERY DEPTH")
print("=" * 60)

# build the functional remainder
sqrt2_fn = make_newton_fn("sqrt2", lambda x: (x + VDR(2) / x) / VDR(2))

print("\nDepth | Correct digits | Residual x^2 - 2")
print("-" * 55)

for depth in range(1, 11):
    val = sqrt2_fn.expand(depth)
    residual = val * val - VDR(2)
    frac = residual.to_fraction()

    # count correct digits
    if frac == 0:
        digits = "exact"
    else:
        # approximate digit count from residual magnitude
        r_float = abs(float(frac))
        if r_float > 0:
            import math
            digits = int(-math.log10(r_float))
        else:
            digits = ">300"

    print("  %2d   | %12s   | numerator: %d digits, denom: %d digits" % (
        depth,
        str(digits),
        len(str(abs(frac.numerator))),
        len(str(abs(frac.denominator))),
    ))

# show the value at depth 7
print("\n--- Value at Depth 7 ---")
val = sqrt2_fn.expand(7)
dec = to_decimal(val, digits=50)
print("sqrt(2) ≈", dec)
print("actual  = 1.4142135623730950488016887242096980785696718753769...")

# demonstrate the functional remainder as VDR object
print("\n--- As VDR Object ---")
obj = VDR(0, 1, sqrt2_fn)
print("Object:", repr(obj))
print("This VDR has a function in R that returns sqrt(2) at any depth.")
print("resolve(obj, depth=10) gives >100 correct digits.")

resolved = resolve(obj, depth=10)
print("Resolved value:", to_decimal(resolved, digits=30))

# key point: the residual is EXACT
print("\n--- Key Point ---")
val = sqrt2_fn.expand(5)
residual = val * val - VDR(2)
print("At depth 5:")
print("  x^2 - 2 =", residual.to_fraction())
print("  This residual is EXACT — not an estimate, not a bound.")
print("  You know precisely how far from sqrt(2) you are.")

print("\n" + "=" * 60)
print("Quadratic convergence: digits double per step.")
print("1 -> 3 -> 6 -> 12 -> 24 -> 48 -> >100 digits.")
print("Every intermediate is an exact rational.")
print("=" * 60)
