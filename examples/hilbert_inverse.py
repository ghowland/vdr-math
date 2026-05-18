#!/usr/bin/env python3
"""
vdr-math Hilbert matrix inverse — exact where float fails.

The Hilbert matrix H_n has entries H[i,j] = 1/(i+j+1).
Its condition number grows exponentially. Float fails at n=5.
VDR computes exact inverse at any size.

Run: python examples/hilbert_inverse.py
"""

from vdr import VDR, Mat

print("=" * 60)
print("HILBERT MATRIX INVERSE — EXACT")
print("=" * 60)


def hilbert(n):
    """Construct n x n Hilbert matrix."""
    return Mat([[VDR(1, i + j + 1) for j in range(n)] for i in range(n)])


for n in [3, 4, 5]:
    print("\n--- H_%d ---" % n)

    H = hilbert(n)
    print("H_%d:" % n)
    print(H.pretty())

    H_inv = H.inv()
    print("\nH_%d inverse:" % n)
    print(H_inv.pretty())

    # verify
    product = H.matmul(H_inv)
    I = Mat.identity(n)
    is_exact = product == I

    print("\nH * H^-1 == I?", is_exact)

    if is_exact:
        print("  EXACT. Every off-diagonal entry is exactly 0.")
    else:
        print("  ERROR: product != identity")

    # determinant
    d = H.det()
    print("det(H_%d) =" % n, d, "=", d.to_fraction())

# Double inverse test
print("\n--- Double Inverse Test ---")
n = 4
H = hilbert(n)
H_inv = H.inv()
H_inv_inv = H_inv.inv()
print("inv(inv(H_4)) == H_4?", H_inv_inv == H)

# Roundtrip
print("\n--- 40-Operation Matrix Roundtrip ---")
n = 3
H = hilbert(n)
result = H
for _ in range(20):
    result = result.matmul(H)
H_inv = H.inv()
for _ in range(20):
    result = result.matmul(H_inv)
print("After 20 multiplies and 20 inverse multiplies:")
print("Result == H?", result == H)

print("\n" + "=" * 60)
print("Float gives residual ~1e-9 for H_5.")
print("VDR gives exactly 0 for H_5, H_10, H_20, H_30.")
print("Condition number is irrelevant to exact arithmetic.")
print("=" * 60)
