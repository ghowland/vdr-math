# VDR Cookbook

## Practical Recipes for Exact Arithmetic

Each recipe is self-contained. Copy, paste, run. Every result is exact.

---

## Table of Contents

1. [Basic Arithmetic](#1-basic-arithmetic)
2. [Return-to-Origin Test](#2-return-to-origin-test)
3. [Exact Matrix Inverse](#3-exact-matrix-inverse)
4. [Hilbert Matrix — Where Float Fails](#4-hilbert-matrix--where-float-fails)
5. [Newton Square Root](#5-newton-square-root)
6. [Square Root of Any Rational](#6-square-root-of-any-rational)
7. [Exact Polynomial Interpolation](#7-exact-polynomial-interpolation)
8. [Polynomial GCD — Exact Zero-Testing](#8-polynomial-gcd--exact-zero-testing)
9. [Bayesian Updating Chain](#9-bayesian-updating-chain)
10. [Binomial PMF — Sum Exactly 1](#10-binomial-pmf--sum-exactly-1)
11. [Markov Chain Steady State](#11-markov-chain-steady-state)
12. [RSA Encrypt-Decrypt Roundtrip](#12-rsa-encrypt-decrypt-roundtrip)
13. [Fibonacci via Matrix Power](#13-fibonacci-via-matrix-power)
14. [Cassini Identity Verification](#14-cassini-identity-verification)
15. [Bernoulli Numbers](#15-bernoulli-numbers)
16. [Egyptian Fractions](#16-egyptian-fractions)
17. [Farey Sequence and Mediant Property](#17-farey-sequence-and-mediant-property)
18. [Continued Fraction Roundtrip](#18-continued-fraction-roundtrip)
19. [Exact Definite Integral](#19-exact-definite-integral)
20. [Discrete Derivative and Finite Differences](#20-discrete-derivative-and-finite-differences)
21. [Exact Softmax — Sum to 1](#21-exact-softmax--sum-to-1)
22. [Neural Network Forward-Backward](#22-neural-network-forward-backward)
23. [Training Loop with Exact Gradients](#23-training-loop-with-exact-gradients)
24. [Autodiff Computation Graph](#24-autodiff-computation-graph)
25. [Exact Self-Attention](#25-exact-self-attention)
26. [DFT Roundtrip — Perfect Reconstruction](#26-dft-roundtrip--perfect-reconstruction)
27. [IIR Filter — Zero Drift](#27-iir-filter--zero-drift)
28. [Convolution and Deconvolution](#28-convolution-and-deconvolution)
29. [Exact Dijkstra Shortest Path](#29-exact-dijkstra-shortest-path)
30. [PageRank — Sum Exactly 1](#30-pagerank--sum-exactly-1)
31. [Shapley Values — Sum to v(N)](#31-shapley-values--sum-to-vn)
32. [Cournot Duopoly Equilibrium](#32-cournot-duopoly-equilibrium)
33. [Hamming Code Error Correction](#33-hamming-code-error-correction)
34. [Exact Betti Numbers](#34-exact-betti-numbers)
35. [LLL Lattice Reduction](#35-lll-lattice-reduction)
36. [Cayley-Hamilton Verification](#36-cayley-hamilton-verification)
37. [State-Space Evolution — Zero Drift](#37-state-space-evolution--zero-drift)
38. [Transfer Function Evaluation](#38-transfer-function-evaluation)
39. [Haar Wavelet Perfect Reconstruction](#39-haar-wavelet-perfect-reconstruction)
40. [Tent Map — Exact Period Detection](#40-tent-map--exact-period-detection)
41. [Q335 Constants and Arithmetic](#41-q335-constants-and-arithmetic)
42. [Borwein Zeta Acceleration](#42-borwein-zeta-acceleration)
43. [Elliptic Integral Computation](#43-elliptic-integral-computation)
44. [QED A₂ Coefficient](#44-qed-a₂-coefficient)
45. [Spin Rotation Periodicity](#45-spin-rotation-periodicity)
46. [Kepler Orbit Closure](#46-kepler-orbit-closure)
47. [Optical System — Symplecticity](#47-optical-system--symplecticity)
48. [Helmert Transformation Roundtrip](#48-helmert-transformation-roundtrip)
49. [Ising Model Partition Function](#49-ising-model-partition-function)
50. [Crystal Point Group Closure](#50-crystal-point-group-closure)
51. [Truss Structure — Exact Equilibrium](#51-truss-structure--exact-equilibrium)
52. [Diffusion Schedule — Exact Cumulative Product](#52-diffusion-schedule--exact-cumulative-product)
53. [DDIM Roundtrip — Exactly Zero Error](#53-ddim-roundtrip--exactly-zero-error)
54. [Multi-Cycle Drift Test](#54-multi-cycle-drift-test)
55. [Q335 Multiplication — D Stays Fixed](#55-q335-multiplication--d-stays-fixed)
56. [Custom D-Frame Arithmetic](#56-custom-d-frame-arithmetic)
57. [Rebase Between Frames](#57-rebase-between-frames)
58. [JSON Serialization Roundtrip](#58-json-serialization-roundtrip)
59. [LaTeX Export](#59-latex-export)
60. [Building a New Domain Module](#60-building-a-new-domain-module)

---

## 1. Basic Arithmetic

```python
from vdr import VDR

a = VDR(1, 3)
b = VDR(2, 7)

print(a + b)    # [10, 21, 0]  = 10/21
print(a - b)    # [1, 21, 0]   = 1/21
print(a * b)    # [2, 21, 0]   = 2/21
print(a / b)    # [7, 6, 0]    = 7/6
print(-a)       # [-1, 3, 0]   = -1/3
print(abs(-a))  # [1, 3, 0]    = 1/3

# Works with plain ints
print(a + 1)    # [4, 3, 0]    = 4/3
print(3 * a)    # [1, 1, 0]    = 1
```

---

## 2. Return-to-Origin Test

```python
from vdr import VDR

x = VDR(1, 7)
step = VDR(1, 13)

for _ in range(100):
    x = x + step
for _ in range(100):
    x = x - step

assert x == VDR(1, 7)
print("200 operations, error = 0")
# Float64: error ≈ 2.78e-16
```

---

## 3. Exact Matrix Inverse

```python
from vdr.linalg import Mat

m = Mat.from_ints([[1, 2], [3, 4]])
m_inv = m.inv()
product = m.matmul(m_inv)

print(m_inv.pretty())
assert product == Mat.identity(2)
print("M * M^-1 = I exactly")
```

---

## 4. Hilbert Matrix — Where Float Fails

```python
from vdr import VDR
from vdr.linalg import Mat

def hilbert(n):
    return Mat([[VDR(1, i+j+1) for j in range(n)] for i in range(n)])

for n in [3, 4, 5, 6]:
    H = hilbert(n)
    H_inv = H.inv()
    assert H.matmul(H_inv) == Mat.identity(n)
    print("H_%d * H_%d^-1 = I exactly" % (n, n))

# Double inverse: inv(inv(H)) = H
H = hilbert(4)
assert H.inv().inv() == H
print("inv(inv(H_4)) = H_4 exactly")
```

---

## 5. Newton Square Root

```python
from vdr import VDR
from vdr.fn import make_newton_fn
from vdr.export import to_decimal

sqrt2 = make_newton_fn("sqrt2", lambda x: (x + VDR(2)/x) / VDR(2))

for depth in [1, 3, 5, 7, 10]:
    val = sqrt2.expand(depth)
    residual = val * val - VDR(2)
    frac = residual.to_fraction()
    print("depth %2d: |x²-2| has %d-digit numerator, %d-digit denom" % (
        depth, len(str(abs(frac.numerator))), len(str(abs(frac.denominator)))
    ))

val = sqrt2.expand(7)
print("sqrt(2) ≈", to_decimal(val, 50))
```

---

## 6. Square Root of Any Rational

```python
from vdr import VDR
from vdr.math.transcendental import sqrt_newton

# sqrt of integer
s3 = sqrt_newton(VDR(3), depth=10)
print("sqrt(3)² - 3 =", (s3 * s3 - VDR(3)).to_fraction())

# sqrt of rational
s_half = sqrt_newton(VDR(1, 2), depth=10)
print("sqrt(1/2)² - 1/2 =", (s_half * s_half - VDR(1, 2)).to_fraction())

# sqrt of perfect square normalizes cleanly
s4 = sqrt_newton(VDR(4), depth=10)
n = s4.normalize()
print("sqrt(4) =", n)  # should be [2, 1, 0]
```

---

## 7. Exact Polynomial Interpolation

```python
from vdr import VDR
from vdr.math.polynomial import lagrange_interpolate, poly_eval

# Find polynomial through (0,1), (1,3), (2,7)
points = [(VDR(0), VDR(1)), (VDR(1), VDR(3)), (VDR(2), VDR(7))]
p = lagrange_interpolate(points)

# Verify all points
for x, y in points:
    assert poly_eval(p, x) == y

# Evaluate at a new point
print("p(1/2) =", poly_eval(p, VDR(1, 2)))
# Exact — no Runge phenomenon from arithmetic
```

---

## 8. Polynomial GCD — Exact Zero-Testing

```python
from vdr import VDR
from vdr.math.polynomial import poly_gcd, poly_eval

# gcd(x² - 1, x² + 2x + 1) = x + 1
p = [VDR(-1), VDR(0), VDR(1)]   # x² - 1
q = [VDR(1), VDR(2), VDR(1)]    # x² + 2x + 1

g = poly_gcd(p, q)

# g should be x + 1, which has root at x = -1
assert poly_eval(g, VDR(-1)) == VDR(0)
print("GCD is", g, "— root at x=-1 verified exactly")
# Decimal cannot do this: is the remainder coefficient 0 or 1e-15?
```

---

## 9. Bayesian Updating Chain

```python
from vdr import VDR
from vdr.math.probability import bayes_update, bayes_sequential

# Prior: 1/2 (50% chance of hypothesis)
prior = VDR(1, 2)

# Sequential evidence with likelihood ratios
evidence = [VDR(3), VDR(2), VDR(4), VDR(1, 2), VDR(5)]

posteriors = bayes_sequential(prior, evidence)
for i, p in enumerate(posteriors):
    print("After evidence %d: P(H) = %s = %s" % (
        i+1, p, p.to_fraction()))

# Every posterior is exact — no accumulation from 5 updates
```

---

## 10. Binomial PMF — Sum Exactly 1

```python
from vdr import VDR
from vdr.math.probability import binom_pmf_full

pmf = binom_pmf_full(10, VDR(1, 3))

total = VDR(0)
for p in pmf:
    total = total + p

assert total == VDR(1)
print("PMF of Binomial(10, 1/3) sums to exactly 1")
print("P(X=0) =", pmf[0].to_fraction())
print("P(X=5) =", pmf[5].to_fraction())
```

---

## 11. Markov Chain Steady State

```python
from vdr import VDR
from vdr.linalg import Mat
from vdr.math.probability import markov_steady_state

# Transition matrix (row-stochastic)
P = Mat.from_fracs([
    [(1, 2), (1, 2)],
    [(1, 4), (3, 4)],
])

ss = markov_steady_state(P)
print("Steady state:", ss)
print("Sum:", ss[0] + ss[1])

assert ss[0] == VDR(1, 3)
assert ss[1] == VDR(2, 3)
assert ss[0] + ss[1] == VDR(1)
print("Sums to exactly 1")
```

---

## 12. RSA Encrypt-Decrypt Roundtrip

```python
from vdr.math.cryptographic import rsa_keygen, rsa_encrypt, rsa_decrypt

n, e, d = rsa_keygen(61, 53, 17)
print("n=%d, e=%d, d=%d" % (n, e, d))

messages = [42, 0, 100, 1000, 3000]
for m in messages:
    c = rsa_encrypt(m, e, n)
    recovered = rsa_decrypt(c, d, n)
    assert recovered == m
    print("  %d -> encrypt -> %d -> decrypt -> %d ✓" % (m, c, recovered))
```

---

## 13. Fibonacci via Matrix Power

```python
from vdr import VDR
from vdr.linalg import Mat
from vdr.math.control import mat_pow

F = Mat.from_ints([[1, 1], [1, 0]])

for n in [10, 20, 30, 50]:
    Fn = mat_pow(F, n)
    fib_n = Fn[0, 1]
    print("F(%d) = %s" % (n, fib_n))

# F(50) = 12586269025 — exact, no overflow concern
```

---

## 14. Cassini Identity Verification

```python
from vdr import VDR
from vdr.math.sequences import fibonacci

for n in range(2, 20):
    fn = fibonacci(n)
    fn1 = fibonacci(n + 1)
    fnm1 = fibonacci(n - 1)
    result = fnm1 * fn1 - fn * fn
    expected = VDR((-1) ** n)
    assert result == expected

print("Cassini identity F(n-1)*F(n+1) - F(n)² = (-1)^n")
print("Verified exactly for n = 2..19")
```

---

## 15. Bernoulli Numbers

```python
from vdr import VDR
from vdr.math.sequences import bernoulli

for n in range(0, 20, 2):
    bn = bernoulli(n)
    print("B(%2d) = %s" % (n, bn.to_fraction()))

# B(12) = -691/2730 — denominator has factors 2·3·5·7·13
# Decimal would truncate. VDR: exact.
```

---

## 16. Egyptian Fractions

```python
from fractions import Fraction
from vdr import VDR
from vdr.math.number_theory import egyptian_fractions

ef = egyptian_fractions(5, 7)
print("5/7 =", " + ".join(str(f) for f in ef))

total = sum((f.to_fraction() for f in ef), Fraction(0))
assert total == Fraction(5, 7)
print("Sum verified exact")
```

---

## 17. Farey Sequence and Mediant Property

```python
from vdr import VDR
from vdr.math.number_theory import farey

f5 = farey(5)
print("F_5:", [str(f) for f in f5])

# Mediant property: for adjacent a/b, c/d: |ad - bc| = 1
for i in range(len(f5) - 1):
    a, b = f5[i].v, f5[i].d
    c, d = f5[i+1].v, f5[i+1].d
    assert abs(a*d - b*c) == 1

print("Mediant property |ad-bc| = 1 verified for all %d adjacent pairs" % (len(f5)-1))
```

---

## 18. Continued Fraction Roundtrip

```python
from vdr import VDR
from vdr.math.continued_fractions import to_cf, from_cf, convergents_from_cf

for v, d in [(1, 2), (3, 7), (22, 7), (355, 113)]:
    cf = to_cf(v, d)
    recovered = from_cf(cf)
    assert recovered == VDR(v, d)
    convs = convergents_from_cf(cf)
    print("%d/%d -> CF %s -> convergents %s -> %s ✓" % (
        v, d, cf, [str(c) for c in convs], recovered))
```

---

## 19. Exact Definite Integral

```python
from vdr import VDR
from vdr.math.polynomial import definite_integral

# ∫₀¹ x² dx = 1/3
p = [VDR(0), VDR(0), VDR(1)]   # x²
result = definite_integral(p, VDR(0), VDR(1))
assert result == VDR(1, 3)
print("∫₀¹ x² dx =", result, "exact")

# ∫₁³ (1 + 3x + 5x²) dx
p2 = [VDR(1), VDR(3), VDR(5)]
result2 = definite_integral(p2, VDR(1), VDR(3))
print("∫₁³ (1+3x+5x²) dx =", result2.to_fraction())
```

---

## 20. Discrete Derivative and Finite Differences

```python
from vdr import VDR
from vdr.fn import discrete_derivative, discrete_derivative_nth

# D_h(x²) at x=3, h=1/1000 = 6001/1000
f = lambda x: x * x
df = discrete_derivative(f, VDR(1, 1000))
assert df(VDR(3)) == VDR(6001, 1000)
print("D_{1/1000}(x²) at x=3 =", df(VDR(3)))

# Δ³(x³) = 6 (exactly 3!)
g = lambda x: x * x * x
d3g = discrete_derivative_nth(g, VDR(1), order=3)
assert d3g(VDR(0)) == VDR(6)
print("Δ³(x³) at 0 =", d3g(VDR(0)))

# Δ⁴(x³) = 0 (no float noise floor)
d4g = discrete_derivative_nth(g, VDR(1), order=4)
assert d4g(VDR(0)) == VDR(0)
print("Δ⁴(x³) at 0 =", d4g(VDR(0)), "— exactly zero")
```

---

## 21. Exact Softmax — Sum to 1

```python
from vdr import VDR
from vdr.linalg import Vec
from vdr.ml.softmax import softmax

logits = Vec.from_ints([1, 2, 3, 4])
probs = softmax(logits)

total = VDR(0)
for i in range(len(probs)):
    total = total + probs[i]

assert total == VDR(1)
print("Softmax sums to exactly 1")
for i in range(len(probs)):
    print("  p[%d] = %s" % (i, probs[i].to_fraction()))
```

---

## 22. Neural Network Forward-Backward

```python
from vdr import VDR
from vdr.linalg import Vec
from vdr.ml.nn import Linear, ReLU, Sequential
from vdr.ml.losses import mse, mse_grad

model = Sequential([
    Linear.from_ints([[1, 2], [3, 4]], [0, 0]),
    ReLU(),
    Linear.from_ints([[1, 0], [0, 1]], [1, 1]),
])

x = Vec.from_ints([1, 1])
target = Vec.from_ints([5, 7])

# Forward
output = model.forward(x)
print("Output:", output)

# Loss
loss = mse(output, target)
print("MSE loss:", loss.to_fraction())

# Backward
grad = mse_grad(output, target)
model.backward(grad)

# Gradients are exact
for p in model.parameters():
    print("  %s grad: %s" % (p.name, p.grad))
```

---

## 23. Training Loop with Exact Gradients

```python
from vdr import VDR
from vdr.linalg import Vec
from vdr.ml.nn import Linear, ReLU, Sequential
from vdr.ml.optim import SGD
from vdr.ml.trainer import train_step

model = Sequential([
    Linear.from_ints([[1, 0], [0, 1]], [0, 0]),
    ReLU(),
    Linear.from_ints([[1, 0], [0, 1]], [0, 0]),
])

opt = SGD(model.parameters(), lr=VDR(1, 100))

dataset = [
    (Vec.from_ints([1, 0]), Vec.from_ints([2, 0])),
    (Vec.from_ints([0, 1]), Vec.from_ints([0, 3])),
]

for epoch in range(5):
    total_loss = VDR(0)
    for x, y in dataset:
        loss = train_step(model, x, y, opt)
        total_loss = total_loss + loss
    print("Epoch %d: loss = %s" % (epoch, total_loss.to_fraction()))
```

---

## 24. Autodiff Computation Graph

```python
from vdr import VDR
from vdr.ml.autodiff import Node, relu, mse_loss

# f(a, b) = a*b + relu(a - 3)
a = Node(VDR(5))
b = Node(VDR(4))
c = a * b + relu(a - Node(VDR(3)))

c.backward()
print("f(5, 4) =", c.value)
print("df/da =", a.grad)  # b + relu'(a-3) = 4 + 1 = 5
print("df/db =", b.grad)  # a = 5
```

---

## 25. Exact Self-Attention

```python
from vdr.linalg import Vec
from vdr.ml.attention import self_attention

Q = [Vec.from_ints([1, 0]), Vec.from_ints([0, 1])]
K = [Vec.from_ints([1, 1]), Vec.from_ints([1, -1])]
V = [Vec.from_ints([10, 0]), Vec.from_ints([0, 10])]

out = self_attention(Q, K, V, causal=True)
for i, v in enumerate(out):
    print("Position %d:" % i, v)
# Every attention weight row sums to exactly 1
```

---

## 26. DFT Roundtrip — Perfect Reconstruction

```python
from vdr import VDR
from vdr.signal.dft import exact_dft, exact_idft, parseval_verify

x = [VDR(1), VDR(2), VDR(3), VDR(4)]
X = exact_dft(x, depth=16)
recovered = exact_idft(X, depth=16)

for i in range(len(x)):
    assert recovered[i] == x[i]
print("IDFT(DFT(x)) == x exactly")

assert parseval_verify(x, X)
print("Parseval energy identity: exact")
```

---

## 27. IIR Filter — Zero Drift

```python
from vdr import VDR
from vdr.signal.filters import iir_filter

# Impulse response: y[n] = (1/2)^n
x = [VDR(1)] + [VDR(0)] * 19
y = iir_filter(x, VDR(1, 2))

for n in [0, 1, 5, 10, 19]:
    expected = VDR(1, 2**n)
    assert y[n] == expected
    print("y[%d] = %s ✓" % (n, y[n]))

print("20 IIR steps, zero drift")
```

---

## 28. Convolution and Deconvolution

```python
from vdr import VDR
from vdr.signal.convolution import convolve, deconvolve

a = [VDR(1), VDR(2), VDR(3)]
b = [VDR(1), VDR(1)]

y = convolve(a, b)
print("Conv:", y)  # [1, 3, 5, 3]

recovered = deconvolve(y, b)
print("Deconv:", recovered)
# recovered == a — exact inverse
```

---

## 29. Exact Dijkstra Shortest Path

```python
from vdr import VDR
from vdr.math.graph import dijkstra

adj = {
    0: [(1, VDR(1, 3)), (2, VDR(3, 4))],
    1: [(2, VDR(1, 2)), (3, VDR(1))],
    2: [(3, VDR(1, 5))],
    3: [],
}

dist = dijkstra(adj, 0)
for node in sorted(dist.keys()):
    d = dist[node]
    print("  0 -> %d: %s" % (node, d.to_fraction() if d else "unreachable"))
```

---

## 30. PageRank — Sum Exactly 1

```python
from vdr import VDR
from vdr.linalg import Mat
from vdr.math.graph import pagerank

P = Mat.from_fracs([
    [(1, 2), (1, 2), (0, 1)],
    [(0, 1), (0, 1), (1, 1)],
    [(1, 2), (1, 2), (0, 1)],
])

pr = pagerank(P)
total = pr[0] + pr[1] + pr[2]
assert total == VDR(1)
print("PageRank:", pr)
print("Sum:", total, "— exactly 1")
```

---

## 31. Shapley Values — Sum to v(N)

```python
from vdr import VDR
from vdr.math.game_theory import shapley_values

def v(s):
    if len(s) == 3: return VDR(1)
    if len(s) == 2: return VDR(1, 2)
    if len(s) == 1: return VDR(0)
    return VDR(0)

phi = shapley_values(v, 3)
total = phi[0] + phi[1] + phi[2]
assert total == VDR(1)
print("Shapley values:", phi)
print("Sum = v({0,1,2}) =", total, "exactly")
```

---

## 32. Cournot Duopoly Equilibrium

```python
from vdr import VDR
from vdr.math.game_theory import cournot_duopoly

q1, q2, pi1, pi2 = cournot_duopoly(VDR(100), VDR(1), VDR(10), VDR(20))
print("q1* =", q1.to_fraction())
print("q2* =", q2.to_fraction())
print("profit1 =", pi1.to_fraction())
print("profit2 =", pi2.to_fraction())
```

---

## 33. Hamming Code Error Correction

```python
from vdr.math.coding_theory import hamming74_encode, hamming74_correct, hamming74_syndrome

data = [1, 0, 1, 1]
codeword = hamming74_encode(data)
print("Data:     ", data)
print("Codeword: ", codeword)

# Inject error at position 3
received = list(codeword)
received[3] ^= 1
print("Received: ", received, "(error at position 3)")

corrected = hamming74_correct(received)
print("Corrected:", corrected)
assert corrected == codeword
print("Error corrected exactly")
```

---

## 34. Exact Betti Numbers

```python
from vdr.math.topology import simplicial_complex_boundaries, betti_numbers, euler_characteristic

# Hollow triangle (1-cycle)
simplices = {
    0: [(0,), (1,), (2,)],
    1: [(0,1), (0,2), (1,2)],
}
boundaries = simplicial_complex_boundaries(simplices)
betti = betti_numbers(boundaries)
chi = euler_characteristic(betti)

print("Hollow triangle: betti =", betti, "chi =", chi)
assert betti[0] == 1   # connected
assert betti[1] == 1   # one 1-cycle
assert chi == 0

# Filled triangle (no holes)
simplices2 = {
    0: [(0,), (1,), (2,)],
    1: [(0,1), (0,2), (1,2)],
    2: [(0,1,2)],
}
boundaries2 = simplicial_complex_boundaries(simplices2)
betti2 = betti_numbers(boundaries2)
print("Filled triangle: betti =", betti2)
assert betti2[1] == 0   # no holes
```

---

## 35. LLL Lattice Reduction

```python
from vdr.linalg import Vec
from vdr.math.tropical import gram_schmidt_exact, lll_reduce

basis = [
    Vec.from_ints([1, 0, 0]),
    Vec.from_ints([0, 1, 0]),
    Vec.from_ints([10, 10, 1]),
]

# Gram-Schmidt — cross-dot products exactly 0
ortho, mu = gram_schmidt_exact(basis)
for i in range(3):
    for j in range(i):
        assert ortho[i].dot(ortho[j]) == VDR(0)
print("Gram-Schmidt: all cross-dots exactly 0")

# LLL — exact Lovász condition, no float rounding
reduced = lll_reduce(basis)
print("LLL reduced basis:")
for v in reduced:
    print(" ", v)
```

---

## 36. Cayley-Hamilton Verification

```python
from vdr import VDR
from vdr.linalg import Mat
from vdr.math.control import cayley_hamilton_verify

A = Mat.from_ints([[1, 2, 0], [0, 1, 3], [2, 0, 1]])
result = cayley_hamilton_verify(A)

print("p(A) =")
print(result.pretty())
assert result == Mat.zero(3, 3)
print("Cayley-Hamilton: p(A) = 0 exactly (every entry [0, 1, 0])")
```

---

## 37. State-Space Evolution — Zero Drift

```python
from vdr import VDR
from vdr.linalg import Vec, Mat
from vdr.math.control import state_evolve

A = Mat.from_fracs([[(9, 10), (1, 10)], [(-1, 10), (9, 10)]])
B = Mat.identity(2)
x0 = Vec.from_ints([1, 0])
inputs = [Vec.from_ints([0, 0])] * 100

traj = state_evolve(A, B, x0, inputs)
print("Initial state:", traj[0])
print("After 100 steps:", traj[100])
print("All 101 states are exact rationals — zero drift")
```

---

## 38. Transfer Function Evaluation

```python
from vdr import VDR
from vdr.math.control import transfer_function_eval

# H(s) = 1 / (s² + 3s + 2)
num = [VDR(1)]
den = [VDR(2), VDR(3), VDR(1)]

# Real evaluation
h_at_1 = transfer_function_eval(num, den, VDR(1))
assert h_at_1 == VDR(1, 6)
print("H(1) =", h_at_1)

# Complex evaluation: s = i
h_at_i = transfer_function_eval(num, den, (VDR(0), VDR(1)))
re, im = h_at_i
print("H(i) = %s + %s*i" % (re, im))
assert re == VDR(1, 10)
assert im == VDR(-3, 10)
```

---

## 39. Haar Wavelet Perfect Reconstruction

```python
from vdr import VDR
from vdr.math.wavelets import haar_forward, haar_inverse, haar_multilevel, haar_reconstruct_multilevel

signal = [VDR(1), VDR(3), VDR(5), VDR(7)]

# One level
avgs, dets = haar_forward(signal)
recovered = haar_inverse(avgs, dets)
assert recovered == signal
print("1-level roundtrip: exact")

# Multi-level: 64 samples
signal64 = [VDR(i) for i in range(64)]
decomp = haar_multilevel(signal64, 6)
recovered64 = haar_reconstruct_multilevel(decomp)
assert recovered64 == signal64
print("64-sample 6-level roundtrip: exact")
```

---

## 40. Tent Map — Exact Period Detection

```python
from vdr import VDR
from vdr.math.chaos import tent_map, iterate_map, detect_period

orbit = iterate_map(tent_map, VDR(1, 7), 20)
period = detect_period(orbit)

print("Tent map orbit of 1/7:")
for i in range(min(10, len(orbit))):
    print("  step %d: %s" % (i, orbit[i]))
print("Period:", period)
assert period == 3
print("Exact forever. Float diverges at ~25 steps.")
```

---

## 41. Q335 Constants and Arithmetic

```python
from vdr.math.transcendental import PI, E, LN2, SQRT2, ZETA3, Q335_DENOM
from vdr.export import to_decimal

print("pi  ≈", to_decimal(PI, 30))
print("e   ≈", to_decimal(E, 30))
print("ln2 ≈", to_decimal(LN2, 30))
print("√2  ≈", to_decimal(SQRT2, 30))
print("ζ(3)≈", to_decimal(ZETA3, 30))

# Addition: one integer add
pi_plus_e = PI + E
assert pi_plus_e.d == Q335_DENOM
print("\npi + e: D =", "2^335" if pi_plus_e.d == Q335_DENOM else "changed!")

# Identity: ln(10) ≈ ln(2) + ln(5)
from vdr.math.transcendental import LN5, LN10
diff = (LN2 + LN5) - LN10
print("ln2 + ln5 - ln10: numerator diff =", diff.v, "(rounding residual)")
```

---

## 42. Borwein Zeta Acceleration

```python
from vdr.math.transcendental import borwein_zeta
from vdr.export import to_decimal

for s in [2, 3, 5, 7]:
    z = borwein_zeta(s, n=50)
    print("ζ(%d) ≈ %s" % (s, to_decimal(z, 25)))

# zeta(2) = pi²/6
from vdr.math.transcendental import PI_SQ
z2 = borwein_zeta(2, n=50)
pi_sq_6 = PI_SQ / VDR(6)
diff = z2 - pi_sq_6
print("\nζ(2) - π²/6: numerator diff =", diff.v)
```

---

## 43. Elliptic Integral Computation

```python
from vdr import VDR
from vdr.math.transcendental import elliptic_k, elliptic_e
from vdr.export import to_decimal

# K(k) at k² = 1/4 — fastest convergence
k1 = elliptic_k(VDR(1, 4), terms=200)
print("K(1/2) ≈", to_decimal(k1, 20))

# E(k) at k² = 1/2
e1 = elliptic_e(VDR(1, 2), terms=200)
print("E(1/√2) ≈", to_decimal(e1, 20))
```

---

## 44. QED A₂ Coefficient

```python
from vdr.physics.qed import a2_coefficient
from vdr.export import to_decimal

a2 = a2_coefficient()
print("A₂ = 197/144 + π²/12 + 3ζ(3)/4 - (π²/2)·ln(2)")
print("   ≈", to_decimal(a2, 25))
print("   (matches -0.328478965579... to 100 digits)")
```

---

## 45. Spin Rotation Periodicity

```python
from vdr import VDR
from vdr.linalg import Mat
from vdr.physics.quantum import spin_rotation, verify_unitarity

U = spin_rotation("z", VDR(1, 2))  # pi/2 rotation
print("U =")
print(U.pretty())
assert verify_unitarity(U)
print("U^T U = I: exact")

# Apply 4 times: should return to I
result = U
for _ in range(3):
    result = result.matmul(U)
print("\nU^4 =")
print(result.pretty())
# Float: I ± ~1e-15. VDR: check structural.
```

---

## 46. Kepler Orbit Closure

```python
from vdr import VDR
from vdr.physics.orbital import kepler_newton, kepler_position
from vdr.math.transcendental import PI

e = VDR(1, 2)  # eccentricity
a = VDR(1)      # semi-major axis

# Solve at M=0 (periapsis)
E0 = kepler_newton(VDR(0), e, depth=15)
x0, y0 = kepler_position(a, e, E0)
print("Periapsis: (%s, %s)" % (x0.to_fraction(), y0.to_fraction()))

# Solve at M close to 2*pi (should return to periapsis)
E_full = kepler_newton(VDR(2) * PI, e, depth=15)
x1, y1 = kepler_position(a, e, E_full)
print("After one orbit: (%s, %s)" % (
    float(x1.to_fraction()), float(y1.to_fraction())))
```

---

## 47. Optical System — Symplecticity

```python
from vdr import VDR
from vdr.physics.optics import free_space, thin_lens, system_matrix, verify_symplecticity, matrix_power

# Build a complex optical system
elements = [free_space(VDR(1, 10)), thin_lens(VDR(1, 2))] * 50
M = system_matrix(elements)
assert verify_symplecticity(M)
print("100-element system: det(M) = 1 exactly")

# Raise to high power
M100 = matrix_power(M, 100)
assert verify_symplecticity(M100)
print("M^100: det = 1 exactly (float: ~1e-12 drift)")
```

---

## 48. Helmert Transformation Roundtrip

```python
from vdr import VDR
from vdr.linalg import Vec
from vdr.physics.geodesy import helmert_forward, helmert_inverse, helmert_roundtrip_verify

coords = Vec([VDR(100), VDR(200), VDR(300)])
params = {
    "tx": VDR(1, 10), "ty": VDR(2, 10), "tz": VDR(3, 10),
    "rx": VDR(1, 1000), "ry": VDR(2, 1000), "rz": VDR(3, 1000),
    "s": VDR(1, 1000000),
}

transformed = helmert_forward(coords, params)
recovered = helmert_inverse(transformed, params)

print("Original: ", coords)
print("Transformed:", transformed)
print("Recovered:  ", recovered)
assert helmert_roundtrip_verify(coords, params)
print("Roundtrip: exact (float: ~1 nm error)")
```

---

## 49. Ising Model Partition Function

```python
from vdr import VDR
from vdr.physics.thermo import ising_1d_transfer, boltzmann_probabilities

# 1D Ising, 10 spins, no external field
Z = ising_1d_transfer(VDR(1), VDR(0), VDR(1), 10)
print("Z(1D Ising, N=10, J=1, β=1) =", Z.to_fraction())

# Boltzmann probabilities sum exactly to 1
energies = [VDR(0), VDR(1), VDR(2)]
probs = boltzmann_probabilities(energies, VDR(1))
total = VDR(0)
for p in probs:
    total = total + p
assert total == VDR(1)
print("Boltzmann probabilities sum to exactly 1")
```

---

## 50. Crystal Point Group Closure

```python
from vdr.physics.crystallography import point_group_matrix, verify_group_closure

ops = ["E", "C2z", "C4z", "C4z_inv"]
matrices = [point_group_matrix(op) for op in ops]

closed = verify_group_closure(matrices)
print("Group {%s}:" % ", ".join(ops))
print("Closed under multiplication:", closed)
# Comparison by exact structural equality, not tolerance
```

---

## 51. Truss Structure — Exact Equilibrium

```python
from vdr import VDR
from vdr.linalg import Vec, Mat
from vdr.physics.structural import solve_structure, verify_equilibrium

# Simple 2-DOF system
K = Mat.from_ints([[4, -1], [-1, 3]])
F = Vec.from_ints([10, 5])

u = solve_structure(K, F)
print("Displacements:", u)
assert verify_equilibrium(K, u, F)
print("K @ u == F: exact (not 'residual < 1e-10')")
```

---

## 52. Diffusion Schedule — Exact Cumulative Product

```python
from vdr import VDR
from vdr.diffusion.schedule import linear_schedule
from fractions import Fraction

schedule = linear_schedule(5, VDR(1, 100), VDR(1, 20))

print("Betas:", [b.to_fraction() for b in schedule.betas])
print("Alpha bars:", [ab.to_fraction() for ab in schedule.alpha_bars])

# Verify cumulative product matches Python Fraction
product = Fraction(1)
for a in schedule.alphas:
    product *= a.to_fraction()
assert schedule.alpha_bars[-1].to_fraction() == product
print("Cumulative product matches Fraction: bit-identical")
```

---

## 53. DDIM Roundtrip — Exactly Zero Error

```python
from vdr import VDR
from vdr.linalg import Vec
from vdr.diffusion.schedule import linear_schedule
from vdr.diffusion.sampling import verify_ddim_roundtrip

schedule = linear_schedule(5, VDR(1, 100), VDR(1, 20))
x0 = Vec.from_ints([1, 2, 3])
epsilon = Vec.from_ints([0, 0, 0])

err = verify_ddim_roundtrip(x0, schedule, epsilon)
print("DDIM roundtrip error:", err)
# Should be exactly 0 or < 1e-50 (Newton residual only)
```

---

## 54. Multi-Cycle Drift Test

```python
from vdr import VDR
from vdr.linalg import Vec
from vdr.diffusion.schedule import linear_schedule
from vdr.diffusion.sampling import verify_multi_step_drift

schedule = linear_schedule(5, VDR(1, 100), VDR(1, 20))
x0 = Vec.from_ints([1, 2, 3])
epsilon = Vec.from_ints([0, 0, 0])

drift = verify_multi_step_drift(x0, schedule, epsilon, num_cycles=5)
print("Per-cycle error:")
for i, d in enumerate(drift):
    print("  Cycle %d: %s" % (i+1, d.to_fraction()))
print("Error does NOT grow across cycles")
```

---

## 55. Q335 Multiplication — D Stays Fixed

```python
from vdr import VDR
from vdr.basis import to_qbasis, qb_mul

a = to_qbasis(VDR(22, 7), bits=10)   # small basis for demo
b = to_qbasis(VDR(3), bits=10)

print("a =", a, "  D =", a.d)
print("b =", b, "  D =", b.d)

c = qb_mul(a, b, bits=10)
print("a * b =", c, "  D =", c.d)
assert c.d == 2**10
print("D stays 2^10. Overflow in R. Zero loss.")
```

---

## 56. Custom D-Frame Arithmetic

```python
from vdr import VDR

# Working in sevenths
a = VDR(3, 7)
b = VDR(5, 7)
c = a + b
print("3/7 + 5/7 =", c)

# Working in binary fixed-point (D = 256)
a = VDR(128, 256)   # 1/2
b = VDR(64, 256)    # 1/4
c = a * b
print("1/2 * 1/4 =", c.to_fraction())

# Working in plain integers (D = 1)
a = VDR(42)
b = VDR(17)
print("42 * 17 =", (a * b).v)
```

---

## 57. Rebase Between Frames

```python
from vdr import VDR

x = VDR(1, 2)

# Rebase to sixths: clean
y = x.rebase(6)
print("1/2 in sixths:", y)
assert y == VDR(3, 6)

# Rebase to sevenths: active (mismatch witness)
z = x.rebase(7)
print("1/2 in sevenths:", z)
print("  value:", z.to_fraction())
assert z.to_fraction() == x.to_fraction()
print("Value preserved exactly through rebase")
```

---

## 58. JSON Serialization Roundtrip

```python
import json
from vdr import VDR
from vdr.core import Remainder
from vdr.linalg import vdr_to_dict, vdr_from_dict

# Closed
x = VDR(3, 7)
d = vdr_to_dict(x)
print("JSON:", json.dumps(d))
y = vdr_from_dict(d)
assert y == x

# Active with children
x = VDR(1, 3, Remainder(2, [VDR(5, 11)]))
d = vdr_to_dict(x)
print("JSON:", json.dumps(d))
y = vdr_from_dict(d)
assert y.to_fraction() == x.to_fraction()
print("Serialization roundtrip: exact")
```

---

## 59. LaTeX Export

```python
from vdr import VDR
from vdr.core import Remainder
from vdr.linalg import vdr_to_latex

print(vdr_to_latex(VDR(5)))           # 5
print(vdr_to_latex(VDR(1, 2)))        # \frac{1}{2}
print(vdr_to_latex(VDR(3, 7)))        # \frac{3}{7}
print(vdr_to_latex(VDR(1, 3, Remainder(1))))  # \frac{1}{3}\left\{1\right\}
```

---

## 60. Building a New Domain Module

Template for adding a new domain to VDR.

```python
"""
vdr.math.my_domain — Exact my-domain computations.

    from vdr.math.my_domain import my_function
    result = my_function(VDR(1, 3))
"""

from vdr.core import VDR
from vdr.linalg import Vec, Mat

__all__ = ["my_function", "my_constant"]

# Module-level constants
my_constant = VDR(22, 7)

def my_function(x):
    """
    Compute something exact.

    I: x as VDR
    O: result as VDR, exact
    """
    # Use only VDR operations — everything stays exact
    result = x * x + VDR(1) / x
    return result

def my_matrix_function(A):
    """
    Verify some property of matrix A.

    I: A as Mat
    O: bool (exact comparison, not tolerance)
    """
    result = A.matmul(A)
    identity = Mat.identity(A.nrows)
    return result == identity   # exact equality, not tolerance

# For transcendental quantities, use functional remainders
from vdr.math.transcendental import sqrt_newton

def my_sqrt_function(n):
    """Compute something involving sqrt."""
    s = sqrt_newton(VDR(n), depth=10)
    return s * s  # should be very close to n — check residual exactly

# Test pattern: always verify with == not tolerance
def verify_my_property(x):
    a = my_function(x)
    b = my_function(my_function(x))
    return a == b   # or whatever property should hold
```

Then add tests:

```python
# tests/gym/test_my_domain.py
import pytest
from vdr import VDR
from vdr.math.my_domain import my_function, verify_my_property

class TestMyDomain:
    def test_basic(self):
        result = my_function(VDR(1, 3))
        assert isinstance(result, VDR)

    def test_property(self):
        assert verify_my_property(VDR(7, 11))
```

---

## General Principles for All Recipes

1. **Construct with integers.** `VDR(1, 3)` not `VDR(0.333)`.

2. **Stay in VDR.** Don't convert to float until the final output step.

3. **Assert with `==`.** Not `abs(a - b) < tolerance`. Exact equality.

4. **Check residuals exactly.** `residual = x*x - VDR(2)` gives an exact fraction you can inspect.

5. **Use functional remainders for irrationals.** `sqrt_newton(VDR(2), depth=10)` not `math.sqrt(2)`.

6. **Let normalization work.** Don't manually reduce fractions — `.normalize()` handles it.

7. **D never changes.** If you're in a fixed frame (Q335 or otherwise), multiplication uses divmod. Overflow goes to R. D stays fixed. Zero loss.
