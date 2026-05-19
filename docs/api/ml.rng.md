<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/rng.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `ml.rng`
vdr.ml.rng — Deterministic exact rational random number generator. 

 from vdr.ml.rng import VDRRandom 

 rng = VDRRandom(seed=42)  val = rng.rand_fraction()    # exact rational in [0, 1)  idx = rng.randbelow(10)      # integer in [0, 10) 

Linear congruential generator with exact integer arithmetic. Deterministic: same seed = same sequence on any platform. 



---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/rng.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `VDRRandom`
Deterministic pseudo-random number generator for VDR. 

Uses linear congruential generator (LCG):  state_{n+1} = (a * state_n + c) mod m 

Constants from Numerical Recipes (period 2^32):  a = 1664525  c = 1013904223  m = 2^32 

All operations exact integer arithmetic. Platform-independent: same seed, same sequence, always. 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/rng.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(seed=1)
```

I: integer seed (determines entire sequence) 




---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/rng.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `next_int`

```python
next_int()
```

Advance state and return raw integer in [0, m). 

O: integer 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/rng.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `permutation`

```python
permutation(n)
```

Random permutation of [0, 1, ..., n-1]. 

I: size n O: list of integers 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/rng.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `rand_fraction`

```python
rand_fraction()
```

Random VDR rational in [0, 1). 

Returns exact rational with denominator 2^32. 

O: VDR in [0, 1) 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/rng.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `rand_vec`

```python
rand_vec(dim, denom=100)
```

Random VDR Vec with entries in [-1, 1) with given denominator. 

I: dimension, denominator for precision O: Vec of exact VDR rationals 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/rng.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `randbelow`

```python
randbelow(n)
```

Random integer in [0, n). 

I: upper bound n (positive integer) O: integer in [0, n) 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/rng.py#L79"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `randint`

```python
randint(lo, hi)
```

Random integer in [lo, hi] inclusive. 

I: lower bound lo, upper bound hi O: integer 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/rng.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `shuffle_in_place`

```python
shuffle_in_place(xs)
```

Fisher-Yates shuffle of a list. 

I: mutable list S: shuffles list in place 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
