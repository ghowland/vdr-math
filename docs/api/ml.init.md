<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/init.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `ml.init`
vdr.ml.init — Exact rational weight initialization for neural networks. 

 from vdr.ml.init import rational_uniform_mat, xavier_like_mat, zero_bias 

 W = xavier_like_mat(4, 3, denom=100, seed=42)  b = zero_bias(3) 

All initial weights are exact VDR rationals. Deterministic: same seed produces identical weights on any platform. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/init.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rational_uniform_vec`

```python
rational_uniform_vec(dim, denom=100, seed=1, lo=-1, hi=1)
```

Random Vec with entries uniformly distributed in [lo, hi). 

Each entry is an exact rational with the given denominator. 

I: dimension, denominator, seed, range [lo, hi) O: Vec of exact VDR rationals 

 v = rational_uniform_vec(4, denom=100, seed=42) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/init.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rational_uniform_mat`

```python
rational_uniform_mat(nrows, ncols, denom=100, seed=1, lo=-1, hi=1)
```

Random Mat with entries uniformly distributed in [lo, hi). 

I: dimensions, denominator, seed, range O: Mat of exact VDR rationals 

 W = rational_uniform_mat(3, 4, denom=100, seed=42) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/init.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `xavier_like_mat`

```python
xavier_like_mat(nrows, ncols, denom=100, seed=1)
```

Xavier-like initialization with rational bounds. 

Standard Xavier uniform: U[-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))] 

We approximate the bound as a rational: isqrt(6*denom^2 / (fan_in+fan_out)) / denom. This gives a rational approximation of the Xavier bound. 

I: dimensions nrows (fan_out) x ncols (fan_in), denominator, seed O: Mat of exact VDR rationals in approximately Xavier range 

 W = xavier_like_mat(4, 3, denom=100, seed=42) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/init.py#L111"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `zero_bias`

```python
zero_bias(dim)
```

Zero bias vector. 

I: dimension O: Vec of zeros 

 b = zero_bias(4)  # Vec([VDR(0), VDR(0), VDR(0), VDR(0)]) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/init.py#L123"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `constant_vec`

```python
constant_vec(dim, value=None)
```

Constant-value vector. 

I: dimension, value (VDR or int, default 0) O: Vec with all entries equal to value 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/init.py#L137"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `constant_mat`

```python
constant_mat(nrows, ncols, value=None)
```

Constant-value matrix. 

I: dimensions, value (VDR or int, default 0) O: Mat with all entries equal to value 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
