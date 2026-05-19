<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/exp.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `ml.exp`
vdr.ml.exp — Exact exponential function for ML pipelines. 

 from vdr.ml.exp import exp_series, exp_range_reduced, exp_neg 

 e_val = exp_series(VDR(1), depth=20)   # e to ~18 digits  safe = exp_neg(VDR(5), depth=20)       # exp(-5), always positive 

Taylor series with exact VDR rational arithmetic. Range reduction for large arguments. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/exp.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `exp_series`

```python
exp_series(x, depth=16)
```

Taylor series for exp(x) = sum_{k=0}^{depth} x^k / k! 

I: x (VDR or int), depth (number of terms beyond constant) O: exact rational partial sum 

Super-geometric convergence: ~35 terms for 100 digits at x=1/2. 

 exp_series(VDR(1), 20)        # e ≈ 2.71828...  exp_series(VDR(1, 2), 16)     # exp(0.5)  exp_series(VDR(-1), 20)       # exp(-1) ≈ 0.36788... 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/exp.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `exp_range_reduced`

```python
exp_range_reduced(x, depth=16)
```

Range-reduced exponential for large |x|. 

Uses exp(x) = exp(x - n*ln2) * 2^n where n = round(x / ln2). The reduced argument is in [-ln2/2, ln2/2] for fast convergence. 

For exact work: n is chosen so that the reduced argument is small, then exp of the small part is computed by Taylor, and the 2^n factor is an exact power of two (just changes D in VDR frame). 

I: x (VDR), depth O: exact rational approximation 

 exp_range_reduced(VDR(10), 20) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/exp.py#L95"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `exp_neg`

```python
exp_neg(x, depth=16)
```

Compute exp(-|x|), ensuring positive result. 

Convenience for ML contexts where we need exp of negative values (softmax, sigmoid, etc.). 

I: x (VDR, non-negative), depth O: exp(-x) as VDR, exact rational, positive 

 exp_neg(VDR(5), 20) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
