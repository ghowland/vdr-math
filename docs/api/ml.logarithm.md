<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/logarithm.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `ml.logarithm`
vdr.ml.logarithm — Exact logarithm for ML pipelines. 

 from vdr.ml.logarithm import log_series, log1p_series 

 ln_val = log_series(VDR(2), depth=20)        # ln(2)  ln1p = log1p_series(VDR(1, 10), depth=20)    # ln(1.1) 

Arctanh-based series for general positive arguments. Direct series for arguments near 1. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/logarithm.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `log1p_series`

```python
log1p_series(x, depth=16)
```

ln(1 + x) via Taylor series: x - x^2/2 + x^3/3 - ... 

Converges for -1 < x <= 1. 

I: x (VDR), depth (number of terms) O: exact rational partial sum 

 log1p_series(VDR(1, 10), 20)  # ln(1.1) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/logarithm.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `log_series`

```python
log_series(x, depth=16)
```

Natural logarithm via arctanh series. 

ln(x) = 2 * arctanh((x-1)/(x+1)) = 2 * sum u^(2k+1)/(2k+1) where u = (x-1)/(x+1). 

Converges for all x > 0. Fastest near x = 1. 

I: x (VDR, positive), depth O: exact rational partial sum of ln(x) 

 log_series(VDR(2), 20)     # ln(2) ≈ 0.693...  log_series(VDR(10), 30)    # ln(10) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/logarithm.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `log_ratio_near_one`

```python
log_ratio_near_one(num, den, depth=16)
```

ln(num/den) when num ≈ den, avoiding catastrophic cancellation. 

Uses ln(1 + (num-den)/den) = log1p((num-den)/den). Exact VDR subtraction — no cancellation. 

I: numerator num (VDR), denominator den (VDR), depth O: ln(num/den) as VDR 

 log_ratio_near_one(VDR(101, 100), VDR(1), 20)  # ln(1.01) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
