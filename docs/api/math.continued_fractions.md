<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/continued_fractions.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `math.continued_fractions`
vdr.math.continued_fractions — Exact continued fraction operations. 

 from vdr.math.continued_fractions import to_cf, from_cf, sqrt_cf_period 

 cf = to_cf(355, 113)              # [3, 7, 16]  val = from_cf([3, 7, 16])         # VDR(355, 113)  period = sqrt_cf_period(2)        # (1, [2]) 

All operations exact. Convergents are exact VDR rationals. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/continued_fractions.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_cf`

```python
to_cf(v, d)
```

Convert fraction v/d to continued fraction coefficients. 

I: numerator v, denominator d (integers) O: list of CF coefficients [a0; a1, a2, ...] 

 to_cf(355, 113) -> [3, 7, 16] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/continued_fractions.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `from_cf`

```python
from_cf(coeffs)
```

Convert continued fraction coefficients to VDR rational. 

I: list of CF coefficients [a0, a1, ...] O: exact VDR rational 

 from_cf([3, 7, 16]) -> VDR(355, 113) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/continued_fractions.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `convergents_from_cf`

```python
convergents_from_cf(coeffs)
```

Compute all convergents from CF coefficients. 

I: list of CF coefficients O: list of VDR convergent fractions 

 convergents_from_cf([3, 7, 16]) -> [VDR(3,1), VDR(22,7), VDR(355,113)] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/continued_fractions.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sb_path`

```python
sb_path(p, q)
```

Stern-Brocot path for fraction p/q. 

I: numerator p, denominator q (positive, coprime) O: path string of L and R moves 

 sb_path(3, 5) -> "LLRL" 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/continued_fractions.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sqrt_cf_period`

```python
sqrt_cf_period(n)
```

Continued fraction expansion of sqrt(n) for non-square n. 

I: positive non-square integer O: (integer_part, periodic_block) where sqrt(n) = [a0; a1, a2, ..., a_k, ...]  and [a1, ..., a_k] is the repeating period. 

 sqrt_cf_period(2) -> (1, [2])  sqrt_cf_period(3) -> (1, [1, 2])  sqrt_cf_period(7) -> (2, [1, 1, 1, 4]) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/continued_fractions.py#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `best_rational`

```python
best_rational(x, max_denom)
```

Best rational approximation to x with denominator <= max_denom. 

I: target VDR value, maximum denominator O: best approximation as VDR 

Uses continued fraction convergents. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
