<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/symbolic.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `math.symbolic`
vdr.math.symbolic — Exact symbolic algebra over VDR. 

 from vdr.math.symbolic import partial_fractions_simple, power_sum 

 pf = partial_fractions_simple([VDR(1)], [VDR(1), VDR(2)])  # 1/((x-1)(x-2)) = -1/(x-1) + 1/(x-2) 

 s = power_sum(3, 100)  # 1^3 + 2^3 + ... + 100^3 exact 

Rational function arithmetic, partial fractions, power sums. All exact VDR arithmetic. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/symbolic.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `partial_fractions_simple`

```python
partial_fractions_simple(numerator, roots)
```

Partial fraction decomposition for simple (non-repeated) linear roots. 

Decomposes N(x) / ((x-r1)(x-r2)...(x-rn)) into  c1/(x-r1) + c2/(x-r2) + ... + cn/(x-rn) 

I: numerator polynomial coefficients, list of root VDR values O: list of (coefficient, root) tuples [(c1, r1), (c2, r2), ...] 

Uses the cover-up method: ci = N(ri) / product_{j!=i}(ri - rj) 

 partial_fractions_simple([VDR(1)], [VDR(1), VDR(2)]) 
    -> [(VDR(-1), VDR(1)), (VDR(1), VDR(2))]  # meaning -1/(x-1) + 1/(x-2) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/symbolic.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `ratfun_add`

```python
ratfun_add(pq1, pq2)
```

Add two rational functions. 

Each rational function is (numerator_poly, denominator_poly). 

I: two rational functions as (num, den) tuples of coefficient lists O: sum as (num, den) tuple 

 ratfun_add(([VDR(1)], [VDR(-1), VDR(1)]),  ([VDR(1)], [VDR(-2), VDR(1)]))  # 1/(x-1) + 1/(x-2) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/symbolic.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `ratfun_mul`

```python
ratfun_mul(pq1, pq2)
```

Multiply two rational functions. 

I: two rational functions as (num, den) tuples O: product as (num, den) tuple 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/symbolic.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `ratfun_eval`

```python
ratfun_eval(pq, x)
```

Evaluate rational function at x. 

I: rational function as (num, den), evaluation point x O: N(x)/D(x) as VDR 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/symbolic.py#L116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `power_sum`

```python
power_sum(k, n)
```

Power sum: 1^k + 2^k + ... + n^k. 

I: exponent k (non-negative int), upper limit n (positive int) O: exact VDR value 

 power_sum(2, 100)  -> VDR(338350)  power_sum(3, 100)  -> VDR(25502500) 

Computed directly by summation. For large n, Faulhaber's formula could be used but direct summation is exact and simple. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
