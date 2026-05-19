<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/sequences.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `math.sequences`
vdr.math.sequences — Exact recursive sequences. 

 from vdr.math.sequences import fibonacci, bernoulli, lucas 

 f30 = fibonacci(30)       # VDR(832040)  b12 = bernoulli(12)       # VDR(-691, 2730)  l10 = lucas(10)           # VDR(123) 

All values exact. Bernoulli numbers cached on demand. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/sequences.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fibonacci`

```python
fibonacci(n)
```

Fibonacci number F(n). 

I: non-negative integer n O: F(n) as VDR 

 fibonacci(30) -> VDR(832040) 

Uses iterative computation. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/sequences.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fibonacci_seq`

```python
fibonacci_seq(n)
```

Fibonacci sequence F(0) through F(n). 

I: non-negative integer n O: list of VDR [F(0), F(1), ..., F(n)] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/sequences.py#L87"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `lucas`

```python
lucas(n)
```

Lucas number L(n). L(0)=2, L(1)=1, L(n)=L(n-1)+L(n-2). 

I: non-negative integer n O: L(n) as VDR 

 lucas(10) -> VDR(123) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/sequences.py#L107"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `lucas_seq`

```python
lucas_seq(n)
```

Lucas sequence L(0) through L(n). 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/sequences.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `catalan_seq`

```python
catalan_seq(n)
```

Catalan numbers C(0) through C(n). 

C(n) = C(2n, n) / (n+1) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/sequences.py#L138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bernoulli`

```python
bernoulli(n)
```

Bernoulli number B(n). 

I: non-negative integer n O: B(n) as exact VDR 

 bernoulli(0) -> VDR(1)  bernoulli(1) -> VDR(-1, 2)  bernoulli(12) -> VDR(-691, 2730) 

Uses the recurrence:  B(0) = 1  sum_{k=0}^{n} C(n+1, k) * B(k) = 0  for n >= 1 

Results cached for repeated access. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/sequences.py#L184"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bernoulli_seq`

```python
bernoulli_seq(n)
```

Bernoulli numbers B(0) through B(n). 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/sequences.py#L193"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `tribonacci`

```python
tribonacci(n)
```

Tribonacci number T(n). T(0)=0, T(1)=0, T(2)=1, T(n)=T(n-1)+T(n-2)+T(n-3). 

I: non-negative integer n O: T(n) as VDR 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/sequences.py#L211"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `tribonacci_seq`

```python
tribonacci_seq(n)
```

Tribonacci sequence T(0) through T(n). 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/sequences.py#L234"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rational_recurrence`

```python
rational_recurrence(coeffs, initial, n)
```

General linear recurrence with VDR coefficients. 

x(k) = c0*x(k-1) + c1*x(k-2) + ... + c_{m-1}*x(k-m) 

I: coeffs = [c0, c1, ..., c_{m-1}], initial = [x(0), x(1), ..., x(m-1)],  n = number of terms to generate O: list of VDR values [x(0), x(1), ..., x(n-1)] 

 # a(n) = 3 - 2^(1-n)  # via recurrence a(n) = (3/2)*a(n-1) - (1/2)*a(n-2)  rational_recurrence(  [VDR(3,2), VDR(-1,2)],  [VDR(1), VDR(2)],  14  ) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
