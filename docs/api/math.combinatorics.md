<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/combinatorics.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `math.combinatorics`
vdr.math.combinatorics — Exact combinatorial functions. 

 from vdr.math.combinatorics import binom, bell, multinomial 

 c = binom(20, 10)        # VDR(184756)  b = bell(5)              # VDR(52)  m = multinomial(10, [3,3,4])  # VDR(4200) 

All results exact VDR integers or rationals. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/combinatorics.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `factorial`

```python
factorial(n)
```

n! as VDR. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/combinatorics.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `falling_factorial`

```python
falling_factorial(n, k)
```

Falling factorial: n * (n-1) * ... * (n-k+1). 

I: VDR n, int k O: exact VDR 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/combinatorics.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `binom`

```python
binom(n, k)
```

Binomial coefficient C(n, k). 

I: non-negative integers n, k O: C(n,k) as VDR 

 binom(20, 10) -> VDR(184756) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/combinatorics.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `stirling2`

```python
stirling2(n, k)
```

Stirling number of the second kind S(n, k). Number of ways to partition n elements into k non-empty subsets. 

I: non-negative integers n, k O: S(n,k) as VDR 

Uses the explicit formula:  S(n,k) = (1/k!) * sum_{j=0}^{k} (-1)^(k-j) * C(k,j) * j^n 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/combinatorics.py#L112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bell`

```python
bell(n)
```

Bell number B(n): total number of partitions of n elements. 

I: non-negative integer n O: B(n) as VDR 

 bell(5) -> VDR(52) 

Uses Bell triangle. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/combinatorics.py#L137"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `derangement`

```python
derangement(n)
```

Derangement D(n): number of permutations with no fixed points. 

I: non-negative integer n O: D(n) as VDR 

 derangement(7) -> VDR(1854) 

Uses recurrence: D(n) = (n-1)(D(n-1) + D(n-2)) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/combinatorics.py#L162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `catalan`

```python
catalan(n)
```

Catalan number C_n = C(2n,n) / (n+1). 

I: non-negative integer n O: C_n as VDR 

 catalan(5) -> VDR(42) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/combinatorics.py#L174"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `catalan_gf`

```python
catalan_gf(x, terms)
```

Catalan generating function partial sum:  sum_{n=0}^{terms} C_n * x^n 

I: VDR x, number of terms O: partial sum as VDR 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/combinatorics.py#L190"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `multinomial`

```python
multinomial(n, ks)
```

Multinomial coefficient: n! / (k1! * k2! * ... * km!) 

I: total n, list of group sizes [k1, k2, ..., km] O: multinomial coefficient as VDR 

 multinomial(10, [3, 3, 4]) -> VDR(4200) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
