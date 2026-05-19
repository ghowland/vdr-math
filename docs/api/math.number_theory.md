<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/number_theory.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `math.number_theory`
vdr.math.number_theory — Exact integer and rational number theory. 

 from vdr.math.number_theory import harmonic, euler_totient, farey 

 h10 = harmonic(10)           # VDR(7381, 2520) exact  phi = euler_totient(100)     # 40  f5 = farey(5)                # all fractions p/q with q <= 5 

All operations exact. No float. No truncation. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/number_theory.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `vdr_gcd`

```python
vdr_gcd(a, b)
```

GCD of two integer VDR values. 

I: two closed VDR with D=1 O: GCD as VDR 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/number_theory.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `vdr_lcm`

```python
vdr_lcm(a, b)
```

LCM of two integer VDR values. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/number_theory.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `factorial`

```python
factorial(n)
```

n! as VDR. Exact. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/number_theory.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `egyptian_fractions`

```python
egyptian_fractions(v, d, max_terms=20)
```

Decompose v/d into sum of unit fractions (greedy algorithm). 

I: numerator v, denominator d, max terms O: list of VDR unit fractions [1/a1, 1/a2, ...] summing to v/d 

 egyptian_fractions(3, 7) -> [VDR(1,3), VDR(1,11), VDR(1,231)] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/number_theory.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `stern_brocot`

```python
stern_brocot(depth)
```

Generate Stern-Brocot tree fractions at given depth. 

I: tree depth (0 = just 1/1) O: sorted list of VDR fractions 

 stern_brocot(2) -> [VDR(1,3), VDR(1,2), VDR(2,3), VDR(1,1), ...] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/number_theory.py#L112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `farey`

```python
farey(n)
```

Farey sequence F_n: all fractions p/q with 0 <= p/q <= 1, q <= n, in ascending order. 

I: order n O: list of VDR fractions 

 farey(5) -> [VDR(0,1), VDR(1,5), VDR(1,4), ..., VDR(1,1)] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/number_theory.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `euler_totient`

```python
euler_totient(n)
```

Euler's totient function phi(n). 

I: positive integer O: int (count of integers 1..n coprime to n) 

 euler_totient(100) -> 40 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/number_theory.py#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `harmonic`

```python
harmonic(n)
```

Harmonic number H_n = 1 + 1/2 + 1/3 + ... + 1/n. 

I: positive integer n O: exact VDR rational 

 harmonic(10) -> VDR(7381, 2520) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/number_theory.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `vdr_pow`

```python
vdr_pow(base, exp)
```

Exact integer power: base^exp. 

I: VDR base, non-negative int exponent O: base^exp as VDR 

Uses repeated squaring for efficiency. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/number_theory.py#L197"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `vdr_mod`

```python
vdr_mod(a, m)
```

Modular reduction: a mod m. 

I: integer VDR a (D=1), int modulus m O: VDR(a.v % m) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/number_theory.py#L207"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `convergents`

```python
convergents(cf_coeffs)
```

Compute convergents from continued fraction coefficients. 

I: list of CF coefficients [a0, a1, a2, ...] O: list of VDR convergents [p0/q0, p1/q1, ...] 

Uses the recurrence:  p_n = a_n * p_{n-1} + p_{n-2}  q_n = a_n * q_{n-1} + q_{n-2} 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/number_theory.py#L235"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `e_cf`

```python
e_cf(n)
```

First n continued fraction coefficients of e. 

e = [2; 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8, ...] Pattern after a0=2: repeating (1, 2k, 1) for k=1,2,3,... 

I: number of coefficients O: list of ints 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
