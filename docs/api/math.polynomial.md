<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/polynomial.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `math.polynomial`
vdr.math.polynomial — Exact polynomial arithmetic over VDR. 

Polynomials are represented as coefficient lists, constant term first:  [a0, a1, a2, ..., an] represents a0 + a1*x + a2*x^2 + ... + an*x^n 

 from vdr.math.polynomial import poly_eval, poly_mul, lagrange_interpolate 

 p = [VDR(1), VDR(1), VDR(1)]  # 1 + x + x^2  poly_eval(p, VDR(2))          # VDR(7) 

All operations exact. Zero-testing is structural: coefficient is VDR(0) or not. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/polynomial.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `poly_trim`

```python
poly_trim(p)
```

Remove trailing zero coefficients. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/polynomial.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `poly_degree`

```python
poly_degree(p)
```

Degree of polynomial. Zero polynomial has degree -1. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/polynomial.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `poly_eval`

```python
poly_eval(p, x)
```

Evaluate polynomial at x using Horner's method. 

I: coefficient list [a0, a1, ...], evaluation point x (VDR) O: p(x) as VDR, exact 

 poly_eval([VDR(1), VDR(1), VDR(1)], VDR(2)) -> VDR(7) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/polynomial.py#L84"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `poly_add`

```python
poly_add(p, q)
```

Add two polynomials. 

I: two coefficient lists O: sum coefficient list, trimmed 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/polynomial.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `poly_sub`

```python
poly_sub(p, q)
```

Subtract two polynomials. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/polynomial.py#L111"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `poly_mul`

```python
poly_mul(p, q)
```

Multiply two polynomials. 

I: two coefficient lists O: product coefficient list 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/polynomial.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `poly_scale`

```python
poly_scale(p, s)
```

Multiply polynomial by scalar. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/polynomial.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `poly_neg`

```python
poly_neg(p)
```

Negate polynomial. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/polynomial.py#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `poly_divmod`

```python
poly_divmod(p, q)
```

Polynomial division with remainder. 

I: dividend p, divisor q (coefficient lists) O: (quotient, remainder) as coefficient lists 

Exact zero-testing at each step. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/polynomial.py#L177"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `poly_gcd`

```python
poly_gcd(p, q)
```

Polynomial GCD via Euclidean algorithm. 

I: two polynomials as coefficient lists O: GCD polynomial (monic), exact 

Zero-testing is structural: VDR(0) or not. 

 poly_gcd([VDR(-1),VDR(0),VDR(1)], [VDR(1),VDR(2),VDR(1)]) 
    -> represents (x+1) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/polynomial.py#L205"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rational_roots`

```python
rational_roots(p)
```

Find all rational roots of polynomial via rational root theorem. 

Tests +/-(factors of a0) / (factors of an). 

I: coefficient list O: list of VDR roots 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/polynomial.py#L263"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `lagrange_interpolate`

```python
lagrange_interpolate(points)
```

Lagrange interpolation through given points. 

I: list of (x, y) pairs as (VDR, VDR) tuples O: polynomial coefficient list [a0, a1, ..., an] 

 lagrange_interpolate([(VDR(0),VDR(1)), (VDR(1),VDR(3)), (VDR(2),VDR(7))]) 
    -> [VDR(1), VDR(1), VDR(1)]  # 1 + x + x^2 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/polynomial.py#L306"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `char_poly_2x2`

```python
char_poly_2x2(m)
```

Characteristic polynomial of a 2x2 matrix. 

p(lambda) = lambda^2 - trace*lambda + det 

I: 2x2 Mat O: coefficient list [det, -trace, 1] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/polynomial.py#L324"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `poly_derivative`

```python
poly_derivative(p)
```

Symbolic differentiation of polynomial. 

I: coefficient list [a0, a1, a2, ..., an] O: derivative [a1, 2*a2, 3*a3, ..., n*an] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/polynomial.py#L339"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `poly_integral`

```python
poly_integral(p, c=None)
```

Symbolic antiderivative of polynomial. 

I: coefficient list, optional constant of integration c O: antiderivative [c, a0, a1/2, a2/3, ...] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/polynomial.py#L356"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `definite_integral`

```python
definite_integral(p, a, b)
```

Exact definite integral of polynomial from a to b. 

I: coefficient list, bounds a and b (VDR) O: integral value as VDR, exact 

 definite_integral([VDR(0), VDR(0), VDR(1)], VDR(0), VDR(1)) 
    -> VDR(1, 3)  # integral of x^2 from 0 to 1 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/polynomial.py#L372"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `poly_str`

```python
poly_str(p)
```

Pretty-print polynomial as string. 

 poly_str([VDR(1), VDR(-2), VDR(3)]) -> "1 - 2x + 3x^2" 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
