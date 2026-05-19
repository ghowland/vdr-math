<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/transcendental.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `math.transcendental`
vdr.math.transcendental — Exact transcendental arithmetic. 

Named constants available both as Q335 precomputed values (100-digit precision) and as FnRemainder functions (arbitrary depth). 

 from vdr.math.transcendental import PI, E, SQRT2, ZETA3  from vdr.math.transcendental import PI_FN, sqrt_newton, borwein_zeta 

 PI                        # VDR(2198864..., 2^335) ready to use  resolve(PI_FN, depth=10)  # arbitrary precision via functional remainder  borwein_zeta(5, n=210)    # zeta(5) to 100+ digits 

Q335 basis: 22 constants as [p, 2^335, 0]. Minimal universal exponent: at n=334 Catalan's G fails at position 101. At n=335 all 22 pass. 

**Global Variables**
---------------
- **Q335_DENOM**

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/transcendental.py#L179"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sqrt_newton`

```python
sqrt_newton(n, depth=10, start=None)
```

Newton-Raphson square root of n. 

x_{k+1} = (x_k + n/x_k) / 2 

I: n (VDR or int), depth (iterations), optional start O: exact rational approximation. Depth 10 = >100 correct digits.  Quadratic convergence: digits double per step. 

 sqrt_newton(VDR(2), depth=10)  # >100 digits of sqrt(2) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/transcendental.py#L210"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `exp_series`

```python
exp_series(x, depth=16)
```

Taylor series for exp(x) = sum x^n / n! 

I: x (VDR), depth (number of terms) O: exact rational partial sum 

 exp_series(VDR(1), 20)  # e to ~18 digits 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/transcendental.py#L229"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sin_series`

```python
sin_series(x, depth=16)
```

Taylor series for sin(x) = x - x^3/3! + x^5/5! - ... 

I: x (VDR), depth (number of terms) O: exact rational partial sum 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/transcendental.py#L247"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `cos_series`

```python
cos_series(x, depth=16)
```

Taylor series for cos(x) = 1 - x^2/2! + x^4/4! - ... 

I: x (VDR), depth (number of terms) O: exact rational partial sum 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/transcendental.py#L264"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `ln_series`

```python
ln_series(x, depth=16)
```

Natural logarithm via ln(1+u) = u - u^2/2 + u^3/3 - ... where u = (x-1)/x for faster convergence when x > 1, or u = x-1 for x near 1. 

I: x (VDR, positive), depth O: exact rational partial sum of ln(x) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/transcendental.py#L302"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `arctan_series`

```python
arctan_series(x, depth=16)
```

Taylor series for arctan(x) = x - x^3/3 + x^5/5 - ... 

I: x (VDR, |x| <= 1), depth O: exact rational partial sum 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/transcendental.py#L323"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `arcsin_series`

```python
arcsin_series(x, depth=16)
```

Taylor series for arcsin(x) using central binomial coefficients. 

arcsin(x) = sum_{n=0}^{depth} C(2n,n) * x^(2n+1) / (4^n * (2n+1)) 

I: x (VDR, |x| < 1), depth O: exact rational partial sum 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/transcendental.py#L354"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `pi_machin`

```python
pi_machin(terms=160)
```

Compute pi via Machin's formula:  pi/4 = 4*arctan(1/5) - arctan(1/239) 

Geometric convergence ~1.4 bits/term. 160 terms = ~224 bits > 67 digits. 

I: number of arctan series terms O: exact rational approximation of pi 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/transcendental.py#L373"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `borwein_coefficients`

```python
borwein_coefficients(n)
```

Compute Borwein acceleration coefficients d_k for k=0..n. 

d_k = n * sum_{i=0}^{k} (n+i-1)! * 4^i / ((n-i)! * (2i)!) 

All coefficients are exact integers (rational with denominator 1). 

I: parameter n (typically 210 for 100 digits) O: list of VDR [d_0, d_1, ..., d_n] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/transcendental.py#L407"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `borwein_eta`

```python
borwein_eta(s, n=210)
```

Dirichlet eta function via Borwein acceleration. 

eta(s) = -1/d_n * sum_{k=0}^{n-1} (-1)^k * (d_k - d_n) / (k+1)^s 

Geometric convergence 3^(-n). n=210 gives ~100 digits for any s. 

I: integer s >= 1, acceleration parameter n O: exact VDR rational 

 borwein_eta(2, 210)  # eta(2) = pi^2/12 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/transcendental.py#L436"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `borwein_zeta`

```python
borwein_zeta(s, n=210)
```

Riemann zeta function via Borwein acceleration. 

zeta(s) = eta(s) / (1 - 2^(1-s)) 

I: integer s >= 2, acceleration parameter n O: exact VDR rational, ~100 digits at n=210 

 borwein_zeta(3, 210)  # Apery's constant  borwein_zeta(5, 210)  # zeta(5) to 100+ digits 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/transcendental.py#L461"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `hypergeometric_2f1`

```python
hypergeometric_2f1(a, b, c, z, terms=500)
```

Gauss hypergeometric function 2F1(a, b; c; z). 

2F1 = sum_{n=0}^{terms} (a)_n (b)_n / ((c)_n * n!) * z^n 

where (x)_n = x(x+1)...(x+n-1) is the rising factorial. 

All coefficients rational when a, b, c, z are rational. 

I: parameters a, b, c, z (VDR), number of terms O: exact rational partial sum 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/transcendental.py#L489"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `elliptic_k`

```python
elliptic_k(k_sq, terms=500)
```

Complete elliptic integral of the first kind K(k). 

K(k) = (pi/2) * 2F1(1/2, 1/2; 1; k^2) 

I: k^2 as VDR (must be < 1), number of hypergeometric terms O: exact rational times Q335 pi/2 

Every series coefficient is rational. Product with Q335 pi/2 gives a standard VDR closed object. 

 elliptic_k(VDR(1, 2), 500)  # K(1/sqrt(2)), k^2 = 1/2 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/transcendental.py#L509"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `elliptic_e`

```python
elliptic_e(k_sq, terms=500)
```

Complete elliptic integral of the second kind E(k). 

E(k) = (pi/2) * 2F1(-1/2, 1/2; 1; k^2) 

I: k^2 as VDR (must be < 1), number of terms O: exact rational times Q335 pi/2 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/transcendental.py#L523"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `clausen_2`

```python
clausen_2(x, terms=210)
```

Clausen function Cl_2(x) = -integral_0^x ln|2 sin(t/2)| dt  = sum_{n=1}^{terms} sin(nx) / n^2 

I: x as VDR (rational multiple of pi), terms O: exact rational (requires sin evaluation via series) 

For x = pi/3: Cl_2(pi/3) relates to zeta(2) and L-functions. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/transcendental.py#L540"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `clausen_3`

```python
clausen_3(x, terms=210)
```

Clausen function Cl_3(x) = sum_{n=1}^{terms} cos(nx) / n^3. 

I: x as VDR, terms O: exact rational 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
