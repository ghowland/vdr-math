<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `fn`
vdr.fn — Functional remainder for VDR objects. 

Extends the remainder slot to hold a callable that produces VDR structure on demand. This is how VDR handles square roots, trig, exp, log — every depth gives a complete exact rational, not an approximation of a limit. 

A functional remainder is: 
    - a Python callable: f(depth: int) -> VDR 
    - a name string for inspectability 
    - optional metadata dict 

The function is finite. The structure it produces at any depth is finite and exact. Expansion is on demand via resolve(obj, depth). 

 from vdr.fn import FnRemainder, resolve, make_newton_fn  from vdr.core import VDR 

 sqrt2_fn = make_newton_fn("sqrt2", lambda x: (x + VDR(2)/x) / VDR(2))  obj = VDR(0, 1, sqrt2_fn)  val = resolve(obj, depth=10)  # exact rational, >100 correct digits 

Each depth is a complete exact value, not an approximation. The depth parameter controls how many recursive expansion steps are taken. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `vdr_fn`

```python
vdr_fn(name=None)
```

Decorator for creating named VDR remainder functions. 

 @vdr_fn("sqrt2")  def sqrt2(depth):  x = VDR(1)  for _ in range(depth):  x = (x + VDR(2)/x) / VDR(2)  return x 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L241"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `resolve`

```python
resolve(x, depth=1)
```

Resolve a VDR object by expanding any functional remainder. 

If the remainder is functional, expand it at the given depth and construct a concrete VDR result. 

If the remainder is already concrete, return unchanged. 

 resolved = resolve(obj, depth=10) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L270"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_functional`

```python
is_functional(x)
```

Check if a VDR object has a functional remainder. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L275"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `resolve_recursive`

```python
resolve_recursive(x, depth=1)
```

Resolve all functional remainders in a VDR tree, recursively. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L296"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_constant_fn`

```python
make_constant_fn(name, value_func)
```

Functional remainder that always returns the same value. 

 pi_approx = make_constant_fn("pi_22_7", lambda: VDR(22, 7)) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L307"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_series_fn`

```python
make_series_fn(name, term_func, initial=None)
```

Functional remainder from a series. 

term_func(n) returns the nth exact rational term. At depth N, result is sum of terms 0..N plus initial. 

 # Leibniz series for pi/4  def leibniz_term(n):  sign = 1 if n % 2 == 0 else -1  return VDR(sign, 2*n + 1) 

 pi_fn = make_series_fn("leibniz_pi4", leibniz_term) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L333"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_newton_fn`

```python
make_newton_fn(name, step_fn, start=None)
```

Functional remainder from Newton-Raphson iteration. 

step_fn(x) takes current VDR and returns next iterate. Each step is exact rational arithmetic. Quadratic convergence: digits double per step. 

 # sqrt(2): x_{n+1} = (x + 2/x) / 2  sqrt2_fn = make_newton_fn("sqrt2", lambda x: (x + VDR(2)/x) / VDR(2)) 

At depth N, applies step_fn N times starting from start (default VDR(1)). 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L358"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_iterative_fn`

```python
make_iterative_fn(name, step, start)
```

General iterative function: apply step N times to start. 

 fn = make_iterative_fn("collatz", collatz_step, VDR(27)) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L379"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `discrete_derivative`

```python
discrete_derivative(f, h)
```

Discrete derivative operator. 

Given f: VDR -> VDR and step size h (VDR rational), returns Df where:  Df(x) = (f(x + h) - f(x)) / h 

Every evaluation is exact VDR arithmetic. No limits. 

 f = lambda x: x * x  df = discrete_derivative(f, VDR(1, 1000))  print(df(VDR(3)))  # exact 6001/1000 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L401"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `discrete_derivative_nth`

```python
discrete_derivative_nth(f, h, order=1)
```

Nth-order discrete derivative by repeated application. D^n f = D(D^(n-1) f) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L412"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `discrete_integral`

```python
discrete_integral(f, a, b, n)
```

Discrete integral (left Riemann sum). 

Computes the exact sum:  sum f(a + k*h) * h   for k = 0, 1, ..., n-1 

where h = (b - a) / n. 

Every term is exact VDR arithmetic. No limits. Each n gives an exact answer. 

 result = discrete_integral(lambda x: x*x, VDR(0), VDR(1), 100) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L436"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `discrete_integral_trapz`

```python
discrete_integral_trapz(f, a, b, n)
```

Discrete integral (trapezoidal rule). 

Computes:  h/2 * (f(a) + 2*f(a+h) + ... + 2*f(a+(n-1)*h) + f(b)) 

More accurate than left Riemann for the same n. Still exact VDR arithmetic at every step. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L487"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `install`

```python
install()
```

Patch VDR to be aware of functional remainders. 

After this, to_fraction() raises on functional remainders (forcing explicit resolve), and is_closed/is_active work correctly. 

Called automatically by vdr.__init__. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L513"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `uninstall`

```python
uninstall()
```

Restore original VDR behavior. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FnRemainder`
A remainder slot holding a callable instead of concrete structure. 

The callable signature is: f(depth: int) -> VDR 

At any given depth, the function produces an exact finite VDR object. There is no limit process. Each depth is a complete exact answer. 

 fr = FnRemainder(my_func, name="sqrt2")  result = fr.expand(5)   # calls my_func(5), returns VDR 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(func, name=None, meta=None)
```






---

#### <kbd>property</kbd> is_atomic





---

#### <kbd>property</kbd> is_functional





---

#### <kbd>property</kbd> is_globally_zero





---

#### <kbd>property</kbd> is_zero







---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `combine`

```python
combine(other, sign=1)
```

Combining a functional remainder with another remainder. If both functional, compose them. If one is concrete, wrap in a hybrid. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `expand`

```python
expand(depth)
```

Expand the function at the given depth. Returns an exact finite VDR object. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `legacy_value`

```python
legacy_value()
```

Functional remainder has no default scalar projection. Expand first, then project. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L137"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `lift`

```python
lift(k)
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L131"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `negate`

```python
negate()
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L152"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `normalize`

```python
normalize()
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/fn.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `structural_eq`

```python
structural_eq(other)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
