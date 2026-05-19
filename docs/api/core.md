<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `core`
VDR — Value, Denominator, Remainder. Exact finite discrete representation in irreducible triple form. 

A VDR object is [V, D, R] where:  V in Z           — value slot (settled numerator)  D in Z \ {0}     — denominator slot (frame)  R in Remainder    — remainder slot (exact unresolved structure) 

Remainder is first-class. Never error. Never residue. 

R is either:  an integer (atomic remainder), or  an integer base plus a finite list of child VDR objects (composite remainder) 

Recursion exists only in R. Every valid object is a finite tree. No limits, no approximation, no infinity. 

Closed object:  [V, D, 0]  — projects to V/D exactly Active object:  [V, D, R]  with R != 0 — carries exact remainder state 

**Global Variables**
---------------
- **ZERO**
- **ONE**
- **NEG_ONE**


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `VDRError`
Base error for all VDR operations. 





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ZeroDenominatorError`








---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `InvalidStructureError`








---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RebaseError`








---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ArithmeticFailure`








---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Remainder`
The remainder slot of a VDR triple. 

Atomic form:   Remainder(base=r, children=[]) Composite form: Remainder(base=r, children=[X1, X2, ...]) 

R = r + X1 + X2 + ... + Xn where r is an integer and each Xi is a valid VDR object. 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L137"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(base=0, children=None)
```






---

#### <kbd>property</kbd> is_atomic





---

#### <kbd>property</kbd> is_functional





---

#### <kbd>property</kbd> is_globally_zero

True when base is 0 and all children are globally closed with value 0. 

---

#### <kbd>property</kbd> is_zero

True when base is 0 and no children. 



---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `combine`

```python
combine(other, sign=1)
```

Same-frame remainder combination. sign=1  for addition:    R1 + R2 sign=-1 for subtraction: R1 - R2 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L234"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `legacy_value`

```python
legacy_value()
```

Additive flattening for scalar comparison. legacy(r) = r legacy(r + X1+...+Xn) = r + Fraction(X1) + ... + Fraction(Xn) 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L216"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `lift`

```python
lift(k)
```

Transport remainder into a scaled denominator frame. 

lift(r, k) = k * r lift(r + X1+...+Xn, k) = k*r + lift(X1,k) + ... + lift(Xn,k) 

Child VDR lift: lift([V, D, R], k) = [k*V, D, lift(R, k)] 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L193"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `negate`

```python
negate()
```


-(r + X1 + ... + Xn) = -r + (-X1) + ... + (-Xn) 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L247"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `normalize`

```python
normalize()
```

Normalize remainder structure: 1. Normalize all children recursively 2. Remove zero-value closed children 3. Merge closed children sharing a denominator 4. Absorb D=1 closed children into base 5. Sort children canonically 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L179"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `structural_eq`

```python
structural_eq(other)
```






---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L303"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `VDR`
Exact finite discrete triple: [V, D, R] 

Construction:  VDR(3)              -> [3, 1, 0]   integer  VDR(1, 2)           -> [1, 2, 0]   rational 1/2  VDR(1, 3, Remainder(1))  -> [1, 3, 1]   active  VDR.from_fraction(Fraction(5, 6))  -> [5, 6, 0] 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L316"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(v, d=1, r=None)
```






---

#### <kbd>property</kbd> is_active





---

#### <kbd>property</kbd> is_closed

Top-level remainder is zero. 

---

#### <kbd>property</kbd> is_globally_closed

All remainders in the entire tree are zero. 



---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L684"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bracket`

```python
bracket()
```

Native bracket notation. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L479"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compact`

```python
compact()
```

Absorb atomic remainder into V, keep D. VDR(0, 7, 7).compact() -> VDR(1, 7, 0) 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L666"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `den_complexity`

```python
den_complexity()
```

Denominator complexity: (distinct_count, magnitude_sum, node_count). 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L654"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `depth`

```python
depth()
```

Recursive depth of the VDR tree. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L344"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `from_fraction`

```python
from_fraction(frac)
```

Exact construction from fractions.Fraction. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L349"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `from_int`

```python
from_int(n)
```

Exact construction from integer. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L649"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `negate`

```python
negate()
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L410"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `normalize`

```python
normalize()
```

Produce canonical form: 1. Normalize remainder recursively 2. Sign convention: D > 0 3. Check if remainder is value-equivalent to zero -> collapse to closed 4. GCD reduce (V, D) for closed nodes 5. For active nodes, reduce if remainder is cleanly divisible 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L607"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `rebase`

```python
rebase(target_d)
```

Change top-level denominator to target_d, preserving exact value. 

Closed rebase: succeeds when V*B/D is integer Active rebase: [Q, B, [S,D,0] + lift(R, B)]  where V*B = Q*D + S 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L662"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `size`

```python
size()
```

Total node count. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L372"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `structural_eq`

```python
structural_eq(other)
```

X =s Y iff every slot matches exactly, recursively. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L471"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_float`

```python
to_float()
```

Lossy export to float. Loss belongs to float format. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L460"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_fraction`

```python
to_fraction()
```

Exact projection to fractions.Fraction. 

Closed: V/D Active: (V + legacy(R)) / D 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/core.py#L382"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `value_eq`

```python
value_eq(other)
```

X =n Y iff norm(X) =s norm(Y). 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
