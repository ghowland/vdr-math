<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `linalg`
vdr.linalg — Exact rational linear algebra over VDR objects. 

 from vdr.linalg import Vec, Mat 

 v = Vec([VDR(1,2), VDR(1,3), VDR(1,7)])  m = Mat.identity(3)  d = m.det()           # exact  m_inv = m.inv()       # exact 

All operations use exact VDR arithmetic. Zero drift. 

Determinant, inverse, and solve dispatch automatically:  n <= 4: cofactor / adjugate / Cramer (simple, exact)  n >= 5: Gaussian elimination O(n^3) (practical, exact) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L713"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `parse_vdr`

```python
parse_vdr(text)
```

Parse bracket notation into a VDR object. 

 parse_vdr("[1, 2, 0]")       -> VDR(1, 2)  parse_vdr("[1, 3, [1, 6, 0]]") -> VDR(1, 3, Remainder(0, [VDR(1, 6)])) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L828"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `vdr_to_dict`

```python
vdr_to_dict(x)
```

Serialize VDR to a JSON-compatible dict. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L837"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `vdr_from_dict`

```python
vdr_from_dict(d)
```

Deserialize VDR from a JSON-compatible dict. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L858"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `vdr_to_latex`

```python
vdr_to_latex(x)
```

Export VDR to LaTeX notation. 

 [1, 2, 0]          -> \frac{1}{2}  [1, 3, [1, 6, 0]]  -> \frac{1}{3}\left\{\frac{1}{6}\right\} 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LinAlgError`
Linear algebra specific errors. 





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Vec`
Exact VDR vector — an ordered list of VDR objects. 

 v = Vec([VDR(1,2), VDR(1,3)])  w = Vec.from_ints([1, 2, 3])  v + w, v - w, v * VDR(2), v.dot(w) 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(data)
```






---

#### <kbd>property</kbd> dim







---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dot`

```python
dot(other)
```

Exact dot product: v . w = sum(vi * wi) 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `from_fracs`

```python
from_fracs(pairs)
```

Vec.from_fracs([(1,2), (3,4)]) -> Vec([VDR(1,2), VDR(3,4)]) 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L84"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `from_ints`

```python
from_ints(ns)
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `norm_sq`

```python
norm_sq()
```

Squared norm: v . v (no sqrt, stays exact). 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L125"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `scale`

```python
scale(s)
```

Scalar multiplication: s * v 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L168"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_fractions`

```python
to_fractions()
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `zero`

```python
zero(n)
```






---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L176"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Mat`
Exact VDR matrix — list of row Vecs, all same dimension. 

 m = Mat.from_ints([[1,2],[3,4]])  m.det(), m.inv(), m.solve(b), m.rank() 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L186"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(rows)
```






---

#### <kbd>property</kbd> T





---

#### <kbd>property</kbd> is_square





---

#### <kbd>property</kbd> ncols





---

#### <kbd>property</kbd> nrows





---

#### <kbd>property</kbd> shape







---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L252"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `col`

```python
col(j)
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L333"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `det`

```python
det()
```

Exact determinant. Dispatches:  n <= 4: cofactor expansion  n >= 5: Gaussian elimination 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L345"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `det_cofactor`

```python
det_cofactor()
```

Exact determinant by cofactor expansion. O(n!). 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L364"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `det_gauss`

```python
det_gauss()
```

Exact determinant by Gaussian elimination. O(n^3). 

Uses exact VDR division for pivot operations. Tracks sign from row swaps. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L207"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `from_fracs`

```python
from_fracs(data)
```

Mat.from_fracs([[(1,2),(3,4)],[(5,6),(7,8)]]) 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L203"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `from_ints`

```python
from_ints(data)
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `identity`

```python
identity(n)
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L424"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `inv`

```python
inv()
```

Exact matrix inverse. Dispatches:  n <= 4: adjugate method  n >= 5: Gaussian elimination 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L436"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `inv_adjugate`

```python
inv_adjugate()
```

Exact inverse via adjugate: A^-1 = adj(A) / det(A). 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L460"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `inv_gauss`

```python
inv_gauss()
```

Exact inverse via Gaussian elimination with augmented matrix. [A | I] -> [I | A^-1] 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L282"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `matmul`

```python
matmul(other)
```

Exact matrix multiplication: C[i,j] = sum A[i,k]*B[k,j] 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L301"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `matvec`

```python
matvec(v)
```

Matrix-vector product: Ax 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L690"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `pretty`

```python
pretty()
```

Human-readable matrix display. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L604"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `rank`

```python
rank()
```

Exact rank via Gaussian elimination. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L249"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `row`

```python
row(i)
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L630"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `rref`

```python
rref()
```

Reduced row echelon form via Gaussian elimination. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L268"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `scale`

```python
scale(s)
```

Scalar multiplication. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L515"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `solve`

```python
solve(b)
```

Solve Ax = b. Dispatches:  n <= 4: Cramer's rule  n >= 5: Gaussian elimination 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L529"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `solve_cramer`

```python
solve_cramer(b)
```

Solve Ax = b by Cramer's rule. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L554"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `solve_gauss`

```python
solve_gauss(b)
```

Solve Ax = b by Gaussian elimination with back-substitution. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L702"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_fractions`

```python
to_fractions()
```

Export as list of lists of Fraction. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L322"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `trace`

```python
trace()
```

Sum of diagonal elements. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/linalg.py#L221"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `zero`

```python
zero(nrows, ncols)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
