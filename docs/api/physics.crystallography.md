<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/crystallography.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `physics.crystallography`
vdr.physics.crystallography — Exact crystallographic computations. 

 from vdr.physics.crystallography import point_group_matrix, verify_group_closure 

 matrices = [point_group_matrix(op) for op in ["E", "C4z", "C2z", "C4z_inv"]]  assert verify_group_closure(matrices)  # exact structural equality 

Point group operations as 3x3 integer matrices. Group closure verified by exact comparison, not tolerance. Structure factors with exact complex exponentials. 

**Global Variables**
---------------
- **CUBIC_GENERATORS**
- **HEXAGONAL_GENERATORS**

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/crystallography.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `point_group_matrix`

```python
point_group_matrix(operation)
```

Get 3x3 rotation/reflection matrix for named point group operation. 

I: operation name string (e.g. "C4z", "sigma_h", "i") O: 3x3 Mat with exact integer entries 

Cubic operations have entries from {-1, 0, 1}. Hexagonal operations (C3z, C6z) would need sqrt(3)/2. 

 M = point_group_matrix("C4z")  # [[0, -1, 0], [1, 0, 0], [0, 0, 1]] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/crystallography.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `verify_group_closure`

```python
verify_group_closure(matrices)
```

Verify that a set of matrices is closed under multiplication. 

For every pair (A, B), the product A*B must equal some matrix in the set. Comparison by exact structural equality, not tolerance. 

I: list of Mat O: bool, True if closed 

Float: "within tolerance." VDR: structural equality. 

 matrices = [point_group_matrix(op) for op in ["E", "C2z", "C4z", "C4z_inv"]]  verify_group_closure(matrices)  # True 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/crystallography.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `group_multiplication_table`

```python
group_multiplication_table(matrices, names=None)
```

Build group multiplication table. 

I: list of Mat, optional list of name strings O: list of lists of indices (or names) showing product[i][j] 

 matrices = [point_group_matrix(op) for op in ops]  table = group_multiplication_table(matrices, ops) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/crystallography.py#L164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `structure_factor`

```python
structure_factor(atoms, hkl, depth=16)
```

Structure factor F(hkl) = sum f_j * exp(2*pi*i*(h*xj + k*yj + l*zj)). 

I: list of (f, x, y, z) tuples where f is scattering factor (VDR)  and x, y, z are fractional coordinates (VDR),  hkl as (h, k, l) tuple of integers,  trig series depth O: (real, imag) as VDR tuple 

Complex exponentials of rational arguments computed via exact sin/cos series. |F|^2 exact. 

 atoms = [(VDR(1), VDR(0), VDR(0), VDR(0)),  (VDR(1), VDR(1,2), VDR(1,2), VDR(0))]  Fr, Fi = structure_factor(atoms, (1, 1, 0)) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
