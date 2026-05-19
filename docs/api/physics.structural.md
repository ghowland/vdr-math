<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/structural.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `physics.structural`
vdr.physics.structural — Exact structural mechanics. 

 from vdr.physics.structural import solve_structure, verify_equilibrium 

 u = solve_structure(K, F)  assert verify_equilibrium(K, u, F)  # True, exact 

Direct stiffness method with exact VDR arithmetic. Equilibrium verified by exact equality, not residual tolerance. Ill-conditioned structures solved exactly — condition number irrelevant. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/structural.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `truss_element_stiffness`

```python
truss_element_stiffness(EA, L, angle_cos, angle_sin)
```

2D truss element stiffness matrix in global coordinates (4x4). 

k_local = (EA/L) * [[1, -1], [-1, 1]] Transformed to global by rotation matrix T. 

I: axial stiffness EA (VDR), length L (VDR),  direction cosines cos(theta) and sin(theta) (VDR) O: 4x4 Mat (DOFs: u1, v1, u2, v2) 

 k = truss_element_stiffness(VDR(100), VDR(1), VDR(1), VDR(0))  # horizontal member, EA=100, L=1 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/structural.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `assemble_stiffness`

```python
assemble_stiffness(elements, n_dof)
```

Assemble global stiffness matrix from element contributions. 

I: list of (element_stiffness_Mat, dof_indices) tuples,  total number of DOFs O: global stiffness Mat (n_dof x n_dof) 

Each element contributes to specific DOF positions. 

 elements = [  (k1, [0, 1, 2, 3]),   # element 1 connects DOFs 0,1,2,3  (k2, [2, 3, 4, 5]),   # element 2 connects DOFs 2,3,4,5  ]  K = assemble_stiffness(elements, 6) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/structural.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `apply_boundary_conditions`

```python
apply_boundary_conditions(K, F, fixed_dofs)
```

Apply boundary conditions by zeroing rows/columns of fixed DOFs. 

Uses the penalty method alternative: replace fixed DOF equation with identity row (1 on diagonal, 0 elsewhere, 0 in F). 

I: stiffness Mat K, force Vec F, list of fixed DOF indices O: (K_modified, F_modified) as (Mat, Vec) 

 K_bc, F_bc = apply_boundary_conditions(K, F, [0, 1]) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/structural.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `solve_structure`

```python
solve_structure(K, F)
```

Solve Ku = F for displacements. 

I: global stiffness Mat K, force Vec F O: displacement Vec u, exact 

Uses Gaussian elimination for n >= 5, Cramer for smaller. Condition number irrelevant — exact arithmetic. 

 u = solve_structure(K, F) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/structural.py#L132"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `verify_equilibrium`

```python
verify_equilibrium(K, u, F)
```

Verify K @ u == F exactly. 

I: stiffness Mat, displacement Vec, force Vec O: bool, True if exact structural equality 

Float gives residual ~1e-10. VDR gives True (exact zero residual). 

 assert verify_equilibrium(K, u, F) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/structural.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `element_forces`

```python
element_forces(k_elem, u_elem)
```

Compute element internal forces from element stiffness and displacements. 

f_elem = k_elem @ u_elem 

I: element stiffness Mat, element displacement Vec O: element force Vec, exact 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/structural.py#L159"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `reaction_forces`

```python
reaction_forces(K_full, u, F_applied)
```

Compute reaction forces at supports. 

R = K_full @ u - F_applied 

At free DOFs, R should be zero. At fixed DOFs, R gives the reaction. 

I: full (unreduced) stiffness Mat, displacement Vec, applied force Vec O: reaction force Vec, exact 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
