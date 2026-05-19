<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/optics.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `physics.optics`
vdr.physics.optics — Exact paraxial optics via ABCD matrices. 

 from vdr.physics.optics import free_space, thin_lens, system_matrix  from vdr.physics.optics import verify_symplecticity, resonator_stability 

 M = system_matrix([free_space(VDR(1)), thin_lens(VDR(2)), free_space(VDR(1))])  verify_symplecticity(M)  # True, det(M) == 1 exactly 

det(M) = 1 (symplecticity) exact after any number of elements. M^1000 via repeated squaring: exact. Float accumulates ~1e-12. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/optics.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `free_space`

```python
free_space(d)
```

ABCD matrix for free-space propagation of distance d. 

[[1, d], [0, 1]] 

I: distance d (VDR) O: 2x2 Mat 

 free_space(VDR(1, 2))  # half-unit propagation 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/optics.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `thin_lens`

```python
thin_lens(f)
```

ABCD matrix for thin lens of focal length f. 

[[1, 0], [-1/f, 1]] 

I: focal length f (VDR, nonzero) O: 2x2 Mat 

 thin_lens(VDR(10))  # f = 10 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/optics.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `flat_mirror`

```python
flat_mirror()
```

ABCD matrix for flat mirror (reflection). 

[[1, 0], [0, 1]]  (identity — flat mirror just reverses direction) 

O: 2x2 identity Mat 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/optics.py#L76"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `curved_mirror`

```python
curved_mirror(R)
```

ABCD matrix for curved mirror with radius of curvature R. 

[[1, 0], [-2/R, 1]] 

Equivalent to thin lens with f = R/2. 

I: radius of curvature R (VDR) O: 2x2 Mat 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/optics.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `thick_lens`

```python
thick_lens(n, R1, R2, d)
```

ABCD matrix for thick lens. 

Composed of: interface(1, n, R1) * free_space(d/n) * interface(n, 1, R2) 

I: refractive index n, front radius R1, back radius R2, thickness d (all VDR) O: 2x2 Mat 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/optics.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `interface`

```python
interface(n1, n2, R)
```

ABCD matrix for refraction at a curved interface. 

[[1, 0], [(n1-n2)/(n2*R), n1/n2]] 

I: refractive indices n1, n2, radius of curvature R (all VDR) O: 2x2 Mat 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/optics.py#L125"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `system_matrix`

```python
system_matrix(elements)
```

System ABCD matrix from sequence of optical elements. 

Product of element matrices, right-to-left (first element is rightmost in the product since light passes through it first). 

I: list of 2x2 Mat [M1, M2, M3, ...] in order light encounters them O: system Mat = Mn * ... * M2 * M1 

 M = system_matrix([free_space(VDR(1)), thin_lens(VDR(2)), free_space(VDR(1))]) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/optics.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `verify_symplecticity`

```python
verify_symplecticity(M)
```

Verify det(M) == 1 exactly (symplecticity / energy conservation). 

I: 2x2 Mat O: bool, True if det is exactly 1 

Float gives 1.0 +/- ~1e-12 after 1000 elements. VDR gives exactly 1. 

 verify_symplecticity(system_matrix(elements))  # True 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/optics.py#L162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `resonator_stability`

```python
resonator_stability(M)
```

Resonator stability criterion. 

A resonator is stable when |(A+D)/2| < 1. 

I: round-trip system Mat (2x2) O: (is_stable, half_trace) where is_stable is bool (exact comparison)  and half_trace is VDR 

Exact rational comparison — no borderline float ambiguity. 

 stable, ht = resonator_stability(roundtrip_matrix) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/optics.py#L185"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `matrix_power`

```python
matrix_power(M, n)
```

Exact matrix power via repeated squaring. 

M^1000 in 10 multiplications, exact. 

I: 2x2 Mat, non-negative integer n O: M^n as Mat 

Float accumulates ~1000 butterfly-equivalent roundings. VDR stays exact. 

 M1000 = matrix_power(system_matrix(elements), 1000)  verify_symplecticity(M1000)  # still True 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/optics.py#L218"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `beam_parameter`

```python
beam_parameter(M, q_in)
```

Transform complex beam parameter through ABCD matrix. 

q_out = (A*q_in + B) / (C*q_in + D) 

For real beam parameters (geometric optics): 

I: 2x2 Mat, input beam parameter q (VDR) O: output beam parameter as VDR 

 q_out = beam_parameter(M, VDR(10)) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
