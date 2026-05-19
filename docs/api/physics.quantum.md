<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/quantum.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `physics.quantum`
vdr.physics.quantum — Exact quantum mechanics computations. 

 from vdr.physics.quantum import SIGMA_X, SIGMA_Z, spin_rotation  from vdr.physics.quantum import measurement_probabilities, verify_unitarity 

 U = spin_rotation("z", VDR(1, 4))  # pi/4 rotation about z  probs = measurement_probabilities(state)  # sums to exactly 1  verify_unitarity(U)  # True, exact 

Pauli algebra verified as structural identity. Spin rotation periodicity: U(pi/2 about z) applied 4 times = I exactly. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/quantum.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `pauli_multiply`

```python
pauli_multiply(a, b)
```

Multiply two Pauli matrices by label. 

I: Pauli labels a, b from {"I", "x", "y", "z"} O: ((real_phase, imag_phase), result_label) 

 pauli_multiply("x", "y") -> ((0, 1), "z")   # sigma_x * sigma_y = i * sigma_z  pauli_multiply("x", "x") -> ((1, 0), "I")    # sigma_x^2 = I 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/quantum.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `spin_rotation_components`

```python
spin_rotation_components(axis, angle_over_pi, depth=16)
```

Compute cos(theta/2) and sin(theta/2) for spin rotation. 

theta = angle_over_pi * pi 

I: axis ("x", "y", "z"), angle as rational multiple of pi (VDR), depth O: (cos_half, sin_half) as VDR tuple 

 spin_rotation_components("z", VDR(1, 2))  # theta = pi/2, cos(pi/4) and sin(pi/4) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/quantum.py#L114"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `spin_rotation`

```python
spin_rotation(axis, angle_over_pi, depth=16)
```

Spin-1/2 rotation matrix (real part only, for real-axis rotations). 

U = cos(theta/2)*I - i*sin(theta/2)*sigma_axis 

For axis = "z":  U = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]  (where the -i*sigma_z produces off-diagonal terms) 

Actually for z-axis the full unitary is:  U = [[exp(-i*t/2), 0], [0, exp(i*t/2)]]  = [[cos(t/2) - i*sin(t/2), 0], [0, cos(t/2) + i*sin(t/2)]] 

For pure real representation, we use the rotation matrix form:  R_z(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]] 

I: axis ("x", "y", "z"), angle/pi as VDR, trig depth O: 2x2 Mat (real rotation part) 

U applied 4 times at angle pi/2 returns to I exactly. Float gives I +/- ~1e-15. 

 U = spin_rotation("z", VDR(1, 2))  # pi/2 rotation 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/quantum.py#L151"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `measurement_probabilities`

```python
measurement_probabilities(state)
```

Born rule measurement probabilities. 

P(i) = |a_i|^2 for state vector |psi> = sum a_i |i>. 

I: state as Vec of VDR amplitudes (real) O: list of VDR probabilities, sums to exactly 1 

For complex amplitudes, pass squared magnitudes directly. 

 state = Vec([VDR(3, 5), VDR(4, 5)])  probs = measurement_probabilities(state)  # [VDR(9, 25), VDR(16, 25)] -> sum = VDR(1) exactly 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/quantum.py#L172"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `density_matrix`

```python
density_matrix(state)
```

Density matrix rho = |psi><psi| (outer product). 

I: state as Vec O: Mat (n x n) 

 state = Vec([VDR(1, 2), VDR(1, 2)])  rho = density_matrix(state)  # trace(rho) == 1 exactly 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/quantum.py#L193"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `verify_unitarity`

```python
verify_unitarity(U)
```

Verify U^T * U == I (for real unitary / orthogonal matrices). 

I: Mat U O: bool, True if U^T * U is exactly the identity matrix 

For complex unitarity (U†U = I), pass the appropriate conjugate-transpose product. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/quantum.py#L208"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `commutator`

```python
commutator(A, B)
```

Matrix commutator [A, B] = AB - BA. 

I: two square Mat of same size O: commutator Mat, exact 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/quantum.py#L218"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `anticommutator`

```python
anticommutator(A, B)
```

Matrix anticommutator {A, B} = AB + BA. 

I: two square Mat of same size O: anticommutator Mat, exact 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/quantum.py#L228"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `expectation_value`

```python
expectation_value(operator, state)
```

Expectation value <psi|O|psi>. 

I: operator as Mat, state as Vec O: VDR, exact 

 ev = expectation_value(SIGMA_Z, Vec([VDR(3,5), VDR(4,5)])) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
