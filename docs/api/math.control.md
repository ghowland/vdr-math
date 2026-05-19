<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/control.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `math.control`
vdr.math.control — Exact control theory computations. 

 from vdr.math.control import state_evolve, is_controllable, cayley_hamilton_verify 

 trajectory = state_evolve(A, B, x0, inputs)  # zero drift  ctrl = is_controllable(A, B)                  # exact rank  zero_mat = cayley_hamilton_verify(A)           # exact zero matrix 

State-space evolution, controllability, observability, transfer functions. All exact VDR arithmetic. Cayley-Hamilton verified as structural equality. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/control.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mat_pow`

```python
mat_pow(M, n)
```

Exact matrix power via repeated squaring. 

I: square Mat M, non-negative integer n O: M^n as Mat, exact 

 mat_pow(Mat.from_ints([[1,1],[1,0]]), 10)  # Fibonacci matrix 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/control.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `state_evolve`

```python
state_evolve(A, B, x0, inputs)
```

Discrete-time state-space evolution. 

x[n+1] = A * x[n] + B * u[n] 

I: system matrix A (Mat), input matrix B (Mat), initial state x0 (Vec),  input sequence (list of Vec) O: list of state Vec [x0, x1, x2, ...], all exact, zero drift 

 A = Mat.from_fracs([[(9,10), (1,10)], [(-1,10), (9,10)]])  B = Mat.from_ints([[1, 0], [0, 1]])  x0 = Vec.from_ints([1, 0])  inputs = [Vec.from_ints([0, 0])] * 100  trajectory = state_evolve(A, B, x0, inputs) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/control.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `controllability_matrix`

```python
controllability_matrix(A, B)
```

Controllability matrix: [B, AB, A^2B, ..., A^(n-1)B]. 

I: A (n x n Mat), B (n x m Mat) O: (n x n*m) Mat 

 C = controllability_matrix(A, B)  is_controllable = (C.rank() == A.nrows) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/control.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `observability_matrix`

```python
observability_matrix(A, C)
```

Observability matrix: [C; CA; CA^2; ...; CA^(n-1)]. 

I: A (n x n Mat), C (p x n Mat) O: (p*n x n) Mat 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/control.py#L145"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_controllable`

```python
is_controllable(A, B)
```

Test controllability: controllability matrix has full rank. 

I: A (n x n Mat), B (n x m Mat) O: bool, exact (rank via Gaussian) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/control.py#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_observable`

```python
is_observable(A, C)
```

Test observability: observability matrix has full rank. 

I: A (n x n Mat), C (p x n Mat) O: bool, exact 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/control.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `char_poly`

```python
char_poly(A)
```

Characteristic polynomial of a square matrix. 

For 2x2: lambda^2 - trace*lambda + det For general n: uses Faddeev-LeVerrier algorithm. 

I: square Mat A O: coefficient list [c0, c1, ..., cn] (constant first, leading = 1)  where p(lambda) = c0 + c1*lambda + ... + cn*lambda^n 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/control.py#L205"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `cayley_hamilton_verify`

```python
cayley_hamilton_verify(A)
```

Verify Cayley-Hamilton theorem: p(A) = 0 where p is characteristic polynomial. 

I: square Mat A O: p(A) as Mat — should be exact zero matrix 

 result = cayley_hamilton_verify(A)  assert result == Mat.zero(A.nrows, A.ncols) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/control.py#L232"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `transfer_function_eval`

```python
transfer_function_eval(num_coeffs, den_coeffs, s)
```

Evaluate transfer function H(s) = N(s) / D(s). 

I: numerator polynomial coefficients, denominator polynomial coefficients,  complex frequency s (VDR for real, or tuple (real, imag) for complex) O: H(s) as VDR (real) or tuple (real, imag) 

 # H(s) = 1 / (s^2 + 3s + 2)  transfer_function_eval([VDR(1)], [VDR(2), VDR(3), VDR(1)], VDR(1)) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/control.py#L290"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `controllability_gramian`

```python
controllability_gramian(A, B, steps)
```

Discrete-time controllability Gramian. 

W = sum_{k=0}^{steps-1} A^k * B * B^T * (A^T)^k 

I: A (n x n), B (n x m), number of steps O: Gramian Mat (n x n), symmetric, exact 

 W = controllability_gramian(A, B, 10)  # det(W) > 0 means controllable within horizon 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
