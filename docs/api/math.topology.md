<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/topology.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `math.topology`
vdr.math.topology — Exact algebraic topology computations. 

 from vdr.math.topology import betti_numbers, euler_characteristic 

 betti = betti_numbers([d1, d0])   # exact via rank  chi = euler_characteristic(betti) # alternating sum 

All boundary operators, ranks, and Betti numbers computed with exact VDR arithmetic. d∘d = 0 verified as exact zero matrix. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/topology.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `boundary_matrix`

```python
boundary_matrix(simplices_k, simplices_k_minus_1)
```

Construct boundary operator matrix d_k. 

d_k maps k-chains to (k-1)-chains. Entry [i, j] = coefficient of (k-1)-simplex i in the boundary of k-simplex j. 

I: list of k-simplices (as sorted tuples), list of (k-1)-simplices O: Mat with entries in {-1, 0, 1} 

 # Triangle [0,1,2]: edges [0,1], [0,2], [1,2]  d1 = boundary_matrix(  [(0,1,2)],          # 2-simplices  [(0,1), (0,2), (1,2)]  # 1-simplices  ) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/topology.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `verify_d_squared_zero`

```python
verify_d_squared_zero(d_high, d_low)
```

Verify d∘d = 0 (fundamental property of boundary operators). 

I: d_{k+1} (Mat) and d_k (Mat) O: bool, True if d_low * d_high is the exact zero matrix 

 verify_d_squared_zero(d2, d1) -> True 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/topology.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `betti_numbers`

```python
betti_numbers(boundary_matrices)
```

Compute Betti numbers from a sequence of boundary matrices. 

beta_k = dim(ker(d_k)) - dim(im(d_{k+1}))  = (n_k - rank(d_k)) - rank(d_{k+1}) 

I: list of boundary matrices [d_1, d_2, ...] where d_k maps  k-chains to (k-1)-chains. d_0 is omitted (maps to 0). O: list of Betti numbers [beta_0, beta_1, ...] 

Ranks computed via exact Gaussian elimination. 

 betti = betti_numbers([d1, d2])  # For hollow triangle: [1, 1]  # For filled triangle: [1, 0] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/topology.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `euler_characteristic`

```python
euler_characteristic(betti)
```

Euler characteristic from Betti numbers. 

chi = sum (-1)^k * beta_k 

I: list of Betti numbers O: integer 

 euler_characteristic([1, 0, 1]) -> 2  (sphere) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/topology.py#L181"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `simplicial_complex_boundaries`

```python
simplicial_complex_boundaries(simplices)
```

Build all boundary matrices for a simplicial complex. 

I: dict mapping dimension -> list of simplices (as sorted tuples)  e.g. {0: [(0,),(1,),(2,)], 1: [(0,1),(0,2),(1,2)], 2: [(0,1,2)]} O: list of boundary matrices [d_1, d_2, ...] 

 simplices = {  0: [(0,), (1,), (2,)],  1: [(0,1), (0,2), (1,2)],  2: [(0,1,2)]  }  boundaries = simplicial_complex_boundaries(simplices) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
