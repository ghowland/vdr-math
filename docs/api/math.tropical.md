<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/tropical.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `math.tropical`
vdr.math.tropical — Exact tropical algebra and lattice computations. 

 from vdr.math.tropical import trop_mat_mul, gram_schmidt_exact, lll_reduce 

 shortest = trop_mat_mul(A, B, n)          # min-plus matrix multiply  ortho, mu = gram_schmidt_exact(basis)     # exact Gram-Schmidt  reduced = lll_reduce(basis)               # exact LLL, no float in threshold 

Tropical: min-plus semiring. Lattice: exact Gram-Schmidt and Lovasz condition. 

**Global Variables**
---------------
- **TROP_INF**

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/tropical.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `trop_add`

```python
trop_add(a, b)
```

Tropical addition: min(a, b). 

None represents +infinity. 

 trop_add(VDR(3), VDR(5)) -> VDR(3)  trop_add(VDR(3), None) -> VDR(3) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/tropical.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `trop_mul`

```python
trop_mul(a, b)
```

Tropical multiplication: a + b (ordinary addition). 

None represents +infinity. inf + anything = inf. 

 trop_mul(VDR(3), VDR(5)) -> VDR(8) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/tropical.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `trop_mat_mul`

```python
trop_mat_mul(A, B, n)
```

Tropical matrix multiplication (min-plus). 

C[i][j] = min_k (A[i][k] + B[k][j]) 

I: A, B as list-of-lists (n x n), entries VDR or None O: product as list-of-lists 

 # 2-hop shortest paths  result = trop_mat_mul(dist, dist, n) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/tropical.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `trop_det`

```python
trop_det(M, n)
```

Tropical determinant: minimum weight perfect matching. 

det_trop(A) = min over permutations sigma of sum A[i][sigma(i)] 

I: matrix as list-of-lists (n x n), entries VDR or None O: tropical determinant as VDR or None 

 trop_det([[VDR(0), VDR(1)], [VDR(1), VDR(0)]], 2) -> VDR(0) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/tropical.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gram_matrix`

```python
gram_matrix(vectors)
```

Gram matrix: G[i][j] = vectors[i] . vectors[j]. 

I: list of Vec O: Mat, exact 

 gram_matrix([Vec.from_ints([1,0]), Vec.from_ints([1,1])]) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/tropical.py#L141"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gram_schmidt_exact`

```python
gram_schmidt_exact(vectors)
```

Exact Gram-Schmidt orthogonalization. 

I: list of Vec (basis vectors) O: (orthogonalized, mu) where: 
   - orthogonalized is list of Vec (Gram-Schmidt vectors b*_i) 
   - mu is list-of-lists of VDR (GSO coefficients mu[i][j] for j < i) 

Cross-dot products are exactly 0. mu[i][j] = b_i . b*_j / (b*_j . b*_j), exact rational. 

 ortho, mu = gram_schmidt_exact([Vec.from_ints([1,1,0]), ...]) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/tropical.py#L181"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `lovasz_condition`

```python
lovasz_condition(b_star, mu, i, delta=None)
```

Check Lovasz condition at index i:  ||b*_i||^2 >= (delta - mu[i][i-1]^2) * ||b*_{i-1}||^2 

I: orthogonalized vectors b_star, mu coefficients, index i,  delta parameter (VDR, default 3/4) O: bool, exact rational comparison 

No float rounding in this comparison. This is where float LLL makes wrong decisions on borderline cases. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/tropical.py#L208"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `size_reduce`

```python
size_reduce(vectors, b_star, mu, i, j)
```

Size reduction step: ensure |mu[i][j]| <= 1/2. 

If |mu[i][j]| > 1/2, subtract round(mu[i][j]) * b_j from b_i. 

I: basis vectors, GSO vectors, mu coefficients, indices i, j O: updated (vectors, mu) — modifies in place and returns 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/tropical.py#L250"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `lll_reduce`

```python
lll_reduce(basis, delta=None)
```

LLL lattice basis reduction with exact VDR arithmetic. 

I: list of Vec (basis vectors), delta parameter (VDR, default 3/4) O: LLL-reduced list of Vec 

All Gram-Schmidt coefficients exact. Lovasz condition checked with exact rational comparison — no float rounding in threshold decisions. 

 basis = [Vec.from_ints([1, 1, 1]),  Vec.from_ints([-1, 0, 2]),  Vec.from_ints([3, 5, 6])]  reduced = lll_reduce(basis) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
