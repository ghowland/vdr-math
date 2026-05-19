<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/tensor.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `ml.tensor`
vdr.ml.tensor — Batch tensor operations for VDR. 

 from vdr.ml.tensor import Tensor3D, batched_matvec 

 t = Tensor3D([[row1_vecs], [row2_vecs]])  results = batched_matvec(matrices, vectors) 

Tensor3D is batch x sequence x dimension. All operations exact VDR arithmetic across batches. Fill and zero values projected to basis frame. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/tensor.py#L138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `batched_matvec`

```python
batched_matvec(mats, vecs)
```

Batched matrix-vector product. 

I: list of Mat, list of Vec (same length) O: list of Vec where result[i] = mats[i] @ vecs[i] 

 results = batched_matvec([M1, M2], [v1, v2]) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/tensor.py#L152"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rowwise_add_bias`

```python
rowwise_add_bias(rows, bias)
```

Add bias vector to each row. 

I: list of Vec (rows), bias Vec O: list of Vec where result[i] = rows[i] + bias 

 biased = rowwise_add_bias([v1, v2, v3], bias) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/tensor.py#L164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `masked_fill_rows`

```python
masked_fill_rows(rows, mask, fill=None)
```

Apply element-wise mask to rows of Vec. 

Fill value projected to basis frame once. 

I: list of Vec, mask as list of list of bool,  fill value (VDR, default 0 in basis) O: list of Vec with masked positions replaced by fill 

 masked = masked_fill_rows(rows, mask, VDR(-1000)) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/tensor.py#L194"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `reduce_sum_rows`

```python
reduce_sum_rows(rows)
```

Element-wise sum across a list of Vec. 

I: list of Vec (all same dimension) O: Vec where result[j] = sum_i rows[i][j] 

 total = reduce_sum_rows([v1, v2, v3]) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/tensor.py#L213"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `stack_vecs`

```python
stack_vecs(vecs)
```

Stack list of Vec into a Mat (each Vec becomes a row). 

I: list of Vec (all same dimension) O: Mat 

 m = stack_vecs([Vec.from_ints([1,2]), Vec.from_ints([3,4])]) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/tensor.py#L225"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `unstack_vecs`

```python
unstack_vecs(m)
```

Unstack Mat into list of row Vec. 

I: Mat O: list of Vec 

 vecs = unstack_vecs(m) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/tensor.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Tensor3D`
3D tensor: batch x sequence x dimension. 

Stored as list of list of Vec. 

 data = [  [Vec.from_ints([1,2]), Vec.from_ints([3,4])],   # batch 0  [Vec.from_ints([5,6]), Vec.from_ints([7,8])],   # batch 1  ]  t = Tensor3D(data)  t.shape  # (2, 2, 2) 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/tensor.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(data)
```

I: list of list of Vec (batch x sequence) 


---

#### <kbd>property</kbd> batch





---

#### <kbd>property</kbd> dim

Feature dimension. 

---

#### <kbd>property</kbd> nrows

Sequence length. 

---

#### <kbd>property</kbd> shape

(batch, sequence, dimension). 



---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/tensor.py#L127"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_fractions`

```python
to_fractions()
```

Export as nested list of Fraction. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/tensor.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `zero`

```python
zero(b, n, d)
```

Zero tensor of shape (b, n, d) in basis frame. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
