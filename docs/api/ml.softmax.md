<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/softmax.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `ml.softmax`
vdr.ml.softmax — Exact softmax and logsumexp. 

 from vdr.ml.softmax import softmax 

 probs = softmax(Vec.from_ints([1, 2, 3]))  # sums to exactly 1 

Uses max-subtraction for stability (exact, no overflow). Exp via Taylor series at configurable depth. All constants projected to basis frame to avoid D mixing. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/softmax.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `softmax`

```python
softmax(logits, exp_depth=16)
```

Exact softmax with max-subtraction for numerical stability. 

softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x)) 

I: logits as Vec, exp Taylor series depth O: probability Vec, sums to exactly 1 

Max-subtraction is exact (VDR comparison and subtraction). Each exp is exact rational at given depth. The sum is exact. Division is exact. 

 softmax(Vec.from_ints([1, 2, 3])) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/softmax.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `logsumexp`

```python
logsumexp(logits, exp_depth=16, log_depth=16)
```

Exact log-sum-exp: log(sum(exp(x_i))). 

Uses max-subtraction: lse(x) = max(x) + log(sum(exp(x_i - max(x)))) 

I: logits as Vec, exp and log depths O: VDR scalar 

 logsumexp(Vec.from_ints([1, 2, 3])) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/softmax.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `softmax_matrix_rows`

```python
softmax_matrix_rows(M, exp_depth=16)
```

Apply softmax independently to each row of a matrix. 

I: Mat of logits, exp depth O: Mat where each row is a probability distribution summing to exactly 1 

 weights = softmax_matrix_rows(score_matrix) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/softmax.py#L150"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `softmax_surrogate_square`

```python
softmax_surrogate_square(logits, shift=None)
```

Polynomial softmax surrogate using squared values. 

p_i = (x_i - shift)^2 / sum((x_j - shift)^2) 

Avoids exp entirely. Still sums to exactly 1. Shift parameter centers logits to keep V slots small. Default shift is the minimum logit. 

I: logits as Vec, optional shift as VDR O: probability Vec, sums to exactly 1 

 softmax_surrogate_square(Vec.from_ints([1, 2, 3]))  softmax_surrogate_square(logits, shift=VDR(0)) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
