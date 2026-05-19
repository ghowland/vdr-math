<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/attention.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `ml.attention`
vdr.ml.attention — Exact self-attention mechanism. 

 from vdr.ml.attention import self_attention 

 outputs = self_attention(Q, K, V, causal=True)  # every weight row sums to exactly 1 

Attention scores exact. Softmax weights sum to exactly 1. Weighted sums exact. No float drift in long sequences. All mask fill values projected to basis frame to avoid D mixing. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/attention.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `attention_scores`

```python
attention_scores(Q, K)
```

Compute raw attention score matrix: S[i,j] = Q[i] . K[j]. 

I: Q, K as lists of Vec (sequence of query/key vectors) O: list of Vec (score rows), forming the score matrix 

 Q = [Vec.from_ints([1, 0]), Vec.from_ints([0, 1])]  K = [Vec.from_ints([1, 1]), Vec.from_ints([1, -1])]  scores = attention_scores(Q, K) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/attention.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `causal_mask`

```python
causal_mask(n)
```

Lower-triangular boolean mask for causal (autoregressive) attention. 

mask[i][j] = True if j <= i, False otherwise. 

I: sequence length n O: list of lists of bool 

 causal_mask(3) -> [[True,False,False],[True,True,False],[True,True,True]] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/attention.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `apply_boolean_mask`

```python
apply_boolean_mask(scores, mask, fill=None)
```

Apply boolean mask to attention scores. 

Where mask[i][j] is False, replace score with fill value (very negative for softmax to produce ~0 weight). Fill value is projected to basis frame once. 

I: score rows (list of Vec), mask (list of lists of bool),  fill value (VDR, default -1000 in basis) O: masked score rows (list of Vec) 

 masked = apply_boolean_mask(scores, causal_mask(n)) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/attention.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `attention_weights`

```python
attention_weights(scores, mask=None, exp_depth=16)
```

Convert scores to attention weights via softmax. 

Optionally applies causal mask before softmax. 

I: score rows (list of Vec), optional mask, exp depth O: weight rows (list of Vec), each row sums to exactly 1 

 weights = attention_weights(scores, mask=causal_mask(n)) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/attention.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `weighted_sum`

```python
weighted_sum(weights, values)
```

Weighted sum of value vectors: out = sum_j w_j * V[j]. 

I: weights as Vec, values as list of Vec O: Vec, exact 

 ws = Vec([VDR(1,2), VDR(1,2)])  vs = [Vec.from_ints([1, 0]), Vec.from_ints([0, 1])]  weighted_sum(ws, vs)  # Vec([VDR(1,2), VDR(1,2)]) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/attention.py#L160"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `attention_mix`

```python
attention_mix(weight_rows, V)
```

Apply attention weights to value vectors. 

output[i] = sum_j weight_rows[i][j] * V[j] 

I: weight_rows (list of Vec), V (list of Vec) O: list of Vec (attended outputs) 

 outputs = attention_mix(weights, V) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/attention.py#L174"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `self_attention`

```python
self_attention(Q, K, V, causal=False, exp_depth=16)
```

Full self-attention pipeline. 

1. Compute scores: S[i,j] = Q[i] . K[j] 2. Optionally apply causal mask 3. Softmax each row (weights sum to exactly 1) 4. Weighted sum of values 

I: Q, K, V as lists of Vec, causal flag, exp depth O: list of Vec (attended outputs) 

 Q = [Vec.from_ints([1, 0]), Vec.from_ints([0, 1])]  K = [Vec.from_ints([1, 1]), Vec.from_ints([1, -1])]  V = [Vec.from_ints([10, 0]), Vec.from_ints([0, 10])]  out = self_attention(Q, K, V, causal=True) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/attention.py#L202"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `multi_head_split`

```python
multi_head_split(vecs, n_heads)
```

Split vectors into multiple attention heads. 

Each Vec of dimension d is split into n_heads Vecs of dimension d // n_heads. 

I: list of Vec, number of heads O: list of lists — outer list is heads, inner is sequence 

 heads = multi_head_split([Vec.from_ints([1,2,3,4])], 2) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/attention.py#L233"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `multi_head_concat`

```python
multi_head_concat(heads)
```

Concatenate attention head outputs back into full vectors. 

I: list of lists — outer is heads, inner is sequence of Vec O: list of Vec with concatenated dimensions 

 full = multi_head_concat(heads) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
