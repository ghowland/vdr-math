<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/sampling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `ml.sampling`
vdr.ml.sampling — Exact sampling from probability distributions. 

 from vdr.ml.sampling import categorical_sample, top_k_probs, nucleus_probs 

 rng = VDRRandom(seed=42)  probs = softmax(logits)  idx = categorical_sample(probs, rng)  # exact rational CDF comparison 

All CDF values exact. Sampling decisions are exact comparisons, not float threshold tests. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/sampling.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `cdf_from_probs`

```python
cdf_from_probs(probs)
```

Cumulative distribution function from probability vector. 

I: probability Vec (should sum to exactly 1) O: CDF Vec where cdf[i] = sum_{j=0}^{i} probs[j] 

Final entry is exactly 1 when input sums to exactly 1. 

 probs = Vec([VDR(1,4), VDR(1,2), VDR(1,4)])  cdf = cdf_from_probs(probs)  # Vec([VDR(1,4), VDR(3,4), VDR(1)]) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/sampling.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `categorical_sample`

```python
categorical_sample(probs, rng)
```

Sample from a categorical distribution using exact CDF comparison. 

Generates a uniform random rational in [0, 1) and finds the first CDF bin that exceeds it. 

I: probability Vec (sums to exactly 1), VDRRandom instance O: integer index of sampled category 

The comparison is exact rational — no float threshold ambiguity. 

 from vdr.ml.rng import VDRRandom  rng = VDRRandom(seed=42)  idx = categorical_sample(probs, rng) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/sampling.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `argmax_sample`

```python
argmax_sample(probs)
```

Deterministic argmax sampling (greedy decoding). 

I: probability or logit Vec O: integer index of maximum element 

 argmax_sample(Vec([VDR(1,4), VDR(1,2), VDR(1,4)])) -> 1 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/sampling.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `top_k_probs`

```python
top_k_probs(probs, k)
```

Top-k filtering: keep only the k largest probabilities, renormalize. 

I: probability Vec, k (number of top entries to keep) O: filtered probability Vec (sums to exactly 1) 

Non-top-k entries set to zero. Remaining entries renormalized by exact division by their sum. 

 filtered = top_k_probs(Vec([VDR(1,10), VDR(6,10), VDR(3,10)]), 2)  # VDR(1,10) -> VDR(0), renormalize [VDR(6,10), VDR(3,10)] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/sampling.py#L145"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `nucleus_probs`

```python
nucleus_probs(probs, threshold)
```

Nucleus (top-p) sampling: keep smallest set of tokens whose cumulative probability exceeds threshold. 

I: probability Vec, threshold (VDR, e.g. VDR(9, 10) for top-90%) O: filtered probability Vec (sums to exactly 1) 

The threshold comparison is exact rational. 

 filtered = nucleus_probs(probs, VDR(9, 10)) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/sampling.py#L189"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `temperature_scale`

```python
temperature_scale(logits, temperature)
```

Scale logits by temperature before softmax. 

I: logits Vec, temperature (VDR, > 0) O: scaled logits Vec 

temperature < 1: sharper distribution (more greedy) temperature > 1: flatter distribution (more random) temperature = 1: unchanged 

 scaled = temperature_scale(logits, VDR(1, 2))  # temperature 0.5 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
