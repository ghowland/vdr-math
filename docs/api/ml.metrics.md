<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/metrics.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `ml.metrics`
vdr.ml.metrics — Exact evaluation metrics. 

 from vdr.ml.metrics import exact_accuracy, denominator_complexity_vec 

 acc = exact_accuracy([0, 1, 2], [0, 1, 1])  # VDR(2, 3) = 66.7% accuracy, exact rational 

 complexity = denominator_complexity_vec(v)  # (distinct_denoms, sum_denoms, node_count) 

Accuracy as exact fraction. Denominator complexity as structural metric. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/metrics.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `exact_accuracy`

```python
exact_accuracy(pred_ids, target_ids)
```

Classification accuracy as exact rational. 

I: list of predicted class ids, list of target class ids O: accuracy as VDR (correct / total) 

 exact_accuracy([0, 1, 2, 1], [0, 1, 1, 1]) 
    -> VDR(3, 4) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/metrics.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `argmax_vec`

```python
argmax_vec(v)
```

Index of maximum element in Vec. 

I: Vec O: integer index 

 argmax_vec(Vec([VDR(1,3), VDR(1,2), VDR(1,4)])) -> 1 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/metrics.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `denominator_complexity_vec`

```python
denominator_complexity_vec(v)
```

Denominator complexity of a Vec. 

I: Vec O: (distinct_denoms, sum_denoms, node_count) tuple 

Measures structural complexity of the VDR representation. Lower is simpler. Used to track model complexity during training. 

 dc = denominator_complexity_vec(Vec([VDR(1,2), VDR(1,3)]))  # (2, 5, 2) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/metrics.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `denominator_complexity_mat`

```python
denominator_complexity_mat(m)
```

Denominator complexity of a Mat. 

I: Mat O: (distinct_denoms, sum_denoms, node_count) tuple 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/metrics.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `parameter_denominator_complexity`

```python
parameter_denominator_complexity(params)
```

Aggregate denominator complexity across all model parameters. 

I: list of VecParam or MatParam O: (total_distinct, total_sum, total_count) tuple 

 complexity = parameter_denominator_complexity(model.parameters()) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/metrics.py#L143"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `total_parameters`

```python
total_parameters(params)
```

Count total number of scalar parameters. 

I: list of VecParam or MatParam O: integer count 

 n = total_parameters(model.parameters()) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/metrics.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mean_loss`

```python
mean_loss(losses)
```

Mean of a list of loss values. 

I: list of VDR loss values O: mean as VDR, exact rational 

 avg = mean_loss([VDR(1,3), VDR(1,2), VDR(1,4)]) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
