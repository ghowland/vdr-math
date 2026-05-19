<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/trainer.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `ml.trainer`
vdr.ml.trainer — Exact training loops for VDR neural networks. 

 from vdr.ml.trainer import train_step, train_epoch, evaluate_classification 

 loss = train_step(model, x, y, optimizer)  losses = train_epoch(model, dataset, optimizer)  accuracy = evaluate_classification(model, dataset) 

Every forward pass, gradient, and parameter update exact VDR rational. Loss values are exact fractions, not approximate floats. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/trainer.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `train_step`

```python
train_step(model, x, y, optimizer)
```

One training step: forward, loss, backward, update. 

I: model (Module with forward/backward), input x (Vec),  target y (Vec), optimizer (SGD or Momentum) O: loss value as VDR, exact 

S: updates model parameters via optimizer 

 loss = train_step(model, x, y, opt) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/trainer.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `train_epoch`

```python
train_epoch(model, dataset, optimizer)
```

Train on entire dataset for one epoch. 

I: model, dataset as list of (x, y) tuples, optimizer O: list of per-sample loss values (VDR) 

 losses = train_epoch(model, [(x1,y1), (x2,y2), ...], opt) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/trainer.py#L79"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `evaluate_epoch`

```python
evaluate_epoch(model, dataset)
```

Evaluate model on dataset without updating parameters. 

I: model, dataset as list of (x, y) tuples O: list of per-sample loss values (VDR) 

 eval_losses = evaluate_epoch(model, test_set) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/trainer.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `predict_class`

```python
predict_class(model, x)
```

Predict class label by argmax of model output. 

I: model, input x (Vec) O: integer class index 

 label = predict_class(model, x) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/trainer.py#L115"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `evaluate_classification`

```python
evaluate_classification(model, dataset)
```

Evaluate classification accuracy on dataset. 

I: model, dataset as list of (x, y) tuples where y encodes target class  (either as one-hot Vec or as int) O: accuracy as VDR (correct / total), exact rational 

 accuracy = evaluate_classification(model, test_set)  # e.g. VDR(7, 10) = 70% accuracy 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
