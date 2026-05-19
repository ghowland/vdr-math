<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/losses.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `ml.losses`
vdr.ml.losses — Exact loss functions. 

 from vdr.ml.losses import mse, l1, mse_grad 

 loss = mse(pred, target)      # exact MSE as VDR  grad = mse_grad(pred, target) # exact gradient as Vec 

All losses exact VDR rationals. Gradients exact via closed-form expressions. Constants projected to basis frame to avoid D mixing in hot paths. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/losses.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mse`

```python
mse(pred, target)
```

Mean squared error: (1/n) * sum (pred_i - target_i)^2. 

I: pred Vec, target Vec (same dimension) O: MSE as VDR, exact 

 mse(Vec.from_ints([1, 2, 3]), Vec.from_ints([1, 2, 4])) 
    -> VDR(1, 3) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/losses.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `l1`

```python
l1(pred, target)
```

Mean absolute error: (1/n) * sum |pred_i - target_i|. 

I: pred Vec, target Vec O: L1 loss as VDR, exact 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/losses.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `hinge_binary`

```python
hinge_binary(score, label)
```

Binary hinge loss: max(0, 1 - label * score). 

I: score (VDR), label (int, +1 or -1) O: hinge loss as VDR 

 hinge_binary(VDR(3, 2), 1) -> to_qbasis(0)  hinge_binary(VDR(1, 2), 1) -> VDR(1, 2) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/losses.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mse_grad`

```python
mse_grad(pred, target)
```

Gradient of MSE with respect to pred. 

d/d(pred_i) [(1/n) sum (pred_j - target_j)^2] = (2/n) * (pred_i - target_i) 

I: pred Vec, target Vec O: gradient Vec, exact 

 grad = mse_grad(Vec.from_ints([1, 2, 3]), Vec.from_ints([1, 2, 4])) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/losses.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `l1_grad`

```python
l1_grad(pred, target)
```

Subgradient of L1 loss with respect to pred. 

d/d(pred_i) |pred_i - target_i| = sign(pred_i - target_i) / n 

I: pred Vec, target Vec O: subgradient Vec 

At pred_i == target_i, returns 0 (subgradient choice). 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/losses.py#L151"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `cross_entropy_binary`

```python
cross_entropy_binary(pred_prob, label, log_depth=16)
```

Binary cross-entropy loss: -[y*ln(p) + (1-y)*ln(1-p)]. 

I: predicted probability pred_prob (VDR in (0,1)),  true label (VDR, 0 or 1), log depth O: loss as VDR 

Uses exact log series. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
