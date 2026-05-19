<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/optim.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `ml.optim`
vdr.ml.optim — Exact optimizers for VDR neural networks. 

 from vdr.ml.optim import SGD, Momentum 

 opt = SGD(model.parameters(), lr=VDR(1, 100))  opt.zero_grad()  loss = ...  model.backward(grad)  opt.step()  # w = w - lr * grad, exact 

All parameter updates exact VDR arithmetic. Hyperparameters projected to basis frame at construction to avoid per-element D mixing on every step. 



---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/optim.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SGD`
Stochastic Gradient Descent optimizer. 

update: w = w - lr * grad 

I: list of parameters (VecParam or MatParam), learning rate (VDR) 

lr is projected to basis frame at construction. 

 opt = SGD(model.parameters(), lr=VDR(1, 100))  opt.step() 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/optim.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(params, lr)
```








---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/optim.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `step`

```python
step()
```

Apply one SGD update to all parameters. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/optim.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `zero_grad`

```python
zero_grad()
```

Zero all parameter gradients. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/optim.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Momentum`
SGD with momentum optimizer. 

v = beta * v + grad w = w - lr * v 

I: params list, learning rate (VDR), momentum beta (VDR, default 9/10) 

lr and beta are projected to basis frame at construction. Velocity buffers initialized in basis frame. 

 opt = Momentum(model.parameters(), lr=VDR(1, 100), beta=VDR(9, 10))  opt.step() 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/optim.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(params, lr, beta=None)
```








---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/optim.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `step`

```python
step()
```

Apply one momentum update to all parameters. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/optim.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `zero_grad`

```python
zero_grad()
```

Zero all parameter gradients. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
