<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `ml.nn`
vdr.ml.nn — Exact neural network layers and parameters. 

 from vdr.ml.nn import Linear, ReLU, Sequential 

 model = Sequential([  Linear.from_ints([[1,2],[3,4]], [0,0]),  ReLU(),  Linear.from_ints([[1,0],[0,1]], [1,1]),  ])  output = model.forward(Vec.from_ints([1, 2]))  model.backward(mse_grad(output, target)) 

All forward passes exact. All gradients exact via chain rule. No float drift in training loops. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `relu_scalar`

```python
relu_scalar(x)
```

ReLU on a single VDR: max(0, x). 

I: VDR O: VDR (0 if x < 0, x otherwise) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `relu_prime_scalar`

```python
relu_prime_scalar(x)
```

ReLU derivative on a single VDR. 

I: VDR O: VDR(1) if x > 0, VDR(0) if x <= 0 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `VecParam`
Trainable vector parameter with gradient accumulator. 

 p = VecParam(Vec.from_ints([1, 2, 3]), name="bias")  p.zero_grad()  p.step(lr=VDR(1, 100)) 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(value, name=None)
```








---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply_update`

```python
apply_update(update, lr)
```

Apply an arbitrary update vector: value = value - lr * update. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `combine_scaled`

```python
combine_scaled(a, a_scale, b, b_scale)
```

Weighted combination: a_scale * a + b_scale * b. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `grad_like`

```python
grad_like()
```

Return zero Vec with same shape as value. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `step`

```python
step(lr)
```

SGD update: value = value - lr * grad. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_qbasis`

```python
to_qbasis(bits=None)
```

Project value and reset grad in basis frame. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `zero_grad`

```python
zero_grad()
```

Reset gradient to zero. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `zeros_like`

```python
zeros_like()
```

Return zero Vec with same shape. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L130"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MatParam`
Trainable matrix parameter with gradient accumulator. 

 p = MatParam(Mat.from_ints([[1,2],[3,4]]), name="weight")  p.zero_grad()  p.step(lr=VDR(1, 100)) 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L141"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(value, name=None)
```








---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply_update`

```python
apply_update(update, lr)
```

Apply an arbitrary update matrix: value = value - lr * update. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L163"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `combine_scaled`

```python
combine_scaled(a, a_scale, b, b_scale)
```

Weighted combination of two Mat values. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L150"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `grad_like`

```python
grad_like()
```

Return zero Mat with same shape. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `step`

```python
step(lr)
```

SGD update: value = value - lr * grad. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L172"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_qbasis`

```python
to_qbasis(bits=None)
```

Project value and reset grad in basis frame. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L146"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `zero_grad`

```python
zero_grad()
```

Reset gradient to zero. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `zeros_like`

```python
zeros_like()
```

Return zero Mat with same shape. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L199"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Module`
Base class for neural network modules. 




---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L214"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `backward`

```python
backward(grad_out)
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L211"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L202"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `parameters`

```python
parameters()
```

Return list of trainable parameters (VecParam or MatParam). 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L217"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_qbasis`

```python
to_qbasis(bits=None)
```

Project all parameters to basis frame. Override in subclasses. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L206"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `zero_grad`

```python
zero_grad()
```

Zero all parameter gradients. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L228"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Linear`
Exact linear layer: y = W @ x + b. 

Forward: matrix-vector multiply + bias add. Backward: exact gradients for W, b, and input. 

 layer = Linear.from_ints([[1,2],[3,4]], [0,0])  y = layer.forward(Vec.from_ints([1, 1])) 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L239"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(weight, bias, name=None)
```

I: weight as Mat, bias as Vec, optional name 




---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L271"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `backward`

```python
backward(grad_out)
```

Backpropagate gradient through linear layer. 

Given dL/dy (grad_out as Vec):  dL/dW = grad_out (outer) input  dL/db = grad_out  dL/dx = W^T @ grad_out 

I: grad_out Vec (gradient of loss w.r.t. output) O: grad_input Vec (gradient of loss w.r.t. input) S: accumulates into weight.grad and bias.grad 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L260"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

y = W @ x + b. 

I: input Vec x O: output Vec y, exact S: caches input for backward 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L252"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `from_fracs`

```python
from_fracs(weight, bias, name=None)
```

Construct from fraction tuples. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L247"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `from_ints`

```python
from_ints(weight, bias, name=None)
```

Construct from integer lists. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L257"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `parameters`

```python
parameters()
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L217"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_qbasis`

```python
to_qbasis(bits=None)
```

Project all parameters to basis frame. Override in subclasses. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L206"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `zero_grad`

```python
zero_grad()
```

Zero all parameter gradients. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L302"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ReLU`
Exact ReLU activation: y_i = max(0, x_i). 

Forward: piecewise linear, exact. Backward: 0 or 1 per element, exact. 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L310"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(name=None)
```








---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L327"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `backward`

```python
backward(grad_out)
```

dL/dx_i = dL/dy_i * relu'(x_i). 

relu'(x) = 1 if x > 0, 0 otherwise. 

I: grad_out Vec O: grad_input Vec, exact 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L317"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

y_i = max(0, x_i). 

I: input Vec O: output Vec, exact piecewise linear 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L314"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `parameters`

```python
parameters()
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L342"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_qbasis`

```python
to_qbasis(bits=None)
```

ReLU has no parameters. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L206"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `zero_grad`

```python
zero_grad()
```

Zero all parameter gradients. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L351"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Sequential`
Sequential composition of layers. 

Forward: passes through layers in order. Backward: passes gradient through layers in reverse order. 

 model = Sequential([  Linear.from_ints([[1,2],[3,4]], [0,0]),  ReLU(),  Linear.from_ints([[1,0],[0,1]], [1,1]),  ]) 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L365"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(layers)
```








---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L380"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `backward`

```python
backward(grad_out)
```

Pass gradient through all layers in reverse. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L374"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Pass x through all layers in order. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L368"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `parameters`

```python
parameters()
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L386"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_qbasis`

```python
to_qbasis(bits=None)
```

Project all layers to basis frame. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L206"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `zero_grad`

```python
zero_grad()
```

Zero all parameter gradients. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L397"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FFN`
Two-layer feed-forward network with ReLU activation. 

y = W2 @ relu(W1 @ x + b1) + b2 

Common building block for transformers and MLPs. 

 ffn = FFN(  Linear.from_ints([[1,2],[3,4]], [0,0]),  ReLU(),  Linear.from_ints([[1,0],[0,1]], [1,1]),  ) 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L412"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(l1, act, l2)
```








---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L429"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `backward`

```python
backward(grad_out)
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L424"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L417"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `parameters`

```python
parameters()
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L434"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_qbasis`

```python
to_qbasis(bits=None)
```

Project both linear layers to basis frame. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/nn.py#L420"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `zero_grad`

```python
zero_grad()
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
