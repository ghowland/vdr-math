<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/autodiff.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `ml.autodiff`
vdr.ml.autodiff — Exact automatic differentiation via computation graph. 

 from vdr.ml.autodiff import Node, ensure_node, relu, mse_loss 

 a = Node(VDR(3))  b = Node(VDR(4))  c = a * b + a  c.backward()  print(a.grad)  # VDR(5) = d/da(a*b + a) = b + 1 

All gradients exact VDR rationals via reverse-mode AD. Chain rule applied symbolically — no numerical differentiation. Node values and gradients projected to basis frame to avoid D mixing. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/autodiff.py#L252"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `ensure_node`

```python
ensure_node(x)
```

Convert scalar to Node if needed. Projects to basis frame. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/autodiff.py#L267"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `relu`

```python
relu(x)
```

ReLU as autodiff Node. 

I: Node O: Node with relu applied and backward wired 

 a = Node(VDR(-3))  b = relu(a)  b.backward()  # a.grad == VDR(0) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/autodiff.py#L298"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sum_nodes`

```python
sum_nodes(xs)
```

Sum a sequence of Nodes. 

I: list of Nodes O: Node representing the sum 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/autodiff.py#L313"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mean_nodes`

```python
mean_nodes(xs)
```

Mean of a sequence of Nodes. 

I: list of Nodes O: Node representing the mean 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/autodiff.py#L331"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mse_loss`

```python
mse_loss(pred, target)
```

Mean squared error loss as autodiff graph. 

I: pred (list of Nodes), target (list of VDR or int) O: Node representing (1/n) * sum (pred_i - target_i)^2 

 pred = [Node(VDR(1)), Node(VDR(2))]  loss = mse_loss(pred, [VDR(0), VDR(3)])  loss.backward() 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/autodiff.py#L358"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `dot_nodes`

```python
dot_nodes(a, b)
```

Dot product of two Node sequences. 

I: two lists of Nodes O: Node representing sum a_i * b_i 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/autodiff.py#L371"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `linear_node`

```python
linear_node(weights, xs, bias)
```

Linear combination: sum(w_i * x_i) + bias. 

I: weights (list of VDR/int), xs (list of Nodes), bias (VDR/int) O: Node 

 out = linear_node([VDR(1,2), VDR(3,4)], [x1, x2], VDR(1)) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/autodiff.py#L387"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `zero_grads`

```python
zero_grads(nodes)
```

Reset gradients of all nodes to basis-frame zero. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/autodiff.py#L393"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `value_of_vec`

```python
value_of_vec(nodes)
```

Extract values from Node list as Vec. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/autodiff.py#L398"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `grad_of_vec`

```python
grad_of_vec(nodes)
```

Extract gradients from Node list as Vec. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/autodiff.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Node`
Computation graph node for reverse-mode automatic differentiation. 

Each node holds: 
    - value: exact VDR result of the forward computation 
    - grad: accumulated gradient (VDR), initialized to basis-frame zero 
    - _backward: closure that propagates gradient to children 
    - _children: set of parent nodes for topological sort 

 a = Node(VDR(3))  b = Node(VDR(4))  c = a * b  c.backward()  # a.grad == VDR(4), b.grad == VDR(3) 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/autodiff.py#L85"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(value, children=None, backward_fn=None)
```








---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/autodiff.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `backward`

```python
backward()
```

Reverse-mode automatic differentiation. 

Sets self.grad = 1 (in basis frame), then propagates through the graph in reverse topological order. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/autodiff.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `zero_grad`

```python
zero_grad()
```

Reset gradient to basis-frame zero. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
