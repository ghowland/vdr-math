<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/convolution.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `signal.convolution`
vdr.signal.convolution — Exact discrete convolution and correlation. 

 from vdr.signal.convolution import convolve, correlate 

 y = convolve([VDR(1), VDR(2), VDR(3)], [VDR(1), VDR(1)])  # [VDR(1), VDR(3), VDR(5), VDR(3)] exact 

Each product exact. Each sum exact. No accumulation error. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/convolution.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `convolve`

```python
convolve(a, b)
```

Discrete linear convolution. 

(a * b)[n] = sum_k a[k] * b[n-k] 

I: two lists of VDR O: convolution result, length len(a) + len(b) - 1 

 convolve([VDR(1), VDR(2), VDR(3)], [VDR(1), VDR(1)]) 
    -> [VDR(1), VDR(3), VDR(5), VDR(3)] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/convolution.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `correlate`

```python
correlate(a, b)
```

Discrete cross-correlation. 

(a ⋆ b)[n] = sum_k a[k] * b[k+n] 

Equivalent to convolve(a, reversed(b)). 

I: two lists of VDR O: cross-correlation result 

 correlate([VDR(1), VDR(2), VDR(3)], [VDR(1), VDR(1)]) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/convolution.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `toeplitz_mat`

```python
toeplitz_mat(h, n)
```

Build Toeplitz convolution matrix from impulse response h. 

For h of length m and output length n, builds an n x n matrix where row i has h shifted by i positions. 

I: impulse response h (list of VDR), output dimension n O: Toeplitz Mat (n x n) 

Used to verify that matrix-vector product equals direct convolution. 

 T = toeplitz_mat([VDR(1), VDR(2), VDR(3)], 4) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/convolution.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `convolve_via_toeplitz`

```python
convolve_via_toeplitz(a, b)
```

Convolution via Toeplitz matrix-vector product. 

Verifies that direct convolution equals matrix form. 

I: two lists of VDR O: convolution result via matrix multiply 

 direct = convolve(a, b)  matrix = convolve_via_toeplitz(a, b)  assert direct == matrix  # exact 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/convolution.py#L137"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `deconvolve`

```python
deconvolve(y, h)
```

Deconvolution: given y = h * x, recover x. 

Uses polynomial division: x = y / h in polynomial ring. 

I: output signal y, impulse response h (lists of VDR) O: input signal x (list of VDR), exact 

Only works when h divides y exactly (no remainder). 

 x = deconvolve(convolve(a, b), b)  assert x == a  # exact recovery 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
