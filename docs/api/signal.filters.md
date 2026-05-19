<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/filters.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `signal.filters`
vdr.signal.filters — Exact digital filters. 

 from vdr.signal.filters import iir_filter, moving_average, z_transform 

 y = iir_filter([VDR(1),VDR(0),VDR(0)], VDR(1,2))  # y[n] = (1/2)*y[n-1] + x[n], exact at every step 

IIR chains: each step nests one R level. When rational powers collapse (e.g. (1/sqrt(2))^20 = 1/1024), normalization rule N7 fires. Year-long operation same precision as second 1. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/filters.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `iir_filter`

```python
iir_filter(x, a_coeff)
```

First-order IIR filter: y[n] = a * y[n-1] + x[n]. 

I: input signal x (list of VDR), feedback coefficient a (VDR) O: output signal y (list of VDR), exact at every step 

Each step is one VDR multiplication and one addition. No drift regardless of signal length. 

 iir_filter([VDR(1), VDR(0), VDR(0), VDR(0)], VDR(1, 2)) 
    -> [VDR(1), VDR(1,2), VDR(1,4), VDR(1,8)] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/filters.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `iir_filter_general`

```python
iir_filter_general(x, b_coeffs, a_coeffs)
```

General IIR filter (direct form I). 

y[n] = sum_{k=0}^{M} b[k]*x[n-k] - sum_{k=1}^{N} a[k]*y[n-k] 

Convention: a[0] = 1 (normalized). If a[0] != 1, all coefficients are divided by a[0]. 

I: input x (list of VDR), feedforward b (list of VDR),  feedback a (list of VDR, a[0]=1) O: output y (list of VDR) 

 # Second-order: y[n] = x[n] + 0.5*x[n-1] - 0.25*y[n-1]  iir_filter_general(x, [VDR(1), VDR(1,2)], [VDR(1), VDR(1,4)]) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/filters.py#L107"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `moving_average`

```python
moving_average(signal, window)
```

Moving average filter. 

y[n] = (1/window) * sum_{k=0}^{window-1} x[n-k] 

I: input signal (list of VDR), window size (int) O: filtered signal (list of VDR), exact 

Note: division by window introduces 1/window in denominator. If window has odd factors, this is where the decimal trap bites float but VDR handles exactly. 

 moving_average([VDR(1), VDR(2), VDR(3), VDR(4)], 3) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/filters.py#L146"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `z_transform`

```python
z_transform(h, z)
```

Evaluate z-transform H(z) = sum_{n=0}^{N-1} h[n] * z^{-n}. 

I: impulse response h (list of VDR), evaluation point z (VDR) O: H(z) as VDR, exact 

 z_transform([VDR(1), VDR(1, 2), VDR(1, 4)], VDR(2))  # H(2) = 1 + 1/2 * 1/2 + 1/4 * 1/4 = 1 + 1/4 + 1/16 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/filters.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `frequency_response`

```python
frequency_response(b_coeffs, a_coeffs, omega, depth=16)
```

Frequency response H(e^{j*omega}) of a digital filter. 

H(z) = B(z) / A(z) evaluated at z = e^{j*omega}. 

I: feedforward b, feedback a coefficients (lists of VDR),  frequency omega (VDR), trig series depth O: (magnitude_sq, real, imag) as VDR tuple 

Uses exact sin/cos series for the complex exponential. 

 frequency_response([VDR(1)], [VDR(1), VDR(-1,2)], VDR(1,4)) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
