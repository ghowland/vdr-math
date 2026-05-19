<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/dft.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `signal.dft`
vdr.signal.dft — Exact Discrete Fourier Transform. 

 from vdr.signal.dft import exact_dft, exact_idft, parseval_verify 

 X = exact_dft([VDR(1), VDR(2), VDR(3), VDR(4)])  x = exact_idft(X)  assert x == [VDR(1), VDR(2), VDR(3), VDR(4)]  # exact roundtrip 

All digital signals are rational. Every frequency bin is an exact complex value (real, imag) pair of VDR. 

Twiddle factors use sin/cos series at configurable depth. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/dft.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `twiddle_factor`

```python
twiddle_factor(k, n, N, depth=16)
```

DFT twiddle factor W_N^{kn} = exp(-2*pi*i*k*n/N). 

Returns (cos_part, sin_part) where W = cos - i*sin. 

The angle is 2*pi*k*n/N. For rational k*n/N, the trig values are computed via Taylor series at configurable depth. 

I: frequency index k, time index n, DFT size N, series depth O: (cos_val, neg_sin_val) as VDR tuple 

Special cases for exact values:  angle = 0: (1, 0)  angle = pi/2: (0, -1)  angle = pi: (-1, 0)  angle = 3pi/2: (0, 1) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/dft.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `exact_dft`

```python
exact_dft(x, depth=16)
```

Exact Discrete Fourier Transform. 

X[k] = sum_{n=0}^{N-1} x[n] * W_N^{kn} 

where W_N = exp(-2*pi*i/N). 

I: real signal as list of VDR, trig series depth O: list of (real, imag) VDR tuples, one per frequency bin 

 X = exact_dft([VDR(1), VDR(2), VDR(3), VDR(4)]) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/dft.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `exact_idft`

```python
exact_idft(X, depth=16)
```

Exact Inverse Discrete Fourier Transform. 

x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * W_N^{-kn} 

I: frequency domain as list of (real, imag) VDR tuples, depth O: real signal as list of VDR 

IDFT(DFT(x)) == x exactly. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/dft.py#L154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `parseval_verify`

```python
parseval_verify(x, X)
```

Verify Parseval's theorem: sum |x[n]|^2 == (1/N) * sum |X[k]|^2. 

I: time-domain signal (list of VDR), frequency-domain (list of (re,im) tuples) O: bool, True if identity holds exactly 

 X = exact_dft(x)  parseval_verify(x, X)  # True 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/dft.py#L182"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `dft_matrix`

```python
dft_matrix(N, depth=16)
```

Build the N x N DFT matrix explicitly. 

W[k,n] = twiddle_factor(k, n, N) 

Returns as pair of Mat (real_part, imag_part). 

I: DFT size N, series depth O: (real_mat, imag_mat) as Mat pair 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
