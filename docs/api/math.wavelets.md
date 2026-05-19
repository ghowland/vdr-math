<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/wavelets.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `math.wavelets`
vdr.math.wavelets — Exact Haar wavelet transforms. 

 from vdr.math.wavelets import haar_forward, haar_inverse 

 avgs, dets = haar_forward([VDR(1), VDR(3), VDR(5), VDR(7)])  signal = haar_inverse(avgs, dets)  # signal == [VDR(1), VDR(3), VDR(5), VDR(7)] exactly 

Perfect reconstruction. Parseval energy identity exact. All operations exact VDR rational arithmetic. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/wavelets.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `haar_forward`

```python
haar_forward(signal)
```

One-level Haar forward transform. 

I: signal as list of VDR (length must be even) O: (averages, details) each list of VDR, half the length 

averages[k] = (signal[2k] + signal[2k+1]) / 2 details[k]  = (signal[2k] - signal[2k+1]) / 2 

 haar_forward([VDR(1), VDR(3), VDR(5), VDR(7)]) 
    -> ([VDR(2), VDR(6)], [VDR(-1), VDR(-1)]) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/wavelets.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `haar_inverse`

```python
haar_inverse(avgs, dets)
```

One-level Haar inverse transform. 

I: averages and details (lists of VDR, same length) O: reconstructed signal (list of VDR, double the length) 

signal[2k]   = avgs[k] + dets[k] signal[2k+1] = avgs[k] - dets[k] 

Perfect reconstruction: haar_inverse(*haar_forward(x)) == x exactly. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/wavelets.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `haar_multilevel`

```python
haar_multilevel(signal, levels)
```

Multi-level Haar decomposition. 

I: signal (list of VDR, length power of 2), number of levels O: list of (averages, details) per level, from finest to coarsest 

At each level, the averages from the previous level become the input signal for the next level. 

 decomp = haar_multilevel([VDR(1),VDR(3),VDR(5),VDR(7)], 2)  # level 0: avgs=[2,6], dets=[-1,-1]  # level 1: avgs=[4], dets=[-2] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/wavelets.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `haar_reconstruct_multilevel`

```python
haar_reconstruct_multilevel(decomposition)
```

Reconstruct signal from multi-level Haar decomposition. 

I: list of (averages, details) from haar_multilevel O: reconstructed signal (list of VDR), exact 

Perfect reconstruction: reconstruct(multilevel(x, n)) == x exactly. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/wavelets.py#L131"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `energy`

```python
energy(signal)
```

Signal energy: sum |x[n]|^2. 

I: signal as list of VDR O: energy as VDR, exact 

 energy([VDR(1), VDR(3), VDR(5), VDR(7)]) -> VDR(84) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/wavelets.py#L146"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `parseval_verify`

```python
parseval_verify(signal, decomposition)
```

Verify Parseval energy identity for Haar transform. 

For our definition (divide by 2), the energy splits as:  energy(signal) = 2^levels * energy(coarsest_avgs) +  sum over levels of 2^(level+1) * energy(details) 

Simpler check: reconstruct and compare energies. 

I: original signal, decomposition from haar_multilevel O: bool, True if energy preserved exactly 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/wavelets.py#L163"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `threshold_hard`

```python
threshold_hard(details, thresh)
```

Hard thresholding for wavelet denoising. 

I: detail coefficients (list of VDR), threshold (VDR) O: thresholded details (list of VDR) 

Coefficients with |d| < thresh are set to zero. Exact rational comparison — no float ambiguity at boundary. 

 threshold_hard([VDR(1,10), VDR(3), VDR(-1,5)], VDR(1,2)) 
    -> [VDR(0), VDR(3), VDR(0)] 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
