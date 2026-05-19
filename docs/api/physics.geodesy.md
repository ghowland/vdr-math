<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/geodesy.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `physics.geodesy`
vdr.physics.geodesy — Exact geodetic computations. 

 from vdr.physics.geodesy import helmert_forward, helmert_roundtrip_verify 

 transformed = helmert_forward(coords, params)  assert helmert_roundtrip_verify(coords, params)  # True, exact 

Helmert 7-parameter transformation: exact forward and inverse. Roundtrip recovers original coordinates identically. Zero residual. Float gives ~1 nm error. VDR gives zero. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/geodesy.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rotation_matrix_small_angle`

```python
rotation_matrix_small_angle(rx, ry, rz)
```

Small-angle rotation matrix for Helmert transformation. 

R = [[1, -rz, ry], [rz, 1, -rx], [-ry, rx, 1]] 

I: rotation angles rx, ry, rz (VDR, in radians, small) O: 3x3 rotation Mat 

For exact Helmert, angles are rational (e.g. arcseconds converted to radians). 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/geodesy.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `helmert_forward`

```python
helmert_forward(coords, params)
```

Helmert 7-parameter forward transformation. 

X' = T + (1 + s) * R * X 

where T = translation vector, s = scale factor, R = rotation matrix. 

I: coordinates as Vec (3D), params dict with keys:  "tx", "ty", "tz" (translation VDR),  "rx", "ry", "rz" (rotation VDR),  "s" (scale factor VDR) O: transformed Vec, exact 

 params = {  "tx": VDR(1, 10), "ty": VDR(2, 10), "tz": VDR(3, 10),  "rx": VDR(1, 1000), "ry": VDR(2, 1000), "rz": VDR(3, 1000),  "s": VDR(1, 1000000),  }  result = helmert_forward(Vec.from_ints([100, 200, 300]), params) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/geodesy.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `helmert_inverse`

```python
helmert_inverse(coords, params)
```

Helmert 7-parameter inverse transformation. 

X = R^{-1} * (X' - T) / (1 + s) 

For small-angle R, R^{-1} is R with negated angles. 

I: transformed coordinates as Vec, params dict O: original coordinates as Vec, exact 

 original = helmert_inverse(transformed, params) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/geodesy.py#L123"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `helmert_roundtrip_verify`

```python
helmert_roundtrip_verify(coords, params)
```

Verify Helmert roundtrip: inverse(forward(X)) == X. 

I: original coordinates Vec, params dict O: bool, True if roundtrip recovers original exactly 

Float gives ~1 nm error. VDR gives True. 

 assert helmert_roundtrip_verify(coords, params) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/geodesy.py#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `coordinate_transform`

```python
coordinate_transform(coords, M, offset=None)
```

General affine coordinate transformation. 

X' = M * X + offset 

I: coordinates Vec, transformation Mat, optional offset Vec O: transformed Vec, exact 

 rotated = coordinate_transform(coords, rotation_matrix, translation) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/geodesy.py#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `weighted_average`

```python
weighted_average(values, weights)
```

Weighted average with exact rational weights. 

result = sum(w_i * x_i) / sum(w_i) 

Weights sum to exactly 1 after normalization (same structural property as softmax). 

I: list of VDR values, list of VDR weights O: weighted average as VDR, exact 

 weighted_average([VDR(1), VDR(2), VDR(3)], [VDR(1,3), VDR(1,3), VDR(1,3)]) 
    -> VDR(2) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/geodesy.py#L185"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `misclosure`

```python
misclosure(legs)
```

Survey misclosure: sum of leg vectors. 

For a closed traverse, the sum should be exactly zero. Any nonzero result is measurement error (not arithmetic error). 

I: list of Vec (survey leg vectors) O: misclosure Vec. VDR arithmetic contributes nothing — if nonzero,  it's pure measurement error. 

 legs = [Vec.from_ints([100, 0, 0]),  Vec.from_ints([0, 100, 0]),  Vec.from_ints([-100, 0, 0]),  Vec.from_ints([0, -100, 0])]  m = misclosure(legs)  # Vec([VDR(0), VDR(0), VDR(0)]) exactly 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
