<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/export.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `export`
vdr.export — Lossy export boundary. 

This is where exact VDR precision ends and target-format precision begins. Any loss belongs to the target format, not to VDR. 

 from vdr.export import to_decimal, to_float 

 x = VDR(1, 7)  print(to_decimal(x, 50))   # 50 decimal digits  print(to_float(x))         # lossy 64-bit float 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/export.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_fraction`

```python
to_fraction(x)
```

Exact projection to fractions.Fraction. Lossless for closed objects. Legacy-flattened for active objects. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/export.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_float`

```python
to_float(x)
```

Lossy projection to Python float (64-bit IEEE 754). Loss belongs to the float format. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/export.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_decimal`

```python
to_decimal(x, digits=50)
```

Render VDR value as a decimal string with `digits` significant figures. 

Uses mpmath if available for arbitrary precision. Falls back to manual long division from Fraction if not. 

The VDR value is exact. The decimal rendering is the lossy step. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
