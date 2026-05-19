<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/active.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `active`
vdr.active — Active multiplication and division for VDR objects. 

Extends VDR arithmetic beyond the closed subclass. 

For [V1,D1,R1] * [V2,D2,R2]:  Frame: D1*D2  Closed part: V1*V2  Remainder: V1*R2 + V2*R1 + R1*R2 (cross-terms) 

All cross-terms captured as exact remainder structure. No approximation. 

 import vdr.active  vdr.active.install()   # patches VDR.__mul__ and __truediv__ 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/active.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `active_mul`

```python
active_mul(a, b)
```

Exact multiplication of two VDR objects, including active. 

Both in basis (closed or active): flatten to frame, divmod back. One in basis: rebase other to frame, divmod back. Both closed: direct formula V1*V2 / D1*D2. At least one active: construct product with cross-term remainder. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/active.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `active_div`

```python
active_div(a, b)
```

Exact division of two VDR objects. 

Both in basis (closed or active): flatten to frame, divmod back. One in basis: rebase other to frame, divmod back. By closed: multiply by reciprocal. By active: project divisor to exact rational, invert, multiply.  Divisor remainder structure lost (v1 compromise). 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/active.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `install`

```python
install()
```

Patch VDR with active multiplication and division. 

After this call, VDR * and / operators handle active objects. Called automatically by vdr.__init__. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/active.py#L238"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `uninstall`

```python
uninstall()
```

Restore original VDR operators (active mul/div raises). 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
