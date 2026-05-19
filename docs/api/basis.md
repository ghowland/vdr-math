<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/basis.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `basis`
vdr.basis — D-frame management and Q335 basis. 

The D-frame is the denominator you keep. divmod keeps it stable, overflow goes to R. D never explodes. 

Q335 (D = 2^335) is the default — proven across ~1000 tests in physics, transcendental, and ML domains. But any D is valid. If you're doing pure rational arithmetic with small denominators, D might be 1. If you're doing binary fixed-point, it might be 2^16. 

 import vdr.basis  vdr.basis.set_default(bits=335)   # Q335, the default 

 from vdr.basis import to_qbasis, qb_mul  a = to_qbasis(3.14159, bits=335)  # project onto 2^335 grid  b = to_qbasis(2.71828, bits=335)  c = qb_mul(a, b)                  # D stays 2^335, overflow in R 

**Global Variables**
---------------
- **TYPE_CHECKING**
- **DEFAULT_BITS**
- **Q335**

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/basis.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_default`

```python
get_default()
```

Return current default bit width. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/basis.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `set_default`

```python
set_default(bits)
```

Set the default basis bit width. 

 vdr.basis.set_default(668)  # 200-digit precision 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/basis.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `q_basis_denominator`

```python
q_basis_denominator(bits=None)
```

Return the basis denominator 2^bits. 

If bits is None, uses DEFAULT_BITS. 

 q_basis_denominator()      # 2^335 by default  q_basis_denominator(668)   # 2^668 for 200 digits 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/basis.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_qbasis`

```python
to_qbasis(x, bits=None)
```

Project a value onto the 2^bits grid as [round(x * 2^bits), 2^bits, 0]. 

Accepts VDR, int, float, or Fraction. 

 to_qbasis(VDR(22, 7))         # 22/7 projected onto Q335  to_qbasis(3.14159)            # float projected (one-time boundary loss)  to_qbasis(Fraction(1, 3))     # exact rational projected 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/basis.py#L125"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `vec_to_qbasis`

```python
vec_to_qbasis(v, bits=None)
```

Project each element of a Vec onto the basis. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/basis.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mat_to_qbasis`

```python
mat_to_qbasis(m, bits=None)
```

Project each element of a Mat onto the basis. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/basis.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `qb_add`

```python
qb_add(a, b, bits=None)
```

Addition staying in basis frame. 

Both operands should share the same D = 2^bits. Result: [(p1 + p2), 2^bits, 0] — one integer add. 

If operands have different D, rebases both first. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/basis.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `qb_mul`

```python
qb_mul(a, b, bits=None)
```

Multiplication with divmod back to basis frame. 

D stays 2^bits. Overflow in R. Zero loss. 

Product p1*p2 is a big integer. divmod by 2^bits:  Q, S = divmod(p1 * p2, 2^bits)  Result: [Q, 2^bits, [S, 2^bits, 0]] 

D never changed. R caught what V couldn't absorb — exactly. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/basis.py#L201"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `qb_div`

```python
qb_div(a, b, bits=None)
```

Division with divmod back to basis frame. 

a / b where both are in 2^bits frame. Multiply a.v by 2^bits, divmod by b.v. Odd factors go into R. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
