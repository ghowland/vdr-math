<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/qed.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `physics.qed`
vdr.physics.qed — Exact QED electron anomalous magnetic moment. 

 from vdr.physics.qed import a2_coefficient, anomalous_moment 

 a2 = a2_coefficient()  # matches -0.328478965579... to 100 digits  ae = anomalous_moment(n_loops=3) 

The perturbation series a_e = A1*(alpha/pi) + A2*(alpha/pi)^2 + ... Each coefficient computed with Q335 basis constants. Odd denominator factors go into R via divmod. D stays 2^335. 

**Global Variables**
---------------
- **Q335_DENOM**

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/qed.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `a2_coefficient`

```python
a2_coefficient()
```

2-loop QED coefficient A2. 

A2 = 197/144 + pi^2/12 + 3*zeta(3)/4 - (pi^2/2)*ln(2) 

All terms use Q335 basis constants. Odd denominators (144 = 2^4 * 3^2) handled via exact VDR division — odd factors go into R. 

O: A2 as VDR, matches -0.328478965579... to 100 digits 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/qed.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `a3_coefficient`

```python
a3_coefficient()
```

3-loop QED coefficient A3 (Laporta & Remiddi). 

A3 involves zeta(5), pi^2*zeta(3), Li4(1/2), and products up to weight 5. All in Q335 basis or computable via Borwein. 

This is the structural form — full numerical value requires the complete analytical expression with ~100 terms. 

O: simplified A3 contribution as VDR 

Note: the full A3 = 1.181241456... is a known constant. Here we demonstrate the mechanism with the dominant terms. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/qed.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `anomalous_moment`

```python
anomalous_moment(n_loops=2)
```

Electron anomalous magnetic moment a_e = (g-2)/2. 

a_e = A1*(alpha/pi) + A2*(alpha/pi)^2 + A3*(alpha/pi)^3 + ... 

I: number of loops to include (1, 2, or 3) O: a_e as VDR, exact on exact inputs 

The fine-structure constant alpha is measured, not computed. It enters as Q335 at 100 digits. Series evaluation is exact on exact inputs. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/qed.py#L125"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `transcendental_weight`

```python
transcendental_weight(constant_name)
```

Transcendental weight assignment. 

rational = 0, pi = ln(2) = 1, zeta(n) = n, Li_n(1/2) = n, K(k) = 1. Maximal weight at L-loop QED = 2L - 1. 

I: constant name string O: integer weight 

 transcendental_weight("pi") -> 1  transcendental_weight("zeta3") -> 3 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
