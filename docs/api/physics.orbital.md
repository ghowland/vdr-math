<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/orbital.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `physics.orbital`
vdr.physics.orbital — Exact orbital mechanics. 

 from vdr.physics.orbital import kepler_newton, orbit_closure_verify 

 E = kepler_newton(VDR(1), VDR(1, 2), depth=20)  # eccentric anomaly, >100 digits 

Two-body orbits close exactly. Float gives ~1e-12 position error. VDR gives zero. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/orbital.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `kepler_newton`

```python
kepler_newton(M, e, depth=20, start=None)
```

Solve Kepler's equation M = E - e*sin(E) via Newton iteration. 

x_{n+1} = x_n - (x_n - e*sin(x_n) - M) / (1 - e*cos(x_n)) 

Quadratic convergence: digits double per step. Depth 20 = >100 digits. 

I: mean anomaly M (VDR), eccentricity e (VDR), Newton depth,  optional starting guess O: eccentric anomaly E as VDR, exact rational at given depth 

 kepler_newton(VDR(1), VDR(1, 2), depth=20) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/orbital.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `kepler_position`

```python
kepler_position(a, e, E)
```

Position in orbital plane from eccentric anomaly. 

x = a * (cos(E) - e) y = a * sqrt(1 - e^2) * sin(E) 

I: semi-major axis a (VDR), eccentricity e (VDR),  eccentric anomaly E (VDR) O: (x, y) as VDR tuple, exact 

 x, y = kepler_position(VDR(1), VDR(1, 2), E_solved) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/orbital.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `kepler_velocity`

```python
kepler_velocity(a, e, E, mu=None)
```

Velocity in orbital plane from eccentric anomaly. 

vx = -sqrt(mu/a) * sin(E) / (1 - e*cos(E)) vy = sqrt(mu/a) * sqrt(1-e^2) * cos(E) / (1 - e*cos(E)) 

I: semi-major axis a, eccentricity e, eccentric anomaly E,  gravitational parameter mu (default 1) O: (vx, vy) as VDR tuple 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/orbital.py#L121"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `true_anomaly`

```python
true_anomaly(e, E)
```

True anomaly from eccentric anomaly. 

tan(nu/2) = sqrt((1+e)/(1-e)) * tan(E/2) 

I: eccentricity e, eccentric anomaly E (VDR) O: true anomaly nu as VDR (via atan2-like computation) 

For exact work, it's often better to use E directly. This returns the exact rational from the tan half-angle formula. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/orbital.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `orbit_propagate`

```python
orbit_propagate(a, e, M_start, n_steps, dM)
```

Propagate orbit for n_steps, advancing mean anomaly by dM each step. 

I: semi-major axis, eccentricity, starting mean anomaly,  number of steps, mean anomaly increment per step O: list of (x, y) positions 

 # one full orbit in 100 steps  positions = orbit_propagate(VDR(1), VDR(1,2), VDR(0), 100,  VDR(2) * PI / VDR(100)) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/orbital.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `orbit_closure_verify`

```python
orbit_closure_verify(positions)
```

Verify orbit closure: distance between first and last position. 

I: list of (x, y) positions from orbit_propagate O: squared distance as VDR. Should be exactly 0 for closed orbit. 

Float gives ~1e-12. VDR gives exact zero. 

 positions = orbit_propagate(...)  err = orbit_closure_verify(positions)  assert err == VDR(0) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/orbital.py#L198"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `orbital_energy`

```python
orbital_energy(a, e, E, mu=None)
```

Specific orbital energy (vis-viva). 

epsilon = -mu / (2a)  (for bound orbit) 

Also computable as v^2/2 - mu/r where r = a*(1 - e*cos(E)). 

I: semi-major axis, eccentricity, eccentric anomaly, mu O: specific energy as VDR, exact 

Energy should be conserved exactly across orbit. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
