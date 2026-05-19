<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/differential_eq.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `math.differential_eq`
vdr.math.differential_eq — Exact discrete ODE solvers over VDR. 

 from vdr.math.differential_eq import euler_solve, rk4_solve 

 trajectory = euler_solve(lambda x,y: y, VDR(1), VDR(0), VDR(1,10), 10)  # Euler method for dy/dx = y, exact at each step 

VDR eliminates arithmetic error but not method error. Euler h=1/10 gives (11/10)^10 regardless of exact or approximate arithmetic. The advantage is zero drift and exact reproducibility. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/differential_eq.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `euler_step`

```python
euler_step(f, x, y, h)
```

One Euler step: y_{n+1} = y_n + h * f(x_n, y_n). 

I: f(x, y) -> VDR, current x, current y, step size h (all VDR) O: next y as VDR, exact 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/differential_eq.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `euler_solve`

```python
euler_solve(f, y0, x0, h, n_steps)
```

Euler method ODE solver. 

dy/dx = f(x, y), y(x0) = y0, step size h, n steps. 

I: function f(x,y)->VDR, initial y0, initial x0, step h, steps O: list of (x, y) tuples, all exact VDR 

 # dy/dx = y, y(0) = 1, h = 1/10, 10 steps  euler_solve(lambda x,y: y, VDR(1), VDR(0), VDR(1,10), 10)  # y(1) = (11/10)^10 exact 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/differential_eq.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rk4_step`

```python
rk4_step(f, x, y, h)
```

One RK4 step. 

k1 = f(x, y) k2 = f(x + h/2, y + h*k1/2) k3 = f(x + h/2, y + h*k2/2) k4 = f(x + h, y + h*k3) y_{n+1} = y_n + (h/6)(k1 + 2*k2 + 2*k3 + k4) 

Butcher coefficients 1/6, 1/3, 1/3, 1/6 are all exact rationals. 

I: f(x,y)->VDR, current x, y, step h (all VDR) O: next y as VDR, exact 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/differential_eq.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rk4_solve`

```python
rk4_solve(f, y0, x0, h, n_steps)
```

RK4 ODE solver. 

I: function f(x,y)->VDR, initial y0, initial x0, step h, steps O: list of (x, y) tuples, all exact 

 rk4_solve(lambda x,y: y, VDR(1), VDR(0), VDR(1,10), 10)  # ~140x more accurate than Euler for same step count 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/differential_eq.py#L112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mat_exp`

```python
mat_exp(A, t, terms)
```

Truncated matrix exponential: sum_{k=0}^{terms} (tA)^k / k! 

I: square Mat A, scalar t (VDR), number of terms O: approximate exp(tA) as Mat, exact rational at given term count 

Each term is exact. More terms = closer to true exp. 

 mat_exp(Mat.from_ints([[0,1],[-1,0]]), VDR(1,10), 10) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/differential_eq.py#L140"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `picard_iterate`

```python
picard_iterate(f, y0, n_iterations)
```

Picard iteration for successive approximations. 

For dy/dx = f(x, y), y(0) = y0:  y_0(x) = y0  y_{n+1}(x) = y0 + integral_0^x f(t, y_n(t)) dt 

I: function f(x, y) -> VDR, initial value y0, iterations O: callable y_n(x) representing the nth Picard iterate 

For dy/dx = y, y(0) = 1:  After n iterations, y_n(x) = sum_{k=0}^n x^k / k!  The coefficients are exactly 1/k! 

 picard = picard_iterate(lambda x, y: y, VDR(1), 8)  picard(VDR(1))  # sum_{k=0}^8 1/k! 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/differential_eq.py#L194"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `picard_exp_coefficients`

```python
picard_exp_coefficients(n_iterations)
```

Picard iteration for dy/dx = y, y(0) = 1. 

Returns exact polynomial coefficients [1, 1, 1/2, 1/6, ..., 1/n!] up to degree n_iterations. 

This is the clean case where Picard gives Taylor coefficients exactly. 

I: number of iterations O: list of VDR coefficients [a0, a1, ..., an] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/differential_eq.py#L214"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `lotka_volterra_step`

```python
lotka_volterra_step(state, h, a, b, c, d_param)
```

One Euler step of the Lotka-Volterra predator-prey system. 

dx/dt = a*x - b*x*y dy/dt = -c*y + d*x*y 

I: state (x, y) as VDR tuple, step size h, parameters a, b, c, d (VDR) O: next state (x, y) as VDR tuple, exact 

 lotka_volterra_step((VDR(10), VDR(5)), VDR(1,100),  VDR(1,10), VDR(1,100), VDR(1,10), VDR(1,100)) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/differential_eq.py#L236"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `lotka_volterra_solve`

```python
lotka_volterra_solve(state0, h, a, b, c, d_param, n_steps)
```

Solve Lotka-Volterra system for n_steps Euler steps. 

I: initial (x, y), step h, parameters, steps O: list of (x, y) tuples, all exact 

 lotka_volterra_solve(  (VDR(10), VDR(5)), VDR(1,100),  VDR(1,10), VDR(1,100), VDR(1,10), VDR(1,100),  200  ) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
