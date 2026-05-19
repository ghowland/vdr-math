<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/optimization.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `math.optimization`
vdr.math.optimization — Exact optimization methods over VDR. 

 from vdr.math.optimization import newton_optimize, simplex_2d, bisection 

 x_opt = newton_optimize(f_prime, f_double_prime, VDR(5), 10)  x_root = bisection(f, VDR(0), VDR(2), 30) 

All iterates exact rational. Convergence rate determined by algorithm, not arithmetic precision. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/optimization.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `newton_optimize`

```python
newton_optimize(f_prime, f_double_prime, x0, n_steps)
```

Newton's method for optimization: find x where f'(x) = 0. 

x_{n+1} = x_n - f'(x_n) / f''(x_n) 

I: first derivative callable, second derivative callable,  starting point x0 (VDR), number of steps O: approximate minimizer as VDR, exact at each step 

 # minimize x^2 - 4x + 4: f'(x) = 2x-4, f''(x) = 2  newton_optimize(lambda x: VDR(2)*x - VDR(4),  lambda x: VDR(2),  VDR(0), 1) 
    -> VDR(2)  # converges in 1 step for quadratic 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/optimization.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `newton_root`

```python
newton_root(f, f_prime, x0, n_steps)
```

Newton's method for root finding: find x where f(x) = 0. 

x_{n+1} = x_n - f(x_n) / f'(x_n) 

I: function callable, derivative callable, start (VDR), steps O: approximate root as VDR, exact at each step 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/optimization.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gradient_descent_2d`

```python
gradient_descent_2d(grad, x0, y0, lr, steps)
```

Gradient descent in 2D. 

I: gradient function (x, y) -> (gx, gy), initial point,  learning rate lr (VDR), number of steps O: (x, y) tuple of VDR after optimization 

 gradient_descent_2d(  lambda x, y: (VDR(2)*x, VDR(2)*y),  # grad of x^2+y^2  VDR(5), VDR(3), VDR(1, 10), 20  ) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/optimization.py#L95"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `simplex_2d`

```python
simplex_2d(c, A, b)
```

Exact rational simplex method for 2D linear programming. 

Minimize c^T x subject to Ax <= b, x >= 0. 

I: cost vector c (list of 2 VDR), constraint matrix A (list of lists),  constraint RHS b (list of VDR) O: optimal solution as Vec, exact 

Simple implementation for small problems. Enumerates vertices. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/optimization.py#L178"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bisection`

```python
bisection(f, a, b, n_steps)
```

Bisection method for root finding. 

I: function f (VDR -> VDR), interval [a, b], number of bisection steps O: midpoint after n_steps as VDR, exact rational 

Each step halves the interval. After n steps, interval width is (b-a) / 2^n. 

 bisection(lambda x: x*x - VDR(2), VDR(1), VDR(2), 30)  # |x^2 - 2| < 10^-8 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/optimization.py#L210"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `golden_section`

```python
golden_section(f, a, b, n_steps)
```

Golden section search for unimodal function minimum. 

I: function f (VDR -> VDR), interval [a, b], steps O: approximate minimizer as VDR 

Uses rational approximation of golden ratio: phi ~ 610/987 (Fibonacci approximation). 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
