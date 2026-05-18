"""
vdr.math.optimization — Exact optimization methods over VDR.

    from vdr.math.optimization import newton_optimize, simplex_2d, bisection

    x_opt = newton_optimize(f_prime, f_double_prime, VDR(5), 10)
    x_root = bisection(f, VDR(0), VDR(2), 30)

All iterates exact rational. Convergence rate determined by algorithm,
not arithmetic precision.
"""

from __future__ import annotations
from typing import Callable, Tuple, List

from vdr.core import VDR
from vdr.linalg import Vec, Mat

__all__ = [
    "newton_optimize",
    "newton_root",
    "gradient_descent_2d",
    "simplex_2d",
    "bisection",
    "golden_section",
]


def newton_optimize(f_prime, f_double_prime, x0, n_steps):
    """
    Newton's method for optimization: find x where f'(x) = 0.

    x_{n+1} = x_n - f'(x_n) / f''(x_n)

    I: first derivative callable, second derivative callable,
       starting point x0 (VDR), number of steps
    O: approximate minimizer as VDR, exact at each step

        # minimize x^2 - 4x + 4: f'(x) = 2x-4, f''(x) = 2
        newton_optimize(lambda x: VDR(2)*x - VDR(4),
                       lambda x: VDR(2),
                       VDR(0), 1)
        -> VDR(2)  # converges in 1 step for quadratic
    """
    x = x0
    for _ in range(n_steps):
        fp = f_prime(x)
        fpp = f_double_prime(x)
        if fpp == VDR(0):
            break
        x = x - fp / fpp
    return x


def newton_root(f, f_prime, x0, n_steps):
    """
    Newton's method for root finding: find x where f(x) = 0.

    x_{n+1} = x_n - f(x_n) / f'(x_n)

    I: function callable, derivative callable, start (VDR), steps
    O: approximate root as VDR, exact at each step
    """
    x = x0
    for _ in range(n_steps):
        fx = f(x)
        fpx = f_prime(x)
        if fpx == VDR(0):
            break
        x = x - fx / fpx
    return x


def gradient_descent_2d(grad, x0, y0, lr, steps):
    """
    Gradient descent in 2D.

    I: gradient function (x, y) -> (gx, gy), initial point,
       learning rate lr (VDR), number of steps
    O: (x, y) tuple of VDR after optimization

        gradient_descent_2d(
            lambda x, y: (VDR(2)*x, VDR(2)*y),  # grad of x^2+y^2
            VDR(5), VDR(3), VDR(1, 10), 20
        )
    """
    x, y = x0, y0
    for _ in range(steps):
        gx, gy = grad(x, y)
        x = x - lr * gx
        y = y - lr * gy
    return (x, y)


def simplex_2d(c, A, b):
    """
    Exact rational simplex method for 2D linear programming.

    Minimize c^T x subject to Ax <= b, x >= 0.

    I: cost vector c (list of 2 VDR), constraint matrix A (list of lists),
       constraint RHS b (list of VDR)
    O: optimal solution as Vec, exact

    Simple implementation for small problems. Enumerates vertices.
    """
    # for 2D, enumerate intersection points of constraints
    # and check feasibility

    n_constraints = len(b)
    c0 = c[0] if isinstance(c[0], VDR) else VDR(c[0])
    c1 = c[1] if isinstance(c[1], VDR) else VDR(c[1])

    # candidate vertices: intersections of constraint boundaries
    # plus axis intersections
    candidates = []

    # add origin
    candidates.append((VDR(0), VDR(0)))

    # axis intersections with each constraint
    for i in range(n_constraints):
        a0 = A[i][0] if isinstance(A[i][0], VDR) else VDR(A[i][0])
        a1 = A[i][1] if isinstance(A[i][1], VDR) else VDR(A[i][1])
        bi = b[i] if isinstance(b[i], VDR) else VDR(b[i])

        if a0 != VDR(0):
            candidates.append((bi / a0, VDR(0)))
        if a1 != VDR(0):
            candidates.append((VDR(0), bi / a1))

    # pairwise intersections
    for i in range(n_constraints):
        for j in range(i + 1, n_constraints):
            a_i0 = A[i][0] if isinstance(A[i][0], VDR) else VDR(A[i][0])
            a_i1 = A[i][1] if isinstance(A[i][1], VDR) else VDR(A[i][1])
            bi = b[i] if isinstance(b[i], VDR) else VDR(b[i])

            a_j0 = A[j][0] if isinstance(A[j][0], VDR) else VDR(A[j][0])
            a_j1 = A[j][1] if isinstance(A[j][1], VDR) else VDR(A[j][1])
            bj = b[j] if isinstance(b[j], VDR) else VDR(b[j])

            det = a_i0 * a_j1 - a_i1 * a_j0
            if det != VDR(0):
                x = (bi * a_j1 - bj * a_i1) / det
                y = (a_i0 * bj - a_j0 * bi) / det
                candidates.append((x, y))

    # filter feasible
    def is_feasible(x, y):
        if x < VDR(0) or y < VDR(0):
            return False
        for i in range(n_constraints):
            a0 = A[i][0] if isinstance(A[i][0], VDR) else VDR(A[i][0])
            a1 = A[i][1] if isinstance(A[i][1], VDR) else VDR(A[i][1])
            bi_val = b[i] if isinstance(b[i], VDR) else VDR(b[i])
            if a0 * x + a1 * y > bi_val:
                return False
        return True

    feasible = [(x, y) for x, y in candidates if is_feasible(x, y)]

    if not feasible:
        raise ValueError("No feasible solution")

    # find minimum
    best = feasible[0]
    best_val = c0 * best[0] + c1 * best[1]
    for x, y in feasible[1:]:
        val = c0 * x + c1 * y
        if val < best_val:
            best = (x, y)
            best_val = val

    return Vec([best[0], best[1]])


def bisection(f, a, b, n_steps):
    """
    Bisection method for root finding.

    I: function f (VDR -> VDR), interval [a, b], number of bisection steps
    O: midpoint after n_steps as VDR, exact rational

    Each step halves the interval. After n steps, interval width
    is (b-a) / 2^n.

        bisection(lambda x: x*x - VDR(2), VDR(1), VDR(2), 30)
        # |x^2 - 2| < 10^-8
    """
    a_val = a
    b_val = b
    fa = f(a_val)

    for _ in range(n_steps):
        mid = (a_val + b_val) / VDR(2)
        fmid = f(mid)
        if fmid == VDR(0):
            return mid
        # check sign
        if (fa > VDR(0)) != (fmid > VDR(0)):
            b_val = mid
        else:
            a_val = mid
            fa = fmid

    return (a_val + b_val) / VDR(2)


def golden_section(f, a, b, n_steps):
    """
    Golden section search for unimodal function minimum.

    I: function f (VDR -> VDR), interval [a, b], steps
    O: approximate minimizer as VDR

    Uses rational approximation of golden ratio:
    phi ~ 610/987 (Fibonacci approximation).
    """
    # rational golden ratio approximation
    phi = VDR(610, 987)

    a_val = a
    b_val = b

    for _ in range(n_steps):
        d = b_val - a_val
        x1 = b_val - phi * d
        x2 = a_val + phi * d

        f1 = f(x1)
        f2 = f(x2)

        if f1 < f2:
            b_val = x2
        else:
            a_val = x1

    return (a_val + b_val) / VDR(2)
