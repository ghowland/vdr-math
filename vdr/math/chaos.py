"""
vdr.math.chaos — Exact chaotic map iteration and period detection.

    from vdr.math.chaos import tent_map, iterate_map, detect_period

    orbit = iterate_map(tent_map, VDR(1, 7), 20)
    period = detect_period(orbit)  # 3, exact forever

Periodic rational orbits are computationally free (bounded denominators).
Chaotic orbits have exponential denominator growth — information-theoretic,
not a VDR defect. Float hides this cost by silent truncation.
"""

from __future__ import annotations
from typing import Callable, List, Optional

from vdr.core import VDR

__all__ = [
    "tent_map",
    "bernoulli_shift",
    "arnold_cat",
    "logistic_map",
    "iterate_map",
    "detect_period",
    "lyapunov_product",
]


def tent_map(x):
    """
    Tent map T(x) on [0, 1].

    T(x) = 2x       if x < 1/2
    T(x) = 2(1-x)   if x >= 1/2

    I: VDR x in [0, 1]
    O: T(x) as VDR, exact

    On 1/7: period 3 exact forever while float diverges at ~25 steps.

        tent_map(VDR(1, 7)) -> VDR(2, 7)
        tent_map(VDR(2, 7)) -> VDR(4, 7)
        tent_map(VDR(4, 7)) -> VDR(6, 7) -> VDR(2, 7) ... period 3
    """
    half = VDR(1, 2)
    if x < half:
        return VDR(2) * x
    else:
        return VDR(2) * (VDR(1) - x)


def bernoulli_shift(x):
    """
    Bernoulli shift (doubling map): B(x) = 2x mod 1.

    I: VDR x in [0, 1)
    O: B(x) as VDR, exact

    For rational x = p/q, orbit is periodic with period dividing
    the multiplicative order of 2 modulo q.

        bernoulli_shift(VDR(1, 3)) -> VDR(2, 3)
        bernoulli_shift(VDR(2, 3)) -> VDR(1, 3)  # period 2
    """
    doubled = VDR(2) * x
    # mod 1: subtract integer part
    frac = doubled.to_fraction()
    int_part = int(frac)
    return doubled - VDR(int_part)


def arnold_cat(x, y):
    """
    Arnold cat map on the torus [0,1) x [0,1).

    (x', y') = ((2x + y) mod 1, (x + y) mod 1)

    I: VDR x, y in [0, 1)
    O: (x', y') as tuple of VDR, exact

    On (1/7, 3/11): period 40 exact.

        arnold_cat(VDR(1, 7), VDR(3, 11))
    """
    new_x = VDR(2) * x + y
    new_y = x + y

    # mod 1
    fx = new_x.to_fraction()
    fy = new_y.to_fraction()
    new_x = new_x - VDR(int(fx))
    new_y = new_y - VDR(int(fy))

    return (new_x, new_y)


def logistic_map(x, r):
    """
    Logistic map: L(x) = r * x * (1 - x).

    I: VDR x in [0, 1], parameter r (VDR)
    O: L(x) as VDR, exact

    NOTE: In flat Fraction representation, denominator digits grow as ~2^n
    at r=4. In VDR with Q335 fixed-frame, D stays at 2^335 and tree depth
    grows by 1 per step. Cost is tree depth, not denominator explosion.
    Periodic orbits have bounded denominators and zero growth.

    Practical for ~10-15 steps at r=4. Unlimited for non-chaotic r.

        logistic_map(VDR(1, 3), VDR(4)) -> VDR(8, 9)
    """
    return r * x * (VDR(1) - x)


def iterate_map(f, x0, steps):
    """
    Iterate a map and collect the orbit.

    I: map function f (VDR -> VDR or tuple -> tuple),
       initial point x0, number of steps
    O: list [x0, f(x0), f(f(x0)), ...] of length steps+1

        orbit = iterate_map(tent_map, VDR(1, 7), 20)
    """
    orbit = [x0]
    x = x0
    for _ in range(steps):
        x = f(x)
        orbit.append(x)
    return orbit


def iterate_map_2d(f, x0, y0, steps):
    """
    Iterate a 2D map and collect the orbit.

    I: map function f(x, y) -> (x', y'), initial point, steps
    O: list of (x, y) tuples

        orbit = iterate_map_2d(arnold_cat, VDR(1,7), VDR(3,11), 50)
    """
    orbit = [(x0, y0)]
    x, y = x0, y0
    for _ in range(steps):
        x, y = f(x, y)
        orbit.append((x, y))
    return orbit


def detect_period(orbit):
    """
    Detect period in an exact orbit.

    I: list of VDR values (exact, so equality testing is reliable)
    O: period as int, or None if no period found within orbit length

    Checks all pairs: finds smallest p > 0 where orbit[i+p] == orbit[i]
    for some i.

        orbit = iterate_map(tent_map, VDR(1, 7), 20)
        detect_period(orbit) -> 3
    """
    if len(orbit) < 2:
        return None

    n = len(orbit)
    # try each period length
    for p in range(1, n):
        # check if orbit[i] == orbit[i+p] for some valid i
        found = False
        for i in range(n - p):
            if orbit[i] == orbit[i + p]:
                # verify it holds for subsequent steps too
                valid = True
                for j in range(i, min(i + p, n - p)):
                    if orbit[j] != orbit[j + p]:
                        valid = False
                        break
                if valid:
                    return p
    return None


def detect_period_2d(orbit):
    """
    Detect period in a 2D exact orbit.

    I: list of (VDR, VDR) tuples
    O: period as int, or None
    """
    if len(orbit) < 2:
        return None

    x0, y0 = orbit[0]
    for i in range(1, len(orbit)):
        xi, yi = orbit[i]
        if xi == x0 and yi == y0:
            return i

    return None


def lyapunov_product(derivatives):
    """
    Product of derivative magnitudes along an orbit.

    For a 1D map f, the Lyapunov exponent is:
        lambda = (1/N) * ln(product |f'(x_i)|)

    This function computes the product exactly.
    The logarithm is taken externally if needed.

    I: list of |f'(x_i)| values as VDR
    O: product as VDR, exact

        # tent map: |f'| = 2 everywhere
        lyapunov_product([VDR(2)] * 20) -> VDR(1048576)  # 2^20
    """
    product = VDR(1)
    for d in derivatives:
        product = product * d
    return product
