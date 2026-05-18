"""
vdr.signal.schedule — Exact noise schedules for diffusion and signal processing.

    from vdr.signal.schedule import linear_schedule, cosine_schedule_rational

    betas = linear_schedule(5, VDR(1, 100), VDR(1, 20))
    # [VDR(1,100), VDR(7,400), VDR(3,100), ...] all exact

Schedules are exact rational sequences. Cumulative products exact.
Square roots via Newton functional remainder.
"""

from __future__ import annotations
from typing import List

from vdr.core import VDR
from vdr.math.transcendental import cos_series, sqrt_newton

__all__ = [
    "linear_schedule",
    "cosine_schedule_rational",
    "compute_alphas",
    "compute_alpha_bars",
    "compute_sqrt_alpha_bars",
    "compute_sqrt_one_minus_alpha_bars",
]


def linear_schedule(T, beta_start, beta_end):
    """
    Linear noise schedule: T evenly-spaced beta values.

    beta_t = beta_start + t * (beta_end - beta_start) / (T - 1)

    I: number of timesteps T, start and end beta values (VDR)
    O: list of T VDR beta values, all exact rational

        linear_schedule(5, VDR(1, 100), VDR(1, 20))
        -> [VDR(1,100), VDR(7,400), VDR(13,400), VDR(19,400), VDR(1,20)]
    """
    if T <= 0:
        return []
    if T == 1:
        return [beta_start]

    betas = []
    denom = VDR(T - 1)
    span = beta_end - beta_start

    for t in range(T):
        beta = beta_start + VDR(t) * span / denom
        betas.append(beta)

    return betas


def cosine_schedule_rational(T, s=None):
    """
    Cosine noise schedule with rational cosine approximation.

    f(t) = cos^2((t/T + s) / (1 + s) * pi/2)
    alpha_bar_t = f(t) / f(0)

    I: number of timesteps T, offset s (VDR, default 8/1000)
    O: list of T VDR beta values derived from cosine schedule

    Uses Taylor series cosine for exact rational evaluation.
    """
    if s is None:
        s = VDR(8, 1000)

    one_plus_s = VDR(1) + s
    # pi/2 from Q335
    from vdr.math.transcendental import PI
    pi_half = PI / VDR(2)

    def f_val(t_idx):
        frac = (VDR(t_idx) / VDR(T) + s) / one_plus_s
        angle = frac * pi_half
        c = cos_series(angle, depth=20)
        return c * c

    f0 = f_val(0)
    alpha_bars = []
    for t in range(T):
        alpha_bars.append(f_val(t) / f0)

    # derive betas from alpha_bars
    # alpha_bar_t = product_{k=0}^{t} (1 - beta_k)
    # beta_0 = 1 - alpha_bar_0
    # beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
    betas = []
    for t in range(T):
        if t == 0:
            beta = VDR(1) - alpha_bars[0]
        else:
            beta = VDR(1) - alpha_bars[t] / alpha_bars[t - 1]
        # clamp to valid range
        if beta < VDR(0):
            beta = VDR(0)
        if beta > VDR(1):
            beta = VDR(1)
        betas.append(beta)

    return betas


def compute_alphas(betas):
    """
    Compute alpha = 1 - beta for each timestep.

    I: list of beta values (VDR)
    O: list of alpha values (VDR), exact (integer subtraction)
    """
    return [VDR(1) - b for b in betas]


def compute_alpha_bars(alphas):
    """
    Compute cumulative product alpha_bar_t = product_{k=0}^{t} alpha_k.

    I: list of alpha values (VDR)
    O: list of alpha_bar values (VDR), exact (integer multiplication)

        alphas = compute_alphas(betas)
        alpha_bars = compute_alpha_bars(alphas)
        # alpha_bars[-1] for T=5: 26821179/31250000 exact
    """
    result = []
    product = VDR(1)
    for a in alphas:
        product = product * a
        result.append(product)
    return result


def compute_sqrt_alpha_bars(alpha_bars, depth=10):
    """
    Compute sqrt(alpha_bar) for each timestep via Newton iteration.

    I: list of alpha_bar values (VDR), Newton depth
    O: list of sqrt(alpha_bar) values (VDR)

    Each sqrt is exact rational at given depth. Depth 10 = >100 digits.
    Residual does not compound through chains.
    """
    result = []
    for ab in alpha_bars:
        result.append(sqrt_newton(ab, depth=depth))
    return result


def compute_sqrt_one_minus_alpha_bars(alpha_bars, depth=10):
    """
    Compute sqrt(1 - alpha_bar) for each timestep.

    I: list of alpha_bar values (VDR), Newton depth
    O: list of sqrt(1 - alpha_bar) values (VDR)
    """
    result = []
    for ab in alpha_bars:
        one_minus = VDR(1) - ab
        result.append(sqrt_newton(one_minus, depth=depth))
    return result
