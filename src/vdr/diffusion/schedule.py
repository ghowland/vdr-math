"""
vdr.diffusion.schedule — Exact noise schedules for diffusion models.

    from vdr.diffusion.schedule import DiffusionSchedule, linear_schedule

    schedule = linear_schedule(T=5, beta_start=VDR(1,100), beta_end=VDR(1,20))
    print(schedule.alpha_bars[-1])  # 26821179/31250000 exact

All betas, alphas, alpha_bars are exact rationals.
Cumulative products are exact integer multiplication.
Square roots via Newton functional remainder.
"""

from __future__ import annotations
from typing import List, Optional

from vdr.core import VDR
from vdr.math.transcendental import sqrt_newton, cos_series, PI

__all__ = [
    "DiffusionSchedule",
    "linear_schedule",
    "cosine_schedule",
    "exact_sqrt",
]


# ---------------------------------------------------------------------------
# Cached sqrt
# ---------------------------------------------------------------------------

_sqrt_cache = {}


def exact_sqrt(a, depth=10):
    """
    Newton sqrt with caching.

    Cache key is the normalized VDR value. Avoids recomputing sqrt
    for schedule values used at every step.

    I: value a (VDR), Newton depth
    O: exact rational approximation of sqrt(a)

        exact_sqrt(VDR(1, 2), depth=10)
    """
    key = (a.v, a.d, depth)
    if key in _sqrt_cache:
        return _sqrt_cache[key]

    result = sqrt_newton(a, depth=depth)
    _sqrt_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# DiffusionSchedule
# ---------------------------------------------------------------------------

class DiffusionSchedule:
    """
    Precomputed noise schedule for diffusion models.

    Stores betas, alphas, alpha_bars, and their square roots.
    All values exact VDR rationals. Square roots via Newton at chosen depth.

        schedule = DiffusionSchedule(betas, sqrt_depth=10)
        schedule.sqrt_alpha_bars[t]       # sqrt(alpha_bar_t)
        schedule.sqrt_one_minus_alpha_bars[t]  # sqrt(1 - alpha_bar_t)
    """

    def __init__(self, betas, sqrt_depth=10):
        """
        I: list of beta values (VDR), Newton depth for sqrt
        """
        self.betas = list(betas)
        self.T = len(betas)
        self.sqrt_depth = sqrt_depth

        # alpha_t = 1 - beta_t
        self.alphas = [VDR(1) - b for b in self.betas]

        # alpha_bar_t = cumulative product of alphas
        self.alpha_bars = []
        product = VDR(1)
        for a in self.alphas:
            product = product * a
            self.alpha_bars.append(product)

        # precompute sqrt values
        self.sqrt_alpha_bars = [
            exact_sqrt(ab, sqrt_depth) for ab in self.alpha_bars
        ]
        self.sqrt_one_minus_alpha_bars = [
            exact_sqrt(VDR(1) - ab, sqrt_depth) for ab in self.alpha_bars
        ]

    def posterior_variance(self, t):
        """
        Posterior variance beta_tilde_t for reverse process.

        beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)

        For t=0, returns beta_0 (no previous alpha_bar).

        I: timestep t (0-indexed)
        O: posterior variance as VDR, exact closed positive rational

            var = schedule.posterior_variance(3)
        """
        if t == 0:
            return self.betas[0]

        one_minus_ab_prev = VDR(1) - self.alpha_bars[t - 1]
        one_minus_ab = VDR(1) - self.alpha_bars[t]

        return self.betas[t] * one_minus_ab_prev / one_minus_ab


def linear_schedule(T, beta_start, beta_end):
    """
    Linear noise schedule.

    beta_t = beta_start + t * (beta_end - beta_start) / (T - 1)

    I: number of timesteps T, start and end beta (VDR)
    O: DiffusionSchedule with exact rational betas

        schedule = linear_schedule(5, VDR(1, 100), VDR(1, 20))
    """
    if T <= 0:
        return DiffusionSchedule([])
    if T == 1:
        return DiffusionSchedule([beta_start])

    betas = []
    span = beta_end - beta_start
    denom = VDR(T - 1)

    for t in range(T):
        beta = beta_start + VDR(t) * span / denom
        betas.append(beta)

    return DiffusionSchedule(betas)


def cosine_schedule(T, s=None, sqrt_depth=10):
    """
    Cosine noise schedule with exact rational cosine approximation.

    f(t) = cos^2((t/T + s) / (1 + s) * pi/2)
    alpha_bar_t = f(t) / f(0)
    beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}

    I: timesteps T, offset s (VDR, default 8/1000), sqrt depth
    O: DiffusionSchedule

        schedule = cosine_schedule(10, s=VDR(8, 1000))
    """
    if s is None:
        s = VDR(8, 1000)

    one_plus_s = VDR(1) + s
    pi_half = PI / VDR(2)

    def f_val(t_idx):
        frac = (VDR(t_idx) / VDR(T) + s) / one_plus_s
        angle = frac * pi_half
        c = cos_series(angle, depth=20)
        return c * c

    f0 = f_val(0)
    alpha_bars = [f_val(t) / f0 for t in range(T)]

    # derive betas
    betas = []
    for t in range(T):
        if t == 0:
            beta = VDR(1) - alpha_bars[0]
        else:
            beta = VDR(1) - alpha_bars[t] / alpha_bars[t - 1]
        # clamp
        if beta < VDR(0):
            beta = VDR(0)
        if beta > VDR(1):
            beta = VDR(1)
        betas.append(beta)

    return DiffusionSchedule(betas, sqrt_depth=sqrt_depth)
