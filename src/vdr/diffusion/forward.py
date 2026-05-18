"""
vdr.diffusion.forward — Exact forward diffusion process.

    from vdr.diffusion.forward import forward_sample, forward_trajectory

    xt = forward_sample(x0, t, schedule, epsilon)
    # xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon

Exact rational scaling and addition. No accumulation error.
"""

from __future__ import annotations
from typing import List

from vdr.core import VDR
from vdr.linalg import Vec

__all__ = [
    "forward_sample",
    "forward_sample_step",
    "forward_trajectory",
]


def forward_sample(x0, t, schedule, epsilon):
    """
    Forward diffusion: sample x_t from x_0.

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

    I: original signal x0 (Vec), timestep t (int), DiffusionSchedule,
       noise epsilon (Vec)
    O: noisy signal x_t (Vec), exact

    Two exact rational scalings and one exact addition.

        x0 = Vec.from_ints([1, 2, 3])
        epsilon = Vec.from_ints([0, 0, 0])
        xt = forward_sample(x0, 2, schedule, epsilon)
    """
    sqrt_ab = schedule.sqrt_alpha_bars[t]
    sqrt_one_minus_ab = schedule.sqrt_one_minus_alpha_bars[t]

    # x_t = sqrt_ab * x0 + sqrt_one_minus_ab * epsilon
    signal_part = x0.scale(sqrt_ab)
    noise_part = epsilon.scale(sqrt_one_minus_ab)

    return signal_part + noise_part


def forward_sample_step(x_prev, t, schedule, epsilon):
    """
    One-step forward diffusion from x_{t-1} to x_t.

    x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon

    I: previous signal x_{t-1} (Vec), timestep t, schedule, noise (Vec)
    O: next noisy signal x_t (Vec), exact
    """
    from vdr.diffusion.schedule import exact_sqrt

    alpha_t = schedule.alphas[t]
    sqrt_alpha = exact_sqrt(alpha_t, schedule.sqrt_depth)
    sqrt_one_minus_alpha = exact_sqrt(VDR(1) - alpha_t, schedule.sqrt_depth)

    return x_prev.scale(sqrt_alpha) + epsilon.scale(sqrt_one_minus_alpha)


def forward_trajectory(x0, schedule, epsilons):
    """
    Complete forward diffusion trajectory.

    Produces [x_0, x_1, ..., x_T] where each x_t is computed
    directly from x_0 (not chained), so no sequential error.

    I: original x0 (Vec), DiffusionSchedule,
       list of noise vectors [epsilon_0, ..., epsilon_{T-1}] (list of Vec)
    O: list of Vec [x_0, x_1, ..., x_T], length T+1

    x_0 is identity (the original signal).
    x_t for t >= 1 computed via forward_sample.

        trajectory = forward_trajectory(x0, schedule, epsilons)
        assert trajectory[0] == x0  # identity, exact
    """
    trajectory = [x0]

    for t in range(schedule.T):
        xt = forward_sample(x0, t, schedule, epsilons[t])
        trajectory.append(xt)

    return trajectory
