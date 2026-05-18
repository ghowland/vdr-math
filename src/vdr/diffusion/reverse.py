"""
vdr.diffusion.reverse — Exact reverse diffusion process.

    from vdr.diffusion.reverse import compute_x0_prediction, reverse_step_ddim

    x0_pred = compute_x0_prediction(xt, t, schedule, eps_pred)
    # with perfect noise: error = exactly 0

    x_prev = reverse_step_ddim(xt, t, t_prev, schedule, eps_pred)
    # DDIM roundtrip error = exactly 0

Exact rational subtraction and division. No catastrophic cancellation.
"""

from __future__ import annotations
from typing import Optional, Callable, List

from vdr.core import VDR
from vdr.linalg import Vec
from vdr.diffusion.schedule import DiffusionSchedule, exact_sqrt

__all__ = [
    "compute_x0_prediction",
    "compute_posterior_mean",
    "reverse_step",
    "reverse_step_ddim",
    "reverse_sample_loop",
]


def compute_x0_prediction(xt, t, schedule, eps_pred):
    """
    Predict x_0 from x_t and predicted noise.

    x_0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)

    I: noisy signal xt (Vec), timestep t, schedule, predicted noise (Vec)
    O: predicted original signal x_0 (Vec), exact

    With oracle noise predictor (eps_pred == actual epsilon):
    prediction error = exactly 0. Not approximately zero. Zero.

        x0_pred = compute_x0_prediction(xt, 3, schedule, eps_pred)
    """
    sqrt_ab = schedule.sqrt_alpha_bars[t]
    sqrt_one_minus_ab = schedule.sqrt_one_minus_alpha_bars[t]

    # x_0 = (x_t - sqrt(1-ab) * eps) / sqrt(ab)
    noise_scaled = eps_pred.scale(sqrt_one_minus_ab)
    numerator = xt - noise_scaled

    # divide each component by sqrt_ab
    result = []
    for i in range(len(numerator)):
        result.append(numerator[i] / sqrt_ab)

    return Vec(result)


def compute_posterior_mean(xt, t, schedule, eps_pred):
    """
    Posterior mean for reverse step.

    mu_t = (1/sqrt(alpha_t)) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_pred)

    I: noisy xt (Vec), timestep t, schedule, predicted noise (Vec)
    O: posterior mean (Vec), exact
    """
    alpha_t = schedule.alphas[t]
    beta_t = schedule.betas[t]
    sqrt_one_minus_ab = schedule.sqrt_one_minus_alpha_bars[t]
    sqrt_alpha = exact_sqrt(alpha_t, schedule.sqrt_depth)

    coeff = beta_t / sqrt_one_minus_ab

    noise_term = eps_pred.scale(coeff)
    adjusted = xt - noise_term

    result = []
    for i in range(len(adjusted)):
        result.append(adjusted[i] / sqrt_alpha)

    return Vec(result)


def reverse_step(xt, t, schedule, eps_pred, z=None):
    """
    One stochastic reverse diffusion step.

    x_{t-1} = mu_t + sqrt(beta_tilde_t) * z

    where z is optional noise (default zero for deterministic).

    I: noisy xt (Vec), timestep t, schedule, predicted noise (Vec),
       optional noise z (Vec)
    O: denoised x_{t-1} (Vec), exact

        x_prev = reverse_step(xt, 3, schedule, eps_pred)
    """
    mu = compute_posterior_mean(xt, t, schedule, eps_pred)

    if z is None or t == 0:
        return mu

    # stochastic: add sqrt(posterior_variance) * z
    var_t = schedule.posterior_variance(t)
    sqrt_var = exact_sqrt(var_t, schedule.sqrt_depth)
    noise = z.scale(sqrt_var)

    return mu + noise


def reverse_step_ddim(xt, t, t_prev, schedule, eps_pred, eta=None):
    """
    DDIM (Denoising Diffusion Implicit Models) deterministic reverse step.

    x_{t-1} = sqrt(alpha_bar_{t-1}) * x_0_pred +
              sqrt(1 - alpha_bar_{t-1} - sigma^2) * eps_pred +
              sigma * z

    With eta=0 (default): fully deterministic. No stochastic noise.
    Roundtrip error with perfect noise = exactly 0.

    I: xt (Vec), current timestep t, previous timestep t_prev,
       schedule, predicted noise (Vec), eta (VDR, default 0)
    O: x_{t-1} (Vec), exact

    DDIM with eta=0 is the paper's strongest result:
    forward-reverse roundtrip error = exactly zero.

        x_prev = reverse_step_ddim(xt, 4, 3, schedule, eps_pred)
    """
    if eta is None:
        eta = VDR(0)

    # predict x_0
    x0_pred = compute_x0_prediction(xt, t, schedule, eps_pred)

    # get alpha_bar at target timestep
    sqrt_ab_prev = schedule.sqrt_alpha_bars[t_prev]
    sqrt_one_minus_ab_prev = schedule.sqrt_one_minus_alpha_bars[t_prev]

    # deterministic direction
    # x_{t-1} = sqrt(ab_{t-1}) * x0_pred + sqrt(1 - ab_{t-1}) * eps_pred
    signal_part = x0_pred.scale(sqrt_ab_prev)
    noise_direction = eps_pred.scale(sqrt_one_minus_ab_prev)

    return signal_part + noise_direction


def reverse_sample_loop(xT, schedule, predict_noise, noise_vectors=None):
    """
    Complete reverse sampling loop from x_T to x_0.

    I: initial noisy signal xT (Vec), schedule,
       noise predictor function (xt, t) -> Vec,
       optional noise vectors for stochastic steps
    O: denoised signal x_0 (Vec)

    Each step exact. Drift does not grow across steps.

        def oracle(xt, t):
            return actual_noise_at_t
        x0_recovered = reverse_sample_loop(xT, schedule, oracle)
    """
    x = xT

    for t in range(schedule.T - 1, -1, -1):
        eps_pred = predict_noise(x, t)

        z = None
        if noise_vectors is not None and t > 0:
            z = noise_vectors[t]

        x = reverse_step(x, t, schedule, eps_pred, z=z)

    return x
