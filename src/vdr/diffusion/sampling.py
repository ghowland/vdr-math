"""
vdr.diffusion.sampling — Verification and testing utilities for diffusion.

    from vdr.diffusion.sampling import verify_forward_reverse_roundtrip
    from vdr.diffusion.sampling import verify_multi_step_drift

    err = verify_forward_reverse_roundtrip(x0, schedule, epsilon)
    # err < 1e-50 or exactly 0 for DDIM

    drift = verify_multi_step_drift(x0, schedule, epsilon, num_cycles=10)
    # drift does NOT grow across cycles — central result

Oracle predictor separates arithmetic error from model error.
"""

from __future__ import annotations
from typing import Callable, List

from vdr.core import VDR
from vdr.linalg import Vec
from vdr.diffusion.schedule import DiffusionSchedule
from vdr.diffusion.forward import forward_sample
from vdr.diffusion.reverse import (
    compute_x0_prediction,
    reverse_step,
    reverse_step_ddim,
    reverse_sample_loop,
)

__all__ = [
    "verify_schedule_consistency",
    "verify_snr_monotonic",
    "verify_coefficient_identity",
    "make_oracle_predictor",
    "verify_forward_reverse_roundtrip",
    "verify_ddim_roundtrip",
    "verify_multi_step_drift",
    "roundtrip_error",
]


def verify_schedule_consistency(schedule):
    """
    Verify internal consistency of diffusion schedule.

    Checks:
    - alpha = 1 - beta for all t
    - alpha_bar = cumulative product of alphas
    - all betas in (0, 1)

    I: DiffusionSchedule
    O: bool, True if all checks pass

        assert verify_schedule_consistency(schedule)
    """
    for t in range(schedule.T):
        # alpha = 1 - beta
        if schedule.alphas[t] != VDR(1) - schedule.betas[t]:
            return False

        # beta in (0, 1)
        if schedule.betas[t] <= VDR(0) or schedule.betas[t] >= VDR(1):
            return False

    # alpha_bar = cumulative product
    product = VDR(1)
    for t in range(schedule.T):
        product = product * schedule.alphas[t]
        if schedule.alpha_bars[t] != product:
            return False

    return True


def verify_snr_monotonic(schedule):
    """
    Verify signal-to-noise ratio is strictly decreasing.

    SNR_t = alpha_bar_t / (1 - alpha_bar_t)

    I: DiffusionSchedule
    O: bool, True if SNR is strictly decreasing

    Exact rational comparison — no float ambiguity.
    """
    for t in range(1, schedule.T):
        ab_curr = schedule.alpha_bars[t]
        ab_prev = schedule.alpha_bars[t - 1]

        snr_curr = ab_curr / (VDR(1) - ab_curr)
        snr_prev = ab_prev / (VDR(1) - ab_prev)

        if snr_curr >= snr_prev:
            return False

    return True


def verify_coefficient_identity(schedule):
    """
    Verify (sqrt(alpha_bar))^2 + (sqrt(1-alpha_bar))^2 ≈ 1 for all t.

    Due to Newton sqrt approximation, residual is nonzero but tiny
    (< 1e-50 at depth 10).

    I: DiffusionSchedule
    O: list of residual VDR values (one per timestep)

        residuals = verify_coefficient_identity(schedule)
        all(abs(r) < VDR(1, 10**20) for r in residuals)
    """
    residuals = []
    for t in range(schedule.T):
        s1 = schedule.sqrt_alpha_bars[t]
        s2 = schedule.sqrt_one_minus_alpha_bars[t]
        identity = s1 * s1 + s2 * s2
        residual = identity - VDR(1)
        residuals.append(residual)
    return residuals


def make_oracle_predictor(x0, schedule):
    """
    Create a perfect noise predictor (oracle) for testing.

    Given x0 and schedule, the oracle knows the exact noise at each timestep.
    Using this predictor, roundtrip error measures only arithmetic error.

    I: original signal x0 (Vec), DiffusionSchedule
    O: callable predict(xt, t) -> epsilon_pred

    With oracle, compute_x0_prediction gives error = exactly 0.
    This separates arithmetic error from model error.

        oracle = make_oracle_predictor(x0, schedule)
        eps_pred = oracle(xt, t)
    """
    def predict(xt, t):
        """
        Recover the exact noise from xt and x0.

        epsilon = (xt - sqrt(alpha_bar_t) * x0) / sqrt(1 - alpha_bar_t)
        """
        sqrt_ab = schedule.sqrt_alpha_bars[t]
        sqrt_one_minus_ab = schedule.sqrt_one_minus_alpha_bars[t]

        signal_part = x0.scale(sqrt_ab)
        noise_unnormed = xt - signal_part

        result = []
        for i in range(len(noise_unnormed)):
            result.append(noise_unnormed[i] / sqrt_one_minus_ab)

        return Vec(result)

    return predict


def roundtrip_error(x0, recovered):
    """
    Compute roundtrip error as sum of squared differences.

    I: original x0 (Vec), recovered signal (Vec)
    O: error as VDR (sum of (x0_i - recovered_i)^2)

    For perfect roundtrip: exactly 0.
    """
    total = VDR(0)
    for i in range(len(x0)):
        diff = x0[i] - recovered[i]
        total = total + diff * diff
    return total


def verify_forward_reverse_roundtrip(x0, schedule, epsilon):
    """
    Verify forward-reverse roundtrip error.

    1. Forward: x_T = sqrt(ab_T) * x0 + sqrt(1-ab_T) * epsilon
    2. Oracle predictor from x0
    3. Reverse loop from x_T to recovered x0
    4. Measure error

    I: original x0 (Vec), schedule, noise epsilon (Vec)
    O: error as VDR. Should be < 1e-50 (Newton residual only).

        err = verify_forward_reverse_roundtrip(x0, schedule, epsilon)
    """
    T = schedule.T - 1
    xT = forward_sample(x0, T, schedule, epsilon)
    oracle = make_oracle_predictor(x0, schedule)
    recovered = reverse_sample_loop(xT, schedule, oracle)

    return roundtrip_error(x0, recovered)


def verify_ddim_roundtrip(x0, schedule, epsilon):
    """
    Verify DDIM deterministic roundtrip.

    Forward to x_T, then DDIM reverse to x_0.
    Error should be exactly 0 (strongest result).

    I: original x0 (Vec), schedule, noise epsilon (Vec)
    O: error as VDR

        err = verify_ddim_roundtrip(x0, schedule, epsilon)
        # err == VDR(0) for DDIM with oracle
    """
    T = schedule.T - 1
    xT = forward_sample(x0, T, schedule, epsilon)
    oracle = make_oracle_predictor(x0, schedule)

    x = xT
    for t in range(T, 0, -1):
        eps_pred = oracle(x, t)
        x = reverse_step_ddim(x, t, t - 1, schedule, eps_pred)

    # final step
    eps_pred = oracle(x, 0)
    x0_pred = compute_x0_prediction(x, 0, schedule, eps_pred)

    return roundtrip_error(x0, x0_pred)


def verify_multi_step_drift(x0, schedule, epsilon, num_cycles=3):
    """
    Verify that drift does NOT grow across multiple forward-reverse cycles.

    Each cycle: forward x0 -> xT, reverse xT -> x0_recovered.
    The next cycle starts from x0_recovered.

    Central result of VDR-26: error at cycle N = error at cycle 1.
    Newton residual is fixed. Rational ops contribute zero error.
    Chain length irrelevant.

    I: original x0 (Vec), schedule, noise epsilon (Vec), number of cycles
    O: list of per-cycle error VDR values

    All errors should be approximately equal (bounded by Newton residual).
    Float: errors grow linearly. VDR: constant.

        drift = verify_multi_step_drift(x0, schedule, epsilon, num_cycles=10)
        # drift[0] ≈ drift[9] — does not grow
    """
    errors = []
    current = x0

    T = schedule.T - 1

    for cycle in range(num_cycles):
        # forward
        xT = forward_sample(current, T, schedule, epsilon)

        # oracle from current (not original x0 — tests actual drift)
        oracle = make_oracle_predictor(current, schedule)

        # reverse
        recovered = reverse_sample_loop(xT, schedule, oracle)

        # measure error against this cycle's input
        err = roundtrip_error(current, recovered)
        errors.append(err)

        # next cycle starts from recovered
        current = recovered

    return errors
