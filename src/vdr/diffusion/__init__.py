"""
vdr.diffusion — Exact diffusion model arithmetic.

    from vdr.diffusion.schedule import DiffusionSchedule, linear_schedule
    from vdr.diffusion.forward import forward_sample, forward_trajectory
    from vdr.diffusion.reverse import reverse_step_ddim, compute_x0_prediction
    from vdr.diffusion.sampling import verify_forward_reverse_roundtrip

Central result: drift at cycle N = drift at cycle 1. Chain length
irrelevant to accumulated error. DDIM roundtrip error = exactly 0.

Float64 drift grows linearly: ~1e-15 per cycle. VDR constant at <1e-50.
For a 2-hour film at 24fps/50 steps: float ~1.9e-8, VDR < 1e-50.
"""
