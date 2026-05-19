<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/sampling.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `diffusion.sampling`
vdr.diffusion.sampling — Verification and testing utilities for diffusion. 

 from vdr.diffusion.sampling import verify_forward_reverse_roundtrip  from vdr.diffusion.sampling import verify_multi_step_drift 

 err = verify_forward_reverse_roundtrip(x0, schedule, epsilon)  # err < 1e-50 or exactly 0 for DDIM 

 drift = verify_multi_step_drift(x0, schedule, epsilon, num_cycles=10)  # drift does NOT grow across cycles — central result 

Oracle predictor separates arithmetic error from model error. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/sampling.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `verify_schedule_consistency`

```python
verify_schedule_consistency(schedule)
```

Verify internal consistency of diffusion schedule. 

Checks: 
- alpha = 1 - beta for all t 
- alpha_bar = cumulative product of alphas 
- all betas in (0, 1) 

I: DiffusionSchedule O: bool, True if all checks pass 

 assert verify_schedule_consistency(schedule) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/sampling.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `verify_snr_monotonic`

```python
verify_snr_monotonic(schedule)
```

Verify signal-to-noise ratio is strictly decreasing. 

SNR_t = alpha_bar_t / (1 - alpha_bar_t) 

I: DiffusionSchedule O: bool, True if SNR is strictly decreasing 

Exact rational comparison — no float ambiguity. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/sampling.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `verify_coefficient_identity`

```python
verify_coefficient_identity(schedule)
```

Verify (sqrt(alpha_bar))^2 + (sqrt(1-alpha_bar))^2 ≈ 1 for all t. 

Due to Newton sqrt approximation, residual is nonzero but tiny (< 1e-50 at depth 10). 

I: DiffusionSchedule O: list of residual VDR values (one per timestep) 

 residuals = verify_coefficient_identity(schedule)  all(abs(r) < VDR(1, 10**20) for r in residuals) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/sampling.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_oracle_predictor`

```python
make_oracle_predictor(x0, schedule)
```

Create a perfect noise predictor (oracle) for testing. 

Given x0 and schedule, the oracle knows the exact noise at each timestep. Using this predictor, roundtrip error measures only arithmetic error. 

I: original signal x0 (Vec), DiffusionSchedule O: callable predict(xt, t) -> epsilon_pred 

With oracle, compute_x0_prediction gives error = exactly 0. This separates arithmetic error from model error. 

 oracle = make_oracle_predictor(x0, schedule)  eps_pred = oracle(xt, t) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/sampling.py#L159"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `roundtrip_error`

```python
roundtrip_error(x0, recovered)
```

Compute roundtrip error as sum of squared differences. 

I: original x0 (Vec), recovered signal (Vec) O: error as VDR (sum of (x0_i - recovered_i)^2) 

For perfect roundtrip: exactly 0. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/sampling.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `verify_forward_reverse_roundtrip`

```python
verify_forward_reverse_roundtrip(x0, schedule, epsilon)
```

Verify forward-reverse roundtrip error. 

1. Forward: x_T = sqrt(ab_T) * x0 + sqrt(1-ab_T) * epsilon 2. Oracle predictor from x0 3. Reverse loop from x_T to recovered x0 4. Measure error 

I: original x0 (Vec), schedule, noise epsilon (Vec) O: error as VDR. Should be < 1e-50 (Newton residual only). 

 err = verify_forward_reverse_roundtrip(x0, schedule, epsilon) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/sampling.py#L197"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `verify_ddim_roundtrip`

```python
verify_ddim_roundtrip(x0, schedule, epsilon)
```

Verify DDIM deterministic roundtrip. 

Forward to x_T, then DDIM reverse to x_0. Error should be exactly 0 (strongest result). 

I: original x0 (Vec), schedule, noise epsilon (Vec) O: error as VDR 

 err = verify_ddim_roundtrip(x0, schedule, epsilon)  # err == VDR(0) for DDIM with oracle 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/sampling.py#L226"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `verify_multi_step_drift`

```python
verify_multi_step_drift(x0, schedule, epsilon, num_cycles=3)
```

Verify that drift does NOT grow across multiple forward-reverse cycles. 

Each cycle: forward x0 -> xT, reverse xT -> x0_recovered. The next cycle starts from x0_recovered. 

Central result of VDR-26: error at cycle N = error at cycle 1. Newton residual is fixed. Rational ops contribute zero error. Chain length irrelevant. 

I: original x0 (Vec), schedule, noise epsilon (Vec), number of cycles O: list of per-cycle error VDR values 

All errors should be approximately equal (bounded by Newton residual). Float: errors grow linearly. VDR: constant. 

 drift = verify_multi_step_drift(x0, schedule, epsilon, num_cycles=10)  # drift[0] ≈ drift[9] — does not grow 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
