<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/schedule.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `diffusion.schedule`
vdr.diffusion.schedule — Exact noise schedules for diffusion models. 

 from vdr.diffusion.schedule import DiffusionSchedule, linear_schedule 

 schedule = linear_schedule(T=5, beta_start=VDR(1,100), beta_end=VDR(1,20))  print(schedule.alpha_bars[-1])  # 26821179/31250000 exact 

All betas, alphas, alpha_bars are exact rationals. Cumulative products are exact integer multiplication. Square roots via Newton functional remainder. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/schedule.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `exact_sqrt`

```python
exact_sqrt(a, depth=10)
```

Newton sqrt with caching. 

Cache key is the normalized VDR value. Avoids recomputing sqrt for schedule values used at every step. 

I: value a (VDR), Newton depth O: exact rational approximation of sqrt(a) 

 exact_sqrt(VDR(1, 2), depth=10) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/schedule.py#L120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `linear_schedule`

```python
linear_schedule(T, beta_start, beta_end)
```

Linear noise schedule. 

beta_t = beta_start + t * (beta_end - beta_start) / (T - 1) 

I: number of timesteps T, start and end beta (VDR) O: DiffusionSchedule with exact rational betas 

 schedule = linear_schedule(5, VDR(1, 100), VDR(1, 20)) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/schedule.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `cosine_schedule`

```python
cosine_schedule(T, s=None, sqrt_depth=10)
```

Cosine noise schedule with exact rational cosine approximation. 

f(t) = cos^2((t/T + s) / (1 + s) * pi/2) alpha_bar_t = f(t) / f(0) beta_t = 1 - alpha_bar_t / alpha_bar_{t-1} 

I: timesteps T, offset s (VDR, default 8/1000), sqrt depth O: DiffusionSchedule 

 schedule = cosine_schedule(10, s=VDR(8, 1000)) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/schedule.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DiffusionSchedule`
Precomputed noise schedule for diffusion models. 

Stores betas, alphas, alpha_bars, and their square roots. All values exact VDR rationals. Square roots via Newton at chosen depth. 

 schedule = DiffusionSchedule(betas, sqrt_depth=10)  schedule.sqrt_alpha_bars[t]       # sqrt(alpha_bar_t)  schedule.sqrt_one_minus_alpha_bars[t]  # sqrt(1 - alpha_bar_t) 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/schedule.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(betas, sqrt_depth=10)
```

I: list of beta values (VDR), Newton depth for sqrt 




---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/schedule.py#L98"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `posterior_variance`

```python
posterior_variance(t)
```

Posterior variance beta_tilde_t for reverse process. 

beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) 

For t=0, returns beta_0 (no previous alpha_bar). 

I: timestep t (0-indexed) O: posterior variance as VDR, exact closed positive rational 

 var = schedule.posterior_variance(3) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
