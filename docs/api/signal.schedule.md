<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/schedule.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `signal.schedule`
vdr.signal.schedule — Exact noise schedules for diffusion and signal processing. 

 from vdr.signal.schedule import linear_schedule, cosine_schedule_rational 

 betas = linear_schedule(5, VDR(1, 100), VDR(1, 20))  # [VDR(1,100), VDR(7,400), VDR(3,100), ...] all exact 

Schedules are exact rational sequences. Cumulative products exact. Square roots via Newton functional remainder. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/schedule.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `linear_schedule`

```python
linear_schedule(T, beta_start, beta_end)
```

Linear noise schedule: T evenly-spaced beta values. 

beta_t = beta_start + t * (beta_end - beta_start) / (T - 1) 

I: number of timesteps T, start and end beta values (VDR) O: list of T VDR beta values, all exact rational 

 linear_schedule(5, VDR(1, 100), VDR(1, 20)) 
    -> [VDR(1,100), VDR(7,400), VDR(13,400), VDR(19,400), VDR(1,20)] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/schedule.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `cosine_schedule_rational`

```python
cosine_schedule_rational(T, s=None)
```

Cosine noise schedule with rational cosine approximation. 

f(t) = cos^2((t/T + s) / (1 + s) * pi/2) alpha_bar_t = f(t) / f(0) 

I: number of timesteps T, offset s (VDR, default 8/1000) O: list of T VDR beta values derived from cosine schedule 

Uses Taylor series cosine for exact rational evaluation. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/schedule.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_alphas`

```python
compute_alphas(betas)
```

Compute alpha = 1 - beta for each timestep. 

I: list of beta values (VDR) O: list of alpha values (VDR), exact (integer subtraction) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/schedule.py#L118"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_alpha_bars`

```python
compute_alpha_bars(alphas)
```

Compute cumulative product alpha_bar_t = product_{k=0}^{t} alpha_k. 

I: list of alpha values (VDR) O: list of alpha_bar values (VDR), exact (integer multiplication) 

 alphas = compute_alphas(betas)  alpha_bars = compute_alpha_bars(alphas)  # alpha_bars[-1] for T=5: 26821179/31250000 exact 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/schedule.py#L137"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_sqrt_alpha_bars`

```python
compute_sqrt_alpha_bars(alpha_bars, depth=10)
```

Compute sqrt(alpha_bar) for each timestep via Newton iteration. 

I: list of alpha_bar values (VDR), Newton depth O: list of sqrt(alpha_bar) values (VDR) 

Each sqrt is exact rational at given depth. Depth 10 = >100 digits. Residual does not compound through chains. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/schedule.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_sqrt_one_minus_alpha_bars`

```python
compute_sqrt_one_minus_alpha_bars(alpha_bars, depth=10)
```

Compute sqrt(1 - alpha_bar) for each timestep. 

I: list of alpha_bar values (VDR), Newton depth O: list of sqrt(1 - alpha_bar) values (VDR) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
