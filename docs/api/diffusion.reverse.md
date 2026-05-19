<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/reverse.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `diffusion.reverse`
vdr.diffusion.reverse — Exact reverse diffusion process. 

 from vdr.diffusion.reverse import compute_x0_prediction, reverse_step_ddim 

 x0_pred = compute_x0_prediction(xt, t, schedule, eps_pred)  # with perfect noise: error = exactly 0 

 x_prev = reverse_step_ddim(xt, t, t_prev, schedule, eps_pred)  # DDIM roundtrip error = exactly 0 

Exact rational subtraction and division. No catastrophic cancellation. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/reverse.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_x0_prediction`

```python
compute_x0_prediction(xt, t, schedule, eps_pred)
```

Predict x_0 from x_t and predicted noise. 

x_0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t) 

I: noisy signal xt (Vec), timestep t, schedule, predicted noise (Vec) O: predicted original signal x_0 (Vec), exact 

With oracle noise predictor (eps_pred == actual epsilon): prediction error = exactly 0. Not approximately zero. Zero. 

 x0_pred = compute_x0_prediction(xt, 3, schedule, eps_pred) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/reverse.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_posterior_mean`

```python
compute_posterior_mean(xt, t, schedule, eps_pred)
```

Posterior mean for reverse step. 

mu_t = (1/sqrt(alpha_t)) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_pred) 

I: noisy xt (Vec), timestep t, schedule, predicted noise (Vec) O: posterior mean (Vec), exact 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/reverse.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `reverse_step`

```python
reverse_step(xt, t, schedule, eps_pred, z=None)
```

One stochastic reverse diffusion step. 

x_{t-1} = mu_t + sqrt(beta_tilde_t) * z 

where z is optional noise (default zero for deterministic). 

I: noisy xt (Vec), timestep t, schedule, predicted noise (Vec),  optional noise z (Vec) O: denoised x_{t-1} (Vec), exact 

 x_prev = reverse_step(xt, 3, schedule, eps_pred) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/reverse.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `reverse_step_ddim`

```python
reverse_step_ddim(xt, t, t_prev, schedule, eps_pred, eta=None)
```

DDIM (Denoising Diffusion Implicit Models) deterministic reverse step. 

x_{t-1} = sqrt(alpha_bar_{t-1}) * x_0_pred +  sqrt(1 - alpha_bar_{t-1} - sigma^2) * eps_pred +  sigma * z 

With eta=0 (default): fully deterministic. No stochastic noise. Roundtrip error with perfect noise = exactly 0. 

I: xt (Vec), current timestep t, previous timestep t_prev,  schedule, predicted noise (Vec), eta (VDR, default 0) O: x_{t-1} (Vec), exact 

DDIM with eta=0 is the paper's strongest result: forward-reverse roundtrip error = exactly zero. 

 x_prev = reverse_step_ddim(xt, 4, 3, schedule, eps_pred) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/diffusion/reverse.py#L151"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `reverse_sample_loop`

```python
reverse_sample_loop(xT, schedule, predict_noise, noise_vectors=None)
```

Complete reverse sampling loop from x_T to x_0. 

I: initial noisy signal xT (Vec), schedule,  noise predictor function (xt, t) -> Vec,  optional noise vectors for stochastic steps O: denoised signal x_0 (Vec) 

Each step exact. Drift does not grow across steps. 

 def oracle(xt, t):  return actual_noise_at_t  x0_recovered = reverse_sample_loop(xT, schedule, oracle) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
