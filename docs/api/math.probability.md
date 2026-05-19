<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/probability.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `math.probability`
vdr.math.probability — Exact probability computations. 

 from vdr.math.probability import binom_pmf, bayes_update, markov_steady_state 

 pmf = binom_pmf_full(10, VDR(1, 3))  # PMF sums to exactly 1 

 posterior = bayes_update(VDR(1, 2), VDR(3))  # exact Bayesian update 

 steady = markov_steady_state(transition_matrix)  # sums to exactly 1 

All probabilities exact. PMFs sum to exactly 1. Posteriors exact. No float rounding in conditional logic. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/probability.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `binom_pmf`

```python
binom_pmf(n, k, p)
```

Binomial PMF: P(X = k) for X ~ Binomial(n, p). 

I: trials n, successes k, success probability p (VDR) O: exact probability as VDR 

 binom_pmf(10, 3, VDR(1, 3)) -> exact 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/probability.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `binom_pmf_full`

```python
binom_pmf_full(n, p)
```

Full binomial PMF vector for X ~ Binomial(n, p). 

I: trials n, success probability p (VDR) O: list of VDR [P(X=0), P(X=1), ..., P(X=n)]  Sums to exactly 1. 

 pmf = binom_pmf_full(10, VDR(1, 3))  sum(pmf) == VDR(1)  # True, exactly 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/probability.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bayes_update`

```python
bayes_update(prior, likelihood_ratio)
```

Single Bayesian update. 

P(H|D) = P(H) * LR / (P(H) * LR + (1 - P(H))) 

where LR = P(D|H) / P(D|not H) 

I: prior probability P(H) as VDR, likelihood ratio as VDR O: posterior probability as VDR, exact 

 bayes_update(VDR(1, 2), VDR(3)) -> VDR(3, 4) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/probability.py#L112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bayes_sequential`

```python
bayes_sequential(prior, likelihood_ratios)
```

Sequential Bayesian updating through multiple observations. 

I: initial prior as VDR, list of likelihood ratios O: list of posteriors after each update 

 bayes_sequential(VDR(1, 2), [VDR(3), VDR(2), VDR(4)]) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/probability.py#L129"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `markov_steady_state`

```python
markov_steady_state(transition)
```

Steady-state distribution of a Markov chain. 

Solves pi * P = pi with sum(pi) = 1. 

I: row-stochastic transition matrix (Mat) O: steady-state vector (Vec), sums to exactly 1 

Method: solve (P^T - I) augmented with sum=1 constraint. Uses Gaussian elimination for exact result. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/probability.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `markov_step`

```python
markov_step(state, transition)
```

One step of Markov chain: state * transition. 

I: state as Vec (row vector), transition as Mat O: next state as Vec 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/probability.py#L184"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `markov_power`

```python
markov_power(state, transition, steps)
```

Evolve Markov chain for given number of steps. 

I: initial state Vec, transition Mat, number of steps O: state after steps 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/probability.py#L197"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gamblers_ruin`

```python
gamblers_ruin(k, n)
```

Gambler's ruin probability for fair game. 

P(ruin starting with k, total capital n) = (n - k) / n 

I: starting capital k, total capital n O: ruin probability as VDR 

 gamblers_ruin(3, 10) -> VDR(7, 10) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/probability.py#L211"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `expected_value`

```python
expected_value(values, probs)
```

Expected value E[X] = sum(xi * pi). 

I: list of values (VDR), list of probabilities (VDR) O: exact expected value as VDR 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/probability.py#L226"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `variance`

```python
variance(values, probs)
```

Variance Var(X) = E[X^2] - E[X]^2. 

I: list of values (VDR), list of probabilities (VDR) O: exact variance as VDR 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
