"""
vdr.math.probability — Exact probability computations.

    from vdr.math.probability import binom_pmf, bayes_update, markov_steady_state

    pmf = binom_pmf_full(10, VDR(1, 3))
    # PMF sums to exactly 1

    posterior = bayes_update(VDR(1, 2), VDR(3))
    # exact Bayesian update

    steady = markov_steady_state(transition_matrix)
    # sums to exactly 1

All probabilities exact. PMFs sum to exactly 1. Posteriors exact.
No float rounding in conditional logic.
"""

from __future__ import annotations
from typing import List, Callable

from vdr.core import VDR
from vdr.linalg import Vec, Mat

__all__ = [
    "binom_pmf",
    "binom_pmf_full",
    "bayes_update",
    "bayes_sequential",
    "markov_steady_state",
    "markov_step",
    "markov_power",
    "gamblers_ruin",
    "expected_value",
    "variance",
]


def _binom_coeff(n, k):
    """Internal binomial coefficient as integer."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n - k:
        k = n - k
    result = 1
    for i in range(k):
        result = result * (n - i)
        result = result // (i + 1)
    return result


def binom_pmf(n, k, p):
    """
    Binomial PMF: P(X = k) for X ~ Binomial(n, p).

    I: trials n, successes k, success probability p (VDR)
    O: exact probability as VDR

        binom_pmf(10, 3, VDR(1, 3)) -> exact
    """
    if k < 0 or k > n:
        return VDR(0)

    coeff = VDR(_binom_coeff(n, k))
    # p^k
    p_k = VDR(1)
    for _ in range(k):
        p_k = p_k * p
    # (1-p)^(n-k)
    q = VDR(1) - p
    q_nk = VDR(1)
    for _ in range(n - k):
        q_nk = q_nk * q

    return coeff * p_k * q_nk


def binom_pmf_full(n, p):
    """
    Full binomial PMF vector for X ~ Binomial(n, p).

    I: trials n, success probability p (VDR)
    O: list of VDR [P(X=0), P(X=1), ..., P(X=n)]
       Sums to exactly 1.

        pmf = binom_pmf_full(10, VDR(1, 3))
        sum(pmf) == VDR(1)  # True, exactly
    """
    return [binom_pmf(n, k, p) for k in range(n + 1)]


def bayes_update(prior, likelihood_ratio):
    """
    Single Bayesian update.

    P(H|D) = P(H) * LR / (P(H) * LR + (1 - P(H)))

    where LR = P(D|H) / P(D|not H)

    I: prior probability P(H) as VDR, likelihood ratio as VDR
    O: posterior probability as VDR, exact

        bayes_update(VDR(1, 2), VDR(3)) -> VDR(3, 4)
    """
    numerator = prior * likelihood_ratio
    denominator = numerator + (VDR(1) - prior)
    return numerator / denominator


def bayes_sequential(prior, likelihood_ratios):
    """
    Sequential Bayesian updating through multiple observations.

    I: initial prior as VDR, list of likelihood ratios
    O: list of posteriors after each update

        bayes_sequential(VDR(1, 2), [VDR(3), VDR(2), VDR(4)])
    """
    posteriors = []
    current = prior
    for lr in likelihood_ratios:
        current = bayes_update(current, lr)
        posteriors.append(current)
    return posteriors


def markov_steady_state(transition):
    """
    Steady-state distribution of a Markov chain.

    Solves pi * P = pi with sum(pi) = 1.

    I: row-stochastic transition matrix (Mat)
    O: steady-state vector (Vec), sums to exactly 1

    Method: solve (P^T - I) augmented with sum=1 constraint.
    Uses Gaussian elimination for exact result.
    """
    n = transition.nrows

    # Build system: (P^T - I)^T's null space with sum constraint
    # Equivalently, solve [P^T - I; 1 1 ... 1] * pi = [0; 0; ...; 1]
    # Replace last equation of (P^T - I) with sum = 1

    # A = P^T - I
    pt = transition.T
    I_mat = Mat.identity(n)
    A = pt - I_mat

    # Replace last row with [1, 1, ..., 1]
    rows = []
    for i in range(n - 1):
        rows.append([A[i, j] for j in range(n)])
    rows.append([VDR(1)] * n)

    system = Mat(rows)

    # RHS: [0, 0, ..., 0, 1]
    rhs_data = [VDR(0)] * (n - 1) + [VDR(1)]
    rhs = Vec(rhs_data)

    return system.solve(rhs)


def markov_step(state, transition):
    """
    One step of Markov chain: state * transition.

    I: state as Vec (row vector), transition as Mat
    O: next state as Vec
    """
    n = len(state)
    result = []
    for j in range(n):
        total = VDR(0)
        for i in range(n):
            total = total + state[i] * transition[i, j]
        result.append(total)
    return Vec(result)


def markov_power(state, transition, steps):
    """
    Evolve Markov chain for given number of steps.

    I: initial state Vec, transition Mat, number of steps
    O: state after steps
    """
    current = state
    for _ in range(steps):
        current = markov_step(current, transition)
    return current


def gamblers_ruin(k, n):
    """
    Gambler's ruin probability for fair game.

    P(ruin starting with k, total capital n) = (n - k) / n

    I: starting capital k, total capital n
    O: ruin probability as VDR

        gamblers_ruin(3, 10) -> VDR(7, 10)
    """
    return VDR(n - k, n)


def expected_value(values, probs):
    """
    Expected value E[X] = sum(xi * pi).

    I: list of values (VDR), list of probabilities (VDR)
    O: exact expected value as VDR
    """
    if len(values) != len(probs):
        raise ValueError("values and probs must have same length")
    total = VDR(0)
    for v, p in zip(values, probs):
        total = total + v * p
    return total


def variance(values, probs):
    """
    Variance Var(X) = E[X^2] - E[X]^2.

    I: list of values (VDR), list of probabilities (VDR)
    O: exact variance as VDR
    """
    ex = expected_value(values, probs)
    ex2 = expected_value([v * v for v in values], probs)
    return ex2 - ex * ex
