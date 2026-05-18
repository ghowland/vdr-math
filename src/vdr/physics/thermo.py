"""
vdr.physics.thermo — Exact thermodynamic computations.

    from vdr.physics.thermo import partition_function, free_energy, entropy

    Z = partition_function(energies, beta)
    F = free_energy(Z, beta)
    S = entropy(energies, beta)

Partition functions via exact Taylor exp. Free energy via exact ln.
Discrete calculus for thermodynamic derivatives.
Small Ising lattices by exact enumeration — no Monte Carlo, no sign problems.
"""

from __future__ import annotations
from typing import List, Tuple

from vdr.core import VDR
from vdr.linalg import Mat
from vdr.math.transcendental import exp_series, ln_series
from vdr.math.control import mat_pow

__all__ = [
    "partition_function",
    "free_energy",
    "entropy",
    "specific_heat",
    "boltzmann_weight",
    "boltzmann_probabilities",
    "internal_energy",
    "ising_1d_transfer",
    "ising_1d_partition",
    "ising_2d_exact_enumerate",
]


def boltzmann_weight(energy, beta, exp_depth=16):
    """
    Boltzmann weight exp(-beta * E).

    I: energy E (VDR), inverse temperature beta (VDR), exp depth
    O: exact rational via Taylor series

        boltzmann_weight(VDR(1, 2), VDR(1), 16)
    """
    return exp_series(-beta * energy, exp_depth)


def partition_function(energies, beta, exp_depth=16):
    """
    Canonical partition function Z = sum exp(-beta * E_i).

    I: list of energy levels (VDR), inverse temperature beta (VDR), exp depth
    O: Z as VDR, exact rational at given depth

    Each exp is a Taylor series producing exact rational.
    The sum is exact.

        Z = partition_function([VDR(0), VDR(1), VDR(2)], VDR(1))
    """
    total = VDR(0)
    for E in energies:
        total = total + boltzmann_weight(E, beta, exp_depth)
    return total


def boltzmann_probabilities(energies, beta, exp_depth=16):
    """
    Boltzmann probabilities p_i = exp(-beta*E_i) / Z.

    I: energy levels, beta, exp depth
    O: list of VDR probabilities, sums to exactly 1

        probs = boltzmann_probabilities([VDR(0), VDR(1)], VDR(1))
    """
    weights = [boltzmann_weight(E, beta, exp_depth) for E in energies]
    Z = VDR(0)
    for w in weights:
        Z = Z + w
    return [w / Z for w in weights]


def free_energy(Z, beta, log_depth=16):
    """
    Helmholtz free energy F = -ln(Z) / beta.

    I: partition function Z (VDR), inverse temperature beta (VDR), ln depth
    O: F as VDR, exact rational at given depth

        F = free_energy(Z, VDR(1))
    """
    return -ln_series(Z, log_depth) / beta


def internal_energy(energies, beta, exp_depth=16):
    """
    Internal energy U = sum E_i * p_i = <E>.

    I: energy levels, beta, exp depth
    O: U as VDR, exact
    """
    probs = boltzmann_probabilities(energies, beta, exp_depth)
    total = VDR(0)
    for E, p in zip(energies, probs):
        total = total + E * p
    return total


def entropy(energies, beta, exp_depth=16, log_depth=16):
    """
    Entropy S = beta * (U - F) = beta * U + ln(Z).

    I: energy levels, beta, depths
    O: S as VDR, exact rational at given depth

        S = entropy([VDR(0), VDR(1)], VDR(1))
    """
    Z = partition_function(energies, beta, exp_depth)
    U = internal_energy(energies, beta, exp_depth)
    F = free_energy(Z, beta, log_depth)
    return beta * (U - F)


def specific_heat(energies, beta, h=None, exp_depth=16):
    """
    Specific heat via discrete derivative of internal energy.

    C = -beta^2 * dU/dbeta (discrete derivative)

    I: energy levels, beta, step size h (default beta/100), exp depth
    O: C as VDR, exact

    Uses exact discrete derivative — no cancellation issues.
    """
    if h is None:
        h = beta / VDR(100)

    U_plus = internal_energy(energies, beta + h, exp_depth)
    U_minus = internal_energy(energies, beta, exp_depth)

    dU_dbeta = (U_plus - U_minus) / h

    return -beta * beta * dU_dbeta


def ising_1d_transfer(J, h, beta, N, exp_depth=16):
    """
    1D Ising model partition function via transfer matrix method.

    H = -J * sum s_i * s_{i+1} - h * sum s_i
    Transfer matrix T[s, s'] = exp(beta * (J*s*s' + h*(s+s')/2))

    I: coupling J, field h, inverse temperature beta, chain length N,
       exp depth
    O: partition function Z as VDR, exact

    2x2 transfer matrix raised to power N, then trace.
    Exact matrix power via repeated squaring.

        Z = ising_1d_transfer(VDR(1), VDR(0), VDR(1), 10)
    """
    # states: s = +1 (index 0), s = -1 (index 1)
    # T[0,0] = exp(beta*(J + h))      s=+1, s'=+1
    # T[0,1] = exp(beta*(-J))         s=+1, s'=-1, field: h*(1-1)/2 = 0
    # T[1,0] = exp(beta*(-J))         s=-1, s'=+1
    # T[1,1] = exp(beta*(J - h))      s=-1, s'=-1

    # more carefully:
    # T[s,s'] = exp(beta * J * s * s') * exp(beta * h * (s + s') / 2)
    # s = +1 -> index 0, s = -1 -> index 1

    def T_entry(s_val, sp_val):
        interaction = J * VDR(s_val * sp_val)
        field_term = h * VDR(s_val + sp_val) / VDR(2)
        arg = beta * (interaction + field_term)
        return exp_series(arg, exp_depth)

    T = Mat([
        [T_entry(1, 1), T_entry(1, -1)],
        [T_entry(-1, 1), T_entry(-1, -1)],
    ])

    # Z = Tr(T^N)
    T_N = mat_pow(T, N)
    return T_N.trace()


def ising_1d_partition(J, beta, N, exp_depth=16):
    """
    1D Ising partition function with zero external field.

    Simplified version of ising_1d_transfer with h=0.

    I: coupling J, beta, chain length N
    O: Z as VDR
    """
    return ising_1d_transfer(J, VDR(0), beta, N, exp_depth)


def ising_2d_exact_enumerate(J, beta, Lx, Ly, exp_depth=16):
    """
    2D Ising model by exact enumeration of all 2^(Lx*Ly) states.

    H = -J * sum_{<ij>} s_i * s_j (nearest neighbor on rectangular lattice)

    I: coupling J, beta, lattice dimensions Lx, Ly, exp depth
    O: partition function Z as VDR, exact

    Practical for small lattices (Lx*Ly <= 20).
    No Monte Carlo. No sign problems.

        Z = ising_2d_exact_enumerate(VDR(1), VDR(1), 3, 3)
    """
    N = Lx * Ly
    if N > 25:
        raise ValueError(
            "Exact enumeration limited to N<=25, got %d (=%d*%d)" % (N, Lx, Ly)
        )

    Z = VDR(0)

    for config in range(2 ** N):
        # decode configuration: bit i -> spin at site i
        # spin = +1 if bit set, -1 if not
        spins = []
        for i in range(N):
            if (config >> i) & 1:
                spins.append(1)
            else:
                spins.append(-1)

        # compute energy
        energy = 0
        for x in range(Lx):
            for y in range(Ly):
                site = x * Ly + y
                # right neighbor
                if x + 1 < Lx:
                    nb = (x + 1) * Ly + y
                    energy -= spins[site] * spins[nb]
                # up neighbor
                if y + 1 < Ly:
                    nb = x * Ly + (y + 1)
                    energy -= spins[site] * spins[nb]

        E = J * VDR(energy)
        Z = Z + exp_series(-beta * E, exp_depth)

    return Z
