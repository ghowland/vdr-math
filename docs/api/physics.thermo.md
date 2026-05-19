<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/thermo.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `physics.thermo`
vdr.physics.thermo — Exact thermodynamic computations. 

 from vdr.physics.thermo import partition_function, free_energy, entropy 

 Z = partition_function(energies, beta)  F = free_energy(Z, beta)  S = entropy(energies, beta) 

Partition functions via exact Taylor exp. Free energy via exact ln. Discrete calculus for thermodynamic derivatives. Small Ising lattices by exact enumeration — no Monte Carlo, no sign problems. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/thermo.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `boltzmann_weight`

```python
boltzmann_weight(energy, beta, exp_depth=16)
```

Boltzmann weight exp(-beta * E). 

I: energy E (VDR), inverse temperature beta (VDR), exp depth O: exact rational via Taylor series 

 boltzmann_weight(VDR(1, 2), VDR(1), 16) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/thermo.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `partition_function`

```python
partition_function(energies, beta, exp_depth=16)
```

Canonical partition function Z = sum exp(-beta * E_i). 

I: list of energy levels (VDR), inverse temperature beta (VDR), exp depth O: Z as VDR, exact rational at given depth 

Each exp is a Taylor series producing exact rational. The sum is exact. 

 Z = partition_function([VDR(0), VDR(1), VDR(2)], VDR(1)) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/thermo.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `boltzmann_probabilities`

```python
boltzmann_probabilities(energies, beta, exp_depth=16)
```

Boltzmann probabilities p_i = exp(-beta*E_i) / Z. 

I: energy levels, beta, exp depth O: list of VDR probabilities, sums to exactly 1 

 probs = boltzmann_probabilities([VDR(0), VDR(1)], VDR(1)) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/thermo.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `free_energy`

```python
free_energy(Z, beta, log_depth=16)
```

Helmholtz free energy F = -ln(Z) / beta. 

I: partition function Z (VDR), inverse temperature beta (VDR), ln depth O: F as VDR, exact rational at given depth 

 F = free_energy(Z, VDR(1)) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/thermo.py#L95"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `internal_energy`

```python
internal_energy(energies, beta, exp_depth=16)
```

Internal energy U = sum E_i * p_i = <E>. 

I: energy levels, beta, exp depth O: U as VDR, exact 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/thermo.py#L109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `entropy`

```python
entropy(energies, beta, exp_depth=16, log_depth=16)
```

Entropy S = beta * (U - F) = beta * U + ln(Z). 

I: energy levels, beta, depths O: S as VDR, exact rational at given depth 

 S = entropy([VDR(0), VDR(1)], VDR(1)) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/thermo.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `specific_heat`

```python
specific_heat(energies, beta, h=None, exp_depth=16)
```

Specific heat via discrete derivative of internal energy. 

C = -beta^2 * dU/dbeta (discrete derivative) 

I: energy levels, beta, step size h (default beta/100), exp depth O: C as VDR, exact 

Uses exact discrete derivative — no cancellation issues. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/thermo.py#L146"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `ising_1d_transfer`

```python
ising_1d_transfer(J, h, beta, N, exp_depth=16)
```

1D Ising model partition function via transfer matrix method. 

H = -J * sum s_i * s_{i+1} - h * sum s_i Transfer matrix T[s, s'] = exp(beta * (J*s*s' + h*(s+s')/2)) 

I: coupling J, field h, inverse temperature beta, chain length N,  exp depth O: partition function Z as VDR, exact 

2x2 transfer matrix raised to power N, then trace. Exact matrix power via repeated squaring. 

 Z = ising_1d_transfer(VDR(1), VDR(0), VDR(1), 10) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/thermo.py#L188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `ising_1d_partition`

```python
ising_1d_partition(J, beta, N, exp_depth=16)
```

1D Ising partition function with zero external field. 

Simplified version of ising_1d_transfer with h=0. 

I: coupling J, beta, chain length N O: Z as VDR 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/thermo.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `ising_2d_exact_enumerate`

```python
ising_2d_exact_enumerate(J, beta, Lx, Ly, exp_depth=16)
```

2D Ising model by exact enumeration of all 2^(Lx*Ly) states. 

H = -J * sum_{<ij>} s_i * s_j (nearest neighbor on rectangular lattice) 

I: coupling J, beta, lattice dimensions Lx, Ly, exp depth O: partition function Z as VDR, exact 

Practical for small lattices (Lx*Ly <= 20). No Monte Carlo. No sign problems. 

 Z = ising_2d_exact_enumerate(VDR(1), VDR(1), 3, 3) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
