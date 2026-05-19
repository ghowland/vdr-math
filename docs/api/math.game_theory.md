<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/game_theory.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `math.game_theory`
vdr.math.game_theory — Exact game-theoretic computations. 

 from vdr.math.game_theory import shapley_values, minimax_2x2, cournot_duopoly 

 phi = shapley_values(v_func, 3)  # sums to exactly v(N)  p, q, val = minimax_2x2(payoff)  # exact mixed strategies 

All equilibria, values, and allocations exact VDR rationals. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/game_theory.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `minimax_2x2`

```python
minimax_2x2(payoff)
```

Solve 2x2 zero-sum game for minimax strategies. 

I: 2x2 payoff Mat (row player) O: (p_star, q_star, game_value) where p_star is row player's  probability on row 0, q_star is column player's probability  on column 0, game_value is the exact value of the game. 

 payoff = Mat.from_ints([[3, -1], [-2, 4]])  p, q, v = minimax_2x2(payoff) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/game_theory.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `nash_2x2`

```python
nash_2x2(payoff_a, payoff_b)
```

Find Nash equilibrium of a 2x2 bimatrix game. 

I: payoff_a (2x2 Mat for player A), payoff_b (2x2 Mat for player B) O: (p_star, q_star, ea, eb) — mixed strategy probabilities and  expected payoffs for each player. 

 # Battle of the Sexes  A = Mat.from_ints([[3, 0], [0, 2]])  B = Mat.from_ints([[2, 0], [0, 3]])  p, q, ea, eb = nash_2x2(A, B) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/game_theory.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `shapley_values`

```python
shapley_values(v_func, n)
```

Shapley values for an n-player cooperative game. 

I: characteristic function v_func(frozenset) -> VDR,  number of players n O: Vec of Shapley values, sums to exactly v(grand_coalition) 

phi_i = sum over permutations:  [v(predecessors_of_i union {i}) - v(predecessors_of_i)] / n! 

 def v(s):  if len(s) == 3: return VDR(1)  if len(s) == 2: return VDR(1, 2)  return VDR(0)  shapley_values(v, 3) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/game_theory.py#L163"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `dominated_elimination`

```python
dominated_elimination(payoff)
```

Iterated elimination of strictly dominated strategies. 

I: payoff Mat (row player) O: reduced payoff Mat with dominated rows/cols removed 

A row i is dominated if there exists row j such that payoff[j, k] > payoff[i, k] for all k. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/game_theory.py#L221"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `cournot_duopoly`

```python
cournot_duopoly(a, b, c1, c2)
```

Cournot duopoly equilibrium. 

Inverse demand: P = a - b*(q1 + q2) Cost functions: C1 = c1*q1, C2 = c2*q2 

Profit: pi_i = (a - b*(q1+q2)) * qi - ci*qi 

Nash equilibrium (first-order conditions):  q1* = (a - 2*b*q1 - b*q2 - c1) = 0 => q1 = (a - c1 - b*q2) / (2b)  q2* = (a - c2 - b*q1) / (2b) 

Solving simultaneously:  q1* = (a - 2*c1 + c2) / (3*b)  q2* = (a - 2*c2 + c1) / (3*b) 

I: demand intercept a, slope b, marginal costs c1, c2 (all VDR) O: (q1_star, q2_star, profit1, profit2) as VDR tuple 

 cournot_duopoly(VDR(100), VDR(1), VDR(10), VDR(20)) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
