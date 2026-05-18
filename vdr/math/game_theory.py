"""
vdr.math.game_theory — Exact game-theoretic computations.

    from vdr.math.game_theory import shapley_values, minimax_2x2, cournot_duopoly

    phi = shapley_values(v_func, 3)  # sums to exactly v(N)
    p, q, val = minimax_2x2(payoff)  # exact mixed strategies

All equilibria, values, and allocations exact VDR rationals.
"""

from __future__ import annotations
from typing import Callable, List, Tuple
from itertools import permutations, combinations

from vdr.core import VDR
from vdr.linalg import Vec, Mat

__all__ = [
    "minimax_2x2",
    "nash_2x2",
    "shapley_values",
    "dominated_elimination",
    "cournot_duopoly",
]


def minimax_2x2(payoff):
    """
    Solve 2x2 zero-sum game for minimax strategies.

    I: 2x2 payoff Mat (row player)
    O: (p_star, q_star, game_value) where p_star is row player's
       probability on row 0, q_star is column player's probability
       on column 0, game_value is the exact value of the game.

        payoff = Mat.from_ints([[3, -1], [-2, 4]])
        p, q, v = minimax_2x2(payoff)
    """
    a = payoff[0, 0]
    b = payoff[0, 1]
    c = payoff[1, 0]
    d = payoff[1, 1]

    # check for saddle point
    row_mins = [min(a, b), min(c, d)]
    col_maxs = [max(a, c), max(b, d)]
    maximin = max(row_mins)
    minimax = min(col_maxs)

    if maximin == minimax:
        # pure strategy saddle point
        for i in range(2):
            for j in range(2):
                if payoff[i, j] == maximin:
                    p = VDR(1) if i == 0 else VDR(0)
                    q = VDR(1) if j == 0 else VDR(0)
                    return (p, q, maximin)

    # mixed strategy
    # p* = (d - c) / (a - b - c + d)
    denom = a - b - c + d
    if denom == VDR(0):
        raise ValueError("Degenerate game (denom = 0)")

    p_star = (d - c) / denom
    q_star = (d - b) / denom
    value = (a * d - b * c) / denom

    return (p_star, q_star, value)


def nash_2x2(payoff_a, payoff_b):
    """
    Find Nash equilibrium of a 2x2 bimatrix game.

    I: payoff_a (2x2 Mat for player A), payoff_b (2x2 Mat for player B)
    O: (p_star, q_star, ea, eb) — mixed strategy probabilities and
       expected payoffs for each player.

        # Battle of the Sexes
        A = Mat.from_ints([[3, 0], [0, 2]])
        B = Mat.from_ints([[2, 0], [0, 3]])
        p, q, ea, eb = nash_2x2(A, B)
    """
    # Player A's mixed strategy: p on row 0
    # A is indifferent when B mixes: q * a[0,0] + (1-q) * a[0,1] = q * a[1,0] + (1-q) * a[1,1]
    # Solve for q (B's mix that makes A indifferent)

    a00, a01 = payoff_a[0, 0], payoff_a[0, 1]
    a10, a11 = payoff_a[1, 0], payoff_a[1, 1]

    b00, b01 = payoff_b[0, 0], payoff_b[0, 1]
    b10, b11 = payoff_b[1, 0], payoff_b[1, 1]

    # q makes A indifferent:
    # q(a00 - a10) + (1-q)(a01 - a11) = 0
    # q(a00 - a10 - a01 + a11) = a11 - a01
    denom_q = a00 - a10 - a01 + a11
    if denom_q == VDR(0):
        q_star = VDR(1, 2)  # fallback to uniform
    else:
        q_star = (a11 - a01) / denom_q

    # p makes B indifferent:
    # p(b00 - b01) + (1-p)(b10 - b11) = 0
    denom_p = b00 - b01 - b10 + b11
    if denom_p == VDR(0):
        p_star = VDR(1, 2)
    else:
        p_star = (b11 - b10) / denom_p

    # expected payoffs
    ea = q_star * (p_star * a00 + (VDR(1) - p_star) * a10) + \
         (VDR(1) - q_star) * (p_star * a01 + (VDR(1) - p_star) * a11)
    eb = p_star * (q_star * b00 + (VDR(1) - q_star) * b01) + \
         (VDR(1) - p_star) * (q_star * b10 + (VDR(1) - q_star) * b11)

    return (p_star, q_star, ea, eb)


def shapley_values(v_func, n):
    """
    Shapley values for an n-player cooperative game.

    I: characteristic function v_func(frozenset) -> VDR,
       number of players n
    O: Vec of Shapley values, sums to exactly v(grand_coalition)

    phi_i = sum over permutations:
        [v(predecessors_of_i union {i}) - v(predecessors_of_i)] / n!

        def v(s):
            if len(s) == 3: return VDR(1)
            if len(s) == 2: return VDR(1, 2)
            return VDR(0)
        shapley_values(v, 3)
    """
    players = list(range(n))

    # n! as integer
    n_fact = 1
    for i in range(2, n + 1):
        n_fact *= i

    values = [VDR(0)] * n

    for perm in permutations(players):
        predecessors = set()
        for player in perm:
            # marginal contribution
            with_player = frozenset(predecessors | {player})
            without_player = frozenset(predecessors)
            marginal = v_func(with_player) - v_func(without_player)
            values[player] = values[player] + marginal
            predecessors.add(player)

    # divide by n!
    result = [v / VDR(n_fact) for v in values]
    return Vec(result)


def dominated_elimination(payoff):
    """
    Iterated elimination of strictly dominated strategies.

    I: payoff Mat (row player)
    O: reduced payoff Mat with dominated rows/cols removed

    A row i is dominated if there exists row j such that
    payoff[j, k] > payoff[i, k] for all k.
    """
    rows = list(range(payoff.nrows))
    cols = list(range(payoff.ncols))
    changed = True

    while changed:
        changed = False

        # eliminate dominated rows
        new_rows = []
        for i in rows:
            dominated = False
            for j in rows:
                if j == i:
                    continue
                if all(payoff[j, c] > payoff[i, c] for c in cols):
                    dominated = True
                    break
            if not dominated:
                new_rows.append(i)
        if len(new_rows) < len(rows):
            changed = True
            rows = new_rows

        # eliminate dominated columns (for column player, lower is better
        # in zero-sum; for general games this is the column player's view)
        new_cols = []
        for c1 in cols:
            dominated = False
            for c2 in cols:
                if c2 == c1:
                    continue
                if all(payoff[r, c2] < payoff[r, c1] for r in rows):
                    dominated = True
                    break
            if not dominated:
                new_cols.append(c1)
        if len(new_cols) < len(cols):
            changed = True
            cols = new_cols

    # build reduced matrix
    result = []
    for i in rows:
        row = [payoff[i, j] for j in cols]
        result.append(row)
    return Mat(result)


def cournot_duopoly(a, b, c1, c2):
    """
    Cournot duopoly equilibrium.

    Inverse demand: P = a - b*(q1 + q2)
    Cost functions: C1 = c1*q1, C2 = c2*q2

    Profit: pi_i = (a - b*(q1+q2)) * qi - ci*qi

    Nash equilibrium (first-order conditions):
        q1* = (a - 2*b*q1 - b*q2 - c1) = 0 => q1 = (a - c1 - b*q2) / (2b)
        q2* = (a - c2 - b*q1) / (2b)

    Solving simultaneously:
        q1* = (a - 2*c1 + c2) / (3*b)
        q2* = (a - 2*c2 + c1) / (3*b)

    I: demand intercept a, slope b, marginal costs c1, c2 (all VDR)
    O: (q1_star, q2_star, profit1, profit2) as VDR tuple

        cournot_duopoly(VDR(100), VDR(1), VDR(10), VDR(20))
    """
    three_b = VDR(3) * b

    q1 = (a - VDR(2) * c1 + c2) / three_b
    q2 = (a - VDR(2) * c2 + c1) / three_b

    # price at equilibrium
    total_q = q1 + q2
    price = a - b * total_q

    # profits
    profit1 = (price - c1) * q1
    profit2 = (price - c2) * q2

    return (q1, q2, profit1, profit2)
