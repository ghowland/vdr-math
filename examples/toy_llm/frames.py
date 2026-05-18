"""
examples/toy_llm/frames.py

D-frame management — rebase utilities for small fixed denominators.

Every layer boundary rebases values to the target frame.
D never grows. Overflow goes to R. Denominators stay machine-word sized.

    from frames import rebase_vec, init_weight_mat, frame_check_vec

Power-of-two denominators: divmod is bit-shift (free in hardware, fast in Python).
"""

from vdr.core import VDR, Remainder
from vdr.linalg import Vec, Mat


# -- rebase operations ----------------------------------------------------

def rebase_value(x, target_d):
    """
    Rebase a VDR value to target denominator frame.

    Uses divmod: V * target_d = Q * D + S
    Result: [Q, target_d, remainder] — D stays at target_d.

    For closed VDR [V, D, 0]:
      N = V * target_d
      Q = N // D
      S = N % D
      if S == 0: result is [Q, target_d, 0] (closed, clean rebase)
      if S != 0: result is [Q, target_d, [S, D, 0]] (active, mismatch in R)

    I: VDR value, target denominator (int, power of two)
    O: VDR with d == target_d

        rebase_value(VDR(1, 3), 128) -> VDR(42, 128, [2, 3, 0])
        rebase_value(VDR(1, 2), 128) -> VDR(64, 128, 0)
    """
    if x == VDR(0):
        return VDR(0, target_d)

    if x.d == target_d and x.is_closed:
        return x

    return x.rebase(target_d)


def rebase_vec(v, target_d):
    """
    Rebase every element of a Vec to target denominator frame.

    I: Vec of VDR values, target denominator
    O: new Vec with every element rebased

        rebase_vec(Vec([VDR(1, 3), VDR(1, 2)]), 128)
        -> Vec([VDR(42, 128, ...), VDR(64, 128)])
    """
    return Vec([rebase_value(v[i], target_d) for i in range(len(v))])


def rebase_vec_list(vs, target_d):
    """
    Rebase a list of Vec to target frame.

    I: list of Vec, target denominator
    O: list of rebased Vec
    """
    return [rebase_vec(v, target_d) for v in vs]


def rebase_mat(m, target_d):
    """
    Rebase every element of a Mat to target denominator frame.

    I: Mat of VDR values, target denominator
    O: new Mat with every element rebased
    """
    data = []
    for i in range(m.nrows):
        r = m.row(i)
        row = [rebase_value(r[j], target_d) for j in range(m.ncols)]
        data.append(row)
    return Mat(data)


def rebase_params(params, target_d):
    """
    Rebase all parameter values to target frame, in place.
    Handles VDR scalar, Vec, and Mat parameter values.

    I: list of parameter objects (each has .value), target denominator
    O: None
    S: mutates each param.value
    """
    for p in params:
        v = p.value
        if isinstance(v, Mat):
            p.value = rebase_mat(v, target_d)
        elif isinstance(v, Vec):
            p.value = rebase_vec(v, target_d)
        else:
            p.value = rebase_value(v, target_d)


def rebase_grads(params, target_d):
    """
    Rebase all parameter gradients to target frame, in place.
    Handles VDR scalar, Vec, and Mat parameter gradients.

    I: list of parameter objects (each has .grad), target denominator
    O: None
    S: mutates each param.grad
    """
    for p in params:
        g = p.grad
        if g is None:
            continue
        if isinstance(g, Mat):
            p.grad = rebase_mat(g, target_d)
        elif isinstance(g, Vec):
            p.grad = rebase_vec(g, target_d)
        else:
            p.grad = rebase_value(g, target_d)


# -- initialization -------------------------------------------------------

def _simple_rng(seed):
    """
    Deterministic integer PRNG for weight initialization.

    Linear congruential generator. Returns function that produces
    next integer on each call.

    I: seed int
    O: callable() -> int in range
    """
    state = [seed]

    def next_val(lo, hi):
        # LCG: state = (a * state + c) mod m
        state[0] = (1103515245 * state[0] + 12345) & 0x7FFFFFFF
        # map to [lo, hi] inclusive
        span = hi - lo + 1
        return lo + (state[0] % span)

    return next_val


def init_weight_vec(dim, denom, seed, idx):
    """
    Initialize a weight vector in fixed D-frame.

    Values are VDR(k, denom) where k is small integer from deterministic RNG.
    Range: k in [-(denom//4), denom//4] for Xavier-like scale.

    I: dimension, denominator (D_WEIGHT), rng seed, index offset
    O: Vec with small-integer numerators in D_WEIGHT frame

        init_weight_vec(4, 128, 42, 0)
        -> Vec of 4 values like VDR(12, 128), VDR(-7, 128), ...
    """
    rng = _simple_rng(seed + idx)
    bound = denom // 4  # Xavier-like: values in [-0.25, 0.25]
    data = [VDR(rng(-bound, bound), denom) for _ in range(dim)]
    return Vec(data)


def init_weight_mat(rows, cols, denom, seed, idx):
    """
    Initialize a weight matrix in fixed D-frame.

    Each row is an independently seeded weight vector.

    I: rows, cols, denominator, seed, index offset
    O: Mat with small-integer numerators in D_WEIGHT frame

        init_weight_mat(4, 4, 128, 42, 0)
        -> 4x4 Mat of VDR(k, 128) values
    """
    data = []
    for r in range(rows):
        rng = _simple_rng(seed + idx + r * 1000)
        bound = denom // 4
        row = [VDR(rng(-bound, bound), denom) for _ in range(cols)]
        data.append(row)
    return Mat(data)


def init_bias_vec(dim, denom):
    """
    Initialize a bias vector to zero in fixed D-frame.

    I: dimension, denominator
    O: Vec of VDR(0, denom) values

        init_bias_vec(4, 128) -> Vec([VDR(0,128), VDR(0,128), ...])
    """
    return Vec([VDR(0, denom) for _ in range(dim)])


def zero_vec(dim):
    """
    Zero vector with default denominator 1.

    I: dimension
    O: Vec of VDR(0) values
    """
    return Vec([VDR(0)] * dim)


# -- frame discipline checks ---------------------------------------------

def frame_check(x, expected_d):
    """
    Check that a VDR value is in the expected D-frame.

    I: VDR value, expected denominator
    O: True if x.d == expected_d, or x is exactly zero

        frame_check(VDR(42, 128), 128) -> True
        frame_check(VDR(42, 256), 128) -> False
        frame_check(VDR(0), 128)       -> True
    """
    if x == VDR(0):
        return True
    return x.d == expected_d


def frame_check_vec(v, expected_d):
    """
    Check that every element of a Vec is in the expected D-frame.

    I: Vec, expected denominator
    O: True if all elements pass frame_check

        frame_check_vec(rebase_vec(v, 128), 128) -> True
    """
    for i in range(len(v)):
        if not frame_check(v[i], expected_d):
            return False
    return True


def frame_report(model):
    """
    Report D-frame status of all model parameters.

    I: model with .parameters() method
    O: dict with counts per denominator value

        report = frame_report(model)
        # {"128": 200, "other": 0}
    """
    counts = {}
    for p in model.parameters():
        d = p.value.d
        key = str(d)
        counts[key] = counts.get(key, 0) + 1
    return counts

