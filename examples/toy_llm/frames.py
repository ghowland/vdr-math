"""
examples/toy_llm/frames.py

Fixed-frame arithmetic — all operations keep D at 2^bits.
Uses qb_mul/qb_add from vdr.basis so D never compounds.

    from frames import qb_dot, qb_matvec, qb_linear_forward
"""

from vdr.core import VDR
from vdr.linalg import Vec, Mat
from vdr.basis import qb_mul, qb_add, to_qbasis


# -- core fixed-frame arithmetic -----------------------------------------

def qb_sub(a, b, bits):
    """
    Subtraction in fixed frame: a - b.

    I: a VDR, b VDR, bits
    O: VDR in D=2^bits frame
    """
    neg_b = VDR(-b.v, b.d, b.r)
    return qb_add(a, neg_b, bits=bits)


def qb_dot(a, b, bits):
    """
    Dot product of two Vec in fixed frame.
    Each multiply via qb_mul, accumulate via qb_add.
    D stays at 2^bits throughout.

    I: a Vec, b Vec, bits
    O: VDR scalar in D=2^bits frame
    """
    d = 2 ** bits
    acc = VDR(0, d)
    for i in range(len(a)):
        prod = qb_mul(a[i], b[i], bits=bits)
        acc = qb_add(acc, prod, bits=bits)
    return acc


def qb_matvec(m, v, bits):
    """
    Matrix-vector product in fixed frame.

    I: Mat m, Vec v, bits
    O: Vec in D=2^bits frame
    """
    result = []
    for i in range(m.nrows):
        row = m.row(i)
        result.append(qb_dot(row, v, bits))
    return Vec(result)


def qb_vec_add(a, b, bits):
    """
    Element-wise Vec addition in fixed frame.

    I: a Vec, b Vec, bits
    O: Vec in D=2^bits frame
    """
    return Vec([qb_add(a[i], b[i], bits=bits) for i in range(len(a))])


def qb_vec_sub(a, b, bits):
    """
    Element-wise Vec subtraction in fixed frame.

    I: a Vec, b Vec, bits
    O: Vec in D=2^bits frame
    """
    return Vec([qb_sub(a[i], b[i], bits=bits) for i in range(len(a))])


def qb_vec_scale(v, s, bits):
    """
    Scale Vec by scalar in fixed frame.

    I: v Vec, s VDR scalar, bits
    O: Vec in D=2^bits frame
    """
    return Vec([qb_mul(v[i], s, bits=bits) for i in range(len(v))])


def qb_linear_forward(weight_mat, bias_vec, x, bits):
    """
    Linear layer forward: y = Wx + b in fixed frame.

    I: weight_mat Mat, bias_vec Vec, x Vec, bits
    O: Vec in D=2^bits frame
    """
    y = qb_matvec(weight_mat, x, bits)
    return qb_vec_add(y, bias_vec, bits)


# -- initialization -------------------------------------------------------

def _simple_rng(seed):
    """Deterministic integer PRNG."""
    state = [seed]
    def next_val(lo, hi):
        state[0] = (1103515245 * state[0] + 12345) & 0x7FFFFFFF
        span = hi - lo + 1
        return lo + (state[0] % span)
    return next_val


def init_weight_vec(dim, denom, seed, idx):
    """
    Initialize weight vector in fixed D-frame.
    Values VDR(k, denom) where k in [-denom//4, denom//4].

    I: dim, denom (D_WEIGHT), seed, index offset
    O: Vec in D=denom frame
    """
    rng = _simple_rng(seed + idx)
    bound = denom // 4
    return Vec([VDR(rng(-bound, bound), denom) for _ in range(dim)])


def init_weight_mat(rows, cols, denom, seed, idx):
    """
    Initialize weight matrix in fixed D-frame.

    I: rows, cols, denom, seed, index offset
    O: Mat in D=denom frame
    """
    data = []
    for r in range(rows):
        rng = _simple_rng(seed + idx + r * 1000)
        bound = denom // 4
        row = [VDR(rng(-bound, bound), denom) for _ in range(cols)]
        data.append(row)
    return Mat(data)


def init_bias_vec(dim, denom):
    """Zero bias in fixed D-frame."""
    return Vec([VDR(0, denom) for _ in range(dim)])


def zero_vec(dim):
    """Zero vector, D=1."""
    return Vec([VDR(0)] * dim)


def zero_vec_in_frame(dim, bits):
    """Zero vector in fixed D-frame."""
    d = 2 ** bits
    return Vec([VDR(0, d) for _ in range(dim)])


# -- rebase for params/grads (used by optimizer step) --------------------

def rebase_value(x, target_d):
    """Rebase scalar VDR to target D."""
    if x == VDR(0):
        return VDR(0, target_d)
    if x.d == target_d and x.is_closed:
        return x
    return x.rebase(target_d)


def rebase_vec(v, target_d):
    """Rebase every element of Vec."""
    return Vec([rebase_value(v[i], target_d) for i in range(len(v))])


def rebase_mat(m, target_d):
    """Rebase every element of Mat."""
    data = []
    for i in range(m.nrows):
        r = m.row(i)
        row = [rebase_value(r[j], target_d) for j in range(m.ncols)]
        data.append(row)
    return Mat(data)


def rebase_params(params, target_d):
    """Rebase all parameter values in place. Handles VDR, Vec, Mat."""
    for p in params:
        v = p.value
        if isinstance(v, Mat):
            p.value = rebase_mat(v, target_d)
        elif isinstance(v, Vec):
            p.value = rebase_vec(v, target_d)
        else:
            p.value = rebase_value(v, target_d)


def rebase_grads(params, target_d):
    """Rebase all parameter gradients in place. Handles VDR, Vec, Mat."""
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


# -- frame discipline checks ---------------------------------------------

def frame_check(x, expected_d):
    """Check VDR is in expected D-frame."""
    if x == VDR(0):
        return True
    return x.d == expected_d


def frame_check_vec(v, expected_d):
    """Check all elements of Vec in expected D-frame."""
    for i in range(len(v)):
        if not frame_check(v[i], expected_d):
            return False
    return True


def frame_report(model):
    """Report D-frame of all parameters."""
    counts = {}
    for p in model.parameters():
        v = p.value
        if isinstance(v, Mat):
            for i in range(v.nrows):
                row = v.row(i)
                for j in range(v.ncols):
                    d = row[j].d
                    key = str(d)
                    counts[key] = counts.get(key, 0) + 1
        elif isinstance(v, Vec):
            for i in range(len(v)):
                d = v[i].d
                key = str(d)
                counts[key] = counts.get(key, 0) + 1
        else:
            key = str(v.d)
            counts[key] = counts.get(key, 0) + 1
    return counts
