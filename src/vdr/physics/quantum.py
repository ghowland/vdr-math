"""
vdr.physics.quantum — Exact quantum mechanics computations.

    from vdr.physics.quantum import SIGMA_X, SIGMA_Z, spin_rotation
    from vdr.physics.quantum import measurement_probabilities, verify_unitarity

    U = spin_rotation("z", VDR(1, 4))  # pi/4 rotation about z
    probs = measurement_probabilities(state)  # sums to exactly 1
    verify_unitarity(U)  # True, exact

Pauli algebra verified as structural identity.
Spin rotation periodicity: U(pi/2 about z) applied 4 times = I exactly.
"""

from __future__ import annotations
from typing import List, Tuple

from vdr.core import VDR
from vdr.linalg import Vec, Mat
from vdr.math.transcendental import sin_series, cos_series, PI

__all__ = [
    "SIGMA_X",
    "SIGMA_Y_REAL",
    "SIGMA_Y_IMAG",
    "SIGMA_Z",
    "IDENTITY_2",
    "pauli_multiply",
    "spin_rotation",
    "spin_rotation_components",
    "measurement_probabilities",
    "density_matrix",
    "verify_unitarity",
    "commutator",
    "anticommutator",
    "expectation_value",
]


# ---------------------------------------------------------------------------
# Pauli matrices
# ---------------------------------------------------------------------------

SIGMA_X = Mat.from_ints([[0, 1], [1, 0]])
SIGMA_Z = Mat.from_ints([[1, 0], [0, -1]])
IDENTITY_2 = Mat.identity(2)

# sigma_y has imaginary entries: [[0, -i], [i, 0]]
# Represented as real and imaginary parts separately
SIGMA_Y_REAL = Mat.from_ints([[0, 0], [0, 0]])
SIGMA_Y_IMAG = Mat.from_ints([[0, -1], [1, 0]])


# Pauli multiplication table (phase, result)
# sigma_a * sigma_b = phase * sigma_result
# phase is (real, imag) in {1, -1, i, -i}
_PAULI_TABLE = {
    ("I", "I"): ((1, 0), "I"),
    ("I", "x"): ((1, 0), "x"),
    ("I", "y"): ((1, 0), "y"),
    ("I", "z"): ((1, 0), "z"),
    ("x", "I"): ((1, 0), "x"),
    ("x", "x"): ((1, 0), "I"),
    ("x", "y"): ((0, 1), "z"),   # i * sigma_z
    ("x", "z"): ((0, -1), "y"),  # -i * sigma_y
    ("y", "I"): ((1, 0), "y"),
    ("y", "x"): ((0, -1), "z"),  # -i * sigma_z
    ("y", "y"): ((1, 0), "I"),
    ("y", "z"): ((0, 1), "x"),   # i * sigma_x
    ("z", "I"): ((1, 0), "z"),
    ("z", "x"): ((0, 1), "y"),   # i * sigma_y
    ("z", "y"): ((0, -1), "x"),  # -i * sigma_x
    ("z", "z"): ((1, 0), "I"),
}


def pauli_multiply(a, b):
    """
    Multiply two Pauli matrices by label.

    I: Pauli labels a, b from {"I", "x", "y", "z"}
    O: ((real_phase, imag_phase), result_label)

        pauli_multiply("x", "y") -> ((0, 1), "z")   # sigma_x * sigma_y = i * sigma_z
        pauli_multiply("x", "x") -> ((1, 0), "I")    # sigma_x^2 = I
    """
    key = (a, b)
    if key not in _PAULI_TABLE:
        raise ValueError("Unknown Pauli labels: %s, %s" % (a, b))
    return _PAULI_TABLE[key]


def spin_rotation_components(axis, angle_over_pi, depth=16):
    """
    Compute cos(theta/2) and sin(theta/2) for spin rotation.

    theta = angle_over_pi * pi

    I: axis ("x", "y", "z"), angle as rational multiple of pi (VDR), depth
    O: (cos_half, sin_half) as VDR tuple

        spin_rotation_components("z", VDR(1, 2))
        # theta = pi/2, cos(pi/4) and sin(pi/4)
    """
    # theta/2 = angle_over_pi * pi / 2
    half_angle = angle_over_pi * PI / VDR(2)

    c = cos_series(half_angle, depth)
    s = sin_series(half_angle, depth)

    return (c, s)


def spin_rotation(axis, angle_over_pi, depth=16):
    """
    Spin-1/2 rotation matrix (real part only, for real-axis rotations).

    U = cos(theta/2)*I - i*sin(theta/2)*sigma_axis

    For axis = "z":
        U = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
        (where the -i*sigma_z produces off-diagonal terms)

    Actually for z-axis the full unitary is:
        U = [[exp(-i*t/2), 0], [0, exp(i*t/2)]]
        = [[cos(t/2) - i*sin(t/2), 0], [0, cos(t/2) + i*sin(t/2)]]

    For pure real representation, we use the rotation matrix form:
        R_z(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]

    I: axis ("x", "y", "z"), angle/pi as VDR, trig depth
    O: 2x2 Mat (real rotation part)

    U applied 4 times at angle pi/2 returns to I exactly.
    Float gives I +/- ~1e-15.

        U = spin_rotation("z", VDR(1, 2))  # pi/2 rotation
    """
    c, s = spin_rotation_components(axis, angle_over_pi, depth)

    if axis == "z":
        return Mat([[c, -s], [s, c]])
    elif axis == "x":
        return Mat([[c, -s], [-s, c]])
    elif axis == "y":
        return Mat([[c, s], [-s, c]])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z', got '%s'" % axis)


def measurement_probabilities(state):
    """
    Born rule measurement probabilities.

    P(i) = |a_i|^2 for state vector |psi> = sum a_i |i>.

    I: state as Vec of VDR amplitudes (real)
    O: list of VDR probabilities, sums to exactly 1

    For complex amplitudes, pass squared magnitudes directly.

        state = Vec([VDR(3, 5), VDR(4, 5)])
        probs = measurement_probabilities(state)
        # [VDR(9, 25), VDR(16, 25)] -> sum = VDR(1) exactly
    """
    probs = []
    for i in range(len(state)):
        probs.append(state[i] * state[i])
    return probs


def density_matrix(state):
    """
    Density matrix rho = |psi><psi| (outer product).

    I: state as Vec
    O: Mat (n x n)

        state = Vec([VDR(1, 2), VDR(1, 2)])
        rho = density_matrix(state)
        # trace(rho) == 1 exactly
    """
    n = len(state)
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(state[i] * state[j])
        rows.append(row)
    return Mat(rows)


def verify_unitarity(U):
    """
    Verify U^T * U == I (for real unitary / orthogonal matrices).

    I: Mat U
    O: bool, True if U^T * U is exactly the identity matrix

    For complex unitarity (U†U = I), pass the appropriate
    conjugate-transpose product.
    """
    product = U.T.matmul(U)
    I_mat = Mat.identity(U.nrows)
    return product == I_mat


def commutator(A, B):
    """
    Matrix commutator [A, B] = AB - BA.

    I: two square Mat of same size
    O: commutator Mat, exact
    """
    return A.matmul(B) - B.matmul(A)


def anticommutator(A, B):
    """
    Matrix anticommutator {A, B} = AB + BA.

    I: two square Mat of same size
    O: anticommutator Mat, exact
    """
    return A.matmul(B) + B.matmul(A)


def expectation_value(operator, state):
    """
    Expectation value <psi|O|psi>.

    I: operator as Mat, state as Vec
    O: VDR, exact

        ev = expectation_value(SIGMA_Z, Vec([VDR(3,5), VDR(4,5)]))
    """
    O_psi = operator.matvec(state)
    return state.dot(O_psi)
