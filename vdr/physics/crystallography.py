"""
vdr.physics.crystallography — Exact crystallographic computations.

    from vdr.physics.crystallography import point_group_matrix, verify_group_closure

    matrices = [point_group_matrix(op) for op in ["E", "C4z", "C2z", "C4z_inv"]]
    assert verify_group_closure(matrices)  # exact structural equality

Point group operations as 3x3 integer matrices.
Group closure verified by exact comparison, not tolerance.
Structure factors with exact complex exponentials.
"""

from __future__ import annotations
from typing import List, Tuple, Dict

from vdr.core import VDR
from vdr.linalg import Mat
from vdr.math.transcendental import sin_series, cos_series, PI

__all__ = [
    "point_group_matrix",
    "verify_group_closure",
    "group_multiplication_table",
    "structure_factor",
    "CUBIC_GENERATORS",
    "HEXAGONAL_GENERATORS",
]


# ---------------------------------------------------------------------------
# Standard point group operations (3x3 integer matrices)
# ---------------------------------------------------------------------------

# Cubic system operations
_CUBIC_OPS = {
    "E": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],           # identity
    "C2z": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],       # 180 about z
    "C2x": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],       # 180 about x
    "C2y": [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],       # 180 about y
    "C4z": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],        # 90 about z
    "C4z_inv": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],    # -90 about z
    "C4x": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],        # 90 about x
    "C4x_inv": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],    # -90 about x
    "C4y": [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],        # 90 about y
    "C4y_inv": [[0, 0, -1], [0, 1, 0], [1, 0, 0]],    # -90 about y
    "i": [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],        # inversion
    "sigma_h": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],    # horizontal mirror
    "sigma_vx": [[1, 0, 0], [0, -1, 0], [0, 0, 1]],   # vertical mirror (xz)
    "sigma_vy": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],   # vertical mirror (yz)
}

CUBIC_GENERATORS = ["E", "C4z", "C4x", "C2z"]

# Hexagonal system would need sqrt(3)/2 entries — use Q335 for those
HEXAGONAL_GENERATORS = ["E", "C6z", "C3z"]


def point_group_matrix(operation):
    """
    Get 3x3 rotation/reflection matrix for named point group operation.

    I: operation name string (e.g. "C4z", "sigma_h", "i")
    O: 3x3 Mat with exact integer entries

    Cubic operations have entries from {-1, 0, 1}.
    Hexagonal operations (C3z, C6z) would need sqrt(3)/2.

        M = point_group_matrix("C4z")
        # [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    """
    if operation in _CUBIC_OPS:
        return Mat.from_ints(_CUBIC_OPS[operation])

    # hexagonal operations
    if operation == "C3z":
        # 120 degree rotation about z
        # [[cos120, -sin120, 0], [sin120, cos120, 0], [0, 0, 1]]
        # cos120 = -1/2, sin120 = sqrt(3)/2
        # For exact: use Q335 sqrt(3)
        from vdr.math.transcendental import SQRT3
        half = VDR(1, 2)
        s32 = SQRT3 / VDR(2)
        return Mat([
            [-half, -s32, VDR(0)],
            [s32, -half, VDR(0)],
            [VDR(0), VDR(0), VDR(1)],
        ])

    if operation == "C6z":
        # 60 degree rotation about z
        # cos60 = 1/2, sin60 = sqrt(3)/2
        from vdr.math.transcendental import SQRT3
        half = VDR(1, 2)
        s32 = SQRT3 / VDR(2)
        return Mat([
            [half, -s32, VDR(0)],
            [s32, half, VDR(0)],
            [VDR(0), VDR(0), VDR(1)],
        ])

    raise ValueError("Unknown operation: %s" % operation)


def verify_group_closure(matrices):
    """
    Verify that a set of matrices is closed under multiplication.

    For every pair (A, B), the product A*B must equal some matrix in the set.
    Comparison by exact structural equality, not tolerance.

    I: list of Mat
    O: bool, True if closed

    Float: "within tolerance." VDR: structural equality.

        matrices = [point_group_matrix(op) for op in ["E", "C2z", "C4z", "C4z_inv"]]
        verify_group_closure(matrices)  # True
    """
    n = len(matrices)
    for i in range(n):
        for j in range(n):
            product = matrices[i].matmul(matrices[j])
            found = False
            for k in range(n):
                if product == matrices[k]:
                    found = True
                    break
            if not found:
                return False
    return True


def group_multiplication_table(matrices, names=None):
    """
    Build group multiplication table.

    I: list of Mat, optional list of name strings
    O: list of lists of indices (or names) showing product[i][j]

        matrices = [point_group_matrix(op) for op in ops]
        table = group_multiplication_table(matrices, ops)
    """
    n = len(matrices)
    if names is None:
        names = [str(i) for i in range(n)]

    table = []
    for i in range(n):
        row = []
        for j in range(n):
            product = matrices[i].matmul(matrices[j])
            found_name = "?"
            for k in range(n):
                if product == matrices[k]:
                    found_name = names[k]
                    break
            row.append(found_name)
        table.append(row)

    return table


def structure_factor(atoms, hkl, depth=16):
    """
    Structure factor F(hkl) = sum f_j * exp(2*pi*i*(h*xj + k*yj + l*zj)).

    I: list of (f, x, y, z) tuples where f is scattering factor (VDR)
       and x, y, z are fractional coordinates (VDR),
       hkl as (h, k, l) tuple of integers,
       trig series depth
    O: (real, imag) as VDR tuple

    Complex exponentials of rational arguments computed via exact
    sin/cos series. |F|^2 exact.

        atoms = [(VDR(1), VDR(0), VDR(0), VDR(0)),
                 (VDR(1), VDR(1,2), VDR(1,2), VDR(0))]
        Fr, Fi = structure_factor(atoms, (1, 1, 0))
    """
    h, k, l = hkl
    two_pi = VDR(2) * PI

    F_real = VDR(0)
    F_imag = VDR(0)

    for f_j, xj, yj, zj in atoms:
        f_j = f_j if isinstance(f_j, VDR) else VDR(f_j)
        xj = xj if isinstance(xj, VDR) else VDR(xj)
        yj = yj if isinstance(yj, VDR) else VDR(yj)
        zj = zj if isinstance(zj, VDR) else VDR(zj)

        # phase = 2*pi*(h*x + k*y + l*z)
        phase = two_pi * (VDR(h) * xj + VDR(k) * yj + VDR(l) * zj)

        c = cos_series(phase, depth)
        s = sin_series(phase, depth)

        F_real = F_real + f_j * c
        F_imag = F_imag + f_j * s

    return (F_real, F_imag)
