"""
vdr.physics.structural — Exact structural mechanics.

    from vdr.physics.structural import solve_structure, verify_equilibrium

    u = solve_structure(K, F)
    assert verify_equilibrium(K, u, F)  # True, exact

Direct stiffness method with exact VDR arithmetic.
Equilibrium verified by exact equality, not residual tolerance.
Ill-conditioned structures solved exactly — condition number irrelevant.
"""

from __future__ import annotations
from typing import List, Tuple

from vdr.core import VDR
from vdr.linalg import Vec, Mat

__all__ = [
    "truss_element_stiffness",
    "assemble_stiffness",
    "apply_boundary_conditions",
    "solve_structure",
    "verify_equilibrium",
    "element_forces",
    "reaction_forces",
]


def truss_element_stiffness(EA, L, angle_cos, angle_sin):
    """
    2D truss element stiffness matrix in global coordinates (4x4).

    k_local = (EA/L) * [[1, -1], [-1, 1]]
    Transformed to global by rotation matrix T.

    I: axial stiffness EA (VDR), length L (VDR),
       direction cosines cos(theta) and sin(theta) (VDR)
    O: 4x4 Mat (DOFs: u1, v1, u2, v2)

        k = truss_element_stiffness(VDR(100), VDR(1), VDR(1), VDR(0))
        # horizontal member, EA=100, L=1
    """
    c = angle_cos
    s = angle_sin
    factor = EA / L

    cc = c * c * factor
    ss = s * s * factor
    cs = c * s * factor

    return Mat([
        [cc, cs, -cc, -cs],
        [cs, ss, -cs, -ss],
        [-cc, -cs, cc, cs],
        [-cs, -ss, cs, ss],
    ])


def assemble_stiffness(elements, n_dof):
    """
    Assemble global stiffness matrix from element contributions.

    I: list of (element_stiffness_Mat, dof_indices) tuples,
       total number of DOFs
    O: global stiffness Mat (n_dof x n_dof)

    Each element contributes to specific DOF positions.

        elements = [
            (k1, [0, 1, 2, 3]),   # element 1 connects DOFs 0,1,2,3
            (k2, [2, 3, 4, 5]),   # element 2 connects DOFs 2,3,4,5
        ]
        K = assemble_stiffness(elements, 6)
    """
    K = Mat.zero(n_dof, n_dof)
    rows = [[K[i, j] for j in range(n_dof)] for i in range(n_dof)]

    for k_elem, dofs in elements:
        n_elem = len(dofs)
        for i in range(n_elem):
            for j in range(n_elem):
                gi = dofs[i]
                gj = dofs[j]
                rows[gi][gj] = rows[gi][gj] + k_elem[i, j]

    return Mat(rows)


def apply_boundary_conditions(K, F, fixed_dofs):
    """
    Apply boundary conditions by zeroing rows/columns of fixed DOFs.

    Uses the penalty method alternative: replace fixed DOF equation
    with identity row (1 on diagonal, 0 elsewhere, 0 in F).

    I: stiffness Mat K, force Vec F, list of fixed DOF indices
    O: (K_modified, F_modified) as (Mat, Vec)

        K_bc, F_bc = apply_boundary_conditions(K, F, [0, 1])
    """
    n = K.nrows
    rows = [[K[i, j] for j in range(n)] for i in range(n)]
    f_data = [F[i] for i in range(n)]

    for dof in fixed_dofs:
        for j in range(n):
            rows[dof][j] = VDR(0)
            rows[j][dof] = VDR(0)
        rows[dof][dof] = VDR(1)
        f_data[dof] = VDR(0)

    return Mat(rows), Vec(f_data)


def solve_structure(K, F):
    """
    Solve Ku = F for displacements.

    I: global stiffness Mat K, force Vec F
    O: displacement Vec u, exact

    Uses Gaussian elimination for n >= 5, Cramer for smaller.
    Condition number irrelevant — exact arithmetic.

        u = solve_structure(K, F)
    """
    return K.solve(F)


def verify_equilibrium(K, u, F):
    """
    Verify K @ u == F exactly.

    I: stiffness Mat, displacement Vec, force Vec
    O: bool, True if exact structural equality

    Float gives residual ~1e-10. VDR gives True (exact zero residual).

        assert verify_equilibrium(K, u, F)
    """
    Ku = K.matvec(u)
    return Ku == F


def element_forces(k_elem, u_elem):
    """
    Compute element internal forces from element stiffness and displacements.

    f_elem = k_elem @ u_elem

    I: element stiffness Mat, element displacement Vec
    O: element force Vec, exact
    """
    return k_elem.matvec(u_elem)


def reaction_forces(K_full, u, F_applied):
    """
    Compute reaction forces at supports.

    R = K_full @ u - F_applied

    At free DOFs, R should be zero. At fixed DOFs, R gives the reaction.

    I: full (unreduced) stiffness Mat, displacement Vec, applied force Vec
    O: reaction force Vec, exact
    """
    Ku = K_full.matvec(u)
    n = len(u)
    result = []
    for i in range(n):
        result.append(Ku[i] - F_applied[i])
    return Vec(result)
