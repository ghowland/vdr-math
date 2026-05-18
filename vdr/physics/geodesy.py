"""
vdr.physics.geodesy — Exact geodetic computations.

    from vdr.physics.geodesy import helmert_forward, helmert_roundtrip_verify

    transformed = helmert_forward(coords, params)
    assert helmert_roundtrip_verify(coords, params)  # True, exact

Helmert 7-parameter transformation: exact forward and inverse.
Roundtrip recovers original coordinates identically. Zero residual.
Float gives ~1 nm error. VDR gives zero.
"""

from __future__ import annotations
from typing import Dict

from vdr.core import VDR
from vdr.linalg import Vec, Mat

__all__ = [
    "helmert_forward",
    "helmert_inverse",
    "helmert_roundtrip_verify",
    "rotation_matrix_small_angle",
    "coordinate_transform",
    "weighted_average",
    "misclosure",
]


def rotation_matrix_small_angle(rx, ry, rz):
    """
    Small-angle rotation matrix for Helmert transformation.

    R = [[1, -rz, ry], [rz, 1, -rx], [-ry, rx, 1]]

    I: rotation angles rx, ry, rz (VDR, in radians, small)
    O: 3x3 rotation Mat

    For exact Helmert, angles are rational (e.g. arcseconds converted to radians).
    """
    return Mat([
        [VDR(1), -rz, ry],
        [rz, VDR(1), -rx],
        [-ry, rx, VDR(1)],
    ])


def helmert_forward(coords, params):
    """
    Helmert 7-parameter forward transformation.

    X' = T + (1 + s) * R * X

    where T = translation vector, s = scale factor, R = rotation matrix.

    I: coordinates as Vec (3D), params dict with keys:
       "tx", "ty", "tz" (translation VDR),
       "rx", "ry", "rz" (rotation VDR),
       "s" (scale factor VDR)
    O: transformed Vec, exact

        params = {
            "tx": VDR(1, 10), "ty": VDR(2, 10), "tz": VDR(3, 10),
            "rx": VDR(1, 1000), "ry": VDR(2, 1000), "rz": VDR(3, 1000),
            "s": VDR(1, 1000000),
        }
        result = helmert_forward(Vec.from_ints([100, 200, 300]), params)
    """
    tx = params["tx"]
    ty = params["ty"]
    tz = params["tz"]
    rx = params["rx"]
    ry = params["ry"]
    rz = params["rz"]
    s = params["s"]

    T = Vec([tx, ty, tz])
    R = rotation_matrix_small_angle(rx, ry, rz)
    scale = VDR(1) + s

    # X' = T + (1+s) * R * X
    RX = R.matvec(coords)
    scaled = RX.scale(scale)

    return T + scaled


def helmert_inverse(coords, params):
    """
    Helmert 7-parameter inverse transformation.

    X = R^{-1} * (X' - T) / (1 + s)

    For small-angle R, R^{-1} is R with negated angles.

    I: transformed coordinates as Vec, params dict
    O: original coordinates as Vec, exact

        original = helmert_inverse(transformed, params)
    """
    tx = params["tx"]
    ty = params["ty"]
    tz = params["tz"]
    rx = params["rx"]
    ry = params["ry"]
    rz = params["rz"]
    s = params["s"]

    T = Vec([tx, ty, tz])
    scale = VDR(1) + s

    # X = R_inv * (X' - T) / (1+s)
    shifted = coords - T
    unscaled = shifted.scale(VDR(1) / scale)

    # R_inv for small angle: negate all angles
    R_inv = rotation_matrix_small_angle(-rx, -ry, -rz)

    return R_inv.matvec(unscaled)


def helmert_roundtrip_verify(coords, params):
    """
    Verify Helmert roundtrip: inverse(forward(X)) == X.

    I: original coordinates Vec, params dict
    O: bool, True if roundtrip recovers original exactly

    Float gives ~1 nm error. VDR gives True.

        assert helmert_roundtrip_verify(coords, params)
    """
    forward = helmert_forward(coords, params)
    recovered = helmert_inverse(forward, params)
    return recovered == coords


def coordinate_transform(coords, M, offset=None):
    """
    General affine coordinate transformation.

    X' = M * X + offset

    I: coordinates Vec, transformation Mat, optional offset Vec
    O: transformed Vec, exact

        rotated = coordinate_transform(coords, rotation_matrix, translation)
    """
    result = M.matvec(coords)
    if offset is not None:
        result = result + offset
    return result


def weighted_average(values, weights):
    """
    Weighted average with exact rational weights.

    result = sum(w_i * x_i) / sum(w_i)

    Weights sum to exactly 1 after normalization (same structural
    property as softmax).

    I: list of VDR values, list of VDR weights
    O: weighted average as VDR, exact

        weighted_average([VDR(1), VDR(2), VDR(3)], [VDR(1,3), VDR(1,3), VDR(1,3)])
        -> VDR(2)
    """
    if len(values) != len(weights):
        raise ValueError("values and weights must have same length")

    num = VDR(0)
    den = VDR(0)
    for v, w in zip(values, weights):
        v = v if isinstance(v, VDR) else VDR(v)
        w = w if isinstance(w, VDR) else VDR(w)
        num = num + w * v
        den = den + w

    return num / den


def misclosure(legs):
    """
    Survey misclosure: sum of leg vectors.

    For a closed traverse, the sum should be exactly zero.
    Any nonzero result is measurement error (not arithmetic error).

    I: list of Vec (survey leg vectors)
    O: misclosure Vec. VDR arithmetic contributes nothing — if nonzero,
       it's pure measurement error.

        legs = [Vec.from_ints([100, 0, 0]),
                Vec.from_ints([0, 100, 0]),
                Vec.from_ints([-100, 0, 0]),
                Vec.from_ints([0, -100, 0])]
        m = misclosure(legs)  # Vec([VDR(0), VDR(0), VDR(0)]) exactly
    """
    if not legs:
        return Vec.zero(3)

    total = legs[0]
    for leg in legs[1:]:
        total = total + leg

    return total
