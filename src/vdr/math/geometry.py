"""
vdr.math.geometry — Exact computational geometry over VDR.

    from vdr.math.geometry import line_intersect, polygon_area, circumcenter

    pt = line_intersect((VDR(0),VDR(0)), (VDR(2),VDR(2)),
                        (VDR(0),VDR(2)), (VDR(2),VDR(0)))
    # (VDR(1), VDR(1)) exact

No epsilon. No tolerance. Point-in-triangle is exact boolean.
Barycentric coordinates sum to exactly 1.
"""

from __future__ import annotations
from typing import Tuple, List

from vdr.core import VDR

__all__ = [
    "line_intersect",
    "polygon_area",
    "shoelace_signed",
    "barycentric",
    "point_in_triangle",
    "dist_sq",
    "circumcenter",
    "midpoint",
    "collinear",
]

Point = Tuple[VDR, VDR]


def _v(x):
    if isinstance(x, VDR):
        return x
    return VDR(x)


def line_intersect(p1, p2, p3, p4):
    """
    Intersection of lines through (p1,p2) and (p3,p4).

    I: four points as (VDR, VDR) tuples
    O: intersection point (x, y) as tuple of VDR, exact

    Uses Cramer's rule on the line equations.
    Raises ValueError if lines are parallel.

        line_intersect((VDR(0),VDR(0)), (VDR(2),VDR(2)),
                       (VDR(0),VDR(2)), (VDR(2),VDR(0)))
        -> (VDR(1), VDR(1))
    """
    x1, y1 = _v(p1[0]), _v(p1[1])
    x2, y2 = _v(p2[0]), _v(p2[1])
    x3, y3 = _v(p3[0]), _v(p3[1])
    x4, y4 = _v(p4[0]), _v(p4[1])

    # line 1: (y2-y1)x - (x2-x1)y = (y2-y1)x1 - (x2-x1)y1
    # line 2: (y4-y3)x - (x4-x3)y = (y4-y3)x3 - (x4-x3)y3
    a1 = y2 - y1
    b1 = -(x2 - x1)
    c1 = a1 * x1 + b1 * y1

    a2 = y4 - y3
    b2 = -(x4 - x3)
    c2 = a2 * x3 + b2 * y3

    det = a1 * b2 - a2 * b1
    if det == VDR(0):
        raise ValueError("Lines are parallel (det = 0)")

    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det

    return (x, y)


def shoelace_signed(vertices):
    """
    Signed area of polygon via Shoelace formula.

    I: list of vertices as (VDR, VDR) tuples, in order
    O: signed area as VDR (positive = counterclockwise)

    A = (1/2) * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    """
    n = len(vertices)
    if n < 3:
        raise ValueError("Polygon needs at least 3 vertices")

    total = VDR(0)
    for i in range(n):
        j = (i + 1) % n
        xi, yi = _v(vertices[i][0]), _v(vertices[i][1])
        xj, yj = _v(vertices[j][0]), _v(vertices[j][1])
        total = total + xi * yj - xj * yi

    return total / VDR(2)


def polygon_area(vertices):
    """
    Unsigned area of polygon via Shoelace formula.

    I: list of vertices as (VDR, VDR) tuples
    O: area as VDR, exact
    """
    signed = shoelace_signed(vertices)
    return abs(signed)


def barycentric(p, a, b, c):
    """
    Barycentric coordinates of point p with respect to triangle (a, b, c).

    I: point p and triangle vertices a, b, c as (VDR, VDR) tuples
    O: (lambda1, lambda2, lambda3) as VDR tuple, sums to exactly 1

    Uses the area method:
        lambda_i = area(p, other two vertices) / area(a, b, c)

        barycentric((VDR(1,3), VDR(1,3)),
                    (VDR(0),VDR(0)), (VDR(1),VDR(0)), (VDR(0),VDR(1)))
        -> (VDR(1,3), VDR(1,3), VDR(1,3))
    """
    px, py = _v(p[0]), _v(p[1])
    ax, ay = _v(a[0]), _v(a[1])
    bx, by = _v(b[0]), _v(b[1])
    cx, cy = _v(c[0]), _v(c[1])

    # area of full triangle (signed, 2x)
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)

    if denom == VDR(0):
        raise ValueError("Triangle is degenerate (zero area)")

    # barycentric coords
    l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
    l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
    l3 = VDR(1) - l1 - l2

    return (l1, l2, l3)


def point_in_triangle(p, a, b, c):
    """
    Test whether point p is inside triangle (a, b, c).

    I: point and triangle vertices as (VDR, VDR) tuples
    O: bool — True if inside or on boundary, exact

    No epsilon. Exact rational comparison.
    """
    l1, l2, l3 = barycentric(p, a, b, c)
    return l1 >= VDR(0) and l2 >= VDR(0) and l3 >= VDR(0)


def dist_sq(p1, p2):
    """
    Squared Euclidean distance between two points.

    I: two points as (VDR, VDR) tuples
    O: squared distance as VDR, exact (avoids sqrt)

        dist_sq((VDR(0),VDR(0)), (VDR(3),VDR(4))) -> VDR(25)
    """
    dx = _v(p2[0]) - _v(p1[0])
    dy = _v(p2[1]) - _v(p1[1])
    return dx * dx + dy * dy


def circumcenter(a, b, c):
    """
    Circumcenter of triangle (a, b, c).

    I: three vertices as (VDR, VDR) tuples
    O: circumcenter (x, y) as tuple of VDR, exact

    Equidistant from all three vertices:
        dist_sq(center, a) == dist_sq(center, b) == dist_sq(center, c)
    """
    ax, ay = _v(a[0]), _v(a[1])
    bx, by = _v(b[0]), _v(b[1])
    cx, cy = _v(c[0]), _v(c[1])

    # solve the perpendicular bisector system
    # |PA|^2 = |PB|^2 and |PA|^2 = |PC|^2
    # expanding and simplifying gives two linear equations in (x, y)

    # 2(bx-ax)x + 2(by-ay)y = bx^2 - ax^2 + by^2 - ay^2
    # 2(cx-ax)x + 2(cy-ay)y = cx^2 - ax^2 + cy^2 - ay^2

    a1 = VDR(2) * (bx - ax)
    b1 = VDR(2) * (by - ay)
    c1 = bx * bx - ax * ax + by * by - ay * ay

    a2 = VDR(2) * (cx - ax)
    b2 = VDR(2) * (cy - ay)
    c2 = cx * cx - ax * ax + cy * cy - ay * ay

    det = a1 * b2 - a2 * b1
    if det == VDR(0):
        raise ValueError("Triangle is degenerate (collinear points)")

    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det

    return (x, y)


def midpoint(p1, p2):
    """
    Midpoint of two points.

    I: two points as (VDR, VDR) tuples
    O: midpoint as (VDR, VDR), exact
    """
    x = (_v(p1[0]) + _v(p2[0])) / VDR(2)
    y = (_v(p1[1]) + _v(p2[1])) / VDR(2)
    return (x, y)


def collinear(p1, p2, p3):
    """
    Test whether three points are collinear.

    I: three points as (VDR, VDR) tuples
    O: bool, exact

    Uses signed area = 0 test.
    """
    area = shoelace_signed([p1, p2, p3])
    return area == VDR(0)
