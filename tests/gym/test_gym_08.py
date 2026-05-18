"""Gym 08 — Computational geometry. 19/19 in VDR-2."""

import pytest
from vdr.core import VDR
from vdr.math.geometry import (
    line_intersect, polygon_area, barycentric, point_in_triangle,
    dist_sq, circumcenter, collinear,
)


class TestLineIntersect:
    def test_basic(self):
        pt = line_intersect(
            (VDR(0), VDR(0)), (VDR(2), VDR(2)),
            (VDR(0), VDR(2)), (VDR(2), VDR(0)),
        )
        assert pt == (VDR(1), VDR(1))

    def test_parallel_raises(self):
        with pytest.raises(ValueError):
            line_intersect(
                (VDR(0), VDR(0)), (VDR(1), VDR(1)),
                (VDR(0), VDR(1)), (VDR(1), VDR(2)),
            )


class TestPolygonArea:
    def test_unit_square(self):
        verts = [(VDR(0), VDR(0)), (VDR(1), VDR(0)),
                 (VDR(1), VDR(1)), (VDR(0), VDR(1))]
        assert polygon_area(verts) == VDR(1)

    def test_triangle(self):
        verts = [(VDR(0), VDR(0)), (VDR(4), VDR(0)), (VDR(0), VDR(3))]
        assert polygon_area(verts) == VDR(6)


class TestBarycentric:
    def test_centroid(self):
        a = (VDR(0), VDR(0))
        b = (VDR(1), VDR(0))
        c = (VDR(0), VDR(1))
        l1, l2, l3 = barycentric((VDR(1, 3), VDR(1, 3)), a, b, c)
        assert l1 == VDR(1, 3)
        assert l2 == VDR(1, 3)
        assert l3 == VDR(1, 3)
        assert l1 + l2 + l3 == VDR(1)

    def test_on_edge(self):
        a = (VDR(0), VDR(0))
        b = (VDR(1), VDR(0))
        c = (VDR(0), VDR(1))
        l1, l2, l3 = barycentric((VDR(1, 3), VDR(0)), a, b, c)
        assert l3 == VDR(0)  # on edge AB


class TestPointInTriangle:
    def test_inside(self):
        a = (VDR(0), VDR(0))
        b = (VDR(4), VDR(0))
        c = (VDR(0), VDR(4))
        assert point_in_triangle((VDR(1), VDR(1)), a, b, c)

    def test_outside(self):
        a = (VDR(0), VDR(0))
        b = (VDR(1), VDR(0))
        c = (VDR(0), VDR(1))
        assert not point_in_triangle((VDR(1), VDR(1)), a, b, c)


class TestCircumcenter:
    def test_known(self):
        a = (VDR(0), VDR(0))
        b = (VDR(6), VDR(0))
        c = (VDR(0), VDR(8))
        cx, cy = circumcenter(a, b, c)
        assert cx == VDR(3)
        assert cy == VDR(4)
        # verify equidistant
        d0 = dist_sq((cx, cy), a)
        d1 = dist_sq((cx, cy), b)
        d2 = dist_sq((cx, cy), c)
        assert d0 == d1
        assert d1 == d2
