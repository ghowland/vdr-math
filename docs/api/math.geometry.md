<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/geometry.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `math.geometry`
vdr.math.geometry — Exact computational geometry over VDR. 

 from vdr.math.geometry import line_intersect, polygon_area, circumcenter 

 pt = line_intersect((VDR(0),VDR(0)), (VDR(2),VDR(2)),  (VDR(0),VDR(2)), (VDR(2),VDR(0)))  # (VDR(1), VDR(1)) exact 

No epsilon. No tolerance. Point-in-triangle is exact boolean. Barycentric coordinates sum to exactly 1. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/geometry.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `line_intersect`

```python
line_intersect(p1, p2, p3, p4)
```

Intersection of lines through (p1,p2) and (p3,p4). 

I: four points as (VDR, VDR) tuples O: intersection point (x, y) as tuple of VDR, exact 

Uses Cramer's rule on the line equations. Raises ValueError if lines are parallel. 

 line_intersect((VDR(0),VDR(0)), (VDR(2),VDR(2)),  (VDR(0),VDR(2)), (VDR(2),VDR(0))) 
    -> (VDR(1), VDR(1)) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/geometry.py#L79"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `shoelace_signed`

```python
shoelace_signed(vertices)
```

Signed area of polygon via Shoelace formula. 

I: list of vertices as (VDR, VDR) tuples, in order O: signed area as VDR (positive = counterclockwise) 

A = (1/2) * |sum(x_i * y_{i+1} - x_{i+1} * y_i)| 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/geometry.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `polygon_area`

```python
polygon_area(vertices)
```

Unsigned area of polygon via Shoelace formula. 

I: list of vertices as (VDR, VDR) tuples O: area as VDR, exact 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/geometry.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `barycentric`

```python
barycentric(p, a, b, c)
```

Barycentric coordinates of point p with respect to triangle (a, b, c). 

I: point p and triangle vertices a, b, c as (VDR, VDR) tuples O: (lambda1, lambda2, lambda3) as VDR tuple, sums to exactly 1 

Uses the area method:  lambda_i = area(p, other two vertices) / area(a, b, c) 

 barycentric((VDR(1,3), VDR(1,3)),  (VDR(0),VDR(0)), (VDR(1),VDR(0)), (VDR(0),VDR(1))) 
    -> (VDR(1,3), VDR(1,3), VDR(1,3)) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/geometry.py#L146"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `point_in_triangle`

```python
point_in_triangle(p, a, b, c)
```

Test whether point p is inside triangle (a, b, c). 

I: point and triangle vertices as (VDR, VDR) tuples O: bool — True if inside or on boundary, exact 

No epsilon. Exact rational comparison. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/geometry.py#L159"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `dist_sq`

```python
dist_sq(p1, p2)
```

Squared Euclidean distance between two points. 

I: two points as (VDR, VDR) tuples O: squared distance as VDR, exact (avoids sqrt) 

 dist_sq((VDR(0),VDR(0)), (VDR(3),VDR(4))) -> VDR(25) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/geometry.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `circumcenter`

```python
circumcenter(a, b, c)
```

Circumcenter of triangle (a, b, c). 

I: three vertices as (VDR, VDR) tuples O: circumcenter (x, y) as tuple of VDR, exact 

Equidistant from all three vertices:  dist_sq(center, a) == dist_sq(center, b) == dist_sq(center, c) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/geometry.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `midpoint`

```python
midpoint(p1, p2)
```

Midpoint of two points. 

I: two points as (VDR, VDR) tuples O: midpoint as (VDR, VDR), exact 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/geometry.py#L224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `collinear`

```python
collinear(p1, p2, p3)
```

Test whether three points are collinear. 

I: three points as (VDR, VDR) tuples O: bool, exact 

Uses signed area = 0 test. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
