"""Gym 16 — Graph theory. 19/20 in VDR-3 (1 max-flow BFS bug)."""

import pytest
from vdr.core import VDR
from vdr.linalg import Vec, Mat
from vdr.math.graph import dijkstra, bellman_ford, prim_mst, floyd_warshall, pagerank


class TestDijkstra:
    def test_basic(self):
        adj = {
            0: [(1, VDR(1, 3)), (2, VDR(3, 4))],
            1: [(2, VDR(1, 2))],
            2: [],
        }
        dist = dijkstra(adj, 0)
        assert dist[0] == VDR(0)
        assert dist[1] == VDR(1, 3)
        # 0->2 direct = 3/4, 0->1->2 = 1/3 + 1/2 = 5/6
        assert dist[2] == VDR(3, 4)  # direct is shorter


class TestBellmanFord:
    def test_negative_weight(self):
        edges = [
            (0, 1, VDR(1)),
            (1, 2, VDR(-1, 2)),
            (0, 2, VDR(2)),
        ]
        dist = bellman_ford(3, edges, 0)
        assert dist[0] == VDR(0)
        assert dist[2] == VDR(1, 2)  # 0->1->2 = 1 - 1/2 = 1/2


class TestPrimMST:
    def test_triangle(self):
        adj = {
            0: [(1, VDR(1, 3)), (2, VDR(1, 2))],
            1: [(0, VDR(1, 3)), (2, VDR(1, 4))],
            2: [(0, VDR(1, 2)), (1, VDR(1, 4))],
        }
        edges, total = prim_mst(3, adj)
        assert len(edges) == 2
        assert total == VDR(1, 3) + VDR(1, 4)  # 7/12


class TestFloydWarshall:
    def test_3_nodes(self):
        dist = [
            [VDR(0), VDR(1, 3), None],
            [None, VDR(0), VDR(1, 2)],
            [VDR(2), None, VDR(0)],
        ]
        result = floyd_warshall(3, dist)
        assert result[0][0] == VDR(0)
        assert result[0][2] == VDR(5, 6)  # 0->1->2 = 1/3 + 1/2


class TestPageRank:
    def test_sum_to_one(self):
        P = Mat.from_fracs([
            [(1, 2), (1, 2)],
            [(1, 1), (0, 1)],
        ])
        pr = pagerank(P)
        total = pr[0] + pr[1]
        assert total == VDR(1)
