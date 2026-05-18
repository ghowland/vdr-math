"""
vdr.math.graph — Exact graph algorithms over VDR.

    from vdr.math.graph import dijkstra, pagerank, floyd_warshall

    dist = dijkstra(adj, 0)        # exact shortest paths
    pr = pagerank(adj_matrix)      # sums to exactly 1

All weights, distances, and probabilities are exact VDR rationals.
No float rounding in comparisons or accumulation.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

from vdr.core import VDR
from vdr.linalg import Vec, Mat

__all__ = [
    "dijkstra",
    "bellman_ford",
    "prim_mst",
    "floyd_warshall",
    "pagerank",
    "max_flow",
]


# Sentinel for infinity in graph algorithms
_INF = None


def _less(a, b):
    """Compare two values where None represents infinity."""
    if a is None:
        return False
    if b is None:
        return True
    return a < b


def _add_dist(a, b):
    """Add two distances where None represents infinity."""
    if a is None or b is None:
        return None
    return a + b


def dijkstra(adj, src):
    """
    Dijkstra's shortest path with exact rational weights.

    I: adjacency dict {node: [(neighbor, weight_VDR), ...]}, source node
    O: dict {node: distance_VDR} from source

    All comparisons exact. No float rounding in priority decisions.

        adj = {
            0: [(1, VDR(1,3)), (2, VDR(3,4))],
            1: [(2, VDR(1,2))],
            2: []
        }
        dijkstra(adj, 0) -> {0: VDR(0), 1: VDR(1,3), 2: VDR(3,4)}
    """
    # collect all nodes
    nodes = set(adj.keys())
    for node in list(adj.keys()):
        for neighbor, _ in adj[node]:
            nodes.add(neighbor)

    dist = {n: _INF for n in nodes}
    dist[src] = VDR(0)
    visited = set()

    while len(visited) < len(nodes):
        # find unvisited node with smallest distance
        current = None
        for n in nodes:
            if n in visited:
                continue
            if dist[n] is None:
                continue
            if current is None or dist[n] < dist[current]:
                current = n

        if current is None:
            break

        visited.add(current)

        if current not in adj:
            continue

        for neighbor, weight in adj[current]:
            new_dist = dist[current] + weight
            if dist[neighbor] is None or new_dist < dist[neighbor]:
                dist[neighbor] = new_dist

    return dist


def bellman_ford(n, edges, src):
    """
    Bellman-Ford shortest path with exact rational weights.
    Handles negative weights.

    I: number of nodes n, edge list [(u, v, weight_VDR), ...], source node
    O: list of VDR distances indexed by node, None for unreachable

        bellman_ford(3, [(0,1,VDR(1,2)), (1,2,VDR(-1,3))], 0)
    """
    dist = [_INF] * n
    dist[src] = VDR(0)

    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] is not None:
                new_dist = dist[u] + w
                if dist[v] is None or new_dist < dist[v]:
                    dist[v] = new_dist
                    updated = True
        if not updated:
            break

    # check for negative cycles
    for u, v, w in edges:
        if dist[u] is not None:
            if dist[v] is None or dist[u] + w < dist[v]:
                raise ValueError("Graph contains a negative-weight cycle")

    return dist


def prim_mst(n, adj_list):
    """
    Prim's minimum spanning tree with exact rational weights.

    I: number of nodes, adjacency list {node: [(neighbor, weight), ...]}
    O: list of (u, v, weight) edges in MST, total weight as VDR

        mst, total = prim_mst(4, {0:[(1,VDR(1,3)),(2,VDR(1,2))], ...})
    """
    if n == 0:
        return [], VDR(0)

    in_mst = set()
    mst_edges = []
    total_weight = VDR(0)

    # start from node 0
    in_mst.add(0)

    while len(in_mst) < n:
        best_edge = None
        best_weight = None

        for u in in_mst:
            if u not in adj_list:
                continue
            for v, w in adj_list[u]:
                if v not in in_mst:
                    if best_weight is None or w < best_weight:
                        best_edge = (u, v, w)
                        best_weight = w

        if best_edge is None:
            break  # disconnected graph

        u, v, w = best_edge
        in_mst.add(v)
        mst_edges.append(best_edge)
        total_weight = total_weight + w

    return mst_edges, total_weight


def floyd_warshall(n, dist_mat):
    """
    Floyd-Warshall all-pairs shortest paths.

    I: number of nodes, distance matrix as Mat (use large VDR for no-edge)
    O: shortest-path distance Mat, exact

    Use None/large value convention: entries that are "infinity" should
    be represented as a very large VDR or handled via the adj parameter.

    For simplicity, this version works with a plain list-of-lists
    where None means no direct edge.

        dist = [[VDR(0), VDR(1,3), None],
                [None, VDR(0), VDR(1,2)],
                [VDR(2), None, VDR(0)]]
        result = floyd_warshall(3, dist)
    """
    # work on mutable copy
    d = []
    for i in range(n):
        row = []
        for j in range(n):
            if isinstance(dist_mat, Mat):
                val = dist_mat[i, j]
                row.append(val)
            else:
                row.append(dist_mat[i][j])
        d.append(row)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                through_k = _add_dist(d[i][k], d[k][j])
                if through_k is not None:
                    if d[i][j] is None or through_k < d[i][j]:
                        d[i][j] = through_k

    return d


def pagerank(adj_matrix, damping=None, method="linear"):
    """
    PageRank via exact linear system solve.

    I: adjacency/transition matrix as Mat (row-stochastic),
       damping factor (VDR, default 85/100)
    O: PageRank vector as Vec, sums to exactly 1

        P = Mat.from_fracs([[(1,2),(1,2)],[(1,1),(0,1)]])
        pr = pagerank(P)  # sums to exactly 1
    """
    if damping is None:
        damping = VDR(85, 100)

    n = adj_matrix.nrows

    # PageRank: pi = d * pi * P + (1-d)/n * e
    # Rearranging: pi * (I - d*P^T) = (1-d)/n * e
    # Or solve (I - d*P^T)^T * pi^T = (1-d)/n * e^T

    I_mat = Mat.identity(n)
    PT = adj_matrix.T
    dPT = PT.scale(damping)
    A = I_mat - dPT

    # RHS: (1-d)/n for each component
    one_minus_d = VDR(1) - damping
    rhs_val = one_minus_d / VDR(n)
    rhs = Vec([rhs_val] * n)

    # Solve A * pi = rhs
    pi = A.solve(rhs)

    return pi


def max_flow(n, cap, s, t):
    """
    Ford-Fulkerson max flow with BFS augmenting paths (Edmonds-Karp).

    I: number of nodes, capacity as list-of-lists or dict,
       source s, sink t
    O: max flow value as VDR, exact

        cap = [[VDR(0), VDR(3,4), VDR(1,2)],
               [VDR(0), VDR(0), VDR(1,3)],
               [VDR(0), VDR(0), VDR(0)]]
        max_flow(3, cap, 0, 2)
    """
    # build residual capacity as mutable structure
    res = []
    for i in range(n):
        row = []
        for j in range(n):
            if isinstance(cap, Mat):
                row.append(cap[i, j])
            elif isinstance(cap[i], list):
                row.append(cap[i][j] if isinstance(cap[i][j], VDR) else VDR(cap[i][j]))
            else:
                row.append(VDR(0))
        res.append(row)

    total_flow = VDR(0)

    while True:
        # BFS to find augmenting path
        parent = [None] * n
        visited = [False] * n
        visited[s] = True
        queue = [s]
        found = False

        while queue and not found:
            u = queue.pop(0)
            for v in range(n):
                if not visited[v] and res[u][v] > VDR(0):
                    visited[v] = True
                    parent[v] = u
                    if v == t:
                        found = True
                        break
                    queue.append(v)

        if not found:
            break

        # find bottleneck
        path_flow = None
        v = t
        while v != s:
            u = parent[v]
            if path_flow is None or res[u][v] < path_flow:
                path_flow = res[u][v]
            v = u

        # update residual
        v = t
        while v != s:
            u = parent[v]
            res[u][v] = res[u][v] - path_flow
            res[v][u] = res[v][u] + path_flow
            v = u

        total_flow = total_flow + path_flow

    return total_flow

