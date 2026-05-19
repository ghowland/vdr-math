<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/graph.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `math.graph`
vdr.math.graph — Exact graph algorithms over VDR. 

 from vdr.math.graph import dijkstra, pagerank, floyd_warshall 

 dist = dijkstra(adj, 0)        # exact shortest paths  pr = pagerank(adj_matrix)      # sums to exactly 1 

All weights, distances, and probabilities are exact VDR rationals. No float rounding in comparisons or accumulation. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/graph.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `dijkstra`

```python
dijkstra(adj, src)
```

Dijkstra's shortest path with exact rational weights. 

I: adjacency dict {node: [(neighbor, weight_VDR), ...]}, source node O: dict {node: distance_VDR} from source 

All comparisons exact. No float rounding in priority decisions. 

 adj = {  0: [(1, VDR(1,3)), (2, VDR(3,4))],  1: [(2, VDR(1,2))],  2: []  }  dijkstra(adj, 0) -> {0: VDR(0), 1: VDR(1,3), 2: VDR(3,4)} 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/graph.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bellman_ford`

```python
bellman_ford(n, edges, src)
```

Bellman-Ford shortest path with exact rational weights. Handles negative weights. 

I: number of nodes n, edge list [(u, v, weight_VDR), ...], source node O: list of VDR distances indexed by node, None for unreachable 

 bellman_ford(3, [(0,1,VDR(1,2)), (1,2,VDR(-1,3))], 0) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/graph.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `prim_mst`

```python
prim_mst(n, adj_list)
```

Prim's minimum spanning tree with exact rational weights. 

I: number of nodes, adjacency list {node: [(neighbor, weight), ...]} O: list of (u, v, weight) edges in MST, total weight as VDR 

 mst, total = prim_mst(4, {0:[(1,VDR(1,3)),(2,VDR(1,2))], ...}) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/graph.py#L178"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `floyd_warshall`

```python
floyd_warshall(n, dist_mat)
```

Floyd-Warshall all-pairs shortest paths. 

I: number of nodes, distance matrix as Mat (use large VDR for no-edge) O: shortest-path distance Mat, exact 

Use None/large value convention: entries that are "infinity" should be represented as a very large VDR or handled via the adj parameter. 

For simplicity, this version works with a plain list-of-lists where None means no direct edge. 

 dist = [[VDR(0), VDR(1,3), None],  [None, VDR(0), VDR(1,2)],  [VDR(2), None, VDR(0)]]  result = floyd_warshall(3, dist) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/graph.py#L219"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `pagerank`

```python
pagerank(adj_matrix, damping=None, method='linear')
```

PageRank via exact linear system solve. 

I: adjacency/transition matrix as Mat (row-stochastic),  damping factor (VDR, default 85/100) O: PageRank vector as Vec, sums to exactly 1 

 P = Mat.from_fracs([[(1,2),(1,2)],[(1,1),(0,1)]])  pr = pagerank(P)  # sums to exactly 1 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/graph.py#L255"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `max_flow`

```python
max_flow(n, cap, s, t)
```

Ford-Fulkerson max flow with BFS augmenting paths (Edmonds-Karp). 

I: number of nodes, capacity as list-of-lists or dict,  source s, sink t O: max flow value as VDR, exact 

 cap = [[VDR(0), VDR(3,4), VDR(1,2)],  [VDR(0), VDR(0), VDR(1,3)],  [VDR(0), VDR(0), VDR(0)]]  max_flow(3, cap, 0, 2) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
