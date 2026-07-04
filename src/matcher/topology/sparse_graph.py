"""Sparse graph primitives using scipy CSR matrices.

This module provides NetworkX-free graph operations for road network analysis,
designed to be Spark-compatible (CSR matrices serialize efficiently).

Core data structure: scipy.sparse.csr_matrix adjacency matrix
- Efficient for graph algorithms (connected components, BFS, etc.)
- Works well with numba for custom algorithms
- Serializes efficiently for Spark UDFs
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as scipy_connected_components


@dataclass
class SparseGraph:
    """CSR-based graph representation for road networks.

    Attributes:
        adjacency: CSR adjacency matrix (n_nodes x n_nodes)
        node_ids: List of original node IDs (maps matrix index -> node ID)
        node_to_idx: Dict mapping node ID -> matrix index
        edge_data: Optional dict mapping (u, v) -> edge attributes
    """

    adjacency: csr_matrix
    node_ids: list[Any]
    node_to_idx: dict[Any, int]
    edge_data: dict[tuple[Any, Any], dict[str, Any]] | None = None

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self.adjacency.shape[0]

    @property
    def n_edges(self) -> int:
        """Number of edges in the graph (undirected, so nnz/2)."""
        return self.adjacency.nnz // 2

    def degree(self, node: Any) -> int:
        """Get degree of a node.

        Args:
            node: Node ID

        Returns:
            Number of edges incident to node
        """
        if node not in self.node_to_idx:
            return 0
        idx = self.node_to_idx[node]
        return int(self.adjacency.indptr[idx + 1] - self.adjacency.indptr[idx])

    def degrees(self) -> dict[Any, int]:
        """Get degrees of all nodes.

        Returns:
            Dict mapping node ID -> degree
        """
        indptr = self.adjacency.indptr
        return {node_id: int(indptr[i + 1] - indptr[i]) for i, node_id in enumerate(self.node_ids)}

    def neighbors(self, node: Any) -> list[Any]:
        """Get neighbors of a node.

        Args:
            node: Node ID

        Returns:
            List of neighbor node IDs
        """
        if node not in self.node_to_idx:
            return []
        idx = self.node_to_idx[node]
        start = self.adjacency.indptr[idx]
        end = self.adjacency.indptr[idx + 1]
        neighbor_indices = self.adjacency.indices[start:end]
        return [self.node_ids[i] for i in neighbor_indices]

    def has_edge(self, u: Any, v: Any) -> bool:
        """Check if an edge exists between two nodes.

        Args:
            u: First node ID
            v: Second node ID

        Returns:
            True if edge exists
        """
        if u not in self.node_to_idx or v not in self.node_to_idx:
            return False
        i, j = self.node_to_idx[u], self.node_to_idx[v]
        return self.adjacency[i, j] != 0

    def get_edge_data(self, u: Any, v: Any) -> dict[str, Any] | None:
        """Get edge attributes.

        Args:
            u: First node ID
            v: Second node ID

        Returns:
            Edge attribute dict or None if no edge/no data
        """
        if self.edge_data is None:
            return None
        return self.edge_data.get((u, v)) or self.edge_data.get((v, u))


def build_graph_from_edges(
    edges: list[tuple[Any, Any]],
    node_attrs: dict[Any, dict[str, Any]] | None = None,
    edge_attrs: dict[tuple[Any, Any], dict[str, Any]] | None = None,
) -> SparseGraph:
    """Build a sparse graph from an edge list.

    Args:
        edges: List of (u, v) tuples representing edges
        node_attrs: Optional dict mapping node ID -> attributes (not stored in CSR)
        edge_attrs: Optional dict mapping (u, v) -> attributes

    Returns:
        SparseGraph instance
    """
    # Collect all unique nodes
    nodes = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)

    # Also include nodes from node_attrs that might not have edges
    if node_attrs:
        nodes.update(node_attrs.keys())

    node_ids = sorted(nodes) if all(isinstance(n, (int, float)) for n in nodes) else list(nodes)
    node_to_idx = {n: i for i, n in enumerate(node_ids)}
    n = len(node_ids)

    # Build COO format data
    rows = []
    cols = []
    for u, v in edges:
        i, j = node_to_idx[u], node_to_idx[v]
        # Add both directions for undirected graph
        rows.extend([i, j])
        cols.extend([j, i])

    # Create CSR matrix
    if rows:
        data = np.ones(len(rows), dtype=np.int32)
        adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))
        adjacency.sort_indices()  # Required for merge-based intersection in numba
    else:
        adjacency = csr_matrix((n, n), dtype=np.int32)

    return SparseGraph(
        adjacency=adjacency,
        node_ids=node_ids,
        node_to_idx=node_to_idx,
        edge_data=edge_attrs,
    )


def build_graph_from_node_pairs(
    from_nodes: list[Any],
    to_nodes: list[Any],
    node_ids: list[Any] | None = None,
) -> SparseGraph:
    """Build a sparse graph from parallel arrays of node pairs.

    More efficient than build_graph_from_edges when you have arrays.

    Args:
        from_nodes: Array of source node IDs
        to_nodes: Array of target node IDs
        node_ids: Optional list of all node IDs (to include isolated nodes)

    Returns:
        SparseGraph instance
    """
    # Collect all unique nodes
    nodes = set(from_nodes) | set(to_nodes)
    if node_ids is not None:
        nodes.update(node_ids)

    node_list = sorted(nodes) if all(isinstance(n, (int, float)) for n in nodes) else list(nodes)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)

    # Convert to indices
    rows = []
    cols = []
    for u, v in zip(from_nodes, to_nodes):
        i, j = node_to_idx[u], node_to_idx[v]
        rows.extend([i, j])
        cols.extend([j, i])

    # Create CSR matrix
    if rows:
        data = np.ones(len(rows), dtype=np.int32)
        adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))
        adjacency.sort_indices()
    else:
        adjacency = csr_matrix((n, n), dtype=np.int32)

    return SparseGraph(
        adjacency=adjacency,
        node_ids=node_list,
        node_to_idx=node_to_idx,
        edge_data=None,
    )


def connected_components(graph: SparseGraph) -> tuple[int, list[set[Any]]]:
    """Find connected components in the graph.

    Args:
        graph: SparseGraph instance

    Returns:
        Tuple of:
        - n_components: Number of connected components
        - components: List of sets, each containing node IDs in a component
    """
    if graph.n_nodes == 0:
        return 0, []

    n_components, labels = scipy_connected_components(
        graph.adjacency, directed=False, return_labels=True
    )

    # Group nodes by component
    component_nodes: dict[int, set[Any]] = {i: set() for i in range(n_components)}
    for idx, label in enumerate(labels):
        component_nodes[label].add(graph.node_ids[idx])

    return n_components, list(component_nodes.values())


def is_connected(graph: SparseGraph) -> bool:
    """Check if the graph is connected.

    Args:
        graph: SparseGraph instance

    Returns:
        True if graph has exactly one connected component
    """
    if graph.n_nodes == 0:
        return True
    n_components, _ = scipy_connected_components(
        graph.adjacency, directed=False, return_labels=True
    )
    return n_components == 1


def has_path(graph: SparseGraph, source: Any, target: Any) -> bool:
    """Check if a path exists between two nodes.

    Uses BFS for efficiency.

    Args:
        graph: SparseGraph instance
        source: Source node ID
        target: Target node ID

    Returns:
        True if a path exists
    """
    if source not in graph.node_to_idx or target not in graph.node_to_idx:
        return False

    if source == target:
        return True

    source_idx = graph.node_to_idx[source]
    target_idx = graph.node_to_idx[target]

    # BFS
    visited = np.zeros(graph.n_nodes, dtype=bool)
    visited[source_idx] = True
    frontier = [source_idx]
    indptr = graph.adjacency.indptr
    indices = graph.adjacency.indices

    while frontier:
        next_frontier = []
        for node_idx in frontier:
            start = indptr[node_idx]
            end = indptr[node_idx + 1]
            for neighbor_idx in indices[start:end]:
                if neighbor_idx == target_idx:
                    return True
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    next_frontier.append(neighbor_idx)
        frontier = next_frontier

    return False


def bfs_neighbors(
    graph: SparseGraph,
    start: Any,
    depth: int,
) -> set[Any]:
    """Find all nodes within `depth` hops of start node.

    Args:
        graph: SparseGraph instance
        start: Starting node ID
        depth: Maximum number of hops

    Returns:
        Set of node IDs within depth hops (including start)
    """
    if start not in graph.node_to_idx:
        return set()

    start_idx = graph.node_to_idx[start]
    visited = {start_idx}
    frontier = [start_idx]
    indptr = graph.adjacency.indptr
    indices = graph.adjacency.indices

    for _ in range(depth):
        next_frontier = []
        for node_idx in frontier:
            start_ptr = indptr[node_idx]
            end_ptr = indptr[node_idx + 1]
            for neighbor_idx in indices[start_ptr:end_ptr]:
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    next_frontier.append(neighbor_idx)
        frontier = next_frontier
        if not frontier:
            break

    return {graph.node_ids[i] for i in visited}


def compute_degree_signature(
    graph: SparseGraph,
    edge: tuple[Any, Any],
    radius: int = 2,
) -> tuple[int, ...]:
    """Compute degree signature for an edge's local neighborhood.

    Creates a sorted tuple of degrees for nodes within `radius` hops.
    Two edges with similar signatures have similar local topology.

    Args:
        graph: SparseGraph instance
        edge: Tuple of (from_node, to_node)
        radius: Number of hops to include

    Returns:
        Sorted tuple of degrees in the neighborhood
    """
    from_node, to_node = edge

    # Collect nodes within radius of both endpoints
    neighborhood = bfs_neighbors(graph, from_node, radius)
    neighborhood.update(bfs_neighbors(graph, to_node, radius))

    # Get sorted degrees
    indptr = graph.adjacency.indptr
    degrees = []
    for node in neighborhood:
        if node in graph.node_to_idx:
            idx = graph.node_to_idx[node]
            degrees.append(int(indptr[idx + 1] - indptr[idx]))

    return tuple(sorted(degrees))


def edge_is_bridge(graph: SparseGraph, edge: tuple[Any, Any]) -> bool:
    """Check if removing an edge would disconnect its endpoints.

    A bridge edge is one whose removal increases the number of
    connected components (i.e., disconnects the graph locally).

    Args:
        graph: SparseGraph instance
        edge: Tuple of (from_node, to_node)

    Returns:
        True if the edge is a bridge
    """
    from_node, to_node = edge

    if from_node not in graph.node_to_idx or to_node not in graph.node_to_idx:
        return False

    if not graph.has_edge(from_node, to_node):
        return False

    # Check if nodes are still connected without this edge
    # We do BFS from from_node, avoiding the direct edge to to_node
    from_idx = graph.node_to_idx[from_node]
    to_idx = graph.node_to_idx[to_node]

    visited = np.zeros(graph.n_nodes, dtype=bool)
    visited[from_idx] = True
    frontier = [from_idx]
    indptr = graph.adjacency.indptr
    indices = graph.adjacency.indices

    while frontier:
        next_frontier = []
        for node_idx in frontier:
            start = indptr[node_idx]
            end = indptr[node_idx + 1]
            for neighbor_idx in indices[start:end]:
                # Skip the direct edge we're testing
                if node_idx == from_idx and neighbor_idx == to_idx:
                    continue
                if node_idx == to_idx and neighbor_idx == from_idx:
                    continue

                if neighbor_idx == to_idx:
                    # Found alternative path
                    return False
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    next_frontier.append(neighbor_idx)
        frontier = next_frontier

    # Didn't find to_node via alternative path
    return True


# Numba-accelerated functions for graphlet features
_NUMBA_TRIANGLE_FUNCS: tuple | None = None


def _get_numba_triangle_functions():
    """Get numba-accelerated triangle/clustering functions (cached after first call).

    Functions are compiled on first call and cached to disk via numba's cache=True.

    Returns:
        Tuple of (count_triangles_per_node, compute_clustering_coefficients) functions
    """
    global _NUMBA_TRIANGLE_FUNCS
    if _NUMBA_TRIANGLE_FUNCS is not None:
        return _NUMBA_TRIANGLE_FUNCS

    from numba import njit

    @njit(cache=True)
    def count_triangles_per_node(
        n_nodes: int, indptr: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        """Count triangles involving each node using CSR arrays.

        A triangle involving node v is a 3-clique {v, u, w} where v-u, v-w, u-w all exist.
        Each triangle is counted once for each of its three vertices (matches NetworkX behavior).

        Args:
            n_nodes: Number of nodes
            indptr: CSR row pointers
            indices: CSR column indices (must be sorted)

        Returns:
            Array of triangle counts per node
        """
        result = np.zeros(n_nodes, dtype=np.int64)

        # First pass: count triangles with v as lowest index, add to all 3 nodes
        for v in range(n_nodes):
            start_v = indptr[v]
            end_v = indptr[v + 1]
            neighbors_v = indices[start_v:end_v]

            for i in range(len(neighbors_v)):
                u = neighbors_v[i]
                if u <= v:  # Only consider neighbors with higher index
                    continue

                start_u = indptr[u]
                end_u = indptr[u + 1]
                neighbors_u = indices[start_u:end_u]

                # Count common neighbors > u (each triangle counted once)
                # Merge-based intersection of sorted arrays
                p_v = i + 1  # Start after u in v's neighbors
                p_u = 0

                while p_v < len(neighbors_v) and p_u < len(neighbors_u):
                    w_v = neighbors_v[p_v]
                    w_u = neighbors_u[p_u]

                    if w_v == w_u and w_v > u:
                        # Found triangle (v, u, w) - add to all three nodes
                        result[v] += 1
                        result[u] += 1
                        result[w_v] += 1
                        p_v += 1
                        p_u += 1
                    elif w_v < w_u:
                        p_v += 1
                    else:
                        p_u += 1

        return result

    @njit(cache=True)
    def compute_clustering_coefficients(
        n_nodes: int, indptr: np.ndarray, indices: np.ndarray, triangles: np.ndarray
    ) -> np.ndarray:
        """Compute local clustering coefficient for each node.

        clustering(v) = triangles(v) / (degree(v) * (degree(v) - 1) / 2)

        For degree < 2 nodes the clustering coefficient is UNDEFINED (there is no
        pair of neighbors that could be connected), so we return NaN rather than
        0.0. Road networks are dominated by degree-1/2 nodes; forcing 0.0 makes an
        "undefined" node indistinguishable from a genuine hub measured at zero,
        which collapses the clustering feature to a constant. NaN preserves the
        distinction and passes through to XGBoost as a missing value.

        Args:
            n_nodes: Number of nodes
            indptr: CSR row pointers
            indices: CSR column indices
            triangles: Triangle counts per node

        Returns:
            Array of clustering coefficients per node (NaN where degree < 2)
        """
        result = np.zeros(n_nodes, dtype=np.float64)

        for v in range(n_nodes):
            degree = indptr[v + 1] - indptr[v]
            if degree < 2:
                result[v] = np.nan
            else:
                possible = degree * (degree - 1) // 2
                result[v] = triangles[v] / possible if possible > 0 else 0.0

        return result

    # Cache the functions
    _NUMBA_TRIANGLE_FUNCS = (count_triangles_per_node, compute_clustering_coefficients)
    return _NUMBA_TRIANGLE_FUNCS


def compute_triangles(graph: SparseGraph) -> dict[Any, int]:
    """Count triangles involving each node.

    Args:
        graph: SparseGraph instance

    Returns:
        Dict mapping node ID -> triangle count
    """
    if graph.n_nodes == 0:
        return {}

    count_triangles_numba, _ = _get_numba_triangle_functions()

    if count_triangles_numba is not None:
        # Numba-accelerated path
        triangles_arr = count_triangles_numba(
            graph.n_nodes,
            graph.adjacency.indptr.astype(np.int64),
            graph.adjacency.indices.astype(np.int64),
        )
        return {node_id: int(triangles_arr[i]) for i, node_id in enumerate(graph.node_ids)}
    else:
        # Pure Python fallback
        triangles = {node: 0 for node in graph.node_ids}
        indptr = graph.adjacency.indptr
        indices = graph.adjacency.indices

        for v_idx, v in enumerate(graph.node_ids):
            start_v = indptr[v_idx]
            end_v = indptr[v_idx + 1]
            neighbors_v = set(indices[start_v:end_v])

            for u_idx in indices[start_v:end_v]:
                if u_idx <= v_idx:
                    continue
                start_u = indptr[u_idx]
                end_u = indptr[u_idx + 1]
                neighbors_u = set(indices[start_u:end_u])

                # Common neighbors > u (each triangle counted once)
                common = [w for w in (neighbors_v & neighbors_u) if w > u_idx]
                # Add triangle count to all 3 nodes
                for w_idx in common:
                    triangles[v] += 1
                    triangles[graph.node_ids[u_idx]] += 1
                    triangles[graph.node_ids[w_idx]] += 1

        return triangles


def compute_clustering(graph: SparseGraph) -> dict[Any, float]:
    """Compute local clustering coefficient for each node.

    Args:
        graph: SparseGraph instance

    Returns:
        Dict mapping node ID -> clustering coefficient
    """
    if graph.n_nodes == 0:
        return {}

    count_triangles_numba, compute_clustering_numba = _get_numba_triangle_functions()

    if count_triangles_numba is not None and compute_clustering_numba is not None:
        # Numba-accelerated path
        indptr = graph.adjacency.indptr.astype(np.int64)
        indices_arr = graph.adjacency.indices.astype(np.int64)

        triangles_arr = count_triangles_numba(graph.n_nodes, indptr, indices_arr)
        clustering_arr = compute_clustering_numba(graph.n_nodes, indptr, indices_arr, triangles_arr)

        return {node_id: float(clustering_arr[i]) for i, node_id in enumerate(graph.node_ids)}
    else:
        # Pure Python fallback
        triangles = compute_triangles(graph)
        degrees = graph.degrees()
        clustering = {}

        for node in graph.node_ids:
            d = degrees[node]
            if d < 2:
                # Clustering is undefined for degree < 2 (see numba path above).
                # Return NaN so "undefined" is distinguishable from a measured zero.
                clustering[node] = float("nan")
            else:
                possible = d * (d - 1) // 2
                clustering[node] = triangles[node] / possible if possible > 0 else 0.0

        return clustering


def find_articulation_points(graph: SparseGraph) -> set[Any]:
    """Find articulation points (cut vertices) using Tarjan's algorithm.

    An articulation point is a node whose removal disconnects the graph.

    Args:
        graph: SparseGraph instance

    Returns:
        Set of node IDs that are articulation points
    """
    if graph.n_nodes < 3:
        return set()

    # Run Tarjan's algorithm (iterative to avoid recursion limit)
    indptr = graph.adjacency.indptr
    indices = graph.adjacency.indices
    n = graph.n_nodes

    visited = np.zeros(n, dtype=bool)
    disc = np.zeros(n, dtype=np.int64)  # Discovery times
    low = np.zeros(n, dtype=np.int64)  # Lowest reachable discovery time
    parent = np.full(n, -1, dtype=np.int64)
    ap = np.zeros(n, dtype=bool)  # Articulation points
    children_count = np.zeros(n, dtype=np.int64)  # Child count per node

    time_counter = 0

    # Process each connected component
    for start_node in range(n):
        if visited[start_node]:
            continue

        # Iterative DFS using explicit stack
        # Stack entries: (node, neighbor_iterator_index)
        stack: list[tuple[int, int]] = [(start_node, indptr[start_node])]
        visited[start_node] = True
        disc[start_node] = low[start_node] = time_counter
        time_counter += 1

        while stack:
            u, ptr = stack[-1]
            u_end = indptr[u + 1]

            # Find next unvisited neighbor
            found_child = False
            while ptr < u_end:
                v = indices[ptr]
                ptr += 1

                if not visited[v]:
                    # Tree edge: u -> v
                    children_count[u] += 1
                    parent[v] = u
                    visited[v] = True
                    disc[v] = low[v] = time_counter
                    time_counter += 1

                    # Update u's position and push v
                    stack[-1] = (u, ptr)
                    stack.append((v, indptr[v]))
                    found_child = True
                    break
                elif v != parent[u]:
                    # Back edge: update low[u]
                    low[u] = min(low[u], disc[v])

            if not found_child:
                # Done with u, backtrack
                stack.pop()
                if stack:
                    # Update parent's low value and check articulation
                    p = parent[u]
                    low[p] = min(low[p], low[u])

                    # Non-root node p is articulation point if low[u] >= disc[p]
                    if parent[p] != -1 and low[u] >= disc[p]:
                        ap[p] = True

        # Root is articulation point if it has > 1 DFS tree children
        if children_count[start_node] > 1:
            ap[start_node] = True

    return {graph.node_ids[i] for i in range(n) if ap[i]}


def validate_network(graph: SparseGraph) -> dict[str, Any]:
    """Validate network topology and return diagnostics.

    Args:
        graph: SparseGraph instance

    Returns:
        Dictionary with validation results
    """
    n_components, components = connected_components(graph)

    # Find isolated nodes (islands)
    islands = [c for c in components if len(c) == 1]

    # Find small disconnected fragments
    small_fragments = [c for c in components if 1 < len(c) < 5]

    # Degree distribution
    degrees_dict = graph.degrees()
    degree_dist: dict[int, int] = {}
    for d in degrees_dict.values():
        degree_dist[d] = degree_dist.get(d, 0) + 1

    validation = {
        "valid": n_components == 1,
        "n_components": n_components,
        "n_islands": len(islands),
        "n_small_fragments": len(small_fragments),
        "degree_distribution": degree_dist,
        "has_dangling_ends": any(d == 1 for d in degrees_dict.values()),
    }

    if not validation["valid"]:
        logger.warning(f"Network has {n_components} disconnected components")
        if islands:
            logger.warning(f"  {len(islands)} isolated nodes")
        if small_fragments:
            logger.warning(f"  {len(small_fragments)} small fragments (2-4 nodes)")

    return validation
