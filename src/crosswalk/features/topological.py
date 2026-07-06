"""Topological feature extraction for graph-based matching.

Uses scipy CSR-based SparseGraph instead of NetworkX for Spark compatibility.
"""

from collections import Counter
from typing import Any

from crosswalk.topology.sparse_graph import (
    SparseGraph,
    bfs_neighbors,
)
from crosswalk.topology.sparse_graph import (
    edge_is_bridge as sparse_edge_is_bridge,
)


def compute_topological_features(
    graph: SparseGraph,
    edge_key: tuple[int, int],
) -> dict[str, Any]:
    """Compute topological features for an edge.

    Args:
        graph: SparseGraph instance
        edge_key: Tuple of (from_node, to_node)

    Returns:
        Dictionary of topological features
    """
    from_node, to_node = edge_key

    # Basic degree features
    from_degree = graph.degree(from_node)
    to_degree = graph.degree(to_node)

    # Edge connectivity features
    features = {
        "from_degree": from_degree,
        "to_degree": to_degree,
        "avg_degree": (from_degree + to_degree) / 2,
        "min_degree": min(from_degree, to_degree),
        "max_degree": max(from_degree, to_degree),
        "is_dead_end": min(from_degree, to_degree) == 1,
        "is_intersection_edge": max(from_degree, to_degree) > 2,
    }

    return features


def compute_degree_signature(
    graph: SparseGraph,
    edge_key: tuple[int, int],
    radius: int = 2,
) -> tuple[int, ...]:
    """Compute a degree signature for an edge's local neighborhood.

    Creates a sorted tuple of degrees for nodes within `radius` hops.
    Two edges with similar signatures have similar local topology.

    Args:
        graph: SparseGraph instance
        edge_key: Tuple of (from_node, to_node)
        radius: Number of hops to include

    Returns:
        Sorted tuple of degrees in the neighborhood
    """
    from_node, to_node = edge_key

    # Collect nodes within radius of both endpoints
    neighborhood = bfs_neighbors(graph, from_node, radius)
    neighborhood.update(bfs_neighbors(graph, to_node, radius))

    # Get sorted degrees
    degrees = sorted(graph.degree(n) for n in neighborhood)
    return tuple(degrees)


def degree_signature_similarity(
    sig_a: tuple[int, ...],
    sig_b: tuple[int, ...],
) -> float:
    """Compute similarity between two degree signatures.

    Uses a simple overlap-based metric.

    Args:
        sig_a: First degree signature
        sig_b: Second degree signature

    Returns:
        Similarity score (0-1)
    """
    if not sig_a or not sig_b:
        return 0.0

    counter_a = Counter(sig_a)
    counter_b = Counter(sig_b)

    # Compute intersection and union
    intersection = sum((counter_a & counter_b).values())
    union = sum((counter_a | counter_b).values())

    return intersection / union if union > 0 else 0.0


def compute_local_pattern(
    graph: SparseGraph,
    node: int,
) -> str:
    """Compute a local pattern string for a node.

    The pattern encodes the node's degree and its neighbors' degrees,
    useful for quick pattern matching.

    Args:
        graph: SparseGraph instance
        node: Node ID

    Returns:
        Pattern string like "4-[2,3,3,4]" (degree-[neighbor_degrees])
    """
    if node not in graph.node_to_idx:
        return "0-[]"

    degree = graph.degree(node)
    neighbors = graph.neighbors(node)
    neighbor_degrees = sorted(graph.degree(n) for n in neighbors)

    return f"{degree}-{neighbor_degrees}"


def edge_is_bridge(graph: SparseGraph, edge_key: tuple[int, int]) -> bool:
    """Check if an edge is a bridge (its removal disconnects the graph).

    Bridge edges are topologically important - removing them would
    split the network.

    Args:
        graph: SparseGraph instance
        edge_key: Tuple of (from_node, to_node)

    Returns:
        True if the edge is a bridge
    """
    return sparse_edge_is_bridge(graph, edge_key)


def compare_topology_features(
    features_a: dict[str, Any],
    features_b: dict[str, Any],
) -> dict[str, float]:
    """Compare two sets of topological features.

    Args:
        features_a: Features from reference edge
        features_b: Features from target edge

    Returns:
        Dictionary of comparison metrics
    """
    comparison = {}

    # Degree comparison
    if "from_degree" in features_a and "from_degree" in features_b:
        deg_a = (features_a["from_degree"], features_a["to_degree"])
        deg_b = (features_b["from_degree"], features_b["to_degree"])

        # Best matching of degrees (endpoints might be swapped)
        diff1 = abs(deg_a[0] - deg_b[0]) + abs(deg_a[1] - deg_b[1])
        diff2 = abs(deg_a[0] - deg_b[1]) + abs(deg_a[1] - deg_b[0])
        min_diff = min(diff1, diff2)

        # Normalize to similarity (0-1)
        max_possible = max(sum(deg_a), sum(deg_b))
        comparison["degree_similarity"] = 1.0 - (min_diff / max(max_possible, 1))

    # Dead end comparison
    if "is_dead_end" in features_a and "is_dead_end" in features_b:
        comparison["dead_end_match"] = float(features_a["is_dead_end"] == features_b["is_dead_end"])

    # Intersection edge comparison
    if "is_intersection_edge" in features_a and "is_intersection_edge" in features_b:
        comparison["intersection_match"] = float(
            features_a["is_intersection_edge"] == features_b["is_intersection_edge"]
        )

    return comparison
