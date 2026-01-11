"""Topological feature extraction for graph-based matching."""

from typing import Any, Optional

import networkx as nx


def compute_topological_features(
    G: nx.Graph,
    edge_key: tuple[int, int],
) -> dict[str, Any]:
    """Compute topological features for an edge.

    Args:
        G: NetworkX graph
        edge_key: Tuple of (from_node, to_node)

    Returns:
        Dictionary of topological features
    """
    from_node, to_node = edge_key

    # Basic degree features
    from_degree = G.degree(from_node) if from_node in G else 0
    to_degree = G.degree(to_node) if to_node in G else 0

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
    G: nx.Graph,
    edge_key: tuple[int, int],
    radius: int = 2,
) -> tuple[int, ...]:
    """Compute a degree signature for an edge's local neighborhood.

    Creates a sorted tuple of degrees for nodes within `radius` hops.
    Two edges with similar signatures have similar local topology.

    Args:
        G: NetworkX graph
        edge_key: Tuple of (from_node, to_node)
        radius: Number of hops to include

    Returns:
        Sorted tuple of degrees in the neighborhood
    """
    from_node, to_node = edge_key

    # Collect nodes within radius
    neighborhood = set()

    for start_node in [from_node, to_node]:
        if start_node not in G:
            continue

        # BFS to find nodes within radius
        visited = {start_node}
        frontier = [start_node]

        for _ in range(radius):
            next_frontier = []
            for node in frontier:
                for neighbor in G.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier

        neighborhood.update(visited)

    # Get sorted degrees
    degrees = sorted(G.degree(n) for n in neighborhood if n in G)
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

    # Convert to multisets (count occurrences)
    from collections import Counter

    counter_a = Counter(sig_a)
    counter_b = Counter(sig_b)

    # Compute intersection and union
    intersection = sum((counter_a & counter_b).values())
    union = sum((counter_a | counter_b).values())

    return intersection / union if union > 0 else 0.0


def compute_local_pattern(
    G: nx.Graph,
    node: int,
) -> str:
    """Compute a local pattern string for a node.

    The pattern encodes the node's degree and its neighbors' degrees,
    useful for quick pattern matching.

    Args:
        G: NetworkX graph
        node: Node ID

    Returns:
        Pattern string like "4-[2,3,3,4]" (degree-[neighbor_degrees])
    """
    if node not in G:
        return "0-[]"

    degree = G.degree(node)
    neighbor_degrees = sorted(G.degree(n) for n in G.neighbors(node))

    return f"{degree}-{neighbor_degrees}"


def edge_is_bridge(G: nx.Graph, edge_key: tuple[int, int]) -> bool:
    """Check if an edge is a bridge (its removal disconnects the graph).

    Bridge edges are topologically important - removing them would
    split the network.

    Args:
        G: NetworkX graph
        edge_key: Tuple of (from_node, to_node)

    Returns:
        True if the edge is a bridge
    """
    from_node, to_node = edge_key

    if from_node not in G or to_node not in G:
        return False

    if not G.has_edge(from_node, to_node):
        return False

    # Temporarily remove the edge
    edge_data = G.edges[from_node, to_node]
    G.remove_edge(from_node, to_node)

    # Check if nodes are still connected
    is_bridge = not nx.has_path(G, from_node, to_node)

    # Restore the edge
    G.add_edge(from_node, to_node, **edge_data)

    return is_bridge


def compute_edge_centrality_features(
    G: nx.Graph,
    edge_key: tuple[int, int],
    node_pagerank: Optional[dict[int, float]] = None,
    node_betweenness: Optional[dict[int, float]] = None,
) -> dict[str, float]:
    """Compute centrality-based features for an edge.

    Args:
        G: NetworkX graph
        edge_key: Tuple of (from_node, to_node)
        node_pagerank: Pre-computed PageRank (optional)
        node_betweenness: Pre-computed betweenness centrality (optional)

    Returns:
        Dictionary of centrality features
    """
    from_node, to_node = edge_key

    features = {}

    # PageRank features
    if node_pagerank is not None:
        pr_from = node_pagerank.get(from_node, 0)
        pr_to = node_pagerank.get(to_node, 0)
        features["pagerank_sum"] = pr_from + pr_to
        features["pagerank_max"] = max(pr_from, pr_to)
        features["pagerank_min"] = min(pr_from, pr_to)

    # Betweenness features
    if node_betweenness is not None:
        bc_from = node_betweenness.get(from_node, 0)
        bc_to = node_betweenness.get(to_node, 0)
        features["betweenness_sum"] = bc_from + bc_to
        features["betweenness_max"] = max(bc_from, bc_to)

    return features


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
        comparison["dead_end_match"] = float(
            features_a["is_dead_end"] == features_b["is_dead_end"]
        )

    # Intersection edge comparison
    if "is_intersection_edge" in features_a and "is_intersection_edge" in features_b:
        comparison["intersection_match"] = float(
            features_a["is_intersection_edge"] == features_b["is_intersection_edge"]
        )

    return comparison
