"""NetworkX graph construction from planarized edges."""

from typing import Any

import networkx as nx
from loguru import logger

from .planarize import PlanarizedNetwork


def build_graph(network: PlanarizedNetwork) -> nx.Graph:
    """Build NetworkX graph from planarized network.

    Node attributes: geometry, x, y
    Edge attributes: edge_id, geometry, length, original_id, name, road_class, etc.

    Args:
        network: PlanarizedNetwork with nodes and edges

    Returns:
        NetworkX Graph
    """
    logger.info("Building NetworkX graph...")

    G = nx.Graph()

    # Add nodes
    for _, row in network.nodes.iterrows():
        G.add_node(
            row["node_id"],
            geometry=row.geometry,
            x=row.geometry.x,
            y=row.geometry.y,
        )

    # Add edges
    for _, row in network.edges.iterrows():
        attrs = {
            "edge_id": row["edge_id"],
            "geometry": row.geometry,
            "length": row.geometry.length,
        }

        # Add optional attributes if present
        for col in ["original_id", "name", "road_class", "is_bridge", "is_tunnel", "layer"]:
            if col in row.index:
                attrs[col] = row[col]

        G.add_edge(row["from_node"], row["to_node"], **attrs)

    logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_topology_features(G: nx.Graph) -> dict[str, Any]:
    """Compute graph-level topology features.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary of topology metrics
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_components = nx.number_connected_components(G)

    # Degree statistics
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    avg_degree = sum(degree_values) / n_nodes if n_nodes > 0 else 0

    # Count dangling ends (degree 1)
    dangling_count = sum(1 for d in degree_values if d == 1)

    # Count intersections (degree > 2)
    intersection_count = sum(1 for d in degree_values if d > 2)

    # Total network length
    total_length = sum(data.get("length", 0) for _, _, data in G.edges(data=True))

    features = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_components": n_components,
        "avg_degree": avg_degree,
        "dangling_count": dangling_count,
        "intersection_count": intersection_count,
        "total_length": total_length,
    }

    logger.info(f"Topology features: {features}")
    return features


def compute_node_features(G: nx.Graph) -> dict[int, dict[str, Any]]:
    """Compute per-node topology features.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary mapping node_id to feature dict
    """
    features = {}

    # Degree
    degrees = dict(G.degree())

    # PageRank (importance)
    try:
        pagerank = nx.pagerank(G)
    except nx.NetworkXError:
        pagerank = {n: 0 for n in G.nodes()}

    # Betweenness centrality (for smaller graphs)
    if G.number_of_nodes() < 10000:
        try:
            betweenness = nx.betweenness_centrality(G)
        except nx.NetworkXError:
            betweenness = {n: 0 for n in G.nodes()}
    else:
        betweenness = {n: 0 for n in G.nodes()}

    for node in G.nodes():
        features[node] = {
            "degree": degrees[node],
            "pagerank": pagerank.get(node, 0),
            "betweenness": betweenness.get(node, 0),
        }

    return features


def compute_edge_features(G: nx.Graph) -> dict[tuple[int, int], dict[str, Any]]:
    """Compute per-edge topology features.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary mapping (from_node, to_node) to feature dict
    """
    features = {}

    # Edge betweenness centrality (for smaller graphs)
    if G.number_of_edges() < 10000:
        try:
            edge_betweenness = nx.edge_betweenness_centrality(G)
        except nx.NetworkXError:
            edge_betweenness = {}
    else:
        edge_betweenness = {}

    for u, v, data in G.edges(data=True):
        features[(u, v)] = {
            "length": data.get("length", 0),
            "edge_betweenness": edge_betweenness.get((u, v), 0),
            "from_degree": G.degree(u),
            "to_degree": G.degree(v),
        }

    return features


def find_connected_components(G: nx.Graph) -> list[set[int]]:
    """Find connected components in the graph.

    Args:
        G: NetworkX graph

    Returns:
        List of sets of node IDs, one per component
    """
    return [set(c) for c in nx.connected_components(G)]


def validate_network(G: nx.Graph) -> dict[str, Any]:
    """Validate network topology and return diagnostics.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary with validation results
    """
    components = list(nx.connected_components(G))
    n_components = len(components)

    # Find isolated nodes (islands)
    islands = [c for c in components if len(c) == 1]

    # Find small disconnected fragments
    small_fragments = [c for c in components if 1 < len(c) < 5]

    # Degree distribution
    degrees = dict(G.degree())
    degree_dist = {}
    for d in degrees.values():
        degree_dist[d] = degree_dist.get(d, 0) + 1

    validation = {
        "valid": n_components == 1,
        "n_components": n_components,
        "n_islands": len(islands),
        "n_small_fragments": len(small_fragments),
        "degree_distribution": degree_dist,
        "has_dangling_ends": any(d == 1 for d in degrees.values()),
    }

    if not validation["valid"]:
        logger.warning(f"Network has {n_components} disconnected components")
        if islands:
            logger.warning(f"  {len(islands)} isolated nodes")
        if small_fragments:
            logger.warning(f"  {len(small_fragments)} small fragments (2-4 nodes)")

    return validation
