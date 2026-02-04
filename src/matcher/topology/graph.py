"""Sparse graph construction from planarized edges.

Uses scipy CSR matrices instead of NetworkX for Spark compatibility.
"""

from typing import Any

from loguru import logger

from .planarize import PlanarizedNetwork
from .sparse_graph import (
    SparseGraph,
    build_graph_from_node_pairs,
    connected_components,
)
from .sparse_graph import (
    validate_network as sparse_validate_network,
)


def build_graph(network: PlanarizedNetwork) -> SparseGraph:
    """Build sparse graph from planarized network.

    Args:
        network: PlanarizedNetwork with nodes and edges

    Returns:
        SparseGraph instance
    """
    logger.info("Building sparse graph...")

    # Extract node IDs
    node_ids = list(network.nodes["node_id"].values)

    # Extract edge endpoints
    from_nodes = list(network.edges["from_node"].values)
    to_nodes = list(network.edges["to_node"].values)

    # Build edge data dict for optional attributes
    edge_data: dict[tuple[Any, Any], dict[str, Any]] = {}
    for _, row in network.edges.iterrows():
        u, v = row["from_node"], row["to_node"]
        attrs = {
            "edge_id": row["edge_id"],
            "geometry": row.geometry,
            "length": row.geometry.length,
        }
        # Add optional attributes if present
        for col in ["original_id", "name", "road_class", "is_bridge", "is_tunnel", "layer"]:
            if col in row.index:
                attrs[col] = row[col]
        edge_data[(u, v)] = attrs

    # Build graph
    graph = build_graph_from_node_pairs(from_nodes, to_nodes, node_ids)
    graph.edge_data = edge_data

    logger.info(f"Graph built: {graph.n_nodes} nodes, {graph.n_edges} edges")
    return graph


def compute_topology_features(graph: SparseGraph) -> dict[str, Any]:
    """Compute graph-level topology features.

    Args:
        graph: SparseGraph instance

    Returns:
        Dictionary of topology metrics
    """
    n_nodes = graph.n_nodes
    n_edges = graph.n_edges
    n_components, _ = connected_components(graph)

    # Degree statistics
    degrees = graph.degrees()
    degree_values = list(degrees.values())
    avg_degree = sum(degree_values) / n_nodes if n_nodes > 0 else 0

    # Count dangling ends (degree 1)
    dangling_count = sum(1 for d in degree_values if d == 1)

    # Count intersections (degree > 2)
    intersection_count = sum(1 for d in degree_values if d > 2)

    # Total network length (from edge data)
    total_length = 0.0
    if graph.edge_data:
        total_length = sum(data.get("length", 0) for data in graph.edge_data.values())

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


def find_connected_components(graph: SparseGraph) -> list[set[int]]:
    """Find connected components in the graph.

    Args:
        graph: SparseGraph instance

    Returns:
        List of sets of node IDs, one per component
    """
    _, components = connected_components(graph)
    return components


def validate_network(graph: SparseGraph) -> dict[str, Any]:
    """Validate network topology and return diagnostics.

    Args:
        graph: SparseGraph instance

    Returns:
        Dictionary with validation results
    """
    return sparse_validate_network(graph)
