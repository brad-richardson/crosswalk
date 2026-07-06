"""Output generation for integration pipeline.

Writes integration results to parquet files with proper schemas.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
from loguru import logger

from .provenance import IntegrationResult, IntegrationStatistics

# Legacy files that should be removed on new runs
_LEGACY_FILES = ["orphans.parquet"]


def _cleanup_stale_files(output_dir: Path) -> None:
    """Remove legacy files from previous pipeline versions."""
    for name in _LEGACY_FILES:
        path = output_dir / name
        if path.exists():
            path.unlink()
            logger.info(f"Removed legacy file: {path}")


def write_integration_outputs(
    result: IntegrationResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write integration outputs to parquet files.

    Creates:
    - nodes.parquet: All nodes with component annotations
    - edges.parquet: All edges with provenance and component annotations
    - disconnected.parquet: Truly disconnected segments for QA review
    - filtered.parquet: Connected segments with insufficient net-new coverage
    - net_new.parquet: Net-new geometry portions (for visualization)
    - bridges.parquet: Bridge segments promoted via connectivity gating (for QA)
    - dropped_overlaps.parquet: Segments dropped due to priority conflicts
    - statistics.json: Integration statistics

    Args:
        result: IntegrationResult from pipeline
        output_dir: Directory for output files

    Returns:
        Dictionary mapping output type to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up legacy files from previous runs
    _cleanup_stale_files(output_dir)

    output_paths = {}

    # Write nodes
    nodes_path = output_dir / "nodes.parquet"
    _write_nodes(result.nodes, nodes_path)
    output_paths["nodes"] = nodes_path

    # Write edges
    edges_path = output_dir / "edges.parquet"
    _write_edges(result.edges, edges_path)
    output_paths["edges"] = edges_path

    # Write disconnected edges
    disconnected_path = output_dir / "disconnected.parquet"
    _write_orphan_layer(result.disconnected_edges, disconnected_path, "disconnected")
    output_paths["disconnected"] = disconnected_path

    # Write filtered edges
    filtered_path = output_dir / "filtered.parquet"
    _write_orphan_layer(result.filtered_edges, filtered_path, "filtered")
    output_paths["filtered"] = filtered_path

    # Write dropped overlaps
    dropped_path = output_dir / "dropped_overlaps.parquet"
    _write_dropped_overlaps(result.dropped_overlaps, dropped_path)
    output_paths["dropped_overlaps"] = dropped_path

    # Write net-new edges (geometry portions that add new coverage)
    net_new_path = output_dir / "net_new.parquet"
    if result.net_new_edges is not None and len(result.net_new_edges) > 0:
        _write_net_new(result.net_new_edges, net_new_path)
        output_paths["net_new"] = net_new_path
    elif net_new_path.exists():
        net_new_path.unlink()
        logger.info(f"Removed stale {net_new_path} (no net-new edges this run)")

    # Write bridge edges (promoted connectors between disconnected components)
    bridges_path = output_dir / "bridges.parquet"
    if result.bridge_edges is not None and len(result.bridge_edges) > 0:
        result.bridge_edges.to_parquet(bridges_path)
        output_paths["bridges"] = bridges_path
        logger.info(f"Wrote {len(result.bridge_edges)} bridge edges to {bridges_path}")
    elif bridges_path.exists():
        bridges_path.unlink()
        logger.info(f"Removed stale {bridges_path} (no bridge edges this run)")

    # Write statistics
    stats_path = output_dir / "statistics.json"
    _write_statistics(result.statistics, result.created_at, stats_path)
    output_paths["statistics"] = stats_path

    logger.info(f"Integration outputs written to {output_dir}")
    for name, path in output_paths.items():
        logger.info(f"  {name}: {path}")

    return output_paths


def _write_nodes(nodes: gpd.GeoDataFrame, path: Path) -> None:
    """Write nodes to parquet."""
    if nodes is None or len(nodes) == 0:
        logger.warning("No nodes to write")
        # Write empty file with schema
        empty = gpd.GeoDataFrame(
            columns=["node_id", "geometry", "component_id", "component_status"]
        )
        empty.to_parquet(path)
        return

    nodes.to_parquet(path)
    logger.info(f"Wrote {len(nodes)} nodes to {path}")


def _write_edges(edges: gpd.GeoDataFrame, path: Path) -> None:
    """Write edges to parquet."""
    if edges is None or len(edges) == 0:
        logger.warning("No edges to write")
        # Write empty file with schema
        empty = gpd.GeoDataFrame(
            columns=[
                "edge_id",
                "from_node",
                "to_node",
                "geometry",
                "_source",
                "_original_id",
                "_source_dataset",
                "_priority",
                "_match_ref_id",
                "_match_confidence",
                "component_id",
                "component_status",
                "component_size",
            ]
        )
        empty.to_parquet(path)
        return

    edges.to_parquet(path)
    logger.info(f"Wrote {len(edges)} edges to {path}")


def _write_orphan_layer(gdf: gpd.GeoDataFrame, path: Path, label: str) -> None:
    """Write a disconnected or filtered edges layer to parquet for QA."""
    if gdf is None or len(gdf) == 0:
        logger.info(f"No {label} edges to write")
        # Write empty file with schema
        empty = gpd.GeoDataFrame(
            columns=[
                "edge_id",
                "geometry",
                "_source",
                "_original_id",
                "_source_dataset",
                "component_id",
                "component_size",
                "nearest_main_distance",
                "qa_priority",
            ]
        )
        empty.to_parquet(path)
        return

    gdf.to_parquet(path)
    logger.info(f"Wrote {len(gdf)} {label} edges to {path}")


def _write_dropped_overlaps(dropped: gpd.GeoDataFrame, path: Path) -> None:
    """Write dropped overlaps to parquet."""
    if dropped is None or len(dropped) == 0:
        logger.info("No dropped overlaps to write")
        # Write empty file with schema
        empty = gpd.GeoDataFrame(
            columns=[
                "geometry",
                "original_id",
                "source_dataset",
                "source_type",
                "dropped_reason",
                "overlapping_edge_id",
                "overlap_iou",
                "priority",
            ]
        )
        empty.to_parquet(path)
        return

    dropped.to_parquet(path)
    logger.info(f"Wrote {len(dropped)} dropped overlaps to {path}")


def _write_net_new(net_new: gpd.GeoDataFrame, path: Path) -> None:
    """Write net-new geometry portions to parquet."""
    if net_new is None or len(net_new) == 0:
        logger.info("No net-new edges to write")
        return

    net_new.to_parquet(path)
    logger.info(f"Wrote {len(net_new)} net-new edges to {path}")


def _write_statistics(
    stats: IntegrationStatistics,
    created_at: datetime,
    path: Path,
) -> None:
    """Write statistics to JSON."""
    data = stats.to_dict()
    data["created_at"] = created_at.isoformat()
    data["pipeline_version"] = "0.1.0"

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Wrote statistics to {path}")


def load_integration_result(output_dir: Path) -> IntegrationResult:
    """Load integration result from output directory.

    Args:
        output_dir: Directory containing integration outputs

    Returns:
        IntegrationResult
    """
    output_dir = Path(output_dir)

    # Load nodes
    nodes_path = output_dir / "nodes.parquet"
    nodes = gpd.read_parquet(nodes_path) if nodes_path.exists() else gpd.GeoDataFrame()

    # Load edges
    edges_path = output_dir / "edges.parquet"
    edges = gpd.read_parquet(edges_path) if edges_path.exists() else gpd.GeoDataFrame()

    # Load disconnected and filtered edges
    disconnected_path = output_dir / "disconnected.parquet"
    disconnected = (
        gpd.read_parquet(disconnected_path) if disconnected_path.exists() else gpd.GeoDataFrame()
    )

    filtered_path = output_dir / "filtered.parquet"
    filtered = gpd.read_parquet(filtered_path) if filtered_path.exists() else gpd.GeoDataFrame()

    # Load dropped overlaps
    dropped_path = output_dir / "dropped_overlaps.parquet"
    dropped = gpd.read_parquet(dropped_path) if dropped_path.exists() else gpd.GeoDataFrame()

    # Load net-new edges
    net_new_path = output_dir / "net_new.parquet"
    net_new = gpd.read_parquet(net_new_path) if net_new_path.exists() else None

    # Load bridge edges (restore _full_geometry from WKB)
    bridges_path = output_dir / "bridges.parquet"
    bridge_edges = None
    if bridges_path.exists():
        bridge_edges = gpd.read_parquet(bridges_path)
        if "_full_geometry_wkb" in bridge_edges.columns:
            import shapely

            bridge_edges["_full_geometry"] = shapely.from_wkb(bridge_edges["_full_geometry_wkb"])
            bridge_edges = bridge_edges.drop(columns=["_full_geometry_wkb"])

    # Load statistics
    stats_path = output_dir / "statistics.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats_data = json.load(f)
        created_at = datetime.fromisoformat(
            stats_data.pop("created_at", datetime.now(UTC).isoformat())
        )
        stats_data.pop("pipeline_version", None)
        # Strip unknown keys from older pipeline versions
        known_fields = {f.name for f in IntegrationStatistics.__dataclass_fields__.values()}
        stats_data = {k: v for k, v in stats_data.items() if k in known_fields}
        statistics = IntegrationStatistics(**stats_data)
    else:
        statistics = IntegrationStatistics()
        created_at = datetime.now(UTC)

    return IntegrationResult(
        nodes=nodes,
        edges=edges,
        disconnected_edges=disconnected,
        filtered_edges=filtered,
        dropped_overlaps=dropped,
        net_new_edges=net_new,
        bridge_edges=bridge_edges,
        statistics=statistics,
        created_at=created_at,
    )
