"""Output generation for integration pipeline.

Writes integration results to parquet files with proper schemas.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
from loguru import logger

from .provenance import IntegrationResult, IntegrationStatistics


def write_integration_outputs(
    result: IntegrationResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write integration outputs to parquet files.

    Creates:
    - nodes.parquet: All nodes with component annotations
    - edges.parquet: All edges with provenance and component annotations
    - orphans.parquet: Orphan edges for QA review
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

    output_paths = {}

    # Write nodes
    nodes_path = output_dir / "nodes.parquet"
    _write_nodes(result.nodes, nodes_path)
    output_paths["nodes"] = nodes_path

    # Write edges
    edges_path = output_dir / "edges.parquet"
    _write_edges(result.edges, edges_path)
    output_paths["edges"] = edges_path

    # Write orphans
    orphans_path = output_dir / "orphans.parquet"
    _write_orphans(result.orphan_edges, orphans_path)
    output_paths["orphans"] = orphans_path

    # Write dropped overlaps
    dropped_path = output_dir / "dropped_overlaps.parquet"
    _write_dropped_overlaps(result.dropped_overlaps, dropped_path)
    output_paths["dropped_overlaps"] = dropped_path

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


def _write_orphans(orphans: gpd.GeoDataFrame, path: Path) -> None:
    """Write orphan edges to parquet for QA."""
    if orphans is None or len(orphans) == 0:
        logger.info("No orphan edges to write")
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

    orphans.to_parquet(path)
    logger.info(f"Wrote {len(orphans)} orphan edges to {path}")


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

    # Load orphans
    orphans_path = output_dir / "orphans.parquet"
    orphans = gpd.read_parquet(orphans_path) if orphans_path.exists() else gpd.GeoDataFrame()

    # Load dropped overlaps
    dropped_path = output_dir / "dropped_overlaps.parquet"
    dropped = gpd.read_parquet(dropped_path) if dropped_path.exists() else gpd.GeoDataFrame()

    # Load statistics
    stats_path = output_dir / "statistics.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats_data = json.load(f)
        created_at = datetime.fromisoformat(
            stats_data.pop("created_at", datetime.now(UTC).isoformat())
        )
        stats_data.pop("pipeline_version", None)
        statistics = IntegrationStatistics(**stats_data)
    else:
        statistics = IntegrationStatistics()
        created_at = datetime.now(UTC)

    return IntegrationResult(
        nodes=nodes,
        edges=edges,
        orphan_edges=orphans,
        dropped_overlaps=dropped,
        statistics=statistics,
        created_at=created_at,
    )
