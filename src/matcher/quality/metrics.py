"""Quality metric computations for road network datasets.

Computes various quality metrics from a GeoDataFrame of road edges.
"""

import geopandas as gpd
import numpy as np
from loguru import logger
from shapely.geometry import LineString

from ..config import CLASS_COLUMN, NAMES_COLUMN
from ..post_integration.constants import SNAP_TOLERANCE_M
from ..post_integration.island_detector import detect_islands
from .fingerprint import QualityFingerprint


def compute_quality_metrics(
    edges_gdf: gpd.GeoDataFrame,
    dataset_name: str,
    name_column: str | None = None,
    class_column: str | None = None,
    snap_tolerance_m: float = SNAP_TOLERANCE_M,
) -> QualityFingerprint:
    """Compute comprehensive quality metrics for a road network dataset.

    Args:
        edges_gdf: GeoDataFrame with road edges (LineString geometries)
        dataset_name: Name identifier for the dataset
        name_column: Column containing road names (auto-detected if None)
        class_column: Column containing road class (auto-detected if None)
        snap_tolerance_m: Tolerance for topology analysis

    Returns:
        QualityFingerprint with computed metrics
    """
    logger.info(f"Computing quality metrics for {dataset_name} ({len(edges_gdf)} edges)")

    if len(edges_gdf) == 0:
        return QualityFingerprint(dataset_name=dataset_name)

    # Ensure CRS is set
    if edges_gdf.crs is None:
        edges_gdf = edges_gdf.set_crs("EPSG:4326")

    # Work in metric CRS for length calculations
    metric_crs = edges_gdf.estimate_utm_crs()
    edges_metric = edges_gdf.to_crs(metric_crs)

    # Basic statistics
    total_segments = len(edges_gdf)
    total_length_m = edges_metric.geometry.length.sum()

    # Geometry metrics
    vertex_density_mean, vertex_density_std = _compute_vertex_density(edges_metric)
    invalid_geometry_count = _count_invalid_geometries(edges_gdf)

    # Filter to LineString geometries for topology analysis
    line_mask = edges_gdf.geometry.apply(lambda g: isinstance(g, LineString) if g else False)
    if not line_mask.all():
        non_line_count = (~line_mask).sum()
        logger.warning(
            f"Filtering {non_line_count} non-LineString geometries for topology analysis"
        )
    edges_lines = edges_gdf[line_mask]
    edges_metric_lines = edges_metric.loc[edges_lines.index]

    # Topology metrics
    island_result = detect_islands(edges_lines, snap_tolerance_m=snap_tolerance_m)
    dead_end_count, dead_end_ratio = _compute_dead_ends(edges_metric_lines, snap_tolerance_m)

    # Attribute metrics (use standardized column names from fetch step)
    name_col = name_column or NAMES_COLUMN
    name_coverage_ratio = _compute_name_coverage(edges_gdf, name_col)

    class_col = class_column or CLASS_COLUMN
    class_distribution = _compute_class_distribution(edges_gdf, class_col)

    fingerprint = QualityFingerprint(
        dataset_name=dataset_name,
        total_segments=total_segments,
        total_length_m=total_length_m,
        vertex_density_mean=vertex_density_mean,
        vertex_density_std=vertex_density_std,
        invalid_geometry_count=invalid_geometry_count,
        island_count=len(island_result.islands),
        dead_end_count=dead_end_count,
        dead_end_ratio=dead_end_ratio,
        connected_components=island_result.total_components,
        largest_component_ratio=island_result.main_component_ratio,
        name_coverage_ratio=name_coverage_ratio,
        class_distribution=class_distribution,
    )

    logger.info(
        f"Quality fingerprint computed: {total_segments} segments, "
        f"{total_length_m / 1000:.1f}km, {name_coverage_ratio:.1%} named"
    )

    return fingerprint


def _compute_vertex_density(edges_metric: gpd.GeoDataFrame) -> tuple[float, float]:
    """Compute vertex density (vertices per meter) statistics.

    Args:
        edges_metric: GeoDataFrame in metric CRS

    Returns:
        Tuple of (mean_density, std_density)
    """
    densities = []

    for geom in edges_metric.geometry:
        if geom is None or geom.is_empty:
            continue
        if not isinstance(geom, LineString):
            continue

        length = geom.length
        if length > 0:
            n_vertices = len(geom.coords)
            density = n_vertices / length
            densities.append(density)

    if not densities:
        return 0.0, 0.0

    return float(np.mean(densities)), float(np.std(densities))


def _count_invalid_geometries(edges_gdf: gpd.GeoDataFrame) -> int:
    """Count geometries that are invalid or problematic.

    Args:
        edges_gdf: GeoDataFrame with geometries

    Returns:
        Count of invalid geometries
    """
    count = 0
    for geom in edges_gdf.geometry:
        if geom is None or geom.is_empty or not geom.is_valid:
            count += 1
    return count


def _compute_dead_ends(
    edges_metric: gpd.GeoDataFrame,
    snap_tolerance_m: float,
) -> tuple[int, float]:
    """Compute dead end count and ratio.

    A dead end is an endpoint that connects to only one edge.

    Args:
        edges_metric: GeoDataFrame in metric CRS
        snap_tolerance_m: Tolerance for endpoint matching

    Returns:
        Tuple of (dead_end_count, dead_end_ratio)
    """
    # Collect all endpoints
    endpoints: list[tuple[float, float]] = []

    for geom in edges_metric.geometry:
        if geom is None or geom.is_empty:
            continue
        if not isinstance(geom, LineString):
            continue

        coords = list(geom.coords)
        if len(coords) >= 2:
            endpoints.append(coords[0][:2])
            endpoints.append(coords[-1][:2])

    if len(endpoints) < 2:
        return 0, 0.0

    # Count endpoint occurrences using spatial proximity
    points = np.array(endpoints)
    endpoint_degrees: dict[int, int] = {}

    for i in range(len(points)):
        dists = np.sqrt(np.sum((points - points[i]) ** 2, axis=1))
        nearby_count = np.sum(dists <= snap_tolerance_m)
        endpoint_degrees[i] = nearby_count

    # Dead ends have degree 1 (only connect to themselves)
    dead_end_count = sum(1 for d in endpoint_degrees.values() if d == 1)
    dead_end_ratio = dead_end_count / len(endpoints) if endpoints else 0.0

    return dead_end_count, dead_end_ratio


def _compute_name_coverage(
    gdf: gpd.GeoDataFrame,
    name_column: str | None,
) -> float:
    """Compute the ratio of edges that have names.

    Args:
        gdf: GeoDataFrame with edges
        name_column: Column containing names

    Returns:
        Ratio of named edges (0.0 to 1.0)
    """
    if name_column is None or name_column not in gdf.columns:
        return 0.0

    # Count non-null, non-empty names
    names = gdf[name_column]
    named_count = 0

    for name in names:
        if name is None:
            continue
        if isinstance(name, str) and name.strip():
            named_count += 1
        elif isinstance(name, dict):
            # Overture format: {"primary": "Main St", ...}
            if name.get("primary"):
                named_count += 1

    return named_count / len(gdf) if len(gdf) > 0 else 0.0


def _compute_class_distribution(
    gdf: gpd.GeoDataFrame,
    class_column: str | None,
) -> dict[str, int]:
    """Compute distribution of road classes.

    Args:
        gdf: GeoDataFrame with edges
        class_column: Column containing road class

    Returns:
        Dictionary mapping class names to counts
    """
    if class_column is None or class_column not in gdf.columns:
        return {}

    distribution: dict[str, int] = {}

    for cls in gdf[class_column]:
        if cls is None:
            cls_str = "unknown"
        else:
            cls_str = str(cls)

        distribution[cls_str] = distribution.get(cls_str, 0) + 1

    # Sort by count descending
    return dict(sorted(distribution.items(), key=lambda x: -x[1]))
