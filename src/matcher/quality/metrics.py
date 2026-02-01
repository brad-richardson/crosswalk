"""Quality metric computations for road network datasets.

Computes various quality metrics from a GeoDataFrame of road edges.
"""

import geopandas as gpd
import numpy as np
from loguru import logger
from scipy.spatial import KDTree
from shapely.geometry import LineString

from ..config import CLASS_COLUMN, NAMES_COLUMN
from ..post_integration.constants import SNAP_TOLERANCE_M
from ..post_integration.gps_drift_detector import detect_gps_drift
from ..post_integration.island_detector import detect_islands
from .fingerprint import QualityFingerprint

# Thresholds for quality metrics
SHARP_TURN_THRESHOLD_DEG = 150.0  # Interior angles > this indicate sharp turns/reversals
HIGH_SINUOSITY_THRESHOLD = 1.5  # Sinuosity > this is considered "high"
NEAR_DUPLICATE_BUFFER_M = 2.0  # Buffer for near-duplicate detection
NEAR_DUPLICATE_IOU_THRESHOLD = 0.9  # IOU threshold for near-duplicates


def compute_quality_metrics(
    edges_gdf: gpd.GeoDataFrame,
    dataset_name: str,
    name_column: str | None = None,
    class_column: str | None = None,
    snap_tolerance_m: float = SNAP_TOLERANCE_M,
    detect_drift: bool = True,
    detect_duplicates: bool = True,
) -> QualityFingerprint:
    """Compute comprehensive quality metrics for a road network dataset.

    Args:
        edges_gdf: GeoDataFrame with road edges (LineString geometries)
        dataset_name: Name identifier for the dataset
        name_column: Column containing road names (auto-detected if None)
        class_column: Column containing road class (auto-detected if None)
        snap_tolerance_m: Tolerance for topology analysis
        detect_drift: Whether to run GPS drift detection (slower)
        detect_duplicates: Whether to run near-duplicate detection (slower)

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

    # Length distribution
    length_stats = _compute_length_distribution(edges_metric)

    # Filter to LineString geometries for topology/geometry analysis
    line_mask = edges_gdf.geometry.apply(lambda g: isinstance(g, LineString) if g else False)
    if not line_mask.all():
        non_line_count = (~line_mask).sum()
        logger.warning(
            f"Filtering {non_line_count} non-LineString geometries for topology analysis"
        )
    edges_lines = edges_gdf[line_mask]
    edges_metric_lines = edges_metric.loc[edges_lines.index]

    # Jaggedness / geometry quality metrics
    jaggedness_stats = _compute_jaggedness_metrics(edges_metric_lines)

    # GPS drift detection (optional, slower)
    drift_stats = {
        "zigzag_segment_count": 0,
        "spike_segment_count": 0,
        "loop_segment_count": 0,
        "drift_affected_ratio": 0.0,
    }
    if detect_drift and len(edges_lines) > 0:
        try:
            drift_result = detect_gps_drift(edges_lines)
            drift_stats = {
                "zigzag_segment_count": drift_result.zigzag_count,
                "spike_segment_count": drift_result.spike_count,
                "loop_segment_count": drift_result.loop_count,
                "drift_affected_ratio": (
                    drift_result.edges_with_drift / drift_result.total_edges
                    if drift_result.total_edges > 0
                    else 0.0
                ),
            }
        except Exception as e:
            logger.warning(f"GPS drift detection failed: {e}")

    # Near-duplicate detection (optional, slower)
    duplicate_stats = {"near_duplicate_count": 0, "near_duplicate_ratio": 0.0}
    if detect_duplicates and len(edges_metric_lines) > 0:
        try:
            duplicate_stats = _compute_near_duplicates(edges_metric_lines)
        except Exception as e:
            logger.warning(f"Near-duplicate detection failed: {e}")

    # Topology metrics
    island_result = detect_islands(edges_lines, snap_tolerance_m=snap_tolerance_m)
    dead_end_count, dead_end_ratio = _compute_dead_ends(edges_metric_lines, snap_tolerance_m)

    # Attribute metrics (use standardized column names from fetch step)
    name_col = name_column or NAMES_COLUMN
    name_coverage_ratio = _compute_name_coverage(edges_gdf, name_col)

    class_col = class_column or CLASS_COLUMN
    class_distribution = _compute_class_distribution(edges_gdf, class_col)
    class_coverage_ratio = _compute_class_coverage(edges_gdf, class_col)

    fingerprint = QualityFingerprint(
        dataset_name=dataset_name,
        # Basic stats
        total_segments=total_segments,
        total_length_m=total_length_m,
        # Geometry
        vertex_density_mean=vertex_density_mean,
        vertex_density_std=vertex_density_std,
        invalid_geometry_count=invalid_geometry_count,
        # Length distribution
        length_min_m=length_stats["min"],
        length_max_m=length_stats["max"],
        length_median_m=length_stats["median"],
        length_p5_m=length_stats["p5"],
        length_p95_m=length_stats["p95"],
        # Jaggedness
        sharp_angle_count=jaggedness_stats["sharp_angle_count"],
        sharp_angle_ratio=jaggedness_stats["sharp_angle_ratio"],
        mean_segment_sinuosity=jaggedness_stats["mean_sinuosity"],
        high_sinuosity_count=jaggedness_stats["high_sinuosity_count"],
        high_sinuosity_ratio=jaggedness_stats["high_sinuosity_ratio"],
        # GPS drift
        zigzag_segment_count=drift_stats["zigzag_segment_count"],
        spike_segment_count=drift_stats["spike_segment_count"],
        loop_segment_count=drift_stats["loop_segment_count"],
        drift_affected_ratio=drift_stats["drift_affected_ratio"],
        # Duplicates
        near_duplicate_count=duplicate_stats["near_duplicate_count"],
        near_duplicate_ratio=duplicate_stats["near_duplicate_ratio"],
        # Topology
        island_count=len(island_result.islands),
        dead_end_count=dead_end_count,
        dead_end_ratio=dead_end_ratio,
        connected_components=island_result.total_components,
        largest_component_ratio=island_result.main_component_ratio,
        # Attributes
        name_coverage_ratio=name_coverage_ratio,
        class_coverage_ratio=class_coverage_ratio,
        class_distribution=class_distribution,
    )

    logger.info(
        f"Quality fingerprint computed: {total_segments} segments, "
        f"{total_length_m / 1000:.1f}km, {name_coverage_ratio:.1%} named, "
        f"{class_coverage_ratio:.1%} classified"
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
    Uses KDTree for O(n log n) spatial queries instead of O(n²) brute force.

    Args:
        edges_metric: GeoDataFrame in metric CRS
        snap_tolerance_m: Tolerance for endpoint matching

    Returns:
        Tuple of (dead_end_count, dead_end_ratio)
    """
    # Collect all endpoints
    endpoints: list[tuple[float, float]] = []
    n_edges = len(edges_metric)

    for idx, geom in enumerate(edges_metric.geometry):
        if geom is None or geom.is_empty:
            continue
        if not isinstance(geom, LineString):
            continue

        coords = list(geom.coords)
        if len(coords) >= 2:
            endpoints.append(coords[0][:2])
            endpoints.append(coords[-1][:2])

        # Log progress for large datasets
        if n_edges > 10000 and (idx + 1) % 100000 == 0:
            logger.info(f"Collected endpoints for dead-end analysis: {idx + 1}/{n_edges} edges")

    if len(endpoints) < 2:
        return 0, 0.0

    # Use KDTree for O(n log n) neighbor counting
    points = np.array(endpoints)
    logger.debug(f"Building KDTree for {len(points)} endpoints (dead-end detection)")
    tree = KDTree(points)

    # Count neighbors for each point using query_ball_point
    # Dead ends have degree 1 (only themselves within tolerance)
    dead_end_count = 0
    for i in range(len(points)):
        neighbors = tree.query_ball_point(points[i], snap_tolerance_m)
        if len(neighbors) == 1:  # Only itself
            dead_end_count += 1

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


def _compute_length_distribution(edges_metric: gpd.GeoDataFrame) -> dict[str, float]:
    """Compute length distribution statistics.

    Args:
        edges_metric: GeoDataFrame in metric CRS

    Returns:
        Dictionary with min, max, median, p5, p95 lengths
    """
    lengths = []
    for geom in edges_metric.geometry:
        if geom is None or geom.is_empty:
            continue
        lengths.append(geom.length)

    if not lengths:
        return {"min": 0.0, "max": 0.0, "median": 0.0, "p5": 0.0, "p95": 0.0}

    lengths_arr = np.array(lengths)
    return {
        "min": float(np.min(lengths_arr)),
        "max": float(np.max(lengths_arr)),
        "median": float(np.median(lengths_arr)),
        "p5": float(np.percentile(lengths_arr, 5)),
        "p95": float(np.percentile(lengths_arr, 95)),
    }


def _compute_jaggedness_metrics(edges_metric: gpd.GeoDataFrame) -> dict[str, float | int]:
    """Compute geometry jaggedness and sinuosity metrics.

    Jaggedness is measured by:
    - Sharp turns: Vertices where the interior angle > 150 degrees (near reversal)
    - Sinuosity: Ratio of actual length to straight-line distance (1.0 = straight)

    Args:
        edges_metric: GeoDataFrame in metric CRS with LineString geometries

    Returns:
        Dictionary with jaggedness metrics
    """
    sharp_turn_segments = 0
    total_segments = 0
    sinuosities = []
    high_sinuosity_count = 0

    for geom in edges_metric.geometry:
        if geom is None or geom.is_empty:
            continue
        if not isinstance(geom, LineString):
            continue

        total_segments += 1
        coords = np.array(geom.coords)

        # Compute sinuosity (length / straight-line distance)
        # Skip closed loops (start == end) as sinuosity is undefined
        if len(coords) >= 2:
            straight_dist = np.sqrt(
                (coords[-1][0] - coords[0][0]) ** 2 + (coords[-1][1] - coords[0][1]) ** 2
            )
            if straight_dist > 0:
                sinuosity = geom.length / straight_dist
                sinuosities.append(sinuosity)
                if sinuosity > HIGH_SINUOSITY_THRESHOLD:
                    high_sinuosity_count += 1

        # Check for sharp turns (only if >= 3 vertices)
        # A sharp turn is where the interior angle between consecutive segments is large,
        # indicating an abrupt direction change (e.g., zigzag or reversal)
        if len(coords) >= 3:
            has_sharp_turn = False
            for i in range(1, len(coords) - 1):
                v1 = coords[i] - coords[i - 1]
                v2 = coords[i + 1] - coords[i]

                # Compute angle between vectors (0° = same direction, 180° = reversal)
                dot = np.dot(v1, v2)
                mag1 = np.linalg.norm(v1)
                mag2 = np.linalg.norm(v2)

                if mag1 > 0 and mag2 > 0:
                    cos_angle = np.clip(dot / (mag1 * mag2), -1, 1)
                    angle_deg = np.degrees(np.arccos(cos_angle))

                    # Large angle = sharp turn (vectors pointing opposite directions)
                    if angle_deg > SHARP_TURN_THRESHOLD_DEG:
                        has_sharp_turn = True
                        break

            if has_sharp_turn:
                sharp_turn_segments += 1

    return {
        "sharp_angle_count": sharp_turn_segments,
        "sharp_angle_ratio": sharp_turn_segments / total_segments if total_segments > 0 else 0.0,
        "mean_sinuosity": float(np.mean(sinuosities)) if sinuosities else 1.0,
        "high_sinuosity_count": high_sinuosity_count,
        "high_sinuosity_ratio": high_sinuosity_count / total_segments
        if total_segments > 0
        else 0.0,
    }


def _compute_near_duplicates(
    edges_metric: gpd.GeoDataFrame,
    buffer_m: float = NEAR_DUPLICATE_BUFFER_M,
    iou_threshold: float = NEAR_DUPLICATE_IOU_THRESHOLD,
    sample_size: int = 5000,
) -> dict[str, float | int]:
    """Detect near-duplicate geometries using buffered IOU.

    For performance, samples the dataset if it's large. When sampling is used,
    the ratio is computed within the sample and extrapolated to estimate the
    full dataset duplicate count.

    Args:
        edges_metric: GeoDataFrame in metric CRS
        buffer_m: Buffer distance for IOU calculation
        iou_threshold: IOU threshold above which geometries are duplicates
        sample_size: Maximum edges to check (for performance)

    Returns:
        Dictionary with duplicate count and ratio (estimated if sampled)
    """
    total_edges = len(edges_metric)
    if total_edges < 2:
        return {"near_duplicate_count": 0, "near_duplicate_ratio": 0.0}

    # Sample if dataset is large
    is_sampled = total_edges > sample_size
    if is_sampled:
        edges_sample = edges_metric.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        edges_sample = edges_metric.reset_index(drop=True)

    sample_count = len(edges_sample)

    # Build spatial index (returns positional indices after reset_index)
    sindex = edges_sample.sindex

    duplicate_positions: set[int] = set()
    checked_pairs: set[tuple[int, int]] = set()
    failed_comparisons = 0

    for pos in range(sample_count):
        geom = edges_sample.iloc[pos].geometry
        if geom is None or geom.is_empty:
            continue

        # Buffer and find candidates (sindex returns positional indices)
        buffered = geom.buffer(buffer_m * 2)
        candidate_positions = list(sindex.intersection(buffered.bounds))

        for cand_pos in candidate_positions:
            if cand_pos == pos:
                continue

            # Avoid checking same pair twice
            pair = (min(pos, cand_pos), max(pos, cand_pos))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)

            cand_geom = edges_sample.iloc[cand_pos].geometry
            if cand_geom is None or cand_geom.is_empty:
                continue

            # Compute buffered IOU
            try:
                buf1 = geom.buffer(buffer_m)
                buf2 = cand_geom.buffer(buffer_m)
                intersection = buf1.intersection(buf2).area
                union = buf1.union(buf2).area

                if union > 0:
                    iou = intersection / union
                    if iou > iou_threshold:
                        duplicate_positions.add(pos)
                        duplicate_positions.add(cand_pos)
            except Exception as e:
                failed_comparisons += 1
                if failed_comparisons <= 5:
                    logger.debug(f"IOU comparison failed: {e}")
                continue

    if failed_comparisons > 5:
        logger.debug(f"Total failed IOU comparisons: {failed_comparisons}")

    sample_duplicate_count = len(duplicate_positions)
    sample_ratio = sample_duplicate_count / sample_count if sample_count > 0 else 0.0

    # If sampled, extrapolate to estimate full dataset
    if is_sampled:
        estimated_count = int(sample_ratio * total_edges)
        return {
            "near_duplicate_count": estimated_count,
            "near_duplicate_ratio": sample_ratio,
        }

    return {
        "near_duplicate_count": sample_duplicate_count,
        "near_duplicate_ratio": sample_ratio,
    }


def _compute_class_coverage(
    gdf: gpd.GeoDataFrame,
    class_column: str | None,
) -> float:
    """Compute the ratio of edges that have a class assigned.

    Args:
        gdf: GeoDataFrame with edges
        class_column: Column containing road class

    Returns:
        Ratio of classified edges (0.0 to 1.0)
    """
    if class_column is None or class_column not in gdf.columns:
        return 0.0

    total_edges = len(gdf)
    if total_edges == 0:
        return 0.0

    # Vectorized classification check
    col = gdf[class_column]
    col_str = col.astype(str)
    mask = col.notna() & (col_str.str.strip() != "") & (col_str.str.lower() != "unknown")
    classified_count = int(mask.sum())

    return classified_count / total_edges
