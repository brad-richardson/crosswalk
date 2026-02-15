"""Shared fixtures for feature correctness audit tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from matcher.config import MAX_DISTANCE_METERS
from matcher.features.compute import _get_error_features, compute_pair_features
from matcher.labeling.label_store import LabelStore
from tests.conftest import MOCK_TOPOLOGY_FEATURES

LABELS_DIR = Path(__file__).parent.parent.parent / "labels"

# Expected bounds for each feature: (min, max)
# None means unbounded in that direction
FEATURE_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    # Geometric - distances (meters, non-negative, capped at MAX_DISTANCE)
    "hausdorff_distance_m": (0, MAX_DISTANCE_METERS),
    "mean_hausdorff_distance_m": (0, MAX_DISTANCE_METERS),
    "hausdorff_p95_m": (0, MAX_DISTANCE_METERS),
    "edge_distance_rmse_m": (0, MAX_DISTANCE_METERS),
    # Geometric - ratios (0-1)
    "buffer_iou_5m": (0, 1),
    "buffer_iou_15m": (0, 1),
    "collinear_gap_ratio": (0, 1),
    "angle_histogram_similarity": (0, 1),
    # Geometric - degrees
    "heading_delta": (0, 90),
    # Name similarity (0-1)
    "name_levenshtein": (0, 1),
    "name_jaro_winkler": (0, 1),
    "name_token_sort": (0, 1),
    "name_soundex": (0, 1),
    "name_metaphone": (0, 1),
    "has_name_ref": (0, 1),
    "has_name_target": (0, 1),
    "name_is_generic": (0, 1),
    "name_numeric_match": (0, 1),
    "route_prefix_match": (0, 1),
    # Class similarity (0-1)
    "class_similarity": (0, 1),
    # Endpoint distances (meters)
    "min_endpoint_proximity_m": (0, MAX_DISTANCE_METERS),
    "max_endpoint_proximity_m": (0, MAX_DISTANCE_METERS),
    "shared_endpoint_count": (0, None),
    # Lateral offset (meters)
    "lateral_offset_m": (0, MAX_DISTANCE_METERS),
    "lateral_offset_iqr_m": (0, MAX_DISTANCE_METERS),
    "lateral_offset_p95_m": (0, MAX_DISTANCE_METERS),
    # Topology - degrees (non-negative integers)
    "from_degree_ref": (0, None),
    "to_degree_ref": (0, None),
    "from_degree_target": (0, None),
    "to_degree_target": (0, None),
    # Topology - scores (0-1)
    "degree_match_score": (0, 1),
    "degree_signature_similarity": (0, 1),
    # Topology - binary (0 or 1)
    "is_dead_end_ref": (0, 1),
    "is_dead_end_target": (0, 1),
    "dead_end_match": (0, 1),
    "is_intersection_ref": (0, 1),
    "is_intersection_target": (0, 1),
    "intersection_match": (0, 1),
    # Coverage (0-1)
    "ref_coverage": (0, 1),
    "target_coverage": (0, 1),
    "min_coverage": (0, 1),
    "coverage_ratio": (0, 1),
    # Graphlet (0-1)
    "graphlet_similarity": (0, 1),
    "endpoint_degree_similarity": (0, 1),
    # Clustering (0-1 for individual, 0-1 for delta)
    "clustering_coef_ref": (0, 1),
    "clustering_coef_target": (0, 1),
    "clustering_coef_delta": (0, 1),
    # Sinuosity (>= 1 for individual, >= 0 for delta)
    "sinuosity_ref": (1, None),
    "sinuosity_target": (1, None),
    "sinuosity_delta": (0, None),
    # Heading consistency (0-1 for individual, 0-1 for delta)
    "heading_consistency_ref": (0, 1),
    "heading_consistency_target": (0, 1),
    "heading_consistency_delta": (0, 1),
    # Vertex density (>= 0)
    "vertex_density_ref": (0, None),
    "vertex_density_target": (0, None),
    "vertex_density_ratio": (0, 1),
    # Length (>= 0)
    "min_length_m": (0, None),
    # Shape complexity (>= 0 for counts, >= 0 for delta)
    "shape_complexity_ref": (0, None),
    "shape_complexity_target": (0, None),
    "shape_complexity_delta": (0, None),
    # Aligned length (meters)
    "aligned_length_m": (0, None),
    # Crossing angle
    "crossing_angle_min_ref": (0, 90),
    "transverse_neighbor_fraction_ref": (0, 1),
    "crossing_angle_min_target": (0, 90),
    "transverse_neighbor_fraction_target": (0, 1),
    # Intersection overlap
    "post_node_continuation_m": (0, None),
    "endpoint_heading_divergence": (0, 90),
    # Parallel sibling
    "has_parallel_sibling_ref": (0, 1),
    "parallel_fraction_ref": (0, 1),
    "offset_vs_half_corridor_ratio": (0, None),
    "offset_over_expected_halfwidth": (0, None),
    "likely_representation_mismatch": (0, 1),
}


def make_projected_line(coords: list[tuple[float, float]]) -> LineString:
    """Create a LineString from projected coordinates (meters).

    Use this for sweep tests where you need control over exact positions.
    Coordinates are in a local projected CRS (just plain meters).
    """
    return LineString(coords)


def compute_features_simple(
    ref_line: LineString,
    target_line: LineString,
    ref_name: str | None = "Test Road",
    target_name: str | None = "Test Road",
    ref_class: str | None = "residential",
    target_class: str | None = "residential",
) -> dict[str, float]:
    """Compute features for a pair of projected LineStrings.

    Convenience wrapper that fills in defaults for non-geometric params.
    No alignment, topology, or graphlet data - just raw geometry comparison.
    """
    # Build names structs from flat convenience params
    ref_names_raw = {"primary": ref_name} if ref_name else None
    target_names_raw = {"primary": target_name} if target_name else None
    # Compute endpoint features manually
    ref_coords = np.array(ref_line.coords)
    target_coords = np.array(target_line.coords)

    # Simple endpoint proximity calculation
    from shapely.geometry import Point

    ref_start = Point(ref_coords[0])
    ref_end = Point(ref_coords[-1])
    target_start = Point(target_coords[0])
    target_end = Point(target_coords[-1])

    distances = [
        ref_start.distance(target_start),
        ref_start.distance(target_end),
        ref_end.distance(target_start),
        ref_end.distance(target_end),
    ]
    min_ep = min(distances)
    max_ep = max(min(distances[:2]), min(distances[2:]))

    endpoint_features = {
        "min_endpoint_proximity_m": min(min_ep, MAX_DISTANCE_METERS),
        "max_endpoint_proximity_m": min(max_ep, MAX_DISTANCE_METERS),
        "shared_endpoint_count": sum(1 for d in distances if d < 5.0),
    }

    return compute_pair_features(
        ref_geom_full=ref_line,
        target_geom_full=target_line,
        ref_class=ref_class,
        target_class=target_class,
        endpoint_features=endpoint_features,
        ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        ref_names_raw=ref_names_raw,
        target_names_raw=target_names_raw,
    )


@pytest.fixture(scope="session")
def labeled_features() -> pd.DataFrame:
    """Load all labeled features. Session-scoped for performance."""
    if not LABELS_DIR.exists():
        pytest.skip("No labels directory found")
    df = LabelStore.load_all(labels_dir=LABELS_DIR)
    if len(df) == 0:
        pytest.skip("No labeled data available")
    return df


@pytest.fixture(scope="session")
def match_features(labeled_features: pd.DataFrame) -> pd.DataFrame:
    """Features for match-labeled pairs only."""
    df = labeled_features[labeled_features["label"] == "match"]
    if len(df) == 0:
        pytest.skip("No match labels found")
    return df


@pytest.fixture(scope="session")
def no_match_features(labeled_features: pd.DataFrame) -> pd.DataFrame:
    """Features for no_match-labeled pairs only."""
    df = labeled_features[labeled_features["label"] == "no_match"]
    if len(df) == 0:
        pytest.skip("No no_match labels found")
    return df


@pytest.fixture
def error_features() -> dict[str, float]:
    """Default error features from _get_error_features()."""
    return _get_error_features()
