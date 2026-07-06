"""Fixtures for ML regression tests.

These fixtures include both synthetic examples and real-world feature patterns
derived from Brad's manual labeling sessions. The real-world examples help
ensure the model behaves correctly on fuzzy, real-world data.
"""

from pathlib import Path

import pytest


@pytest.fixture
def model_path() -> Path:
    """Return path to the trained model."""
    path = Path(__file__).parent.parent.parent / "data" / "models" / "matcher_model_combined.joblib"
    if not path.exists():
        pytest.skip(f"Model not found at {path}. Run 'crosswalk train --combined' first.")
    return path


@pytest.fixture
def trained_matcher(model_path):
    """Return a loaded MLMatcher instance."""
    from crosswalk.matching.ml import MLMatcher

    return MLMatcher(str(model_path))


@pytest.fixture
def perfect_match_features():
    """Features representing a near-perfect match.

    All distance values are in meters (features are computed after projecting
    to a meter-based CRS like UTM).
    """
    return {
        # Geometric features (in meters)
        "hausdorff_distance_m": 0.1,  # 0.1 meter - nearly identical
        "mean_hausdorff_distance_m": 0.08,
        "hausdorff_p95_m": 0.12,
        "buffer_iou_5m": 0.9999,
        "buffer_iou_15m": 0.9999,
        "overlap_ratio": 0.99,
        "heading_delta": 0.1,
        "collinear_gap_ratio": 0.01,
        # Semantic features (exact name match)
        "name_levenshtein": 1.0,
        "name_jaro_winkler": 1.0,
        "name_token_sort": 1.0,
        "name_soundex": 1.0,
        "name_metaphone": 1.0,
        "has_name_ref": 1.0,
        "has_name_target": 1.0,
        "name_is_generic": 0.0,
        "class_similarity": 1.0,
        # Connectivity (in meters)
        "min_endpoint_proximity_m": 1.0,  # 1 meter
        "max_endpoint_proximity_m": 1.0,
        "shared_endpoint_count": 2,
        "lateral_offset_m": 1.0,
        "lateral_offset_iqr_m": 0.5,
        "lateral_offset_p95_m": 1.5,
        # Topology: same pattern
        "from_degree_ref": 3,
        "to_degree_ref": 3,
        "from_degree_target": 3,
        "to_degree_target": 3,
        "degree_match_score": 1.0,
        "degree_signature_similarity": 1.0,
        "is_dead_end_ref": 0,
        "is_dead_end_target": 0,
        "dead_end_match": 1.0,
        "is_intersection_ref": 1.0,
        "is_intersection_target": 1.0,
        "intersection_match": 1.0,
        # Coverage features
        "ref_coverage": 1.0,
        "target_coverage": 1.0,
        "min_coverage": 1.0,
        "coverage_ratio": 1.0,
        # Graphlet features
        "graphlet_similarity": 1.0,
        "endpoint_degree_similarity": 1.0,
        # New features (PR #74) - neutral/matching values
        "sinuosity_ref": 1.0,  # Straight line
        "sinuosity_target": 1.0,
        "sinuosity_delta": 0.0,
        "heading_consistency_ref": 1.0,  # Consistent heading
        "heading_consistency_target": 1.0,
        "heading_consistency_delta": 0.0,
        "vertex_density_ref": 0.1,  # Typical density
        "vertex_density_target": 0.1,
        "vertex_density_ratio": 1.0,
        "length_bin_ref": 1,  # Medium length (10-100m)
        "length_bin_target": 1,
        "length_bin_match": 1.0,
        "min_length_m": 50.0,
        "shape_complexity_ref": 0,  # No significant turns
        "shape_complexity_target": 0,
        "shape_complexity_delta": 0.0,
        "name_numeric_match": 1.0,  # Numeric portions match
    }


@pytest.fixture
def terrible_match_features():
    """Features representing a clear non-match.

    All distance values are in meters. Large distances and poor overlap
    indicate a clear non-match. Values are extreme to ensure confidence < 0.1.
    """
    return {
        # Geometric features (in meters - large distances)
        "hausdorff_distance_m": 150.0,  # 150 meters - very far apart
        "mean_hausdorff_distance_m": 120.0,
        "hausdorff_p95_m": 140.0,
        "buffer_iou_5m": 0.0,  # No overlap at 5m buffer
        "buffer_iou_15m": 0.0,  # No overlap at 15m buffer either
        "overlap_ratio": 0.1,
        "heading_delta": 80.0,  # Nearly perpendicular
        "collinear_gap_ratio": 0.9,
        # Semantic features (completely different names)
        "name_levenshtein": 0.0,
        "name_jaro_winkler": 0.0,
        "name_token_sort": 0.0,
        "name_soundex": 0.0,
        "name_metaphone": 0.0,
        "has_name_ref": 1.0,
        "has_name_target": 1.0,
        "name_is_generic": 0.0,
        "class_similarity": 0.1,
        # Connectivity (far from other segments)
        "min_endpoint_proximity_m": 150.0,  # Far from network
        "max_endpoint_proximity_m": 150.0,
        "shared_endpoint_count": 0,
        "lateral_offset_m": 150.0,  # 150 meters offset
        "lateral_offset_iqr_m": 100.0,
        "lateral_offset_p95_m": 180.0,
        # Topology: different patterns
        "from_degree_ref": 4,
        "to_degree_ref": 4,
        "from_degree_target": 1,
        "to_degree_target": 1,
        "degree_match_score": 0.1,
        "degree_signature_similarity": 0.1,
        "is_dead_end_ref": 0,
        "is_dead_end_target": 1,
        "dead_end_match": 0.0,
        "is_intersection_ref": 1,
        "is_intersection_target": 0,
        "intersection_match": 0.0,
        # Coverage features
        "ref_coverage": 0.2,
        "target_coverage": 0.2,
        "min_coverage": 0.2,
        "coverage_ratio": 0.3,
        # Graphlet features
        "graphlet_similarity": 0.1,
        "endpoint_degree_similarity": 0.1,
        # New features (PR #74) - mismatched values
        "sinuosity_ref": 1.0,  # Straight
        "sinuosity_target": 2.5,  # Very curvy
        "sinuosity_delta": 1.5,
        "heading_consistency_ref": 1.0,  # Consistent
        "heading_consistency_target": 0.3,  # Inconsistent
        "heading_consistency_delta": 0.7,
        "vertex_density_ref": 0.1,
        "vertex_density_target": 0.01,  # Sparse vertices
        "vertex_density_ratio": 0.1,
        "length_bin_ref": 0,  # Short (<10m)
        "length_bin_target": 3,  # Highway (>500m)
        "length_bin_match": 0.0,
        "min_length_m": 5.0,  # Very short
        "shape_complexity_ref": 0,
        "shape_complexity_target": 15,  # Many turns
        "shape_complexity_delta": 15.0,
        "name_numeric_match": 0.0,  # Numerics don't match
    }


@pytest.fixture
def borderline_match_features():
    """Features representing a true borderline match case.

    All distance values are in meters. Moderate geometry quality with
    partial name match - should be in REVIEW range.
    """
    return {
        # Moderate geometry (in meters)
        "hausdorff_distance_m": 10.0,  # 10 meters
        "mean_hausdorff_distance_m": 6.0,
        "hausdorff_p95_m": 12.0,
        "buffer_iou_5m": 0.85,  # Moderate overlap at 5m buffer
        "buffer_iou_15m": 0.95,  # Better overlap at 15m buffer
        "overlap_ratio": 0.7,
        "heading_delta": 3.7,
        "collinear_gap_ratio": 0.3,
        # Partial name match (similar but not identical)
        "name_levenshtein": 0.64,
        "name_jaro_winkler": 0.86,
        "name_token_sort": 0.64,
        "name_soundex": 1.0,
        "name_metaphone": 0.8,
        "has_name_ref": 1.0,
        "has_name_target": 1.0,
        "name_is_generic": 0.0,
        # Similar class
        "class_similarity": 0.8,
        # Moderate connectivity (in meters)
        "min_endpoint_proximity_m": 25.0,  # Not super close
        "max_endpoint_proximity_m": 30.0,
        "shared_endpoint_count": 1,
        # Moderate lateral offset
        "lateral_offset_m": 30.0,  # 30 meters
        "lateral_offset_iqr_m": 20.0,
        "lateral_offset_p95_m": 40.0,
        # Mixed topology
        "from_degree_ref": 3,
        "to_degree_ref": 4,
        "from_degree_target": 2,
        "to_degree_target": 3,
        "degree_match_score": 0.55,
        "degree_signature_similarity": 0.5,
        "is_dead_end_ref": 0,
        "is_dead_end_target": 0,
        "dead_end_match": 1.0,
        "is_intersection_ref": 1,
        "is_intersection_target": 1,
        "intersection_match": 1.0,
        # Coverage features
        "ref_coverage": 0.8,
        "target_coverage": 0.7,
        "min_coverage": 0.7,
        "coverage_ratio": 0.85,
        # Graphlet features
        "graphlet_similarity": 0.6,
        "endpoint_degree_similarity": 0.7,
        # New features (PR #74) - borderline/mixed values (some good, some bad)
        "sinuosity_ref": 1.1,
        "sinuosity_target": 1.5,  # More different
        "sinuosity_delta": 0.4,
        "heading_consistency_ref": 0.9,
        "heading_consistency_target": 0.6,  # Less consistent
        "heading_consistency_delta": 0.3,
        "vertex_density_ref": 0.1,
        "vertex_density_target": 0.05,  # Different density
        "vertex_density_ratio": 0.5,
        "length_bin_ref": 1,  # Medium
        "length_bin_target": 2,  # Different bin
        "length_bin_match": 0.0,
        "min_length_m": 15.0,  # Shorter
        "shape_complexity_ref": 2,
        "shape_complexity_target": 6,  # More complex
        "shape_complexity_delta": 4.0,
        "name_numeric_match": 0.0,  # No match
    }
