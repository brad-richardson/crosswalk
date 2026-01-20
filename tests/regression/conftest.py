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
        pytest.skip(f"Model not found at {path}. Run 'matcher train --combined' first.")
    return path


@pytest.fixture
def trained_matcher(model_path):
    """Return a loaded MLMatcher instance."""
    from matcher.matching.ml import MLMatcher

    return MLMatcher(str(model_path))


@pytest.fixture
def perfect_match_features():
    """Features representing a near-perfect match.

    Values are in WGS84 degrees (model is trained on degree-based features).
    Based on real labeled data from boston_streets where best matches have
    hausdorff_distance ~0.00001 degrees (~1 meter).
    """
    return {
        # Geometric features (in WGS84 degrees)
        "hausdorff_distance": 0.000001,  # ~0.1 meter
        "mean_hausdorff_distance": 0.0000008,
        "buffer_iou": 0.9999,
        "overlap_ratio": 0.99,
        "heading_delta": 0.1,
        "length_ratio": 0.999,
        "centroid_distance": 0.0000005,
        "collinear_gap_ratio": 0.01,
        # Semantic features (exact name match)
        "name_levenshtein": 1.0,
        "name_jaro_winkler": 1.0,
        "name_token_sort": 1.0,
        "name_soundex": 1.0,
        "name_metaphone": 1.0,
        "class_similarity": 1.0,
        # Connectivity (in WGS84 degrees)
        "start_endpoint_proximity": 0.00001,  # ~1 meter
        "end_endpoint_proximity": 0.00001,
        "shared_endpoint_count": 2,
        "lateral_offset": 0.00001,
        "lateral_offset_consistency": 0.000005,
        "projection_distance": 0.000001,
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
    }


@pytest.fixture
def terrible_match_features():
    """Features representing a clear non-match.

    Values are in WGS84 degrees (model is trained on degree-based features).
    Large distances and poor overlap indicate a clear non-match.
    """
    return {
        # Geometric features (in WGS84 degrees - large distances)
        "hausdorff_distance": 0.001,  # ~100 meters - very far apart
        "mean_hausdorff_distance": 0.0008,
        "buffer_iou": 0.5,  # Low overlap
        "overlap_ratio": 0.3,
        "heading_delta": 45.0,  # Significantly different heading
        "length_ratio": 0.34,
        "centroid_distance": 0.0005,  # ~50 meters apart
        "collinear_gap_ratio": 0.8,
        "projection_distance": 0.001,
        # Semantic features (completely different names)
        "name_levenshtein": 0.0,
        "name_jaro_winkler": 0.0,
        "name_token_sort": 0.0,
        "name_soundex": 0.0,
        "name_metaphone": 0.0,
        "class_similarity": 0.3,
        # Connectivity (far from other segments)
        "start_endpoint_proximity": 100.0,  # Far from network
        "end_endpoint_proximity": 100.0,
        "shared_endpoint_count": 0,
        "lateral_offset": 0.001,  # ~100 meters offset
        "lateral_offset_consistency": 0.0008,
        # Topology: different patterns
        "from_degree_ref": 3,
        "to_degree_ref": 3,
        "from_degree_target": 1,
        "to_degree_target": 1,
        "degree_match_score": 0.2,
        "degree_signature_similarity": 0.2,
        "is_dead_end_ref": 0,
        "is_dead_end_target": 1,
        "dead_end_match": 0.0,
        "is_intersection_ref": 1,
        "is_intersection_target": 0,
        "intersection_match": 0.0,
        # Coverage features
        "ref_coverage": 0.3,
        "target_coverage": 0.3,
        "min_coverage": 0.3,
        "coverage_ratio": 0.5,
    }


@pytest.fixture
def borderline_match_features():
    """Features representing a true borderline match case.

    Values are in WGS84 degrees (model is trained on degree-based features).
    Moderate geometry quality with partial name match - should be in REVIEW range.
    """
    return {
        # Moderate geometry (in WGS84 degrees)
        "hausdorff_distance": 0.0001,  # ~10 meters
        "mean_hausdorff_distance": 0.00006,
        "buffer_iou": 0.998,  # Reasonable overlap on aligned sublines
        "overlap_ratio": 0.7,
        "heading_delta": 3.7,
        "length_ratio": 0.74,
        "centroid_distance": 0.00008,
        "collinear_gap_ratio": 0.3,
        "projection_distance": 0.0001,
        # Partial name match (similar but not identical)
        "name_levenshtein": 0.64,
        "name_jaro_winkler": 0.86,
        "name_token_sort": 0.64,
        "name_soundex": 1.0,
        "name_metaphone": 0.8,
        # Similar class
        "class_similarity": 0.8,
        # Moderate connectivity
        "start_endpoint_proximity": 25.0,  # Meters (not super close)
        "end_endpoint_proximity": 30.0,
        "shared_endpoint_count": 1,
        # Moderate lateral offset
        "lateral_offset": 0.0003,  # ~30 meters
        "lateral_offset_consistency": 0.0002,
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
    }
