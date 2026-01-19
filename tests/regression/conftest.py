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

    Based on real labeled data from fort_collins_streets where
    segments had near-identical geometry and names.
    Original confidence: 0.9979
    """
    return {
        # Geometric features (from real labeled match)
        "hausdorff_distance": 0.17,
        "mean_hausdorff_distance": 0.1,
        "buffer_iou": 0.99,
        "overlap_ratio": 0.99,
        "heading_delta": 0.1,
        "length_ratio": 0.999,
        "centroid_distance": 0.07,
        # Semantic features (exact name match)
        "name_levenshtein": 1.0,
        "name_jaro_winkler": 1.0,
        "name_token_sort": 1.0,
        "name_soundex": 1.0,
        "name_metaphone": 1.0,
        "class_similarity": 1.0,
        # Connectivity (reasonable defaults)
        "start_endpoint_proximity": 1.0,
        "end_endpoint_proximity": 1.0,
        "shared_endpoint_count": 2,
        "lateral_offset": 0.5,
        "lateral_offset_consistency": 0.3,
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
    }


@pytest.fixture
def terrible_match_features():
    """Features representing a clear non-match.

    Based on real labeled data from boston_streets where Brad
    labeled pairs as no_match. Features show low IoU and
    completely different names.
    Original confidence: 0.4935
    """
    return {
        # Geometric features (from real labeled no_match)
        "hausdorff_distance": 46.0,
        "mean_hausdorff_distance": 30.0,
        "buffer_iou": 0.27,
        "overlap_ratio": 0.3,
        "heading_delta": 0.5,
        "length_ratio": 0.34,
        "centroid_distance": 17.0,
        # Semantic features (completely different names)
        "name_levenshtein": 0.0,
        "name_jaro_winkler": 0.0,
        "name_token_sort": 0.0,
        "name_soundex": 0.0,
        "name_metaphone": 0.0,
        "class_similarity": 0.6,
        # Connectivity
        "start_endpoint_proximity": 50.0,
        "end_endpoint_proximity": 50.0,
        "shared_endpoint_count": 0,
        "lateral_offset": 20.0,
        "lateral_offset_consistency": 15.0,
        # Topology: different patterns
        "from_degree_ref": 3,
        "to_degree_ref": 3,
        "from_degree_target": 2,
        "to_degree_target": 4,
        "degree_match_score": 0.8,
        "degree_signature_similarity": 0.5,
        "is_dead_end_ref": 0,
        "is_dead_end_target": 0,
        "dead_end_match": 1.0,
        "is_intersection_ref": 1,
        "is_intersection_target": 1,
        "intersection_match": 1.0,
    }


@pytest.fixture
def borderline_match_features():
    """Features representing a true borderline match case.

    Based on real labeled data from boston_streets where Brad
    labeled as match despite moderate confidence (0.59).
    Features show mediocre geometry but partial name match.
    Original confidence: 0.5908
    """
    return {
        # Moderate geometry (from real borderline case)
        "hausdorff_distance": 24.4,
        "mean_hausdorff_distance": 15.0,
        "buffer_iou": 0.30,
        "overlap_ratio": 0.35,
        "heading_delta": 3.7,
        "length_ratio": 0.74,
        "centroid_distance": 20.8,
        # Partial name match (similar but not identical)
        "name_levenshtein": 0.64,
        "name_jaro_winkler": 0.86,
        "name_token_sort": 0.64,
        "name_soundex": 1.0,
        "name_metaphone": 0.8,
        # Similar class
        "class_similarity": 0.8,
        # Moderate connectivity
        "start_endpoint_proximity": 25.0,
        "end_endpoint_proximity": 30.0,
        "shared_endpoint_count": 1,
        # Moderate lateral offset
        "lateral_offset": 12.0,
        "lateral_offset_consistency": 8.0,
        # Mixed topology (from real data)
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
    }
