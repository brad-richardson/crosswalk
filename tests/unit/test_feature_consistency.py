"""Tests for feature computation consistency.

These tests ensure that feature computation is consistent across all paths:
- ML scoring pipeline (ml.py)
- Backfill pipeline (backfill_features.py)
- Labeling UI (compute.py)

The key invariants are:
1. config.py::FEATURE_COLUMNS is the SINGLE SOURCE OF TRUTH
2. compute.py::compute_pair_features() returns exactly FEATURE_COLUMNS
3. compute.py::_get_error_features() returns exactly FEATURE_COLUMNS
4. ml.py delegates to compute_pair_features() - no duplicate logic
"""

import pytest
from shapely import LineString

from matcher.config import FEATURE_COLUMNS, MAX_DISTANCE_METERS
from matcher.features.compute import (
    ALL_FEATURE_COLUMNS,
    _get_error_features,
    compute_pair_features,
)


@pytest.fixture
def simple_pair_geoms():
    """Simple parallel line pair for feature testing."""
    return (
        LineString([(0, 0), (100, 0)]),  # ref
        LineString([(0, 5), (100, 5)]),  # target
    )


class TestSingleSourceOfTruth:
    """Ensure config.py FEATURE_COLUMNS is the single source of truth."""

    def test_all_feature_columns_matches_config(self):
        """ALL_FEATURE_COLUMNS in compute.py should be imported from config.py."""
        assert ALL_FEATURE_COLUMNS is FEATURE_COLUMNS, (
            "ALL_FEATURE_COLUMNS in compute.py should be imported from config.py, "
            "not a separate list."
        )


class TestErrorFeaturesConsistency:
    """Ensure _get_error_features() returns exactly FEATURE_COLUMNS."""

    def test_error_features_match_feature_columns(self):
        """_get_error_features() must return exactly the keys in FEATURE_COLUMNS."""
        error_features = _get_error_features()
        assert set(error_features.keys()) == set(FEATURE_COLUMNS)

    @pytest.mark.parametrize(
        "feature",
        [
            "hausdorff_distance_m",
            "mean_hausdorff_distance_m",
            "hausdorff_p95_m",
            "projection_distance_m",
            "centroid_distance_m",
            "min_endpoint_proximity_m",
            "max_endpoint_proximity_m",
            "lateral_offset_m",
            "lateral_offset_iqr_m",
            "lateral_offset_p95_m",
        ],
    )
    def test_distance_error_defaults_to_max(self, feature):
        """Distance features should default to MAX_DISTANCE_METERS."""
        error_features = _get_error_features()
        assert error_features[feature] == MAX_DISTANCE_METERS

    @pytest.mark.parametrize(
        "feature",
        [
            "buffer_iou_5m",
            "buffer_iou_15m",
            "overlap_ratio",
            "length_ratio",
            "name_levenshtein",
            "name_jaro_winkler",
            "name_token_sort",
            "class_similarity",
        ],
    )
    def test_similarity_error_defaults_to_zero(self, feature):
        """Similarity features should default to 0.0."""
        error_features = _get_error_features()
        assert error_features[feature] == 0.0


class TestComputePairFeaturesConsistency:
    """Ensure compute_pair_features() returns exactly FEATURE_COLUMNS."""

    def test_returns_all_feature_columns(self, simple_pair_geoms):
        """compute_pair_features() must return exactly FEATURE_COLUMNS."""
        ref_geom, target_geom = simple_pair_geoms
        features = compute_pair_features(
            ref_geom, target_geom, "Main St", "Main St", "residential", "residential"
        )
        assert set(features.keys()) == set(FEATURE_COLUMNS)

    @pytest.mark.parametrize(
        "graphlet_features,expected_sim,expected_deg",
        [
            ({"graphlet_similarity": 0.8, "endpoint_degree_similarity": 0.9}, 0.8, 0.9),
            (None, 0.5, 0.5),  # Defaults when no graphlet data
        ],
    )
    def test_graphlet_features_handling(
        self, simple_pair_geoms, graphlet_features, expected_sim, expected_deg
    ):
        """Graphlet features should use provided values or defaults."""
        ref_geom, target_geom = simple_pair_geoms
        features = compute_pair_features(
            ref_geom,
            target_geom,
            "Main St",
            "Main St",
            "residential",
            "residential",
            graphlet_features=graphlet_features,
        )
        assert features["graphlet_similarity"] == expected_sim
        assert features["endpoint_degree_similarity"] == expected_deg


class TestFeatureNaming:
    """Ensure consistent feature naming conventions."""

    @pytest.mark.parametrize(
        "feature",
        [
            "hausdorff_distance_m",
            "mean_hausdorff_distance_m",
            "hausdorff_p95_m",
            "projection_distance_m",
            "centroid_distance_m",
            "min_endpoint_proximity_m",
            "max_endpoint_proximity_m",
            "lateral_offset_m",
            "lateral_offset_iqr_m",
            "lateral_offset_p95_m",
        ],
    )
    def test_distance_features_have_m_suffix(self, feature):
        """Distance features should have _m suffix to indicate meters."""
        assert feature in FEATURE_COLUMNS

    @pytest.mark.parametrize(
        "old_name",
        [
            "hausdorff_distance",
            "buffer_iou",
            "projection_distance",
            "centroid_distance",
            "start_endpoint_proximity",
            "end_endpoint_proximity",
            "lateral_offset",
            "lateral_offset_consistency",
        ],
    )
    def test_no_old_naming_conventions(self, old_name):
        """Ensure old naming conventions are not present."""
        assert old_name not in FEATURE_COLUMNS


class TestGraphletFeatures:
    """Ensure graphlet features are properly integrated."""

    @pytest.mark.parametrize(
        "feature,expected_default",
        [
            ("graphlet_similarity", 0.5),
            ("endpoint_degree_similarity", 0.5),
        ],
    )
    def test_graphlet_features_in_config_and_error(self, feature, expected_default):
        """Graphlet features should be in FEATURE_COLUMNS and _get_error_features()."""
        assert feature in FEATURE_COLUMNS
        error_features = _get_error_features()
        assert error_features[feature] == expected_default
