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

import math

import geopandas as gpd
import pytest
from shapely import LineString

from matcher.config import FEATURE_COLUMNS
from matcher.features.compute import (
    ALL_FEATURE_COLUMNS,
    MissingContextError,
    _get_error_features,
    compute_pair_features,
)
from tests.conftest import MOCK_ENDPOINT_FEATURES, MOCK_TOPOLOGY_FEATURES


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

    @pytest.mark.parametrize("feature", FEATURE_COLUMNS)
    def test_all_error_defaults_are_nan(self, feature):
        """All error features should default to NaN (XGBoost handles natively)."""
        error_features = _get_error_features()
        assert math.isnan(error_features[feature]), (
            f"Error default for {feature} should be NaN, got {error_features[feature]}"
        )


class TestComputePairFeaturesConsistency:
    """Ensure compute_pair_features() returns exactly FEATURE_COLUMNS."""

    def test_returns_all_feature_columns(self, simple_pair_geoms):
        """compute_pair_features() must return exactly FEATURE_COLUMNS.

        Note: Error cases may include _error* metadata fields which are
        internal tracking fields, not ML features.
        """
        ref_geom, target_geom = simple_pair_geoms
        features = compute_pair_features(
            ref_geom,
            target_geom,
            "Main St",
            "Main St",
            "residential",
            "residential",
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )
        # Filter out internal metadata fields (prefixed with _)
        feature_keys = {k for k in features if not k.startswith("_")}
        assert feature_keys == set(FEATURE_COLUMNS)

    def test_graphlet_features_with_values(self, simple_pair_geoms):
        """Graphlet features should use provided values."""
        ref_geom, target_geom = simple_pair_geoms
        features = compute_pair_features(
            ref_geom,
            target_geom,
            "Main St",
            "Main St",
            "residential",
            "residential",
            graphlet_features={"graphlet_similarity": 0.8, "endpoint_degree_similarity": 0.9},
            endpoint_features=MOCK_ENDPOINT_FEATURES,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )
        assert features["graphlet_similarity"] == 0.8
        assert features["endpoint_degree_similarity"] == 0.9

    def test_graphlet_features_default_to_nan(self, simple_pair_geoms):
        """Graphlet features should default to NaN when no graphlet data."""
        ref_geom, target_geom = simple_pair_geoms
        features = compute_pair_features(
            ref_geom,
            target_geom,
            "Main St",
            "Main St",
            "residential",
            "residential",
            graphlet_features=None,
            endpoint_features=MOCK_ENDPOINT_FEATURES,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )
        assert math.isnan(features["graphlet_similarity"])
        assert math.isnan(features["endpoint_degree_similarity"])


class TestFeatureNaming:
    """Ensure consistent feature naming conventions."""

    @pytest.mark.parametrize(
        "feature",
        [
            "hausdorff_distance_m",
            "mean_hausdorff_distance_m",
            "hausdorff_p95_m",
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
        "feature",
        [
            "graphlet_similarity",
            "endpoint_degree_similarity",
        ],
    )
    def test_graphlet_features_in_config_and_error(self, feature):
        """Graphlet features should be in FEATURE_COLUMNS and error defaults should be NaN."""
        assert feature in FEATURE_COLUMNS
        error_features = _get_error_features()
        assert math.isnan(error_features[feature])


class TestCallSiteContextConsistency:
    """Ensure all call sites provide required context to compute_pair_features.

    These tests verify:
    1. Missing topology raises MissingContextError (not silently defaults)
    2. Aligned topology path still works without explicit topology
    3. Real topology values pass through correctly
    """

    def test_missing_topology_raises(self):
        """compute_pair_features must raise MissingContextError when topology omitted."""
        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(0, 5), (100, 5)])

        with pytest.raises(MissingContextError, match="ref_topology is required"):
            compute_pair_features(
                ref_geom=ref,
                target_geom=target,
                ref_name="Main St",
                target_name="Main St",
                ref_class="residential",
                target_class="residential",
                endpoint_features=MOCK_ENDPOINT_FEATURES,
                # ref_topology and target_topology deliberately omitted
            )

    def test_topology_not_required_when_aligned_path_active(self):
        """Aligned topology path (graphlet_data + alignment + seg_ids) should work
        without explicit topology parameters."""
        from matcher.features.alignment import AlignmentResult
        from matcher.features.compute import precompute_graphlet_features

        gdf = gpd.GeoDataFrame(
            {
                "id": ["ref_1", "target_1"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(0, 5), (100, 5)]),
                ],
            },
            crs="EPSG:32632",
        )

        graphlet_data = precompute_graphlet_features(gdf, id_column="id", tolerance_m=5.0)
        alignment = AlignmentResult(
            overture_start_frac=0.0,
            overture_end_frac=1.0,
            dataset_start_frac=0.0,
            dataset_end_frac=1.0,
        )

        # Should NOT raise - aligned path computes topology from graphlet data
        features = compute_pair_features(
            ref_geom=gdf.geometry.iloc[0],
            target_geom=gdf.geometry.iloc[1],
            ref_name=None,
            target_name=None,
            ref_class=None,
            target_class=None,
            endpoint_features=MOCK_ENDPOINT_FEATURES,
            alignment=alignment,
            ref_graphlet_data=graphlet_data,
            target_graphlet_data=graphlet_data,
            ref_seg_id="ref_1",
            target_seg_id="target_1",
        )

        assert "from_degree_ref" in features
        assert "to_degree_ref" in features

    def test_topology_features_match_real_network(self):
        """Topology from compute_all_topology should produce non-default features
        that match actual network structure."""
        from matcher.features.spatial_context import compute_all_topology

        # Build T-intersection: main road + side street
        gdf = gpd.GeoDataFrame(
            {
                "id": ["main_w", "main_e", "side"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),  # Main road west
                    LineString([(100, 0), (200, 0)]),  # Main road east
                    LineString([(100, 0), (100, 100)]),  # Side street
                ],
            },
            crs="EPSG:32632",
        )

        topology = compute_all_topology(gdf, id_column="id", tolerance_m=5.0)

        # main_w: from_degree=1 (dead end), to_degree=3 (T-junction)
        assert topology["main_w"]["to_degree"] == 3
        assert topology["main_w"]["from_degree"] == 1

        # Pass real topology to compute_pair_features
        features = compute_pair_features(
            ref_geom=gdf.geometry.iloc[0],
            target_geom=gdf.geometry.iloc[1],
            ref_name=None,
            target_name=None,
            ref_class=None,
            target_class=None,
            endpoint_features=MOCK_ENDPOINT_FEATURES,
            ref_topology=topology["main_w"],
            target_topology=topology["main_e"],
        )

        # Verify non-default values came through
        # (defaults would be from_degree=1, to_degree=1 for dead ends)
        assert features["to_degree_ref"] == 3  # T-junction
        assert features["from_degree_ref"] == 1  # Dead end
        assert features["is_dead_end_ref"] == 1.0  # main_w from-end is dead end
        assert features["is_intersection_ref"] == 1.0  # main_w to-end is intersection (degree >= 3)
