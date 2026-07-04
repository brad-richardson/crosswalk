"""Integration tests for candidate generation to scoring pipeline.

Tests that candidates are correctly generated and that scoring
preserves the expected data flow through the pipeline.
"""

import math

import geopandas as gpd
import pytest
from shapely import LineString

from matcher.blocking.spatial_index import generate_candidates


@pytest.fixture
def reference_gdf():
    """Reference network with identifiable segments."""
    return gpd.GeoDataFrame(
        {
            "id": ["ref_1", "ref_2", "ref_3"],
            "names": ["Main Street", "Oak Avenue", "Elm Road"],
            "class": ["primary", "secondary", "tertiary"],
            "geometry": [
                LineString([(0, 0), (100, 0)]),  # Horizontal
                LineString([(100, 0), (100, 100)]),  # Vertical, connects to ref_1
                LineString([(200, 200), (300, 200)]),  # Isolated
            ],
        },
        crs="EPSG:32610",
    )


@pytest.fixture
def target_gdf():
    """Target segments with varying match quality."""
    return gpd.GeoDataFrame(
        {
            "local_id": ["t_1", "t_2", "t_3", "t_4"],
            "names": ["Main St", "Oak Ave", "Different Name", "Elm Rd"],
            "class": ["primary", "secondary", "residential", "tertiary"],
            "geometry": [
                LineString([(0, 5), (100, 5)]),  # Close to ref_1, similar name
                LineString([(95, 0), (95, 100)]),  # Close to ref_2, similar name
                LineString([(50, 50), (150, 50)]),  # Crosses both but different name
                LineString([(205, 195), (295, 195)]),  # Close to ref_3, similar name
            ],
        },
        crs="EPSG:32610",
    )


class TestCandidateGeneration:
    """Tests for candidate generation behavior."""

    def test_candidate_indices_match_gdf_positions(self, reference_gdf, target_gdf):
        """CandidatePair indices should correctly reference GeoDataFrame positions."""
        candidates = generate_candidates(
            reference_gdf,
            target_gdf,
            buffer_distance_m=30.0,
        )

        for cand in candidates:
            # ref_idx should be a valid index into reference_gdf
            assert 0 <= cand.ref_idx < len(reference_gdf)

            # target_idx should be a valid index into target_gdf
            assert 0 <= cand.target_idx < len(target_gdf)

            # ref_id should match the id at ref_idx
            assert cand.ref_id == reference_gdf.iloc[cand.ref_idx]["id"]

            # target_id should match the local_id at target_idx
            assert cand.target_id == target_gdf.iloc[cand.target_idx]["local_id"]

    def test_candidates_found_for_nearby_segments(self, reference_gdf, target_gdf):
        """Nearby segments should generate candidate pairs."""
        candidates = generate_candidates(
            reference_gdf,
            target_gdf,
            buffer_distance_m=30.0,
        )

        # Should find at least some candidates
        assert len(candidates) > 0

        # Check for expected pairs (t_1 near ref_1, t_4 near ref_3)
        ref_target_pairs = {(c.ref_id, c.target_id) for c in candidates}

        # t_1 is 5m from ref_1 - should be a candidate
        assert ("ref_1", "t_1") in ref_target_pairs

        # t_4 is ~7m from ref_3 - should be a candidate
        assert ("ref_3", "t_4") in ref_target_pairs

    def test_buffer_distance_affects_candidates(self, reference_gdf, target_gdf):
        """Buffer distance should control which segments are candidates."""
        candidates_narrow = generate_candidates(
            reference_gdf,
            target_gdf,
            buffer_distance_m=3.0,
        )

        candidates_wide = generate_candidates(
            reference_gdf,
            target_gdf,
            buffer_distance_m=50.0,
        )

        # Wider buffer should produce more candidates
        assert len(candidates_wide) >= len(candidates_narrow)


class TestAlignmentIntegration:
    """Tests for alignment integration in the pipeline."""

    def test_alignment_coverage_features_computed(self, reference_gdf, target_gdf):
        """Coverage features should be computed when alignment is enabled."""
        from matcher.features.alignment import linestring_alignment
        from matcher.features.compute import compute_pair_features
        from tests.conftest import MOCK_TOPOLOGY_FEATURES

        ref_geom = reference_gdf.iloc[0].geometry
        target_geom = target_gdf.iloc[0].geometry

        # Compute alignment
        alignment = linestring_alignment(ref_geom, target_geom)

        # Mock endpoint features (required parameter - test is checking coverage features)
        mock_endpoint_features = {
            "min_endpoint_proximity_m": 0.0,
            "max_endpoint_proximity_m": 0.0,
            "shared_endpoint_count": 0,
        }

        # Compute features with alignment
        features = compute_pair_features(
            ref_geom_full=ref_geom,
            target_geom_full=target_geom,
            ref_class="primary",
            target_class="primary",
            alignment=alignment,
            endpoint_features=mock_endpoint_features,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        # Verify coverage features are present and valid
        assert "ref_coverage" in features
        assert "target_coverage" in features
        assert "min_coverage" in features
        assert "coverage_ratio" in features

        # Coverage values should be between 0 and 1
        assert 0 <= features["ref_coverage"] <= 1
        assert 0 <= features["target_coverage"] <= 1
        assert 0 <= features["min_coverage"] <= 1
        assert 0 <= features["coverage_ratio"] <= 1

        # For parallel lines with similar length, coverage should be high
        assert features["ref_coverage"] > 0.8
        assert features["target_coverage"] > 0.8

    def test_partial_overlap_produces_lower_coverage(self):
        """Partial overlap should produce lower coverage than full overlap."""
        from matcher.features.alignment import compute_coverage_features, linestring_alignment

        # Full overlap case
        ref_full = LineString([(0, 0), (100, 0)])
        target_full = LineString([(0, 0), (100, 0)])
        alignment_full = linestring_alignment(ref_full, target_full)
        coverage_full = compute_coverage_features(alignment_full)

        # Partial overlap case - target is half the length
        ref_partial = LineString([(0, 0), (100, 0)])
        target_partial = LineString([(50, 0), (100, 0)])
        alignment_partial = linestring_alignment(ref_partial, target_partial)
        coverage_partial = compute_coverage_features(alignment_partial)

        # Full overlap should have higher ref_coverage
        assert coverage_full["ref_coverage"] > coverage_partial["ref_coverage"]

        # But target coverage for partial should still be high (target is fully used)
        assert coverage_partial["target_coverage"] > 0.8

    def test_aligned_features_differ_from_full_geometry_features(self):
        """Aligned features should differ from full geometry features for partial overlap."""
        from matcher.features.alignment import linestring_alignment
        from matcher.features.compute import compute_pair_features
        from tests.conftest import MOCK_TOPOLOGY_FEATURES

        # Reference covers more area than target
        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(50, 2), (100, 2)])  # Only overlaps second half

        alignment = linestring_alignment(ref, target)

        # Mock endpoint features (required parameter - test is checking alignment behavior)
        mock_endpoint_features = {
            "min_endpoint_proximity_m": 0.0,
            "max_endpoint_proximity_m": 0.0,
            "shared_endpoint_count": 0,
        }

        features_aligned = compute_pair_features(
            ref_geom_full=ref,
            target_geom_full=target,
            ref_class=None,
            target_class=None,
            alignment=alignment,
            endpoint_features=mock_endpoint_features,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        features_unaligned = compute_pair_features(
            ref_geom_full=ref,
            target_geom_full=target,
            ref_class=None,
            target_class=None,
            alignment=None,
            endpoint_features=mock_endpoint_features,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        # Aligned features should have better hausdorff (comparing matching portions)
        # The unaligned version compares full 100m vs 50m, adding 50m mismatch
        assert (
            features_aligned["hausdorff_distance_m"] <= features_unaligned["hausdorff_distance_m"]
        )

        # Coverage features should only be present with alignment; without
        # one, coverage is NaN (a computation failure, not zero overlap)
        assert features_aligned["ref_coverage"] > 0
        assert math.isnan(features_unaligned["ref_coverage"])


class TestFeatureComputation:
    """Tests for feature computation through the pipeline."""

    def test_all_expected_features_computed(self, reference_gdf, target_gdf):
        """Feature computation should produce all expected features."""
        from matcher.features.alignment import linestring_alignment
        from matcher.features.compute import compute_pair_features
        from tests.conftest import MOCK_TOPOLOGY_FEATURES

        ref_geom = reference_gdf.iloc[0].geometry
        target_geom = target_gdf.iloc[0].geometry
        alignment = linestring_alignment(ref_geom, target_geom)

        features = compute_pair_features(
            ref_geom_full=ref_geom,
            target_geom_full=target_geom,
            ref_class="primary",
            target_class="primary",
            alignment=alignment,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
        )

        # Check for geometric features (buffer_iou replaced with buffer_iou_5m and buffer_iou_15m)
        expected_geometric = [
            "hausdorff_distance_m",
            "buffer_iou_5m",
            "buffer_iou_15m",
            "heading_delta",
        ]
        for feat in expected_geometric:
            assert feat in features, f"Missing geometric feature: {feat}"

        # Check for semantic features
        expected_semantic = ["name_levenshtein", "name_jaro_winkler"]
        for feat in expected_semantic:
            assert feat in features, f"Missing semantic feature: {feat}"

    def test_feature_values_are_reasonable(self, reference_gdf, target_gdf):
        """Feature values should be within expected ranges."""
        from matcher.features.alignment import linestring_alignment
        from matcher.features.compute import compute_pair_features
        from tests.conftest import MOCK_ENDPOINT_FEATURES, MOCK_TOPOLOGY_FEATURES

        ref_geom = reference_gdf.iloc[0].geometry
        target_geom = target_gdf.iloc[0].geometry
        alignment = linestring_alignment(ref_geom, target_geom)

        features = compute_pair_features(
            ref_geom_full=ref_geom,
            target_geom_full=target_geom,
            ref_class="primary",
            target_class="primary",
            alignment=alignment,
            endpoint_features=MOCK_ENDPOINT_FEATURES,
            ref_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            target_topology=MOCK_TOPOLOGY_FEATURES.copy(),
            ref_names_raw={"primary": "Main Street"},
            target_names_raw={"primary": "Main St"},
        )

        # Hausdorff distance should be positive
        assert features["hausdorff_distance_m"] >= 0

        # IoU should be between 0 and 1 (both 5m and 15m variants)
        assert 0 <= features["buffer_iou_5m"] <= 1
        assert 0 <= features["buffer_iou_15m"] <= 1

        # Heading delta should be between 0 and 180
        assert 0 <= features["heading_delta"] <= 180

        # Name similarity should be between 0 and 1
        assert 0 <= features["name_levenshtein"] <= 1
