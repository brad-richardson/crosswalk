"""Integration tests for candidate generation to scoring pipeline.

Tests that candidates are correctly generated and that scoring
preserves the expected data flow through the pipeline.
"""

import geopandas as gpd
import pytest
from shapely import LineString

from matcher.blocking.spatial_index import CandidatePair, generate_candidates
from matcher.matching.rules import MatchDecision, score_candidates


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
            buffer_distance=30.0,
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
            buffer_distance=30.0,
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
            buffer_distance=3.0,
        )

        candidates_wide = generate_candidates(
            reference_gdf,
            target_gdf,
            buffer_distance=50.0,
        )

        # Wider buffer should produce more candidates
        assert len(candidates_wide) >= len(candidates_narrow)


class TestScoringPreservesIds:
    """Tests that scoring preserves ID linkage through the pipeline."""

    def test_candidate_ids_preserve_through_scoring(self, reference_gdf, target_gdf):
        """ref_id/target_id should be preserved from CandidatePair to MatchResult."""
        candidates = generate_candidates(
            reference_gdf,
            target_gdf,
            buffer_distance=30.0,
        )

        results = score_candidates(
            candidates,
            reference_gdf,
            target_gdf,
            ref_name_column="names",
            target_name_column="names",
        )

        # Each result should have valid ref_id and target_id
        for result in results:
            assert result.ref_id in reference_gdf["id"].values
            assert result.target_id in target_gdf["local_id"].values

        # Results should cover all candidates
        assert len(results) == len(candidates)

    def test_scoring_uses_correct_geometries(self, reference_gdf, target_gdf):
        """Scoring should use geometries from the correct GDF indices."""
        # Create a specific candidate we can verify
        candidates = [
            CandidatePair(
                ref_id="ref_1",
                ref_idx=0,
                target_id="t_1",
                target_idx=0,
                distance_estimate=5.0,
                heading_diff=0.0,
                length_ratio=1.0,
            )
        ]

        results = score_candidates(
            candidates,
            reference_gdf,
            target_gdf,
        )

        assert len(results) == 1
        result = results[0]

        # Check that result references correct IDs
        assert result.ref_id == "ref_1"
        assert result.target_id == "t_1"

        # Check that features were computed (indicates geometries were accessed)
        assert "hausdorff_distance" in result.features
        assert result.features["hausdorff_distance"] >= 0


class TestScoringDecisions:
    """Tests for scoring decision behavior."""

    def test_similar_segments_get_high_confidence(self, reference_gdf, target_gdf):
        """Geometrically similar segments with similar names should score high."""
        # t_1 is parallel to ref_1, 5m offset, similar name
        candidates = [
            CandidatePair(
                ref_id="ref_1",
                ref_idx=0,
                target_id="t_1",
                target_idx=0,
                distance_estimate=5.0,
                heading_diff=0.0,
                length_ratio=1.0,
            )
        ]

        results = score_candidates(
            candidates,
            reference_gdf,
            target_gdf,
            ref_name_column="names",
            target_name_column="names",
        )

        assert len(results) == 1
        result = results[0]

        # Should have reasonable confidence due to good geometry and similar names
        assert result.confidence > 0.5
        assert result.decision in [MatchDecision.MATCH, MatchDecision.REVIEW]

    def test_dissimilar_segments_get_low_confidence(self, reference_gdf, target_gdf):
        """Dissimilar segments should score low."""
        # Create a bad candidate: ref_1 (horizontal) vs t_3 (different area, different name)
        # t_3 is "Different Name" at a different location
        candidates = [
            CandidatePair(
                ref_id="ref_1",
                ref_idx=0,
                target_id="t_3",
                target_idx=2,
                distance_estimate=50.0,
                heading_diff=45.0,
                length_ratio=0.5,
            )
        ]

        results = score_candidates(
            candidates,
            reference_gdf,
            target_gdf,
            ref_name_column="names",
            target_name_column="names",
        )

        assert len(results) == 1
        result = results[0]

        # Should have lower confidence due to different geometry and name
        assert result.confidence < 0.8


class TestFeatureComputation:
    """Tests for feature computation through the pipeline."""

    def test_all_expected_features_computed(self, reference_gdf, target_gdf):
        """Scoring should compute all expected geometric and semantic features."""
        candidates = generate_candidates(
            reference_gdf,
            target_gdf,
            buffer_distance=30.0,
        )

        if not candidates:
            pytest.skip("No candidates generated for feature test")

        results = score_candidates(
            candidates[:1],  # Just test one
            reference_gdf,
            target_gdf,
            ref_name_column="names",
            target_name_column="names",
        )

        result = results[0]

        # Check for geometric features
        expected_geometric = [
            "hausdorff_distance",
            "buffer_iou",
            "heading_delta",
            "length_ratio",
        ]
        for feat in expected_geometric:
            assert feat in result.features, f"Missing geometric feature: {feat}"

        # Check for semantic features
        expected_semantic = ["name_levenshtein", "name_jaro_winkler"]
        for feat in expected_semantic:
            assert feat in result.features, f"Missing semantic feature: {feat}"

    def test_feature_values_are_reasonable(self, reference_gdf, target_gdf):
        """Feature values should be within expected ranges."""
        candidates = [
            CandidatePair(
                ref_id="ref_1",
                ref_idx=0,
                target_id="t_1",
                target_idx=0,
                distance_estimate=5.0,
                heading_diff=0.0,
                length_ratio=1.0,
            )
        ]

        results = score_candidates(
            candidates,
            reference_gdf,
            target_gdf,
            ref_name_column="names",
            target_name_column="names",
        )

        features = results[0].features

        # Hausdorff distance should be positive
        assert features["hausdorff_distance"] >= 0

        # IoU should be between 0 and 1
        assert 0 <= features["buffer_iou"] <= 1

        # Heading delta should be between 0 and 180
        assert 0 <= features["heading_delta"] <= 180

        # Length ratio should be between 0 and 1 (normalized)
        assert 0 <= features["length_ratio"] <= 1

        # Name similarity should be between 0 and 1
        assert 0 <= features["name_levenshtein"] <= 1


class TestAlignmentIntegration:
    """Tests for alignment integration in the pipeline."""

    def test_alignment_coverage_features_computed(self, reference_gdf, target_gdf):
        """Coverage features should be computed when alignment is enabled."""
        from matcher.features.alignment import linestring_alignment
        from matcher.features.compute import compute_pair_features

        ref_geom = reference_gdf.iloc[0].geometry
        target_geom = target_gdf.iloc[0].geometry

        # Compute alignment
        alignment = linestring_alignment(ref_geom, target_geom)

        # Compute features with alignment
        features = compute_pair_features(
            ref_geom=ref_geom,
            target_geom=target_geom,
            ref_name="Main Street",
            target_name="Main St",
            ref_class="primary",
            target_class="primary",
            alignment=alignment,
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

        # Reference covers more area than target
        ref = LineString([(0, 0), (100, 0)])
        target = LineString([(50, 2), (100, 2)])  # Only overlaps second half

        alignment = linestring_alignment(ref, target)

        features_aligned = compute_pair_features(
            ref_geom=ref,
            target_geom=target,
            ref_name=None,
            target_name=None,
            ref_class=None,
            target_class=None,
            alignment=alignment,
        )

        features_unaligned = compute_pair_features(
            ref_geom=ref,
            target_geom=target,
            ref_name=None,
            target_name=None,
            ref_class=None,
            target_class=None,
            alignment=None,
        )

        # Aligned features should have better hausdorff (comparing matching portions)
        # The unaligned version compares full 100m vs 50m, adding 50m mismatch
        assert features_aligned["hausdorff_distance"] <= features_unaligned["hausdorff_distance"]

        # Coverage features should only be present with alignment
        assert features_aligned["ref_coverage"] > 0
        assert features_unaligned["ref_coverage"] == 0
