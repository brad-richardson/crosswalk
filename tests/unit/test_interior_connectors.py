"""Tests for interior connector and shared anchor features."""

import pytest

from matcher.features.spatial_context import (
    _connectors_near_endpoint,
    compute_interior_connector_features,
    compute_shared_anchor_features,
)

# Shared test fixtures: node_features with degree >= 2 (junctions)
JUNCTIONS = {10: 3, 20: 2, 30: 4, 40: 2, 50: 3}
# Dead-end node (degree 1) — should be filtered out by interior connector logic
DEAD_END = {10: 1, 20: 1}


class TestInteriorConnectorJaccard:
    """Test interior_connector_jaccard for various overlap scenarios."""

    @pytest.mark.parametrize(
        "ref_connectors,target_connectors,expected_jaccard",
        [
            # Perfect match: same junction IDs on both sides
            ([(0.3, 10), (0.7, 20)], [(0.3, 10), (0.7, 20)], 1.0),
            # No overlap: completely different junction IDs
            ([(0.3, 10), (0.7, 20)], [(0.3, 30), (0.7, 40)], 0.0),
            # Partial overlap: 1 shared out of 3 unique = 1/3
            ([(0.3, 10), (0.7, 20)], [(0.3, 10), (0.7, 30)], 1 / 3),
            # Both empty: perfect match by convention
            ([], [], 1.0),
            # One empty, one has junctions
            ([(0.5, 10)], [], 0.0),
            ([], [(0.5, 10)], 0.0),
            # Single shared junction
            ([(0.5, 10)], [(0.5, 10)], 1.0),
            # 2 shared out of 4 unique = 2/4 = 0.5
            ([(0.3, 10), (0.5, 20), (0.7, 30)], [(0.3, 10), (0.5, 20), (0.7, 40)], 2 / 4),
        ],
        ids=[
            "perfect_match",
            "no_overlap",
            "partial_1of3",
            "both_empty",
            "ref_only",
            "target_only",
            "single_shared",
            "partial_2of4",
        ],
    )
    def test_jaccard(self, ref_connectors, target_connectors, expected_jaccard):
        result = compute_interior_connector_features(
            "ref",
            "tgt",
            {"ref": [(0.0, 99)] + ref_connectors + [(1.0, 98)]},  # endpoints at 0/1
            {"tgt": [(0.0, 99)] + target_connectors + [(1.0, 98)]},
            JUNCTIONS,
            JUNCTIONS,
            0.0,
            1.0,
            0.0,
            1.0,
        )
        assert result["interior_connector_jaccard"] == pytest.approx(expected_jaccard)


class TestInteriorConnectorCounts:
    """Test junction counting and delta."""

    @pytest.mark.parametrize(
        "ref_interior,target_interior,exp_ref,exp_target,exp_delta",
        [
            ([(0.5, 10)], [(0.5, 10)], 1, 1, 0),
            ([(0.3, 10), (0.7, 20)], [(0.5, 10)], 2, 1, 1),
            ([], [], 0, 0, 0),
            ([(0.3, 10), (0.5, 20), (0.7, 30)], [], 3, 0, 3),
        ],
        ids=["equal", "ref_more", "both_empty", "target_empty"],
    )
    def test_counts(self, ref_interior, target_interior, exp_ref, exp_target, exp_delta):
        result = compute_interior_connector_features(
            "ref",
            "tgt",
            {"ref": ref_interior},
            {"tgt": target_interior},
            JUNCTIONS,
            JUNCTIONS,
            0.0,
            1.0,
            0.0,
            1.0,
        )
        assert result["interior_junction_count_ref"] == exp_ref
        assert result["interior_junction_count_target"] == exp_target
        assert result["interior_junction_count_delta"] == exp_delta


class TestInteriorConnectorFiltering:
    """Test that dead-end connectors are excluded and alignment range is respected."""

    def test_endpoint_connectors_included(self):
        """Connectors at alignment endpoints are included (inclusive range)."""
        result = compute_interior_connector_features(
            "ref",
            "tgt",
            {"ref": [(0.0, 10), (0.005, 20), (0.5, 30), (0.995, 40), (1.0, 50)]},
            {"tgt": []},
            JUNCTIONS,
            JUNCTIONS,
            0.0,
            1.0,
            0.0,
            1.0,
        )
        # All 5 nodes are junctions (degree >= 2) and within [0.0, 1.0]
        assert result["interior_junction_count_ref"] == 5

    def test_dead_ends_excluded(self):
        """Nodes with degree < 2 are not counted as junctions."""
        result = compute_interior_connector_features(
            "ref",
            "tgt",
            {"ref": [(0.5, 10)]},  # node 10 has degree 1 in DEAD_END
            {"tgt": []},
            DEAD_END,
            DEAD_END,
            0.0,
            1.0,
            0.0,
            1.0,
        )
        assert result["interior_junction_count_ref"] == 0

    def test_partial_alignment_range(self):
        """Only connectors within the aligned fraction range are counted."""
        # Alignment covers 0.2–0.8; connectors at 0.1, 0.5, 0.9
        result = compute_interior_connector_features(
            "ref",
            "tgt",
            {"ref": [(0.1, 10), (0.5, 20), (0.9, 30)]},
            {"tgt": []},
            JUNCTIONS,
            JUNCTIONS,
            0.2,
            0.8,
            0.0,
            1.0,  # ref aligned 0.2–0.8
        )
        # Only node 20 at 0.5 is inside [0.2, 0.8]
        assert result["interior_junction_count_ref"] == 1

    def test_epsilon_captures_boundary_drift(self):
        """Connector just outside alignment range (within 1e-4 eps) is included."""
        # Alignment is 0.2–0.8, connector at 0.19995 (half of eps outside start)
        result = compute_interior_connector_features(
            "ref",
            "tgt",
            {"ref": [(0.19995, 10), (0.5, 20), (0.80005, 30)]},
            {"tgt": []},
            JUNCTIONS,
            JUNCTIONS,
            0.2,
            0.8,
            0.0,
            1.0,
        )
        # All 3 within eps tolerance of [0.2, 0.8]
        assert result["interior_junction_count_ref"] == 3

    def test_epsilon_does_not_overreach(self):
        """Connector well outside alignment range is excluded despite epsilon."""
        result = compute_interior_connector_features(
            "ref",
            "tgt",
            {"ref": [(0.19, 10), (0.5, 20), (0.81, 30)]},
            {"tgt": []},
            JUNCTIONS,
            JUNCTIONS,
            0.2,
            0.8,
            0.0,
            1.0,
        )
        # Only node 20 at 0.5; 0.19 and 0.81 are > 1e-4 outside range
        assert result["interior_junction_count_ref"] == 1


class TestInteriorPositionSimilarity:
    """Test position similarity scoring."""

    @pytest.mark.parametrize(
        "ref_fracs,target_fracs,expected_sim",
        [
            # Identical positions
            ([0.5], [0.5], 1.0),
            ([0.3, 0.7], [0.3, 0.7], 1.0),
            # Both empty
            ([], [], 1.0),
            # One empty
            ([0.5], [], 0.0),
            # Offset positions: mean diff = 0.1
            ([0.5], [0.6], 0.9),
        ],
        ids=["single_exact", "two_exact", "both_empty", "one_empty", "offset_0.1"],
    )
    def test_position_similarity(self, ref_fracs, target_fracs, expected_sim):
        ref_conns = [(f, i * 10) for i, f in enumerate(ref_fracs)]
        tgt_conns = [(f, i * 10) for i, f in enumerate(target_fracs)]
        # All nodes are junctions
        nf = {i * 10: 3 for i in range(10)}
        result = compute_interior_connector_features(
            "ref",
            "tgt",
            {"ref": ref_conns},
            {"tgt": tgt_conns},
            nf,
            nf,
            0.0,
            1.0,
            0.0,
            1.0,
        )
        assert result["interior_junction_position_sim"] == pytest.approx(expected_sim, abs=1e-9)


class TestSharedAnchorCount:
    """Test shared_anchor_count for endpoint connector matching."""

    @pytest.mark.parametrize(
        "ref_conns,target_conns,expected_count",
        [
            # Both endpoints share same connector
            ([(0.0, 10), (1.0, 20)], [(0.0, 10), (1.0, 20)], 2),
            # Only start shared
            ([(0.0, 10), (1.0, 20)], [(0.0, 10), (1.0, 30)], 1),
            # Only end shared
            ([(0.0, 10), (1.0, 20)], [(0.0, 30), (1.0, 20)], 1),
            # No shared connectors
            ([(0.0, 10), (1.0, 20)], [(0.0, 30), (1.0, 40)], 0),
            # Empty connectors
            ([], [(0.0, 10)], 0),
            ([(0.0, 10)], [], 0),
        ],
        ids=["both_shared", "start_only", "end_only", "none_shared", "ref_empty", "target_empty"],
    )
    def test_shared_anchor_count(self, ref_conns, target_conns, expected_count):
        result = compute_shared_anchor_features(
            "ref",
            "tgt",
            {"ref": ref_conns},
            {"tgt": target_conns},
            0.0,
            1.0,
            0.0,
            1.0,
            ref_length_m=100.0,
            target_length_m=100.0,
            tolerance_m=5.0,
        )
        assert result["shared_anchor_count"] == expected_count

    def test_tolerance_converts_to_frac(self):
        """5m tolerance on a 100m segment = 0.05 frac; connector at 0.04 should match."""
        result = compute_shared_anchor_features(
            "ref",
            "tgt",
            {"ref": [(0.04, 10)]},  # 4m from start on 100m segment
            {"tgt": [(0.03, 10)]},  # 3m from start on 100m segment
            0.0,
            1.0,
            0.0,
            1.0,
            ref_length_m=100.0,
            target_length_m=100.0,
            tolerance_m=5.0,
        )
        assert result["shared_anchor_count"] == 1

    def test_tolerance_too_tight(self):
        """Connector outside fractional tolerance shouldn't match."""
        result = compute_shared_anchor_features(
            "ref",
            "tgt",
            {"ref": [(0.1, 10)]},  # 10m from start on 100m segment
            {"tgt": [(0.0, 10)]},  # at start
            0.0,
            1.0,
            0.0,
            1.0,
            ref_length_m=100.0,
            target_length_m=100.0,
            tolerance_m=5.0,  # 5m = 0.05 frac, but gap is 0.1
        )
        assert result["shared_anchor_count"] == 0

    def test_long_segment_small_coverage(self):
        """Works correctly with long segments and small alignment fractions."""
        # 10km segment, alignment covers 0.001–0.006 (50m)
        result = compute_shared_anchor_features(
            "ref",
            "tgt",
            {"ref": [(0.001, 10), (0.006, 20)]},
            {"tgt": [(0.0, 10), (1.0, 20)]},
            0.001,
            0.006,
            0.0,
            1.0,
            ref_length_m=10000.0,
            target_length_m=50.0,
            tolerance_m=5.0,  # 0.0005 frac on ref, 0.1 frac on target
        )
        assert result["shared_anchor_count"] == 2


class TestConnectorsNearEndpoint:
    """Test the _connectors_near_endpoint helper."""

    @pytest.mark.parametrize(
        "connectors,endpoint,tolerance,expected_ids",
        [
            ([(0.0, 10), (0.05, 20), (0.5, 30)], 0.0, 0.05, {10, 20}),
            ([(0.0, 10), (0.5, 20), (1.0, 30)], 1.0, 0.05, {30}),
            ([(0.5, 10)], 0.0, 0.05, set()),
            ([], 0.0, 0.05, set()),
        ],
        ids=["start_two_within", "end_one_within", "none_within", "empty"],
    )
    def test_near_endpoint(self, connectors, endpoint, tolerance, expected_ids):
        assert _connectors_near_endpoint(connectors, endpoint, tolerance) == expected_ids
