"""Unit tests for integration module.

Fixtures used (from conftest.py):
- reference_gdf: Reference network with two connected segments
- target_matched_gdf: Target segments that match reference
- target_unmatched_gdf: Target segments without matches
- match_results: Mock match results for target_matched
"""

import geopandas as gpd
from shapely import LineString, Point

from matcher.fetch.overture import BoundingBox
from matcher.integration import (
    EdgeSource,
    IntegrationStatistics,
    TargetInput,
    combine_networks,
    compute_reference_coverage,
    detect_orphan_components,
    detect_orphans_by_proximity,
    extract_unmatched_remnants,
    filter_fringe_segments,
    filter_short_segments,
)
from matcher.integration.combiner import (
    _build_multi_match_ranges,
    _complement_ranges,
    _merge_ranges,
)
from matcher.screen.constants import FRINGE_BUFFER_M, FRINGE_MIN_INSIDE_LENGTH_M
from matcher.topology.planarize import planarize


class TestCombineNetworks:
    """Tests for combine_networks function."""

    def test_combine_reference_only(self, reference_gdf):
        """Reference-only combination works."""
        combined, dropped = combine_networks(
            reference=reference_gdf,
            target_inputs=[],
            ref_id_column="id",
        )

        assert len(combined) == 2
        assert len(dropped) == 0
        assert "_source" in combined.columns
        assert all(combined["_source"] == EdgeSource.REFERENCE.value)

    def test_combine_with_targets(
        self, reference_gdf, target_matched_gdf, target_unmatched_gdf, match_results
    ):
        """Combined network includes reference and targets."""
        target_input = TargetInput(
            name="test_target",
            matched=target_matched_gdf,
            unmatched=target_unmatched_gdf,
            match_results=match_results,
            priority=1,
        )

        combined, dropped = combine_networks(
            reference=reference_gdf,
            target_inputs=[target_input],
            ref_id_column="id",
            target_id_column="local_id",
        )

        # Should have: 2 reference + 1 matched + 2 unmatched = 5
        assert len(combined) == 5

        # Check sources
        sources = combined["_source"].value_counts()
        assert sources.get(EdgeSource.REFERENCE.value, 0) == 2
        assert sources.get(EdgeSource.TARGET_MATCHED.value, 0) == 1
        assert sources.get(EdgeSource.TARGET_UNMATCHED.value, 0) == 2

    def test_provenance_columns_present(self, reference_gdf):
        """Provenance columns are added to combined network."""
        combined, _ = combine_networks(
            reference=reference_gdf,
            target_inputs=[],
            ref_id_column="id",
        )

        expected_columns = [
            "_source",
            "_original_id",
            "_source_dataset",
            "_priority",
            "_match_ref_id",
            "_match_confidence",
        ]
        for col in expected_columns:
            assert col in combined.columns, f"Missing column: {col}"


class TestOrphanDetection:
    """Tests for orphan detection."""

    def test_connected_network_no_orphans(self, reference_gdf):
        """Fully connected network has no orphans."""
        # Planarize reference
        planarized = planarize(reference_gdf, id_column="id")

        # Add provenance columns
        edges = planarized.edges.copy()
        edges["_source"] = EdgeSource.REFERENCE.value

        # Detect orphans
        main_edges, orphan_edges, stats = detect_orphan_components(planarized, edges)

        assert len(orphan_edges) == 0
        assert stats["orphan_components"] == 0

    def test_disconnected_segment_is_orphan(self, reference_gdf, target_unmatched_gdf):
        """Disconnected segment is flagged as orphan."""
        # Combine with orphan
        target_input = TargetInput(
            name="test",
            matched=gpd.GeoDataFrame(),
            unmatched=target_unmatched_gdf,
            match_results=[],
            priority=1,
        )

        combined, _ = combine_networks(
            reference=reference_gdf,
            target_inputs=[target_input],
            ref_id_column="id",
            target_id_column="local_id",
        )

        # Planarize
        planarized = planarize(combined, id_column="_original_id")

        # Add edge IDs to edges for provenance lookup
        edges = planarized.edges.copy()

        # Detect orphans
        main_edges, orphan_edges, stats = detect_orphan_components(planarized, edges)

        # Should have at least one orphan (the disconnected segment)
        assert len(orphan_edges) > 0
        assert stats["orphan_components"] > 0


class TestFilters:
    """Tests for optional filters."""

    def test_filter_short_segments(self):
        """Short segments are filtered."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["short", "long"],
                "geometry": [
                    LineString([(0, 0), (1, 0)]),  # 1m - should be filtered
                    LineString([(0, 0), (100, 0)]),  # 100m - should be kept
                ],
            },
            crs="EPSG:32610",
        )

        kept, filtered = filter_short_segments(gdf, min_length_m=5.0)

        assert len(kept) == 1
        assert len(filtered) == 1
        assert kept.iloc[0]["id"] == "long"
        assert filtered.iloc[0]["id"] == "short"


class TestIntegrationStatistics:
    """Tests for IntegrationStatistics."""

    def test_to_dict(self):
        """Statistics convert to dictionary."""
        stats = IntegrationStatistics(
            reference_edges=100,
            target_edges_matched=50,
            target_edges_unmatched=25,
            dropped_overlaps=5,
            total_nodes=200,
            total_edges=175,
            main_component_edges=170,
            disconnected_edges=3,
            filtered_edges=2,
            datasets_integrated=["boston_streets"],
        )

        d = stats.to_dict()

        assert d["reference_edges"] == 100
        assert d["disconnected_edges"] == 3
        assert d["filtered_edges"] == 2
        assert "orphan_edges" not in d
        assert d["datasets_integrated"] == ["boston_streets"]


class TestTransitiveConnectivity:
    """Tests for transitive connectivity propagation."""

    def test_direct_connection_hop_0(self):
        """Directly connected targets are hop 0."""
        # Reference: horizontal line
        reference = gpd.GeoDataFrame(
            {
                "_original_id": ["ref_1"],
                "_source": [EdgeSource.REFERENCE.value],
                "geometry": [LineString([(0, 0), (100, 0)])],
            },
            crs="EPSG:32610",
        )

        # Target: connects to reference at (50, 0)
        target = gpd.GeoDataFrame(
            {
                "_original_id": ["t_1"],
                "_source": [EdgeSource.TARGET_UNMATCHED.value],
                "geometry": [LineString([(50, 0), (50, 50)])],  # Connects at (50, 0)
            },
            crs="EPSG:32610",
        )

        combined = gpd.GeoDataFrame(
            data={
                "_original_id": ["ref_1", "t_1"],
                "_source": [EdgeSource.REFERENCE.value, EdgeSource.TARGET_UNMATCHED.value],
                "geometry": [reference.geometry.iloc[0], target.geometry.iloc[0]],
            },
            crs="EPSG:32610",
        )

        main, disconnected, filtered, net_new, stats = detect_orphans_by_proximity(
            combined, connection_tolerance_m=3.0, min_merge_length_m=0, max_hops=2
        )

        # Target should be connected (hop 0)
        connected = main[main["_source"] == EdgeSource.TARGET_UNMATCHED.value]
        assert len(connected) == 1
        assert connected.iloc[0]["_connectivity_hop"] == 0

    def test_transitive_connection_hop_1(self):
        """Segments connecting to connected targets are hop 1."""
        # Reference: horizontal line
        reference_geom = LineString([(0, 0), (100, 0)])

        # Target 1: connects to reference
        t1_geom = LineString([(50, 0), (50, 50)])

        # Target 2: connects to Target 1, not reference
        t2_geom = LineString([(50, 50), (100, 50)])

        combined = gpd.GeoDataFrame(
            {
                "_original_id": ["ref_1", "t_1", "t_2"],
                "_source": [
                    EdgeSource.REFERENCE.value,
                    EdgeSource.TARGET_UNMATCHED.value,
                    EdgeSource.TARGET_UNMATCHED.value,
                ],
                "geometry": [reference_geom, t1_geom, t2_geom],
            },
            crs="EPSG:32610",
        )

        main, disconnected, filtered, net_new, stats = detect_orphans_by_proximity(
            combined, connection_tolerance_m=3.0, min_merge_length_m=0, max_hops=2
        )

        # Both targets should be connected
        connected = main[main["_source"] == EdgeSource.TARGET_UNMATCHED.value]
        assert len(connected) == 2

        # Check hop levels
        t1 = connected[connected["_original_id"] == "t_1"]
        t2 = connected[connected["_original_id"] == "t_2"]
        assert t1.iloc[0]["_connectivity_hop"] == 0
        assert t2.iloc[0]["_connectivity_hop"] == 1

    def test_max_hops_limits_depth(self):
        """Max hops parameter limits transitive depth."""
        # Chain: ref -> t1 -> t2 -> t3
        reference_geom = LineString([(0, 0), (100, 0)])
        t1_geom = LineString([(50, 0), (50, 50)])
        t2_geom = LineString([(50, 50), (100, 50)])
        t3_geom = LineString([(100, 50), (100, 100)])

        combined = gpd.GeoDataFrame(
            {
                "_original_id": ["ref_1", "t_1", "t_2", "t_3"],
                "_source": [
                    EdgeSource.REFERENCE.value,
                    EdgeSource.TARGET_UNMATCHED.value,
                    EdgeSource.TARGET_UNMATCHED.value,
                    EdgeSource.TARGET_UNMATCHED.value,
                ],
                "geometry": [reference_geom, t1_geom, t2_geom, t3_geom],
            },
            crs="EPSG:32610",
        )

        # With max_hops=1, t3 should be disconnected
        main, disconnected, filtered, net_new, stats = detect_orphans_by_proximity(
            combined, connection_tolerance_m=3.0, min_merge_length_m=0, max_hops=1
        )

        connected = main[main["_source"] == EdgeSource.TARGET_UNMATCHED.value]
        assert len(connected) == 2  # t1 and t2

        disconnected_targets = disconnected[
            disconnected["_source"] == EdgeSource.TARGET_UNMATCHED.value
        ]
        assert len(disconnected_targets) == 1  # t3

    def test_disconnected_vs_filtered_separation(self):
        """Disconnected and filtered segments are separated correctly."""
        # Reference: horizontal line
        reference_geom = LineString([(0, 0), (100, 0)])

        # Target 1: connects to reference but short (5m), will be filtered
        t1_geom = LineString([(50, 0), (50, 5)])

        # Target 2: far from reference, will be disconnected
        t2_geom = LineString([(500, 500), (500, 550)])

        combined = gpd.GeoDataFrame(
            {
                "_original_id": ["ref_1", "t_1", "t_2"],
                "_source": [
                    EdgeSource.REFERENCE.value,
                    EdgeSource.TARGET_UNMATCHED.value,
                    EdgeSource.TARGET_UNMATCHED.value,
                ],
                "geometry": [reference_geom, t1_geom, t2_geom],
            },
            crs="EPSG:32610",
        )

        main, disconnected, filtered, net_new, stats = detect_orphans_by_proximity(
            combined,
            connection_tolerance_m=3.0,
            min_merge_length_m=20.0,
            net_new_buffer_m=5.0,
            max_hops=2,
        )

        # t1 should be filtered (connected but too short net-new)
        assert len(filtered) > 0
        assert filtered.iloc[0]["component_status"] == "filtered"
        assert filtered.iloc[0]["unmatched_reason"] == "insufficient_net_new_length"

        # t2 should be disconnected
        assert len(disconnected) > 0
        assert disconnected.iloc[0]["component_status"] == "disconnected"
        assert disconnected.iloc[0]["unmatched_reason"] == "not_connected_to_network"

        # Stats should reflect the split
        assert stats["disconnected_edges"] == len(disconnected)
        assert stats["filtered_edges"] == len(filtered)


class TestFringeDetection:
    """Tests for fringe/coverage detection."""

    def test_compute_reference_coverage(self):
        """Coverage polygon is computed from reference network."""
        reference = gpd.GeoDataFrame(
            {
                "id": ["1", "2"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(0, 0), (0, 100)]),
                ],
            },
            crs="EPSG:32610",
        )

        coverage = compute_reference_coverage(reference, buffer_distance_m=10.0)

        assert coverage is not None
        # Coverage should contain the reference network points
        assert coverage.contains(Point(50, 0))
        assert coverage.contains(Point(0, 50))

    def test_filter_fringe_segments(self):
        """Segments outside coverage are marked as fringe."""
        # Create a more realistic reference network (L-shape)
        reference = gpd.GeoDataFrame(
            {
                "id": ["1", "2"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),  # Horizontal
                    LineString([(100, 0), (100, 100)]),  # Vertical
                ],
            },
            crs="EPSG:32610",
        )

        # Target inside coverage (parallel to reference, within buffer)
        inside = gpd.GeoDataFrame(
            {
                "id": ["inside"],
                "geometry": [LineString([(10, 10), (90, 10)])],  # 10m from horizontal ref
            },
            crs="EPSG:32610",
        )

        # Target outside coverage (far from reference)
        outside = gpd.GeoDataFrame(
            {
                "id": ["outside"],
                "geometry": [LineString([(500, 500), (600, 500)])],  # Far from reference
            },
            crs="EPSG:32610",
        )

        # Test inside segment - should be valid (inside_length >= min threshold)
        valid, fringe = filter_fringe_segments(
            inside,
            reference,
            buffer_distance_m=FRINGE_BUFFER_M,
            min_inside_length_m=FRINGE_MIN_INSIDE_LENGTH_M,
        )
        assert len(valid) == 1
        assert len(fringe) == 0

        # Test outside segment - should be fringe
        valid, fringe = filter_fringe_segments(
            outside,
            reference,
            buffer_distance_m=FRINGE_BUFFER_M,
            min_inside_length_m=FRINGE_MIN_INSIDE_LENGTH_M,
        )
        assert len(valid) == 0
        assert len(fringe) == 1
        assert fringe.iloc[0]["unmatched_reason"] == "outside_reference_coverage"


class TestBoundingBoxExpand:
    """Tests for BoundingBox expand method."""

    def test_expand_increases_bbox(self):
        """Expanding a bbox increases its dimensions."""
        bbox = BoundingBox(xmin=-122.0, ymin=45.0, xmax=-121.0, ymax=46.0)
        expanded = bbox.expand(1000.0)  # 1km buffer

        assert expanded.xmin < bbox.xmin
        assert expanded.ymin < bbox.ymin
        assert expanded.xmax > bbox.xmax
        assert expanded.ymax > bbox.ymax

    def test_expand_by_zero_unchanged(self):
        """Expanding by 0 returns same dimensions."""
        bbox = BoundingBox(xmin=-122.0, ymin=45.0, xmax=-121.0, ymax=46.0)
        expanded = bbox.expand(0.0)

        assert expanded.xmin == bbox.xmin
        assert expanded.ymin == bbox.ymin
        assert expanded.xmax == bbox.xmax
        assert expanded.ymax == bbox.ymax

    def test_expand_symmetric(self):
        """Expansion is symmetric around center."""
        bbox = BoundingBox(xmin=-122.0, ymin=45.0, xmax=-121.0, ymax=46.0)
        expanded = bbox.expand(1000.0)

        center_x = (bbox.xmin + bbox.xmax) / 2
        center_y = (bbox.ymin + bbox.ymax) / 2

        expanded_center_x = (expanded.xmin + expanded.xmax) / 2
        expanded_center_y = (expanded.ymin + expanded.ymax) / 2

        # Centers should remain the same (within floating point tolerance)
        assert abs(center_x - expanded_center_x) < 1e-6
        assert abs(center_y - expanded_center_y) < 1e-6


class TestMergeRanges:
    """Tests for _merge_ranges helper."""

    def test_empty_input(self):
        assert _merge_ranges([]) == []

    def test_single_range(self):
        assert _merge_ranges([(0.2, 0.5)]) == [(0.2, 0.5)]

    def test_non_overlapping(self):
        result = _merge_ranges([(0.1, 0.3), (0.5, 0.8)])
        assert result == [(0.1, 0.3), (0.5, 0.8)]

    def test_overlapping(self):
        result = _merge_ranges([(0.1, 0.5), (0.3, 0.8)])
        assert result == [(0.1, 0.8)]

    def test_adjacent_within_tolerance(self):
        """Ranges within tolerance gap are merged."""
        result = _merge_ranges([(0.1, 0.5), (0.505, 0.8)], tolerance=0.01)
        assert result == [(0.1, 0.8)]

    def test_unsorted_input(self):
        result = _merge_ranges([(0.5, 0.8), (0.1, 0.3)])
        assert result == [(0.1, 0.3), (0.5, 0.8)]

    def test_three_ranges_with_chain_merge(self):
        result = _merge_ranges([(0.0, 0.3), (0.25, 0.6), (0.55, 0.9)])
        assert result == [(0.0, 0.9)]


class TestComplementRanges:
    """Tests for _complement_ranges helper."""

    def test_empty_matched(self):
        assert _complement_ranges([]) == [(0.0, 1.0)]

    def test_full_match(self):
        assert _complement_ranges([(0.0, 1.0)]) == []

    def test_middle_match(self):
        result = _complement_ranges([(0.3, 0.7)])
        assert result == [(0.0, 0.3), (0.7, 1.0)]

    def test_start_match(self):
        result = _complement_ranges([(0.0, 0.5)])
        assert result == [(0.5, 1.0)]

    def test_end_match(self):
        result = _complement_ranges([(0.5, 1.0)])
        assert result == [(0.0, 0.5)]

    def test_two_matches_with_gaps(self):
        result = _complement_ranges([(0.1, 0.3), (0.6, 0.9)])
        assert result == [(0.0, 0.1), (0.3, 0.6), (0.9, 1.0)]


class TestBuildMultiMatchRanges:
    """Tests for _build_multi_match_ranges."""

    def test_single_match_dict(self):
        results = [
            {
                "local_id": "seg1",
                "local_start_frac": 0.3,
                "local_end_frac": 0.7,
                "match_decision": "match",
            }
        ]
        ranges = _build_multi_match_ranges(results)
        assert ranges == {"seg1": [(0.3, 0.7)]}

    def test_multiple_matches_same_target(self):
        results = [
            {
                "local_id": "seg1",
                "local_start_frac": 0.0,
                "local_end_frac": 0.3,
                "match_decision": "match",
            },
            {
                "local_id": "seg1",
                "local_start_frac": 0.6,
                "local_end_frac": 1.0,
                "match_decision": "match",
            },
        ]
        ranges = _build_multi_match_ranges(results)
        assert len(ranges["seg1"]) == 2
        assert (0.0, 0.3) in ranges["seg1"]
        assert (0.6, 1.0) in ranges["seg1"]

    def test_review_decisions_excluded(self):
        results = [
            {
                "local_id": "seg1",
                "local_start_frac": 0.3,
                "local_end_frac": 0.7,
                "match_decision": "review",
            },
        ]
        ranges = _build_multi_match_ranges(results)
        assert ranges == {}

    def test_none_fractions_skipped(self):
        results = [
            {
                "local_id": "seg1",
                "local_start_frac": None,
                "local_end_frac": None,
                "match_decision": "match",
            },
        ]
        ranges = _build_multi_match_ranges(results)
        assert ranges == {}


class TestExtractUnmatchedRemnants:
    """Tests for extract_unmatched_remnants."""

    def test_partial_match_produces_remnants(self):
        """A segment matched at 30-70% should produce remnants at 0-30% and 70-100%."""
        # 100m horizontal line in projected CRS
        line = LineString([(0, 0), (100, 0)])
        matched = gpd.GeoDataFrame(
            {"local_id": ["seg1"], "geometry": [line]},
            crs="EPSG:32610",
        )

        match_results = [
            {
                "local_id": "seg1",
                "local_start_frac": 0.3,
                "local_end_frac": 0.7,
                "match_decision": "match",
            },
        ]

        remnants = extract_unmatched_remnants(matched, match_results, min_remnant_length_m=1.0)

        assert len(remnants) == 2
        # Remnants should be ~30m each
        for _, row in remnants.iterrows():
            assert 25 < row.geometry.length < 35

        # Check IDs
        ids = set(remnants["local_id"])
        assert "seg1_remnant_0" in ids
        assert "seg1_remnant_1" in ids

    def test_full_match_no_remnants(self):
        """A fully matched segment produces no remnants."""
        line = LineString([(0, 0), (100, 0)])
        matched = gpd.GeoDataFrame(
            {"local_id": ["seg1"], "geometry": [line]},
            crs="EPSG:32610",
        )

        match_results = [
            {
                "local_id": "seg1",
                "local_start_frac": 0.0,
                "local_end_frac": 1.0,
                "match_decision": "match",
            },
        ]

        remnants = extract_unmatched_remnants(matched, match_results, min_remnant_length_m=1.0)
        assert len(remnants) == 0

    def test_short_remnants_filtered(self):
        """Remnants shorter than min_remnant_length_m are filtered out."""
        line = LineString([(0, 0), (100, 0)])
        matched = gpd.GeoDataFrame(
            {"local_id": ["seg1"], "geometry": [line]},
            crs="EPSG:32610",
        )

        # Match covers 0-98%, leaving only 2m remnant
        match_results = [
            {
                "local_id": "seg1",
                "local_start_frac": 0.0,
                "local_end_frac": 0.98,
                "match_decision": "match",
            },
        ]

        remnants = extract_unmatched_remnants(matched, match_results, min_remnant_length_m=3.0)
        assert len(remnants) == 0

    def test_1_to_n_match_merges_ranges(self):
        """Multiple matches on same target merge their ranges correctly."""
        line = LineString([(0, 0), (100, 0)])
        matched = gpd.GeoDataFrame(
            {"local_id": ["seg1"], "geometry": [line]},
            crs="EPSG:32610",
        )

        # Two matches covering 0-40% and 60-100%, leaving 40-60% unmatched
        match_results = [
            {
                "local_id": "seg1",
                "local_start_frac": 0.0,
                "local_end_frac": 0.4,
                "match_decision": "match",
            },
            {
                "local_id": "seg1",
                "local_start_frac": 0.6,
                "local_end_frac": 1.0,
                "match_decision": "match",
            },
        ]

        remnants = extract_unmatched_remnants(matched, match_results, min_remnant_length_m=1.0)

        assert len(remnants) == 1
        # The gap remnant should be ~20m (40% to 60% of 100m)
        assert 15 < remnants.iloc[0].geometry.length < 25

    def test_empty_matched_returns_empty(self):
        """Empty matched GeoDataFrame returns empty result."""
        matched = gpd.GeoDataFrame(columns=["local_id", "geometry"], crs="EPSG:32610")
        remnants = extract_unmatched_remnants(matched, [], min_remnant_length_m=1.0)
        assert len(remnants) == 0

    def test_no_matching_ids_returns_empty(self):
        """If no match_results reference any matched segment, returns empty."""
        line = LineString([(0, 0), (100, 0)])
        matched = gpd.GeoDataFrame(
            {"local_id": ["seg1"], "geometry": [line]},
            crs="EPSG:32610",
        )

        match_results = [
            {
                "local_id": "seg_other",
                "local_start_frac": 0.3,
                "local_end_frac": 0.7,
                "match_decision": "match",
            },
        ]

        remnants = extract_unmatched_remnants(matched, match_results, min_remnant_length_m=1.0)
        assert len(remnants) == 0

    def test_attributes_carried_over(self):
        """Remnants carry over non-geometry, non-ID attributes from parent."""
        line = LineString([(0, 0), (100, 0)])
        matched = gpd.GeoDataFrame(
            {"local_id": ["seg1"], "name": ["Main St"], "geometry": [line]},
            crs="EPSG:32610",
        )

        match_results = [
            {
                "local_id": "seg1",
                "local_start_frac": 0.3,
                "local_end_frac": 0.7,
                "match_decision": "match",
            },
        ]

        remnants = extract_unmatched_remnants(matched, match_results, min_remnant_length_m=1.0)

        assert len(remnants) == 2
        assert all(remnants["name"] == "Main St")


class TestConnectivityGating:
    """Tests for connectivity gating (bridge promotion)."""

    def _make_bridge_scenario(self):
        """Create a scenario with two disconnected reference segments and a bridge target.

        Layout (projected CRS, meters):
            ref_1: (0,0) -> (100,0)     horizontal
            ref_2: (200,0) -> (300,0)   horizontal, gap of 100m

            bridge: (80,0) -> (220,0)   spans the gap, overlapping both refs
                start half (80,0)-(150,0) overlaps ref_1 by 20m
                end half (150,0)-(220,0) overlaps ref_2 by 20m
        """
        ref_1 = LineString([(0, 0), (100, 0)])
        ref_2 = LineString([(200, 0), (300, 0)])
        bridge = LineString([(80, 0), (220, 0)])

        combined = gpd.GeoDataFrame(
            {
                "_original_id": ["ref_1", "ref_2", "bridge_1"],
                "_source": [
                    EdgeSource.REFERENCE.value,
                    EdgeSource.REFERENCE.value,
                    EdgeSource.TARGET_UNMATCHED.value,
                ],
                "geometry": [ref_1, ref_2, bridge],
            },
            crs="EPSG:32610",
        )
        return combined

    def test_bridge_promoted(self):
        """A segment bridging two disconnected reference components is promoted."""
        combined = self._make_bridge_scenario()

        main, disconnected, filtered, net_new, stats = detect_orphans_by_proximity(
            combined,
            connection_tolerance_m=3.0,
            min_merge_length_m=200.0,  # High threshold so bridge gets filtered first
            net_new_buffer_m=5.0,
            max_hops=0,
            enable_connectivity_gating=True,
            min_bridge_overlap_m=10.0,
        )

        # Bridge should be in main (promoted)
        target_main = main[main["_source"] == EdgeSource.TARGET_UNMATCHED.value]
        assert len(target_main) == 1
        assert target_main.iloc[0]["_connectivity_role"] == "bridge"
        assert stats["bridge_promoted"] == 1
        assert len(filtered) == 0

    def test_spur_rejected(self):
        """A spur overlapping one reference at one end only is not promoted."""
        ref_1 = LineString([(0, 0), (100, 0)])
        ref_2 = LineString([(200, 0), (300, 0)])
        # Spur only overlaps ref_1, other end goes to empty space
        spur = LineString([(80, 0), (150, 50)])

        combined = gpd.GeoDataFrame(
            {
                "_original_id": ["ref_1", "ref_2", "spur_1"],
                "_source": [
                    EdgeSource.REFERENCE.value,
                    EdgeSource.REFERENCE.value,
                    EdgeSource.TARGET_UNMATCHED.value,
                ],
                "geometry": [ref_1, ref_2, spur],
            },
            crs="EPSG:32610",
        )

        main, disconnected, filtered, net_new, stats = detect_orphans_by_proximity(
            combined,
            connection_tolerance_m=3.0,
            min_merge_length_m=200.0,
            net_new_buffer_m=5.0,
            max_hops=0,
            enable_connectivity_gating=True,
            min_bridge_overlap_m=10.0,
        )

        # Spur should stay filtered (no overlap at far end)
        target_main = main[main["_source"] == EdgeSource.TARGET_UNMATCHED.value]
        assert len(target_main) == 0
        assert stats["bridge_promoted"] == 0

    def test_same_component_rejected(self):
        """A segment overlapping two reference segments in the same component is not promoted."""
        # Two connected reference segments (same component)
        ref_1 = LineString([(0, 0), (100, 0)])
        ref_2 = LineString([(100, 0), (200, 0)])  # Connected at (100, 0)
        # Target overlaps both refs
        target = LineString([(50, 2), (150, 2)])

        combined = gpd.GeoDataFrame(
            {
                "_original_id": ["ref_1", "ref_2", "t_1"],
                "_source": [
                    EdgeSource.REFERENCE.value,
                    EdgeSource.REFERENCE.value,
                    EdgeSource.TARGET_UNMATCHED.value,
                ],
                "geometry": [ref_1, ref_2, target],
            },
            crs="EPSG:32610",
        )

        main, disconnected, filtered, net_new, stats = detect_orphans_by_proximity(
            combined,
            connection_tolerance_m=3.0,
            min_merge_length_m=200.0,
            net_new_buffer_m=5.0,
            max_hops=0,
            enable_connectivity_gating=True,
            min_bridge_overlap_m=10.0,
        )

        # Should stay filtered — refs are in same component
        target_main = main[main["_source"] == EdgeSource.TARGET_UNMATCHED.value]
        assert len(target_main) == 0
        assert stats["bridge_promoted"] == 0

    def test_below_overlap_threshold(self):
        """A bridge with insufficient overlap at one end is not promoted."""
        ref_1 = LineString([(0, 0), (100, 0)])
        ref_2 = LineString([(200, 0), (300, 0)])
        # Only 5m overlap at start end (95-100 of ref_1), 20m at end
        short_bridge = LineString([(95, 0), (220, 0)])

        combined = gpd.GeoDataFrame(
            {
                "_original_id": ["ref_1", "ref_2", "bridge_1"],
                "_source": [
                    EdgeSource.REFERENCE.value,
                    EdgeSource.REFERENCE.value,
                    EdgeSource.TARGET_UNMATCHED.value,
                ],
                "geometry": [ref_1, ref_2, short_bridge],
            },
            crs="EPSG:32610",
        )

        main, disconnected, filtered, net_new, stats = detect_orphans_by_proximity(
            combined,
            connection_tolerance_m=3.0,
            min_merge_length_m=200.0,
            net_new_buffer_m=5.0,
            max_hops=0,
            enable_connectivity_gating=True,
            min_bridge_overlap_m=15.0,  # Higher threshold
        )

        # Start half is (95,0)-(157.5,0) — overlap with ref_1 is only 5m
        # Should not be promoted
        target_main = main[main["_source"] == EdgeSource.TARGET_UNMATCHED.value]
        assert len(target_main) == 0
        assert stats["bridge_promoted"] == 0

    def test_gating_disabled(self):
        """Bridge scenario with gating disabled stays filtered."""
        combined = self._make_bridge_scenario()

        main, disconnected, filtered, net_new, stats = detect_orphans_by_proximity(
            combined,
            connection_tolerance_m=3.0,
            min_merge_length_m=200.0,
            net_new_buffer_m=5.0,
            max_hops=0,
            enable_connectivity_gating=False,
            min_bridge_overlap_m=10.0,
        )

        # Bridge should stay filtered
        target_main = main[main["_source"] == EdgeSource.TARGET_UNMATCHED.value]
        assert len(target_main) == 0
        assert stats["bridge_promoted"] == 0
        assert len(filtered) == 1
