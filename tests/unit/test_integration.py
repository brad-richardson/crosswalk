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
    filter_fringe_segments,
    filter_short_segments,
)
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
            orphan_edges=5,
            orphan_components=2,
            datasets_integrated=["boston_streets"],
        )

        d = stats.to_dict()

        assert d["reference_edges"] == 100
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

        main, orphans, net_new, stats = detect_orphans_by_proximity(
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

        main, orphans, net_new, stats = detect_orphans_by_proximity(
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

        # With max_hops=1, t3 should be orphan
        main, orphans, net_new, stats = detect_orphans_by_proximity(
            combined, connection_tolerance_m=3.0, min_merge_length_m=0, max_hops=1
        )

        connected = main[main["_source"] == EdgeSource.TARGET_UNMATCHED.value]
        assert len(connected) == 2  # t1 and t2

        orphan_targets = orphans[orphans["_source"] == EdgeSource.TARGET_UNMATCHED.value]
        assert len(orphan_targets) == 1  # t3


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

        # Test inside segment - should be valid (inside_length >= 5m)
        valid, fringe = filter_fringe_segments(
            inside, reference, buffer_distance_m=30.0, min_inside_length_m=5.0
        )
        assert len(valid) == 1
        assert len(fringe) == 0

        # Test outside segment - should be fringe
        valid, fringe = filter_fringe_segments(
            outside, reference, buffer_distance_m=30.0, min_inside_length_m=5.0
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
