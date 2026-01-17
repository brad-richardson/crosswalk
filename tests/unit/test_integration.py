"""Unit tests for integration module.

Fixtures used (from conftest.py):
- reference_gdf: Reference network with two connected segments
- target_matched_gdf: Target segments that match reference
- target_unmatched_gdf: Target segments without matches
- match_results: Mock match results for target_matched
"""

import geopandas as gpd
from shapely import LineString

from matcher.integration import (
    EdgeSource,
    IntegrationStatistics,
    TargetInput,
    combine_networks,
    detect_orphan_components,
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

        kept, filtered = filter_short_segments(gdf, min_length=5.0)

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
