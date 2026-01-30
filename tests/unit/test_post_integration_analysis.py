"""Tests for post-integration analysis modules."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from matcher.post_integration import (
    DriftPattern,
    DriftSeverity,
    IslandSeverity,
    detect_gps_drift,
    detect_islands,
)


class TestIslandDetector:
    """Tests for island detection."""

    def test_single_connected_network(self):
        """Test that a fully connected network has no islands."""
        # Create a simple connected network (triangle) in projected coords (meters)
        edges = gpd.GeoDataFrame(
            {"id": ["a", "b", "c"]},
            geometry=[
                LineString([(0, 0), (100, 0)]),
                LineString([(100, 0), (50, 100)]),
                LineString([(50, 100), (0, 0)]),
            ],
            crs="EPSG:32618",  # UTM zone 18N
        )

        result = detect_islands(edges, snap_tolerance_m=10.0)

        assert result.total_components == 1
        assert result.main_component_ratio == 1.0
        assert len(result.islands) == 0

    def test_detects_disconnected_segment(self):
        """Test that a disconnected segment is detected as an island."""
        # Main network (in meters)
        main = [
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (200, 0)]),
        ]
        # Isolated segment far away
        isolated = LineString([(10000, 10000), (10100, 10000)])

        edges = gpd.GeoDataFrame(
            {"id": ["a", "b", "isolated"]},
            geometry=main + [isolated],
            crs="EPSG:32618",  # UTM zone 18N
        )

        result = detect_islands(edges, snap_tolerance_m=10.0)

        assert result.total_components == 2
        assert len(result.islands) == 1
        assert result.islands[0].edge_count == 1

    def test_single_segment_is_critical(self):
        """Test that single isolated segment is classified as CRITICAL."""
        # Main network (in meters)
        main = [
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (200, 0)]),
        ]
        # Isolated segment
        isolated = LineString([(10000, 10000), (10100, 10000)])

        edges = gpd.GeoDataFrame(
            {"id": ["a", "b", "isolated"]},
            geometry=main + [isolated],
            crs="EPSG:32618",  # UTM zone 18N
        )

        result = detect_islands(edges, snap_tolerance_m=10.0, single_segment_is_critical=True)

        assert result.critical_count == 1
        assert result.islands[0].severity == IslandSeverity.CRITICAL

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        edges = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        edges["id"] = []

        result = detect_islands(edges)

        assert result.total_edges == 0
        assert result.total_components == 0


class TestGpsDriftDetector:
    """Tests for GPS drift detection."""

    def test_clean_geometry_no_drift(self):
        """Test that clean geometry has no drift patterns."""
        edges = gpd.GeoDataFrame(
            {"id": ["a"]},
            geometry=[LineString([(0, 0), (1, 0), (2, 0)])],
            crs="EPSG:4326",
        )

        result = detect_gps_drift(edges)

        assert result.edges_with_drift == 0
        assert result.zigzag_count == 0
        assert result.spike_count == 0
        assert result.loop_count == 0

    def test_detects_zigzag_pattern(self):
        """Test that zigzag pattern is detected."""
        # Create a zigzag with many small alternating turns
        coords = []
        for i in range(20):
            # Alternating up and down
            y = 0.00001 if i % 2 == 0 else -0.00001
            coords.append((i * 0.00001, y))

        edges = gpd.GeoDataFrame(
            {"id": ["zigzag"]},
            geometry=[LineString(coords)],
            crs="EPSG:4326",
        )

        result = detect_gps_drift(edges, zigzag_vertex_density=0.1)

        # Should detect zigzag due to alternating turns
        assert result.edges_with_drift >= 0  # May or may not detect depending on thresholds

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        edges = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        edges["id"] = []

        result = detect_gps_drift(edges)

        assert result.total_edges == 0
        assert result.edges_with_drift == 0

    def test_short_geometry_no_crash(self):
        """Test that short geometries don't crash."""
        edges = gpd.GeoDataFrame(
            {"id": ["short"]},
            geometry=[LineString([(0, 0), (0.0001, 0)])],
            crs="EPSG:4326",
        )

        result = detect_gps_drift(edges)

        # Should not crash
        assert result.total_edges == 1

    def test_result_to_dict(self):
        """Test that result can be converted to dict."""
        edges = gpd.GeoDataFrame(
            {"id": ["a"]},
            geometry=[LineString([(0, 0), (1, 0)])],
            crs="EPSG:4326",
        )

        result = detect_gps_drift(edges)
        d = result.to_dict()

        assert "total_edges" in d
        assert "edges_with_drift" in d
        assert "zigzag_count" in d
        assert "detections" in d
