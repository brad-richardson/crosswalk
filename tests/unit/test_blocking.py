"""Tests for blocking and candidate generation."""

import geopandas as gpd
import pytest
from shapely import LineString

from matcher.blocking.spatial_index import (
    _angle_diff,
    _compute_overall_heading,
    generate_candidates,
)


class TestGenerateCandidates:
    """Tests for candidate generation."""

    def test_generates_candidates_for_nearby_lines(self):
        """Should generate candidates for nearby parallel lines."""
        # Reference: single line
        reference = gpd.GeoDataFrame(
            {
                "id": ["ref_1"],
                "geometry": [LineString([(0, 0), (100, 0)])],
            },
            crs="EPSG:32610",
        )

        # Target: parallel line 10m away
        target = gpd.GeoDataFrame(
            {
                "local_id": ["target_1"],
                "geometry": [LineString([(0, 10), (100, 10)])],
            },
            crs="EPSG:32610",
        )

        candidates = generate_candidates(
            reference, target, buffer_distance=50.0, ref_id_column="id"
        )

        assert len(candidates) == 1
        assert candidates[0].ref_id == "ref_1"
        assert candidates[0].target_id == "target_1"

    def test_no_candidates_for_distant_lines(self):
        """Should not generate candidates for distant lines."""
        reference = gpd.GeoDataFrame(
            {
                "id": ["ref_1"],
                "geometry": [LineString([(0, 0), (100, 0)])],
            },
            crs="EPSG:32610",
        )

        # Target: line 200m away
        target = gpd.GeoDataFrame(
            {
                "local_id": ["target_1"],
                "geometry": [LineString([(0, 200), (100, 200)])],
            },
            crs="EPSG:32610",
        )

        candidates = generate_candidates(
            reference, target, buffer_distance=50.0, ref_id_column="id"
        )

        assert len(candidates) == 0

    def test_computes_heading_difference(self):
        """Should compute heading difference as a feature (not filter)."""
        reference = gpd.GeoDataFrame(
            {
                "id": ["ref_1"],
                "geometry": [LineString([(0, 0), (100, 0)])],  # East-west
            },
            crs="EPSG:32610",
        )

        # Target: perpendicular line nearby
        target = gpd.GeoDataFrame(
            {
                "local_id": ["target_1"],
                "geometry": [LineString([(50, -50), (50, 50)])],  # North-south
            },
            crs="EPSG:32610",
        )

        candidates = generate_candidates(
            reference,
            target,
            buffer_distance=100.0,
            ref_id_column="id",
        )

        # Heading difference is computed as a feature, not used as a filter
        # The ML model uses this as a scoring feature
        assert len(candidates) == 1
        assert candidates[0].heading_diff == 90.0  # 90° difference computed

    def test_computes_length_ratio(self):
        """Should compute length ratio as a feature (not filter)."""
        reference = gpd.GeoDataFrame(
            {
                "id": ["ref_1"],
                "geometry": [LineString([(0, 0), (100, 0)])],  # Length 100
            },
            crs="EPSG:32610",
        )

        # Target: much shorter line nearby
        target = gpd.GeoDataFrame(
            {
                "local_id": ["target_1"],
                "geometry": [LineString([(0, 10), (10, 10)])],  # Length 10
            },
            crs="EPSG:32610",
        )

        candidates = generate_candidates(
            reference,
            target,
            buffer_distance=50.0,
            ref_id_column="id",
        )

        # Length ratio is computed as a feature, not used as a filter
        # The ML model uses this as a scoring feature
        assert len(candidates) == 1
        assert candidates[0].length_ratio == 0.1  # 10/100 = 0.1

    def test_multiple_candidates_per_target(self):
        """Should generate multiple candidates when target matches multiple references."""
        # Two parallel reference lines
        reference = gpd.GeoDataFrame(
            {
                "id": ["ref_1", "ref_2"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(0, 20), (100, 20)]),
                ],
            },
            crs="EPSG:32610",
        )

        # Target: line between them
        target = gpd.GeoDataFrame(
            {
                "local_id": ["target_1"],
                "geometry": [LineString([(0, 10), (100, 10)])],
            },
            crs="EPSG:32610",
        )

        candidates = generate_candidates(
            reference, target, buffer_distance=50.0, ref_id_column="id"
        )

        # Should match both reference lines
        assert len(candidates) == 2


class TestHeadingCalculations:
    """Tests for heading calculation utilities."""

    def test_compute_overall_heading_east(self):
        """East-pointing line should have ~0 degree heading."""
        line = LineString([(0, 0), (100, 0)])
        heading = _compute_overall_heading(line)

        assert heading == pytest.approx(0.0, abs=1.0)

    def test_compute_overall_heading_north(self):
        """North-pointing line should have ~90 degree heading."""
        line = LineString([(0, 0), (0, 100)])
        heading = _compute_overall_heading(line)

        assert heading == pytest.approx(90.0, abs=1.0)

    def test_angle_diff_same(self):
        """Same heading should have 0 difference."""
        assert _angle_diff(45.0, 45.0) == pytest.approx(0.0)

    def test_angle_diff_opposite(self):
        """Opposite headings should have 0 difference (bidirectional roads)."""
        assert _angle_diff(0.0, 180.0) == pytest.approx(0.0)

    def test_angle_diff_perpendicular(self):
        """Perpendicular headings should have 90 degree difference."""
        assert _angle_diff(0.0, 90.0) == pytest.approx(90.0)

    def test_angle_diff_wraparound(self):
        """Should handle wraparound correctly."""
        assert _angle_diff(350.0, 10.0) == pytest.approx(20.0)
