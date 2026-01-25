"""Tests for CRS utility functions."""

import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import LineString

from matcher.utils.crs import (
    ProjectionResult,
    ensure_projected_crs,
    ensure_single_projected_crs,
    validate_projected_crs,
)


class TestValidateProjectedCrs:
    """Tests for validate_projected_crs()."""

    def test_raises_on_geographic_crs(self):
        """Geographic CRS should raise ValueError."""
        gdf = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 1)])]},
            crs="EPSG:4326",
        )
        with pytest.raises(ValueError, match="geographic CRS"):
            validate_projected_crs(gdf, "test")

    def test_raises_on_no_crs(self):
        """No CRS should raise ValueError."""
        gdf = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 1)])]},
        )
        with pytest.raises(ValueError, match="no CRS set"):
            validate_projected_crs(gdf, "test")

    def test_accepts_projected_crs(self):
        """Projected CRS should not raise."""
        gdf = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1000, 1000)])]},
            crs="EPSG:32618",  # UTM zone 18N
        )
        # Should not raise
        validate_projected_crs(gdf, "test")

    def test_error_message_includes_name(self):
        """Error message should include the provided name."""
        gdf = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 1)])]},
            crs="EPSG:4326",
        )
        with pytest.raises(ValueError, match="reference"):
            validate_projected_crs(gdf, "reference")


class TestEnsureProjectedCrs:
    """Tests for ensure_projected_crs()."""

    def test_projects_geographic_to_utm(self):
        """Geographic CRS should be projected to UTM."""
        # Boston area coordinates
        reference = gpd.GeoDataFrame(
            {"id": [1], "geometry": [LineString([(-71.06, 42.36), (-71.05, 42.36)])]},
            crs="EPSG:4326",
        )
        target = gpd.GeoDataFrame(
            {"id": [2], "geometry": [LineString([(-71.06, 42.36), (-71.05, 42.37)])]},
            crs="EPSG:4326",
        )

        result = ensure_projected_crs(reference, target)

        assert isinstance(result, ProjectionResult)
        assert result.was_reprojected is True
        assert result.original_crs == CRS.from_epsg(4326)
        assert result.projected_crs is not None
        assert not result.reference.crs.is_geographic
        assert not result.target.crs.is_geographic
        assert result.reference.crs == result.target.crs

    def test_already_projected_returns_unchanged(self):
        """Already projected data should return as-is."""
        reference = gpd.GeoDataFrame(
            {"id": [1], "geometry": [LineString([(0, 0), (1000, 0)])]},
            crs="EPSG:32618",
        )
        target = gpd.GeoDataFrame(
            {"id": [2], "geometry": [LineString([(0, 0), (0, 1000)])]},
            crs="EPSG:32618",
        )

        result = ensure_projected_crs(reference, target)

        assert result.was_reprojected is False
        assert result.original_crs is None
        assert result.projected_crs == CRS.from_epsg(32618)

    def test_aligns_target_crs_to_reference(self):
        """Target with different CRS should be aligned to reference."""
        reference = gpd.GeoDataFrame(
            {"id": [1], "geometry": [LineString([(-71.06, 42.36), (-71.05, 42.36)])]},
            crs="EPSG:4326",
        )
        # Target in a different CRS
        target = gpd.GeoDataFrame(
            {"id": [2], "geometry": [LineString([(300000, 4700000), (300100, 4700000)])]},
            crs="EPSG:32619",  # UTM 19N
        )

        result = ensure_projected_crs(reference, target)

        # Both should now be in the same projected CRS
        assert result.reference.crs == result.target.crs
        assert result.was_reprojected is True

    def test_preserves_data_columns(self):
        """Data columns should be preserved after projection."""
        reference = gpd.GeoDataFrame(
            {
                "id": [1, 2],
                "name": ["Main St", "Oak Ave"],
                "geometry": [
                    LineString([(-71.06, 42.36), (-71.05, 42.36)]),
                    LineString([(-71.07, 42.37), (-71.06, 42.37)]),
                ],
            },
            crs="EPSG:4326",
        )
        target = gpd.GeoDataFrame(
            {
                "id": [3],
                "name": ["Elm St"],
                "geometry": [LineString([(-71.06, 42.36), (-71.05, 42.37)])],
            },
            crs="EPSG:4326",
        )

        result = ensure_projected_crs(reference, target)

        assert list(result.reference["id"]) == [1, 2]
        assert list(result.reference["name"]) == ["Main St", "Oak Ave"]
        assert list(result.target["id"]) == [3]
        assert list(result.target["name"]) == ["Elm St"]


class TestEnsureSingleProjectedCrs:
    """Tests for ensure_single_projected_crs()."""

    def test_projects_geographic_crs(self):
        """Geographic CRS should be projected."""
        gdf = gpd.GeoDataFrame(
            {"geometry": [LineString([(-71.06, 42.36), (-71.05, 42.36)])]},
            crs="EPSG:4326",
        )

        projected, original_crs = ensure_single_projected_crs(gdf, "test")

        assert not projected.crs.is_geographic
        assert original_crs == CRS.from_epsg(4326)

    def test_already_projected_returns_none_original(self):
        """Already projected data should return None for original_crs."""
        gdf = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1000, 0)])]},
            crs="EPSG:32618",
        )

        projected, original_crs = ensure_single_projected_crs(gdf, "test")

        assert projected.crs == CRS.from_epsg(32618)
        assert original_crs is None


class TestProjectionResultUsage:
    """Tests for using ProjectionResult in typical workflows."""

    def test_restore_original_crs(self):
        """Original CRS can be restored after computation."""
        reference = gpd.GeoDataFrame(
            {"id": [1], "geometry": [LineString([(-71.06, 42.36), (-71.05, 42.36)])]},
            crs="EPSG:4326",
        )
        target = gpd.GeoDataFrame(
            {"id": [2], "geometry": [LineString([(-71.06, 42.36), (-71.05, 42.37)])]},
            crs="EPSG:4326",
        )

        result = ensure_projected_crs(reference, target)

        # Simulate some computation...
        output = result.reference.copy()

        # Restore to original CRS
        if result.original_crs:
            output = output.to_crs(result.original_crs)

        assert output.crs == CRS.from_epsg(4326)

    def test_distance_computation_accuracy(self):
        """Distances should be accurate in meters after projection."""
        # Two points ~111m apart (1 degree longitude at equator ~111km)
        # At 42.36N latitude, 0.001 degrees longitude ~82m
        reference = gpd.GeoDataFrame(
            {"id": [1], "geometry": [LineString([(-71.060, 42.36), (-71.059, 42.36)])]},
            crs="EPSG:4326",
        )
        target = gpd.GeoDataFrame(
            {"id": [2], "geometry": [LineString([(-71.060, 42.36), (-71.059, 42.36)])]},
            crs="EPSG:4326",
        )

        result = ensure_projected_crs(reference, target)

        # Length should be ~82 meters (not 0.001 degrees)
        length = result.reference.geometry.iloc[0].length
        assert 70 < length < 100  # Should be ~82m, not ~0.001
