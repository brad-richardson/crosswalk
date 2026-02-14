"""Tests for helper functions in data_loader.py.

Tests _resolve_lr_name(), _extract_pair_attributes(), and _build_aligned_geometries().
"""

import pytest
from pyproj import Transformer
from shapely.geometry import LineString

from matcher.labeling.data_loader import (
    _build_aligned_geometries,
    _extract_pair_attributes,
    _resolve_lr_name,
)


class TestResolveLrName:
    """Tests for _resolve_lr_name() which resolves display names using LR data."""

    def test_fallback_to_primary_name_when_no_lr_data(self):
        """When names_lr is None, should fall back to the raw name."""
        result = _resolve_lr_name(
            names_lr_data=None,
            name_raw={"primary": "Main Street"},
            start_frac=0.0,
            end_frac=1.0,
        )
        assert result == "Main Street"

    def test_fallback_to_string_name_when_no_lr_data(self):
        """When names_lr is None and name_raw is a plain string, return it."""
        result = _resolve_lr_name(
            names_lr_data=None,
            name_raw="Main Street",
            start_frac=0.0,
            end_frac=1.0,
        )
        assert result == "Main Street"

    def test_returns_none_when_no_name_data(self):
        """When both names_lr and name_raw are None, return None."""
        result = _resolve_lr_name(
            names_lr_data=None,
            name_raw=None,
            start_frac=0.0,
            end_frac=1.0,
        )
        assert result is None

    def test_resolves_lr_name_for_first_segment(self):
        """Should resolve the LR name covering the first half of the segment."""
        lr_data = [
            {"between": [0.0, 0.5], "value": "First Avenue"},
            {"between": [0.5, 1.0], "value": "Second Street"},
        ]
        result = _resolve_lr_name(
            names_lr_data=lr_data,
            name_raw={"primary": "Second Street"},
            start_frac=0.0,
            end_frac=0.4,
        )
        assert result == "First Avenue"

    def test_resolves_lr_name_for_second_segment(self):
        """Should resolve the LR name covering the second half of the segment."""
        lr_data = [
            {"between": [0.0, 0.5], "value": "First Avenue"},
            {"between": [0.5, 1.0], "value": "Second Street"},
        ]
        result = _resolve_lr_name(
            names_lr_data=lr_data,
            name_raw={"primary": "Second Street"},
            start_frac=0.6,
            end_frac=1.0,
        )
        assert result == "Second Street"

    def test_resolves_majority_name_across_boundary(self):
        """When aligned portion spans both names, should return the majority one."""
        lr_data = [
            {"between": [0.0, 0.5], "value": "First Avenue"},
            {"between": [0.5, 1.0], "value": "Second Street"},
        ]
        # 0.3 to 0.8 spans both, but more is in the second half (0.5-0.8 = 0.3 vs 0.3-0.5 = 0.2)
        result = _resolve_lr_name(
            names_lr_data=lr_data,
            name_raw={"primary": "Second Street"},
            start_frac=0.3,
            end_frac=0.8,
        )
        assert result == "Second Street"

    def test_resolves_uniform_lr_name(self):
        """When LR data has a single name covering the full segment, return it."""
        lr_data = [{"between": [0.0, 1.0], "value": "Main Street"}]
        result = _resolve_lr_name(
            names_lr_data=lr_data,
            name_raw={"primary": "Main Street"},
            start_frac=0.0,
            end_frac=1.0,
        )
        assert result == "Main Street"

    def test_fallback_on_malformed_lr_data(self):
        """Should fall back to raw name when LR data is malformed."""
        result = _resolve_lr_name(
            names_lr_data=[{"invalid": "data"}],
            name_raw={"primary": "Fallback Street"},
            start_frac=0.0,
            end_frac=1.0,
        )
        assert result == "Fallback Street"

    def test_fallback_on_empty_lr_data(self):
        """Should fall back to raw name when LR data is an empty list."""
        result = _resolve_lr_name(
            names_lr_data=[],
            name_raw={"primary": "Fallback Street"},
            start_frac=0.0,
            end_frac=1.0,
        )
        assert result == "Fallback Street"

    def test_lr_data_with_none_values(self):
        """LR data where all values are None should fall back to raw name."""
        lr_data = [{"between": [0.0, 1.0], "value": None}]
        result = _resolve_lr_name(
            names_lr_data=lr_data,
            name_raw={"primary": "Fallback Street"},
            start_frac=0.0,
            end_frac=1.0,
        )
        # extract_aligned_attributes returns None for the name, so fallback kicks in
        assert result == "Fallback Street"


class TestExtractPairAttributes:
    """Tests for _extract_pair_attributes() which extracts names/classes from pair data."""

    @pytest.fixture
    def ref_data(self):
        """Reference segment data dict."""
        return {
            "names": {"primary": "Main Street"},
            "class": "primary",
            "subclass": "highway",
            "names_lr": [{"between": [0.0, 1.0], "value": "Main Street"}],
        }

    @pytest.fixture
    def target_data(self):
        """Target segment data dict."""
        return {
            "names": {"primary": "Main St"},
            "class": "residential",
            "subclass": "local",
        }

    def test_extracts_all_attributes(self, ref_data, target_data):
        """Should extract all six attributes when all columns are present."""
        ref_name, target_name, ref_class, target_class, ref_subclass, target_subclass = (
            _extract_pair_attributes(
                ref_data=ref_data,
                target_data=target_data,
                ref_name_column="names",
                target_name_column="names",
                ref_class_column="class",
                target_class_column="class",
                ref_start_frac=0.0,
                ref_end_frac=1.0,
                target_start_frac=0.0,
                target_end_frac=1.0,
                has_ref_name=True,
                has_target_name=True,
                has_ref_class=True,
                has_target_class=True,
                has_ref_subclass=True,
                has_target_subclass=True,
                has_ref_names_lr=True,
                has_target_names_lr=False,
            )
        )
        assert ref_name == "Main Street"
        assert target_name == "Main St"
        assert ref_class == "primary"
        assert target_class == "residential"
        assert ref_subclass == "highway"
        assert target_subclass == "local"

    def test_missing_name_columns(self, ref_data, target_data):
        """When name columns are absent, names should be None."""
        ref_name, target_name, _, _, _, _ = _extract_pair_attributes(
            ref_data=ref_data,
            target_data=target_data,
            ref_name_column="names",
            target_name_column="names",
            ref_class_column="class",
            target_class_column="class",
            ref_start_frac=0.0,
            ref_end_frac=1.0,
            target_start_frac=0.0,
            target_end_frac=1.0,
            has_ref_name=False,
            has_target_name=False,
            has_ref_class=True,
            has_target_class=True,
            has_ref_subclass=False,
            has_target_subclass=False,
            has_ref_names_lr=False,
            has_target_names_lr=False,
        )
        assert ref_name is None
        assert target_name is None

    def test_missing_class_and_subclass(self, ref_data, target_data):
        """When class/subclass columns are absent, they should be None."""
        _, _, ref_class, target_class, ref_subclass, target_subclass = _extract_pair_attributes(
            ref_data=ref_data,
            target_data=target_data,
            ref_name_column="names",
            target_name_column="names",
            ref_class_column="class",
            target_class_column="class",
            ref_start_frac=0.0,
            ref_end_frac=1.0,
            target_start_frac=0.0,
            target_end_frac=1.0,
            has_ref_name=True,
            has_target_name=True,
            has_ref_class=False,
            has_target_class=False,
            has_ref_subclass=False,
            has_target_subclass=False,
            has_ref_names_lr=False,
            has_target_names_lr=False,
        )
        assert ref_class is None
        assert target_class is None
        assert ref_subclass is None
        assert target_subclass is None

    def test_lr_name_resolution_for_partial_alignment(self):
        """LR names should resolve based on alignment fractions, not primary name."""
        ref_data = {
            "names": {"primary": "Second Street"},
            "names_lr": [
                {"between": [0.0, 0.5], "value": "First Avenue"},
                {"between": [0.5, 1.0], "value": "Second Street"},
            ],
            "class": "primary",
        }
        target_data = {
            "names": {"primary": "FIRST AVENUE"},
            "class": "residential",
        }
        ref_name, target_name, _, _, _, _ = _extract_pair_attributes(
            ref_data=ref_data,
            target_data=target_data,
            ref_name_column="names",
            target_name_column="names",
            ref_class_column="class",
            target_class_column="class",
            ref_start_frac=0.0,
            ref_end_frac=0.4,
            target_start_frac=0.0,
            target_end_frac=1.0,
            has_ref_name=True,
            has_target_name=True,
            has_ref_class=True,
            has_target_class=True,
            has_ref_subclass=False,
            has_target_subclass=False,
            has_ref_names_lr=True,
            has_target_names_lr=False,
        )
        # The ref aligned portion (0-40%) falls entirely within "First Avenue" (0-50%)
        assert ref_name == "First Avenue"
        assert target_name == "FIRST AVENUE"

    def test_works_with_pandas_series(self):
        """Should work with pandas Series objects (from .iloc[0])."""
        import pandas as pd

        ref_series = pd.Series(
            {"names": {"primary": "Oak Ave"}, "class": "secondary", "subclass": "local"}
        )
        target_series = pd.Series({"names": {"primary": "Oak Avenue"}, "class": "tertiary"})
        ref_name, target_name, ref_class, target_class, _, _ = _extract_pair_attributes(
            ref_data=ref_series,
            target_data=target_series,
            ref_name_column="names",
            target_name_column="names",
            ref_class_column="class",
            target_class_column="class",
            ref_start_frac=0.0,
            ref_end_frac=1.0,
            target_start_frac=0.0,
            target_end_frac=1.0,
            has_ref_name=True,
            has_target_name=True,
            has_ref_class=True,
            has_target_class=True,
            has_ref_subclass=True,
            has_target_subclass=False,
            has_ref_names_lr=False,
            has_target_names_lr=False,
        )
        assert ref_name == "Oak Ave"
        assert target_name == "Oak Avenue"
        assert ref_class == "secondary"
        assert target_class == "tertiary"


class TestBuildAlignedGeometries:
    """Tests for _build_aligned_geometries() which creates aligned sublines."""

    @pytest.fixture
    def ref_proj_geom(self):
        """Reference geometry in projected CRS."""
        return LineString([(0, 0), (100, 0)])

    @pytest.fixture
    def target_proj_geom(self):
        """Target geometry in projected CRS."""
        return LineString([(0, 10), (100, 10)])

    def test_full_segment_no_transform(self, ref_proj_geom, target_proj_geom):
        """Full alignment (0-1) with no CRS transform should return original geoms."""
        ref_aligned, target_aligned = _build_aligned_geometries(
            ref_proj_geom=ref_proj_geom,
            target_proj_geom=target_proj_geom,
            ref_start_frac=0.0,
            ref_end_frac=1.0,
            target_start_frac=0.0,
            target_end_frac=1.0,
            proj_to_wgs84=None,
        )
        # Full segment — should match the input geometries
        assert ref_aligned.length == pytest.approx(ref_proj_geom.length, abs=0.1)
        assert target_aligned.length == pytest.approx(target_proj_geom.length, abs=0.1)

    def test_partial_alignment(self, ref_proj_geom, target_proj_geom):
        """Partial alignment should return shorter sublines."""
        ref_aligned, target_aligned = _build_aligned_geometries(
            ref_proj_geom=ref_proj_geom,
            target_proj_geom=target_proj_geom,
            ref_start_frac=0.0,
            ref_end_frac=0.5,
            target_start_frac=0.25,
            target_end_frac=0.75,
            proj_to_wgs84=None,
        )
        # Ref: 50% of 100m = 50m
        assert ref_aligned.length == pytest.approx(50.0, abs=1.0)
        # Target: 50% of 100m = 50m
        assert target_aligned.length == pytest.approx(50.0, abs=1.0)

    def test_with_crs_transform(self, ref_proj_geom, target_proj_geom):
        """CRS transform should be applied when proj_to_wgs84 is provided."""
        # Create a real transformer: UTM zone 10N -> WGS84
        proj_to_wgs84 = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True).transform
        ref_aligned, target_aligned = _build_aligned_geometries(
            ref_proj_geom=ref_proj_geom,
            target_proj_geom=target_proj_geom,
            ref_start_frac=0.0,
            ref_end_frac=1.0,
            target_start_frac=0.0,
            target_end_frac=1.0,
            proj_to_wgs84=proj_to_wgs84,
        )
        # After transform to WGS84, coordinates should be in degrees (small values)
        ref_coords = list(ref_aligned.coords)
        assert all(abs(x) < 180 and abs(y) < 90 for x, y in ref_coords)
        target_coords = list(target_aligned.coords)
        assert all(abs(x) < 180 and abs(y) < 90 for x, y in target_coords)

    def test_returns_none_for_degenerate_subline(self):
        """When start and end fracs are equal, create_subline returns None."""
        geom = LineString([(0, 0), (100, 0)])
        ref_aligned, target_aligned = _build_aligned_geometries(
            ref_proj_geom=geom,
            target_proj_geom=geom,
            ref_start_frac=0.5,
            ref_end_frac=0.5,
            target_start_frac=0.0,
            target_end_frac=1.0,
            proj_to_wgs84=None,
        )
        # create_subline with equal fracs produces a zero-length line or None
        # The function should handle this gracefully
        assert ref_aligned is None or ref_aligned.length == pytest.approx(0.0, abs=0.1)
