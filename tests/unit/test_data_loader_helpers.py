"""Tests for helper functions in data_loader.py.

Tests _resolve_lr_name(), _extract_pair_attributes(), and _build_aligned_geometries().
"""

import pandas as pd
import pytest
from pyproj import Transformer
from shapely.geometry import LineString

from matcher.labeling.data_loader import (
    _build_aligned_geometries,
    _extract_pair_attributes,
    _resolve_lr_name,
)

# Reusable LR data: "First Avenue" 0-50%, "Second Street" 50-100%
SPLIT_LR = [
    {"between": [0.0, 0.5], "value": "First Avenue"},
    {"between": [0.5, 1.0], "value": "Second Street"},
]


class TestResolveLrName:
    """Tests for _resolve_lr_name()."""

    @pytest.mark.parametrize(
        "lr_data, name_raw, start, end, expected",
        [
            # Fallback cases: no LR data
            (None, {"primary": "Main Street"}, 0.0, 1.0, "Main Street"),
            (None, "Main Street", 0.0, 1.0, "Main Street"),
            (None, None, 0.0, 1.0, None),
            # LR resolution: aligned portion selects correct name
            (SPLIT_LR, {"primary": "Second Street"}, 0.0, 0.4, "First Avenue"),
            (SPLIT_LR, {"primary": "Second Street"}, 0.6, 1.0, "Second Street"),
            # Majority wins when spanning boundary (0.5-0.8 > 0.3-0.5)
            (SPLIT_LR, {"primary": "Second Street"}, 0.3, 0.8, "Second Street"),
            # Uniform LR
            ([{"between": [0.0, 1.0], "value": "Main Street"}], None, 0.0, 1.0, "Main Street"),
        ],
        ids=[
            "no_lr_dict_name",
            "no_lr_string_name",
            "no_lr_no_name",
            "lr_first_half",
            "lr_second_half",
            "lr_majority_wins",
            "lr_uniform",
        ],
    )
    def test_name_resolution(self, lr_data, name_raw, start, end, expected):
        assert _resolve_lr_name(lr_data, name_raw, start, end) == expected

    @pytest.mark.parametrize(
        "lr_data, name_raw, expected",
        [
            ([{"invalid": "data"}], {"primary": "Fallback"}, "Fallback"),
            ([], {"primary": "Fallback"}, "Fallback"),
            ([{"between": [0.0, 1.0], "value": None}], {"primary": "Fallback"}, "Fallback"),
        ],
        ids=["malformed", "empty_list", "none_value"],
    )
    def test_fallback_on_bad_lr_data(self, lr_data, name_raw, expected):
        """Should fall back to raw name when LR data is unusable."""
        assert _resolve_lr_name(lr_data, name_raw, 0.0, 1.0) == expected


class TestExtractPairAttributes:
    """Tests for _extract_pair_attributes()."""

    def _call(
        self,
        ref_data,
        target_data,
        *,
        has_ref_name=True,
        has_target_name=True,
        has_ref_class=True,
        has_target_class=True,
        has_ref_subclass=False,
        has_target_subclass=False,
        has_ref_names_lr=False,
        has_target_names_lr=False,
        ref_start=0.0,
        ref_end=1.0,
        target_start=0.0,
        target_end=1.0,
    ):
        """Thin wrapper to reduce boilerplate in tests."""
        return _extract_pair_attributes(
            ref_data=ref_data,
            target_data=target_data,
            ref_name_column="names",
            target_name_column="names",
            ref_class_column="class",
            target_class_column="class",
            ref_start_frac=ref_start,
            ref_end_frac=ref_end,
            target_start_frac=target_start,
            target_end_frac=target_end,
            has_ref_name=has_ref_name,
            has_target_name=has_target_name,
            has_ref_class=has_ref_class,
            has_target_class=has_target_class,
            has_ref_subclass=has_ref_subclass,
            has_target_subclass=has_target_subclass,
            has_ref_names_lr=has_ref_names_lr,
            has_target_names_lr=has_target_names_lr,
        )

    def test_extracts_all_attributes(self):
        """All six attributes extracted when all columns present."""
        ref = {
            "names": {"primary": "Main Street"},
            "class": "primary",
            "subclass": "highway",
            "names_lr": [{"between": [0.0, 1.0], "value": "Main Street"}],
        }
        target = {"names": {"primary": "Main St"}, "class": "residential", "subclass": "local"}
        result = self._call(
            ref, target, has_ref_subclass=True, has_target_subclass=True, has_ref_names_lr=True
        )
        assert result == ("Main Street", "Main St", "primary", "residential", "highway", "local")

    @pytest.mark.parametrize(
        "has_ref_name, has_target_name, has_ref_class, has_target_class",
        [
            (False, False, True, True),
            (True, True, False, False),
        ],
        ids=["missing_names", "missing_classes"],
    )
    def test_missing_columns_return_none(
        self, has_ref_name, has_target_name, has_ref_class, has_target_class
    ):
        """Missing columns should produce None for those attributes."""
        ref = {"names": {"primary": "A"}, "class": "primary"}
        target = {"names": {"primary": "B"}, "class": "secondary"}
        result = self._call(
            ref,
            target,
            has_ref_name=has_ref_name,
            has_target_name=has_target_name,
            has_ref_class=has_ref_class,
            has_target_class=has_target_class,
        )
        ref_name, target_name, ref_class, target_class, _, _ = result
        if not has_ref_name:
            assert ref_name is None and target_name is None
        if not has_ref_class:
            assert ref_class is None and target_class is None

    def test_lr_overrides_primary_name_for_partial_alignment(self):
        """LR name for the aligned portion should override the primary name (original bug)."""
        ref = {"names": {"primary": "Second Street"}, "names_lr": SPLIT_LR, "class": "primary"}
        target = {"names": {"primary": "FIRST AVENUE"}, "class": "residential"}
        ref_name, target_name, *_ = self._call(
            ref,
            target,
            has_ref_names_lr=True,
            ref_start=0.0,
            ref_end=0.4,
        )
        assert ref_name == "First Avenue"  # Not "Second Street"
        assert target_name == "FIRST AVENUE"

    def test_works_with_pandas_series(self):
        """Should accept both dicts and pandas Series (from .iloc[0])."""
        ref = pd.Series({"names": {"primary": "Oak Ave"}, "class": "secondary"})
        target = pd.Series({"names": {"primary": "Oak Avenue"}, "class": "tertiary"})
        ref_name, target_name, ref_class, target_class, _, _ = self._call(ref, target)
        assert (ref_name, target_name) == ("Oak Ave", "Oak Avenue")
        assert (ref_class, target_class) == ("secondary", "tertiary")


class TestBuildAlignedGeometries:
    """Tests for _build_aligned_geometries()."""

    REF_GEOM = LineString([(0, 0), (100, 0)])
    TARGET_GEOM = LineString([(0, 10), (100, 10)])

    @pytest.mark.parametrize(
        "ref_start, ref_end, target_start, target_end, expected_ref_len, expected_target_len",
        [
            (0.0, 1.0, 0.0, 1.0, 100.0, 100.0),
            (0.0, 0.5, 0.25, 0.75, 50.0, 50.0),
        ],
        ids=["full_segment", "partial_alignment"],
    )
    def test_subline_lengths(
        self, ref_start, ref_end, target_start, target_end, expected_ref_len, expected_target_len
    ):
        ref_aligned, target_aligned = _build_aligned_geometries(
            self.REF_GEOM,
            self.TARGET_GEOM,
            ref_start,
            ref_end,
            target_start,
            target_end,
            proj_to_wgs84=None,
        )
        assert ref_aligned.length == pytest.approx(expected_ref_len, abs=1.0)
        assert target_aligned.length == pytest.approx(expected_target_len, abs=1.0)

    def test_crs_transform_produces_wgs84_coords(self):
        """Transformed geometries should have coordinates in degree range."""
        proj_to_wgs84 = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True).transform
        ref_aligned, target_aligned = _build_aligned_geometries(
            self.REF_GEOM,
            self.TARGET_GEOM,
            0.0,
            1.0,
            0.0,
            1.0,
            proj_to_wgs84=proj_to_wgs84,
        )
        for geom in (ref_aligned, target_aligned):
            assert all(abs(x) < 180 and abs(y) < 90 for x, y in geom.coords)

    def test_degenerate_subline(self):
        """Equal start/end fracs should produce None or zero-length line."""
        ref_aligned, _ = _build_aligned_geometries(
            self.REF_GEOM,
            self.TARGET_GEOM,
            0.5,
            0.5,
            0.0,
            1.0,
            proj_to_wgs84=None,
        )
        assert ref_aligned is None or ref_aligned.length == pytest.approx(0.0, abs=0.1)
