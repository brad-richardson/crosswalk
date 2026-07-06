"""Tests for helper functions in data_loader.py.

Tests extract_pair_attributes() and _build_aligned_geometries().
LR name extraction tests live in test_linear_ref.py (TestExtractLrName).
"""

import pandas as pd
import pytest
from pyproj import Transformer
from shapely.geometry import LineString

from crosswalk.labeling.data_loader import (
    _build_aligned_geometries,
    extract_pair_attributes,
)

# Reusable LR data: "First Avenue" 0-50%, "Second Street" 50-100%
SPLIT_LR = [
    {"between": [0.0, 0.5], "value": "First Avenue"},
    {"between": [0.5, 1.0], "value": "Second Street"},
]


class TestExtractPairAttributes:
    """Tests for extract_pair_attributes()."""

    def _call(
        self,
        ref_data,
        target_data,
        *,
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
        return extract_pair_attributes(
            ref_data=ref_data,
            target_data=target_data,
            ref_class_column="class",
            target_class_column="class",
            ref_start_frac=ref_start,
            ref_end_frac=ref_end,
            target_start_frac=target_start,
            target_end_frac=target_end,
            has_ref_names_lr=has_ref_names_lr,
            has_target_names_lr=has_target_names_lr,
            has_ref_class=has_ref_class,
            has_target_class=has_target_class,
            has_ref_subclass=has_ref_subclass,
            has_target_subclass=has_target_subclass,
        )

    def test_extracts_all_attributes(self):
        """All six attributes extracted when all columns present."""
        ref = {
            "class": "primary",
            "subclass": "highway",
            "names_lr": [{"between": [0.0, 1.0], "value": "Main Street"}],
        }
        target = {
            "class": "residential",
            "subclass": "local",
            "names_lr": [{"between": [0.0, 1.0], "value": "Main St"}],
        }
        result = self._call(
            ref,
            target,
            has_ref_subclass=True,
            has_target_subclass=True,
            has_ref_names_lr=True,
            has_target_names_lr=True,
        )
        assert result == ("Main Street", "Main St", "primary", "residential", "highway", "local")

    @pytest.mark.parametrize(
        "has_ref_names_lr, has_target_names_lr, has_ref_class, has_target_class",
        [
            (False, False, True, True),
            (True, True, False, False),
        ],
        ids=["missing_names_lr", "missing_classes"],
    )
    def test_missing_columns_return_none(
        self, has_ref_names_lr, has_target_names_lr, has_ref_class, has_target_class
    ):
        """Missing columns should produce None for those attributes."""
        ref = {
            "class": "primary",
            "names_lr": [{"between": [0.0, 1.0], "value": "A"}],
        }
        target = {
            "class": "secondary",
            "names_lr": [{"between": [0.0, 1.0], "value": "B"}],
        }
        result = self._call(
            ref,
            target,
            has_ref_names_lr=has_ref_names_lr,
            has_target_names_lr=has_target_names_lr,
            has_ref_class=has_ref_class,
            has_target_class=has_target_class,
        )
        ref_name, target_name, ref_class, target_class, _, _ = result
        if not has_ref_names_lr:
            assert ref_name is None and target_name is None
        if not has_ref_class:
            assert ref_class is None and target_class is None

    def test_lr_overrides_primary_name_for_partial_alignment(self):
        """LR name for the aligned portion should override the primary name (original bug)."""
        ref = {"names_lr": SPLIT_LR, "class": "primary"}
        target = {
            "names_lr": [{"between": [0.0, 1.0], "value": "FIRST AVENUE"}],
            "class": "residential",
        }
        ref_name, target_name, *_ = self._call(
            ref,
            target,
            has_ref_names_lr=True,
            has_target_names_lr=True,
            ref_start=0.0,
            ref_end=0.4,
        )
        assert ref_name == "First Avenue"  # Not "Second Street"
        assert target_name == "FIRST AVENUE"

    def test_works_with_pandas_series(self):
        """Should accept both dicts and pandas Series (from .iloc[0])."""
        ref = pd.Series(
            {
                "class": "secondary",
                "names_lr": [{"between": [0.0, 1.0], "value": "Oak Ave"}],
            }
        )
        target = pd.Series(
            {
                "class": "tertiary",
                "names_lr": [{"between": [0.0, 1.0], "value": "Oak Avenue"}],
            }
        )
        ref_name, target_name, ref_class, target_class, _, _ = self._call(
            ref, target, has_ref_names_lr=True, has_target_names_lr=True
        )
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
