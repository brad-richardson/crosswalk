"""Tests for the blocking-stage recall metric."""

import math

import geopandas as gpd
import pandas as pd
import pytest
from shapely import LineString, MultiLineString

from matcher.quality.blocking_recall import (
    BlockingRecallResult,
    compute_blocking_recall,
)


def _make_reference() -> gpd.GeoDataFrame:
    """Three reference lines, well separated so buffers are unambiguous."""
    return gpd.GeoDataFrame(
        {
            "id": ["ref_a", "ref_b", "ref_c"],
            "geometry": [
                LineString([(0, 0), (100, 0)]),
                LineString([(0, 5000), (100, 5000)]),
                LineString([(0, 10000), (100, 10000)]),
            ],
        },
        crs="EPSG:32610",  # Projected CRS: coordinates are meters
    )


def _make_target() -> gpd.GeoDataFrame:
    """Targets: one near ref_a, one 200m from ref_b, one MultiLineString near ref_c."""
    return gpd.GeoDataFrame(
        {
            "id": ["t_near", "t_far", "t_mls"],
            "geometry": [
                LineString([(0, 10), (100, 10)]),  # 10m from ref_a
                LineString([(0, 5200), (100, 5200)]),  # 200m from ref_b
                MultiLineString(
                    [[(0, 10010), (50, 10010)], [(60, 10010), (100, 10010)]]
                ),  # near ref_c, but dropped by the pipeline
            ],
        },
        crs="EPSG:32610",
    )


def _make_labels(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["gers_id", "target_id", "label"])


class TestComputeBlockingRecall:
    def test_full_scenario(self):
        """Blocked, missed, unresolvable, and MultiLineString cases together."""
        labels = _make_labels(
            [
                ("ref_a", "t_near", "match"),  # within 50m buffer -> blocked
                ("ref_b", "t_far", "match"),  # 200m away -> missed at 50m
                ("ref_missing", "t_near", "match"),  # unknown ref id -> unresolvable
                ("ref_c", "t_mls", "match"),  # MultiLineString target -> dropped bucket
                ("ref_a", "t_far", "no_match"),  # non-match labels are ignored
            ]
        )

        result = compute_blocking_recall(
            reference_gdf=_make_reference(),
            target_gdf=_make_target(),
            labels_df=labels,
            buffer_distance_m=50.0,
            alt_buffer_distances=(250.0,),
        )

        assert isinstance(result, BlockingRecallResult)
        assert result.total_match_labels == 4
        assert result.blocked == 1
        assert len(result.missed) == 1
        assert result.unresolvable == [("ref_missing", "t_near")]
        assert result.multilinestring_dropped == [("ref_c", "t_mls")]

        # Recall counts only resolvable pairs: 1 blocked of 2 resolvable
        assert result.n_resolvable == 2
        assert result.recall == pytest.approx(0.5)

    def test_missed_pair_reports_distance(self):
        """Missed pair carries the minimum geometry-to-geometry distance."""
        labels = _make_labels([("ref_b", "t_far", "match")])

        result = compute_blocking_recall(
            reference_gdf=_make_reference(),
            target_gdf=_make_target(),
            labels_df=labels,
            buffer_distance_m=50.0,
        )

        assert result.blocked == 0
        assert len(result.missed) == 1
        missed = result.missed[0]
        assert missed.gers_id == "ref_b"
        assert missed.target_id == "t_far"
        assert missed.distance_m == pytest.approx(200.0)

    def test_recall_at_alternative_buffers(self):
        """A larger buffer recovers the 200m-apart pair."""
        labels = _make_labels(
            [
                ("ref_a", "t_near", "match"),
                ("ref_b", "t_far", "match"),
            ]
        )

        result = compute_blocking_recall(
            reference_gdf=_make_reference(),
            target_gdf=_make_target(),
            labels_df=labels,
            buffer_distance_m=50.0,
            alt_buffer_distances=(100.0, 250.0),
        )

        assert result.recall == pytest.approx(0.5)
        assert result.recall_at_buffer[50.0] == pytest.approx(0.5)
        assert result.recall_at_buffer[100.0] == pytest.approx(0.5)  # 200m still missed
        assert result.recall_at_buffer[250.0] == pytest.approx(1.0)  # recovered

    def test_pair_within_buffer_is_blocked(self):
        """A pair within the buffer counts as blocked with 100% recall."""
        labels = _make_labels([("ref_a", "t_near", "match")])

        result = compute_blocking_recall(
            reference_gdf=_make_reference(),
            target_gdf=_make_target(),
            labels_df=labels,
            buffer_distance_m=50.0,
        )

        assert result.blocked == 1
        assert result.missed == []
        assert result.recall == pytest.approx(1.0)

    def test_unresolvable_target_id(self):
        """Unknown target IDs are counted as unresolvable, not missed."""
        labels = _make_labels(
            [
                ("ref_a", "t_near", "match"),
                ("ref_a", "t_unknown", "match"),
            ]
        )

        result = compute_blocking_recall(
            reference_gdf=_make_reference(),
            target_gdf=_make_target(),
            labels_df=labels,
            buffer_distance_m=50.0,
        )

        assert result.unresolvable == [("ref_a", "t_unknown")]
        # Unresolvable pair does not count against recall
        assert result.recall == pytest.approx(1.0)

    def test_no_match_labels(self):
        """No labeled matches yields NaN recall, not a crash."""
        labels = _make_labels([("ref_a", "t_near", "no_match")])

        result = compute_blocking_recall(
            reference_gdf=_make_reference(),
            target_gdf=_make_target(),
            labels_df=labels,
            buffer_distance_m=50.0,
        )

        assert result.total_match_labels == 0
        assert math.isnan(result.recall)

    def test_only_unresolvable_labels(self):
        """All-unresolvable labels yields NaN recall with buckets populated."""
        labels = _make_labels([("ref_missing", "t_unknown", "match")])

        result = compute_blocking_recall(
            reference_gdf=_make_reference(),
            target_gdf=_make_target(),
            labels_df=labels,
            buffer_distance_m=50.0,
        )

        assert result.total_match_labels == 1
        assert result.unresolvable == [("ref_missing", "t_unknown")]
        assert math.isnan(result.recall)

    def test_explicit_zero_buffer_is_honored(self):
        """An explicit 0.0 buffer must not be replaced by the settings default."""
        labels = _make_labels([("ref_a", "t_near", "match")])

        result = compute_blocking_recall(
            reference_gdf=_make_reference(),
            target_gdf=_make_target(),
            labels_df=labels,
            buffer_distance_m=0.0,
        )

        # 0.0 is used as-is (settings default would be 50.0), so the 10m-apart
        # pair is missed rather than blocked
        assert result.buffer_distance_m == 0.0
        assert result.blocked == 0
        assert len(result.missed) == 1

    def test_duplicate_labels_counted_once(self):
        """Duplicate (gers_id, target_id) match labels are deduplicated."""
        labels = _make_labels(
            [
                ("ref_a", "t_near", "match"),
                ("ref_a", "t_near", "match"),
            ]
        )

        result = compute_blocking_recall(
            reference_gdf=_make_reference(),
            target_gdf=_make_target(),
            labels_df=labels,
            buffer_distance_m=50.0,
        )

        assert result.total_match_labels == 1
        assert result.blocked == 1
