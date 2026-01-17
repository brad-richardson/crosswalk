"""Tests for validation module."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely import LineString

from matcher.validation.evaluate import (
    analyze_failures,
    compute_metrics,
    evaluate_by_record_id,
    get_osm_way_id,
)
from matcher.validation.holdout import (
    create_holdout,
    drop_random_osm,
    extract_record_ids,
    has_source,
)


class TestGetOsmWayId:
    """Tests for OSM way ID extraction."""

    def test_plain_way_id(self):
        """Should return plain way ID unchanged."""
        assert get_osm_way_id("w123456") == "w123456"

    def test_versioned_way_id(self):
        """Should strip version suffix from way ID."""
        assert get_osm_way_id("w123456@5") == "w123456"

    def test_empty_string(self):
        """Should handle empty string."""
        assert get_osm_way_id("") == ""

    def test_none_handling(self):
        """Should handle None-like values."""
        assert get_osm_way_id(None) == ""

    def test_numeric_input(self):
        """Should convert numeric input to string."""
        assert get_osm_way_id(123456) == "123456"


class TestEvaluateByRecordId:
    """Tests for evaluate_by_record_id function."""

    def test_basic_evaluation(self):
        """Should correctly identify matched and unmatched segments."""
        # Fresh OSM data
        fresh_osm = gpd.GeoDataFrame(
            {
                "id": ["w100", "w200", "w300"],
                "geometry": [
                    LineString([(0, 0), (10, 0)]),
                    LineString([(0, 10), (10, 10)]),
                    LineString([(0, 20), (10, 20)]),
                ],
            },
            crs="EPSG:32610",
        )

        # Bridge (match results) - w100 and w200 were matched
        bridge = pd.DataFrame(
            {
                "target_id": ["w100", "w200"],
                "gers_id": ["gers_1", "gers_2"],
                "confidence": [0.95, 0.85],
            }
        )

        # Unmatched segments
        unmatched = pd.DataFrame({"target_id": ["w300"]})

        # Dropped record IDs - only w100 and w300 were dropped
        dropped_ids = {"w100", "w300"}

        eval_df = evaluate_by_record_id(
            bridge=bridge,
            unmatched=unmatched,
            fresh_osm=fresh_osm,
            dropped_record_ids=dropped_ids,
        )

        assert len(eval_df) == 3

        # w100: should match (dropped), matched (in bridge)
        w100_row = eval_df[eval_df["osm_id"] == "w100"].iloc[0]
        assert w100_row["should_match"]
        assert w100_row["matched"]
        assert w100_row["confidence"] == 0.95

        # w200: should NOT match (not dropped), but was matched
        w200_row = eval_df[eval_df["osm_id"] == "w200"].iloc[0]
        assert not w200_row["should_match"]
        assert w200_row["matched"]

        # w300: should match (dropped), NOT matched (in unmatched)
        w300_row = eval_df[eval_df["osm_id"] == "w300"].iloc[0]
        assert w300_row["should_match"]
        assert not w300_row["matched"]

    def test_versioned_record_id_matching(self):
        """Should match versioned record IDs correctly."""
        fresh_osm = gpd.GeoDataFrame(
            {
                "id": ["w100", "w200"],
                "geometry": [
                    LineString([(0, 0), (10, 0)]),
                    LineString([(0, 10), (10, 10)]),
                ],
            },
            crs="EPSG:32610",
        )

        bridge = pd.DataFrame(
            {
                "target_id": ["w100"],
                "confidence": [0.9],
            }
        )

        unmatched = pd.DataFrame()

        # Dropped IDs include versioned record_ids
        dropped_ids = {"w100@5", "w200@3"}  # Should match w100 and w200

        eval_df = evaluate_by_record_id(
            bridge=bridge,
            unmatched=unmatched,
            fresh_osm=fresh_osm,
            dropped_record_ids=dropped_ids,
        )

        # Both should be marked as should_match
        assert eval_df["should_match"].all()

    def test_missing_osm_id_column_raises(self):
        """Should raise ValueError for missing OSM ID column."""
        fresh_osm = gpd.GeoDataFrame(
            {"other_col": ["a"]},
            geometry=[LineString([(0, 0), (1, 1)])],
            crs="EPSG:32610",
        )
        bridge = pd.DataFrame()
        unmatched = pd.DataFrame()

        with pytest.raises(ValueError, match="OSM ID column"):
            evaluate_by_record_id(
                bridge=bridge,
                unmatched=unmatched,
                fresh_osm=fresh_osm,
                dropped_record_ids=set(),
                osm_id_column="id",
            )


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_perfect_recall(self):
        """Should compute 100% recall when all dropped segments matched."""
        eval_df = pd.DataFrame(
            {
                "osm_id": ["w1", "w2", "w3"],
                "should_match": [True, True, False],
                "matched": [True, True, True],
                "confidence": [0.9, 0.8, 0.95],
            }
        )

        metrics = compute_metrics(eval_df)

        assert metrics["total_dropped"] == 2
        assert metrics["matched_back"] == 2
        assert metrics["orphaned"] == 0
        assert metrics["recall"] == 1.0

    def test_zero_recall(self):
        """Should compute 0% recall when no dropped segments matched."""
        eval_df = pd.DataFrame(
            {
                "osm_id": ["w1", "w2"],
                "should_match": [True, True],
                "matched": [False, False],
                "confidence": [None, None],
            }
        )

        metrics = compute_metrics(eval_df)

        assert metrics["total_dropped"] == 2
        assert metrics["matched_back"] == 0
        assert metrics["orphaned"] == 2
        assert metrics["recall"] == 0.0

    def test_partial_recall(self):
        """Should compute correct partial recall."""
        eval_df = pd.DataFrame(
            {
                "osm_id": ["w1", "w2", "w3", "w4"],
                "should_match": [True, True, True, True],
                "matched": [True, True, False, False],
                "confidence": [0.9, 0.8, None, None],
            }
        )

        metrics = compute_metrics(eval_df)

        assert metrics["total_dropped"] == 4
        assert metrics["matched_back"] == 2
        assert metrics["orphaned"] == 2
        assert metrics["recall"] == 0.5

    def test_mean_confidence(self):
        """Should compute mean confidence of matched segments."""
        eval_df = pd.DataFrame(
            {
                "osm_id": ["w1", "w2", "w3"],
                "should_match": [True, True, True],
                "matched": [True, True, False],
                "confidence": [0.9, 0.7, None],
            }
        )

        metrics = compute_metrics(eval_df)

        assert metrics["mean_confidence"] == pytest.approx(0.8)

    def test_unexpected_matches(self):
        """Should count unexpected matches (matched but should not have)."""
        eval_df = pd.DataFrame(
            {
                "osm_id": ["w1", "w2", "w3"],
                "should_match": [False, False, True],
                "matched": [True, True, True],  # w1, w2 are unexpected matches
                "confidence": [0.9, 0.8, 0.7],
            }
        )

        metrics = compute_metrics(eval_df)

        assert metrics["unexpected_matches"] == 2

    def test_empty_should_match(self):
        """Should handle case with no segments that should match."""
        eval_df = pd.DataFrame(
            {
                "osm_id": ["w1", "w2"],
                "should_match": [False, False],
                "matched": [True, False],
                "confidence": [0.9, None],
            }
        )

        metrics = compute_metrics(eval_df)

        assert metrics["total_dropped"] == 0
        assert metrics["recall"] == 0.0


class TestAnalyzeFailures:
    """Tests for analyze_failures function."""

    def test_identifies_false_negatives(self):
        """Should identify segments that should have matched but didn't."""
        fresh_osm = gpd.GeoDataFrame(
            {
                "id": ["w1", "w2", "w3"],
                "geometry": [
                    LineString([(0, 0), (10, 0)]),
                    LineString([(0, 10), (10, 10)]),
                    LineString([(0, 20), (10, 20)]),
                ],
            },
            crs="EPSG:32610",
        )

        eval_df = pd.DataFrame(
            {
                "osm_id": ["w1", "w2", "w3"],
                "should_match": [True, True, False],
                "matched": [True, False, False],  # w2 is a false negative
            }
        )

        failures = analyze_failures(eval_df, fresh_osm)

        assert len(failures) == 1
        assert "w2" in failures["id"].values


class TestExtractRecordIds:
    """Tests for record ID extraction from sources."""

    def test_extracts_osm_record_ids(self):
        """Should extract OSM record_ids from sources."""
        sources = pd.Series(
            [
                [{"dataset": "OpenStreetMap", "record_id": "w123"}],
                [{"dataset": "TomTom", "record_id": "tt456"}],
                [{"dataset": "OpenStreetMap", "record_id": "w789@5"}],
            ]
        )

        record_ids = extract_record_ids(sources, dataset_filter="OpenStreetMap")

        assert record_ids.iloc[0] == {"w123"}
        assert record_ids.iloc[1] == set()  # TomTom, not OSM
        assert record_ids.iloc[2] == {"w789"}  # Version stripped

    def test_handles_multi_source_segments(self):
        """Should extract multiple record_ids from merged segments."""
        sources = pd.Series(
            [
                [
                    {"dataset": "OpenStreetMap", "record_id": "w100"},
                    {"dataset": "OpenStreetMap", "record_id": "w101"},
                    {"dataset": "TomTom", "record_id": "tt999"},
                ],
            ]
        )

        record_ids = extract_record_ids(sources, dataset_filter="OpenStreetMap")

        assert record_ids.iloc[0] == {"w100", "w101"}

    def test_handles_none_sources(self):
        """Should handle None in sources."""
        sources = pd.Series([None, [{"dataset": "OpenStreetMap", "record_id": "w123"}]])

        record_ids = extract_record_ids(sources, dataset_filter="OpenStreetMap")

        assert record_ids.iloc[0] == set()
        assert record_ids.iloc[1] == {"w123"}


class TestHasSource:
    """Tests for has_source helper function."""

    def test_source_present(self):
        """Should return True when source is present."""
        sources = [{"dataset": "OpenStreetMap", "record_id": "w123"}]
        assert has_source(sources, "OpenStreetMap") is True

    def test_source_absent(self):
        """Should return False when source is not present."""
        sources = [{"dataset": "TomTom", "record_id": "tt456"}]
        assert has_source(sources, "OpenStreetMap") is False

    def test_none_sources(self):
        """Should handle None sources."""
        assert has_source(None, "OpenStreetMap") is False


class TestDropRandomOsm:
    """Tests for drop_random_osm function."""

    def test_fraction_validation_low(self):
        """Should raise ValueError for fraction < 0."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["1"],
                "sources": [[{"dataset": "OpenStreetMap", "record_id": "w1"}]],
            },
            geometry=[LineString([(0, 0), (1, 1)])],
            crs="EPSG:32610",
        )

        with pytest.raises(ValueError, match="fraction must be between"):
            drop_random_osm(gdf, fraction=-0.1)

    def test_fraction_validation_high(self):
        """Should raise ValueError for fraction > 1."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["1"],
                "sources": [[{"dataset": "OpenStreetMap", "record_id": "w1"}]],
            },
            geometry=[LineString([(0, 0), (1, 1)])],
            crs="EPSG:32610",
        )

        with pytest.raises(ValueError, match="fraction must be between"):
            drop_random_osm(gdf, fraction=1.5)

    def test_missing_sources_column(self):
        """Should raise ValueError for missing sources column."""
        gdf = gpd.GeoDataFrame(
            {"id": ["1"]},
            geometry=[LineString([(0, 0), (1, 1)])],
            crs="EPSG:32610",
        )

        with pytest.raises(ValueError, match="sources"):
            drop_random_osm(gdf, fraction=0.5)

    def test_zero_fraction_returns_copy(self):
        """Should return full copy when fraction results in 0 drops."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["1", "2"],
                "sources": [
                    [{"dataset": "OpenStreetMap", "record_id": "w1"}],
                    [{"dataset": "OpenStreetMap", "record_id": "w2"}],
                ],
            },
            geometry=[
                LineString([(0, 0), (1, 1)]),
                LineString([(2, 2), (3, 3)]),
            ],
            crs="EPSG:32610",
        )

        # Very small fraction that results in 0 segments to drop
        reduced, dropped_ids = drop_random_osm(gdf, fraction=0.01)

        assert len(reduced) == 2
        assert len(dropped_ids) == 0

    def test_drops_correct_fraction(self):
        """Should drop approximately correct fraction of segments."""
        n_segments = 100
        sources_data = [
            [{"dataset": "OpenStreetMap", "record_id": f"w{i}"}] for i in range(n_segments)
        ]

        gdf = gpd.GeoDataFrame(
            {
                "id": [str(i) for i in range(n_segments)],
                "sources": sources_data,
            },
            geometry=[LineString([(i, 0), (i + 1, 0)]) for i in range(n_segments)],
            crs="EPSG:32610",
        )

        reduced, dropped_ids = drop_random_osm(gdf, fraction=0.2, seed=42)

        # Should drop ~20% = 20 segments
        assert len(reduced) == 80
        assert len(dropped_ids) == 20

    def test_reproducibility_with_seed(self):
        """Should produce same results with same seed."""
        n_segments = 50
        sources_data = [
            [{"dataset": "OpenStreetMap", "record_id": f"w{i}"}] for i in range(n_segments)
        ]

        gdf = gpd.GeoDataFrame(
            {
                "id": [str(i) for i in range(n_segments)],
                "sources": sources_data,
            },
            geometry=[LineString([(i, 0), (i + 1, 0)]) for i in range(n_segments)],
            crs="EPSG:32610",
        )

        reduced1, dropped1 = drop_random_osm(gdf, fraction=0.3, seed=123)
        reduced2, dropped2 = drop_random_osm(gdf, fraction=0.3, seed=123)

        assert dropped1 == dropped2
        assert set(reduced1.index) == set(reduced2.index)


class TestCreateHoldout:
    """Tests for create_holdout function."""

    def test_creates_reduced_reference(self):
        """Should create reduced reference without dropped segments."""
        overture = gpd.GeoDataFrame(
            {
                "id": ["1", "2", "3"],
                "sources": [
                    [{"dataset": "OpenStreetMap", "record_id": "w1"}],
                    [{"dataset": "OpenStreetMap", "record_id": "w2"}],
                    [{"dataset": "TomTom", "record_id": "tt3"}],
                ],
            },
            geometry=[
                LineString([(0, 0), (1, 1)]),
                LineString([(2, 2), (3, 3)]),
                LineString([(4, 4), (5, 5)]),
            ],
            crs="EPSG:32610",
        )

        # Drop first two segments
        segments_to_drop = overture.iloc[:2]

        reduced, dropped_ids = create_holdout(overture, segments_to_drop)

        assert len(reduced) == 1
        assert reduced.iloc[0]["id"] == "3"
        assert "w1" in dropped_ids
        assert "w2" in dropped_ids
