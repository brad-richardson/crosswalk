"""Tests for features/pipeline.py (prepare_worker_data).

Validates the shared worker_data preparation function that both
score_candidates() and compute_features_only() delegate to.
"""

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from matcher.blocking.spatial_index import CandidatePair
from matcher.features.pipeline import (
    WorkerDataResult,
    _extract_column_array,
    _extract_lr_column,
    prepare_worker_data,
)


class TestExtractColumnArray:
    """Tests for _extract_column_array helper."""

    def test_returns_column_values_when_present(self):
        """Should return the column as a numpy array."""
        gdf = gpd.GeoDataFrame(
            {
                "names": ["Main St", "Oak Ave"],
                "geometry": [LineString([(0, 0), (1, 0)]), LineString([(0, 0), (0, 1)])],
            },
        )
        result = _extract_column_array(gdf, "names", len(gdf))
        assert list(result) == ["Main St", "Oak Ave"]

    def test_returns_none_array_when_column_missing(self):
        """Should return array of Nones when column doesn't exist."""
        gdf = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 0)]), LineString([(0, 0), (0, 1)])]},
        )
        result = _extract_column_array(gdf, "names", len(gdf))
        assert len(result) == 2
        assert all(v is None for v in result)


class TestExtractLrColumn:
    """Tests for _extract_lr_column helper."""

    def test_returns_array_when_column_present(self):
        """Should return the column as a numpy array."""
        gdf = gpd.GeoDataFrame(
            {
                "names_lr": [[{"between": [0.0, 1.0], "value": "A"}], None],
                "geometry": [LineString([(0, 0), (1, 0)]), LineString([(0, 0), (0, 1)])],
            },
        )
        result = _extract_lr_column(gdf, "names_lr")
        assert result is not None
        assert len(result) == 2

    def test_returns_none_when_column_missing(self):
        """Should return None when column doesn't exist."""
        gdf = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 0)])]},
        )
        result = _extract_lr_column(gdf, "names_lr")
        assert result is None


class TestPrepareWorkerData:
    """Tests for prepare_worker_data() — the shared pipeline setup function."""

    @pytest.fixture
    def reference(self):
        """Reference GeoDataFrame with two segments in projected CRS."""
        return gpd.GeoDataFrame(
            {
                "id": ["ref_1", "ref_2"],
                "names": [{"primary": "Main Street"}, {"primary": "Oak Avenue"}],
                "class": ["primary", "secondary"],
                "subclass": ["highway", "local"],
                "geometry": [
                    LineString([(500000, 4000000), (500100, 4000000)]),
                    LineString([(500100, 4000000), (500100, 4000100)]),
                ],
            },
            crs="EPSG:32610",
        )

    @pytest.fixture
    def target(self):
        """Target GeoDataFrame with two segments in projected CRS."""
        return gpd.GeoDataFrame(
            {
                "id": ["target_1", "target_2"],
                "names": [{"primary": "Main St"}, {"primary": "Oak Ave"}],
                "class": ["residential", "tertiary"],
                "geometry": [
                    LineString([(500000, 4000005), (500100, 4000005)]),
                    LineString([(500105, 4000000), (500105, 4000100)]),
                ],
            },
            crs="EPSG:32610",
        )

    @pytest.fixture
    def candidates(self):
        """Two candidate pairs linking ref to target segments."""
        return [
            CandidatePair(
                ref_id="ref_1",
                ref_idx=0,
                target_id="target_1",
                target_idx=0,
                distance_estimate=5.0,
                heading_diff=0.0,
                length_ratio=1.0,
            ),
            CandidatePair(
                ref_id="ref_2",
                ref_idx=1,
                target_id="target_2",
                target_idx=1,
                distance_estimate=5.0,
                heading_diff=0.0,
                length_ratio=1.0,
            ),
        ]

    def test_returns_worker_data_result(self, candidates, reference, target):
        """Should return a WorkerDataResult namedtuple."""
        result = prepare_worker_data(
            candidates=candidates,
            reference=reference,
            target=target,
            ref_id_column="id",
            target_id_column="id",
            ref_name_column="names",
            target_name_column="names",
            ref_class_column="class",
            target_class_column="class",
            n_jobs=1,
        )
        assert isinstance(result, WorkerDataResult)

    def test_worker_data_contains_required_geometry_keys(self, candidates, reference, target):
        """worker_data must contain ref_geoms and target_geoms arrays."""
        result = prepare_worker_data(
            candidates=candidates,
            reference=reference,
            target=target,
            ref_id_column="id",
            target_id_column="id",
            n_jobs=1,
        )
        wd = result.worker_data
        assert "ref_geoms" in wd
        assert "target_geoms" in wd
        assert len(wd["ref_geoms"]) == len(reference)
        assert len(wd["target_geoms"]) == len(target)

    def test_worker_data_contains_name_arrays(self, candidates, reference, target):
        """worker_data must contain ref_names and target_names arrays."""
        result = prepare_worker_data(
            candidates=candidates,
            reference=reference,
            target=target,
            ref_id_column="id",
            target_id_column="id",
            ref_name_column="names",
            target_name_column="names",
            n_jobs=1,
        )
        wd = result.worker_data
        assert "ref_names" in wd
        assert "target_names" in wd
        assert len(wd["ref_names"]) == len(reference)
        assert len(wd["target_names"]) == len(target)

    def test_worker_data_contains_class_arrays(self, candidates, reference, target):
        """worker_data must contain class and subclass arrays."""
        result = prepare_worker_data(
            candidates=candidates,
            reference=reference,
            target=target,
            ref_id_column="id",
            target_id_column="id",
            ref_class_column="class",
            target_class_column="class",
            n_jobs=1,
        )
        wd = result.worker_data
        assert "ref_classes" in wd
        assert "target_classes" in wd

    def test_worker_data_contains_id_arrays(self, candidates, reference, target):
        """worker_data must contain ref_ids and target_ids arrays."""
        result = prepare_worker_data(
            candidates=candidates,
            reference=reference,
            target=target,
            ref_id_column="id",
            target_id_column="id",
            n_jobs=1,
        )
        wd = result.worker_data
        assert "ref_ids" in wd
        assert "target_ids" in wd
        assert list(wd["ref_ids"]) == ["ref_1", "ref_2"]
        assert list(wd["target_ids"]) == ["target_1", "target_2"]

    def test_worker_data_contains_lr_columns(self, candidates, reference, target):
        """worker_data must contain LR column arrays (None when absent)."""
        result = prepare_worker_data(
            candidates=candidates,
            reference=reference,
            target=target,
            ref_id_column="id",
            target_id_column="id",
            n_jobs=1,
        )
        wd = result.worker_data
        # These columns don't exist in our fixture, so should be None
        assert "ref_names_lr" in wd
        assert "target_names_lr" in wd
        assert wd["ref_names_lr"] is None
        assert wd["target_names_lr"] is None

    def test_worker_data_lr_columns_present_when_available(self, candidates, reference, target):
        """When LR columns exist in the GDF, they should be in worker_data."""
        reference = reference.copy()
        reference["names_lr"] = [
            [{"between": [0.0, 1.0], "value": "Main Street"}],
            [{"between": [0.0, 1.0], "value": "Oak Avenue"}],
        ]
        result = prepare_worker_data(
            candidates=candidates,
            reference=reference,
            target=target,
            ref_id_column="id",
            target_id_column="id",
            n_jobs=1,
        )
        wd = result.worker_data
        assert wd["ref_names_lr"] is not None
        assert len(wd["ref_names_lr"]) == 2
        # Target still has no LR
        assert wd["target_names_lr"] is None

    def test_worker_data_contains_topology_and_graphlet(self, candidates, reference, target):
        """worker_data must contain topology and graphlet pre-computed features."""
        result = prepare_worker_data(
            candidates=candidates,
            reference=reference,
            target=target,
            ref_id_column="id",
            target_id_column="id",
            n_jobs=1,
        )
        wd = result.worker_data
        assert "ref_topology" in wd
        assert "target_topology" in wd
        assert "ref_graphlet_data" in wd
        assert "target_graphlet_data" in wd

    def test_worker_data_contains_sibling_contexts(self, candidates, reference, target):
        """worker_data must contain sibling search contexts."""
        result = prepare_worker_data(
            candidates=candidates,
            reference=reference,
            target=target,
            ref_id_column="id",
            target_id_column="id",
            n_jobs=1,
        )
        wd = result.worker_data
        assert "ref_sibling_context" in wd
        assert "target_sibling_context" in wd

    def test_worker_data_contains_alignments(self, candidates, reference, target):
        """worker_data must contain alignments dict and it should match result.alignments."""
        result = prepare_worker_data(
            candidates=candidates,
            reference=reference,
            target=target,
            ref_id_column="id",
            target_id_column="id",
            n_jobs=1,
        )
        wd = result.worker_data
        assert "alignments" in wd
        assert wd["alignments"] is result.alignments

    def test_worker_data_contains_aligned_endpoint_features(self, candidates, reference, target):
        """worker_data must contain aligned_endpoint_features dict."""
        result = prepare_worker_data(
            candidates=candidates,
            reference=reference,
            target=target,
            ref_id_column="id",
            target_id_column="id",
            n_jobs=1,
        )
        wd = result.worker_data
        assert "aligned_endpoint_features" in wd
        assert isinstance(wd["aligned_endpoint_features"], dict)

    def test_unique_indices_match_candidates(self, candidates, reference, target):
        """unique_ref_indices and unique_target_indices should match candidate indices."""
        result = prepare_worker_data(
            candidates=candidates,
            reference=reference,
            target=target,
            ref_id_column="id",
            target_id_column="id",
            n_jobs=1,
        )
        assert result.unique_ref_indices == {0, 1}
        assert result.unique_target_indices == {0, 1}

    def test_topology_features_per_candidate_segment(self, candidates, reference, target):
        """Topology features should be keyed by DataFrame index for each candidate segment."""
        result = prepare_worker_data(
            candidates=candidates,
            reference=reference,
            target=target,
            ref_id_column="id",
            target_id_column="id",
            n_jobs=1,
        )
        wd = result.worker_data
        # Each candidate segment index should have topology features
        for idx in result.unique_ref_indices:
            assert idx in wd["ref_topology"]
            topo = wd["ref_topology"][idx]
            assert "from_degree" in topo
            assert "to_degree" in topo
        for idx in result.unique_target_indices:
            assert idx in wd["target_topology"]

    def test_missing_optional_columns_handled(self):
        """Should work when optional columns (subclass, names_lr) are absent."""
        reference = gpd.GeoDataFrame(
            {
                "id": ["ref_1"],
                "geometry": [LineString([(500000, 4000000), (500100, 4000000)])],
            },
            crs="EPSG:32610",
        )
        target = gpd.GeoDataFrame(
            {
                "id": ["target_1"],
                "geometry": [LineString([(500000, 4000005), (500100, 4000005)])],
            },
            crs="EPSG:32610",
        )
        candidates = [
            CandidatePair(
                ref_id="ref_1",
                ref_idx=0,
                target_id="target_1",
                target_idx=0,
                distance_estimate=5.0,
                heading_diff=0.0,
                length_ratio=1.0,
            ),
        ]
        result = prepare_worker_data(
            candidates=candidates,
            reference=reference,
            target=target,
            ref_id_column="id",
            target_id_column="id",
            ref_name_column="names",
            target_name_column="names",
            ref_class_column="class",
            target_class_column="class",
            n_jobs=1,
        )
        wd = result.worker_data
        # Names should be None arrays since column doesn't exist
        assert all(v is None for v in wd["ref_names"])
        assert all(v is None for v in wd["target_names"])
        # Classes should be None arrays
        assert all(v is None for v in wd["ref_classes"])
        assert all(v is None for v in wd["target_classes"])
        # LR should be None
        assert wd["ref_names_lr"] is None

    def test_background_alignment_produces_same_keys(self, candidates, reference, target):
        """Background alignment should produce the same worker_data keys."""
        result_serial = prepare_worker_data(
            candidates=candidates,
            reference=reference,
            target=target,
            ref_id_column="id",
            target_id_column="id",
            n_jobs=1,
            run_alignment_in_background=False,
        )
        result_bg = prepare_worker_data(
            candidates=candidates,
            reference=reference,
            target=target,
            ref_id_column="id",
            target_id_column="id",
            n_jobs=1,
            run_alignment_in_background=True,
        )
        assert set(result_serial.worker_data.keys()) == set(result_bg.worker_data.keys())


class TestPrepareWorkerDataParity:
    """Verify that prepare_worker_data produces all keys expected by workers.

    This is a regression test for the original bug where compute_features_only()
    was missing LR columns that score_candidates() had.
    """

    REQUIRED_WORKER_DATA_KEYS = {
        "ref_geoms",
        "target_geoms",
        "ref_names",
        "target_names",
        "ref_classes",
        "target_classes",
        "ref_subclasses",
        "target_subclasses",
        "ref_names_lr",
        "target_names_lr",
        "ref_oneway_lr",
        "target_oneway_lr",
        "ref_speed_limit_kph_lr",
        "target_speed_limit_kph_lr",
        "ref_ids",
        "target_ids",
        "aligned_endpoint_features",
        "ref_topology",
        "target_topology",
        "ref_graphlet_data",
        "target_graphlet_data",
        "ref_sibling_context",
        "target_sibling_context",
        "alignments",
    }

    def test_all_required_keys_present(self):
        """worker_data must contain ALL keys that workers expect to find."""
        reference = gpd.GeoDataFrame(
            {
                "id": ["ref_1"],
                "names": [{"primary": "Test"}],
                "class": ["primary"],
                "geometry": [LineString([(500000, 4000000), (500100, 4000000)])],
            },
            crs="EPSG:32610",
        )
        target = gpd.GeoDataFrame(
            {
                "id": ["target_1"],
                "names": [{"primary": "Test"}],
                "class": ["residential"],
                "geometry": [LineString([(500000, 4000005), (500100, 4000005)])],
            },
            crs="EPSG:32610",
        )
        candidates = [
            CandidatePair(
                ref_id="ref_1",
                ref_idx=0,
                target_id="target_1",
                target_idx=0,
                distance_estimate=5.0,
                heading_diff=0.0,
                length_ratio=1.0,
            ),
        ]
        result = prepare_worker_data(
            candidates=candidates,
            reference=reference,
            target=target,
            ref_id_column="id",
            target_id_column="id",
            n_jobs=1,
        )
        actual_keys = set(result.worker_data.keys())
        missing = self.REQUIRED_WORKER_DATA_KEYS - actual_keys
        assert not missing, f"Missing worker_data keys: {missing}"
        # No unexpected keys (would indicate a divergence)
        extra = actual_keys - self.REQUIRED_WORKER_DATA_KEYS
        assert not extra, f"Unexpected worker_data keys: {extra}"
