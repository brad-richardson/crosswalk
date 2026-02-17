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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gdf(ids, names=None, classes=None, coords_list=None, crs="EPSG:32610", **extra):
    """Build a minimal GeoDataFrame for testing."""
    n = len(ids)
    data = {"id": ids}
    if names is not None:
        data["names"] = names
    if classes is not None:
        data["class"] = classes
    if coords_list is None:
        # Default: parallel horizontal lines offset by index * 5m
        coords_list = [[(500000, 4000000 + i * 5), (500100, 4000000 + i * 5)] for i in range(n)]
    data["geometry"] = [LineString(c) for c in coords_list]
    data.update(extra)
    return gpd.GeoDataFrame(data, crs=crs)


def _make_candidates(ref_ids, target_ids, ref_idxs=None, target_idxs=None):
    """Build CandidatePair list from parallel lists of IDs."""
    if ref_idxs is None:
        ref_idxs = list(range(len(ref_ids)))
    if target_idxs is None:
        target_idxs = list(range(len(target_ids)))
    return [
        CandidatePair(
            ref_id=rid,
            ref_idx=ridx,
            target_id=tid,
            target_idx=tidx,
            distance_estimate=5.0,
            heading_diff=0.0,
        )
        for rid, ridx, tid, tidx in zip(ref_ids, ref_idxs, target_ids, target_idxs)
    ]


# ---------------------------------------------------------------------------
# Column extraction helpers
# ---------------------------------------------------------------------------


class TestExtractColumnArray:
    def test_returns_values_when_present(self):
        gdf = _make_gdf(["a", "b"], names=["Main St", "Oak Ave"])
        assert list(_extract_column_array(gdf, "names", 2)) == ["Main St", "Oak Ave"]

    def test_returns_nones_when_missing(self):
        gdf = _make_gdf(["a", "b"])
        result = _extract_column_array(gdf, "names", 2)
        assert len(result) == 2 and all(v is None for v in result)


class TestExtractLrColumn:
    def test_returns_array_when_present(self):
        gdf = _make_gdf(["a"], names_lr=[[{"between": [0.0, 1.0], "value": "A"}]])
        assert _extract_lr_column(gdf, "names_lr") is not None

    def test_returns_none_when_missing(self):
        gdf = _make_gdf(["a"])
        assert _extract_lr_column(gdf, "names_lr") is None


# ---------------------------------------------------------------------------
# prepare_worker_data
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_pair():
    """A minimal ref/target pair with one candidate."""
    ref = _make_gdf(
        ["ref_1", "ref_2"],
        names=[{"primary": "Main Street"}, {"primary": "Oak Avenue"}],
        classes=["primary", "secondary"],
    )
    target = _make_gdf(
        ["target_1", "target_2"],
        names=[{"primary": "Main St"}, {"primary": "Oak Ave"}],
        classes=["residential", "tertiary"],
        coords_list=[
            [(500000, 4000003), (500100, 4000003)],
            [(500000, 4000008), (500100, 4000008)],
        ],
    )
    candidates = _make_candidates(["ref_1", "ref_2"], ["target_1", "target_2"])
    return ref, target, candidates


class TestPrepareWorkerData:
    """Core tests for prepare_worker_data()."""

    def _run(self, ref, target, candidates, **kwargs):
        defaults = dict(ref_id_column="id", target_id_column="id", n_jobs=1)
        defaults.update(kwargs)
        return prepare_worker_data(candidates=candidates, reference=ref, target=target, **defaults)

    def test_returns_named_tuple(self, simple_pair):
        result = self._run(*simple_pair)
        assert isinstance(result, WorkerDataResult)

    @pytest.mark.parametrize(
        "key",
        [
            "ref_geoms_full",
            "target_geoms_full",
            "ref_names",
            "target_names",
            "ref_classes",
            "target_classes",
            "ref_subclasses",
            "target_subclasses",
            "ref_ids",
            "target_ids",
            "ref_names_lr",
            "target_names_lr",
            "ref_oneway_lr",
            "target_oneway_lr",
            "ref_speed_limit_kph_lr",
            "target_speed_limit_kph_lr",
            "aligned_endpoint_features",
            "ref_topology_full",
            "target_topology_full",
            "ref_graphlet_data",
            "target_graphlet_data",
            "ref_sibling_context_full",
            "target_sibling_context_full",
            "alignments",
        ],
    )
    def test_worker_data_contains_key(self, simple_pair, key):
        """Every expected key must be present in worker_data."""
        result = self._run(*simple_pair)
        assert key in result.worker_data

    def test_geometry_array_lengths(self, simple_pair):
        ref, target, candidates = simple_pair
        wd = self._run(ref, target, candidates).worker_data
        assert len(wd["ref_geoms_full"]) == len(ref)
        assert len(wd["target_geoms_full"]) == len(target)

    def test_id_arrays(self, simple_pair):
        wd = self._run(*simple_pair).worker_data
        assert list(wd["ref_ids"]) == ["ref_1", "ref_2"]
        assert list(wd["target_ids"]) == ["target_1", "target_2"]

    def test_lr_columns_none_when_absent(self, simple_pair):
        """LR columns should be None when the GDF doesn't have them."""
        wd = self._run(*simple_pair).worker_data
        assert wd["ref_names_lr"] is None
        assert wd["target_names_lr"] is None

    def test_lr_columns_present_when_available(self, simple_pair):
        ref, target, candidates = simple_pair
        ref = ref.copy()
        ref["names_lr"] = [
            [{"between": [0.0, 1.0], "value": "Main Street"}],
            [{"between": [0.0, 1.0], "value": "Oak Avenue"}],
        ]
        wd = self._run(ref, target, candidates).worker_data
        assert wd["ref_names_lr"] is not None and len(wd["ref_names_lr"]) == 2
        assert wd["target_names_lr"] is None  # target unchanged

    def test_unique_indices(self, simple_pair):
        result = self._run(*simple_pair)
        assert result.unique_ref_indices == {0, 1}
        assert result.unique_target_indices == {0, 1}

    def test_topology_keyed_by_index(self, simple_pair):
        """Topology features should be keyed by DataFrame index."""
        result = self._run(*simple_pair)
        for idx in result.unique_ref_indices:
            assert idx in result.worker_data["ref_topology_full"]
            assert "from_degree" in result.worker_data["ref_topology_full"][idx]
        for idx in result.unique_target_indices:
            assert idx in result.worker_data["target_topology_full"]

    def test_alignments_in_both_places(self, simple_pair):
        """alignments should appear in worker_data and as a top-level result."""
        result = self._run(*simple_pair)
        assert result.worker_data["alignments"] is result.alignments

    def test_missing_optional_columns(self):
        """Works when names, class, subclass columns are all absent."""
        ref = _make_gdf(["ref_1"])
        target = _make_gdf(["target_1"], coords_list=[[(500000, 4000003), (500100, 4000003)]])
        candidates = _make_candidates(["ref_1"], ["target_1"])
        wd = prepare_worker_data(
            candidates=candidates,
            reference=ref,
            target=target,
            ref_id_column="id",
            target_id_column="id",
            n_jobs=1,
        ).worker_data
        assert all(v is None for v in wd["ref_names"])
        assert all(v is None for v in wd["ref_classes"])


class TestPrepareWorkerDataParity:
    """Regression: worker_data must contain exactly the expected keys.

    This test catches the original bug where compute_features_only() was
    missing 6 LR columns that score_candidates() had.
    """

    REQUIRED_KEYS = frozenset(
        {
            "ref_geoms_full",
            "target_geoms_full",
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
            "ref_topology_full",
            "target_topology_full",
            "target_topo_connectors",
            "target_topo_node_features",
            "target_overture_connectors",
            "ref_graphlet_data",
            "target_graphlet_data",
            "ref_sibling_context_full",
            "target_sibling_context_full",
            "ref_node_to_segments",
            "target_node_to_segments",
            "alignments",
        }
    )

    def test_exact_key_set(self):
        ref = _make_gdf(["ref_1"], names=[{"primary": "Test"}], classes=["primary"])
        target = _make_gdf(
            ["t_1"],
            names=[{"primary": "Test"}],
            classes=["res"],
            coords_list=[[(500000, 4000003), (500100, 4000003)]],
        )
        candidates = _make_candidates(["ref_1"], ["t_1"])
        result = prepare_worker_data(
            candidates=candidates,
            reference=ref,
            target=target,
            ref_id_column="id",
            target_id_column="id",
            n_jobs=1,
        )
        actual = set(result.worker_data.keys())
        assert actual == self.REQUIRED_KEYS, (
            f"Missing: {self.REQUIRED_KEYS - actual}, Extra: {actual - self.REQUIRED_KEYS}"
        )
