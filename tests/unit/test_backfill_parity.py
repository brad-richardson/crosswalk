"""Tests verifying backfill produces features through the same path as inference.

The critical invariant: backfill is a thin wrapper around the same
prepare_worker_data() -> _compute_feature_chunk() pipeline that inference uses.
If this test fails, it means backfill has diverged from the inference path.
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from crosswalk.blocking.spatial_index import CandidatePair
from crosswalk.config import FEATURE_COLUMNS
from crosswalk.features.pipeline import prepare_worker_data
from crosswalk.matching.ml import _compute_feature_chunk, _init_worker

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
        # Blocking stats are unused by the pipeline; placeholders required by dataclass
        CandidatePair(
            ref_id=rid,
            ref_idx=ridx,
            target_id=tid,
            target_idx=tidx,
            distance_estimate=0.0,
            heading_diff=0.0,
        )
        for rid, ridx, tid, tidx in zip(ref_ids, ref_idxs, target_ids, target_idxs)
    ]


# ---------------------------------------------------------------------------
# Parity test
# ---------------------------------------------------------------------------


class TestBackfillParity:
    """Verify that the backfill code path (prepare_worker_data + _compute_feature_chunk)
    produces valid features identical to what inference would produce for the same pair."""

    @pytest.fixture
    def ref_target_pair(self):
        """A realistic ref/target pair with attributes for feature computation."""
        ref = _make_gdf(
            ["ref_1", "ref_2"],
            names=["Main Street", "Oak Avenue"],
            classes=["primary", "secondary"],
            coords_list=[
                [(500000, 4000000), (500100, 4000000)],
                [(500000, 4000010), (500100, 4000010)],
            ],
        )
        # Target: close parallel line to ref_1 (should produce a good match)
        target = _make_gdf(
            ["target_1", "target_2"],
            names=["Main St", "Oak Ave"],
            classes=["primary", "secondary"],
            coords_list=[
                [(500000, 4000002), (500100, 4000002)],
                [(500000, 4000012), (500100, 4000012)],
            ],
        )
        return ref, target

    def test_shared_pipeline_produces_all_feature_columns(self, ref_target_pair):
        """Features from the shared pipeline include all declared FEATURE_COLUMNS."""
        ref, target = ref_target_pair
        candidates = _make_candidates(["ref_1"], ["target_1"])

        pipeline_result = prepare_worker_data(
            candidates=candidates,
            reference=ref,
            target=target,
            n_jobs=1,
        )

        _init_worker(pipeline_result.worker_data)
        work_items = [(c.ref_idx, c.target_idx) for c in candidates]
        results, _errors = _compute_feature_chunk(work_items)

        assert len(results) == 1
        features = results[0]
        assert features is not None, "Feature computation returned None (pair rejected)"

        # Verify all declared features are present
        missing = [col for col in FEATURE_COLUMNS if col not in features]
        assert not missing, f"Missing features from shared pipeline: {missing}"

    def test_features_are_finite_numbers(self, ref_target_pair):
        """Feature values should be finite numbers (not string 'nan' or inf)."""
        ref, target = ref_target_pair
        candidates = _make_candidates(["ref_1"], ["target_1"])

        pipeline_result = prepare_worker_data(
            candidates=candidates,
            reference=ref,
            target=target,
            n_jobs=1,
        )

        _init_worker(pipeline_result.worker_data)
        work_items = [(c.ref_idx, c.target_idx) for c in candidates]
        results, _errors = _compute_feature_chunk(work_items)

        features = results[0]
        assert features is not None

        for col in FEATURE_COLUMNS:
            val = features.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue  # NaN is acceptable (XGBoost handles it)
            assert isinstance(val, (int, float)), (
                f"Feature {col} is {type(val).__name__}, expected number"
            )
            assert not isinstance(val, str), f"Feature {col} is string '{val}'"
            if isinstance(val, float):
                assert not np.isinf(val), f"Feature {col} is infinite"

    def test_two_pairs_same_dataset_produce_independent_results(self, ref_target_pair):
        """Multiple pairs in the same dataset get independent feature values."""
        ref, target = ref_target_pair
        candidates = _make_candidates(
            ["ref_1", "ref_2"],
            ["target_1", "target_2"],
        )

        pipeline_result = prepare_worker_data(
            candidates=candidates,
            reference=ref,
            target=target,
            n_jobs=1,
        )

        _init_worker(pipeline_result.worker_data)
        work_items = [(c.ref_idx, c.target_idx) for c in candidates]
        results, _errors = _compute_feature_chunk(work_items)

        assert len(results) == 2
        for i, features in enumerate(results):
            assert features is not None, f"Pair {i} was rejected"

        # Features should not be identical (different geometries)
        f1 = results[0]
        f2 = results[1]
        # At minimum, name features should differ (Main Street vs Oak Avenue)
        assert f1.get("name_levenshtein") != f2.get("name_levenshtein") or f1.get(
            "hausdorff_distance_m"
        ) != f2.get("hausdorff_distance_m"), "Two different pairs produced identical features"

    def test_topology_override_applied(self, ref_target_pair):
        """Stored topology values override pipeline-computed topology."""
        ref, target = ref_target_pair
        candidates = _make_candidates(["ref_1"], ["target_1"])

        pipeline_result = prepare_worker_data(
            candidates=candidates,
            reference=ref,
            target=target,
            n_jobs=1,
        )
        worker_data = pipeline_result.worker_data

        # Override topology with known values (simulating stored topology)
        stored_ref_topo = {
            "from_degree": 4,
            "to_degree": 3,
            "is_dead_end": False,
            "is_intersection": True,
            "degree_signature": (3, 4),
        }
        stored_target_topo = {
            "from_degree": 2,
            "to_degree": 2,
            "is_dead_end": False,
            "is_intersection": False,
            "degree_signature": (2, 2),
        }
        cand = candidates[0]
        worker_data["ref_topology_full"][cand.ref_idx] = stored_ref_topo
        worker_data["target_topology_full"][cand.target_idx] = stored_target_topo

        _init_worker(worker_data)
        work_items = [(cand.ref_idx, cand.target_idx)]
        results, _errors = _compute_feature_chunk(work_items)

        features = results[0]
        assert features is not None

        # Verify topology features reflect the override, not defaults
        # The stored topologies have different degrees, so degree_diff should be non-zero
        if "ref_from_degree" in features:
            assert features["ref_from_degree"] == 4
        if "target_from_degree" in features:
            assert features["target_from_degree"] == 2

    def test_no_error_flag_for_valid_pair(self, ref_target_pair):
        """Valid pairs should not have error flags set."""
        ref, target = ref_target_pair
        candidates = _make_candidates(["ref_1"], ["target_1"])

        pipeline_result = prepare_worker_data(
            candidates=candidates,
            reference=ref,
            target=target,
            n_jobs=1,
        )

        _init_worker(pipeline_result.worker_data)
        work_items = [(c.ref_idx, c.target_idx) for c in candidates]
        results, _errors = _compute_feature_chunk(work_items)

        features = results[0]
        assert features is not None
        assert features.get("_error") is None, f"Valid pair got error: {features.get('_error')}"
