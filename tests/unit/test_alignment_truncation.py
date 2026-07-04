"""Regression tests for the alignment-truncation investigation.

Context
-------
On the live stitching-review map, pairs of near-identical-length, cleanly
parallel road segments (American Legion Highway dual carriageway, group
``701d491e``) appeared to receive alignment intervals that stop far short of
full coverage, e.g. edge R4-T2 stored ``gers=[0.417, 0.999]`` against
``local=[0.0, 1.0]`` for two ~160 m segments.

The investigation (see ``research/alignment_truncation_investigation.md``)
established that ``linestring_alignment`` is NOT the culprit: run directly on
the geometries stored in the batch it returns full coverage on both sides at
every historical code revision. The stored truncation is a geometry-identity
mismatch in the sidecar/batch serialization, not a divergence-truncation bug.

These tests lock in both facts:

1. ``test_real_parallel_carriageway_*`` (PASSING): the real fixture geometries
   align to near-full coverage. This guards the divergence thresholds against a
   future change that would wrongly truncate near-coincident parallel roads
   (the failure mode originally hypothesized).
2. ``test_sidecar_uses_scored_geometry_for_duplicate_ids`` (PASSING): locks in
   the fix -- ``_export_groups_sidecar`` resolves each edge's geometry by its
   scored positional index (``MatchResult.ref_idx``/``target_idx``) instead of a
   global ``dict(zip(reference[id], reference.geometry))`` that silently collapses
   reference rows sharing a GERS id and could serialize the wrong geometry.
"""

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, shape
from shapely.ops import transform

from matcher.features.alignment import (
    _create_local_equidistant_crs,
    linestring_alignment,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "alignment_truncation_701d491e.json"


@pytest.fixture(scope="module")
def truncation_fixture():
    return json.loads(FIXTURE.read_text())


def _project_pair(ref_gj, tgt_gj):
    """Project a WGS84 GeoJSON ref/target pair to a shared local AEQD CRS.

    Mirrors ``compute_alignment_batch``: a local azimuthal-equidistant CRS
    centered on the REFERENCE geometry's centroid (production centers on the
    reference geometries only — see alignment.py `_compute_centroid(ref_geoms)`),
    so that lengths and distances are in meters.
    """
    from pyproj import CRS, Transformer

    ref = shape(ref_gj)
    tgt = shape(tgt_gj)
    c = ref.centroid
    clon, clat = c.x, c.y
    local_crs = _create_local_equidistant_crs(clon, clat)
    tr = Transformer.from_crs(CRS.from_epsg(4326), local_crs, always_xy=True)
    return (
        transform(tr.transform, ref),
        transform(tr.transform, tgt),
    )


@pytest.mark.parametrize("edge_key", ["R4_T2", "R7_T6"])
def test_real_parallel_carriageway_aligns_to_full_coverage(truncation_fixture, edge_key):
    """Real near-equal-length parallel carriageways align to near-full coverage.

    Guards the divergence thresholds: a constant lateral offset between parallel
    carriageways must NOT be treated as divergence. If a future change to
    DIVERGENCE_* constants (or the truncation logic) starts cutting these
    intervals, this test fails.
    """
    edge = truncation_fixture[edge_key]
    ref, tgt = _project_pair(edge["ref_geometry"], edge["target_geometry"])

    # Sanity: the two carriageways are near-equal length.
    assert min(ref.length, tgt.length) / max(ref.length, tgt.length) > 0.95

    result = linestring_alignment(ref, tgt)

    ref_span = result.overture_end_frac - result.overture_start_frac
    tgt_span = result.dataset_end_frac - result.dataset_start_frac

    # Both sides should cover essentially the whole segment (near-coincident,
    # ~1 m apart). Allow a small margin for endpoint sampling.
    assert ref_span > 0.95, f"{edge_key} ref span truncated: {result}"
    assert tgt_span > 0.95, f"{edge_key} target span truncated: {result}"


def test_stored_truncation_is_not_reproducible_from_stored_geometry(truncation_fixture):
    """The stored truncated interval cannot be reproduced from the stored geometry.

    Demonstrates the diagnosis: the stored edge for R4-T2 records
    ``gers=[0.417, 0.999]`` with ``local=[0.0, 1.0]`` -- 100% of the target
    mapping onto only ~58% of the ref -- which is geometrically impossible for
    two near-equal-length segments. Recomputing from the stored geometry yields
    full coverage instead, proving the stored fracs were computed against a
    *different* reference geometry than the one serialized into the batch.
    """
    edge = truncation_fixture["R4_T2"]
    stored = edge["stored_edge"]
    stored_ref_span = stored["gers_end_frac"] - stored["gers_start_frac"]
    stored_tgt_span = stored["local_end_frac"] - stored["local_start_frac"]

    # Stored data: full target, truncated ref -> inconsistent for equal lengths.
    assert stored_tgt_span > 0.99
    assert stored_ref_span < 0.85

    ref, tgt = _project_pair(edge["ref_geometry"], edge["target_geometry"])
    result = linestring_alignment(ref, tgt)
    recomputed_ref_span = result.overture_end_frac - result.overture_start_frac

    # Recomputed span is full -- the stored truncation does not come from
    # aligning these two geometries.
    assert recomputed_ref_span > 0.95
    assert recomputed_ref_span - stored_ref_span > 0.3


def test_dict_zip_lookup_collapses_duplicate_ids():
    """Documents the root-cause mechanism: id-keyed dict-zip drops co-id rows.

    Two reference edges share GERS id ``dup`` (as happens when a long Overture
    segment is split at connectors). ``dict(zip(...))`` keeps only the LAST
    geometry for the id, so an id-keyed lookup cannot recover the edge that was
    actually scored. The sidecar fix resolves geometry by positional index
    instead -- see ``test_sidecar_uses_scored_geometry_for_duplicate_ids``.
    """
    edges = gpd.GeoDataFrame(
        {"id": ["dup", "dup"]},
        geometry=[
            LineString([(0, 0), (100, 0)]),  # edge A (scored, length 100)
            LineString([(100, 0), (162, 0)]),  # edge B (kept by dict-zip)
        ],
        crs="EPSG:4326",
    )

    ref_geom_lookup = dict(zip(edges["id"], edges.geometry))
    # dict-zip collapses to the last row: length 62, NOT the scored edge (100).
    assert ref_geom_lookup["dup"].length == pytest.approx(62.0)


def test_sidecar_uses_scored_geometry_for_duplicate_ids(tmp_path):
    """The sidecar serializes the SCORED edge geometry for a duplicated id.

    Reproduces the display-truncation collapse and verifies the fix. Two
    reference rows share GERS id ``dup``:

    - positional index 0 -> the geometry that was scored (length 100)
    - positional index 1 -> a decoy that an id-keyed ``dict(zip)`` would keep
      (length 62), because it is the LAST row for ``dup``.

    The match group scores against index 0, so a correct sidecar must serialize
    the length-100 geometry. The old id-keyed lookup would emit length 62 (the
    wrong duplicate) -- the ``bold sub-line stops partway`` symptom.
    """
    from matcher.matching.types import MatchDecision, MatchResult
    from matcher.pipeline.runner import (
        _export_groups_sidecar,
        groups_sidecar_path,
    )

    reference = gpd.GeoDataFrame(
        {"id": ["dup", "dup"]},
        geometry=[
            LineString([(0, 0), (100, 0)]),  # idx 0: scored, length 100
            LineString([(100, 0), (162, 0)]),  # idx 1: decoy, length 62
        ],
        crs="EPSG:4326",
    )
    target = gpd.GeoDataFrame(
        {"id": ["t1", "t2"]},
        geometry=[
            LineString([(0, 1), (50, 1)]),
            LineString([(50, 1), (100, 1)]),
        ],
        crs="EPSG:4326",
    )

    # 1:N group: ref "dup" (scored at index 0) matched to two targets.
    results = [
        MatchResult(
            ref_id="dup",
            target_id="t1",
            decision=MatchDecision.MATCH,
            confidence=0.99,
            score_breakdown={},
            features={},
            ref_idx=0,
            target_idx=0,
        ),
        MatchResult(
            ref_id="dup",
            target_id="t2",
            decision=MatchDecision.MATCH,
            confidence=0.99,
            score_breakdown={},
            features={},
            ref_idx=0,
            target_idx=1,
        ),
    ]

    bridge_path = tmp_path / "bridge.parquet"
    sidecar_path = _export_groups_sidecar(
        results=results,
        optimized=[],
        output_path=bridge_path,
        reference=reference,
        target=target,
        min_confidence=0.5,
    )

    assert sidecar_path == groups_sidecar_path(bridge_path)
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["n_groups"] == 1
    group = sidecar["groups"][0]

    ref_geom = shape(group["ref_geometries"]["dup"])
    assert ref_geom.length == pytest.approx(100.0), (
        "sidecar serialized the wrong duplicate geometry for id 'dup': "
        f"got length {ref_geom.length}, expected the scored edge (100)"
    )
    # Targets resolve by their own positional indices too.
    assert shape(group["target_geometries"]["t1"]).length == pytest.approx(50.0)
    assert shape(group["target_geometries"]["t2"]).length == pytest.approx(50.0)
