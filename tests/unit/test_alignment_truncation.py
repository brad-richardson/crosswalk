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
2. ``test_sidecar_geometry_lookup_drops_duplicate_ids`` (XFAIL): documents the
   actual latent defect -- ``dict(zip(reference[id], reference.geometry))``
   silently collapses reference rows that share a GERS id, so the geometry
   serialized into the sidecar can differ from the one that was scored.
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
    centered on the geometries so that lengths and distances are in meters.
    """
    from pyproj import CRS, Transformer

    ref = shape(ref_gj)
    tgt = shape(tgt_gj)
    coords = list(ref.coords) + list(tgt.coords)
    clon = sum(c[0] for c in coords) / len(coords)
    clat = sum(c[1] for c in coords) / len(coords)
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


@pytest.mark.xfail(
    reason=(
        "Latent defect: _export_groups_sidecar builds its geometry lookup with "
        "dict(zip(reference[id], reference.geometry)), which silently drops all "
        "but one row when a reference id is duplicated (e.g. an Overture segment "
        "split into multiple edges that share one GERS id). The scored alignment "
        "fractions then describe a different geometry than the one serialized. "
        "Fixing requires threading the scored edge identity/geometry through "
        "MatchResult into the sidecar rather than re-looking-up by id."
    ),
    strict=True,
)
def test_sidecar_geometry_lookup_drops_duplicate_ids():
    """Encodes the failing behavior at the root of the display truncation.

    Two reference edges share GERS id ``dup`` (as happens when a long Overture
    segment is split at connectors). A correct lookup must be able to recover
    the specific edge that was scored; the current dict-zip cannot -- it keeps
    only the last geometry for the id.
    """
    edges = gpd.GeoDataFrame(
        {"id": ["dup", "dup"]},
        geometry=[
            LineString([(0, 0), (100, 0)]),  # edge A (scored for pair 1)
            LineString([(100, 0), (162, 0)]),  # edge B (kept by dict-zip)
        ],
        crs="EPSG:4326",
    )

    # Current sidecar geometry-lookup construction.
    ref_geom_lookup = dict(zip(edges["id"], edges.geometry))

    # A robust lookup must preserve every edge geometry for a duplicated id.
    # dict-zip collapses them to one, so this assertion fails (xfail=strict).
    recovered = ref_geom_lookup["dup"]
    assert recovered.length == pytest.approx(100.0), (
        "sidecar lookup returned the wrong edge geometry for a duplicated id: "
        f"got length {recovered.length}"
    )
