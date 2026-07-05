"""Unit tests for the bridge-table factory (M4).

Covers: pair discovery + release derivation, manifest staleness-key logic,
scored-candidate cache round-trip / key derivation, and churn-delta computation.
These use synthetic fixtures only (no model, no real stitching), so they are fast
and deterministic.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from matcher.factory import discover_pairs, resolve_release
from matcher.factory.delta import compute_delta
from matcher.factory.discovery import DatasetPair, read_release_from_meta
from matcher.factory.manifest import (
    Manifest,
    compute_full_key,
    compute_optimize_key,
    compute_score_key,
    file_fingerprint,
)
from matcher.factory.runner import FactoryPaths, build_keys, is_up_to_date
from matcher.factory.scored_cache import read_scored_cache, write_scored_cache
from matcher.matching.types import MatchDecision, MatchResult


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def _write_parquet(path, ids):
    import geopandas as gpd
    from shapely.geometry import LineString

    gdf = gpd.GeoDataFrame(
        {"id": ids},
        geometry=[LineString([(0, i), (1, i)]) for i in range(len(ids))],
        crs="EPSG:4326",
    )
    gdf.to_parquet(path)


def _make_triple(raw_dir, name, release="2026-01-21.0"):
    """Create a target + overture segments/connectors triple with a meta.yaml."""
    seg = raw_dir / f"{name}_overture_segments_v1.0.parquet"
    conn = raw_dir / f"{name}_overture_connectors_v1.0.parquet"
    tgt = raw_dir / f"{name}_v1.0.parquet"
    _write_parquet(seg, ["r1", "r2"])
    _write_parquet(conn, ["c1"])
    _write_parquet(tgt, ["t1"])
    if release is not None:
        (raw_dir / f"{name}_overture_segments_v1.0.parquet.meta.yaml").write_text(
            f"release: {release}\nsource: overture\n"
        )
    return DatasetPair(
        name=name, reference_path=seg, target_path=tgt, connectors_path=conn, release=release
    )


def test_discover_pairs_finds_triple(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _make_triple(raw, "xx_test_roads")
    pairs = discover_pairs(raw_dir=raw, names=["xx_test_roads"])
    assert len(pairs) == 1
    p = pairs[0]
    assert p.name == "xx_test_roads"
    assert p.release == "2026-01-21.0"
    assert p.has_connectors


def test_discover_pairs_skips_missing_target(tmp_path):
    """A missing local target must be skipped, not raise (batch isolation)."""
    raw = tmp_path / "raw"
    raw.mkdir()
    # Only overture files, no local target.
    _write_parquet(raw / "yy_only_overture_segments_v1.0.parquet", ["r1"])
    pairs = discover_pairs(raw_dir=raw, names=["yy_only"])
    assert pairs == []


def test_read_release_from_meta_missing(tmp_path):
    seg = tmp_path / "zz_overture_segments_v1.0.parquet"
    seg.write_bytes(b"")
    assert read_release_from_meta(seg) is None


def test_resolve_release_precedence(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pair = _make_triple(raw, "xx_test_roads", release="2026-01-21.0")
    assert resolve_release(pair) == "2026-01-21.0"
    assert resolve_release(pair, override="test2") == "test2"

    no_rel = DatasetPair("nr", pair.reference_path, pair.target_path, None, release=None)
    with pytest.raises(ValueError):
        resolve_release(no_rel)
    assert resolve_release(no_rel, override="forced") == "forced"


# --------------------------------------------------------------------------
# Manifest / staleness keys
# --------------------------------------------------------------------------
def test_score_key_ignores_optimize_settings(tmp_path):
    """score_key must change with inputs/model/buffer, not optimize settings."""
    ref_fp = {"name": "r", "size": 1, "mtime_ns": 10}
    tgt_fp = {"name": "t", "size": 2, "mtime_ns": 20}
    model_fp = {"name": "m", "size": 3, "mtime_ns": 30}

    k1 = compute_score_key(ref_fp, tgt_fp, model_fp, 75.0)
    k2 = compute_score_key(ref_fp, tgt_fp, model_fp, 75.0)
    assert k1 == k2  # deterministic

    # Different buffer -> different score_key
    assert compute_score_key(ref_fp, tgt_fp, model_fp, 50.0) != k1
    # Different input fingerprint -> different score_key
    assert compute_score_key({**ref_fp, "size": 999}, tgt_fp, model_fp, 75.0) != k1
    # Different method -> different score_key
    assert compute_score_key(ref_fp, tgt_fp, model_fp, 75.0, method="other") != k1
    # Different cache schema version -> different score_key (layout invalidation)
    assert compute_score_key(ref_fp, tgt_fp, model_fp, 75.0, cache_schema_version=999) != k1


def test_optimize_and_full_key(tmp_path):
    snap_a = {"resolver_prune_enabled": True, "bridge_min_confidence": 0.5}
    snap_b = {"resolver_prune_enabled": False, "bridge_min_confidence": 0.5}
    oa = compute_optimize_key(snap_a)
    ob = compute_optimize_key(snap_b)
    assert oa != ob
    # min_confidence (optimizer arg) participates in optimize_key
    assert compute_optimize_key(snap_a, min_confidence=0.2) != oa

    score_key = "abc123"
    full_a = compute_full_key(score_key, oa)
    full_b = compute_full_key(score_key, ob)
    assert full_a != full_b  # optimize change flips full_key
    # Same inputs -> same full_key
    assert compute_full_key(score_key, oa) == full_a


def test_settings_snapshot_covers_decision_knobs():
    """Every optimize-phase decision knob must be in the snapshot (else a change
    to it would wrongly skip via full_key). optimizer_review_threshold is the one
    the adversarial review caught missing."""
    from matcher.factory.manifest import settings_snapshot

    snap = settings_snapshot()
    for knob in (
        "optimizer_review_threshold",
        "bridge_min_confidence",
        "optimizer_glue_min_confidence",
        "resolver_prune_enabled",
        "resolver_prune_overrides",
        "contiguity_tolerance_m",
    ):
        assert knob in snap, f"{knob} missing from settings_snapshot()"


def test_is_up_to_date(tmp_path):
    mpath = tmp_path / "manifest.json"
    bpath = tmp_path / "bridge.parquet"
    # No files yet
    assert not is_up_to_date(mpath, bpath, "key1")

    bpath.write_bytes(b"x")
    m = Manifest(dataset="d", release="r", full_key="key1")
    m.write(mpath)
    assert is_up_to_date(mpath, bpath, "key1")
    assert not is_up_to_date(mpath, bpath, "key2")  # key mismatch

    # Missing bridge -> stale even if manifest matches
    bpath.unlink()
    assert not is_up_to_date(mpath, bpath, "key1")


def test_build_keys_reoptimize_semantics(tmp_path, monkeypatch):
    """Changing only an optimize setting keeps score_key, changes full_key."""
    raw = tmp_path / "raw"
    raw.mkdir()
    pair = _make_triple(raw, "xx_test_roads")

    from matcher.config import settings

    monkeypatch.setattr(settings, "resolver_prune_enabled", True)
    k_a = build_keys(pair, 75.0)
    monkeypatch.setattr(settings, "resolver_prune_enabled", False)
    k_b = build_keys(pair, 75.0)

    assert k_a["score_key"] == k_b["score_key"]  # scores unaffected
    assert k_a["optimize_key"] != k_b["optimize_key"]
    assert k_a["full_key"] != k_b["full_key"]

    # min_confidence (optimizer arg) flips optimize_key but not score_key
    k_c = build_keys(pair, 75.0, min_confidence=0.3)
    assert k_c["score_key"] == k_b["score_key"]
    assert k_c["optimize_key"] != k_b["optimize_key"]
    # method flips score_key
    k_d = build_keys(pair, 75.0, method="other")
    assert k_d["score_key"] != k_b["score_key"]


def test_manifest_round_trip(tmp_path):
    m = Manifest(dataset="d", release="2026-01-21.0", n_matched=10, full_key="fk")
    p = tmp_path / "manifest.json"
    m.write(p)
    loaded = Manifest.read(p)
    assert loaded.dataset == "d"
    assert loaded.n_matched == 10
    assert loaded.full_key == "fk"


def test_manifest_read_tolerates_unknown_keys(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"dataset": "d", "release": "r", "future_field": 1}))
    m = Manifest.read(p)
    assert m.dataset == "d"


# --------------------------------------------------------------------------
# Scored-candidate cache
# --------------------------------------------------------------------------
def _sample_results():
    return [
        MatchResult(
            ref_id="gers_a",
            target_id="local_1",
            decision=MatchDecision.MATCH,
            confidence=0.91,
            score_breakdown={"geom": 0.8},
            features={"length_ratio": 1.2, "match_type": "1:1", "nan_feat": float("nan")},
            ref_idx=5,
            target_idx=3,
            gers_start_frac=0.0,
            gers_end_frac=1.0,
            local_start_frac=None,
            local_end_frac=None,
        ),
        MatchResult(
            ref_id="gers_b",
            target_id="local_2",
            decision=MatchDecision.REVIEW,
            confidence=0.42,
            score_breakdown={},
            features={"length_ratio": 0.5},
            ref_idx=None,
            target_idx=None,
        ),
    ]


def test_scored_cache_round_trip(tmp_path):
    results = _sample_results()
    path = tmp_path / "scored.parquet"
    n = write_scored_cache(results, path)
    assert n == 2

    loaded = read_scored_cache(path)
    assert len(loaded) == 2

    a0, b0 = results
    a1, b1 = loaded
    assert a1.ref_id == a0.ref_id
    assert a1.target_id == a0.target_id
    assert a1.decision == a0.decision
    assert a1.confidence == pytest.approx(a0.confidence)
    assert a1.ref_idx == 5 and a1.target_idx == 3
    assert a1.features["length_ratio"] == pytest.approx(1.2)
    assert a1.features["match_type"] == "1:1"
    # NaN feature preserved as NaN
    import math

    assert math.isnan(a1.features["nan_feat"])
    assert a1.gers_start_frac == pytest.approx(0.0)
    assert a1.local_start_frac is None

    # None positional indices survive
    assert b1.ref_idx is None and b1.target_idx is None
    assert b1.decision == MatchDecision.REVIEW


def test_scored_cache_preserves_order(tmp_path):
    results = _sample_results()[::-1]
    path = tmp_path / "scored.parquet"
    write_scored_cache(results, path)
    loaded = read_scored_cache(path)
    assert [r.target_id for r in loaded] == [r.target_id for r in results]


# --------------------------------------------------------------------------
# Delta
# --------------------------------------------------------------------------
def _write_bridge(path, rows):
    """rows: list of (local_id, gers_id)."""
    df = pd.DataFrame(rows, columns=["local_id", "gers_id"])
    df.to_parquet(path)


def test_compute_delta_classifies(tmp_path):
    a = tmp_path / "from.parquet"
    b = tmp_path / "to.parquet"
    # local_1: same (a->x in both)
    # local_2: changed (a->y then a->z)
    # local_3: lost (only in from)
    # local_4: gained (only in to)
    _write_bridge(a, [("local_1", "x"), ("local_2", "y"), ("local_3", "w")])
    _write_bridge(b, [("local_1", "x"), ("local_2", "z"), ("local_4", "v")])

    result = compute_delta("ds", a, b, "r1", "r2")
    s = result.summary
    assert s["same"] == 1
    assert s["changed"] == 1
    assert s["lost"] == 1
    assert s["gained"] == 1

    cats = dict(zip(result.details["local_id"], result.details["category"]))
    assert cats["local_2"] == "changed"
    assert cats["local_3"] == "lost"
    assert cats["local_4"] == "gained"
    assert "local_1" not in cats  # 'same' rows excluded from details

    md = result.to_markdown()
    assert "GERS churn delta" in md
    assert "changed" in md


def test_compute_delta_excludes_review_rows(tmp_path):
    """REVIEW-decision bridge rows must not count as matches in the delta —
    the pipeline routes them to unmatched, so counting them would misreport
    review-band flapping as GERS churn."""
    a = tmp_path / "from.parquet"
    b = tmp_path / "to.parquet"
    pd.DataFrame(
        {
            "local_id": ["l1", "l2"],
            "gers_id": ["x", "y"],
            "match_decision": ["match", "review"],
        }
    ).to_parquet(a)
    pd.DataFrame(
        {
            "local_id": ["l1", "l2"],
            "gers_id": ["x", "y"],
            "match_decision": ["match", "match"],
        }
    ).to_parquet(b)

    result = compute_delta("ds", a, b, "r1", "r2")
    # l1: same. l2: review->match counts as GAINED (was not a match before).
    assert result.summary["same"] == 1
    assert result.summary["gained"] == 1
    assert result.summary["changed"] == 0
    assert result.summary["lost"] == 0


def test_compute_delta_multi_gers_set(tmp_path):
    """A local id matched to multiple GERS is compared as a set."""
    a = tmp_path / "from.parquet"
    b = tmp_path / "to.parquet"
    _write_bridge(a, [("l1", "x"), ("l1", "y")])
    _write_bridge(b, [("l1", "y"), ("l1", "x")])  # same set, different order
    result = compute_delta("ds", a, b, "r1", "r2")
    assert result.summary["same"] == 1
    assert result.summary["changed"] == 0


# --------------------------------------------------------------------------
# FactoryPaths layout
# --------------------------------------------------------------------------
def test_factory_paths_layout(tmp_path):
    paths = FactoryPaths(root=tmp_path / "factory")
    b = paths.bridge("2026-01-21.0", "us_frisco_trails")
    assert b.name == "bridge.parquet"
    assert b.parent.name == "dataset=us_frisco_trails"
    assert b.parent.parent.name == "release=2026-01-21.0"
    assert paths.groups("r", "d").name == "groups.json"
    assert paths.manifest("r", "d").name == "manifest.json"


def test_file_fingerprint(tmp_path):
    f = tmp_path / "x.parquet"
    f.write_bytes(b"hello")
    fp = file_fingerprint(f)
    assert fp["name"] == "x.parquet"
    assert fp["size"] == 5
    assert "mtime_ns" in fp
