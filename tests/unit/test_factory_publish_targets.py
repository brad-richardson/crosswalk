"""Unit tests for target-snapshot publishing (``crosswalk factory publish --targets``).

Covers: license gating (pending_review / unlisted excluded), the persisted
quality-hold gate shared with the bridges path (#370/#338 — see
``publish.py::dataset_quality_hold``), the on-disk key layout
(``targets/dataset=*/snapshot=*/data.parquet`` + ``meta.yaml`` +
``latest.json`` + top-level ``index.json``), snapshot-date resolution (fetch
sidecar vs dataset-yaml + mtime fallback), immutability (refuse overwrite of an
existing snapshot without ``--force``), and dry-run/CLI wiring emits no network
calls. All synthetic and offline — no real R2 credentials, no network, no
pipeline run. Any ``aws`` CLI interaction is monkeypatched.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from crosswalk.cli import app
from crosswalk.factory.licenses import LicenseRegistry
from crosswalk.factory.manifest import Manifest
from crosswalk.factory.publish import BRIDGES_PREFIX, assemble_staging
from crosswalk.factory.publish_sync import (
    R2Config,
    build_aws_sync_argv,
    staged_target_snapshots,
    sync_targets_local,
    sync_targets_r2,
)
from crosswalk.factory.publish_targets import (
    LATEST_JSON_FILENAME,
    TARGET_DATA_FILENAME,
    TARGET_META_FILENAME,
    TARGETS_INDEX_FILENAME,
    TARGETS_PREFIX,
    assemble_targets_staging,
    discover_target_files,
    resolve_snapshot_provenance,
)
from crosswalk.fetch.metadata import FetchMetadata, save_metadata

runner = CliRunner()

BRIDGES_RELEASE = "2026-01-21.0"


def _write_target_parquet(raw_dir: Path, name: str) -> Path:
    """Write a minimal but valid target parquet under `raw_dir`."""
    path = raw_dir / f"{name}_v1.0.parquet"
    pd.DataFrame({"id": ["a", "b"], "geometry_wkt": ["LINESTRING(0 0, 1 1)"] * 2}).to_parquet(
        path, index=False
    )
    return path


def _registry() -> LicenseRegistry:
    return LicenseRegistry(
        {
            "overture": {"attribution": "O", "url": "u", "license": "ODbL-1.0"},
            "datasets": {
                "us_ok_targets": {
                    "status": "approved",
                    "license": "US-PD",
                    "attribution": "Some Agency",
                    "source_url": "https://example.test/api",
                    "display_name": "OK Targets",
                },
                "xx_pending_targets": {
                    "status": "pending_review",
                    "note": "verify terms",
                },
            },
        }
    )


@pytest.fixture
def raw_dir(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    return d


@pytest.fixture
def empty_datasets_dir(tmp_path):
    d = tmp_path / "datasets"
    d.mkdir()
    return d


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def test_discover_target_files_excludes_overture_and_osm(raw_dir):
    _write_target_parquet(raw_dir, "us_ok_targets")
    (raw_dir / "us_ok_overture_segments_v1.0.parquet").write_bytes(b"")
    (raw_dir / "us_ok_overture_connectors_v1.0.parquet").write_bytes(b"")
    (raw_dir / "us_ok_targets_osm_segments_v1.0.parquet").write_bytes(b"")
    (raw_dir / "us_ok_targets_v1.0.parquet.bak").write_bytes(b"")
    (raw_dir / "us_ok_targets_v1.0.parquet.meta.yaml").write_text("fetched_at: '2026-01-01'\n")

    found = discover_target_files(raw_dir)
    assert set(found) == {"us_ok_targets"}


def test_discover_target_files_keeps_target_whose_name_contains_role_tokens(raw_dir):
    """Exclusion keys on the exact role SUFFIX, not a bare ``_osm_`` / ``_overture_``
    substring: a genuine target dataset whose own name embeds those tokens (but
    does not END in a reference role suffix) must NOT be silently dropped."""
    _write_target_parquet(raw_dir, "us_osm_import_roads")
    _write_target_parquet(raw_dir, "us_overture_derived_paths")
    # A real reference file (ends in the role suffix) is still excluded.
    (raw_dir / "us_osm_import_roads_osm_segments_v1.0.parquet").write_bytes(b"")

    found = discover_target_files(raw_dir)
    assert set(found) == {"us_osm_import_roads", "us_overture_derived_paths"}


def test_discover_target_files_empty_dir(tmp_path):
    assert discover_target_files(tmp_path / "does_not_exist") == {}


# --------------------------------------------------------------------------
# License gating + key layout
# --------------------------------------------------------------------------
def test_assembly_publishes_only_licensed(raw_dir, tmp_path, empty_datasets_dir):
    _write_target_parquet(raw_dir, "us_ok_targets")
    _write_target_parquet(raw_dir, "xx_pending_targets")

    staging = tmp_path / "staging"
    report = assemble_targets_staging(
        raw_dir, staging, _registry(), datasets_dir=empty_datasets_dir
    )

    assert report.n_published == 1
    assert report.n_excluded == 1

    ok = next(d for d in report.datasets if d.dataset == "us_ok_targets")
    assert ok.published
    pending = next(d for d in report.datasets if d.dataset == "xx_pending_targets")
    assert not pending.published
    assert "pending_review" in pending.reason

    ds_dir = staging / TARGETS_PREFIX / "dataset=us_ok_targets"
    assert (ds_dir / LATEST_JSON_FILENAME).exists()
    snap_dir = ds_dir / f"snapshot={ok.snapshot}"
    assert (snap_dir / TARGET_DATA_FILENAME).exists()
    assert (snap_dir / TARGET_META_FILENAME).exists()
    # Excluded dataset must NOT have any data copied.
    assert not (staging / TARGETS_PREFIX / "dataset=xx_pending_targets").exists()


def test_unlisted_dataset_excluded(raw_dir, tmp_path, empty_datasets_dir):
    _write_target_parquet(raw_dir, "zz_unlisted_targets")
    report = assemble_targets_staging(
        raw_dir, tmp_path / "staging", _registry(), datasets_dir=empty_datasets_dir
    )
    zz = next(d for d in report.datasets if d.dataset == "zz_unlisted_targets")
    assert not zz.published
    assert "no license registry entry" in zz.reason


# --------------------------------------------------------------------------
# Quality hold (declarative do-not-ship in the dataset YAML; #370/#338)
#
# ``dataset_quality_hold`` (publish.py) is shared by both publish paths. Only
# the bridges path (``assemble_staging``) enforced it before this change —
# ``co_bogota_bike_network`` is license-approved and quality-held, yet its YAML
# hold block says the hold "Blocks publishing (crosswalk factory publish)"
# with no bridges-only qualifier. These tests pin the targets path to the same
# behavior, and (b) pin that the bridges path itself is unchanged.
# --------------------------------------------------------------------------
HOLD = {
    "reason": (
        "cross-mode defect: cycleways matched to parallel road centerlines at 0.82-0.95 confidence"
    ),
    "since": "2026-07-06",
}


def _write_dataset_yaml(datasets_dir, name, *, quality_hold=None):
    """Write a minimal dataset YAML config into the test datasets dir."""
    data = {"name": name, "display_name": f"Display {name}", "type": "bike"}
    if quality_hold is not None:
        data["quality_hold"] = quality_hold
    (datasets_dir / f"{name}.yaml").write_text(yaml.safe_dump(data, sort_keys=False))


def _make_bridges_factory_dataset(factory_root, name, *, release=BRIDGES_RELEASE):
    """Minimal synthetic bridges-path factory output (bridge.parquet + manifest.json)."""
    ds_dir = factory_root / f"release={release}" / f"dataset={name}"
    ds_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "local_id": [f"{name}_0"],
            "gers_id": ["gers-0"],
            "confidence": [0.9],
            "match_type": ["1:1"],
            "match_decision": ["match"],
        }
    ).to_parquet(ds_dir / "bridge.parquet", index=False)
    Manifest(
        dataset=name,
        release=release,
        created_at="2026-07-05T00:00:00+00:00",
        n_reference=1,
        n_target=1,
        n_candidates=1,
        n_matched=1,
        n_review=0,
        n_unmatched=0,
        groups={"n_groups": 1, "n_m_to_n": 0, "n_oversized": 0},
        wall_s=0.1,
    ).write(ds_dir / "manifest.json")
    return ds_dir


def test_quality_hold_excludes_target_snapshot_even_when_license_approved(
    raw_dir, tmp_path, empty_datasets_dir
):
    """(a) The #338 incident this mechanism prevents on the bridges path applies
    identically here: license-approved but quality-held must exclude the TARGET
    snapshot too — the concrete divergence is ``co_bogota_bike_network``, whose
    YAML hold block claims (without a bridges-only qualifier) to block
    ``crosswalk factory publish`` outright."""
    _write_target_parquet(raw_dir, "us_ok_targets")
    _write_dataset_yaml(empty_datasets_dir, "us_ok_targets", quality_hold=HOLD)

    staging = tmp_path / "staging"
    report = assemble_targets_staging(
        raw_dir, staging, _registry(), datasets_dir=empty_datasets_dir
    )

    held = next(d for d in report.datasets if d.dataset == "us_ok_targets")
    assert held.status == "excluded"
    assert held.reason == f"quality hold: {HOLD['reason']} (since 2026-07-06)"
    # The license IS approved — the hold, not the license, is what blocks it.
    assert held.license["approved"] is True
    assert held.quality_hold == HOLD
    assert report.n_published == 0
    # No data copied into the staging tree.
    assert not (staging / TARGETS_PREFIX / "dataset=us_ok_targets").exists()
    # Excluded dataset never appears in the (published-only) index.
    idx = json.loads((staging / TARGETS_PREFIX / TARGETS_INDEX_FILENAME).read_text())
    assert "us_ok_targets" not in idx["datasets"]


def test_quality_hold_takes_precedence_over_pending_target_license(
    raw_dir, tmp_path, empty_datasets_dir
):
    """A held target reports the hold even while its license is still pending, so
    a later license flip can never change its outcome or its stated reason."""
    _write_target_parquet(raw_dir, "xx_pending_targets")
    _write_dataset_yaml(empty_datasets_dir, "xx_pending_targets", quality_hold=HOLD)

    report = assemble_targets_staging(
        raw_dir, tmp_path / "staging", _registry(), datasets_dir=empty_datasets_dir
    )
    d = next(x for x in report.datasets if x.dataset == "xx_pending_targets")
    assert not d.published
    assert d.reason.startswith("quality hold:")
    assert d.license["approved"] is False  # license state still recorded alongside


def test_malformed_quality_hold_still_holds_target_snapshot(raw_dir, tmp_path, empty_datasets_dir):
    """Fail-safe: any truthy quality_hold value holds — a defective dataset must
    never ship on a parsing technicality (mirrors the bridges-path guarantee)."""
    _write_target_parquet(raw_dir, "us_ok_targets")
    (empty_datasets_dir / "us_ok_targets.yaml").write_text(
        "name: us_ok_targets\nquality_hold: true\n"
    )
    report = assemble_targets_staging(
        raw_dir, tmp_path / "staging", _registry(), datasets_dir=empty_datasets_dir
    )
    held = next(d for d in report.datasets if d.dataset == "us_ok_targets")
    assert held.status == "excluded"
    assert held.reason.startswith("quality hold:")


def test_dataset_yaml_without_hold_still_publishes_target(raw_dir, tmp_path, empty_datasets_dir):
    """(c) A YAML config with no quality_hold block changes nothing — an approved,
    un-held dataset still stages exactly as before this change."""
    _write_target_parquet(raw_dir, "us_ok_targets")
    _write_dataset_yaml(empty_datasets_dir, "us_ok_targets")

    report = assemble_targets_staging(
        raw_dir, tmp_path / "staging", _registry(), datasets_dir=empty_datasets_dir
    )
    ok = next(d for d in report.datasets if d.dataset == "us_ok_targets")
    assert ok.published
    assert ok.quality_hold is None
    assert report.n_published == 1


def test_bridges_path_quality_hold_behavior_unchanged(tmp_path, empty_datasets_dir):
    """(b) Pins that this change does not alter the bridges path
    (``publish.assemble_staging``) at all: a license-approved but quality-held
    bridge dataset is still excluded with the hold as its reason, and an
    un-held approved dataset still publishes."""
    factory_root = tmp_path / "factory"
    _make_bridges_factory_dataset(factory_root, "us_ok_roads")
    _make_bridges_factory_dataset(factory_root, "us_held_roads")
    _write_dataset_yaml(empty_datasets_dir, "us_held_roads", quality_hold=HOLD)

    registry = LicenseRegistry(
        {
            "overture": {"attribution": "O", "url": "u", "license": "ODbL-1.0"},
            "datasets": {
                "us_ok_roads": {"status": "approved", "license": "L", "attribution": "A"},
                "us_held_roads": {"status": "approved", "license": "L", "attribution": "A"},
            },
        }
    )
    report = assemble_staging(
        factory_root,
        tmp_path / "staging",
        registry,
        datasets_dir=empty_datasets_dir,
        generated_at="fixed",
    )

    ok = next(d for r in report.releases for d in r.datasets if d.dataset == "us_ok_roads")
    held = next(d for r in report.releases for d in r.datasets if d.dataset == "us_held_roads")
    assert ok.published
    assert not held.published
    assert held.reason == f"quality hold: {HOLD['reason']} (since 2026-07-06)"
    assert held.license["approved"] is True
    assert not (
        tmp_path
        / "staging"
        / BRIDGES_PREFIX
        / f"release={BRIDGES_RELEASE}"
        / "dataset=us_held_roads"
    ).exists()


def test_dataset_filter_restricts_publication(raw_dir, tmp_path, empty_datasets_dir):
    _write_target_parquet(raw_dir, "us_ok_targets")
    reg = LicenseRegistry(
        {
            "overture": {"attribution": "O", "url": "u", "license": "L"},
            "datasets": {
                "us_ok_targets": {"status": "approved", "license": "L", "attribution": "A"},
                "us_other_targets": {"status": "approved", "license": "L", "attribution": "A"},
            },
        }
    )
    _write_target_parquet(raw_dir, "us_other_targets")
    report = assemble_targets_staging(
        raw_dir,
        tmp_path / "staging",
        reg,
        datasets_dir=empty_datasets_dir,
        datasets=["us_ok_targets"],
    )
    assert {d.dataset for d in report.datasets} == {"us_ok_targets"}
    assert report.n_published == 1


def test_latest_json_and_index_json_layout(raw_dir, tmp_path, empty_datasets_dir):
    _write_target_parquet(raw_dir, "us_ok_targets")
    staging = tmp_path / "staging"
    report = assemble_targets_staging(
        raw_dir, staging, _registry(), datasets_dir=empty_datasets_dir
    )
    ok = next(d for d in report.datasets if d.dataset == "us_ok_targets")

    latest = json.loads(
        (staging / TARGETS_PREFIX / "dataset=us_ok_targets" / LATEST_JSON_FILENAME).read_text()
    )
    assert latest == {
        "dataset": "us_ok_targets",
        "latest_snapshot": ok.snapshot,
        "path": f"targets/dataset=us_ok_targets/snapshot={ok.snapshot}/{TARGET_DATA_FILENAME}",
    }

    idx = json.loads((staging / TARGETS_PREFIX / TARGETS_INDEX_FILENAME).read_text())
    assert idx["generated_from"] == str(raw_dir)
    entry = idx["datasets"]["us_ok_targets"]
    assert entry["latest_snapshot"] == ok.snapshot
    assert entry["display_name"] == "OK Targets"
    assert entry["license"] == "US-PD"
    assert entry["attribution"] == "Some Agency"
    assert entry["source_url"] == "https://example.test/api"
    assert entry["size_bytes"] > 0
    # Excluded dataset never appears in the index at all.
    assert "xx_pending_targets" not in idx["datasets"]


def test_index_json_written_even_when_nothing_published(raw_dir, tmp_path, empty_datasets_dir):
    _write_target_parquet(raw_dir, "xx_pending_targets")
    staging = tmp_path / "staging"
    report = assemble_targets_staging(
        raw_dir, staging, _registry(), datasets_dir=empty_datasets_dir
    )
    assert report.n_published == 0
    idx_path = staging / TARGETS_PREFIX / TARGETS_INDEX_FILENAME
    assert idx_path.exists()
    assert json.loads(idx_path.read_text())["datasets"] == {}


# --------------------------------------------------------------------------
# Snapshot-date resolution
# --------------------------------------------------------------------------
def test_snapshot_resolution_prefers_fetch_sidecar(raw_dir, empty_datasets_dir):
    path = _write_target_parquet(raw_dir, "us_ok_targets")
    save_metadata(
        path,
        FetchMetadata(
            fetched_at=datetime(2026, 7, 6, 18, 33, 57, tzinfo=UTC),
            source="arcgis",
            source_url="https://example.test/rest",
            feature_count=42,
            id_column="OBJECTID",
            bbox=(1.0, 2.0, 3.0, 4.0),
        ),
    )
    prov = resolve_snapshot_provenance(path, "us_ok_targets", datasets_dir=empty_datasets_dir)
    assert prov.snapshot == "2026-07-06"  # first 10 chars of fetched_at
    assert prov.source == "arcgis"
    assert prov.source_url == "https://example.test/rest"
    assert prov.feature_count == 42
    assert prov.id_column == "OBJECTID"
    assert prov.bbox == (1.0, 2.0, 3.0, 4.0)
    assert prov.provenance_from == "sidecar"


def test_snapshot_resolution_falls_back_to_yaml_and_mtime(raw_dir, empty_datasets_dir):
    path = _write_target_parquet(raw_dir, "us_ok_targets")
    # No sidecar written. Fall back to dataset YAML `source:` + file mtime.
    (empty_datasets_dir / "us_ok_targets.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "us_ok_targets",
                "source": {"type": "arcgis", "url": "https://example.test/yaml-source"},
                "fetch": {"id_column": "OBJECTID", "bbox": [1.0, 2.0, 3.0, 4.0]},
            }
        )
    )
    import os
    import time

    # An mtime instant of 2026-05-04 20:00 UTC. Force a far-eastern local tz
    # (UTC+14) so the LOCAL calendar date of that instant is the NEXT day
    # (2026-05-05). The snapshot must resolve in UTC (2026-05-04) regardless —
    # this hardcoded assertion would FAIL if the code regressed to local-tz
    # ``date.fromtimestamp``. Save/restore TZ + tzset explicitly so the forced
    # timezone never leaks into other tests in this worker process.
    mtime = datetime(2026, 5, 4, 20, 0, 0, tzinfo=UTC).timestamp()
    os.utime(path, (mtime, mtime))
    _saved_tz = os.environ.get("TZ")
    os.environ["TZ"] = "Pacific/Kiritimati"  # UTC+14
    time.tzset()
    try:
        # Sanity: in this tz the local date really does differ from the UTC date,
        # so the assertion below is a real guard, not a tautology.
        assert date.fromtimestamp(mtime).isoformat() == "2026-05-05"

        prov = resolve_snapshot_provenance(path, "us_ok_targets", datasets_dir=empty_datasets_dir)
        assert prov.snapshot == "2026-05-04"  # UTC calendar date of the mtime instant
    finally:
        if _saved_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = _saved_tz
        time.tzset()
    assert prov.source == "arcgis"
    assert prov.source_url == "https://example.test/yaml-source"
    assert prov.id_column == "OBJECTID"
    assert prov.bbox == (1.0, 2.0, 3.0, 4.0)
    assert prov.provenance_from == "yaml+mtime"


def test_snapshot_resolution_fallback_with_no_yaml_at_all(raw_dir, empty_datasets_dir):
    """No sidecar AND no dataset YAML: still resolves via mtime, with empty provenance."""
    path = _write_target_parquet(raw_dir, "zz_no_config")
    prov = resolve_snapshot_provenance(path, "zz_no_config", datasets_dir=empty_datasets_dir)
    assert prov.provenance_from == "yaml+mtime"
    assert prov.source is None
    assert prov.source_url is None
    assert prov.id_column is None
    assert prov.bbox is None
    # Row count still resolved via parquet metadata, not the (absent) yaml.
    assert prov.feature_count == 2


# --------------------------------------------------------------------------
# Immutability (sync_targets_local)
# --------------------------------------------------------------------------
def test_sync_targets_local_copies_tree(raw_dir, tmp_path, empty_datasets_dir):
    _write_target_parquet(raw_dir, "us_ok_targets")
    staging = tmp_path / "staging"
    report = assemble_targets_staging(
        raw_dir, staging, _registry(), datasets_dir=empty_datasets_dir
    )
    ok = next(d for d in report.datasets if d.dataset == "us_ok_targets")

    target = tmp_path / "target"
    written, plan = sync_targets_local(staging, target, force=False)
    assert written
    assert plan.snapshots == [("us_ok_targets", ok.snapshot)]
    assert plan.skipped_snapshots == []
    assert (
        target
        / TARGETS_PREFIX
        / "dataset=us_ok_targets"
        / f"snapshot={ok.snapshot}"
        / TARGET_DATA_FILENAME
    ).exists()
    assert (target / TARGETS_PREFIX / TARGETS_INDEX_FILENAME).exists()


def test_sync_targets_local_immutable_snapshot_skipped(raw_dir, tmp_path, empty_datasets_dir):
    _write_target_parquet(raw_dir, "us_ok_targets")
    staging = tmp_path / "staging"
    report = assemble_targets_staging(
        raw_dir, staging, _registry(), datasets_dir=empty_datasets_dir
    )
    ok = next(d for d in report.datasets if d.dataset == "us_ok_targets")

    target = tmp_path / "target"
    sync_targets_local(staging, target, force=False)
    marker = (
        target
        / TARGETS_PREFIX
        / "dataset=us_ok_targets"
        / f"snapshot={ok.snapshot}"
        / TARGET_META_FILENAME
    )
    marker.write_text("tampered\n")

    written, plan = sync_targets_local(staging, target, force=False)
    assert plan.skipped_snapshots == [("us_ok_targets", ok.snapshot)]
    assert plan.snapshots == []
    assert marker.read_text() == "tampered\n"  # untouched
    # Mutable top-level files (latest.json / index.json) still re-sync.
    assert any(p.endswith(TARGETS_INDEX_FILENAME) for p in written)
    assert all(f"snapshot={ok.snapshot}" not in p for p in written)

    # force=True replaces the snapshot content.
    written_forced, plan_forced = sync_targets_local(staging, target, force=True)
    assert plan_forced.skipped_snapshots == []
    assert marker.read_text() != "tampered\n"


def test_sync_targets_local_new_snapshot_alongside_existing(raw_dir, tmp_path, empty_datasets_dir):
    """A later re-fetch overwrites ``data/raw/<name>_v1.0.parquet`` in place and
    produces a new ``snapshot=<date>`` in the freshly-rebuilt staging tree. Since
    the sync never deletes (no ``--delete``), publishing the new snapshot must
    leave the previously-published older snapshot dir intact at the target
    alongside it (regression for any future ``--delete``-style behavior that
    would destroy immutable history)."""
    path = _write_target_parquet(raw_dir, "us_ok_targets")
    staging = tmp_path / "staging"
    target = tmp_path / "target"
    report1 = assemble_targets_staging(
        raw_dir, staging, _registry(), datasets_dir=empty_datasets_dir
    )
    old_snapshot = next(d for d in report1.datasets if d.dataset == "us_ok_targets").snapshot
    sync_targets_local(staging, target, force=False)

    import os

    new_mtime = datetime(2026, 12, 25, 0, 0, 0).timestamp()
    os.utime(path, (new_mtime, new_mtime))
    report2 = assemble_targets_staging(
        raw_dir, staging, _registry(), datasets_dir=empty_datasets_dir
    )
    new_snapshot = next(d for d in report2.datasets if d.dataset == "us_ok_targets").snapshot
    assert new_snapshot != old_snapshot
    # The rebuilt staging tree holds only the current (new) snapshot — a target
    # dataset file is a point-in-time overwrite, not an accumulating history.
    assert staged_target_snapshots(staging) == [("us_ok_targets", new_snapshot)]

    written, plan = sync_targets_local(staging, target, force=False)
    # Nothing to skip: the new snapshot was never previously published.
    assert plan.skipped_snapshots == []
    assert plan.snapshots == [("us_ok_targets", new_snapshot)]
    assert (
        target
        / TARGETS_PREFIX
        / "dataset=us_ok_targets"
        / f"snapshot={new_snapshot}"
        / TARGET_DATA_FILENAME
    ).exists()
    # The older snapshot, published in the first sync, is left untouched (no
    # --delete semantics) — both remain queryable at their immutable paths.
    assert (
        target
        / TARGETS_PREFIX
        / "dataset=us_ok_targets"
        / f"snapshot={old_snapshot}"
        / TARGET_DATA_FILENAME
    ).exists()


def test_staged_target_snapshots(raw_dir, tmp_path, empty_datasets_dir):
    _write_target_parquet(raw_dir, "us_ok_targets")
    staging = tmp_path / "staging"
    report = assemble_targets_staging(
        raw_dir, staging, _registry(), datasets_dir=empty_datasets_dir
    )
    ok = next(d for d in report.datasets if d.dataset == "us_ok_targets")
    assert staged_target_snapshots(staging) == [("us_ok_targets", ok.snapshot)]


# --------------------------------------------------------------------------
# R2 sync: argv correctness + immutability, all with a mocked subprocess
# (no network, per the task requirements).
# --------------------------------------------------------------------------
def test_build_aws_sync_argv_excludes_target_snapshots():
    cfg = R2Config(
        endpoint="https://acct.r2.cloudflarestorage.com",
        access_key="k",
        secret_key="s",
        bucket="crosswalk-bridges",
    )
    argv = build_aws_sync_argv(
        Path("/staging"),
        cfg,
        exclude_prefixes=[f"{TARGETS_PREFIX}/dataset=us_ok_targets/snapshot=2026-07-06/"],
    )
    assert "--exclude" in argv
    assert f"{TARGETS_PREFIX}/dataset=us_ok_targets/snapshot=2026-07-06/*" in argv
    assert "k" not in argv and "s" not in argv


def test_sync_targets_r2_skips_existing_snapshot_and_never_hits_network(
    raw_dir, tmp_path, empty_datasets_dir, monkeypatch
):
    _write_target_parquet(raw_dir, "us_ok_targets")
    staging = tmp_path / "staging"
    report = assemble_targets_staging(
        raw_dir, staging, _registry(), datasets_dir=empty_datasets_dir
    )
    ok = next(d for d in report.datasets if d.dataset == "us_ok_targets")
    cfg = R2Config(
        endpoint="https://x.r2.cloudflarestorage.com", access_key="k", secret_key="s", bucket="b"
    )

    calls: list[list[str]] = []

    class _FakeCompleted:
        def __init__(self, returncode=0, stdout=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = ""

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[:3] == ["aws", "s3api", "list-objects-v2"]:
            # Report the snapshot as already present remotely.
            return _FakeCompleted(
                returncode=0,
                stdout=json.dumps({"Contents": [{"Key": "irrelevant"}]}),
            )
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr("crosswalk.factory.publish_sync.subprocess.run", fake_run)

    proc, plan = sync_targets_r2(staging, cfg, force=False)
    assert plan.skipped_snapshots == [("us_ok_targets", ok.snapshot)]
    assert plan.snapshots == []
    # An existence check ran, and the final sync excluded the existing snapshot.
    sync_call = next(c for c in calls if c[:2] == ["aws", "s3"])
    assert "--exclude" in sync_call
    assert f"{TARGETS_PREFIX}/dataset=us_ok_targets/snapshot={ok.snapshot}/*" in sync_call
    # Every "network" interaction went through the mocked subprocess.run only.
    assert all(c[0] == "aws" for c in calls)


def test_sync_targets_r2_force_skips_existence_check(
    raw_dir, tmp_path, empty_datasets_dir, monkeypatch
):
    _write_target_parquet(raw_dir, "us_ok_targets")
    staging = tmp_path / "staging"
    assemble_targets_staging(raw_dir, staging, _registry(), datasets_dir=empty_datasets_dir)
    cfg = R2Config(
        endpoint="https://x.r2.cloudflarestorage.com", access_key="k", secret_key="s", bucket="b"
    )

    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)

        class _FakeCompleted:
            returncode = 0
            stdout = ""
            stderr = ""

        return _FakeCompleted()

    monkeypatch.setattr("crosswalk.factory.publish_sync.subprocess.run", fake_run)
    _, plan = sync_targets_r2(staging, cfg, force=True)
    assert plan.skipped_snapshots == []
    # force=True: no existence (list-objects-v2) check at all, only the sync call.
    assert len(calls) == 1
    assert calls[0][:2] == ["aws", "s3"]


# --------------------------------------------------------------------------
# Dry-run wiring: the CLI must never touch the network under --dry-run (default).
# --------------------------------------------------------------------------
def test_cli_targets_dry_run_emits_no_uploads(raw_dir, tmp_path, empty_datasets_dir, monkeypatch):
    _write_target_parquet(raw_dir, "us_ok_targets")
    staging = tmp_path / "staging"

    monkeypatch.setenv("R2_ENDPOINT_URL", "https://acct.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("R2_BUCKET", "crosswalk-bridges")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess.run must never be called during --dry-run")

    monkeypatch.setattr("crosswalk.factory.publish_sync.subprocess.run", fail_if_called)

    with patch("crosswalk.factory.licenses.LicenseRegistry.load", return_value=_registry()):
        result = runner.invoke(
            app,
            [
                "factory",
                "publish",
                "--targets",
                "--raw-dir",
                str(raw_dir),
                "--staging-dir",
                str(staging),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output
    # Staging tree was still built (the whole point of a dry run is to preview it).
    assert (staging / TARGETS_PREFIX / TARGETS_INDEX_FILENAME).exists()


def test_cli_targets_no_dry_run_local_target_dir(raw_dir, tmp_path, empty_datasets_dir):
    """--no-dry-run with --target-dir performs a real (local, no-network) publish."""
    _write_target_parquet(raw_dir, "us_ok_targets")
    staging = tmp_path / "staging"
    out_dir = tmp_path / "out"

    with patch("crosswalk.factory.licenses.LicenseRegistry.load", return_value=_registry()):
        result = runner.invoke(
            app,
            [
                "factory",
                "publish",
                "--targets",
                "--raw-dir",
                str(raw_dir),
                "--staging-dir",
                str(staging),
                "--target-dir",
                str(out_dir),
                "--no-dry-run",
            ],
        )

    assert result.exit_code == 0, result.output
    assert (out_dir / TARGETS_PREFIX / TARGETS_INDEX_FILENAME).exists()
