"""Unit tests for the R2 publisher (Milestone M5).

Covers: staging-tree assembly from a synthetic factory layout, license-gated
inclusion/exclusion, index.json correctness, checksum stability (idempotent
re-assembly), the unified long table, dry-run behaviour without credentials, and
immutable-release enforcement on the local-dir sync.

All synthetic — no model, no pipeline, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from crosswalk.factory.licenses import LicenseRegistry
from crosswalk.factory.manifest import Manifest
from crosswalk.factory.publish import (
    ALL_BRIDGES_FILENAME,
    BRIDGES_PREFIX,
    INDEX_HTML,
    INDEX_JSON,
    assemble_staging,
    build_all_bridges,
    checksum_of,
)
from crosswalk.factory.publish_sync import (
    missing_r2_env,
    r2_env,
    staged_release_dirs,
    sync_local,
)

RELEASE = "2026-01-21.0"


def _registry() -> LicenseRegistry:
    """Registry: one approved dataset, one explicitly pending, (a third unlisted)."""
    return LicenseRegistry(
        {
            "overture": {
                "attribution": "Overture / © OpenStreetMap contributors (ODbL)",
                "url": "https://overturemaps.org/",
                "license": "ODbL-1.0",
            },
            "datasets": {
                "us_ok_roads": {
                    "status": "approved",
                    "license": "US-PD",
                    "attribution": "Some Agency",
                },
                "xx_pending_roads": {
                    "status": "pending_review",
                    "note": "verify terms",
                },
            },
        }
    )


def _make_factory_dataset(factory_root, name, *, n_rows=3, release=RELEASE, matched=2):
    """Write a synthetic factory dataset output (bridge.parquet + manifest.json)."""
    ds_dir = factory_root / f"release={release}" / f"dataset={name}"
    ds_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "local_id": [f"{name}_{i}" for i in range(n_rows)],
            "gers_id": [f"gers-{name}-{n_rows - i}" for i in range(n_rows)],  # unsorted
            "confidence": [0.9 - 0.1 * i for i in range(n_rows)],
            "match_type": ["1:1"] * n_rows,
            "match_decision": (["match"] * matched + ["review"] * (n_rows - matched)),
        }
    )
    df.to_parquet(ds_dir / "bridge.parquet", index=False)
    m = Manifest(
        dataset=name,
        release=release,
        created_at="2026-07-05T00:00:00+00:00",
        n_reference=10,
        n_target=n_rows,
        n_candidates=5,
        n_matched=matched,
        n_review=n_rows - matched,
        n_unmatched=0,
        groups={"n_groups": 1, "n_m_to_n": 0, "n_oversized": 0},
        wall_s=1.2,
    )
    m.write(ds_dir / "manifest.json")
    return ds_dir


@pytest.fixture
def factory_root(tmp_path):
    root = tmp_path / "factory"
    _make_factory_dataset(root, "us_ok_roads", n_rows=4, matched=3)
    _make_factory_dataset(root, "xx_pending_roads", n_rows=2, matched=2)
    return root


@pytest.fixture
def empty_datasets_dir(tmp_path):
    d = tmp_path / "datasets"
    d.mkdir()
    return d


# --------------------------------------------------------------------------
# Assembly + license gating
# --------------------------------------------------------------------------
def test_assembly_publishes_only_licensed(factory_root, tmp_path, empty_datasets_dir):
    staging = tmp_path / "staging"
    report = assemble_staging(
        factory_root,
        staging,
        _registry(),
        datasets_dir=empty_datasets_dir,
        generated_at="fixed",
    )
    assert report.n_published == 1
    assert report.n_excluded == 1
    assert report.latest_release == RELEASE

    approved = staging / BRIDGES_PREFIX / f"release={RELEASE}" / "dataset=us_ok_roads"
    assert (approved / "bridge.parquet").exists()
    assert (approved / "manifest.json").exists()
    # Excluded dataset must NOT have its data copied.
    excluded = staging / BRIDGES_PREFIX / f"release={RELEASE}" / "dataset=xx_pending_roads"
    assert not excluded.exists()


def test_assembly_writes_page_and_indexes(factory_root, tmp_path, empty_datasets_dir):
    staging = tmp_path / "staging"
    assemble_staging(
        factory_root,
        staging,
        _registry(),
        datasets_dir=empty_datasets_dir,
        generated_at="fixed",
    )
    assert (staging / INDEX_HTML).exists()
    assert (staging / INDEX_JSON).exists()
    html = (staging / INDEX_HTML).read_text()
    assert "GERS Bridge Tables" in html
    assert "us_ok_roads" in html
    # Excluded dataset appears in the "pending review" section, not as published data.
    assert "xx_pending_roads" in html
    rel_index = staging / BRIDGES_PREFIX / f"release={RELEASE}" / INDEX_JSON
    assert rel_index.exists()


def test_index_json_correctness(factory_root, tmp_path, empty_datasets_dir):
    staging = tmp_path / "staging"
    assemble_staging(
        factory_root,
        staging,
        _registry(),
        datasets_dir=empty_datasets_dir,
        generated_at="fixed-ts",
    )
    idx = json.loads((staging / INDEX_JSON).read_text())
    assert idx["generated_at"] == "fixed-ts"
    assert idx["latest_release"] == RELEASE
    assert idx["totals"] == {"n_releases": 1, "n_published": 1, "n_excluded": 1}
    rel = idx["releases"][RELEASE]
    ok = rel["datasets"]["us_ok_roads"]
    assert ok["status"] == "published"
    assert ok["stats"]["n_matched"] == 3
    assert ok["stats"]["match_rate"] == 0.75  # 3/4
    assert set(ok["files"]) == {"bridge.parquet", "manifest.json"}
    pend = rel["datasets"]["xx_pending_roads"]
    assert pend["status"] == "excluded"
    assert "pending_review" in pend["reason"]
    assert "files" not in pend
    assert rel["all_bridges"]["n_rows"] == 4  # only the published dataset's rows


def test_unlisted_dataset_is_excluded(factory_root, tmp_path, empty_datasets_dir):
    _make_factory_dataset(factory_root, "zz_unlisted_roads")
    staging = tmp_path / "staging"
    report = assemble_staging(
        factory_root,
        staging,
        _registry(),
        datasets_dir=empty_datasets_dir,
        generated_at="fixed",
    )
    zz = next(d for r in report.releases for d in r.datasets if d.dataset == "zz_unlisted_roads")
    assert not zz.published
    assert "no license registry entry" in zz.reason


# --------------------------------------------------------------------------
# Unified long table
# --------------------------------------------------------------------------
def test_all_bridges_has_dataset_col_and_sorted(factory_root, tmp_path, empty_datasets_dir):
    # Approve both so all_bridges spans two datasets.
    reg = LicenseRegistry(
        {
            "overture": {"attribution": "O", "url": "u", "license": "ODbL-1.0"},
            "datasets": {
                "us_ok_roads": {"status": "approved", "license": "L", "attribution": "A"},
                "xx_pending_roads": {"status": "approved", "license": "L", "attribution": "A"},
            },
        }
    )
    staging = tmp_path / "staging"
    assemble_staging(
        factory_root, staging, reg, datasets_dir=empty_datasets_dir, generated_at="fixed"
    )
    ab = pd.read_parquet(staging / BRIDGES_PREFIX / f"release={RELEASE}" / ALL_BRIDGES_FILENAME)
    assert "dataset" in ab.columns
    assert set(ab["dataset"].unique()) == {"us_ok_roads", "xx_pending_roads"}
    assert list(ab["gers_id"]) == sorted(ab["gers_id"])  # sorted for row-group pruning
    assert len(ab) == 6  # 4 + 2


def test_build_all_bridges_empty_returns_zero(tmp_path):
    assert build_all_bridges({}, tmp_path / "x.parquet") == 0
    assert not (tmp_path / "x.parquet").exists()


# --------------------------------------------------------------------------
# Checksum stability (idempotent re-assembly)
# --------------------------------------------------------------------------
def test_checksums_stable_across_reassembly(factory_root, tmp_path, empty_datasets_dir):
    def run(dst):
        rep = assemble_staging(
            factory_root,
            dst,
            _registry(),
            datasets_dir=empty_datasets_dir,
            generated_at="fixed",
        )
        pub = next(d for r in rep.releases for d in r.datasets if d.published)
        return (
            {name: ck.sha256 for name, ck in pub.files.items()},
            rep.releases[0].all_bridges.sha256,
        )

    files1, ab1 = run(tmp_path / "s1")
    files2, ab2 = run(tmp_path / "s2")
    assert files1 == files2
    assert ab1 == ab2  # unified table write is deterministic


def test_checksum_matches_written_file(factory_root, tmp_path, empty_datasets_dir):
    staging = tmp_path / "staging"
    assemble_staging(
        factory_root,
        staging,
        _registry(),
        datasets_dir=empty_datasets_dir,
        generated_at="fixed",
    )
    idx = json.loads((staging / INDEX_JSON).read_text())
    ok = idx["releases"][RELEASE]["datasets"]["us_ok_roads"]
    bridge = (
        staging / BRIDGES_PREFIX / f"release={RELEASE}" / "dataset=us_ok_roads" / "bridge.parquet"
    )
    assert checksum_of(bridge).sha256 == ok["files"]["bridge.parquet"]["sha256"]
    # checksums.txt lists it too.
    ck_txt = (staging / BRIDGES_PREFIX / f"release={RELEASE}" / "checksums.txt").read_text()
    assert ok["files"]["bridge.parquet"]["sha256"] in ck_txt


# --------------------------------------------------------------------------
# Sync: local-dir + immutability, dry-run creds
# --------------------------------------------------------------------------
def test_sync_local_copies_tree(factory_root, tmp_path, empty_datasets_dir):
    staging = tmp_path / "staging"
    assemble_staging(
        factory_root,
        staging,
        _registry(),
        datasets_dir=empty_datasets_dir,
        generated_at="fixed",
    )
    target = tmp_path / "target"
    written, plan = sync_local(staging, target, force=False)
    assert written
    assert plan.releases == [RELEASE]
    assert plan.skipped_releases == []
    assert (target / INDEX_JSON).exists()
    assert (
        target / BRIDGES_PREFIX / f"release={RELEASE}" / "dataset=us_ok_roads" / "bridge.parquet"
    ).exists()


def test_sync_local_immutable_release_skipped(factory_root, tmp_path, empty_datasets_dir):
    """A re-publish never overwrites an existing release: it SKIPS it (and still
    updates the mutable top-level index files), so multi-release publishing keeps
    working run after run. ``force=True`` re-publishes."""
    staging = tmp_path / "staging"
    assemble_staging(
        factory_root,
        staging,
        _registry(),
        datasets_dir=empty_datasets_dir,
        generated_at="fixed",
    )
    target = tmp_path / "target"
    sync_local(staging, target, force=False)
    # Tamper with the published release file so we can detect any overwrite.
    marker = target / BRIDGES_PREFIX / f"release={RELEASE}" / "checksums.txt"
    marker.write_text("tampered\n")

    written, plan = sync_local(staging, target, force=False)
    assert plan.skipped_releases == [RELEASE]
    assert plan.releases == []
    # Release files untouched (immutable), top-level index files re-synced.
    assert marker.read_text() == "tampered\n"
    assert INDEX_JSON in {p.split("/")[-1] for p in written}
    assert all(not p.startswith(f"{BRIDGES_PREFIX}/release={RELEASE}/") for p in written)

    # force=True replaces the release content.
    written_forced, plan_forced = sync_local(staging, target, force=True)
    assert plan_forced.skipped_releases == []
    assert marker.read_text() != "tampered\n"


def test_sync_local_publishes_new_release_alongside_existing(
    factory_root, tmp_path, empty_datasets_dir
):
    """The go-live loop: release A already published, staging also contains new
    release B — B must publish while A is skipped (regression for the
    abort-on-any-existing-release bug)."""
    target = tmp_path / "target"
    staging = tmp_path / "staging"
    assemble_staging(
        factory_root, staging, _registry(), datasets_dir=empty_datasets_dir, generated_at="fixed"
    )
    sync_local(staging, target, force=False)

    # A new Overture release lands in the factory root; re-stage everything.
    new_release = "2026-06-17.0"
    _make_factory_dataset(factory_root, "us_ok_roads", n_rows=5, release=new_release, matched=4)
    assemble_staging(
        factory_root, staging, _registry(), datasets_dir=empty_datasets_dir, generated_at="fixed"
    )

    written, plan = sync_local(staging, target, force=False)
    assert plan.skipped_releases == [RELEASE]
    assert plan.releases == [new_release]
    assert (
        target
        / BRIDGES_PREFIX
        / f"release={new_release}"
        / "dataset=us_ok_roads"
        / "bridge.parquet"
    ).exists()


def test_staged_release_dirs(factory_root, tmp_path, empty_datasets_dir):
    staging = tmp_path / "staging"
    assemble_staging(
        factory_root,
        staging,
        _registry(),
        datasets_dir=empty_datasets_dir,
        generated_at="fixed",
    )
    assert staged_release_dirs(staging) == [RELEASE]


def test_build_aws_sync_argv_excludes_published_releases():
    from crosswalk.factory.publish_sync import R2Config, build_aws_sync_argv

    cfg = R2Config(
        endpoint="https://acct.r2.cloudflarestorage.com",
        access_key="k",
        secret_key="s",
        bucket="bridges",
    )
    argv = build_aws_sync_argv(
        Path("/staging"), cfg, exclude_prefixes=[f"{BRIDGES_PREFIX}/release={RELEASE}/"]
    )
    assert "--exclude" in argv
    assert f"{BRIDGES_PREFIX}/release={RELEASE}/*" in argv
    # No secrets on the command line (env-only credentials).
    assert "k" not in argv and "s" not in argv


def test_r2_env_absent(monkeypatch):
    for var in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        monkeypatch.delenv(var, raising=False)
    assert r2_env() is None
    assert set(missing_r2_env()) == {
        "R2_ENDPOINT_URL",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET",
    }


def test_r2_env_present(monkeypatch):
    monkeypatch.setenv("R2_ENDPOINT_URL", "https://acct.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("R2_BUCKET", "bridges")
    cfg = r2_env()
    assert cfg is not None
    assert cfg.bucket == "bridges"
    assert missing_r2_env() == []
