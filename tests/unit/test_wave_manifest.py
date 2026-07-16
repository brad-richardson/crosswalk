"""Unit tests for the wave-manifest contract (write / load_validated / digest)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crosswalk.agent_labeling.stitch_runner import get_panel, panel_descriptor
from crosswalk.agent_labeling.wave_manifest import (
    FIELD_DIGEST,
    FIELD_SCHEMA_VERSION,
    SCHEMA_VERSION,
    WaveManifest,
    compute_digest,
    resolve_batch_dir,
)

PANEL_NAME = "v7-candidate"


def _make_batch(batch_dir: Path, group_id: str = "group-1") -> None:
    group_dir = batch_dir / group_id
    group_dir.mkdir(parents=True)
    (batch_dir / "batch.json").write_text("{}")
    (group_dir / "evidence.json").write_text("{}")


def _content(batch_dir: Path, group_id: str = "group-1") -> dict:
    return {
        "wave": "test_wave",
        "panel": PANEL_NAME,
        "required_panel": panel_descriptor(get_panel(PANEL_NAME)),
        "total_pack_count": 1,
        "batch_dirs": [str(batch_dir)],
        "run_schedule": [
            {
                "run_index": 1,
                "batch_dir": str(batch_dir),
                "group_id": group_id,
                "dataset_id": "dataset",
                "variant": "enriched",
            }
        ],
    }


def test_write_stamps_schema_version_and_digest_and_round_trips(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    _make_batch(batch_dir)
    manifest_path = tmp_path / "manifest.json"

    WaveManifest(_content(batch_dir)).write(manifest_path)

    on_disk = json.loads(manifest_path.read_text())
    assert on_disk[FIELD_SCHEMA_VERSION] == SCHEMA_VERSION
    assert on_disk[FIELD_DIGEST] == compute_digest(on_disk)

    loaded = WaveManifest.load_validated(manifest_path)
    assert loaded.total_pack_count == 1
    assert panel_descriptor(loaded.panel) == panel_descriptor(get_panel(PANEL_NAME))
    # batch_dir is rewritten to the resolved (existing) path.
    assert Path(loaded.run_schedule[0]["batch_dir"]) == batch_dir


def test_digest_excludes_only_itself(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    content = _content(batch_dir)
    with_digest = dict(content)
    with_digest[FIELD_DIGEST] = "whatever-does-not-matter"
    # Adding/removing the digest field itself must not change the digest.
    assert compute_digest(content) == compute_digest(with_digest)


def test_tampered_manifest_is_fatal(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    _make_batch(batch_dir)
    manifest_path = tmp_path / "manifest.json"
    WaveManifest(_content(batch_dir)).write(manifest_path)

    # Mutate content in place WITHOUT re-stamping the digest.
    on_disk = json.loads(manifest_path.read_text())
    on_disk["wave"] = "tampered"
    manifest_path.write_text(json.dumps(on_disk, indent=2))

    with pytest.raises(ValueError, match="integrity digest mismatch"):
        WaveManifest.load_validated(manifest_path)


def test_legacy_manifest_without_digest_warns_but_validates(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    _make_batch(batch_dir)
    manifest_path = tmp_path / "manifest.json"
    legacy = _content(batch_dir)
    legacy[FIELD_SCHEMA_VERSION] = SCHEMA_VERSION  # present, but no manifest_sha256
    manifest_path.write_text(json.dumps(legacy, indent=2) + "\n")

    with pytest.warns(UserWarning, match="no manifest_sha256"):
        loaded = WaveManifest.load_validated(manifest_path)
    assert loaded.total_pack_count == 1


def test_schema_version_mismatch_is_fatal(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    _make_batch(batch_dir)
    manifest_path = tmp_path / "manifest.json"
    content = _content(batch_dir)
    content[FIELD_SCHEMA_VERSION] = SCHEMA_VERSION + 1
    manifest_path.write_text(json.dumps(content, indent=2) + "\n")

    with pytest.raises(ValueError, match="unsupported manifest schema_version"):
        WaveManifest.load_validated(manifest_path)


def test_resolve_batch_dir_absolute_passthrough(tmp_path: Path) -> None:
    absolute = tmp_path / "abs" / "batch"
    resolved = resolve_batch_dir(str(absolute), tmp_path / "manifest.json")
    assert resolved == absolute


def test_resolve_batch_dir_prefers_manifest_sibling_over_cwd(tmp_path: Path) -> None:
    # The v7 layout: a full repo-relative path is stored, but the batch dir is a
    # sibling of the manifest. Resolution must find it via the manifest's parent
    # regardless of cwd, matching by basename.
    manifest_dir = tmp_path / "data" / "agents" / "stitching" / "batches"
    manifest_dir.mkdir(parents=True)
    batch_dir = manifest_dir / "ds_wave"
    batch_dir.mkdir()
    stored = "data/agents/stitching/batches/ds_wave"

    resolved = resolve_batch_dir(stored, manifest_dir / "wave_manifest.json")
    assert resolved == batch_dir


def test_resolve_batch_dir_cwd_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A purely cwd-relative path (no manifest-relative or sibling match) still
    # resolves for a runner invoked from the repo root — the legacy behavior.
    monkeypatch.chdir(tmp_path)
    rel = Path("cwd_only_batch")
    rel.mkdir()
    manifest_path = tmp_path / "elsewhere" / "manifest.json"
    manifest_path.parent.mkdir()

    resolved = resolve_batch_dir("cwd_only_batch", manifest_path)
    assert resolved == rel


def test_load_validated_resolves_sibling_batch_dirs(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "batches"
    manifest_dir.mkdir()
    batch_dir = manifest_dir / "ds_wave"
    _make_batch(batch_dir)
    content = _content(batch_dir)
    # Store a repo-style relative path whose basename is a manifest sibling.
    content["run_schedule"][0]["batch_dir"] = "data/agents/stitching/batches/ds_wave"
    content["batch_dirs"] = ["data/agents/stitching/batches/ds_wave"]
    manifest_path = manifest_dir / "wave_manifest.json"
    WaveManifest(content).write(manifest_path)

    loaded = WaveManifest.load_validated(manifest_path)
    assert Path(loaded.run_schedule[0]["batch_dir"]) == batch_dir


def test_duplicate_scheduled_pack_is_rejected(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    _make_batch(batch_dir)
    content = _content(batch_dir)
    content["total_pack_count"] = 2
    content["run_schedule"].append({**content["run_schedule"][0], "run_index": 2})
    manifest_path = tmp_path / "manifest.json"
    WaveManifest(content).write(manifest_path)

    with pytest.raises(ValueError, match="duplicate scheduled pack"):
        WaveManifest.load_validated(manifest_path)


def test_missing_evidence_pack_is_rejected(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    (batch_dir / "batch.json").write_text("{}")  # no group/evidence.json
    manifest_path = tmp_path / "manifest.json"
    WaveManifest(_content(batch_dir)).write(manifest_path)

    with pytest.raises(FileNotFoundError, match="evidence.json"):
        WaveManifest.load_validated(manifest_path)
