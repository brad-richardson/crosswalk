"""Tests for reproducibility metadata recorded with benchmark results."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import mbench.provenance as provenance_module
from mbench.provenance import collect_provenance, file_fingerprint, git_provenance


def test_file_fingerprint_is_content_addressed(tmp_path: Path):
    path = tmp_path / "input.parquet"
    path.write_bytes(b"benchmark-input")
    fp = file_fingerprint(path)
    assert fp["path"] == str(path.resolve())
    assert fp["size"] == len(b"benchmark-input")
    assert fp["sha256"] == hashlib.sha256(b"benchmark-input").hexdigest()


def test_collect_provenance_records_inputs_and_effective_options(tmp_path: Path):
    reference = tmp_path / "reference.parquet"
    target = tmp_path / "target.parquet"
    connectors = tmp_path / "connectors.parquet"
    for path, contents in (
        (reference, b"reference"),
        (target, b"target"),
        (connectors, b"connectors"),
    ):
        path.write_bytes(contents)
    labels_dir = tmp_path / "labels"
    label_dir = labels_dir / "dataset=demo"
    label_dir.mkdir(parents=True)
    (label_dir / "data.csv").write_text("ref_id,target_id,label\nr1,t1,match\n")

    provenance = collect_provenance(
        tool="fake",
        dataset="demo",
        reference=reference,
        target=target,
        labels_dir=labels_dir,
        stitch_labels_dir=None,
        match_level="target",
        run_options={"connectors": connectors, "threshold": 0.7},
    )

    assert set(provenance["inputs"]) == {"reference", "target", "labels", "connectors"}
    assert provenance["invocation"]["run_options"]["connectors"] == str(connectors.resolve())
    assert provenance["invocation"]["run_options"]["threshold"] == 0.7


def test_git_provenance_separates_tracked_and_untracked_state():
    provenance = git_provenance()
    assert provenance["available"] is True
    assert len(provenance["sha"]) == 40
    assert isinstance(provenance["tracked_dirty"], bool)
    assert isinstance(provenance["untracked_present"], bool)


def test_git_provenance_records_launch_failure(monkeypatch, tmp_path: Path):
    (tmp_path / ".git").mkdir()

    def fail_launch(*_args, **_kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr("mbench.provenance.subprocess.run", fail_launch)
    provenance = git_provenance(tmp_path)
    assert provenance["available"] is False
    assert "git unavailable" in provenance["error"]


def test_git_provenance_does_not_report_clean_when_status_fails(monkeypatch, tmp_path: Path):
    (tmp_path / ".git").mkdir()
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, stdout="a" * 40 + "\n", stderr=""),
            subprocess.CompletedProcess([], 128, stdout="", stderr="status failed"),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        ]
    )
    monkeypatch.setattr(provenance_module, "_git", lambda *_args: next(responses))
    provenance = git_provenance(tmp_path)
    assert provenance["available"] is True
    assert provenance["sha"] == "a" * 40
    assert provenance["worktree_state_available"] is False
    assert provenance["tracked_dirty"] is None
    assert provenance["untracked_present"] is None
    assert "status failed" in provenance["worktree_error"]


def test_collect_provenance_binds_git_to_effective_working_directory(monkeypatch, tmp_path: Path):
    reference = tmp_path / "reference"
    target = tmp_path / "target"
    reference.write_bytes(b"r")
    target.write_bytes(b"t")
    labels = tmp_path / "labels"
    (labels / "dataset=demo").mkdir(parents=True)
    (labels / "dataset=demo" / "data.csv").write_text("ref_id,target_id,label\n")
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    seen = {}

    def capture_git(repo=None):
        seen["repo"] = repo
        return {"available": True, "sha": "abc"}

    monkeypatch.setattr(provenance_module, "git_provenance", capture_git)
    collected = collect_provenance(
        tool="fake",
        dataset="demo",
        reference=reference,
        target=target,
        labels_dir=labels,
        stitch_labels_dir=None,
        match_level="target",
        run_options={},
        adapter_metadata={"working_directory": str(checkout)},
    )
    assert seen["repo"] == checkout
    assert collected["git"]["sha"] == "abc"


def test_custom_crosswalk_command_marks_model_settings_unverifiable(tmp_path: Path):
    reference = tmp_path / "reference"
    target = tmp_path / "target"
    reference.write_bytes(b"r")
    target.write_bytes(b"t")
    labels = tmp_path / "labels"
    (labels / "dataset=demo").mkdir(parents=True)
    (labels / "dataset=demo" / "data.csv").write_text("ref_id,target_id,label\n")

    collected = collect_provenance(
        tool="crosswalk",
        dataset="demo",
        reference=reference,
        target=target,
        labels_dir=labels,
        stitch_labels_dir=None,
        match_level="target",
        run_options={},
        adapter_metadata={
            "working_directory": str(Path.cwd()),
            "crosswalk_command_is_default": False,
        },
    )
    crosswalk = collected["crosswalk"]
    assert crosswalk["verifiable"] is False
    assert "custom or unknown Crosswalk executable" in crosswalk["unverifiable_reasons"]
    assert "model" not in crosswalk


def test_collect_provenance_tolerates_mixed_type_dict_keys(tmp_path: Path):
    """Provenance is best-effort: adapter metadata with int/str dict keys
    (e.g. value_counts of a mixed parquet column) must not raise TypeError."""
    reference = tmp_path / "reference"
    target = tmp_path / "target"
    reference.write_bytes(b"r")
    target.write_bytes(b"t")
    labels = tmp_path / "labels"
    (labels / "dataset=demo").mkdir(parents=True)

    collected = collect_provenance(
        tool="fake",
        dataset="demo",
        reference=reference,
        target=target,
        labels_dir=labels,
        stitch_labels_dir=None,
        match_level="target",
        run_options={"connectors": 12345},
        adapter_metadata={"match_type_counts": {1: 2, "1:N": 3}},
    )
    assert collected["invocation"]["adapter_metadata"]["match_type_counts"] == {
        "1": 2,
        "1:N": 3,
    }


def test_cwd_mismatch_marks_model_settings_unverifiable(tmp_path: Path, monkeypatch):
    """Crosswalk resolves its model path and .env relative to cwd; if mbench
    runs from a different cwd than the adapter subprocess, fingerprinting from
    this process could certify the wrong model."""
    reference = tmp_path / "reference"
    target = tmp_path / "target"
    reference.write_bytes(b"r")
    target.write_bytes(b"t")
    labels = tmp_path / "labels"
    (labels / "dataset=demo").mkdir(parents=True)
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root / "mbench")

    collected = collect_provenance(
        tool="crosswalk",
        dataset="demo",
        reference=reference,
        target=target,
        labels_dir=labels,
        stitch_labels_dir=None,
        match_level="target",
        run_options={},
        adapter_metadata={
            "working_directory": str(repo_root),
            "crosswalk_command_is_default": True,
        },
    )
    crosswalk = collected["crosswalk"]
    assert crosswalk["verifiable"] is False
    assert any("cwd differs" in reason for reason in crosswalk["unverifiable_reasons"])


def test_other_crosswalk_checkout_marks_model_settings_unverifiable(tmp_path: Path):
    reference = tmp_path / "reference"
    target = tmp_path / "target"
    reference.write_bytes(b"r")
    target.write_bytes(b"t")
    labels = tmp_path / "labels"
    (labels / "dataset=demo").mkdir(parents=True)
    (labels / "dataset=demo" / "data.csv").write_text("ref_id,target_id,label\n")
    other_checkout = tmp_path / "other"
    (other_checkout / ".git").mkdir(parents=True)

    collected = collect_provenance(
        tool="crosswalk",
        dataset="demo",
        reference=reference,
        target=target,
        labels_dir=labels,
        stitch_labels_dir=None,
        match_level="target",
        run_options={},
        adapter_metadata={
            "working_directory": str(other_checkout),
            "crosswalk_command_is_default": True,
        },
    )
    crosswalk = collected["crosswalk"]
    assert crosswalk["verifiable"] is False
    assert any("does not match" in reason for reason in crosswalk["unverifiable_reasons"])
    assert "settings" not in crosswalk
