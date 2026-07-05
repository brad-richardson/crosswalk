"""Tests for cbench CLI path resolution.

These lock in the fix for the "labels not found" defect: config-default paths
(``data_dir``/``labels_dir``/``stitch_labels_dir``) must resolve relative to the
datasets.toml file's directory, not the process CWD, so cbench works from any
working directory (e.g. the repo root).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cbench.cli import _resolve_config_default, _resolve_single_run_paths

CONFIG_BODY = """
[defaults]
data_dir = "../data/raw"
labels_dir = "../labels/human"
stitch_labels_dir = "../labels/stitching"

[datasets.demo]
reference = "demo_ref.parquet"
target = "demo_tgt.parquet"
connectors = "demo_conn.parquet"
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Build a fake repo: <root>/cbench/datasets.toml plus data/labels dirs."""
    root = tmp_path / "matcher"
    (root / "cbench").mkdir(parents=True)
    (root / "data" / "raw").mkdir(parents=True)
    (root / "labels" / "human").mkdir(parents=True)
    (root / "labels" / "stitching").mkdir(parents=True)
    (root / "cbench" / "datasets.toml").write_text(CONFIG_BODY)
    return root


class TestResolveConfigDefault:
    def test_relative_anchors_to_config_dir_not_cwd(self, repo, monkeypatch):
        config = repo / "cbench" / "datasets.toml"
        # Run from a wholly unrelated CWD to prove independence from it.
        monkeypatch.chdir(repo.parent)
        resolved = _resolve_config_default(config, "../labels/human")
        assert resolved == (repo / "labels" / "human").resolve()

    def test_absolute_returned_unchanged(self, repo):
        config = repo / "cbench" / "datasets.toml"
        abs_path = repo / "labels" / "human"
        assert _resolve_config_default(config, str(abs_path)) == abs_path

    def test_independent_of_cwd(self, repo, monkeypatch):
        config = repo / "cbench" / "datasets.toml"
        monkeypatch.chdir(repo)
        from_root = _resolve_config_default(config, "../data/raw")
        monkeypatch.chdir(os.sep)
        from_slash = _resolve_config_default(config, "../data/raw")
        assert from_root == from_slash == (repo / "data" / "raw").resolve()


class TestResolveSingleRunPaths:
    def test_fills_all_from_config(self, repo, monkeypatch):
        config = repo / "cbench" / "datasets.toml"
        monkeypatch.chdir(repo.parent)  # non-repo cwd
        ref, tgt, labels, stitch, conn = _resolve_single_run_paths(
            config=config,
            dataset="demo",
            reference=None,
            target=None,
            labels=None,
            stitch_labels=None,
        )
        assert ref == repo / "data" / "raw" / "demo_ref.parquet"
        assert tgt == repo / "data" / "raw" / "demo_tgt.parquet"
        assert labels == (repo / "labels" / "human").resolve()
        assert stitch == (repo / "labels" / "stitching").resolve()
        assert conn == repo / "data" / "raw" / "demo_conn.parquet"

    def test_explicit_paths_win(self, repo):
        config = repo / "cbench" / "datasets.toml"
        explicit_ref = Path("/custom/ref.parquet")
        explicit_labels = Path("/custom/labels")
        # All four provided -> config never consulted, connectors stays None.
        ref, tgt, labels, stitch, conn = _resolve_single_run_paths(
            config=config,
            dataset="demo",
            reference=explicit_ref,
            target=Path("/custom/tgt.parquet"),
            labels=explicit_labels,
            stitch_labels=Path("/custom/stitch"),
        )
        assert ref == explicit_ref
        assert labels == explicit_labels
        assert conn is None

    def test_partial_override_fills_rest_from_config(self, repo):
        config = repo / "cbench" / "datasets.toml"
        ref, tgt, labels, stitch, conn = _resolve_single_run_paths(
            config=config,
            dataset="demo",
            reference=Path("/custom/ref.parquet"),
            target=None,
            labels=None,
            stitch_labels=None,
        )
        assert ref == Path("/custom/ref.parquet")
        assert tgt == repo / "data" / "raw" / "demo_tgt.parquet"
        assert labels == (repo / "labels" / "human").resolve()

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _resolve_single_run_paths(
                config=tmp_path / "nope.toml",
                dataset="demo",
                reference=None,
                target=None,
                labels=None,
                stitch_labels=None,
            )

    def test_unknown_dataset_raises(self, repo):
        config = repo / "cbench" / "datasets.toml"
        with pytest.raises(ValueError, match="not found"):
            _resolve_single_run_paths(
                config=config,
                dataset="ghost",
                reference=None,
                target=None,
                labels=None,
                stitch_labels=None,
            )
