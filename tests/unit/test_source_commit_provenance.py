"""Source-commit provenance for newly trained model artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path

import crosswalk.provenance as provenance_module
from crosswalk.provenance import source_commit_provenance


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "tests@example.com")
    _git(repo, "config", "user.name", "Crosswalk Tests")
    (repo / "tracked.txt").write_text("clean\n")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "--quiet", "-m", "initial")
    return repo


def test_source_commit_provenance_separates_tracked_and_untracked(tmp_path: Path):
    repo = _init_repo(tmp_path)

    clean = source_commit_provenance(repo)
    assert clean["available"] is True
    assert len(clean["sha"]) == 40
    assert clean["worktree_state_available"] is True
    assert clean["tracked_dirty"] is False
    assert clean["tracked_change_count"] == 0
    assert clean["untracked_present"] is False
    assert clean["untracked_count"] == 0

    (repo / "tracked.txt").write_text("dirty\n")
    (repo / "untracked.txt").write_text("new\n")
    dirty = source_commit_provenance(repo)
    assert dirty["sha"] == clean["sha"]
    assert dirty["tracked_dirty"] is True
    assert dirty["tracked_change_count"] == 1
    assert dirty["untracked_present"] is True
    assert dirty["untracked_count"] == 1


def test_source_commit_provenance_has_stable_wheel_fallback(tmp_path: Path):
    provenance = source_commit_provenance(tmp_path)

    assert provenance == {
        "available": False,
        "sha": None,
        "worktree_state_available": False,
        "tracked_dirty": None,
        "tracked_change_count": None,
        "untracked_present": None,
        "untracked_count": None,
        "reason": "not_git_checkout",
    }


def test_wheel_nested_in_checkout_does_not_claim_checkout_sha(monkeypatch, tmp_path: Path):
    repo = _init_repo(tmp_path)
    checkout_module = repo / "src" / "crosswalk" / "provenance.py"
    checkout_module.parent.mkdir(parents=True)
    checkout_module.write_text("# checkout source\n")
    wheel_module = repo / ".venv" / "site-packages" / "crosswalk" / "provenance.py"
    wheel_module.parent.mkdir(parents=True)
    wheel_module.write_text("# installed wheel\n")
    monkeypatch.setattr(provenance_module, "__file__", str(wheel_module))

    provenance = source_commit_provenance()

    assert provenance["available"] is False
    assert provenance["reason"] == "not_git_checkout"


def test_source_commit_provenance_keeps_sha_when_worktree_query_fails(
    monkeypatch,
    tmp_path: Path,
):
    repo = _init_repo(tmp_path)
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, stdout="a" * 40 + "\n", stderr=""),
            subprocess.CompletedProcess([], 128, stdout="", stderr="status failed"),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        ]
    )
    monkeypatch.setattr(provenance_module, "_run_git", lambda *_args: next(responses))

    provenance = source_commit_provenance(repo)

    assert provenance["available"] is True
    assert provenance["sha"] == "a" * 40
    assert provenance["worktree_state_available"] is False
    assert provenance["tracked_dirty"] is None
    assert provenance["untracked_present"] is None
    assert "status failed" in provenance["worktree_error"]
