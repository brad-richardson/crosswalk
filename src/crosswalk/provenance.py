"""Best-effort source provenance for reproducible Crosswalk artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _find_repo_root(start: Path) -> Path | None:
    """Return the nearest Git checkout containing ``start``."""
    resolved = start.resolve()
    if resolved.is_file():
        resolved = resolved.parent
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _source_checkout_root() -> Path | None:
    """Find the checkout that supplies this module, never an unrelated parent repo.

    A wheel installed into ``<checkout>/.venv`` is physically below that
    checkout, but its code does not necessarily correspond to the checkout's
    HEAD. Requiring this exact module to be the checkout's ``src`` copy makes
    that case report unavailable instead of recording misleading provenance.
    """
    module_path = Path(__file__).resolve()
    root = _find_repo_root(module_path)
    if root is None:
        return None
    checkout_module = (root / "src" / "crosswalk" / module_path.name).resolve()
    try:
        return root if checkout_module.samefile(module_path) else None
    except OSError:
        return None


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=10,
    )


def _unavailable(reason: str, error: str | None = None) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "available": False,
        "sha": None,
        "worktree_state_available": False,
        "tracked_dirty": None,
        "tracked_change_count": None,
        "untracked_present": None,
        "untracked_count": None,
        "reason": reason,
    }
    if error:
        provenance["error"] = error
    return provenance


def source_commit_provenance(repo: Path | None = None) -> dict[str, Any]:
    """Capture the source revision and separate tracked/untracked dirt.

    Discovery is bound to the checkout that supplies this module. Installed
    wheels and source distributions without Git metadata return a stable
    unavailable record. Callers may pass ``repo`` explicitly for tooling and
    tests. Collection is best-effort so missing Git or an unreadable worktree
    never prevents model training.
    """
    root = _find_repo_root(repo) if repo is not None else _source_checkout_root()
    if root is None:
        return _unavailable("not_git_checkout")

    try:
        head = _run_git(root, "rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError) as exc:
        return _unavailable("git_unavailable", str(exc))
    if head.returncode != 0 or not head.stdout.strip():
        error = head.stderr.strip() or f"git rev-parse exited {head.returncode}"
        return _unavailable("revision_unavailable", error)

    sha = head.stdout.strip()
    try:
        tracked = _run_git(root, "status", "--short", "--untracked-files=no")
        untracked = _run_git(root, "ls-files", "--others", "--exclude-standard")
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": True,
            "sha": sha,
            "worktree_state_available": False,
            "tracked_dirty": None,
            "tracked_change_count": None,
            "untracked_present": None,
            "untracked_count": None,
            "worktree_error": str(exc),
        }

    if tracked.returncode != 0 or untracked.returncode != 0:
        errors = []
        if tracked.returncode != 0:
            errors.append(tracked.stderr.strip() or f"git status exited {tracked.returncode}")
        if untracked.returncode != 0:
            errors.append(untracked.stderr.strip() or f"git ls-files exited {untracked.returncode}")
        return {
            "available": True,
            "sha": sha,
            "worktree_state_available": False,
            "tracked_dirty": None,
            "tracked_change_count": None,
            "untracked_present": None,
            "untracked_count": None,
            "worktree_error": "; ".join(errors),
        }

    tracked_lines = [line for line in tracked.stdout.splitlines() if line]
    untracked_lines = [line for line in untracked.stdout.splitlines() if line]
    return {
        "available": True,
        "sha": sha,
        "worktree_state_available": True,
        "tracked_dirty": bool(tracked_lines),
        "tracked_change_count": len(tracked_lines),
        "untracked_present": bool(untracked_lines),
        "untracked_count": len(untracked_lines),
    }
