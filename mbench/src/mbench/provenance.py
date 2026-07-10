"""Reproducibility metadata for benchmark result records."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

_FINGERPRINT_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}


def file_fingerprint(path: Path) -> dict[str, Any]:
    """Return a content-addressed fingerprint for one benchmark input.

    The sha256 is a full-content hash. Within one process, results are cached
    by ``(path, size, mtime_ns)`` so a reference file shared across a batch is
    hashed once; a file replaced mid-batch with identical size AND preserved
    mtime (``cp -p`` / ``rsync -t``) would return the cached hash.
    """
    resolved = path.resolve()
    stat = resolved.stat()
    cache_key = (str(resolved), stat.st_size, stat.st_mtime_ns)
    cached = _FINGERPRINT_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    digest = hashlib.sha256()
    with resolved.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    fingerprint = {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }
    _FINGERPRINT_CACHE[cache_key] = fingerprint
    return dict(fingerprint)


def _safe_fingerprint(path: Path) -> dict[str, Any]:
    """Fingerprint best-effort; provenance must never discard a completed run."""
    try:
        return file_fingerprint(path)
    except (OSError, ValueError) as exc:
        return {"path": str(path.resolve()), "error": str(exc)}


def _optional_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return _safe_fingerprint(path)


def _find_label_file(labels_dir: Path, dataset: str) -> Path | None:
    base = labels_dir / f"dataset={dataset}"
    for name in ("data.csv", "data.parquet"):
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def _jsonable(value: Any) -> Any:
    """Convert CLI/adapter options to stable JSON-compatible values."""
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        # Sort on str(key): mixed-type keys (e.g. int/str value_counts) must
        # not raise TypeError — provenance is best-effort by contract.
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _repo_root(start: Path) -> Path | None:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def git_provenance(repo: Path | None = None) -> dict[str, Any]:
    """Capture revision and distinguish tracked dirt from untracked files."""
    root = repo or _repo_root(Path.cwd())
    if root is None:
        return {"available": False}

    try:
        head = _git(root, "rev-parse", "HEAD")
        tracked = _git(root, "status", "--short", "--untracked-files=no")
        untracked = _git(root, "ls-files", "--others", "--exclude-standard")
    except OSError as exc:
        return {"available": False, "root": str(root.resolve()), "error": str(exc)}
    if head.returncode != 0:
        return {"available": False, "root": str(root)}

    if tracked.returncode != 0 or untracked.returncode != 0:
        errors = []
        if tracked.returncode != 0:
            errors.append(f"git status exited {tracked.returncode}: {tracked.stderr.strip()}")
        if untracked.returncode != 0:
            errors.append(f"git ls-files exited {untracked.returncode}: {untracked.stderr.strip()}")
        return {
            "available": True,
            "root": str(root.resolve()),
            "sha": head.stdout.strip(),
            "worktree_state_available": False,
            "worktree_error": "; ".join(errors),
            "tracked_dirty": None,
            "tracked_change_count": None,
            "untracked_present": None,
            "untracked_count": None,
        }

    tracked_lines = [line for line in tracked.stdout.splitlines() if line]
    untracked_lines = [line for line in untracked.stdout.splitlines() if line]
    return {
        "available": True,
        "root": str(root.resolve()),
        "sha": head.stdout.strip(),
        "worktree_state_available": True,
        "tracked_dirty": bool(tracked_lines),
        "tracked_change_count": len(tracked_lines),
        "untracked_present": bool(untracked_lines),
        "untracked_count": len(untracked_lines),
    }


def _crosswalk_provenance(*, verifiable: bool, reasons: list[str]) -> dict[str, Any]:
    """Capture the active Crosswalk model, feature version, and pipeline knobs."""
    if not verifiable:
        return {"verifiable": False, "unverifiable_reasons": reasons}
    try:
        from crosswalk.config import DATA_VERSION, FEATURE_VERSION
        from crosswalk.factory.manifest import settings_snapshot
        from crosswalk.pipeline.runner import _default_model_path

        model_path = _default_model_path()
        return {
            "verifiable": True,
            "feature_version": FEATURE_VERSION,
            "data_version": DATA_VERSION,
            "model": _optional_fingerprint(model_path),
            "settings": _jsonable(settings_snapshot()),
        }
    except Exception as exc:  # pragma: no cover - optional across standalone installs
        return {
            "verifiable": False,
            "unverifiable_reasons": [f"crosswalk provenance unavailable: {exc}"],
        }


def collect_provenance(
    *,
    tool: str,
    dataset: str,
    reference: Path,
    target: Path,
    labels_dir: Path,
    stitch_labels_dir: Path | None,
    match_level: str,
    run_options: dict[str, Any],
    adapter_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a self-contained provenance record for a benchmark run."""
    adapter_metadata = adapter_metadata or {}
    inputs: dict[str, Any] = {
        "reference": _safe_fingerprint(reference),
        "target": _safe_fingerprint(target),
    }
    labels_file = _find_label_file(labels_dir, dataset)
    if labels_file is not None:
        inputs["labels"] = _safe_fingerprint(labels_file)
    if stitch_labels_dir is not None:
        stitch_file = stitch_labels_dir / f"dataset={dataset}" / "data.csv"
        if stitch_file.is_file():
            inputs["stitch_labels"] = _safe_fingerprint(stitch_file)
    connectors = run_options.get("connectors")
    if connectors is not None:
        try:
            connector_path = Path(connectors)
            if connector_path.is_file():
                inputs["connectors"] = _safe_fingerprint(connector_path)
        except (TypeError, OSError, ValueError):
            inputs["connectors"] = {"path": repr(connectors), "error": "not a usable path"}

    working_directory_raw = adapter_metadata.get("working_directory")
    working_directory = (
        Path(working_directory_raw).resolve() if working_directory_raw else Path.cwd().resolve()
    )
    effective_repo = _repo_root(working_directory)
    git = (
        git_provenance(effective_repo)
        if effective_repo is not None
        else {"available": False, "root": str(working_directory)}
    )

    provenance: dict[str, Any] = {
        "schema_version": 1,
        "git": git,
        "inputs": inputs,
        "invocation": {
            "tool": tool,
            "dataset": dataset,
            "match_level": match_level,
            "run_options": _jsonable(run_options),
            "adapter_metadata": _jsonable(adapter_metadata),
        },
    }
    if tool == "crosswalk":
        reasons: list[str] = []
        if adapter_metadata.get("crosswalk_command_is_default") is not True:
            reasons.append("custom or unknown Crosswalk executable")
        local_repo = _repo_root(Path(__file__))
        if effective_repo is None or local_repo is None or effective_repo != local_repo:
            reasons.append("effective working directory does not match the loaded mbench checkout")
        if Path.cwd().resolve() != working_directory:
            # Crosswalk resolves its model path and .env relative to cwd; the
            # subprocess ran in working_directory, so fingerprinting from a
            # different mbench cwd could certify the wrong model/settings.
            reasons.append(
                "mbench process cwd differs from the adapter working directory; "
                "cwd-relative model/settings resolution is unreliable"
            )
        provenance["crosswalk"] = _crosswalk_provenance(
            verifiable=not reasons,
            reasons=reasons,
        )
    return provenance
