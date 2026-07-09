"""Sync the publication staging tree to a target (local dir or Cloudflare R2).

Two targets:

* **local dir** (``--target-dir``) — copies the staging tree to a directory. No
  credentials, fully testable; used to validate publishing end-to-end offline.
* **R2** (default) — Cloudflare R2 is S3-compatible, so we shell out to the ``aws``
  CLI with ``--endpoint-url`` (matching the geocoder repo's pattern). Credentials
  come from environment variables (``R2_*``); they are never read from disk or
  stored.

Both honour **immutable release paths**: a ``release=<X>`` partition already
present at the target is *skipped* (never overwritten); only new releases and the
mutable top-level index files sync, so routine ``publish --all`` keeps working
release after release. ``--force`` intentionally re-publishes existing releases.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .publish import BRIDGES_PREFIX
from .publish_targets import TARGETS_PREFIX

# Environment variable names (documented; never persisted).
R2_ENV_VARS = {
    "endpoint": "R2_ENDPOINT_URL",  # e.g. https://<account>.r2.cloudflarestorage.com
    "access_key": "R2_ACCESS_KEY_ID",
    "secret_key": "R2_SECRET_ACCESS_KEY",
    "bucket": "R2_BUCKET",
}


@dataclass(frozen=True)
class R2Config:
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str


def r2_env() -> R2Config | None:
    """Read R2 credentials from the environment; None if any are missing."""
    vals = {k: os.environ.get(v) for k, v in R2_ENV_VARS.items()}
    if not all(vals.values()):
        return None
    return R2Config(
        endpoint=vals["endpoint"],
        access_key=vals["access_key"],
        secret_key=vals["secret_key"],
        bucket=vals["bucket"],
    )


def missing_r2_env() -> list[str]:
    """Names of the R2_* environment variables that are unset."""
    return [v for v in R2_ENV_VARS.values() if not os.environ.get(v)]


def staged_release_dirs(staging_dir: Path) -> list[str]:
    """Release identifiers present in a staging tree's ``bridges/`` prefix."""
    base = staging_dir / BRIDGES_PREFIX
    if not base.exists():
        return []
    return sorted(d.name.split("=", 1)[1] for d in base.glob("release=*"))


def staged_files(staging_dir: Path) -> list[Path]:
    """All regular files in a staging tree, sorted (for deterministic reporting)."""
    return sorted(p for p in staging_dir.rglob("*") if p.is_file())


# --------------------------------------------------------------------------
# Local-directory target
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SyncPlan:
    """Which staged releases sync vs are skipped as already published.

    Immutable release paths are enforced by *skipping*, not aborting: a release
    already present at the target is left untouched (never overwritten) and only
    new releases + the mutable top (``index.html`` / ``index.json``) sync. This
    keeps the go-live command (``publish --all --no-dry-run``) working release
    after release — aborting on the first existing release would permanently
    block multi-release publishing. ``--force`` re-publishes everything.
    """

    releases: list[str]
    skipped_releases: list[str]

    def excluded_prefixes(self) -> list[str]:
        return [f"{BRIDGES_PREFIX}/release={r}/" for r in self.skipped_releases]


def _plan(staging_dir: Path, existing: set[str], force: bool) -> SyncPlan:
    staged = staged_release_dirs(staging_dir)
    if force:
        return SyncPlan(releases=staged, skipped_releases=[])
    return SyncPlan(
        releases=[r for r in staged if r not in existing],
        skipped_releases=[r for r in staged if r in existing],
    )


def sync_local(
    staging_dir: Path, target_dir: Path, *, force: bool = False
) -> tuple[list[str], SyncPlan]:
    """Copy the staging tree into ``target_dir`` with immutable release paths.

    Releases already present at the target are skipped (never overwritten) unless
    ``force``; new releases and the top-level index files always sync. Returns
    ``(written_relative_paths, plan)``.
    """
    target_dir = Path(target_dir)
    target_bridges = target_dir / BRIDGES_PREFIX
    existing = {d.name.split("=", 1)[1] for d in target_bridges.glob("release=*") if d.is_dir()}
    plan = _plan(staging_dir, existing, force)
    excluded = plan.excluded_prefixes()

    written: list[str] = []
    for src in staged_files(staging_dir):
        rel = src.relative_to(staging_dir)
        rel_str = str(rel)
        if any(rel_str.startswith(p) for p in excluded):
            continue  # immutable: already published, leave untouched
        dst = target_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        written.append(rel_str)
    return written, plan


# --------------------------------------------------------------------------
# R2 target (aws CLI, S3-compatible)
# --------------------------------------------------------------------------
def build_aws_sync_argv(
    staging_dir: Path, cfg: R2Config, exclude_prefixes: list[str] | None = None
) -> list[str]:
    """Build the ``aws s3 sync`` argv for the staging tree.

    Credentials are passed via the environment (see :func:`aws_sync_env`), not on
    the command line, so they never appear in process listings.
    ``exclude_prefixes`` (already-published immutable releases) become
    ``--exclude`` patterns so they are never re-uploaded.
    """
    argv = [
        "aws",
        "s3",
        "sync",
        str(staging_dir),
        f"s3://{cfg.bucket}/",
        "--endpoint-url",
        cfg.endpoint,
        # Immutable release paths + a small mutable top: never delete remote files
        # (no --delete); re-uploads overwrite identical bytes idempotently.
    ]
    for prefix in exclude_prefixes or []:
        argv += ["--exclude", f"{prefix}*"]
    return argv


def aws_sync_env(cfg: R2Config) -> dict[str, str]:
    """Environment for the aws CLI: R2 creds mapped to AWS_* names."""
    env = dict(os.environ)
    env["AWS_ACCESS_KEY_ID"] = cfg.access_key
    env["AWS_SECRET_ACCESS_KEY"] = cfg.secret_key
    # R2 ignores region but the CLI wants one set.
    env.setdefault("AWS_DEFAULT_REGION", "auto")
    return env


def remote_prefix_exists(cfg: R2Config, prefix: str) -> bool:
    """Whether ANY object exists under ``prefix`` in R2. **Fails closed**: an
    error on the check (network/auth) raises rather than reading as "absent",
    so a broken check can never bypass an immutability guard built on top of
    this (release-level for bridges, snapshot-level for targets).

    Uses ``s3api list-objects-v2`` (not ``s3 ls``, whose exit code conflates
    "empty" with "error" across CLI versions).
    """
    import json as _json

    proc = subprocess.run(
        [
            "aws",
            "s3api",
            "list-objects-v2",
            "--bucket",
            cfg.bucket,
            "--prefix",
            prefix,
            "--max-items",
            "1",
            "--output",
            "json",
            "--endpoint-url",
            cfg.endpoint,
        ],
        env=aws_sync_env(cfg),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Could not check whether {prefix} exists in R2 "
            f"(refusing to assume it is absent): {proc.stderr.strip()}"
        )
    out = proc.stdout.strip()
    if not out:
        return False
    return bool(_json.loads(out).get("Contents"))


def remote_release_exists(cfg: R2Config, release: str) -> bool:
    """Whether a ``release=<X>`` prefix already exists in R2 (see :func:`remote_prefix_exists`)."""
    return remote_prefix_exists(cfg, f"{BRIDGES_PREFIX}/release={release}/")


def sync_r2(
    staging_dir: Path, cfg: R2Config, *, force: bool = False
) -> tuple[subprocess.CompletedProcess, SyncPlan]:
    """Run ``aws s3 sync`` to R2 with immutable release paths.

    Releases already present remotely are excluded from the sync (never
    re-uploaded) unless ``force``; new releases and the mutable top-level index
    files always sync. Returns ``(completed_process, plan)``.
    """
    existing: set[str] = set()
    if not force:
        for rel in staged_release_dirs(staging_dir):
            if remote_release_exists(cfg, rel):
                existing.add(rel)
    plan = _plan(staging_dir, existing, force)
    argv = build_aws_sync_argv(staging_dir, cfg, exclude_prefixes=plan.excluded_prefixes())
    proc = subprocess.run(argv, env=aws_sync_env(cfg), check=True)
    return proc, plan


# --------------------------------------------------------------------------
# Target-snapshot sync (targets/dataset=*/snapshot=* prefix)
# --------------------------------------------------------------------------
# Mirrors the bridge sync above, but the immutable unit is a single dataset's
# snapshot (``targets/dataset=<name>/snapshot=<date>/``) rather than a whole
# release — each dataset publishes on its own fetch cadence, so releases would
# be the wrong granularity for locking. Reuses the same R2Config/aws-CLI
# machinery (``build_aws_sync_argv``, ``aws_sync_env``, ``remote_prefix_exists``)
# rather than a second client path.
def target_snapshot_prefix(dataset: str, snapshot: str) -> str:
    return f"{TARGETS_PREFIX}/dataset={dataset}/snapshot={snapshot}/"


def remote_target_snapshot_exists(cfg: R2Config, dataset: str, snapshot: str) -> bool:
    """Whether ``targets/dataset=<dataset>/snapshot=<snapshot>/`` already exists in R2."""
    return remote_prefix_exists(cfg, target_snapshot_prefix(dataset, snapshot))


def staged_target_snapshots(staging_dir: Path) -> list[tuple[str, str]]:
    """``(dataset, snapshot)`` pairs present in a staging tree's ``targets/`` prefix."""
    base = staging_dir / TARGETS_PREFIX
    if not base.exists():
        return []
    out: list[tuple[str, str]] = []
    for ds_dir in sorted(base.glob("dataset=*")):
        name = ds_dir.name.split("=", 1)[1]
        for snap_dir in sorted(ds_dir.glob("snapshot=*")):
            out.append((name, snap_dir.name.split("=", 1)[1]))
    return sorted(out)


@dataclass(frozen=True)
class TargetSyncPlan:
    """Which staged ``(dataset, snapshot)`` pairs sync vs are skipped as already
    published. Same skip-don't-abort semantics as :class:`SyncPlan` (see its
    docstring): a snapshot already at the target is left untouched unless
    ``force``.
    """

    snapshots: list[tuple[str, str]]
    skipped_snapshots: list[tuple[str, str]]

    def excluded_prefixes(self) -> list[str]:
        return [target_snapshot_prefix(ds, snap) for ds, snap in self.skipped_snapshots]


def _plan_targets(staging_dir: Path, existing: set[tuple[str, str]], force: bool) -> TargetSyncPlan:
    staged = staged_target_snapshots(staging_dir)
    if force:
        return TargetSyncPlan(snapshots=staged, skipped_snapshots=[])
    return TargetSyncPlan(
        snapshots=[s for s in staged if s not in existing],
        skipped_snapshots=[s for s in staged if s in existing],
    )


def sync_targets_local(
    staging_dir: Path, target_dir: Path, *, force: bool = False
) -> tuple[list[str], TargetSyncPlan]:
    """Copy the ``targets/`` staging tree into ``target_dir`` with immutable snapshots.

    Snapshots already present at the target are skipped (never overwritten)
    unless ``force``; new snapshots and the mutable ``latest.json`` / top-level
    ``targets/index.json`` always sync. Returns ``(written_relative_paths, plan)``.
    """
    target_dir = Path(target_dir)
    existing = set(staged_target_snapshots(target_dir))
    plan = _plan_targets(staging_dir, existing, force)
    excluded = plan.excluded_prefixes()

    written: list[str] = []
    for src in staged_files(staging_dir):
        rel = src.relative_to(staging_dir)
        rel_str = str(rel)
        if any(rel_str.startswith(p) for p in excluded):
            continue  # immutable: already published, leave untouched
        dst = target_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        written.append(rel_str)
    return written, plan


def sync_targets_r2(
    staging_dir: Path, cfg: R2Config, *, force: bool = False
) -> tuple[subprocess.CompletedProcess, TargetSyncPlan]:
    """Run ``aws s3 sync`` to R2 with immutable target snapshots.

    Snapshots already present remotely are excluded from the sync (never
    re-uploaded) unless ``force``; new snapshots and the mutable top-level
    index files always sync. Returns ``(completed_process, plan)``.
    """
    existing: set[tuple[str, str]] = set()
    if not force:
        for ds, snap in staged_target_snapshots(staging_dir):
            if remote_target_snapshot_exists(cfg, ds, snap):
                existing.add((ds, snap))
    plan = _plan_targets(staging_dir, existing, force)
    argv = build_aws_sync_argv(staging_dir, cfg, exclude_prefixes=plan.excluded_prefixes())
    proc = subprocess.run(argv, env=aws_sync_env(cfg), check=True)
    return proc, plan
