"""Sync the publication staging tree to a target (local dir or Cloudflare R2).

Two targets:

* **local dir** (``--target-dir``) — copies the staging tree to a directory. No
  credentials, fully testable; used to validate publishing end-to-end offline.
* **R2** (default) — Cloudflare R2 is S3-compatible, so we shell out to the ``aws``
  CLI with ``--endpoint-url`` (matching the geocoder repo's pattern). Credentials
  come from environment variables (``R2_*``); they are never read from disk or
  stored.

Both honour **immutable release paths**: a ``release=<X>`` partition already present
at the target is refused unless ``--force`` (a re-run of the *same* release is
idempotent — identical bytes — so ``--force`` is only needed to intentionally
replace one).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .publish import BRIDGES_PREFIX

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
def sync_local(staging_dir: Path, target_dir: Path, *, force: bool = False) -> list[str]:
    """Copy the staging tree into ``target_dir`` with immutable release paths.

    Returns the list of relative file paths written. Raises ``FileExistsError`` if
    a ``release=<X>`` partition already exists at the target and ``force`` is False.
    """
    target_dir = Path(target_dir)
    target_bridges = target_dir / BRIDGES_PREFIX
    if not force:
        for rel in staged_release_dirs(staging_dir):
            existing = target_bridges / f"release={rel}"
            if existing.exists():
                raise FileExistsError(
                    f"release={rel} already published at {existing}. Published "
                    "releases are immutable — pass --force to intentionally replace it."
                )
    written: list[str] = []
    for src in staged_files(staging_dir):
        rel = src.relative_to(staging_dir)
        dst = target_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        written.append(str(rel))
    return written


# --------------------------------------------------------------------------
# R2 target (aws CLI, S3-compatible)
# --------------------------------------------------------------------------
def build_aws_sync_argv(staging_dir: Path, cfg: R2Config) -> list[str]:
    """Build the ``aws s3 sync`` argv for the whole staging tree.

    Credentials are passed via the environment (see :func:`aws_sync_env`), not on
    the command line, so they never appear in process listings.
    """
    return [
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


def aws_sync_env(cfg: R2Config) -> dict[str, str]:
    """Environment for the aws CLI: R2 creds mapped to AWS_* names."""
    env = dict(os.environ)
    env["AWS_ACCESS_KEY_ID"] = cfg.access_key
    env["AWS_SECRET_ACCESS_KEY"] = cfg.secret_key
    # R2 ignores region but the CLI wants one set.
    env.setdefault("AWS_DEFAULT_REGION", "auto")
    return env


def remote_release_exists(cfg: R2Config, release: str) -> bool:
    """Best-effort check whether a ``release=<X>`` prefix already exists in R2."""
    prefix = f"s3://{cfg.bucket}/{BRIDGES_PREFIX}/release={release}/"
    proc = subprocess.run(
        ["aws", "s3", "ls", prefix, "--endpoint-url", cfg.endpoint],
        env=aws_sync_env(cfg),
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def sync_r2(
    staging_dir: Path, cfg: R2Config, *, force: bool = False
) -> subprocess.CompletedProcess:
    """Run ``aws s3 sync`` to R2, refusing to clobber an existing release.

    Returns the completed process. Raises ``FileExistsError`` if a release is
    already present remotely and ``force`` is False.
    """
    if not force:
        for rel in staged_release_dirs(staging_dir):
            if remote_release_exists(cfg, rel):
                raise FileExistsError(
                    f"release={rel} already exists in R2 bucket '{cfg.bucket}'. "
                    "Published releases are immutable — pass --force to replace it."
                )
    argv = build_aws_sync_argv(staging_dir, cfg)
    return subprocess.run(argv, env=aws_sync_env(cfg), check=True)
