"""Matcher tool adapter - shells out to `matcher stitch` CLI."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import pandas as pd
from loguru import logger

from mbench.adapters.base import EvalMode, ToolOutput

# Default invocation. Uses ``uv run`` so the matcher CLI resolves from the
# matcher project environment regardless of what is on ``PATH`` in the caller's
# shell. Executed with ``cwd`` set to the repo root (see ``_find_repo_root``) so
# matcher's relative model path (``data/models/...``) resolves correctly.
DEFAULT_MATCHER_CMD = "uv run matcher"


def _find_repo_root(start: Path | None = None) -> Path:
    """Locate the matcher repo root.

    Walks up from this file (or ``start``) looking for the directory that
    contains ``src/matcher`` — the matcher package root. With an editable
    install (``pip install -e``) this file lives at
    ``<repo>/mbench/src/mbench/adapters/matcher.py``, so the walk finds
    ``<repo>``. Falls back to the 4th parent if no marker is found.
    """
    here = (start or Path(__file__)).resolve()
    for parent in here.parents:
        if (parent / "src" / "matcher").is_dir():
            return parent
    # Fallback: mbench/src/mbench/adapters/matcher.py -> repo root is parents[4]
    return here.parents[4] if len(here.parents) > 4 else here.parent


def _groups_sidecar_path(bridge_path: Path) -> Path:
    """Locate the groups sidecar JSON alongside a bridge parquet.

    Mirrors ``matcher.filenames.groups_sidecar_path``:
    ``.../bridge.parquet`` -> ``.../bridge_groups.json``.
    """
    stem = bridge_path.stem
    if stem.endswith("_bridge"):
        stem = stem[: -len("_bridge")] + "_groups"
    else:
        stem = stem + "_groups"
    return bridge_path.parent / f"{stem}.json"


class MatcherAdapter:
    """Adapter for the matcher road conflation tool.

    Runs `matcher stitch` via subprocess and parses the bridge parquet output.
    """

    name: str = "matcher"
    eval_mode: EvalMode = EvalMode.STITCH

    def run(self, reference: Path, target: Path, output_dir: Path, **kwargs) -> Path:
        """Run matcher stitch and return path to bridge parquet.

        Args:
            reference: Path to reference (Overture) parquet.
            target: Path to target parquet.
            output_dir: Directory for output files.
            **kwargs: Extra options passed as CLI flags.
                model: Model type (default: "xgboost").
                matcher_cmd: How to invoke matcher (default: "uv run matcher").
                    Split with shlex; e.g. "matcher" to use a binary on PATH.
                repo_root: Directory to run matcher from (default: auto-detected
                    matcher repo root). matcher resolves its model path relative
                    to this directory.

        Returns:
            Path to the bridge parquet output.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        # Resolve to absolute paths: the subprocess runs with cwd=repo_root, so
        # any caller-relative paths would otherwise be resolved incorrectly.
        bridge_path = (output_dir / "bridge.parquet").resolve()
        reference = reference.resolve()
        target = target.resolve()

        model = kwargs.get("model", "xgboost")

        matcher_cmd = kwargs.get("matcher_cmd") or DEFAULT_MATCHER_CMD
        base_cmd = shlex.split(str(matcher_cmd))

        repo_root = kwargs.get("repo_root")
        repo_root = Path(repo_root).resolve() if repo_root else _find_repo_root()

        cmd = [
            *base_cmd,
            "stitch",
            "-r",
            str(reference),
            "-t",
            str(target),
            "-m",
            model,
            "-o",
            str(bridge_path),
        ]

        timeout = int(kwargs.get("timeout", 3600))

        logger.info(f"Running (cwd={repo_root}): {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=repo_root
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"matcher stitch timed out after {timeout}s") from exc

        if result.returncode != 0:
            logger.error(f"matcher stitch failed:\n{result.stderr}")
            raise RuntimeError(f"matcher stitch exited with code {result.returncode}")

        if not bridge_path.exists():
            raise FileNotFoundError(f"Expected output not found: {bridge_path}")

        logger.info(f"Matcher output: {bridge_path}")
        return bridge_path

    def parse_output(self, output_path: Path) -> ToolOutput:
        """Parse matcher bridge parquet into standardized ToolOutput.

        Bridge parquet has columns: gers_id, local_id, confidence, match_type, ...
        Maps gers_id -> ref_id, local_id -> target_id.
        """
        bridge = pd.read_parquet(output_path)
        logger.info(f"Loaded bridge with {len(bridge)} matches")

        if "confidence" in bridge.columns:
            confidence = bridge["confidence"].astype(float)
        else:
            confidence = pd.Series(1.0, index=bridge.index)

        matches = pd.DataFrame(
            {
                "ref_id": bridge["gers_id"].astype(str),
                "target_id": bridge["local_id"].astype(str),
                "confidence": confidence,
            }
        )

        if "match_type" in bridge.columns:
            match_type_counts = bridge["match_type"].value_counts().to_dict()
        else:
            match_type_counts = {}

        # Load the M:N groups sidecar (for stitch-level eval), if present.
        groups = None
        sidecar_path = _groups_sidecar_path(output_path)
        if sidecar_path.exists():
            try:
                groups = json.loads(sidecar_path.read_text()).get("groups")
                logger.info(f"Loaded groups sidecar with {len(groups or [])} groups")
            except (ValueError, OSError) as exc:
                logger.warning(f"Failed to read groups sidecar {sidecar_path}: {exc}")

        metadata = {
            "match_type_counts": match_type_counts,
            "total_rows": len(bridge),
            "has_groups_sidecar": groups is not None,
        }

        return ToolOutput(matches=matches, metadata=metadata, groups=groups)
