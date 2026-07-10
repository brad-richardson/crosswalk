"""Crosswalk tool adapter - shells out to `crosswalk stitch` CLI."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import pandas as pd
from loguru import logger

from mbench.adapters.base import EvalMode, ToolOutput

# Default invocation. Uses ``uv run`` so the crosswalk CLI resolves from the
# crosswalk project environment regardless of what is on ``PATH`` in the caller's
# shell. Executed with ``cwd`` set to the repo root (see ``_find_repo_root``) so
# crosswalk's relative model path (``data/models/...``) resolves correctly.
DEFAULT_CROSSWALK_CMD = "uv run crosswalk"


def _find_repo_root(start: Path | None = None) -> Path:
    """Locate the crosswalk repo root.

    Walks up from this file (or ``start``) looking for the directory that
    contains ``src/crosswalk`` — the crosswalk package root. With an editable
    install (``pip install -e``) this file lives at
    ``<repo>/mbench/src/mbench/adapters/crosswalk.py``, so the walk finds
    ``<repo>``. Falls back to the 4th parent if no marker is found.
    """
    here = (start or Path(__file__)).resolve()
    for parent in here.parents:
        if (parent / "src" / "crosswalk").is_dir():
            return parent
    # Fallback: mbench/src/mbench/adapters/crosswalk.py -> repo root is parents[4]
    return here.parents[4] if len(here.parents) > 4 else here.parent


def _groups_sidecar_path(bridge_path: Path) -> Path:
    """Locate the groups sidecar JSON alongside a bridge parquet.

    Mirrors ``crosswalk.filenames.groups_sidecar_path``:
    ``.../bridge.parquet`` -> ``.../bridge_groups.json``.
    """
    stem = bridge_path.stem
    if stem.endswith("_bridge"):
        stem = stem[: -len("_bridge")] + "_groups"
    else:
        stem = stem + "_groups"
    return bridge_path.parent / f"{stem}.json"


def _validated_groups_sidecar(data: object) -> list[dict]:
    """Validate the minimum sidecar schema required for stitch evaluation."""
    if not isinstance(data, dict):
        raise ValueError("groups sidecar root must be an object")
    groups = data.get("groups")
    if not isinstance(groups, list):
        raise ValueError("groups sidecar 'groups' must be a list")
    if not groups:
        raise ValueError("groups sidecar must contain at least one group")

    seen_group_ids: set[str] = set()
    total_edges = 0
    for group_index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"groups[{group_index}] must be an object")
        raw_group_id = group.get("group_id")
        group_id = "" if raw_group_id is None else str(raw_group_id).strip()
        if not group_id:
            raise ValueError(f"groups[{group_index}].group_id must be nonblank")
        if group_id in seen_group_ids:
            raise ValueError(f"duplicate group_id in groups sidecar: {group_id}")
        seen_group_ids.add(group_id)

        edges = group.get("edges")
        if not isinstance(edges, list) or not edges:
            raise ValueError(f"groups[{group_index}].edges must be a nonempty list")
        for edge_index, edge in enumerate(edges):
            if not isinstance(edge, dict):
                raise ValueError(f"groups[{group_index}].edges[{edge_index}] must be an object")
            for key in ("ref_id", "target_id"):
                raw_id = edge.get(key)
                if raw_id is None or not str(raw_id).strip():
                    raise ValueError(
                        f"groups[{group_index}].edges[{edge_index}].{key} must be nonblank"
                    )
        total_edges += len(edges)

    if total_edges == 0:  # Defensive; nonempty per-group edges already imply this.
        raise ValueError("groups sidecar candidate edge universe is empty")
    return groups


class CrosswalkAdapter:
    """Adapter for the crosswalk road conflation tool.

    Runs `crosswalk stitch` via subprocess and parses the bridge parquet output.
    """

    name: str = "crosswalk"
    eval_mode: EvalMode = EvalMode.STITCH
    decision_aware: bool = True

    def __init__(self) -> None:
        self._last_run_metadata: dict = {}

    def run(self, reference: Path, target: Path, output_dir: Path, **kwargs) -> Path:
        """Run crosswalk stitch and return path to bridge parquet.

        Args:
            reference: Path to reference (Overture) parquet.
            target: Path to target parquet.
            output_dir: Directory for output files.
            **kwargs: Extra options passed as CLI flags.
                model: Model type (default: "xgboost").
                dataset: Dataset name (e.g. "us_boston_streets"). When provided it
                    is passed to ``crosswalk stitch`` as the positional dataset
                    argument ALONGSIDE the explicit ``-r``/``-t`` paths. This is
                    what engages crosswalk's resolver-prune allowlist, which keys
                    on dataset identity (never the file paths) since #350. Without
                    it the stitch runs prune-OFF and evaluates a different row set
                    than production — ~5pt below the calibrated gate floor (#372).
                    The mbench runner injects this automatically from the dataset
                    being benchmarked.
                crosswalk_cmd: How to invoke crosswalk (default: "uv run crosswalk").
                    Split with shlex; e.g. "crosswalk" to use a binary on PATH.
                    The deprecated ``matcher_cmd`` key is still accepted.
                repo_root: Directory to run crosswalk from (default: auto-detected
                    crosswalk repo root). crosswalk resolves its model path relative
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

        crosswalk_cmd = (
            kwargs.get("crosswalk_cmd") or kwargs.get("matcher_cmd") or DEFAULT_CROSSWALK_CMD
        )
        base_cmd = shlex.split(str(crosswalk_cmd))

        repo_root = kwargs.get("repo_root")
        repo_root = Path(repo_root).resolve() if repo_root else _find_repo_root()

        # Pass the dataset NAME as the positional argument (in addition to the
        # explicit -r/-t paths) so crosswalk's resolver-prune allowlist engages.
        # `crosswalk stitch <dataset> -r ... -t ...` resolves the prune by dataset
        # identity while still using the exact paths mbench resolved — matching the
        # production/factory path the gate floors were calibrated on (#372). With
        # no dataset name the prune keys to None and stays OFF, scoring a different
        # (unpruned) row set ~5pt below the floor.
        dataset = kwargs.get("dataset")

        cmd = [*base_cmd, "stitch"]
        if dataset:
            cmd.append(str(dataset))
        cmd += [
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

        self._last_run_metadata = {
            "effective_command": cmd,
            "working_directory": str(repo_root),
            "timeout_s": timeout,
            "model": str(model),
            "dataset": str(dataset) if dataset is not None else None,
            "crosswalk_command_is_default": str(crosswalk_cmd) == DEFAULT_CROSSWALK_CMD,
        }

        logger.info(f"Running (cwd={repo_root}): {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=repo_root
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"crosswalk stitch timed out after {timeout}s") from exc

        if result.returncode != 0:
            logger.error(f"crosswalk stitch failed:\n{result.stderr}")
            raise RuntimeError(f"crosswalk stitch exited with code {result.returncode}")

        if not bridge_path.exists():
            raise FileNotFoundError(f"Expected output not found: {bridge_path}")

        logger.info(f"crosswalk output: {bridge_path}")
        return bridge_path

    def parse_output(self, output_path: Path) -> ToolOutput:
        """Parse crosswalk bridge parquet into standardized ToolOutput.

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
        if "match_decision" not in bridge.columns:
            raise ValueError(
                "Crosswalk bridge missing required match_decision column; refusing "
                "to evaluate review rows as published matches"
            )
        if bridge["match_decision"].isna().any():
            raise ValueError("Crosswalk bridge match_decision contains null values")
        normalized_decisions = bridge["match_decision"].astype("string").str.lower().str.strip()
        unknown = sorted(set(normalized_decisions) - {"match", "review", "no_match"})
        if unknown:
            raise ValueError(f"Crosswalk bridge match_decision has unknown values: {unknown}")
        # Preserve the pipeline's publication decision. The runner performs the
        # canonical non-null/allowed-value validation before evaluation.
        matches["match_decision"] = normalized_decisions

        if "match_type" in bridge.columns:
            match_type_counts = bridge["match_type"].value_counts().to_dict()
        else:
            match_type_counts = {}

        # Load the M:N groups sidecar (for stitch-level eval), if present.
        groups = None
        sidecar_path = _groups_sidecar_path(output_path)
        if sidecar_path.exists():
            try:
                groups = _validated_groups_sidecar(json.loads(sidecar_path.read_text()))
                logger.info(f"Loaded groups sidecar with {len(groups or [])} groups")
            except (ValueError, OSError) as exc:
                logger.warning(f"Failed to read groups sidecar {sidecar_path}: {exc}")

        metadata = {
            "match_type_counts": match_type_counts,
            "total_rows": len(bridge),
            "has_groups_sidecar": groups is not None,
            **self._last_run_metadata,
        }

        return ToolOutput(matches=matches, metadata=metadata, groups=groups)
