"""Factory manifest: provenance, staleness keys, and settings snapshot.

Each factory dataset output carries a ``manifest.json`` recording the inputs
(file fingerprints), the model + FEATURE_VERSION, a snapshot of the pipeline
settings (including the resolver-prune allowlist state), timings, and the
resulting counts/group stats. The manifest drives incremental/resume:

* ``score_key`` — hash of everything that changes the SCORES (input fingerprints,
  model fingerprint, FEATURE_VERSION, DATA_VERSION, buffer distance). A cached
  scored-candidate parquet is valid iff its manifest's ``score_key`` still matches.
* ``optimize_key`` — hash of the optimizer/prune/export settings snapshot. These
  can change without invalidating scores (drives ``reoptimize``).
* ``full_key`` — hash of both. ``factory run`` skips a dataset whose existing
  manifest has the same ``full_key`` (unless ``--force``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import DATA_VERSION, FEATURE_VERSION, settings

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"

# Group edge-count threshold above which a group is flagged a "monster" (matches
# the scale-out audit's >20-edge bucket in research/scaleout_readiness_2026_07.md).
MONSTER_EDGE_THRESHOLD = 20


def file_fingerprint(path: Path) -> dict[str, Any]:
    """Cheap, deterministic fingerprint of an input file (size + mtime).

    Mirrors the labeling cache's model-fingerprint philosophy: fast (no content
    hash of large parquets) and stable on a given box, where raw data is fetched
    rather than git-tracked. Returns a dict so it is human-readable in the manifest.
    """
    st = path.stat()
    return {"name": path.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def model_fingerprint() -> dict[str, Any]:
    """Fingerprint the active ML model file (``settings.model_path``)."""
    model_path = Path(settings.model_path)
    if not model_path.exists():
        return {"name": str(model_path), "present": False}
    st = model_path.stat()
    return {
        "name": model_path.name,
        "present": True,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def settings_snapshot() -> dict[str, Any]:
    """Snapshot the pipeline settings that shape optimization/prune/export.

    This is recorded in the manifest for provenance AND hashed into
    ``optimize_key``. Includes the resolver-prune allowlist STATE so a change to
    the tuned thresholds re-triggers a run. ``model_path`` / FEATURE_VERSION are
    tracked separately (they belong to ``score_key``).
    """
    from ..config import DEFAULT_SNAP_TOLERANCE_M

    return {
        "bridge_min_confidence": settings.bridge_min_confidence,
        "enable_calibration": settings.enable_calibration,
        "enable_score_propagation": settings.enable_score_propagation,
        "optimizer_corridor_aware": settings.optimizer_corridor_aware,
        "optimizer_corridor_max_turn_deg": settings.optimizer_corridor_max_turn_deg,
        "optimizer_glue_min_confidence": settings.optimizer_glue_min_confidence,
        "optimizer_glue_min_confidence_raw": settings.optimizer_glue_min_confidence_raw,
        "optimizer_review_threshold": settings.optimizer_review_threshold,
        # The optimizer's contiguity tolerance is the module constant, not the
        # settings.snap_tolerance_m field — snapshot what the optimizer reads.
        "contiguity_tolerance_m": DEFAULT_SNAP_TOLERANCE_M,
        "stitch_export_max_assignment_components": settings.stitch_export_max_assignment_components,
        "stitch_export_soft_max_edges": settings.stitch_export_soft_max_edges,
        "stitch_export_backstop_max_edges": settings.stitch_export_backstop_max_edges,
        "stitch_persist_rejected_edges": settings.stitch_persist_rejected_edges,
        "stitch_rejected_edges_max_per_group": settings.stitch_rejected_edges_max_per_group,
        "stitch_persist_candidate_graph": settings.stitch_persist_candidate_graph,
        "stitch_persist_candidates": settings.stitch_persist_candidates,
        "resolver_prune_enabled": settings.resolver_prune_enabled,
        "resolver_prune_overrides": dict(settings.resolver_prune_overrides or {}),
    }


def _stable_hash(obj: Any) -> str:
    """Deterministic short hash of a JSON-able object (sorted keys)."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def compute_score_key(
    ref_fp: dict,
    target_fp: dict,
    model_fp: dict,
    buffer_distance_m: float,
    method: str = "xgboost",
    cache_schema_version: int | None = None,
) -> str:
    """Hash of everything that changes the scored candidates (or their on-disk form).

    ``cache_schema_version`` folds the scored-cache parquet layout version into the
    key so a reader/writer layout change invalidates old caches even when the
    scores themselves would be unchanged.
    """
    from .scored_cache import SCORED_CACHE_SCHEMA_VERSION

    return _stable_hash(
        {
            "ref": ref_fp,
            "target": target_fp,
            "model": model_fp,
            "feature_version": FEATURE_VERSION,
            "data_version": DATA_VERSION,
            "buffer_distance_m": buffer_distance_m,
            "method": method,
            "cache_schema_version": (
                cache_schema_version
                if cache_schema_version is not None
                else SCORED_CACHE_SCHEMA_VERSION
            ),
        }
    )


def compute_optimize_key(snapshot: dict, min_confidence: float = 0.1) -> str:
    """Hash of the optimizer/prune/export settings snapshot + optimizer args."""
    return _stable_hash({"snapshot": snapshot, "min_confidence": min_confidence})


def compute_full_key(score_key: str, optimize_key: str) -> str:
    """Hash binding scores + optimization; the incremental-skip identity."""
    return _stable_hash({"score": score_key, "optimize": optimize_key})


def compute_group_stats(groups_json_path: Path) -> dict[str, Any]:
    """Summarize a groups sidecar JSON for the manifest.

    Returns zeros when the sidecar is absent (a run with no M:N/1:N/N:1 groups).
    """
    stats: dict[str, Any] = {
        "n_groups": 0,
        "n_one_to_n": 0,
        "n_n_to_one": 0,
        "n_m_to_n": 0,
        "edge_count_mean": 0.0,
        "edge_count_p50": 0,
        "edge_count_p90": 0,
        "edge_count_p99": 0,
        "edge_count_max": 0,
        "n_monster": 0,
        "n_oversized": 0,
    }
    if not groups_json_path.exists():
        return stats

    data = json.loads(groups_json_path.read_text())
    groups = data.get("groups", [])
    if not groups:
        return stats

    import numpy as np

    edge_counts = np.array([int(g.get("n_edges", 0)) for g in groups], dtype=float)
    stats["n_groups"] = len(groups)
    stats["n_one_to_n"] = sum(1 for g in groups if g.get("match_type") == "1:N")
    stats["n_n_to_one"] = sum(1 for g in groups if g.get("match_type") == "N:1")
    stats["n_m_to_n"] = sum(1 for g in groups if g.get("match_type") == "M:N")
    stats["edge_count_mean"] = round(float(edge_counts.mean()), 3)
    stats["edge_count_p50"] = int(np.percentile(edge_counts, 50))
    stats["edge_count_p90"] = int(np.percentile(edge_counts, 90))
    stats["edge_count_p99"] = int(np.percentile(edge_counts, 99))
    stats["edge_count_max"] = int(edge_counts.max())
    stats["n_monster"] = int((edge_counts > MONSTER_EDGE_THRESHOLD).sum())
    stats["n_oversized"] = sum(1 for g in groups if g.get("oversized_group"))
    return stats


@dataclass
class Manifest:
    """Serialized provenance + result record for one factory dataset output."""

    dataset: str
    release: str
    schema_version: int = MANIFEST_SCHEMA_VERSION
    created_at: str = ""
    feature_version: str = FEATURE_VERSION
    data_version: str = DATA_VERSION
    buffer_distance_m: float = 75.0
    method: str = "xgboost"

    # Provenance / staleness
    inputs: dict[str, Any] = field(default_factory=dict)  # {ref, target, connectors}
    model: dict[str, Any] = field(default_factory=dict)
    settings_snapshot: dict[str, Any] = field(default_factory=dict)
    score_key: str = ""
    optimize_key: str = ""
    full_key: str = ""

    # Timings (seconds)
    score_wall_s: float | None = None
    optimize_wall_s: float | None = None
    wall_s: float | None = None

    # Counts
    n_reference: int = 0
    n_target: int = 0
    n_candidates: int = 0
    n_matched: int = 0
    n_review: int = 0
    n_unmatched: int = 0

    # Group stats (from the sidecar)
    groups: dict[str, Any] = field(default_factory=dict)

    # Cache
    scored_cache: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def read(cls, path: Path) -> Manifest:
        data = json.loads(path.read_text())
        # Tolerate unknown/extra keys from newer schema versions.
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @staticmethod
    def now_iso() -> str:
        return datetime.now(UTC).isoformat()
