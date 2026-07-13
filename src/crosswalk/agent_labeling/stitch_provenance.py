"""Deterministic provenance for stitching-panel evidence packs.

The panel's durable unit of evidence is the exact option menu shown to a voter,
not merely the edge set the voter selected.  This module fingerprints that menu,
the displayed edge universe, and every prompt/image file in the generated pack.
The compact manifest is archived with votes so unselected edges and ``NONE``
ballots can later be interpreted without the git-ignored batch directory.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

EVIDENCE_SCHEMA_VERSION = 1
EVIDENCE_MANIFEST = "evidence.json"

_EDGE_PROVENANCE_KEYS = (
    "confidence",
    "selected",
    "decision",
    "review_reason",
    "optimizer_decision",
    "decision_reason",
    "pruned",
    "selected_elsewhere",
    "gers_start_frac",
    "gers_end_frac",
    "local_start_frac",
    "local_end_frac",
)


class EvidenceProvenanceError(ValueError):
    """Raised when a pack no longer matches its evidence manifest."""


def canonical_json(value: Any) -> str:
    """Stable, compact JSON used by every evidence fingerprint."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_descriptor(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    """Content identity for one batch input artifact."""
    path = Path(path)
    display = path
    if root is not None:
        try:
            display = path.resolve().relative_to(Path(root).resolve())
        except ValueError:
            display = path.resolve()
    if not path.exists():
        return {"path": str(display), "available": False}
    return {
        "path": str(display),
        "available": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _stable_id(value: Any, field: str) -> str:
    if value is None or isinstance(value, (bool, dict, list, tuple, set)):
        raise EvidenceProvenanceError(
            f"{field} must be a scalar string/number, got {type(value).__name__}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise EvidenceProvenanceError(f"{field} must be finite")
    text = str(value)
    if not text.strip():
        raise EvidenceProvenanceError(f"{field} must not be blank")
    if field == "group_id" and (text in {".", ".."} or "/" in text or "\\" in text):
        raise EvidenceProvenanceError(f"group_id is not a safe path component: {text!r}")
    return text


def safe_group_id(value: Any) -> str:
    """Return a group ID that is safe to use as one path component."""
    return _stable_id(value, "group_id")


def batch_group_map(batch: dict[str, Any]) -> dict[str, dict]:
    """Return a unique, path-safe group roster from ``batch.json``."""
    groups: dict[str, dict] = {}
    for group in batch.get("groups", []):
        group_id = safe_group_id(group.get("group_id"))
        if group_id in groups:
            raise EvidenceProvenanceError(f"duplicate group_id in batch roster: {group_id}")
        groups[group_id] = group
    return groups


def validate_manifest_against_batch(
    manifest: dict[str, Any],
    batch: dict[str, Any],
    group_id: str,
) -> None:
    """Bind a schema-v2 evidence pack to its exact batch group and inputs."""
    batch_schema = int(batch.get("schema_version") or 0)
    evidence_schema = (manifest.get("evidence") or {}).get("schema_version")
    if batch_schema < 2:
        if evidence_schema == EVIDENCE_SCHEMA_VERSION:
            raise EvidenceProvenanceError(
                "current evidence manifest requires a schema-v2 batch contract"
            )
        return
    if manifest.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise EvidenceProvenanceError("schema-v2 batch requires a current evidence manifest")
    group_id = safe_group_id(group_id)
    groups = batch_group_map(batch)
    if group_id not in groups:
        raise EvidenceProvenanceError(f"group {group_id} is absent from schema-v2 batch roster")
    evidence = manifest.get("evidence") or {}
    if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise EvidenceProvenanceError("schema-v2 batch requires current evidence records")
    for field in ("source_artifacts", "batch_generation_source"):
        if field not in batch or batch[field] is None:
            raise EvidenceProvenanceError(f"schema-v2 batch is missing {field}")
    expected_group_hash = sha256_json(groups[group_id])
    if evidence.get("source_group_sha256") != expected_group_hash:
        raise EvidenceProvenanceError(
            f"evidence source group does not match schema-v2 batch roster for {group_id}"
        )
    if evidence.get("source_artifacts") != batch.get("source_artifacts"):
        raise EvidenceProvenanceError(
            f"evidence source artifacts do not match schema-v2 batch for {group_id}"
        )
    if evidence.get("batch_generation_source") != batch.get("batch_generation_source"):
        raise EvidenceProvenanceError(
            f"evidence generation source does not match schema-v2 batch for {group_id}"
        )


def _edge_identity(edge: dict) -> dict[str, str]:
    return {
        "ref_id": _stable_id(edge.get("ref_id"), "ref_id"),
        "target_id": _stable_id(edge.get("target_id"), "target_id"),
    }


def _edge_record(edge: dict) -> dict[str, Any]:
    out: dict[str, Any] = _edge_identity(edge)
    for key in _EDGE_PROVENANCE_KEYS:
        if key in edge:
            value = edge[key]
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            out[key] = value
    return out


def _dedupe_sorted_edges(edges: list[dict]) -> list[dict[str, str]]:
    pairs = {(e["ref_id"], e["target_id"]) for e in map(_edge_identity, edges)}
    return [{"ref_id": ref_id, "target_id": target_id} for ref_id, target_id in sorted(pairs)]


def build_evidence_record(
    group: dict,
    options_ctx: dict,
    *,
    source_artifacts: dict[str, Any] | None = None,
    batch_generation_source: dict[str, Any] | None = None,
    options_pruned: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the exact post-prune menu made selectable to the panel."""
    group_id = safe_group_id(group.get("group_id"))
    if group.get("candidate_edges"):
        source_edges = list(group.get("candidate_edges") or [])
        source_universe_kind = "candidate_edges"
    else:
        source_edges = list(group.get("edges") or []) + list(group.get("rejected_edges") or [])
        source_universe_kind = "edges+rejected_edges"
    source_universe = _dedupe_sorted_edges(source_edges)

    displayed_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    option_menu: list[dict[str, Any]] = []
    for option in options_ctx.get("options", []):
        option_edges = [_edge_record(edge) for edge in option.get("edges", [])]
        identities = [
            {"ref_id": edge["ref_id"], "target_id": edge["target_id"]} for edge in option_edges
        ]
        for edge in option_edges:
            displayed_by_pair.setdefault((edge["ref_id"], edge["target_id"]), edge)
        option_menu.append(
            {
                "letter": str(option["letter"]),
                "option_id": sha256_json(
                    sorted(identities, key=lambda e: (e["ref_id"], e["target_id"]))
                ),
                "is_optimizer": bool(option.get("is_optimizer", False)),
                "edges": identities,
            }
        )

    displayed = [displayed_by_pair[key] for key in sorted(displayed_by_pair)]
    source_pairs = {(edge["ref_id"], edge["target_id"]) for edge in source_universe}
    displayed_pairs = {(edge["ref_id"], edge["target_id"]) for edge in displayed}
    missing_from_source = sorted(displayed_pairs - source_pairs)
    if missing_from_source:
        raise EvidenceProvenanceError(
            f"displayed options contain edges outside the source candidate universe: "
            f"{missing_from_source}"
        )
    base: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "group_id": group_id,
        "source_group_sha256": sha256_json(group),
        "source_universe_kind": source_universe_kind,
        "source_candidate_count": len(source_universe),
        "source_candidate_edges": source_universe,
        "source_candidate_universe_sha256": sha256_json(source_universe),
        "displayed_candidate_count": len(displayed),
        "displayed_candidate_universe_sha256": sha256_json(
            [{"ref_id": e["ref_id"], "target_id": e["target_id"]} for e in displayed]
        ),
        "displayed_edges": displayed,
        "option_menu": option_menu,
        "option_menu_sha256": sha256_json(option_menu),
        "optimizer_letter": options_ctx.get("optimizer_letter"),
        "selectable_choices": [*[o["letter"] for o in option_menu], "NONE"],
        "options_pruned": options_pruned,
        "source_artifacts": source_artifacts or {"status": "unavailable"},
        "batch_generation_source": batch_generation_source or {"status": "unavailable"},
    }
    base["evidence_id"] = sha256_json(base)
    return base


def _legacy_evidence_record(metadata: dict) -> dict[str, Any]:
    """Reconstruct an honest displayed-menu record for a pre-manifest pack."""
    ref_by_label = {
        str(segment["label"]): _stable_id(segment["id"], "ref_id")
        for segment in metadata.get("segments", {}).get("reference", [])
    }
    target_by_label = {
        str(segment["label"]): _stable_id(segment["id"], "target_id")
        for segment in metadata.get("segments", {}).get("target", [])
    }
    option_menu: list[dict[str, Any]] = []
    displayed: dict[tuple[str, str], dict[str, str]] = {}
    for option in metadata.get("options", []):
        identities: list[dict[str, str]] = []
        for edge in option.get("edges", []):
            ref_id = ref_by_label.get(str(edge.get("ref")), str(edge.get("ref")))
            target_id = target_by_label.get(str(edge.get("target")), str(edge.get("target")))
            item = {"ref_id": ref_id, "target_id": target_id}
            identities.append(item)
            displayed.setdefault((ref_id, target_id), item)
        option_menu.append(
            {
                "letter": str(option["letter"]),
                "option_id": sha256_json(
                    sorted(identities, key=lambda e: (e["ref_id"], e["target_id"]))
                ),
                "is_optimizer": bool(option.get("is_optimizer", False)),
                "edges": identities,
            }
        )
    displayed_edges = [displayed[key] for key in sorted(displayed)]
    base: dict[str, Any] = {
        "schema_version": 0,
        "group_id": safe_group_id(metadata.get("group_id")),
        "source_group_sha256": None,
        "source_universe_kind": "legacy_pack_displayed_only",
        "source_candidate_count": None,
        "source_candidate_edges": None,
        "source_candidate_universe_sha256": None,
        "displayed_candidate_count": len(displayed_edges),
        "displayed_candidate_universe_sha256": sha256_json(displayed_edges),
        "displayed_edges": displayed_edges,
        "option_menu": option_menu,
        "option_menu_sha256": sha256_json(option_menu),
        "optimizer_letter": metadata.get("optimizer_letter"),
        "selectable_choices": [*[o["letter"] for o in option_menu], "NONE"],
        "options_pruned": metadata.get("options_pruned"),
        "source_artifacts": {"status": "unavailable_legacy_pack"},
        "batch_generation_source": {"status": "unavailable_legacy_pack"},
    }
    base["evidence_id"] = sha256_json(base)
    return base


def _managed_pack_files(group_dir: Path) -> list[Path]:
    paths = [group_dir / "metadata.yaml", group_dir / "prompt.txt", group_dir / "overview.png"]
    paths.extend(sorted(group_dir.glob("option_*.png")))
    paths.extend(sorted(group_dir.glob("zoom_*.png")))
    return [path for path in paths if path.is_file()]


def write_evidence_manifest(
    group_dir: Path,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Hash the complete generated pack and write ``evidence.json``."""
    group_dir = Path(group_dir)
    files = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in _managed_pack_files(group_dir)
    ]
    manifest: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "group_id": evidence["group_id"],
        "evidence_id": evidence["evidence_id"],
        "evidence": evidence,
        "files": files,
    }
    manifest["evidence_pack_sha256"] = sha256_json(
        {"evidence_id": evidence["evidence_id"], "files": files}
    )
    (group_dir / EVIDENCE_MANIFEST).write_text(canonical_json(manifest) + "\n")
    return manifest


def load_evidence_manifest(group_dir: Path, *, allow_legacy: bool = True) -> dict[str, Any]:
    """Load and verify a manifest, or materialize one for a legacy pack."""
    group_dir = Path(group_dir)
    path = group_dir / EVIDENCE_MANIFEST
    if not path.exists():
        if not allow_legacy:
            raise EvidenceProvenanceError(f"missing {path}")
        import yaml

        metadata = yaml.safe_load((group_dir / "metadata.yaml").read_text())
        if metadata.get("evidence") is not None:
            raise EvidenceProvenanceError(
                f"missing {path} for a provenance-aware evidence pack; regenerate the pack"
            )
        evidence = _legacy_evidence_record(metadata)
        if evidence["group_id"] != safe_group_id(group_dir.name):
            raise EvidenceProvenanceError(
                "legacy evidence group identity mismatch: "
                f"directory={group_dir.name!r}, metadata={evidence['group_id']!r}"
            )
        return write_evidence_manifest(group_dir, evidence)

    manifest = json.loads(path.read_text())
    evidence = manifest.get("evidence") or {}
    claimed_evidence_id = evidence.pop("evidence_id", None)
    actual_evidence_id = sha256_json(evidence)
    evidence["evidence_id"] = claimed_evidence_id
    if (
        claimed_evidence_id != actual_evidence_id
        or manifest.get("evidence_id") != actual_evidence_id
    ):
        raise EvidenceProvenanceError(
            f"evidence identity mismatch for group {manifest.get('group_id')}"
        )

    directory_group_id = safe_group_id(group_dir.name)
    manifest_group_id = safe_group_id(manifest.get("group_id"))
    evidence_group_id = safe_group_id(evidence.get("group_id"))
    if len({directory_group_id, manifest_group_id, evidence_group_id}) != 1:
        raise EvidenceProvenanceError(
            "evidence group identity mismatch: "
            f"directory={directory_group_id!r}, manifest={manifest_group_id!r}, "
            f"evidence={evidence_group_id!r}"
        )

    expected_names = [item["path"] for item in manifest.get("files", [])]
    current_names = [path.name for path in _managed_pack_files(group_dir)]
    if current_names != expected_names:
        raise EvidenceProvenanceError(
            f"managed evidence files changed for group {manifest.get('group_id')}; "
            "regenerate the pack"
        )
    actual_files = [
        {
            "path": item["path"],
            "bytes": (group_dir / item["path"]).stat().st_size,
            "sha256": sha256_file(group_dir / item["path"]),
        }
        for item in manifest.get("files", [])
    ]
    if actual_files != manifest.get("files", []):
        raise EvidenceProvenanceError(
            f"evidence pack files changed for group {manifest.get('group_id')}; regenerate the pack"
        )
    import yaml

    metadata = yaml.safe_load((group_dir / "metadata.yaml").read_text())
    metadata_group_id = safe_group_id(metadata.get("group_id"))
    if metadata_group_id != directory_group_id:
        raise EvidenceProvenanceError(
            "evidence group identity mismatch: "
            f"directory={directory_group_id!r}, metadata={metadata_group_id!r}"
        )
    embedded_evidence = metadata.get("evidence")
    if embedded_evidence is not None and embedded_evidence != evidence:
        raise EvidenceProvenanceError(
            f"metadata evidence does not match manifest for group {directory_group_id}"
        )
    metadata_menu = _legacy_evidence_record(metadata)["option_menu"]
    if metadata_menu != evidence.get("option_menu"):
        raise EvidenceProvenanceError(
            f"metadata option menu does not match manifest for group {directory_group_id}"
        )
    actual_pack = sha256_json({"evidence_id": actual_evidence_id, "files": actual_files})
    if actual_pack != manifest.get("evidence_pack_sha256"):
        raise EvidenceProvenanceError(
            f"evidence pack hash mismatch for group {manifest.get('group_id')}"
        )
    return manifest


def invocation_signature(
    panel: list[Any],
    *,
    timeout: int | None,
    collect_feedback: bool,
    invocation_budget_s: float,
    effective_timeouts: list[int],
    runtime_contract_sha256: str,
) -> str:
    """Hash every panel/invocation knob that can change a ballot."""
    return sha256_json(
        {
            "providers": [
                {
                    "name": provider.name,
                    "model": provider.model,
                    "effort": provider.effort,
                    "timeout": provider.timeout,
                    "opencode_agent": provider.opencode_agent,
                    "routes": list(provider.routes),
                }
                for provider in panel
            ],
            "timeout_override": timeout,
            "effective_timeouts": effective_timeouts,
            "collect_feedback": collect_feedback,
            "invocation_budget_s": invocation_budget_s,
            "runtime_contract_sha256": runtime_contract_sha256,
        }
    )


def consensus_policy_signature(
    *,
    max_edges: int,
    min_voter_confidence: float,
    runtime_contract_sha256: str,
) -> str:
    """Hash routing policy state so resume cannot reuse stale consensus."""
    return sha256_json(
        {
            "policy_version": "2026-07-12.1",
            "stitch_export_backstop_max_edges": max_edges,
            "stitch_min_voter_confidence": min_voter_confidence,
            "runtime_contract_sha256": runtime_contract_sha256,
        }
    )
