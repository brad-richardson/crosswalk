"""Codex-only repeated-draw diagnostics for immutable stitching evidence packs.

This module is intentionally separate from :mod:`stitch_runner`'s production
panel and consensus paths. It reuses one provider invocation at a time, but it
never writes ``votes.csv`` / ``consensus.csv``, never computes consensus, and
never emits label-export-compatible artifacts. Diagnostic findings are stored
under a dedicated ``.no-export`` tree and remain hypotheses until checked
against human labels and a fresh heterogeneous panel.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import re
import tempfile
import threading
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from . import stitch_runner as _stitch_runner
from .stitch_provenance import (
    invocation_signature,
    load_evidence_manifest,
    sha256_file,
    validate_manifest_against_batch,
)
from .stitch_runner import (
    ProviderSpec,
    Vote,
    get_panel,
    resolve_timeout,
    run_provider_on_group,
)
from .wave_manifest import WaveManifest

DIAGNOSTIC_SCHEMA_VERSION = 1
DEFAULT_CANONICAL_DRAWS = 3
DEFAULT_WORKERS = 5
MAX_WORKERS = 10
NO_EXPORT_MARKER = ".no-export"
PLAN_FILENAME = "diagnostic_manifest.json"
PLAN_DIGEST_FIELD = "diagnostic_manifest_sha256"
RESULT_DIGEST_FIELD = "result_sha256"

NONE_REASONS = {
    "all_edges_no_match",
    "no_exact_option",
    "insufficient_evidence",
}

AUDIT_INSTRUCTION = """

ADDITIONAL DIAGNOSTIC SELF-AUDIT (does not change the matching rubric): include
a fourth key named "pack_feedback" in the SAME JSON object as choice,
confidence, and reasoning. Use this exact object shape:
{
  "none_reason": "all_edges_no_match" | "no_exact_option" |
                 "insufficient_evidence" | null,
  "desired_edges": [{"ref_id": "R1", "target_id": "T2"}],
  "closest_option": "A" | "B" | ... | "NONE",
  "disputed_edges": [
    {
      "ref_id": "R1",
      "target_id": "T2",
      "verdict": "include" | "exclude",
      "reason": "short explanation",
      "missing_evidence": ["facts that would resolve the edge"]
    }
  ],
  "missing_info": ["information absent from the pack"],
  "ambiguities": ["genuine ambiguities"],
  "rubric_gaps": ["rubric wording or policy gaps"],
  "strongest_counterargument": "best case against your answer",
  "confidence_basis": "what your confidence rests on"
}

Use the visible R#/T# segment LABELS in ref_id/target_id fields, not hidden
source identifiers. If choice is NONE, none_reason is required. Otherwise it
must be null. If none_reason is no_exact_option, desired_edges must be the
non-empty exact set you wanted and it must differ from every displayed option.
Use empty arrays or an empty string when a field has nothing to report.
"""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _dict_digest(value: dict[str, Any], excluded_field: str) -> str:
    payload = {key: item for key, item in value.items() if key != excluded_field}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(value, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _safe_component(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    return text or "unknown"


def _group_key(row: dict[str, Any]) -> str:
    return f"{row['dataset_id']}::{row['group_id']}"


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _dataset_holdout_targets(
    groups_by_dataset: dict[str, list[str]], holdout_count: int, total_groups: int
) -> dict[str, int]:
    exact = {
        dataset: len(groups) * holdout_count / total_groups
        for dataset, groups in groups_by_dataset.items()
    }
    targets = {dataset: int(value) for dataset, value in exact.items()}
    remaining = holdout_count - sum(targets.values())
    order = sorted(exact, key=lambda ds: (-(exact[ds] - targets[ds]), ds))
    for dataset in order[:remaining]:
        targets[dataset] += 1
    return targets


def assign_cohorts(
    schedule: list[dict[str, Any]],
    *,
    holdout_groups: int,
    holdout_factorial_groups: int,
    seed: int,
) -> dict[str, str]:
    """Deterministically split unique groups while keeping all variants together."""
    rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in schedule:
        rows_by_group[_group_key(row)].append(row)
    total = len(rows_by_group)
    if not 0 <= holdout_groups < total:
        raise ValueError(f"holdout_groups must be in [0, {total - 1}], got {holdout_groups}")

    factorial = sorted(
        (key for key, rows in rows_by_group.items() if len(rows) > 1),
        key=lambda key: _rank(seed, f"factorial:{key}"),
    )
    if holdout_factorial_groups > len(factorial):
        raise ValueError(
            f"requested {holdout_factorial_groups} factorial holdouts, only {len(factorial)} exist"
        )
    if holdout_factorial_groups > holdout_groups:
        raise ValueError("factorial holdouts cannot exceed total holdout groups")

    selected = set(factorial[:holdout_factorial_groups])
    groups_by_dataset: dict[str, list[str]] = defaultdict(list)
    for key, rows in rows_by_group.items():
        groups_by_dataset[str(rows[0]["dataset_id"])].append(key)
    targets = _dataset_holdout_targets(groups_by_dataset, holdout_groups, total)

    for dataset in sorted(groups_by_dataset):
        candidates = sorted(
            (key for key in groups_by_dataset[dataset] if key not in selected),
            key=lambda key: _rank(seed, f"dataset:{key}"),
        )
        need = max(0, targets[dataset] - len(selected.intersection(groups_by_dataset[dataset])))
        for key in candidates[:need]:
            if len(selected) < holdout_groups:
                selected.add(key)

    if len(selected) < holdout_groups:
        remaining = sorted(
            (key for key in rows_by_group if key not in selected),
            key=lambda key: _rank(seed, f"remainder:{key}"),
        )
        selected.update(remaining[: holdout_groups - len(selected)])
    if len(selected) != holdout_groups:
        raise AssertionError("cohort allocator did not produce the requested holdout size")
    return {key: ("holdout" if key in selected else "development") for key in rows_by_group}


def _codex_provider(panel: list[ProviderSpec]) -> ProviderSpec:
    seats = [provider for provider in panel if provider.name == "codex"]
    if len(seats) != 1:
        raise ValueError(
            f"diagnostic source panel must contain exactly one codex seat, got {len(seats)}"
        )
    provider = seats[0]
    if provider.model != "gpt-5.6-sol" or provider.effort != "high":
        raise ValueError(
            "deep diagnostic requires the v7 Codex seat gpt-5.6-sol/high; "
            f"got {provider.model}/{provider.effort}"
        )
    return provider


def _runtime_hashes() -> dict[str, str]:
    return {
        "diagnostic_runtime_sha256": sha256_file(Path(__file__)),
        "stitch_runner_sha256": sha256_file(Path(_stitch_runner.__file__)),
        "audit_instruction_sha256": hashlib.sha256(AUDIT_INSTRUCTION.encode()).hexdigest(),
    }


def _label_maps(group_dir: Path) -> dict[str, dict[str, str]]:
    metadata = yaml.safe_load((group_dir / "metadata.yaml").read_text())
    segments = metadata.get("segments", {})
    return {
        "reference": {
            str(segment["label"]): str(segment["id"]) for segment in segments.get("reference", [])
        },
        "target": {
            str(segment["label"]): str(segment["id"]) for segment in segments.get("target", [])
        },
    }


def _pack_record(row: dict[str, Any], cohort: str) -> dict[str, Any]:
    batch_dir = Path(str(row["batch_dir"])).resolve()
    group_id = str(row["group_id"])
    group_dir = batch_dir / group_id
    batch_path = batch_dir / "batch.json"
    batch = json.loads(batch_path.read_text())
    evidence_manifest = load_evidence_manifest(group_dir, allow_legacy=False)
    validate_manifest_against_batch(evidence_manifest, batch, group_id)
    evidence = evidence_manifest["evidence"]
    prompt_path = group_dir / "prompt.txt"
    pack_key = "__".join(
        (
            f"{int(row['run_index']):03d}",
            _safe_component(row["dataset_id"]),
            _safe_component(group_id),
            _safe_component(row["variant"]),
        )
    )
    return {
        "pack_key": pack_key,
        "run_index": int(row["run_index"]),
        "dataset_id": str(row["dataset_id"]),
        "group_id": group_id,
        "variant": str(row["variant"]),
        "batch_dir": str(batch_dir),
        "batch_json_sha256": sha256_file(batch_path),
        "group_dir": str(group_dir),
        "cohort": cohort,
        "source_prompt_sha256": sha256_file(prompt_path),
        "evidence_id": str(evidence["evidence_id"]),
        "evidence_pack_sha256": str(evidence_manifest["evidence_pack_sha256"]),
        "displayed_candidate_universe_sha256": str(evidence["displayed_candidate_universe_sha256"]),
        "option_menu_sha256": str(evidence["option_menu_sha256"]),
        "label_maps": _label_maps(group_dir),
    }


def _pass_records(
    provider: ProviderSpec,
    *,
    canonical_draws: int,
    timeout: int,
    invocation_budget_s: float,
    runtime: dict[str, str],
) -> list[dict[str, Any]]:
    if canonical_draws < 2:
        raise ValueError("canonical_draws must be at least 2")
    passes = []
    for draw_index in range(1, canonical_draws + 1):
        passes.append(
            {
                "pass_id": f"canonical-{draw_index}",
                "kind": "canonical",
                "draw_index": draw_index,
                "collect_feedback": False,
            }
        )
    passes.append(
        {
            "pass_id": "audit",
            "kind": "audit",
            "draw_index": None,
            "collect_feedback": True,
        }
    )
    for pass_record in passes:
        pass_record["invocation_signature_sha256"] = invocation_signature(
            [provider],
            timeout=timeout,
            collect_feedback=bool(pass_record["collect_feedback"]),
            invocation_budget_s=invocation_budget_s,
            effective_timeouts=[resolve_timeout(provider, timeout)],
            runtime_contract_sha256=runtime["stitch_runner_sha256"],
        )
    return passes


def build_diagnostic_plan(
    source_manifest_path: Path,
    output_dir: Path,
    *,
    run_id: str,
    canonical_draws: int = DEFAULT_CANONICAL_DRAWS,
    holdout_groups: int = 15,
    holdout_factorial_groups: int = 2,
    seed: int = 20260716,
    timeout: int = 600,
    invocation_budget_s: float = 300.0,
) -> dict[str, Any]:
    """Validate immutable inputs and construct a complete diagnostic plan in memory."""
    source_manifest_path = Path(source_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    timeout = int(timeout)
    invocation_budget_s = float(invocation_budget_s)
    wave = WaveManifest.load_validated(source_manifest_path)
    provider = _codex_provider(wave.panel)
    schedule = wave.run_schedule
    cohorts = assign_cohorts(
        schedule,
        holdout_groups=holdout_groups,
        holdout_factorial_groups=holdout_factorial_groups,
        seed=seed,
    )

    source_batch_dirs = {Path(str(row["batch_dir"])).resolve() for row in schedule}
    for batch_dir in source_batch_dirs:
        if output_dir == batch_dir or output_dir.is_relative_to(batch_dir):
            raise ValueError(f"diagnostic output must not be inside source batch dir {batch_dir}")

    packs = [_pack_record(row, cohorts[_group_key(row)]) for row in schedule]
    runtime = _runtime_hashes()
    passes = _pass_records(
        provider,
        canonical_draws=canonical_draws,
        timeout=timeout,
        invocation_budget_s=invocation_budget_s,
        runtime=runtime,
    )
    pack_counts = Counter(f"{pack['dataset_id']}::{pack['group_id']}" for pack in packs)
    ordinary_candidates = [
        pack for pack in packs if pack_counts[f"{pack['dataset_id']}::{pack['group_id']}"] == 1
    ]
    factorial_candidates = [
        pack for pack in packs if pack_counts[f"{pack['dataset_id']}::{pack['group_id']}"] > 1
    ]
    if not ordinary_candidates or not factorial_candidates:
        raise ValueError("diagnostic slate requires at least one ordinary and one factorial pack")
    factorial = next(
        (pack for pack in factorial_candidates if pack["cohort"] == "development"),
        factorial_candidates[0],
    )
    ordinary = next(
        (
            pack
            for pack in ordinary_candidates
            if pack["cohort"] == "development" and pack["dataset_id"] != factorial["dataset_id"]
        ),
        next(
            (pack for pack in ordinary_candidates if pack["cohort"] == "development"),
            ordinary_candidates[0],
        ),
    )
    assignments = [
        {
            "dataset_id": key.split("::", 1)[0],
            "group_id": key.split("::", 1)[1],
            "cohort": cohort,
        }
        for key, cohort in sorted(cohorts.items())
    ]
    plan: dict[str, Any] = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "purpose": "codex_repeated_draw_diagnostic_only_no_labels_no_consensus",
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output_dir),
        "source_manifest": {
            "path": str(source_manifest_path),
            "sha256": sha256_file(source_manifest_path),
            "wave": str(wave.content.get("wave", "")),
            "total_pack_count": len(schedule),
            "unique_group_count": len(cohorts),
        },
        "provider": {
            "name": provider.name,
            "model": provider.model,
            "effort": provider.effort,
        },
        "timeout": timeout,
        "invocation_budget_s": invocation_budget_s,
        "canonical_draws": canonical_draws,
        "passes": passes,
        "cohort_policy": {
            "seed": seed,
            "holdout_group_count": holdout_groups,
            "holdout_factorial_group_count": holdout_factorial_groups,
            "assignments": assignments,
        },
        "packs": packs,
        "smoke_pack_keys": [ordinary["pack_key"], factorial["pack_key"]],
        "planned_call_count": len(packs) * len(passes),
        "runtime": runtime,
    }
    plan[PLAN_DIGEST_FIELD] = _dict_digest(plan, PLAN_DIGEST_FIELD)
    return plan


def write_diagnostic_plan(plan: dict[str, Any]) -> Path:
    output_dir = Path(plan["output_dir"])
    plan_path = output_dir / PLAN_FILENAME
    if plan_path.exists():
        raise FileExistsError(f"diagnostic plan already exists: {plan_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / NO_EXPORT_MARKER).write_text(
        "Diagnostic-only repeated draws. Never use for label export or panel consensus.\n"
    )
    (output_dir / "holdout").mkdir(exist_ok=True)
    (output_dir / "holdout" / ".sealed").write_text(
        "Holdout results are excluded from analysis until a fix-frozen marker is supplied.\n"
    )
    _atomic_write_json(
        output_dir / "cohort_assignment.json",
        {
            "run_id": plan["run_id"],
            **plan["cohort_policy"],
            "diagnostic_manifest_sha256": plan[PLAN_DIGEST_FIELD],
        },
    )
    _atomic_write_json(plan_path, plan)
    return plan_path


def _load_plan(plan_path: Path) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    if not isinstance(plan, dict):
        raise ValueError("diagnostic manifest root must be an object")
    if plan.get("schema_version") != DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError(f"unsupported diagnostic schema {plan.get('schema_version')!r}")
    expected = _dict_digest(plan, PLAN_DIGEST_FIELD)
    if plan.get(PLAN_DIGEST_FIELD) != expected:
        raise ValueError("diagnostic manifest digest mismatch")
    if plan_path != Path(plan["output_dir"]).resolve() / PLAN_FILENAME:
        raise ValueError("diagnostic manifest is not located in its bound output_dir")
    return plan


def validate_diagnostic_plan(plan_path: Path, *, enforce_runtime: bool = True) -> dict[str, Any]:
    """Revalidate every source/runtime binding before a run or analysis.

    ``enforce_runtime`` re-hashes the *live* diagnostic/stitch-runner runtime and
    rejects drift. This guard exists to keep ballot **collection** isolated: a run
    must execute under the exact runtime the plan was built with. It is disabled
    for pure re-analysis of already-collected results — each frozen result is
    already provably bound to its collection runtime via its per-result digest and
    ``invocation_signature_sha256`` (which incorporates the stored
    ``stitch_runner_sha256``). Analysis-only code in this module is expected to
    evolve (e.g. reporting bug fixes), so re-hashing this file during analysis
    would wrongly block re-analysis without adding any integrity guarantee.
    """
    plan = _load_plan(plan_path)
    marker = Path(plan["output_dir"]) / NO_EXPORT_MARKER
    if not marker.is_file():
        raise ValueError(f"diagnostic output is missing {NO_EXPORT_MARKER}")
    source = plan["source_manifest"]
    if sha256_file(Path(source["path"])) != source["sha256"]:
        raise ValueError("source wave manifest changed after diagnostic planning")
    if enforce_runtime and _runtime_hashes() != plan["runtime"]:
        raise ValueError("diagnostic or stitch-runner runtime changed; create a new run_id")

    provider = _codex_provider(get_panel("v7-candidate"))
    if plan["provider"] != {
        "name": provider.name,
        "model": provider.model,
        "effort": provider.effort,
    }:
        raise ValueError("diagnostic provider binding drifted")
    if plan["planned_call_count"] != len(plan["packs"]) * len(plan["passes"]):
        raise ValueError("planned_call_count does not match packs × passes")
    expected_passes = _pass_records(
        provider,
        canonical_draws=int(plan["canonical_draws"]),
        timeout=int(plan["timeout"]),
        invocation_budget_s=float(plan["invocation_budget_s"]),
        runtime=plan["runtime"],
    )
    if plan["passes"] != expected_passes:
        raise ValueError("diagnostic pass definitions or invocation signatures drifted")

    for pack in plan["packs"]:
        group_dir = Path(pack["group_dir"])
        batch_path = Path(pack["batch_dir"]) / "batch.json"
        if sha256_file(batch_path) != pack["batch_json_sha256"]:
            raise ValueError(f"source batch changed for {pack['pack_key']}")
        if sha256_file(group_dir / "prompt.txt") != pack["source_prompt_sha256"]:
            raise ValueError(f"source prompt changed for {pack['pack_key']}")
        evidence_manifest = load_evidence_manifest(group_dir, allow_legacy=False)
        batch = json.loads(batch_path.read_text())
        validate_manifest_against_batch(evidence_manifest, batch, str(pack["group_id"]))
        evidence = evidence_manifest["evidence"]
        bindings = {
            "evidence_id": evidence["evidence_id"],
            "evidence_pack_sha256": evidence_manifest["evidence_pack_sha256"],
            "displayed_candidate_universe_sha256": evidence["displayed_candidate_universe_sha256"],
            "option_menu_sha256": evidence["option_menu_sha256"],
        }
        if any(str(pack[key]) != str(value) for key, value in bindings.items()):
            raise ValueError(f"evidence binding changed for {pack['pack_key']}")
    return plan


def _option_context(
    evidence_manifest: dict[str, Any],
) -> tuple[list[str], dict[str, list[tuple[str, str]]]]:
    menu = evidence_manifest["evidence"]["option_menu"]
    letters = [str(option["letter"]) for option in menu]
    options = {
        str(option["letter"]): [
            (str(edge["ref_id"]), str(edge["target_id"])) for edge in option["edges"]
        ]
        for option in menu
    }
    return letters, options


def _normalize_feedback_edge(
    raw: Any, pack: dict[str, Any], candidate_edges: set[tuple[str, str]]
) -> tuple[tuple[str, str] | None, str | None]:
    if not isinstance(raw, dict):
        return None, "edge entry must be an object"
    ref = str(raw.get("ref_id", "")).strip()
    target = str(raw.get("target_id", "")).strip()
    ref_id = pack["label_maps"]["reference"].get(ref, ref)
    target_id = pack["label_maps"]["target"].get(target, target)
    edge = (ref_id, target_id)
    if edge not in candidate_edges:
        return None, f"edge {ref!r}->{target!r} is outside the displayed candidate universe"
    return edge, None


def validate_pack_feedback(
    raw_feedback: str,
    *,
    choice: str,
    pack: dict[str, Any],
    evidence_manifest: dict[str, Any],
) -> tuple[str, dict[str, Any] | None, list[str]]:
    """Validate audit feedback independently of the otherwise-valid ballot."""
    if not raw_feedback:
        return "missing", None, ["pack_feedback was not returned"]
    try:
        feedback = json.loads(raw_feedback)
    except (TypeError, ValueError) as exc:
        return "invalid", None, [f"pack_feedback is not JSON: {exc}"]
    if not isinstance(feedback, dict):
        return "invalid", None, ["pack_feedback root must be an object"]

    evidence = evidence_manifest["evidence"]
    candidate_edges = {
        (str(edge["ref_id"]), str(edge["target_id"])) for edge in evidence["displayed_edges"]
    }
    option_sets = {
        str(option["letter"]): frozenset(
            (str(edge["ref_id"]), str(edge["target_id"])) for edge in option["edges"]
        )
        for option in evidence["option_menu"]
    }
    errors: list[str] = []
    raw_none_reason = feedback.get("none_reason")
    none_reason = raw_none_reason if isinstance(raw_none_reason, str) else None
    if raw_none_reason is not None and not isinstance(raw_none_reason, str):
        errors.append("none_reason must be a string or null")
    if choice == "NONE":
        if none_reason not in NONE_REASONS:
            errors.append("NONE choice requires a valid none_reason")
    elif none_reason is not None:
        errors.append("non-NONE choice requires none_reason=null")

    desired_raw = feedback.get("desired_edges")
    desired: list[tuple[str, str]] = []
    if not isinstance(desired_raw, list):
        errors.append("desired_edges must be a list")
    else:
        for edge_raw in desired_raw:
            edge, error = _normalize_feedback_edge(edge_raw, pack, candidate_edges)
            if error:
                errors.append(error)
            elif edge is not None:
                desired.append(edge)
    desired_set = frozenset(desired)
    if none_reason == "no_exact_option":
        if not desired_set:
            errors.append("no_exact_option requires a non-empty desired edge set")
        if desired_set in option_sets.values():
            errors.append("no_exact_option desired edge set exactly matches a displayed option")

    closest = feedback.get("closest_option")
    allowed_choices = set(evidence.get("selectable_choices", []))
    if not isinstance(closest, str) or closest not in allowed_choices:
        errors.append("closest_option must be a displayed choice")

    disputed_normalized = []
    disputed = feedback.get("disputed_edges")
    if not isinstance(disputed, list):
        errors.append("disputed_edges must be a list")
    else:
        for item in disputed:
            edge, error = _normalize_feedback_edge(item, pack, candidate_edges)
            if error:
                errors.append(error)
                continue
            verdict = item.get("verdict") if isinstance(item, dict) else None
            if not isinstance(verdict, str) or verdict not in {"include", "exclude"}:
                errors.append("disputed edge verdict must be include or exclude")
                continue
            missing = item.get("missing_evidence", [])
            if not isinstance(missing, list) or not all(
                isinstance(value, str) for value in missing
            ):
                errors.append("disputed edge missing_evidence must be a list of strings")
                continue
            disputed_normalized.append(
                {
                    "ref_id": edge[0],
                    "target_id": edge[1],
                    "verdict": verdict,
                    "reason": str(item.get("reason", "")),
                    "missing_evidence": missing,
                }
            )

    for field in ("missing_info", "ambiguities", "rubric_gaps"):
        value = feedback.get(field)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            errors.append(f"{field} must be a list of strings")
    for field in ("strongest_counterargument", "confidence_basis"):
        if not isinstance(feedback.get(field), str):
            errors.append(f"{field} must be a string")

    normalized = {
        **feedback,
        "none_reason": none_reason,
        "desired_edges_normalized": [
            {"ref_id": ref_id, "target_id": target_id} for ref_id, target_id in sorted(desired_set)
        ],
        "disputed_edges_normalized": disputed_normalized,
    }
    return ("valid" if not errors else "invalid"), normalized, errors


def _call_id(pack: dict[str, Any], pass_record: dict[str, Any]) -> str:
    return f"{pack['pack_key']}::{pass_record['pass_id']}"


def _result_path(plan: dict[str, Any], pack: dict[str, Any], pass_id: str) -> Path:
    cohort_root = "holdout" if pack["cohort"] == "holdout" else "development"
    return Path(plan["output_dir"]) / cohort_root / "results" / pack["pack_key"] / f"{pass_id}.json"


def _validate_existing_result(
    result: dict[str, Any],
    *,
    plan: dict[str, Any],
    pack: dict[str, Any],
    pass_record: dict[str, Any],
) -> None:
    if result.get(RESULT_DIGEST_FIELD) != _dict_digest(result, RESULT_DIGEST_FIELD):
        raise ValueError(f"result digest mismatch for {_call_id(pack, pass_record)}")
    expected = {
        "diagnostic_manifest_sha256": plan[PLAN_DIGEST_FIELD],
        "call_id": _call_id(pack, pass_record),
        "pack_key": pack["pack_key"],
        "pass_id": pass_record["pass_id"],
        "evidence_pack_sha256": pack["evidence_pack_sha256"],
        "source_prompt_sha256": pack["source_prompt_sha256"],
        "invocation_signature_sha256": pass_record["invocation_signature_sha256"],
        "provider": plan["provider"]["name"],
        "model": plan["provider"]["model"],
        "effort": plan["provider"]["effort"],
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"stale or mismatched result for {_call_id(pack, pass_record)}")


def _run_one_call(
    plan: dict[str, Any],
    pack: dict[str, Any],
    pass_record: dict[str, Any],
    provider: ProviderSpec,
    invoke: Callable[..., Vote],
) -> dict[str, Any]:
    group_dir = Path(pack["group_dir"])
    evidence_manifest = load_evidence_manifest(group_dir, allow_legacy=False)
    letters, options = _option_context(evidence_manifest)
    source_prompt = (group_dir / "prompt.txt").read_text()
    prompt = source_prompt + AUDIT_INSTRUCTION if pass_record["kind"] == "audit" else source_prompt
    vote = invoke(
        provider,
        pack["group_id"],
        group_dir,
        prompt,
        letters,
        options,
        timeout=int(plan["timeout"]),
        retries=1,
        collect_feedback=bool(pass_record["collect_feedback"]),
        invocation_budget_s=float(plan["invocation_budget_s"]),
        evidence_manifest=evidence_manifest,
    )
    if vote.provider != provider.name or vote.model != provider.model:
        raise ValueError(
            f"provider identity drift for {_call_id(pack, pass_record)}: "
            f"got {vote.provider}/{vote.model}, expected {provider.name}/{provider.model}"
        )
    feedback_status = "not_requested"
    normalized_feedback = None
    feedback_errors: list[str] = []
    if pass_record["kind"] == "audit":
        feedback_status, normalized_feedback, feedback_errors = validate_pack_feedback(
            vote.pack_feedback,
            choice=vote.choice,
            pack=pack,
            evidence_manifest=evidence_manifest,
        )
    result: dict[str, Any] = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_manifest_sha256": plan[PLAN_DIGEST_FIELD],
        "run_id": plan["run_id"],
        "call_id": _call_id(pack, pass_record),
        "pack_key": pack["pack_key"],
        "run_index": pack["run_index"],
        "dataset_id": pack["dataset_id"],
        "group_id": pack["group_id"],
        "variant": pack["variant"],
        "cohort": pack["cohort"],
        "pass_id": pass_record["pass_id"],
        "pass_kind": pass_record["kind"],
        "draw_index": pass_record["draw_index"],
        "provider": vote.provider,
        "model": vote.model,
        "effort": provider.effort,
        "choice": vote.choice,
        "confidence": vote.confidence,
        "reasoning": vote.reasoning,
        "edge_set": [list(edge) for edge in sorted(vote.edge_set)],
        "latency_s": vote.latency_s,
        "timestamp": vote.timestamp,
        "raw_response": vote.raw,
        "error": vote.error,
        "abstain_reason": str(vote.abstain_reason),
        "invocation_route": vote.invocation_route,
        "evidence_delivery": vote.evidence_delivery,
        "pack_feedback_raw": vote.pack_feedback,
        "feedback_status": feedback_status,
        "feedback": normalized_feedback,
        "feedback_errors": feedback_errors,
        "source_prompt_sha256": pack["source_prompt_sha256"],
        "audit_instruction_sha256": plan["runtime"]["audit_instruction_sha256"]
        if pass_record["kind"] == "audit"
        else None,
        "evidence_id": pack["evidence_id"],
        "evidence_pack_sha256": pack["evidence_pack_sha256"],
        "displayed_candidate_universe_sha256": pack["displayed_candidate_universe_sha256"],
        "option_menu_sha256": pack["option_menu_sha256"],
        "invocation_signature_sha256": pass_record["invocation_signature_sha256"],
        "runtime": plan["runtime"],
    }
    result[RESULT_DIGEST_FIELD] = _dict_digest(result, RESULT_DIGEST_FIELD)
    return result


class _RunLock:
    def __init__(self, output_dir: Path):
        self.path = output_dir / ".run.lock"
        self.handle = None

    def __enter__(self):
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            raise RuntimeError(f"another diagnostic process holds {self.path}") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"pid={os.getpid()} acquired={datetime.now(UTC).isoformat()}\n")
        self.handle.flush()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def _selected_calls(
    plan: dict[str, Any], *, smoke: bool, pass_ids: set[str] | None
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    packs = plan["packs"]
    if smoke:
        wanted = set(plan["smoke_pack_keys"])
        packs = [pack for pack in packs if pack["pack_key"] in wanted]
    passes = [
        pass_record
        for pass_record in plan["passes"]
        if pass_ids is None or pass_record["pass_id"] in pass_ids
    ]
    if pass_ids is not None and {item["pass_id"] for item in passes} != pass_ids:
        known = {item["pass_id"] for item in plan["passes"]}
        raise ValueError(f"unknown pass id(s): {sorted(pass_ids - known)}")
    return [(pack, pass_record) for pass_record in passes for pack in packs]


def run_diagnostic(
    plan_path: Path,
    *,
    workers: int = DEFAULT_WORKERS,
    resume: bool = True,
    smoke: bool = False,
    pass_ids: set[str] | None = None,
    invoke: Callable[..., Vote] = run_provider_on_group,
    progress: Callable[[str], None] = print,
) -> dict[str, int]:
    """Execute missing diagnostic calls with bounded concurrency and atomic resume."""
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be in [1, {MAX_WORKERS}], got {workers}")
    plan = validate_diagnostic_plan(plan_path)
    provider = _codex_provider(get_panel("v7-candidate"))
    calls = _selected_calls(plan, smoke=smoke, pass_ids=pass_ids)
    pending: list[tuple[dict[str, Any], dict[str, Any]]] = []
    skipped = 0
    for pack, pass_record in calls:
        path = _result_path(plan, pack, pass_record["pass_id"])
        if path.exists():
            if not resume:
                raise FileExistsError(f"result already exists and --no-resume was used: {path}")
            result = json.loads(path.read_text())
            _validate_existing_result(result, plan=plan, pack=pack, pass_record=pass_record)
            skipped += 1
        else:
            pending.append((pack, pass_record))

    completed = 0
    stop = threading.Event()
    claim_lock = threading.Lock()
    count_lock = threading.Lock()
    errors: list[BaseException] = []
    next_index = 0

    def worker() -> None:
        nonlocal completed, next_index
        while not stop.is_set():
            with claim_lock:
                if stop.is_set() or next_index >= len(pending):
                    return
                pack, pass_record = pending[next_index]
                next_index += 1
            try:
                result = _run_one_call(plan, pack, pass_record, provider, invoke)
                path = _result_path(plan, pack, pass_record["pass_id"])
                if path.exists():
                    raise FileExistsError(f"result appeared concurrently: {path}")
                _atomic_write_json(path, result)
                with count_lock:
                    completed += 1
                    displayed_choice = (
                        result["choice"] if pack["cohort"] == "development" else "SEALED"
                    )
                    progress(
                        f"[{completed}/{len(pending)}] {pack['dataset_id']} "
                        f"{pack['group_id']} {pack['variant']} {pass_record['pass_id']} "
                        f"choice={displayed_choice} feedback={result['feedback_status']}"
                    )
            except BaseException as exc:
                with count_lock:
                    errors.append(exc)
                stop.set()
                return

    with _RunLock(Path(plan["output_dir"])):
        threads = [
            threading.Thread(target=worker, name=f"diagnostic-{i + 1}") for i in range(workers)
        ]
        for thread in threads:
            thread.start()
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            stop.set()
            for thread in threads:
                thread.join()
            raise
    if errors:
        raise errors[0]
    status = {"selected": len(calls), "completed": completed, "skipped": skipped}
    _atomic_write_json(Path(plan["output_dir"]) / "run_state.json", status)
    return status


def diagnostic_status(plan_path: Path) -> dict[str, Any]:
    plan = validate_diagnostic_plan(plan_path)
    by_pass: dict[str, dict[str, int]] = {}
    total_complete = 0
    for pass_record in plan["passes"]:
        complete = 0
        for pack in plan["packs"]:
            path = _result_path(plan, pack, pass_record["pass_id"])
            if path.exists():
                _validate_existing_result(
                    json.loads(path.read_text()),
                    plan=plan,
                    pack=pack,
                    pass_record=pass_record,
                )
                complete += 1
        by_pass[pass_record["pass_id"]] = {
            "complete": complete,
            "planned": len(plan["packs"]),
        }
        total_complete += complete
    return {
        "run_id": plan["run_id"],
        "complete": total_complete,
        "planned": plan["planned_call_count"],
        "by_pass": by_pass,
    }


def _strict_human_labels(
    labels_root: Path, dataset_id: str
) -> dict[str, frozenset[tuple[str, str]]]:
    path = labels_root / f"dataset={dataset_id}" / "data.csv"
    if not path.is_file():
        return {}
    out: dict[str, frozenset[tuple[str, str]]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("label_semantics") or "pair") != "pair":
                continue
            try:
                raw = json.loads(row["selected_edges"])
                if not isinstance(raw, list):
                    continue
                edge_set = frozenset((str(edge["ref_id"]), str(edge["target_id"])) for edge in raw)
            except (KeyError, TypeError, ValueError):
                continue
            out[str(row["group_id"])] = edge_set
    return out


def _edge_key(result: dict[str, Any]) -> str:
    return _canonical_json(sorted(result.get("edge_set", [])))


def _jaccard(left: frozenset[tuple[str, str]], right: frozenset[tuple[str, str]]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _human_menu_expressible(
    human: frozenset[tuple[str, str]],
    option_sets: set[frozenset[tuple[str, str]]],
    none_selectable: bool,
) -> bool:
    """Whether a human label's exact edge set is selectable from the option menu.

    A label is expressible iff it matches some lettered option's edge set, or it
    is the empty (reject-all) set and ``NONE`` is selectable. ``NONE`` expresses
    ONLY the empty set — it must never mark a nonempty label expressible.
    """
    if human in option_sets:
        return True
    return not human and none_selectable


def _analyze_pack(
    plan: dict[str, Any],
    pack: dict[str, Any],
    results: dict[str, dict[str, Any]],
    human: frozenset[tuple[str, str]] | None,
) -> dict[str, Any]:
    canonical = [
        results[pass_record["pass_id"]]
        for pass_record in plan["passes"]
        if pass_record["kind"] == "canonical" and pass_record["pass_id"] in results
    ]
    valid = [result for result in canonical if result["choice"] != "ABSTAIN"]
    counts = Counter(_edge_key(result) for result in valid)
    modal_count = counts.most_common(1)[0][1] if counts else 0
    modal_keys = [key for key, count in counts.items() if count == modal_count]
    modal_key = modal_keys[0] if len(modal_keys) == 1 else None
    if len(valid) != plan["canonical_draws"]:
        stability = "incomplete"
    elif len(counts) == 1:
        stability = "stable"
    elif modal_count > len(valid) / 2:
        stability = "majority"
    else:
        stability = "split"
    modal_result = next((result for result in valid if _edge_key(result) == modal_key), None)
    modal_edges = (
        frozenset((str(edge[0]), str(edge[1])) for edge in modal_result["edge_set"])
        if modal_result is not None
        else None
    )
    audit = results.get("audit")
    evidence_manifest = load_evidence_manifest(Path(pack["group_dir"]), allow_legacy=False)
    evidence = evidence_manifest["evidence"]
    option_sets = {
        frozenset((str(edge["ref_id"]), str(edge["target_id"])) for edge in option["edges"])
        for option in evidence["option_menu"]
    }
    none_selectable = "NONE" in set(evidence.get("selectable_choices", []))
    return {
        "pack_key": pack["pack_key"],
        "dataset_id": pack["dataset_id"],
        "group_id": pack["group_id"],
        "variant": pack["variant"],
        "cohort": pack["cohort"],
        "canonical_complete": len(canonical),
        "canonical_valid": len(valid),
        "stability": stability,
        "distinct_edge_sets": len(counts),
        "modal_draw_count": modal_count,
        "modal_choice": modal_result["choice"] if modal_result else None,
        "modal_edge_set": [list(edge) for edge in sorted(modal_edges or [])],
        "audit_present": audit is not None,
        "audit_choice": audit.get("choice") if audit else None,
        "audit_feedback_status": audit.get("feedback_status") if audit else None,
        "audit_agrees_with_modal": (
            _edge_key(audit) == modal_key if audit is not None and modal_key is not None else None
        ),
        "none_reason": (audit.get("feedback") or {}).get("none_reason") if audit else None,
        "human_label_available": human is not None,
        "human_menu_expressible": (
            _human_menu_expressible(human, option_sets, none_selectable)
            if human is not None
            else None
        ),
        "modal_exact_human": modal_edges == human
        if modal_edges is not None and human is not None
        else None,
        "modal_human_jaccard": _jaccard(modal_edges, human)
        if modal_edges is not None and human is not None
        else None,
        "canonical_votes": [
            {
                "pass_id": result["pass_id"],
                "choice": result["choice"],
                "confidence": result["confidence"],
                "edge_set": result["edge_set"],
                "reasoning": result["reasoning"],
                "abstain_reason": result["abstain_reason"],
                "error": result["error"],
            }
            for result in canonical
        ],
        "audit": (
            {
                "choice": audit["choice"],
                "confidence": audit["confidence"],
                "edge_set": audit["edge_set"],
                "reasoning": audit["reasoning"],
                "feedback_status": audit["feedback_status"],
                "feedback": audit["feedback"],
                "feedback_errors": audit["feedback_errors"],
                "abstain_reason": audit["abstain_reason"],
                "error": audit["error"],
            }
            if audit is not None
            else None
        ),
    }


def _summary_report(summary: dict[str, Any]) -> str:
    counts = summary["stability_counts"]
    draws = summary["canonical_draws"]
    lines = [
        f"# {summary['run_id']} — {summary['cohort']} diagnostic summary",
        "",
        "> Diagnostic findings only. This report is not panel consensus and cannot mint labels.",
        "",
        f"- Packs analyzed: {summary['packs_analyzed']}",
        f"- Results present: {summary['results_present']} / {summary['results_planned']}",
        f"- Stable {draws}/{draws}: {counts.get('stable', 0)}",
        f"- Strict-majority stable: {counts.get('majority', 0)}",
        f"- No strict majority: {counts.get('split', 0)}",
        f"- Incomplete: {counts.get('incomplete', 0)}",
        f"- Valid audit feedback: {summary['valid_audit_feedback']}",
        f"- Human pair labels available: {summary['human_labels_available']}",
        f"- Modal exact-human: {summary['modal_exact_human']}",
        f"- Human labels absent from menu: {summary['human_menu_inexpressible']}",
        "",
        "## By dataset",
        "",
        "| Dataset | Packs | Stable | Majority | Split | Incomplete |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, row in sorted(summary["by_dataset"].items()):
        lines.append(
            f"| {dataset} | {row['packs']} | {row.get('stable', 0)} | "
            f"{row.get('majority', 0)} | {row.get('split', 0)} | "
            f"{row.get('incomplete', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Factorial contrasts",
            "",
            "| Dataset / group | Variants | Modal result |",
            "|---|---:|---|",
        ]
    )
    for group in summary["factorial_groups"]:
        lines.append(
            f"| {group['dataset_id']} / {group['group_id']} | {group['pack_count']} | "
            f"{group['modal_contrast']} |"
        )
    if not summary["factorial_groups"]:
        lines.append("| _none in this cohort_ | 0 | n/a |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Codex repetition measures self-consistency and generates failure-mode hypotheses. "
            "A rubric or evidence change still requires human truth or corroboration from the "
            "fresh Claude/Codex/Muse panel. Higher unanimity alone is not success.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_diagnostic(
    plan_path: Path,
    *,
    labels_root: Path = Path("labels/stitching"),
    include_holdout: bool = False,
    fix_frozen_marker: Path | None = None,
) -> dict[str, Any]:
    """Create a deterministic aggregate, hiding holdout findings by default."""
    plan = validate_diagnostic_plan(plan_path, enforce_runtime=False)
    cohort = "holdout" if include_holdout else "development"
    frozen = None
    if include_holdout:
        if fix_frozen_marker is None or not Path(fix_frozen_marker).is_file():
            raise ValueError("holdout analysis requires an explicit fix-frozen marker")
        frozen = json.loads(Path(fix_frozen_marker).read_text())
        if not isinstance(frozen, dict) or not str(frozen.get("fix_id", "")).strip():
            raise ValueError("fix-frozen marker must be JSON with a nonblank fix_id")

    labels_cache: dict[str, dict[str, frozenset[tuple[str, str]]]] = {}
    per_pack = []
    results_present = 0
    selected_packs = [pack for pack in plan["packs"] if pack["cohort"] == cohort]
    for pack in selected_packs:
        results = {}
        for pass_record in plan["passes"]:
            path = _result_path(plan, pack, pass_record["pass_id"])
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            _validate_existing_result(result, plan=plan, pack=pack, pass_record=pass_record)
            results[pass_record["pass_id"]] = result
            results_present += 1
        dataset = pack["dataset_id"]
        if dataset not in labels_cache:
            labels_cache[dataset] = _strict_human_labels(Path(labels_root), dataset)
        human = labels_cache[dataset].get(pack["group_id"])
        per_pack.append(_analyze_pack(plan, pack, results, human))

    stability_counts = Counter(row["stability"] for row in per_pack)
    by_dataset: dict[str, Counter] = defaultdict(Counter)
    by_variant: dict[str, Counter] = defaultdict(Counter)
    for row in per_pack:
        by_dataset[row["dataset_id"]]["packs"] += 1
        by_dataset[row["dataset_id"]][row["stability"]] += 1
        by_variant[row["variant"]]["packs"] += 1
        by_variant[row["variant"]][row["stability"]] += 1
    none_reasons = Counter(row["none_reason"] for row in per_pack if row["none_reason"])
    rows_by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_pack:
        rows_by_group[(row["dataset_id"], row["group_id"])].append(row)
    factorial_groups = []
    for (dataset_id, group_id), rows in sorted(rows_by_group.items()):
        if len(rows) < 2:
            continue
        complete_modal_keys = [
            _canonical_json(row["modal_edge_set"])
            for row in rows
            if row["canonical_valid"] == plan["canonical_draws"]
        ]
        modal_keys = set(complete_modal_keys)
        if len(complete_modal_keys) != len(rows):
            contrast = "incomplete"
        elif len(modal_keys) == 1:
            contrast = "consistent"
        else:
            contrast = "divergent"
        factorial_groups.append(
            {
                "dataset_id": dataset_id,
                "group_id": group_id,
                "pack_count": len(rows),
                "modal_contrast": contrast,
                "distinct_complete_modal_edge_sets": len(modal_keys),
                "variants": [
                    {
                        "pack_key": row["pack_key"],
                        "variant": row["variant"],
                        "stability": row["stability"],
                        "modal_choice": row["modal_choice"],
                        "modal_edge_set": row["modal_edge_set"],
                        "audit_choice": row["audit_choice"],
                        "audit_agrees_with_modal": row["audit_agrees_with_modal"],
                    }
                    for row in sorted(rows, key=lambda item: item["variant"])
                ],
            }
        )
    summary: dict[str, Any] = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "run_id": plan["run_id"],
        "diagnostic_manifest_sha256": plan[PLAN_DIGEST_FIELD],
        "purpose": "diagnostic_only_not_panel_consensus",
        "cohort": cohort,
        "canonical_draws": plan["canonical_draws"],
        "fix_frozen": frozen,
        "packs_analyzed": len(per_pack),
        "results_present": results_present,
        "results_planned": len(selected_packs) * len(plan["passes"]),
        "stability_counts": dict(stability_counts),
        "valid_audit_feedback": sum(row["audit_feedback_status"] == "valid" for row in per_pack),
        "audit_modal_agreement": sum(row["audit_agrees_with_modal"] is True for row in per_pack),
        "none_reason_counts": dict(none_reasons),
        "human_labels_available": sum(row["human_label_available"] for row in per_pack),
        "modal_exact_human": sum(row["modal_exact_human"] is True for row in per_pack),
        "human_menu_inexpressible": sum(row["human_menu_expressible"] is False for row in per_pack),
        "by_dataset": {key: dict(value) for key, value in by_dataset.items()},
        "by_variant": {key: dict(value) for key, value in by_variant.items()},
        "factorial_groups": factorial_groups,
        "per_pack": per_pack,
    }
    output_dir = Path(plan["output_dir"])
    if include_holdout:
        summary_path = output_dir / "holdout" / "summary.json"
        report_path = output_dir / "holdout" / "report.md"
    else:
        summary_path = output_dir / "development_summary.json"
        report_path = output_dir / "development_report.md"
    _atomic_write_json(summary_path, summary)
    report_path.write_text(_summary_report(summary))
    return summary
