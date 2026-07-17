#!/usr/bin/env python3
"""Build the targeted v7 physical/coincidence stitching vote wave.

The wave is deliberately calibration-heavy: it combines known physical-feature
failure pairs, geometry-derived same-side coincidence, fresh physical conflicts
and agreements, plausible NONE cases, ambiguous M:N assignments, and ordinary
controls. Fifty unique groups are packed with full evidence. A small factorial
subset is repeated with physical and coincidence context independently toggled,
so the intervention can distinguish either signal from their interaction.
"""

from __future__ import annotations

import argparse
import copy
import gc
import heapq
import json
import math
import re
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from json import JSONDecodeError, JSONDecoder
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from shapely.geometry import shape as shape_from_geojson

from crosswalk.agent_labeling.stitch_evidence import generate_stitch_evidence
from crosswalk.agent_labeling.stitch_export import NO_EXPORT_MARKER
from crosswalk.agent_labeling.stitch_provenance import artifact_descriptor
from crosswalk.agent_labeling.stitch_runner import get_panel, panel_descriptor
from crosswalk.agent_labeling.wave_manifest import (
    FIELD_BATCH_DIRS,
    FIELD_PANEL,
    FIELD_REQUIRED_PANEL,
    FIELD_RUN_SCHEDULE,
    FIELD_TOTAL_PACK_COUNT,
    ROW_BATCH_DIR,
    ROW_DATASET_ID,
    ROW_GROUP_ID,
    ROW_RUN_INDEX,
    ROW_VARIANT,
    WaveManifest,
)
from crosswalk.cli.data import _fill_spatial_context
from crosswalk.config import settings
from crosswalk.datasets.schema import get_dataset_config
from crosswalk.features.coincidence import compute_same_side_coincidence_context
from crosswalk.features.physical import compute_physical_pair_features
from crosswalk.labeling.stitching_store import StitchingLabelStore
from crosswalk.matching.alternatives import generate_top_k_alternatives
from crosswalk.provenance import source_commit_provenance

ROOT = Path(__file__).parents[1]
DEFAULT_QUOTAS = {
    "au_sydney_roads": 7,
    "fi_helsinki_roads": 7,
    "gb_london_roads": 5,
    "hk_hongkong_roads": 8,
    "de_berlin_roads": 8,
    "nl_amsterdam_roads": 5,
    "ch_grand_geneva_cycle_schema": 5,
    "us_philadelphia_sidewalks": 5,
}
CATEGORY_ORDER = (
    "manual_pair",
    "coincidence",
    "physical_conflict",
    "physical_agreement",
    "frontage_layered",
    "known_none",
    "ambiguous",
    "control",
)
ROLE_PATTERN = re.compile(
    r"frontage|ramp|flyover|bridge|tunnel|covered|tranch|underpass|viaduct",
    re.IGNORECASE,
)
VARIANTS = {
    "enriched": {"physical": True, "coincidence": True},
    "no_physical": {"physical": False, "coincidence": True},
    "no_coincidence": {"physical": True, "coincidence": False},
    "minimal": {"physical": False, "coincidence": False},
}
# The voting panel this wave is built for. The manifest's required_panel block
# is derived from get_panel(WAVE_PANEL) via panel_descriptor so a roster change
# in stitch_runner cannot silently drift from the manifest — the two are the
# same source at build time, not merely cross-checked at run time.
WAVE_PANEL = "v7-candidate"


@dataclass(order=True)
class RankedGroup:
    score: float
    group_id: str
    group: dict = field(compare=False)
    tags: tuple[str, ...] = field(compare=False)
    audit: dict[str, Any] = field(compare=False)


@dataclass(frozen=True)
class TargetPhysicalCapabilities:
    """Target domains that are actually surveyed/configured for one dataset."""

    has_level: bool
    flag_domains: frozenset[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "has_level": self.has_level,
            "flag_domains": sorted(self.flag_domains),
        }


def iter_sidecar_groups(path: Path, *, chunk_size: int = 1024 * 1024) -> Iterator[dict]:
    """Stream the top-level ``groups`` JSON array without loading a huge sidecar."""
    decoder = JSONDecoder()
    buffer = ""
    started = False
    with path.open(encoding="utf-8") as handle:
        eof = False
        while not eof:
            chunk = handle.read(chunk_size)
            eof = not chunk
            buffer += chunk
            if not started:
                marker = buffer.find('"groups"')
                if marker < 0:
                    if eof:
                        raise ValueError(f"No groups array in {path}")
                    buffer = buffer[-32:]
                    continue
                start = buffer.find("[", marker)
                if start < 0:
                    if eof:
                        raise ValueError(f"Malformed groups array in {path}")
                    continue
                buffer = buffer[start + 1 :]
                started = True

            while started:
                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:].lstrip()
                if buffer.startswith("]"):
                    return
                if not buffer:
                    break
                try:
                    group, end = decoder.raw_decode(buffer)
                except JSONDecodeError:
                    break
                if not isinstance(group, dict):
                    raise ValueError(f"Non-object group in {path}")
                yield group
                buffer = buffer[end:]
        if started and buffer.lstrip().startswith("]"):
            return
    raise ValueError(f"Unterminated groups array in {path}")


def _target_capabilities(dataset_id: str) -> TargetPhysicalCapabilities:
    config = get_dataset_config(dataset_id)
    fetch = config.fetch if config is not None else None
    if fetch is None:
        return TargetPhysicalCapabilities(has_level=False, flag_domains=frozenset())
    return TargetPhysicalCapabilities(
        has_level=bool(fetch.level_column),
        flag_domains=fetch.physical_flag_domains(),
    )


def _sanitize_target_physical_block(
    physical: dict[str, Any] | None,
    capabilities: TargetPhysicalCapabilities,
) -> dict[str, Any]:
    """Remove target evidence for domains the provider does not expose."""
    result = copy.deepcopy(physical or {})
    if not capabilities.has_level:
        result.pop("level_lr", None)
    if not capabilities.flag_domains:
        result.pop("road_flags_lr", None)
    elif "road_flags_lr" in result:
        rules = []
        for rule in result.get("road_flags_lr") or []:
            sanitized = copy.deepcopy(rule)
            sanitized["value"] = [
                flag for flag in sanitized.get("value", []) if flag in capabilities.flag_domains
            ]
            rules.append(sanitized)
        result["road_flags_lr"] = rules
    return result


def _sanitize_group_target_physical(
    group: dict[str, Any],
    capabilities: TargetPhysicalCapabilities,
) -> dict[str, Any]:
    """Sanitize group-wide and edge-level target physical blocks in place."""
    target_physical = {}
    for target_id, physical in (group.get("target_physical") or {}).items():
        sanitized = _sanitize_target_physical_block(physical, capabilities)
        if sanitized:
            target_physical[str(target_id)] = sanitized
    group["target_physical"] = target_physical

    for source in ("edges", "candidate_edges", "rejected_edges"):
        for edge in group.get(source, []) or []:
            sanitized = _sanitize_target_physical_block(edge.get("target_physical"), capabilities)
            if sanitized:
                edge["target_physical"] = sanitized
            else:
                edge.pop("target_physical", None)
    return group


def _manual_pairs(path: Path) -> dict[str, set[tuple[str, str]]]:
    payload = json.loads(path.read_text())
    result: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in payload.get("manual_review_queue", []):
        result[str(row["dataset"])].add((str(row["gers_id"]), str(row["target_id"])))
    return result


def _known_none_group_ids(dataset_id: str) -> set[str]:
    frame = StitchingLabelStore(dataset_id).load(dataset_id)
    if frame.empty:
        return set()
    result = set()
    for _, row in frame.iterrows():
        if str(row.get("label_semantics", "pair")) == "set":
            continue
        try:
            selected = json.loads(row.get("selected_edges") or "[]")
        except (TypeError, JSONDecodeError):
            continue
        if not selected:
            result.add(str(row["group_id"]))
    return result


def _offered_edge_keys(group: dict) -> set[tuple[str, str]]:
    """Pairs that can actually appear in a generated assignment option."""
    return {
        (str(edge["ref_id"]), str(edge["target_id"]))
        for edge in group.get("edges", []) or []
        if edge.get("ref_id") is not None and edge.get("target_id") is not None
    }


def _group_coincidence(group: dict) -> tuple[int, int, list[str]]:
    rows = 0
    conflicts = 0
    labels: list[str] = []
    for side, ids_key, geometry_key, classes_key, prefix in (
        ("reference", "ref_ids", "ref_geometries", "ref_classes", "R"),
        ("target", "target_ids", "target_geometries", "target_classes", "T"),
    ):
        ids = [str(value) for value in group.get(ids_key, []) or []]
        if len(ids) < 2:
            continue
        geometries = {}
        for segment_id, value in (group.get(geometry_key, {}) or {}).items():
            try:
                geometries[str(segment_id)] = shape_from_geojson(value)
            except Exception:
                continue
        label_map = {segment_id: f"{prefix}{index + 1}" for index, segment_id in enumerate(ids)}
        result = compute_same_side_coincidence_context(
            geometries,
            roles={str(k): v for k, v in (group.get(classes_key, {}) or {}).items()},
            labels=label_map,
        )
        rows += len(result)
        conflicts += sum(item.has_role_conflict for item in result.values())
        for segment_id, item in result.items():
            labels.append(
                f"{side}:{label_map.get(segment_id, segment_id)}={','.join(item.alternative_ids)}"
            )
    return rows, conflicts, labels


def _group_physical(
    group: dict,
    capabilities: TargetPhysicalCapabilities,
) -> tuple[float, float, int]:
    conflict = 0.0
    agreement = 0.0
    comparable = 0
    for edge in group.get("edges", []) or []:
        ref = edge.get("ref_physical") or {}
        target = edge.get("target_physical") or {}
        features = compute_physical_pair_features(
            ref_level_lr=ref.get("level_lr"),
            target_level_lr=target.get("level_lr"),
            ref_road_flags_lr=ref.get("road_flags_lr"),
            target_road_flags_lr=target.get("road_flags_lr"),
            target_flag_domains=set(capabilities.flag_domains),
        )
        value = features["physical_structure_conflict"]
        if not math.isnan(value):
            conflict = max(conflict, float(value))
        value = features["physical_positive_match"]
        if not math.isnan(value):
            agreement = max(agreement, float(value))
        comparable += int(features["physical_comparable_count"] > 0)
    return conflict, agreement, comparable


def _uncertainty(group: dict) -> float:
    confidences = [float(edge.get("confidence", 0.0)) for edge in group.get("edges", []) or []]
    if not confidences:
        return 0.0
    return max(1.0 - 2.0 * abs(value - 0.5) for value in confidences)


def _role_text(group: dict) -> str:
    values = []
    for key in ("ref_names", "target_names", "ref_classes", "target_classes"):
        values.extend(str(value) for value in (group.get(key, {}) or {}).values())
    return " ".join(values)


def _rank_group(
    group: dict,
    *,
    manual_pairs: set[tuple[str, str]],
    known_none: set[str],
    capabilities: TargetPhysicalCapabilities,
    required_pairs: set[tuple[str, str]],
    forced_ref_ids: set[str],
) -> RankedGroup | None:
    group_id = str(group.get("group_id", ""))
    edge_count = int(group.get("n_candidate_edges", len(group.get("edges", []) or [])))
    if not group_id or edge_count < 1 or edge_count > settings.stitch_export_backstop_max_edges:
        return None
    edge_keys = _offered_edge_keys(group)
    manual_hits = sorted(edge_keys & manual_pairs)
    required_pair_hits = sorted(edge_keys & required_pairs)
    forced = bool(set(map(str, group.get("ref_ids", []) or [])) & forced_ref_ids)
    if (
        group.get("match_type") == "1:1"
        and len(group.get("edges", []) or []) <= 1
        and not required_pair_hits
        and not forced
    ):
        return None

    coincidence_rows, coincidence_conflicts, coincidence_labels = _group_coincidence(group)
    conflict, agreement, comparable = _group_physical(group, capabilities)
    uncertainty = _uncertainty(group)

    tags: list[str] = []
    if manual_hits or required_pair_hits or forced:
        tags.append("manual_pair")
    if coincidence_rows:
        tags.append("coincidence")
    if conflict >= 0.25:
        tags.append("physical_conflict")
    if agreement >= 0.5:
        tags.append("physical_agreement")
    role_context = bool(
        ROLE_PATTERN.search(_role_text(group))
        and (coincidence_rows or conflict > 0 or agreement > 0)
    )
    if role_context:
        tags.append("frontage_layered")
    if group_id in known_none:
        tags.append("known_none")
    if uncertainty >= 0.55 or edge_count >= 5:
        tags.append("ambiguous")
    if not tags and uncertainty <= 0.25:
        tags.append("control")
    if not tags:
        return None

    score = (
        140.0 * bool(required_pair_hits or forced)
        + 120.0 * bool(manual_hits)
        + 85.0 * bool(coincidence_rows)
        + 25.0 * coincidence_conflicts
        + 65.0 * conflict
        + 50.0 * agreement
        + 18.0 * bool(group_id in known_none)
        + 15.0 * role_context
        + 20.0 * uncertainty
        + min(edge_count, 12)
    )
    audit = {
        "tags": tags,
        "manual_pair_hits": [list(pair) for pair in manual_hits],
        "required_pair_hits": [list(pair) for pair in required_pair_hits],
        "forced_regression_member": forced,
        "coincidence_rows": coincidence_rows,
        "coincidence_role_conflicts": coincidence_conflicts,
        "coincidence_labels": coincidence_labels,
        "physical_conflict": round(conflict, 4),
        "physical_agreement": round(agreement, 4),
        "physical_comparable_edges": comparable,
        "uncertainty": round(uncertainty, 4),
        "candidate_edge_count": edge_count,
        "match_type": group.get("match_type"),
        "score": round(score, 4),
        "target_physical_capabilities": capabilities.as_dict(),
    }
    return RankedGroup(score, group_id, group, tuple(tags), audit)


def _push_pool(
    pool: list[tuple[float, str, RankedGroup]],
    ranked: RankedGroup,
    *,
    limit: int = 80,
) -> None:
    item = (ranked.score, ranked.group_id, ranked)
    if len(pool) < limit:
        heapq.heappush(pool, item)
    elif item[:2] > pool[0][:2]:
        heapq.heapreplace(pool, item)


def select_dataset_groups(
    dataset_id: str,
    sidecar: Path,
    *,
    quota: int,
    manual_pairs: set[tuple[str, str]],
    required_pairs: set[tuple[str, str]],
    forced_ref_ids: set[str],
) -> list[RankedGroup]:
    pools: dict[str, list[tuple[float, str, RankedGroup]]] = defaultdict(list)
    capabilities = _target_capabilities(dataset_id)
    known_none = _known_none_group_ids(dataset_id)
    required_pair_groups: dict[tuple[str, str], RankedGroup] = {}
    required_ref_groups: dict[str, RankedGroup] = {}
    scanned = 0
    started_at = time.perf_counter()
    for group in iter_sidecar_groups(sidecar):
        scanned += 1
        if scanned % 5000 == 0:
            elapsed = time.perf_counter() - started_at
            print(
                f"{dataset_id}: scanned {scanned:,} groups in {elapsed:.1f}s",
                flush=True,
            )
        _sanitize_group_target_physical(group, capabilities)
        ranked = _rank_group(
            group,
            manual_pairs=manual_pairs,
            known_none=known_none,
            capabilities=capabilities,
            required_pairs=required_pairs,
            forced_ref_ids=forced_ref_ids,
        )
        if ranked is None:
            continue
        offered = _offered_edge_keys(group)
        for pair in required_pairs & offered:
            previous = required_pair_groups.get(pair)
            if previous is None or ranked.score > previous.score:
                required_pair_groups[pair] = ranked
        ref_ids = set(map(str, group.get("ref_ids", []) or []))
        for ref_id in forced_ref_ids & ref_ids:
            previous = required_ref_groups.get(ref_id)
            if previous is None or ranked.score > previous.score:
                required_ref_groups[ref_id] = ranked
        for tag in ranked.tags:
            _push_pool(pools[tag], ranked)

    missing_pairs = sorted(required_pairs - set(required_pair_groups))
    if missing_pairs:
        raise RuntimeError(
            f"{dataset_id}: required regression pairs are not offered by any group: {missing_pairs}"
        )
    missing_refs = sorted(forced_ref_ids - set(required_ref_groups))
    if missing_refs:
        raise RuntimeError(f"{dataset_id}: required regression refs are absent: {missing_refs}")

    ordered = {tag: [item[2] for item in sorted(pool, reverse=True)] for tag, pool in pools.items()}
    required = {
        ranked.group_id: ranked
        for ranked in [*required_pair_groups.values(), *required_ref_groups.values()]
    }
    selected = sorted(required.values(), reverse=True)
    seen = set(required)
    if len(selected) > quota:
        raise RuntimeError(
            f"{dataset_id}: {len(selected)} forced regression groups exceed quota {quota}"
        )
    while len(selected) < quota:
        progressed = False
        for tag in CATEGORY_ORDER:
            for ranked in ordered.get(tag, []):
                if ranked.group_id in seen:
                    continue
                selected.append(ranked)
                seen.add(ranked.group_id)
                progressed = True
                break
            if len(selected) >= quota:
                break
        if not progressed:
            break
    if len(selected) < quota:
        remainder = sorted(
            {
                ranked.group_id: ranked
                for pool in ordered.values()
                for ranked in pool
                if ranked.group_id not in seen
            }.values(),
            reverse=True,
        )
        selected.extend(remainder[: quota - len(selected)])
    if len(selected) != quota:
        raise RuntimeError(f"{dataset_id}: selected {len(selected)} of requested {quota}")

    selected_pairs = set().union(*(_offered_edge_keys(item.group) for item in selected))
    if not required_pairs <= selected_pairs:
        raise AssertionError(f"{dataset_id}: final roster lost a required offered pair")
    selected_refs = set().union(
        *(set(map(str, item.group.get("ref_ids", []) or [])) for item in selected)
    )
    if not forced_ref_ids <= selected_refs:
        raise AssertionError(f"{dataset_id}: final roster lost a required regression ref")
    print(f"{dataset_id}: scanned {scanned:,}, selected {len(selected)}")
    return selected


def _strip_physical(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_physical(child)
            for key, child in value.items()
            if key not in {"ref_physical", "target_physical"}
        }
    if isinstance(value, list):
        return [_strip_physical(child) for child in value]
    return value


def _write_batch(
    dataset_id: str,
    ranked_groups: list[RankedGroup],
    *,
    sidecar_root: Path,
    output_root: Path,
    wave_name: str,
    variant: str,
    k_alternatives: int,
) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown experiment variant: {variant}")
    treatment = VARIANTS[variant]
    capabilities = _target_capabilities(dataset_id)
    selected = []
    for ranked in ranked_groups:
        group = copy.deepcopy(ranked.group)
        _sanitize_group_target_physical(group, capabilities)
        group["experimental_wave_selection"] = ranked.audit
        if not treatment["physical"]:
            group = _strip_physical(group)
        group["alternatives"] = generate_top_k_alternatives(
            group.get("edges", []),
            ref_geoms=group.get("ref_geometries", {}),
            target_geoms=group.get("target_geometries", {}),
            k=k_alternatives,
        )
        selected.append(group)

    _fill_spatial_context(selected, dataset_id)
    suffix = wave_name if variant == "enriched" else f"{wave_name}_{variant}"
    batch_dir = output_root / f"{dataset_id}_{suffix}"
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty batch {batch_dir}; use a new wave name"
        )
    batch_dir.mkdir(parents=True, exist_ok=True)
    bridge = sidecar_root / f"{dataset_id}_bridge.parquet"
    candidates = sidecar_root / f"{dataset_id}_candidates.parquet"
    groups_path = sidecar_root / f"{dataset_id}_groups.json"
    batch = {
        "schema_version": 2,
        "dataset_id": dataset_id,
        "source_artifacts": {
            "groups_sidecar": artifact_descriptor(groups_path, root=ROOT),
            "candidates_parquet": artifact_descriptor(candidates, root=ROOT),
            "bridge_parquet": artifact_descriptor(bridge, root=ROOT),
        },
        "batch_generation_source": {"source_commit": source_commit_provenance(ROOT)},
        "experiment": {
            "wave": wave_name,
            "variant": variant,
            "physical_metadata_visible": treatment["physical"],
            "same_side_coincidence_visible": treatment["coincidence"],
            "target_physical_capabilities": capabilities.as_dict(),
        },
        "groups": selected,
    }
    (batch_dir / "batch.json").write_text(json.dumps(batch))
    # Belt-and-suspenders: drop a .no-export marker on every ablation-variant
    # batch so filter_exportable_batch_dirs drops it from the export path even if
    # its batch.json ever becomes unreadable. Ablation ballots are experiment
    # data and must never mint; only the enriched (full-context) variant may.
    if variant != "enriched":
        (batch_dir / NO_EXPORT_MARKER).write_text(
            f"ablation variant {variant!r}: experiment data, must not mint panel labels\n"
        )
    generated = generate_stitch_evidence(batch, batch_dir)
    if len(generated) != len(selected):
        raise RuntimeError(
            f"{dataset_id}/{variant}: generated {len(generated)} of {len(selected)} packs"
        )
    return batch_dir


def _factorial_subset(
    selections: dict[str, list[RankedGroup]], factorial_count: int
) -> dict[str, list[RankedGroup]]:
    candidates = [
        (ranked.audit["physical_conflict"] + ranked.audit["physical_agreement"], dataset_id, ranked)
        for dataset_id, groups in selections.items()
        for ranked in groups
        if ranked.audit["physical_comparable_edges"] > 0
        and ranked.audit["coincidence_rows"] > 0
        and (ranked.audit["physical_conflict"] > 0 or ranked.audit["physical_agreement"] > 0)
    ]
    candidates.sort(key=lambda item: (item[0], item[2].score), reverse=True)
    result: dict[str, list[RankedGroup]] = defaultdict(list)
    per_dataset: dict[str, int] = defaultdict(int)
    for _physical_score, dataset_id, ranked in candidates:
        if per_dataset[dataset_id] >= 1:
            continue
        result[dataset_id].append(ranked)
        per_dataset[dataset_id] += 1
        if sum(map(len, result.values())) >= factorial_count:
            break
    if sum(map(len, result.values())) != factorial_count:
        raise RuntimeError(
            f"Only found {sum(map(len, result.values()))} diverse factorial controls"
        )
    return dict(result)


def _planned_batch_dirs(
    *,
    selections: dict[str, list[RankedGroup]],
    factorial: dict[str, list[RankedGroup]],
    output_root: Path,
    wave_name: str,
) -> dict[tuple[str, str], Path]:
    planned = {
        (dataset_id, "enriched"): output_root / f"{dataset_id}_{wave_name}"
        for dataset_id in selections
    }
    for dataset_id in factorial:
        for variant in ("no_physical", "no_coincidence", "minimal"):
            planned[(dataset_id, variant)] = output_root / f"{dataset_id}_{wave_name}_{variant}"
    return planned


def _assert_output_paths_available(
    batch_dirs: dict[tuple[str, str], Path], manifest_path: Path
) -> None:
    collisions = sorted(str(path) for path in batch_dirs.values() if path.exists())
    if manifest_path.exists():
        collisions.append(str(manifest_path))
    if collisions:
        raise FileExistsError(
            "Refusing to create a partial/overwritten wave; output paths already exist: "
            + ", ".join(collisions)
        )


def _assert_required_pairs_in_generated_menus(
    selections: dict[str, list[RankedGroup]],
    required_pairs: dict[str, set[tuple[str, str]]],
    batch_dirs: dict[tuple[str, str], Path],
) -> None:
    """Prove fixture pairs survive option generation and diversity pruning."""
    for dataset_id, pairs in required_pairs.items():
        menu_pairs: set[tuple[str, str]] = set()
        for ranked in selections[dataset_id]:
            evidence_path = batch_dirs[(dataset_id, "enriched")] / ranked.group_id / "evidence.json"
            evidence = json.loads(evidence_path.read_text())["evidence"]
            for option in evidence["option_menu"]:
                menu_pairs.update(
                    (str(edge["ref_id"]), str(edge["target_id"])) for edge in option["edges"]
                )
        missing = sorted(pairs - menu_pairs)
        if missing:
            raise RuntimeError(
                f"{dataset_id}: required regression pairs were pruned from every "
                f"generated option menu: {missing}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sidecar-root", type=Path, default=Path("data/experiments/stitch_physical_v7")
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/agents/stitching/batches"))
    parser.add_argument("--wave-name", default="physical_context_v7_20260715")
    parser.add_argument(
        "--manual-queue",
        type=Path,
        default=Path("research/results/physical_feature_ablation_2026-07-15.json"),
    )
    parser.add_argument(
        "--regressions", type=Path, default=Path("tests/fixtures/physical_match_regressions.json")
    )
    parser.add_argument("--factorial-count", type=int, default=5)
    parser.add_argument("--alternatives", type=int, default=8)
    args = parser.parse_args()

    manual = _manual_pairs(args.manual_queue)
    regression_payload = json.loads(args.regressions.read_text())
    required_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for case in regression_payload.get("pair_cases", []):
        required_pairs[str(case["dataset_id"])].add((str(case["ref_id"]), str(case["target_id"])))
    forced_refs: dict[str, set[str]] = defaultdict(set)
    for case in regression_payload.get("group_cases", []):
        if case.get("ambiguous_ref_id"):
            forced_refs[str(case["dataset_id"])].add(str(case["ambiguous_ref_id"]))

    selections: dict[str, list[RankedGroup]] = {}
    for dataset_id, quota in DEFAULT_QUOTAS.items():
        sidecar = args.sidecar_root / f"{dataset_id}_groups.json"
        if not sidecar.exists():
            raise FileNotFoundError(sidecar)
        selections[dataset_id] = select_dataset_groups(
            dataset_id,
            sidecar,
            quota=quota,
            manual_pairs=manual.get(dataset_id, set()),
            required_pairs=required_pairs.get(dataset_id, set()),
            forced_ref_ids=forced_refs.get(dataset_id, set()),
        )
        gc.collect()

    unknown_regression_datasets = sorted(
        (set(required_pairs) | set(forced_refs)) - set(DEFAULT_QUOTAS)
    )
    if unknown_regression_datasets:
        raise RuntimeError(
            "Regression fixtures reference datasets outside the wave: "
            f"{unknown_regression_datasets}"
        )

    factorial = _factorial_subset(selections, args.factorial_count)
    batch_dirs = _planned_batch_dirs(
        selections=selections,
        factorial=factorial,
        output_root=args.output_root,
        wave_name=args.wave_name,
    )
    manifest_path = args.output_root / f"{args.wave_name}_manifest.json"
    _assert_output_paths_available(batch_dirs, manifest_path)
    for dataset_id, groups in selections.items():
        written = _write_batch(
            dataset_id,
            groups,
            sidecar_root=args.sidecar_root,
            output_root=args.output_root,
            wave_name=args.wave_name,
            variant="enriched",
            k_alternatives=args.alternatives,
        )
        assert written == batch_dirs[(dataset_id, "enriched")]
        gc.collect()
    for dataset_id, groups in factorial.items():
        for variant in ("no_physical", "no_coincidence", "minimal"):
            written = _write_batch(
                dataset_id,
                groups,
                sidecar_root=args.sidecar_root,
                output_root=args.output_root,
                wave_name=args.wave_name,
                variant=variant,
                k_alternatives=args.alternatives,
            )
            assert written == batch_dirs[(dataset_id, variant)]
            gc.collect()

    _assert_required_pairs_in_generated_menus(selections, required_pairs, batch_dirs)

    factorial_rows = sorted(
        (dataset_id, ranked) for dataset_id, groups in factorial.items() for ranked in groups
    )
    factorial_ids = {(dataset_id, ranked.group_id) for dataset_id, ranked in factorial_rows}
    ordinary = [
        {
            ROW_DATASET_ID: dataset_id,
            ROW_GROUP_ID: ranked.group_id,
            ROW_VARIANT: "enriched",
            ROW_BATCH_DIR: str(batch_dirs[(dataset_id, "enriched")]),
        }
        for dataset_id, groups in selections.items()
        for ranked in groups
        if (dataset_id, ranked.group_id) not in factorial_ids
    ]
    schedule = []
    ordinary_index = 0
    condition_order = list(VARIANTS)
    for round_index in range(len(condition_order)):
        remaining_rounds = len(condition_order) - round_index
        chunk_size = math.ceil((len(ordinary) - ordinary_index) / remaining_rounds)
        schedule.extend(ordinary[ordinary_index : ordinary_index + chunk_size])
        ordinary_index += chunk_size
        for slot, (dataset_id, ranked) in enumerate(factorial_rows):
            variant = condition_order[(round_index + slot) % len(condition_order)]
            schedule.append(
                {
                    ROW_DATASET_ID: dataset_id,
                    ROW_GROUP_ID: ranked.group_id,
                    ROW_VARIANT: variant,
                    ROW_BATCH_DIR: str(batch_dirs[(dataset_id, variant)]),
                    "counterbalance_slot": slot,
                    "counterbalance_round": round_index,
                }
            )
    for index, row in enumerate(schedule, start=1):
        row[ROW_RUN_INDEX] = index

    # Factorial conditions must expose the exact same assignment menu. Only the
    # physical/coincidence evidence and associated guidance may differ.
    for dataset_id, ranked in factorial_rows:
        menu_hashes = {}
        for variant in VARIANTS:
            evidence_path = batch_dirs[(dataset_id, variant)] / ranked.group_id / "evidence.json"
            evidence = json.loads(evidence_path.read_text())["evidence"]
            menu_hashes[variant] = evidence["option_menu_sha256"]
        if len(set(menu_hashes.values())) != 1:
            raise AssertionError(
                f"{dataset_id}/{ranked.group_id}: factorial option menus differ: {menu_hashes}"
            )

    manifest_content = {
        "wave": args.wave_name,
        FIELD_PANEL: WAVE_PANEL,
        FIELD_REQUIRED_PANEL: panel_descriptor(get_panel(WAVE_PANEL)),
        "unique_group_count": sum(map(len, selections.values())),
        "factorial_group_count": sum(map(len, factorial.values())),
        "paired_control_count": 3 * sum(map(len, factorial.values())),
        FIELD_TOTAL_PACK_COUNT: len(schedule),
        FIELD_BATCH_DIRS: [str(path) for path in batch_dirs.values()],
        FIELD_RUN_SCHEDULE: schedule,
        "selections": {
            dataset_id: [{"group_id": ranked.group_id, **ranked.audit} for ranked in groups]
            for dataset_id, groups in selections.items()
        },
        "factorial_controls": {
            dataset_id: [ranked.group_id for ranked in groups]
            for dataset_id, groups in factorial.items()
        },
    }
    WaveManifest(manifest_content).write(manifest_path)
    print(
        f"Wrote {manifest_content['unique_group_count']} enriched packs + "
        f"{manifest_content['paired_control_count']} factorial variants -> {manifest_path}"
    )


if __name__ == "__main__":
    main()
