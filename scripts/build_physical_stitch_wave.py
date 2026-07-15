#!/usr/bin/env python3
"""Build the targeted v7 physical/coincidence stitching vote wave.

The wave is deliberately calibration-heavy: it combines known physical-feature
failure pairs, geometry-derived same-side coincidence, fresh physical conflicts
and agreements, plausible NONE cases, ambiguous M:N assignments, and ordinary
controls. Fifty unique groups are packed with full evidence. A smaller paired
control set reuses selected groups after removing only physical metadata; same-
side coincidence remains visible so the control isolates bridge/tunnel/vertical
evidence rather than changing the geometry rubric.
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
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from json import JSONDecodeError, JSONDecoder
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from shapely.geometry import shape as shape_from_geojson

from crosswalk.agent_labeling.stitch_evidence import generate_stitch_evidence
from crosswalk.agent_labeling.stitch_provenance import artifact_descriptor
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
    "au_sydney_roads": 8,
    "fi_helsinki_roads": 8,
    "gb_london_roads": 6,
    "hk_hongkong_roads": 9,
    "de_berlin_roads": 9,
    "nl_amsterdam_roads": 5,
    "ch_grand_geneva_cycle_schema": 5,
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
    r"frontage|service|ramp|flyover|bridge|tunnel|covered|tranch|underpass|viaduct|motorway",
    re.IGNORECASE,
)


@dataclass(order=True)
class RankedGroup:
    score: float
    group_id: str
    group: dict = field(compare=False)
    tags: tuple[str, ...] = field(compare=False)
    audit: dict[str, Any] = field(compare=False)


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


def _target_domains(dataset_id: str) -> set[str]:
    config = get_dataset_config(dataset_id)
    fetch = config.fetch if config is not None else None
    domains: set[str] = set()
    if fetch is not None and fetch.bridge_column:
        domains.add("is_bridge")
    if fetch is not None and fetch.tunnel_column:
        domains.add("is_tunnel")
    return domains


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


def _group_edge_keys(group: dict) -> set[tuple[str, str]]:
    keys = set()
    for source in ("edges", "candidate_edges", "rejected_edges"):
        for edge in group.get(source, []) or []:
            if edge.get("ref_id") is not None and edge.get("target_id") is not None:
                keys.add((str(edge["ref_id"]), str(edge["target_id"])))
    return keys


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
                f"{side}:{label_map.get(segment_id, segment_id)}="
                f"{','.join(item.alternative_ids)}"
            )
    return rows, conflicts, labels


def _group_physical(group: dict, target_domains: set[str]) -> tuple[float, float, int]:
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
            target_flag_domains=target_domains,
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
    target_domains: set[str],
    forced_ref_ids: set[str],
) -> RankedGroup | None:
    group_id = str(group.get("group_id", ""))
    edge_count = int(group.get("n_candidate_edges", len(group.get("edges", []) or [])))
    if not group_id or edge_count < 1 or edge_count > settings.stitch_export_backstop_max_edges:
        return None
    if group.get("match_type") == "1:1" and len(group.get("edges", []) or []) <= 1:
        return None

    edge_keys = _group_edge_keys(group)
    manual_hits = sorted(edge_keys & manual_pairs)
    coincidence_rows, coincidence_conflicts, coincidence_labels = _group_coincidence(group)
    conflict, agreement, comparable = _group_physical(group, target_domains)
    uncertainty = _uncertainty(group)
    forced = bool(set(map(str, group.get("ref_ids", []) or [])) & forced_ref_ids)

    tags: list[str] = []
    if manual_hits or forced:
        tags.append("manual_pair")
    if coincidence_rows:
        tags.append("coincidence")
    if conflict >= 0.25:
        tags.append("physical_conflict")
    if agreement >= 0.5:
        tags.append("physical_agreement")
    if ROLE_PATTERN.search(_role_text(group)):
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
        120.0 * bool(manual_hits or forced)
        + 85.0 * bool(coincidence_rows)
        + 25.0 * coincidence_conflicts
        + 65.0 * conflict
        + 50.0 * agreement
        + 18.0 * bool(group_id in known_none)
        + 15.0 * bool(ROLE_PATTERN.search(_role_text(group)))
        + 20.0 * uncertainty
        + min(edge_count, 12)
    )
    audit = {
        "tags": tags,
        "manual_pair_hits": [list(pair) for pair in manual_hits],
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
    forced_ref_ids: set[str],
) -> list[RankedGroup]:
    pools: dict[str, list[tuple[float, str, RankedGroup]]] = defaultdict(list)
    target_domains = _target_domains(dataset_id)
    known_none = _known_none_group_ids(dataset_id)
    scanned = 0
    for group in iter_sidecar_groups(sidecar):
        scanned += 1
        ranked = _rank_group(
            group,
            manual_pairs=manual_pairs,
            known_none=known_none,
            target_domains=target_domains,
            forced_ref_ids=forced_ref_ids,
        )
        if ranked is None:
            continue
        for tag in ranked.tags:
            _push_pool(pools[tag], ranked)

    ordered = {
        tag: [item[2] for item in sorted(pool, reverse=True)]
        for tag, pool in pools.items()
    }
    selected: list[RankedGroup] = []
    seen: set[str] = set()
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
    selected = []
    for ranked in ranked_groups:
        group = copy.deepcopy(ranked.group)
        group["experimental_wave_selection"] = ranked.audit
        if variant == "no_physical":
            group = _strip_physical(group)
        group["alternatives"] = generate_top_k_alternatives(
            group.get("edges", []),
            ref_geoms=group.get("ref_geometries", {}),
            target_geoms=group.get("target_geometries", {}),
            k=k_alternatives,
        )
        selected.append(group)

    _fill_spatial_context(selected, dataset_id)
    suffix = wave_name if variant == "enriched" else f"{wave_name}_no_physical"
    batch_dir = output_root / f"{dataset_id}_{suffix}"
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
            "physical_metadata_visible": variant == "enriched",
            "same_side_coincidence_visible": True,
        },
        "groups": selected,
    }
    (batch_dir / "batch.json").write_text(json.dumps(batch))
    generated = generate_stitch_evidence(batch, batch_dir)
    if len(generated) != len(selected):
        raise RuntimeError(
            f"{dataset_id}/{variant}: generated {len(generated)} of {len(selected)} packs"
        )
    return batch_dir


def _control_subset(
    selections: dict[str, list[RankedGroup]], control_count: int
) -> dict[str, list[RankedGroup]]:
    candidates = [
        (ranked.audit["physical_conflict"] + ranked.audit["physical_agreement"], dataset_id, ranked)
        for dataset_id, groups in selections.items()
        for ranked in groups
        if ranked.audit["physical_comparable_edges"] > 0
        and (
            ranked.audit["physical_conflict"] > 0
            or ranked.audit["physical_agreement"] > 0
        )
    ]
    candidates.sort(key=lambda item: (item[0], item[2].score), reverse=True)
    result: dict[str, list[RankedGroup]] = defaultdict(list)
    per_dataset: dict[str, int] = defaultdict(int)
    for _physical_score, dataset_id, ranked in candidates:
        if per_dataset[dataset_id] >= 2:
            continue
        result[dataset_id].append(ranked)
        per_dataset[dataset_id] += 1
        if sum(map(len, result.values())) >= control_count:
            break
    if sum(map(len, result.values())) != control_count:
        raise RuntimeError(f"Only found {sum(map(len, result.values()))} A/B controls")
    return dict(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sidecar-root", type=Path, default=Path("data/experiments/stitch_physical_v7")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/agents/stitching/batches")
    )
    parser.add_argument("--wave-name", default="physical_context_v7_20260715")
    parser.add_argument("--manual-queue", type=Path, default=Path("research/results/physical_feature_ablation_2026-07-15.json"))
    parser.add_argument("--regressions", type=Path, default=Path("tests/fixtures/physical_match_regressions.json"))
    parser.add_argument("--control-count", type=int, default=10)
    parser.add_argument("--alternatives", type=int, default=8)
    args = parser.parse_args()

    manual = _manual_pairs(args.manual_queue)
    regression_payload = json.loads(args.regressions.read_text())
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
            forced_ref_ids=forced_refs.get(dataset_id, set()),
        )
        gc.collect()

    controls = _control_subset(selections, args.control_count)
    batch_dirs = []
    for dataset_id, groups in selections.items():
        batch_dirs.append(
            _write_batch(
                dataset_id,
                groups,
                sidecar_root=args.sidecar_root,
                output_root=args.output_root,
                wave_name=args.wave_name,
                variant="enriched",
                k_alternatives=args.alternatives,
            )
        )
        gc.collect()
    control_dirs = []
    for dataset_id, groups in controls.items():
        control_dirs.append(
            _write_batch(
                dataset_id,
                groups,
                sidecar_root=args.sidecar_root,
                output_root=args.output_root,
                wave_name=args.wave_name,
                variant="no_physical",
                k_alternatives=args.alternatives,
            )
        )
        gc.collect()

    manifest = {
        "schema_version": 1,
        "wave": args.wave_name,
        "panel": "v7-candidate",
        "unique_group_count": sum(map(len, selections.values())),
        "paired_control_count": sum(map(len, controls.values())),
        "batch_dirs": [str(path) for path in batch_dirs],
        "control_batch_dirs": [str(path) for path in control_dirs],
        "selections": {
            dataset_id: [
                {"group_id": ranked.group_id, **ranked.audit} for ranked in groups
            ]
            for dataset_id, groups in selections.items()
        },
        "controls": {
            dataset_id: [ranked.group_id for ranked in groups]
            for dataset_id, groups in controls.items()
        },
    }
    manifest_path = args.output_root / f"{args.wave_name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Wrote {manifest['unique_group_count']} enriched packs + "
        f"{manifest['paired_control_count']} paired controls -> {manifest_path}"
    )


if __name__ == "__main__":
    main()
