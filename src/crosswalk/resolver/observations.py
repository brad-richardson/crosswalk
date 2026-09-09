"""Scoped weak supervision from the existing durable panel-vote archive.

An observation describes the candidates actually shown in one evidence snapshot.
It is never a complete label for a parent group. Unknowns carry NaN targets, and
repeated/overlapping observations cannot manufacture independent voters.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from crosswalk.agent_labeling.consensus_desired import map_desired_to_ids, parse_desired_edges
from crosswalk.agent_labeling.stitch_edge_ballots import (
    EDGE_BALLOT_VERSION,
    validate_edge_decisions,
)
from crosswalk.agent_labeling.stitch_provenance import sha256_json

OBSERVATION_SCHEMA_VERSION = 1
SNAPSHOT_COLUMNS = [
    "dataset_id",
    "group_id",
    "evidence_id",
    "evidence_pack_sha256",
    "panel_invocation_sha256",
]
OBSERVATION_KEY = ["dataset_id", "source_snapshot_id", "ref_id", "target_id"]


def text(value) -> str:
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)


def parent_group_id(group_id: str) -> str:
    """Recover the deterministic parent of the existing __p<hash> child IDs."""
    match = re.fullmatch(r"(.+)__p[0-9a-f]{10}", group_id)
    return match.group(1) if match else group_id


def _pairs(raw) -> frozenset[tuple[str, str]]:
    data = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(data, list):
        raise ValueError("edge set must be a list")
    pairs = []
    for edge in data:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2 or not all(edge):
            raise ValueError("edge set must contain pairs of nonempty IDs")
        pairs.append(tuple(map(str, edge)))
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate edge")
    return frozenset(pairs)


def _legacy_decisions(row: dict, displayed: frozenset, label_map: dict | None, menu=None) -> dict:
    if text(row.get("error")) or text(row.get("choice")) == "ABSTAIN":
        return {}
    reason = text(row.get("none_reason"))
    choice = text(row.get("choice"))
    if reason not in {"", "insufficient_evidence", "all_edges_no_match", "no_exact_option"}:
        raise ValueError("unknown NONE reason")
    if reason == "insufficient_evidence":
        return {}
    if reason == "all_edges_no_match":
        if choice != "NONE" or _pairs(row.get("edge_set")):
            raise ValueError("reject-all reason requires NONE")
        selected = frozenset()
    elif reason == "no_exact_option":
        if choice != "NONE" or not label_map:
            return {}
        selected = map_desired_to_ids(parse_desired_edges(row.get("desired_edges")), label_map)
        if selected is None:
            raise ValueError("unmappable desired edges")
    elif choice == "NONE":
        return {}
    else:
        selected = _pairs(row.get("edge_set"))
        if menu is not None:
            options = {
                str(o["letter"]): frozenset(
                    (str(e["ref_id"]), str(e["target_id"])) for e in o["edges"]
                )
                for o in menu
            }
            if choice not in options or options[choice] != selected:
                raise ValueError("chosen edges do not match the recorded option")
        if not selected:
            return {}
    if not selected <= displayed:
        raise ValueError("selected edges outside displayed evidence")
    return {edge: int(edge in selected) for edge in displayed}


def build_vote_observations(
    votes: pd.DataFrame,
    evidence: pd.DataFrame,
    *,
    label_maps: dict | None = None,
    provider_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Aggregate independent seats only within the same evidence/invocation.

    Every displayed edge is emitted, even when every voter abstains on it.
    ``soft_keep`` is a vote fraction, not calibrated correctness probability.
    ``known`` is false for ties and no-opinion cases. Evidence and invalid-ballot
    failures are counted in ``attrs['observation_audit']`` instead of prompting
    a human or manufacturing a label.
    """
    audit: Counter = Counter()
    evidence_index = {}
    invalid_evidence = set()
    for row in evidence.to_dict("records"):
        key = (
            text(row.get("dataset_id")),
            text(row.get("evidence_id")),
            text(row.get("evidence_pack_sha256")),
        )
        try:
            payload = json.loads(row["evidence"])
            if not isinstance(payload, dict):
                raise ValueError("invalid evidence")
            if payload.get("evidence_id"):
                claimed = payload["evidence_id"]
                body = {k: v for k, v in payload.items() if k != "evidence_id"}
                if claimed != key[1] or claimed != sha256_json(body):
                    raise ValueError("evidence identity does not match payload")
            displayed_rows = payload.get("displayed_edges", [])
            if not isinstance(displayed_rows, list) or any(
                not isinstance(e, dict) or not e.get("ref_id") or not e.get("target_id")
                for e in displayed_rows
            ):
                raise ValueError("invalid displayed edge universe")
            old = evidence_index.get(key)
            if old is not None and old != payload:
                raise ValueError(f"contradictory evidence for {key}")
            evidence_index[key] = payload
        except (ValueError, TypeError, KeyError):
            audit["invalid_evidence"] += 1
            invalid_evidence.add(key)

    frame = votes.copy()
    for column in [*SNAPSHOT_COLUMNS, "provider", "model", "timestamp", "source_batch"]:
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].map(text)
    # Legacy records without invocation provenance cannot borrow a quorum from
    # a different batch. Provenanced replays retain the recorded invocation ID.
    frame["panel_invocation_sha256"] = frame["panel_invocation_sha256"].where(
        frame["panel_invocation_sha256"] != "", "legacy:" + frame["source_batch"]
    )
    rows = []
    for snapshot, sub in frame.groupby(SNAPSHOT_COLUMNS, sort=True, dropna=False):
        dataset, group, evidence_id, pack_sha, invocation = snapshot
        evidence_key = (dataset, evidence_id, pack_sha)
        payload = None if evidence_key in invalid_evidence else evidence_index.get(evidence_key)
        if not payload or text(payload.get("group_id", group)) != group:
            audit["missing_or_incompatible_evidence_groups"] += 1
            continue
        displayed = frozenset(
            (str(e["ref_id"]), str(e["target_id"])) for e in payload.get("displayed_edges", [])
        )
        if not displayed:
            audit["empty_evidence_groups"] += 1
            continue
        # A seat can contribute once. Different model configurations claiming
        # one seat in the same invocation are ambiguous, not extra voters.
        ambiguous = set(sub.groupby("provider")["model"].nunique().loc[lambda s: s > 1].index)
        audit["ambiguous_model_seats"] += len(ambiguous)
        sub = sub[~sub["provider"].isin(ambiguous)].sort_values("timestamp", kind="stable")
        sub = sub.drop_duplicates("provider", keep="last")
        roster = sorted((r.provider, r.model) for r in sub.itertuples())
        observation_id = sha256_json([list(snapshot), roster])
        source_artifacts = payload.get("source_artifacts") or {}
        source_hashes = {
            k: v["sha256"]
            for k, v in source_artifacts.items()
            if isinstance(v, dict) and v.get("sha256")
        }
        # Unknown source snapshots must never collapse across evidence packs.
        source_snapshot = sha256_json(source_hashes) if source_hashes else evidence_id
        parent = parent_group_id(group)
        lineage = sha256_json([dataset, parent])
        positive, negative = Counter(), Counter()
        weighted_positive, total_weight = Counter(), Counter()
        for ballot in sub.to_dict("records"):
            try:
                protocol = text(ballot.get("protocol_version"))
                if protocol or payload.get("protocol_version"):
                    if (
                        protocol != EDGE_BALLOT_VERSION
                        or payload.get("protocol_version") != protocol
                    ):
                        raise ValueError("Unsupported or mismatched direct ballot protocol")
                    if text(ballot.get("error")):
                        continue
                    decisions = validate_edge_decisions(
                        json.loads(ballot["edge_decisions"]),
                        set(displayed),
                        text(ballot.get("none_reason")),
                    )
                else:
                    decisions = _legacy_decisions(
                        ballot,
                        displayed,
                        (label_maps or {}).get(evidence_id),
                        payload.get("option_menu"),
                    )
                weight = (provider_weights or {}).get(ballot["provider"], 1.0)
                if not np.isfinite(weight) or weight <= 0:
                    raise ValueError("provider weights must be finite and positive")
            except (ValueError, TypeError, KeyError):
                audit["invalid_ballots"] += 1
                continue
            for edge, keep in decisions.items():
                positive[edge] += keep
                negative[edge] += 1 - keep
                weighted_positive[edge] += keep * weight
                total_weight[edge] += weight
        for ref, target in sorted(displayed):
            edge = (ref, target)
            n_keep, n_drop = positive[edge], negative[edge]
            n_voters = n_keep + n_drop
            probability = weighted_positive[edge] / total_weight[edge] if n_voters else np.nan
            known = bool(n_voters and n_keep != n_drop)
            tier = "unknown"
            if known:
                if n_voters >= 3 and max(n_keep, n_drop) == n_voters:
                    tier = "unanimous"
                elif n_voters == 3 and max(n_keep, n_drop) == 2:
                    tier = "majority"
                else:
                    tier = "below_quorum"
            rows.append(
                {
                    "dataset_id": dataset,
                    "group_id": group,
                    "parent_group_id": parent,
                    "supervision_kind": "resolution",
                    "lineage_id": lineage,
                    "observation_id": observation_id,
                    "source_snapshot_id": source_snapshot,
                    "evidence_id": evidence_id,
                    "evidence_pack_sha256": pack_sha,
                    "panel_invocation_sha256": invocation,
                    "matching_rubric_version": text(payload.get("matching_rubric_version")),
                    "source_artifacts": json.dumps(source_artifacts, sort_keys=True),
                    "source_batch": sub["source_batch"].iloc[-1] if len(sub) else "",
                    "model_roster": json.dumps(roster),
                    "ref_id": ref,
                    "target_id": target,
                    "soft_keep": probability,
                    "known": known,
                    "vote_tier": tier,
                    "n_keep": n_keep,
                    "n_drop": n_drop,
                    "n_providers": n_voters,
                    "n_panel": len(sub) + len(ambiguous),
                    "unanimous": int(n_voters > 0 and n_keep == n_voters),
                    "evidence_complete": 1,
                }
            )
        audit["observation_groups"] += 1
    result = pd.DataFrame(rows)
    audit["rows"] = len(result)
    audit["known_rows"] = int(result["known"].sum()) if len(result) else 0
    result.attrs["observation_audit"] = dict(audit)
    return result


def reconcile_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate overlapping evidence without adding correlated votes.

    Opposite or tied conclusions mask the edge. Compatible conclusions retain
    the least decisive observation; repeated waves cannot raise confidence or
    quorum. All source observation/parent IDs remain available for exclusion.
    """
    if observations.empty:
        return observations.copy()
    rows = []
    for _, sub in observations.groupby(OBSERVATION_KEY, sort=True, dropna=False):
        considered = sub[sub["n_providers"] > 0]
        directions = set(np.sign(considered["n_keep"] - considered["n_drop"]))
        conflict = len(directions) > 1 or 0 in directions
        known = bool(len(considered) and not conflict)
        pool = considered if len(considered) else sub
        chosen = (
            pool.assign(_certainty=(pool["soft_keep"] - 0.5).abs())
            .sort_values(
                ["_certainty", "n_providers", "observation_id"], kind="stable", na_position="last"
            )
            .iloc[0]
            .drop(labels="_certainty")
            .to_dict()
        )
        chosen.update(
            known=known,
            overlap_conflict=conflict,
            observation_ids=json.dumps(sorted(set(sub["observation_id"]))),
            lineage_ids=json.dumps(sorted(set(sub["lineage_id"]))),
            parent_group_ids=json.dumps(sorted(set(sub["parent_group_id"]))),
            n_observations=int(sub["observation_id"].nunique()),
        )
        if not known:
            chosen.update(soft_keep=np.nan, vote_tier="unknown")
        rows.append(chosen)
    result = pd.DataFrame(rows)
    result.attrs["observation_audit"] = {
        **observations.attrs.get("observation_audit", {}),
        "reconciled_rows": len(result),
        "overlap_conflicts": int(result["overlap_conflict"].sum()),
        "reconciled_known_rows": int(result["known"].sum()),
    }
    return result
