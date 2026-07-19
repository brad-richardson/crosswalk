"""Panel-vote soft labels for the resolver prototype.

Panel batches (``data/agents/stitching/batches/*/votes.csv``) record, per group,
each provider's chosen edge set. These cover more groups than the curated
``labels/stitching`` exports, so they are a candidate way to expand training
coverage with *soft* per-edge keep probabilities.

Reliability weighting: ``codex`` is a systematic conservative dissenter (drops
center-junction edges the others keep), so it is down-weighted rather than
dropped. The per-edge soft keep-probability is the weighted fraction of
providers whose chosen edge set includes the edge.

Vote group_ids are batch-vintage hashes, so they are mapped to current sidecar
groups by edge overlap (same churn-robust principle as
``stitch_eval.recover_labeled_groups``).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from crosswalk.agent_labeling.consensus_desired import (
    map_desired_to_ids,
    parse_desired_edges,
)

# Default provider reliability weights (see module docstring).
DEFAULT_PROVIDER_WEIGHTS: dict[str, float] = {
    "claude": 1.0,
    "agy": 1.0,
    "codex": 0.5,
}


def _parse_edge_set(raw) -> frozenset[tuple[str, str]]:
    if raw is None or (isinstance(raw, float)):
        return frozenset()
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return frozenset()
    return frozenset((str(a), str(b)) for a, b in data)


def load_votes(paths: list[str | Path]) -> pd.DataFrame:
    """Load and concatenate votes CSVs, keeping the latest vote per (group, provider).

    A group_id can recur across batch phases (re-votes); the most recent
    timestamp wins.
    """
    frames = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        frame = pd.read_csv(p, dtype={"group_id": str})
        batch_path = p.parent / "batch.json"
        dataset_id = ""
        if batch_path.exists():
            try:
                dataset_id = str(json.loads(batch_path.read_text()).get("dataset_id", ""))
            except (OSError, ValueError, TypeError):
                dataset_id = ""
        elif p.parent.name.startswith("dataset="):
            dataset_id = p.parent.name.removeprefix("dataset=")
        frame["dataset_id"] = dataset_id
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "group_id",
                "provider",
                "choice",
                "confidence",
                "edge_set",
                "timestamp",
            ]
        )
    df = pd.concat(frames, ignore_index=True)
    df = df[df["error"].isna()] if "error" in df.columns else df
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["dataset_id", "group_id", "provider"], keep="last")
    return df.reset_index(drop=True)


def load_evidence(paths: list[str | Path]) -> pd.DataFrame:
    """Load durable per-group displayed candidate universes.

    ``labels/votes/dataset=*/evidence.csv`` is the preferred source.  The
    evidence payload records actual source ids for every edge the voter could
    see, so never-selected candidates and NONE ballots can contribute resolver
    supervision without guessing from the current sidecar.
    """
    frames: list[pd.DataFrame] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"group_id": str})
        if path.parent.name.startswith("dataset="):
            dataset_id = path.parent.name.removeprefix("dataset=")
        else:
            batch_path = path.parent / "batch.json"
            try:
                dataset_id = str(json.loads(batch_path.read_text()).get("dataset_id", ""))
            except (OSError, ValueError, TypeError):
                dataset_id = ""
        frame["dataset_id"] = dataset_id
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["dataset_id", "group_id", "evidence_id", "evidence"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["dataset_id", "group_id", "evidence_id"], keep="last"
    )


def load_archived_label_maps(
    votes_df: pd.DataFrame,
    batches_root: str | Path,
) -> dict[str, dict[str, dict[str, str]]]:
    """Load historical R#/T# maps keyed by ballot ``evidence_id``.

    Desired edges from ``no_exact_option`` ballots use the visible pack labels,
    not source ids.  Only the originating pack's metadata is authoritative.
    Missing archives are skipped conservatively.
    """
    import yaml

    root = Path(batches_root).resolve()
    out: dict[str, dict[str, dict[str, str]]] = {}
    required = {"source_batch", "group_id", "evidence_id"}
    if votes_df.empty or not required <= set(votes_df.columns):
        return out
    for _, row in votes_df[list(required)].drop_duplicates().iterrows():
        evidence_id = str(row.get("evidence_id") or "")
        if not evidence_id:
            continue
        metadata_path = (
            root / str(row.get("source_batch")) / str(row.get("group_id")) / "metadata.yaml"
        )
        try:
            resolved = metadata_path.resolve()
            if not resolved.is_relative_to(root):
                continue
            metadata = yaml.safe_load(resolved.read_text()) or {}
        except (OSError, ValueError, TypeError):
            continue
        segments = metadata.get("segments", {}) or {}
        out[evidence_id] = {
            "reference": {
                str(segment["label"]): str(segment["id"])
                for segment in segments.get("reference", [])
                if isinstance(segment, dict) and segment.get("label") and segment.get("id")
            },
            "target": {
                str(segment["label"]): str(segment["id"])
                for segment in segments.get("target", [])
                if isinstance(segment, dict) and segment.get("label") and segment.get("id")
            },
        }
    return out


def _gather_group_edges(g: dict) -> list[dict]:
    if g.get("candidate_edges"):
        return list(g["candidate_edges"])
    out = list(g.get("edges", []) or [])
    out.extend(g.get("rejected_edges", []) or [])
    return out


def _parse_displayed_edges(raw) -> frozenset[tuple[str, str]]:
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return frozenset()
    if not isinstance(payload, dict):
        return frozenset()
    return frozenset(
        (str(edge.get("ref_id")), str(edge.get("target_id")))
        for edge in payload.get("displayed_edges", [])
        if isinstance(edge, dict) and edge.get("ref_id") and edge.get("target_id")
    )


def _evidence_universes(evidence_df: pd.DataFrame | None) -> dict[str, frozenset[tuple[str, str]]]:
    if evidence_df is None or evidence_df.empty:
        return {}
    out: dict[str, frozenset[tuple[str, str]]] = {}
    for _, row in evidence_df.iterrows():
        evidence_id = str(row.get("evidence_id") or "")
        if evidence_id:
            out[evidence_id] = _parse_displayed_edges(row.get("evidence"))
    return out


def _map_vote_groups_to_sidecar(
    groups: list[dict],
    votes_df: pd.DataFrame,
    evidence_df: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Map each vote group_id -> best sidecar group_id by edge overlap.

    A vote group's candidate proxy is the union of all providers' chosen edges.
    Uses candidate_edges when available (full universe) else
    edges+rejected_edges — otherwise we under-expand soft vote coverage.
    """
    edge_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for g in groups:
        for e in _gather_group_edges(g):
            edge_groups[(str(e["ref_id"]), str(e["target_id"]))].add(g["group_id"])

    mapping: dict[str, str] = {}
    universes = _evidence_universes(evidence_df)
    for vgid, sub in votes_df.groupby("group_id"):
        union: set[tuple[str, str]] = set()
        for _, row in sub.iterrows():
            evidence_id = str(row.get("evidence_id") or "")
            union |= set(universes.get(evidence_id, frozenset()))
            if not universes.get(evidence_id):
                union |= _parse_edge_set(row.get("edge_set"))
        cnt: dict[str, int] = defaultdict(int)
        for e in union:
            for sgid in edge_groups.get(e, ()):
                cnt[sgid] += 1
        if cnt:
            # #354: sort before max() so a count tie resolves to the smallest
            # group_id, not hash-order-dependent set iteration. This feeds the
            # experimental resolver's training table via edge_soft_labels().
            mapping[str(vgid)] = max(sorted(cnt), key=cnt.get)
    return mapping


def edge_soft_labels(
    groups: list[dict],
    votes_df: pd.DataFrame,
    provider_weights: dict[str, float] | None = None,
    dataset_id: str | None = None,
    evidence_df: pd.DataFrame | None = None,
    label_maps: dict[str, dict[str, dict[str, str]]] | None = None,
) -> pd.DataFrame:
    """Per-edge weighted panel keep-probability, mapped to sidecar groups.

    Returns a DataFrame with columns
    ``[group_id, ref_id, target_id, soft_keep, n_providers, unanimous]``
    where ``soft_keep`` is the reliability-weighted fraction of providers whose
    chosen edge set contained the edge.  With ``evidence_df``, every displayed
    candidate is emitted (including never-selected edges and unanimous
    ``all_edges_no_match`` groups).  Without evidence, the conservative legacy
    behavior remains: only an edge selected by at least one voter is emitted.
    """
    if dataset_id is not None and "dataset_id" in votes_df.columns:
        votes_df = votes_df[votes_df["dataset_id"] == dataset_id]
    if evidence_df is not None and dataset_id is not None and "dataset_id" in evidence_df.columns:
        evidence_df = evidence_df[evidence_df["dataset_id"] == dataset_id]
    if votes_df.empty:
        return pd.DataFrame()

    weights = provider_weights or DEFAULT_PROVIDER_WEIGHTS
    vgid_to_sgid = _map_vote_groups_to_sidecar(groups, votes_df, evidence_df)
    gmap = {g["group_id"]: g for g in groups}
    universes = _evidence_universes(evidence_df)

    rows: list[dict] = []
    for vgid, sub in votes_df.groupby("group_id"):
        sgid = vgid_to_sgid.get(str(vgid))
        if sgid is None:
            continue
        sidecar_edges = {
            (str(e["ref_id"]), str(e["target_id"])) for e in _gather_group_edges(gmap[sgid])
        }
        has_group_evidence = any(
            universes.get(str(row.get("evidence_id") or "")) for _, row in sub.iterrows()
        )
        if not has_group_evidence:
            # Backward-compatible conservative path for historical raw batches:
            # the union of chosen edges is observed, but the complement is not.
            legacy_num = defaultdict(float)
            legacy_chosen_by = defaultdict(int)
            legacy_wsum = 0.0
            legacy_n = 0
            observed: set[tuple[str, str]] = set()
            for _, row in sub.iterrows():
                weight = weights.get(row["provider"], 1.0)
                chosen = _parse_edge_set(row.get("edge_set"))
                legacy_wsum += weight
                legacy_n += 1
                observed |= chosen
                for edge in chosen & sidecar_edges:
                    legacy_num[edge] += weight
                    legacy_chosen_by[edge] += 1
            if legacy_wsum > 0:
                for edge in sorted(observed & sidecar_edges):
                    rows.append(
                        {
                            "dataset_id": dataset_id or "",
                            "group_id": sgid,
                            "ref_id": edge[0],
                            "target_id": edge[1],
                            "soft_keep": legacy_num[edge] / legacy_wsum,
                            "n_providers": legacy_n,
                            "unanimous": int(legacy_chosen_by[edge] == legacy_n),
                            "evidence_complete": 0,
                        }
                    )
            continue

        num = defaultdict(float)
        denom = defaultdict(float)
        chosen_by = defaultdict(int)
        considered_by = defaultdict(int)
        any_complete_evidence = False
        for _, vrow in sub.iterrows():
            w = weights.get(vrow["provider"], 1.0)
            es = _parse_edge_set(vrow["edge_set"])
            evidence_id = str(vrow.get("evidence_id") or "")
            displayed = universes.get(evidence_id, frozenset()) & sidecar_edges
            none_reason = str(vrow.get("none_reason") or "").strip()

            # These ballots contain no safe exact selection target.  A desired
            # R#/T# set is usable only with its historical pack label map;
            # insufficient evidence is an abstention by contract.
            if none_reason == "no_exact_option":
                label_map = (label_maps or {}).get(evidence_id)
                desired = (
                    map_desired_to_ids(parse_desired_edges(vrow.get("desired_edges")), label_map)
                    if label_map
                    else None
                )
                if desired is None or not desired <= displayed:
                    continue
                es = desired
            if none_reason == "insufficient_evidence":
                continue
            if not es and not displayed:
                continue
            if not es and none_reason != "all_edges_no_match":
                # Legacy/unknown NONE reason: do not invent negatives.
                continue

            universe = set(displayed) if displayed else set(es & sidecar_edges)
            if displayed:
                any_complete_evidence = True
            for edge in universe:
                denom[edge] += w
                considered_by[edge] += 1
                if edge in es:
                    num[edge] += w
                    chosen_by[edge] += 1

        for e in sorted(denom):
            if denom[e] <= 0:
                continue
            rows.append(
                {
                    "dataset_id": dataset_id or "",
                    "group_id": sgid,
                    "ref_id": e[0],
                    "target_id": e[1],
                    "soft_keep": num[e] / denom[e],
                    "n_providers": considered_by[e],
                    "unanimous": int(chosen_by[e] == considered_by[e]),
                    "evidence_complete": int(any_complete_evidence),
                }
            )
    return pd.DataFrame(rows)


def default_votes_paths(batches_root: str | Path) -> list[Path]:
    """All ``votes.csv`` under a batches root directory."""
    return sorted(Path(batches_root).glob("*/votes.csv"))


def default_evidence_paths(votes_root: str | Path) -> list[Path]:
    """All sibling ``evidence.csv`` exports under a votes root."""
    return sorted(Path(votes_root).glob("*/evidence.csv"))
