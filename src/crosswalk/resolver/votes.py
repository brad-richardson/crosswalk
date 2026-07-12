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
        frames.append(pd.read_csv(p, dtype={"group_id": str}))
    if not frames:
        return pd.DataFrame(
            columns=["group_id", "provider", "choice", "confidence", "edge_set", "timestamp"]
        )
    df = pd.concat(frames, ignore_index=True)
    df = df[df["error"].isna()] if "error" in df.columns else df
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["group_id", "provider"], keep="last")
    return df.reset_index(drop=True)


def _group_edge_sources(group: dict) -> list[dict]:
    """All edge sources that can represent a group's candidate universe.

    Mirrors ``stitch_eval.recover_labeled_groups``: edges + candidate_edges +
    rejected_edges (+ optimizer_assignment if present). This closes the gap where
    vote batches from fresh packs reference an edge that is only in the uncapped
    candidate graph and would otherwise be unmappable.
    """
    sources = []
    for key in ("edges", "candidate_edges", "rejected_edges", "optimizer_assignment"):
        for e in group.get(key, []) or []:
            try:
                # Normalize to ref_id/target_id dict for uniform handling
                sources.append(e)
            except Exception:
                continue
    return sources


def _group_segment_ids(group: dict) -> set[str]:
    """Segment ids (ref + target) for a group, from ref_ids/target_ids + edges."""
    segs: set[str] = set()
    for rid in group.get("ref_ids", []) or []:
        segs.add(str(rid))
    for tid in group.get("target_ids", []) or []:
        segs.add(str(tid))
    for e in _group_edge_sources(group):
        try:
            segs.add(str(e["ref_id"]))
            segs.add(str(e["target_id"]))
        except (KeyError, TypeError):
            continue
    return segs


def _map_vote_groups_to_sidecar(groups: list[dict], votes_df: pd.DataFrame) -> dict[str, str]:
    """Map each vote group_id -> best sidecar group_id by edge + segment overlap.

    A vote group's candidate proxy is the union of all providers' chosen edges.
    Primary signal is edge overlap over the full candidate universe
    (edges + candidate_edges + rejected_edges). Fallback is segment membership
    overlap (ref_ids/target_ids), mirroring ``stitch_eval.recover_labeled_groups``
    and ``evaluateset`` set-label recovery – needed when a batch's edge set is
    empty (NONE) or references edges pruned from the current sidecar.
    """
    edge_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    seg_groups: dict[str, set[str]] = defaultdict(set)

    for g in groups:
        gid = g["group_id"]
        for e in _group_edge_sources(g):
            try:
                key = (str(e["ref_id"]), str(e["target_id"]))
            except (KeyError, TypeError):
                continue
            edge_groups[key].add(gid)
            seg_groups[str(e["ref_id"])].add(gid)
            seg_groups[str(e["target_id"])].add(gid)
        for sid in g.get("ref_ids", []) or []:
            seg_groups[str(sid)].add(gid)
        for sid in g.get("target_ids", []) or []:
            seg_groups[str(sid)].add(gid)

    mapping: dict[str, str] = {}
    for vgid, sub in votes_df.groupby("group_id"):
        union: set[tuple[str, str]] = set()
        seg_union: set[str] = set()
        for raw in sub["edge_set"]:
            es = _parse_edge_set(raw)
            union |= es
            for r, t in es:
                seg_union.add(r)
                seg_union.add(t)

        # Primary: edge overlap
        cnt: dict[str, int] = defaultdict(int)
        for e in union:
            for sgid in edge_groups.get(e, ()):
                cnt[sgid] += 1
        if cnt:
            # #354: sort before max() so a count tie resolves to the smallest
            # group_id, not hash-order-dependent set iteration.
            mapping[str(vgid)] = max(sorted(cnt), key=cnt.get)
            continue

        # Fallback: segment membership overlap (for NONE votes or edge-churned groups)
        if seg_union:
            seg_cnt: dict[str, int] = defaultdict(int)
            for s in seg_union:
                for sgid in seg_groups.get(s, ()):
                    seg_cnt[sgid] += 1
            if seg_cnt:
                mapping[str(vgid)] = max(sorted(seg_cnt), key=seg_cnt.get)
                continue

        # No overlap at all: unmappable (e.g., old batch referencing deleted geography)

    return mapping


def _sidecar_candidate_edges(group: dict) -> set[tuple[str, str]]:
    """Union of all edge keys that represent the group's candidate universe.

    Prefers ``candidate_edges`` when present (uncapped, authoritative), falls back to
    ``edges`` + ``rejected_edges``. This is the universe the resolver trains on,
    so soft labels must cover it, not just the selected assignment.
    """
    # Prefer candidate_edges (stage-1 uncapped) if available
    if group.get("candidate_edges"):
        return {
            (str(e["ref_id"]), str(e["target_id"]))
            for e in group.get("candidate_edges", [])
            if "ref_id" in e and "target_id" in e
        }
    # Legacy fallback: selected + capped rejected
    edges = set()
    for key in ("edges", "rejected_edges"):
        for e in group.get(key, []) or []:
            try:
                edges.add((str(e["ref_id"]), str(e["target_id"])))
            except (KeyError, TypeError):
                continue
    return edges


def edge_soft_labels(
    groups: list[dict],
    votes_df: pd.DataFrame,
    provider_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Per-edge weighted panel keep-probability, mapped to sidecar groups.

    Returns a DataFrame with columns
    ``[group_id, ref_id, target_id, soft_keep, n_providers, unanimous]``
    where ``soft_keep`` is the reliability-weighted fraction of providers whose
    chosen edge set contained the edge, restricted to edges that exist in the
    mapped sidecar group's **full candidate universe** (candidate_edges when
    present, else edges + rejected_edges). This closes the ~95% gap where the
    previous implementation only emitted rows for the selected assignment.
    """
    weights = provider_weights or DEFAULT_PROVIDER_WEIGHTS
    vgid_to_sgid = _map_vote_groups_to_sidecar(groups, votes_df)
    gmap = {g["group_id"]: g for g in groups}

    rows: list[dict] = []
    for vgid, sub in votes_df.groupby("group_id"):
        sgid = vgid_to_sgid.get(str(vgid))
        if sgid is None:
            continue
        group = gmap.get(sgid)
        if group is None:
            continue
        sidecar_edges = _sidecar_candidate_edges(group)
        if not sidecar_edges:
            continue
        num = defaultdict(float)
        chosen_by = defaultdict(int)
        wsum = 0.0
        n_prov = 0
        for _, vrow in sub.iterrows():
            w = weights.get(vrow["provider"], 1.0)
            wsum += w
            n_prov += 1
            es = _parse_edge_set(vrow["edge_set"])
            for e in es & sidecar_edges:
                num[e] += w
                chosen_by[e] += 1
        if wsum == 0:
            continue
        for e in sidecar_edges:
            rows.append(
                {
                    "group_id": sgid,
                    "ref_id": e[0],
                    "target_id": e[1],
                    "soft_keep": num[e] / wsum,
                    "n_providers": n_prov,
                    "unanimous": int(chosen_by[e] == n_prov or chosen_by[e] == 0),
                }
            )
    return pd.DataFrame(rows)


def default_votes_paths(batches_root: str | Path) -> list[Path]:
    """All ``votes.csv`` under a batches root directory."""
    return sorted(Path(batches_root).glob("*/votes.csv"))
