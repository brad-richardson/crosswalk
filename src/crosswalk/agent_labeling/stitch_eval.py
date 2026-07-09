"""Validation of the agent stitching panel against human stitching labels.

CRITICAL FRAMING: the ~33 Boston human stitching labels are old and of varying
quality. Disagreement is NOT treated as agent failure. This module reports
agreement rates AND surfaces panel-vs-human contradictions as *label-quality
review candidates*, plus option-coverage gaps (human edge sets that no current
option can express).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from ..matching.sliver import annotate_group_sliver_flags


def _parse_edge_set(raw: str) -> frozenset:
    """Parse a JSON edge-set string ([[ref,tgt],...]) to a frozenset of tuples."""
    if not raw or (isinstance(raw, float)):
        return frozenset()
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return frozenset()
    return frozenset((str(a), str(b)) for a, b in data)


def _human_edge_set(selected_edges_raw: str) -> frozenset:
    try:
        edges = json.loads(selected_edges_raw)
    except (ValueError, TypeError):
        return frozenset()
    return frozenset((str(e["ref_id"]), str(e["target_id"])) for e in edges)


def _is_set_label(row) -> bool:
    """True when a human row uses SET semantics (membership, not pairs)."""
    val = row.get("label_semantics") if hasattr(row, "get") else row["label_semantics"]
    return str(val or "pair") == "set"


def _parse_id_list(raw) -> frozenset:
    """Parse a JSON id array (``ref_ids`` / ``target_ids``) into a string set."""
    if raw is None or isinstance(raw, float) or not str(raw).strip():
        return frozenset()
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return frozenset()
    return frozenset(str(x) for x in data)


def set_label_metrics(
    pred_edges: frozenset,
    ref_members: frozenset,
    target_members: frozenset,
) -> tuple[bool, float, float]:
    """Score a predicted edge set against a SET label's membership.

    THE PARITY-CRITICAL CORE. Replicated verbatim in
    ``mbench.eval.stitch_metrics.set_label_metrics`` and guarded by
    ``tests/unit/test_mbench_set_metric_parity.py`` — keep the two in lockstep.

    Returns ``(membership_exact, boundary_precision, coverage)``:
      * membership_exact - the segments incident to the predicted edges are
        EXACTLY this membership (both sides).
      * boundary_precision - fraction of predicted edges whose BOTH endpoints are
        members; 1.0 when nothing is predicted.
      * coverage - fraction of members with >= 1 incident predicted edge; 1.0
        when the membership is empty.
    """
    pred_ref = frozenset(r for r, _ in pred_edges)
    pred_tgt = frozenset(t for _, t in pred_edges)
    membership_exact = (pred_ref == ref_members) and (pred_tgt == target_members)

    n_edges = len(pred_edges)
    within = sum(1 for (r, t) in pred_edges if r in ref_members and t in target_members)
    boundary_precision = within / n_edges if n_edges else 1.0

    n_members = len(ref_members) + len(target_members)
    covered = len(ref_members & pred_ref) + len(target_members & pred_tgt)
    coverage = covered / n_members if n_members else 1.0
    return membership_exact, boundary_precision, coverage


def edge_prf(pred: frozenset, truth: frozenset) -> tuple[float, float, float]:
    """Edge-level precision / recall / F1 between two edge sets.

    Two empty sets count as a perfect match (both say 'no edges').
    """
    if not pred and not truth:
        return 1.0, 1.0, 1.0
    tp = len(pred & truth)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(truth) if truth else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


@dataclass
class GroupEval:
    group_id: str
    human_group_id: str
    match_type: str
    human_edge_set: frozenset
    consensus: str
    routing: str
    panel_choice: str
    panel_edge_set: frozenset
    exact_match: bool
    f1: float
    option_covered: bool  # human edge set equals some option's edge set
    provider_votes: dict = field(default_factory=dict)  # provider -> (choice, edge_set)
    # Sliver-filtered variants: junction-sliver edges (near-zero physical
    # overlap) are removed from BOTH the panel and human edge sets before
    # comparison, so agreement is not distorted by whether either side happened
    # to include an artifact edge. Equal to the raw fields when no group edge is
    # a sliver (or when batch geometries are unavailable to classify them).
    human_edge_set_filtered: frozenset = field(default_factory=frozenset)
    panel_edge_set_filtered: frozenset = field(default_factory=frozenset)
    exact_match_filtered: bool = False
    f1_filtered: float = 0.0


def _load_group_metadata(group_dir: Path) -> dict:
    return yaml.safe_load((group_dir / "metadata.yaml").read_text())


def _option_edge_sets(meta: dict) -> dict[str, frozenset]:
    """letter -> frozenset((ref_id, target_id)) using the metadata segment tables."""
    ref_by_label = {s["label"]: str(s["id"]) for s in meta["segments"]["reference"]}
    tgt_by_label = {s["label"]: str(s["id"]) for s in meta["segments"]["target"]}
    out: dict[str, frozenset] = {}
    for opt in meta["options"]:
        pairs = set()
        for e in opt["edges"]:
            pairs.add(
                (ref_by_label.get(e["ref"], e["ref"]), tgt_by_label.get(e["target"], e["target"]))
            )
        out[opt["letter"]] = frozenset(pairs)
    return out


def _group_segment_ids(meta: dict) -> set[str]:
    ids = {str(s["id"]) for s in meta["segments"]["reference"]}
    ids |= {str(s["id"]) for s in meta["segments"]["target"]}
    return ids


def map_human_labels_to_groups(
    human_df: pd.DataFrame,
    group_metas: dict[str, dict],
    group_candidate_edges: dict[str, frozenset] | None = None,
) -> dict[str, str]:
    """Map each panel group_id -> best-matching human group_id.

    Preferred signal is edge-level overlap: when a group's full candidate edge
    set is known (from the batch file), a human label only qualifies if at
    least one of its selected edges exists among the group's candidate edges,
    and ties break by max edge overlap (then segment overlap). Segment-ID
    membership alone can mis-map a label whose specific edges no longer exist
    in the group, skewing option-coverage/agreement metrics — so it is used
    only as a fallback when candidate edges are unavailable (old packs without
    a batch.json).

    Returns {panel_group_id: human_group_id}.
    """
    human_records = []
    for _, row in human_df.iterrows():
        es = _human_edge_set(row["selected_edges"])
        human_records.append((str(row["group_id"]), es))

    group_candidate_edges = group_candidate_edges or {}

    mapping: dict[str, str] = {}
    for gid, meta in group_metas.items():
        seg_ids = _group_segment_ids(meta)
        cand_edges = group_candidate_edges.get(gid)
        best_hgid = None
        best_score = (0, 0)  # (edge_overlap, segment_overlap)
        for hgid, hes in human_records:
            if not hes:
                continue
            seg_overlap = sum(1 for (r, t) in hes if r in seg_ids and t in seg_ids)
            if cand_edges is not None:
                edge_overlap = len(hes & cand_edges)
                if edge_overlap == 0:
                    continue  # require >=1 human edge among the candidate edges
                score = (edge_overlap, seg_overlap)
            else:
                score = (0, seg_overlap)
                if seg_overlap == 0:
                    continue
            if score > best_score:
                best_score = score
                best_hgid = hgid
        if best_hgid is not None:
            mapping[gid] = best_hgid
    return mapping


def _load_batch_candidate_edges(batch_dir: Path) -> dict[str, frozenset]:
    """Load each group's full candidate edge set from the batch file, if present."""
    batch_path = Path(batch_dir) / "batch.json"
    if not batch_path.exists():
        return {}
    try:
        batch = json.loads(batch_path.read_text())
    except (ValueError, OSError):
        return {}
    out: dict[str, frozenset] = {}
    for g in batch.get("groups", []):
        gid = str(g.get("group_id"))
        out[gid] = frozenset((str(e["ref_id"]), str(e["target_id"])) for e in g.get("edges", []))
    return out


def _load_batch_sliver_edges(batch_dir: Path) -> dict[str, frozenset]:
    """Load each group's junction-sliver edge set from ``batch.json``.

    Reads the full groups (with geometries + alignment fractions) and classifies
    each edge with the shared hybrid rule. Returns
    ``{group_id: frozenset((ref_id, target_id))}`` of edges flagged as slivers.
    Kept per-group (not batch-wide) so an edge pair is only filtered within the
    group whose geometries classified it as a sliver. Empty when no
    ``batch.json`` is present (nothing to filter -> filtered == raw).
    """
    batch_path = Path(batch_dir) / "batch.json"
    if not batch_path.exists():
        return {}
    try:
        batch = json.loads(batch_path.read_text())
    except (ValueError, OSError):
        return {}
    out: dict[str, frozenset] = {}
    for g in batch.get("groups", []):
        annotated, _ = annotate_group_sliver_flags(g)
        out[str(g.get("group_id"))] = frozenset(
            (str(e["ref_id"]), str(e["target_id"])) for e in annotated if e.get("is_sliver")
        )
    return out


def recover_labeled_groups(groups: list[dict], human_df: pd.DataFrame) -> dict:
    """Map old human-labeled group_ids to current sidecar groups by edge overlap.

    group_ids are hashes of the exact ref/target id sets, so component shifts
    (e.g. after a model-param change) break exact-id recovery. This maps each
    human label to the current sidecar group containing the most of its
    selected edges.

    SET-semantics labels carry no edges, so they are recovered by MEMBERSHIP
    overlap (ref_ids/target_ids vs each group's segment ids) into their own
    buckets — they must never be misread as reject-all/NONE.

    Returns a dict with:
    - target_group_ids: sorted distinct sidecar group_ids to build the eval batch
      (includes groups recovered for set labels)
    - clean: [(human_gid, sidecar_gid)] where ALL selected edges are in one group
    - split: [(human_gid, sidecar_gid, n_matched, n_total)] edges span groups
    - empty: [human_gid] PAIR labels with no selected edges (NONE / reject-all)
    - lost:  [human_gid] non-empty pair labels whose edges no longer survive
    - set:   [(human_gid, sidecar_gid)] set labels recovered by membership overlap
    - set_lost: [human_gid] set labels whose members appear in no current group
    """
    from collections import defaultdict

    edge_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    seg_groups: dict[str, set[str]] = defaultdict(set)
    for g in groups:
        for e in g.get("edges", []):
            edge_groups[(str(e["ref_id"]), str(e["target_id"]))].add(g["group_id"])
            seg_groups[str(e["ref_id"])].add(g["group_id"])
            seg_groups[str(e["target_id"])].add(g["group_id"])

    clean, split, empty, lost = [], [], [], []
    set_recovered: list[tuple[str, str]] = []
    set_lost: list[str] = []
    for _, row in human_df.iterrows():
        hgid = str(row["group_id"])
        # SET labels carry no edges (selected_edges == "[]") but are MATCH
        # membership assertions, NOT reject-all — classifying them as ``empty``
        # would both miscount them as NONE and exclude their groups from the
        # eval batch. Recover them by MEMBERSHIP overlap instead.
        if _is_set_label(row):
            members = _parse_id_list(row.get("ref_ids")) | _parse_id_list(row.get("target_ids"))
            cnt_s: dict[str, int] = defaultdict(int)
            for s in members:
                for gid in seg_groups.get(s, ()):
                    cnt_s[gid] += 1
            if cnt_s:
                # #354: sort candidates before max() so a count tie breaks on
                # the lexicographically smallest group_id, not dict/set
                # iteration order (hash-seed-dependent for str keys — the
                # source of the ±15 row/process wobble without
                # PYTHONHASHSEED=0 pinned).
                set_recovered.append((hgid, max(sorted(cnt_s), key=cnt_s.get)))
            else:
                set_lost.append(hgid)
            continue
        hes = _human_edge_set(row["selected_edges"])
        if not hes:
            empty.append(hgid)
            continue
        cnt: dict[str, int] = defaultdict(int)
        for e in hes:
            for gid in edge_groups.get(e, ()):
                cnt[gid] += 1
        if not cnt:
            lost.append(hgid)
            continue
        # #354: sort before max() for a deterministic tie-break (smallest
        # group_id wins on a count tie) instead of hash-order-dependent dict
        # iteration.
        best_gid = max(sorted(cnt), key=cnt.get)
        if cnt[best_gid] == len(hes):
            clean.append((hgid, best_gid))
        else:
            split.append((hgid, best_gid, cnt[best_gid], len(hes)))

    targets = sorted(
        {bg for _, bg in clean} | {bg for _, bg, _, _ in split} | {bg for _, bg in set_recovered}
    )
    return {
        "target_group_ids": targets,
        "clean": clean,
        "split": split,
        "empty": empty,
        "lost": lost,
        "set": set_recovered,
        "set_lost": set_lost,
    }


def recover_empty_reject_all(groups: list[dict], human_df: pd.DataFrame) -> dict:
    """Recover which reject-all (empty-edge) human labels map to a current group.

    Reject-all labels (empty ``selected_edges``) record *that* a group was
    rejected but not *which* segments it held, so segment-overlap recovery is
    impossible — the only stored key is the ``group_id`` hash of the original
    ref/target id sets. A label therefore survives only if the exact same sets
    still form a component in the current sidecar (verbatim ``group_id`` match);
    otherwise it is unrecoverable and can only be judged by re-running the panel
    on whatever current group now covers that geography.

    Returns:
        {"recovered": [group_id, ...], "unrecoverable": [group_id, ...]}.
    """
    gids = {g["group_id"] for g in groups}
    recovered: list[str] = []
    unrecoverable: list[str] = []
    for _, row in human_df.iterrows():
        if _is_set_label(row):
            continue  # SET label: a MATCH membership assertion, not a reject-all
        if _human_edge_set(row["selected_edges"]):
            continue  # has edges -> not a reject-all label
        hgid = str(row["group_id"])
        (recovered if hgid in gids else unrecoverable).append(hgid)
    return {"recovered": recovered, "unrecoverable": unrecoverable}


def evaluate_batch(batch_dir: Path, human_df: pd.DataFrame) -> list[GroupEval]:
    """Compare panel results in a batch against the human labels."""
    batch_dir = Path(batch_dir)
    consensus_df = pd.read_csv(batch_dir / "consensus.csv", dtype={"group_id": str})
    votes_df = pd.read_csv(batch_dir / "votes.csv", dtype={"group_id": str})

    group_metas: dict[str, dict] = {}
    for _, row in consensus_df.iterrows():
        gid = str(row["group_id"])
        gdir = batch_dir / gid
        if (gdir / "metadata.yaml").exists():
            group_metas[gid] = _load_group_metadata(gdir)

    candidate_edges = _load_batch_candidate_edges(batch_dir)
    sliver_edges = _load_batch_sliver_edges(batch_dir)
    mapping = map_human_labels_to_groups(human_df, group_metas, candidate_edges)
    human_by_gid = {str(r["group_id"]): r for _, r in human_df.iterrows()}

    results: list[GroupEval] = []
    for _, row in consensus_df.iterrows():
        gid = str(row["group_id"])
        if gid not in mapping:
            continue
        hgid = mapping[gid]
        hrow = human_by_gid[hgid]
        human_es = _human_edge_set(hrow["selected_edges"])
        meta = group_metas[gid]
        opt_sets = _option_edge_sets(meta)

        panel_es = _parse_edge_set(row["edge_set"])
        exact = panel_es == human_es
        _, _, f1 = edge_prf(panel_es, human_es)
        option_covered = any(human_es == s for s in opt_sets.values())

        # Sliver-filtered comparison: drop THIS group's sliver edges from BOTH
        # sides so a disagreement that is only about an artifact edge does not
        # count. Per-group lookup: an edge is only filtered in the group whose
        # geometries classified it as a sliver.
        group_slivers = sliver_edges.get(gid, frozenset())
        panel_es_f = panel_es - group_slivers
        human_es_f = human_es - group_slivers
        exact_f = panel_es_f == human_es_f
        _, _, f1_f = edge_prf(panel_es_f, human_es_f)

        # Per-provider votes for this group.
        prov_votes = {}
        for _, vrow in votes_df[votes_df["group_id"] == gid].iterrows():
            prov_votes[vrow["provider"]] = (
                vrow["choice"],
                _parse_edge_set(vrow["edge_set"]),
            )

        results.append(
            GroupEval(
                group_id=gid,
                human_group_id=hgid,
                match_type=meta.get("match_type", ""),
                human_edge_set=human_es,
                consensus=row["consensus"],
                routing=row["routing"],
                panel_choice=str(row["choice"]),
                panel_edge_set=panel_es,
                exact_match=exact,
                f1=f1,
                option_covered=option_covered,
                provider_votes=prov_votes,
                human_edge_set_filtered=human_es_f,
                panel_edge_set_filtered=panel_es_f,
                exact_match_filtered=exact_f,
                f1_filtered=f1_f,
            )
        )
    return results


def summarize(results: list[GroupEval]) -> dict:
    """Compute per-provider and per-consensus-tier agreement summaries."""
    n = len(results)
    summary: dict = {"n_groups": n}
    if n == 0:
        return summary

    # Panel (consensus) agreement — raw.
    summary["panel_exact_rate"] = round(sum(r.exact_match for r in results) / n, 3)
    summary["panel_mean_f1"] = round(sum(r.f1 for r in results) / n, 3)

    # Panel agreement with junction slivers removed from both sides. Reported
    # alongside the raw numbers so sliver artifacts don't silently inflate or
    # deflate agreement.
    summary["panel_exact_rate_filtered"] = round(
        sum(r.exact_match_filtered for r in results) / n, 3
    )
    summary["panel_mean_f1_filtered"] = round(sum(r.f1_filtered for r in results) / n, 3)
    summary["n_groups_sliver_affected"] = sum(
        1
        for r in results
        if (r.panel_edge_set != r.panel_edge_set_filtered)
        or (r.human_edge_set != r.human_edge_set_filtered)
    )

    # Per consensus tier.
    tiers: dict[str, list[GroupEval]] = {}
    for r in results:
        tiers.setdefault(r.consensus, []).append(r)
    summary["by_consensus"] = {
        tier: {
            "n": len(rs),
            "exact_rate": round(sum(x.exact_match for x in rs) / len(rs), 3),
            "mean_f1": round(sum(x.f1 for x in rs) / len(rs), 3),
        }
        for tier, rs in sorted(tiers.items())
    }

    # Per provider.
    providers: dict[str, list[tuple[bool, float]]] = {}
    for r in results:
        for prov, (_choice, es) in r.provider_votes.items():
            exact = es == r.human_edge_set
            _, _, f1 = edge_prf(es, r.human_edge_set)
            providers.setdefault(prov, []).append((exact, f1))
    summary["by_provider"] = {
        prov: {
            "n": len(vals),
            "exact_rate": round(sum(e for e, _ in vals) / len(vals), 3),
            "mean_f1": round(sum(f for _, f in vals) / len(vals), 3),
        }
        for prov, vals in sorted(providers.items())
    }

    # Option-coverage gap.
    covered = sum(r.option_covered for r in results)
    summary["option_coverage"] = {
        "covered": covered,
        "gap": n - covered,
        "gap_rate": round((n - covered) / n, 3),
    }
    return summary


def disagreement_report(results: list[GroupEval]) -> list[GroupEval]:
    """Groups where the panel contradicts the human label (label-quality review)."""
    return [r for r in results if not r.exact_match]


@dataclass
class SetGroupEval:
    """Set-semantics evaluation of the panel choice for one human set label."""

    group_id: str
    human_group_id: str
    match_type: str
    ref_members: frozenset
    target_members: frozenset
    panel_edge_set: frozenset
    membership_exact: bool
    boundary_precision: float
    coverage: float


def _map_set_labels_to_groups(
    set_df: pd.DataFrame,
    group_members: dict[str, frozenset],
) -> dict[str, str]:
    """Map each SET human row's group_id -> best panel group by membership overlap.

    Mirrors :func:`mbench.eval.stitch_metrics.map_set_labels_to_groups`. Set rows
    carry no edges, so they map by ref_ids ∪ target_ids segment overlap, with a
    verbatim group_id preferred when it still shares a segment.
    """
    from collections import defaultdict

    seg_groups: dict[str, set[str]] = defaultdict(set)
    for gid, members in group_members.items():
        for s in members:
            seg_groups[s].add(gid)

    mapping: dict[str, str] = {}
    for _, row in set_df.iterrows():
        members = _parse_id_list(row.get("ref_ids")) | _parse_id_list(row.get("target_ids"))
        hgid = str(row["group_id"])
        if not members:
            if hgid in group_members:
                mapping[hgid] = hgid
            continue
        counts: dict[str, int] = defaultdict(int)
        for s in members:
            for gid in seg_groups.get(s, ()):
                counts[gid] += 1
        if not counts:
            continue
        # Same deterministic tie-break as recover_labeled_groups (#354): sort
        # before max() so ties resolve to the smallest group_id.
        best = max(sorted(counts), key=counts.get)
        mapping[hgid] = hgid if counts.get(hgid, 0) > 0 else best
    return mapping


def evaluate_set_labels(batch_dir: Path, human_df: pd.DataFrame) -> list[SetGroupEval]:
    """Score panel choices against SET-semantics human labels (membership/boundary/coverage).

    Set labels are excluded from the edge-F1 panel eval (they carry no edges) and
    scored here instead: the panel's consensus edge set is compared against the
    human membership. Empty (no set labels) when the CSV has none.
    """
    batch_dir = Path(batch_dir)
    if "label_semantics" not in human_df.columns:
        return []
    set_df = human_df[human_df.apply(_is_set_label, axis=1)]
    if set_df.empty:
        return []

    consensus_df = pd.read_csv(batch_dir / "consensus.csv", dtype={"group_id": str})
    candidate_edges = _load_batch_candidate_edges(batch_dir)
    group_members = {
        gid: frozenset(r for r, _ in edges) | frozenset(t for _, t in edges)
        for gid, edges in candidate_edges.items()
    }
    panel_es_by_gid = {
        str(row["group_id"]): _parse_edge_set(row["edge_set"]) for _, row in consensus_df.iterrows()
    }

    mapping = _map_set_labels_to_groups(set_df, group_members)
    set_by_hgid = {str(r["group_id"]): r for _, r in set_df.iterrows()}

    results: list[SetGroupEval] = []
    for hgid, gid in mapping.items():
        row = set_by_hgid[hgid]
        ref_members = _parse_id_list(row.get("ref_ids"))
        tgt_members = _parse_id_list(row.get("target_ids"))
        # Panel prediction restricted to this group's candidate edges.
        panel_es = panel_es_by_gid.get(gid, frozenset()) & candidate_edges.get(gid, frozenset())
        exact, boundary, coverage = set_label_metrics(panel_es, ref_members, tgt_members)
        results.append(
            SetGroupEval(
                group_id=gid,
                human_group_id=hgid,
                match_type=str(row.get("match_type") or ""),
                ref_members=ref_members,
                target_members=tgt_members,
                panel_edge_set=panel_es,
                membership_exact=exact,
                boundary_precision=boundary,
                coverage=coverage,
            )
        )
    return results


def summarize_set(results: list[SetGroupEval]) -> dict:
    """Aggregate set-label panel metrics (membership exact / boundary / coverage)."""
    n = len(results)
    if n == 0:
        return {"n_set_groups": 0}
    return {
        "n_set_groups": n,
        "membership_exact_rate": round(sum(r.membership_exact for r in results) / n, 3),
        "boundary_precision": round(sum(r.boundary_precision for r in results) / n, 3),
        "coverage": round(sum(r.coverage for r in results) / n, 3),
    }
