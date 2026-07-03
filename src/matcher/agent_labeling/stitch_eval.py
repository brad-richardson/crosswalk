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
    human_df: pd.DataFrame, group_metas: dict[str, dict]
) -> dict[str, str]:
    """Map each panel group_id -> best-matching human group_id.

    A human label matches a panel group when its selected edges lie among the
    group's segments; the group is assigned the human label with the greatest
    selected-edge overlap. Returns {panel_group_id: human_group_id}.
    """
    human_records = []
    for _, row in human_df.iterrows():
        es = _human_edge_set(row["selected_edges"])
        human_records.append((str(row["group_id"]), es))

    mapping: dict[str, str] = {}
    for gid, meta in group_metas.items():
        seg_ids = _group_segment_ids(meta)
        best_hgid = None
        best_overlap = 0
        for hgid, hes in human_records:
            if not hes:
                continue
            overlap = sum(1 for (r, t) in hes if r in seg_ids and t in seg_ids)
            if overlap > best_overlap:
                best_overlap = overlap
                best_hgid = hgid
        if best_hgid is not None:
            mapping[gid] = best_hgid
    return mapping


def recover_labeled_groups(groups: list[dict], human_df: pd.DataFrame) -> dict:
    """Map old human-labeled group_ids to current sidecar groups by edge overlap.

    group_ids are hashes of the exact ref/target id sets, so component shifts
    (e.g. after a model-param change) break exact-id recovery. This maps each
    human label to the current sidecar group containing the most of its
    selected edges.

    Returns a dict with:
    - target_group_ids: sorted distinct sidecar group_ids to build the eval batch
    - clean: [(human_gid, sidecar_gid)] where ALL selected edges are in one group
    - split: [(human_gid, sidecar_gid, n_matched, n_total)] edges span groups
    - empty: [human_gid] human labels with no selected edges (NONE / reject-all)
    - lost:  [human_gid] non-empty labels whose edges no longer survive
    """
    from collections import defaultdict

    edge_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for g in groups:
        for e in g.get("edges", []):
            edge_groups[(str(e["ref_id"]), str(e["target_id"]))].add(g["group_id"])

    clean, split, empty, lost = [], [], [], []
    for _, row in human_df.iterrows():
        hgid = str(row["group_id"])
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
        best_gid = max(cnt, key=cnt.get)
        if cnt[best_gid] == len(hes):
            clean.append((hgid, best_gid))
        else:
            split.append((hgid, best_gid, cnt[best_gid], len(hes)))

    targets = sorted({bg for _, bg in clean} | {bg for _, bg, _, _ in split})
    return {
        "target_group_ids": targets,
        "clean": clean,
        "split": split,
        "empty": empty,
        "lost": lost,
    }


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

    mapping = map_human_labels_to_groups(human_df, group_metas)
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
            )
        )
    return results


def summarize(results: list[GroupEval]) -> dict:
    """Compute per-provider and per-consensus-tier agreement summaries."""
    n = len(results)
    summary: dict = {"n_groups": n}
    if n == 0:
        return summary

    # Panel (consensus) agreement.
    summary["panel_exact_rate"] = round(sum(r.exact_match for r in results) / n, 3)
    summary["panel_mean_f1"] = round(sum(r.f1 for r in results) / n, 3)

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
