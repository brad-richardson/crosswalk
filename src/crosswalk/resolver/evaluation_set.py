"""Freeze existing exact human labels and block related weak supervision.

This is an experimental resolver benchmark, not a source of corrected labels.
Disputed, incomplete, and contradictory records are quarantined automatically.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from crosswalk.agent_labeling.stitch_eval import parse_selected_edge_set, recover_labeled_groups
from crosswalk.agent_labeling.stitch_provenance import sha256_file, sha256_json
from crosswalk.resolver.extract import build_edge_table
from crosswalk.resolver.observations import parent_group_id, text

EVALUATION_SCHEMA_VERSION = 1


def scope_tokens(dataset: str, groups=(), refs=(), targets=(), lineages=()) -> set[str]:
    """Refs are global GERS IDs; target and historical group IDs are dataset-local."""
    return (
        {f"ref:{r}" for r in refs if text(r)}
        | {f"target:{dataset}:{t}" for t in targets if text(t)}
        | {f"group:{dataset}:{parent_group_id(str(g))}" for g in groups if text(g)}
        | {f"lineage:{v}" for v in lineages if text(v)}
    )


def _list(value) -> list:
    return json.loads(value) if text(value) else []


def observation_tokens(row: dict) -> set[str]:
    groups = [row.get("group_id"), row.get("parent_group_id")]
    for column in ("parent_group_ids", "historical_human_group_ids"):
        groups.extend(_list(row.get(column)))
    lineages = [row.get("lineage_id"), *_list(row.get("lineage_ids"))]
    return set(_list(row.get("context_scope_tokens"))) | scope_tokens(
        str(row["dataset_id"]), groups, [row.get("ref_id")], [row.get("target_id")], lineages
    )


def audit_existing_labels(
    dataset: str,
    groups: list[dict],
    labels: pd.DataFrame,
    *,
    exclusions: dict[tuple[str, str], str] | None = None,
    candidates_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Keep complete exact-edge human claims, with a reason for every human row.

    Clean drift mappings require explicit original membership coverage. A label
    without that scope is usable only on an unchanged group ID. Membership-set
    labels and agent opinions cannot supply exact-edge evaluation truth.
    """
    human = labels[labels["labeler"] == "brad"].copy()
    exact = human[human["label_semantics"] == "pair"]
    exclusions = exclusions or {}
    gmap = {str(g["group_id"]): g for g in groups}
    recovery = recover_labeled_groups(groups, exact)
    mapping = dict(recovery["clean"])
    mapping.update({g: g for g in recovery["empty"] if g in gmap})
    split = {g for g, *_ in recovery["split"]}
    table = build_edge_table(
        groups, exact, dataset, include_split=False, candidates_df=candidates_df
    )
    conflicts = {
        record["current_group_id"]
        for record in table.attrs.get("build_audit", {}).get("quarantined_groups", [])
    }
    retained = set()
    audit = []
    for row in human.to_dict("records"):
        hgid = str(row["group_id"])
        gid = mapping.get(hgid, "")
        group = gmap.get(gid, {})
        selected = parse_selected_edge_set(row.get("selected_edges", "[]"))
        refs = set(_list(row.get("ref_ids"))) | {r for r, _ in selected}
        targets = set(_list(row.get("target_ids"))) | {t for _, t in selected}
        # Even quarantined human scopes block weak data; a model vote must not
        # silently adjudicate a disputed label through the training back door.
        tokens = scope_tokens(
            dataset,
            [hgid, gid],
            refs | set(group.get("ref_ids", [])),
            targets | set(group.get("target_ids", [])),
        )
        part = table[table["group_id"] == gid] if len(table) else table
        reason = "eligible"
        if row.get("label_semantics") != "pair":
            reason = "membership_not_exact_edges"
        elif (dataset, hgid) in exclusions:
            reason = "disputed: " + exclusions[(dataset, hgid)]
        elif "partial_identity" in text(row.get("notes")):
            reason = "partial_identity_not_resolution_truth"
        elif gid in conflicts:
            reason = "contradictory_human_labels"
        elif hgid in split:
            reason = "split_or_missing_selected_edges"
        elif not gid:
            reason = "unmapped_or_missing_source"
        elif not group.get("candidate_edges"):
            reason = "incomplete_legacy_candidate_universe"
        elif gid != hgid and (
            not set(group.get("ref_ids", [])) <= set(_list(row.get("ref_ids")))
            or not set(group.get("target_ids", [])) <= set(_list(row.get("target_ids")))
        ):
            reason = "changed_group_without_original_scope"
        elif part.empty:
            reason = "no_eligible_candidate_edges"
        elif selected != frozenset(
            zip(part.loc[part["keep"] == 1, "ref_id"], part.loc[part["keep"] == 1, "target_id"])
        ):
            reason = "selected_edges_outside_eligible_universe"
        if reason == "eligible":
            retained.add(gid)
            tokens |= set().union(*(observation_tokens(r) for r in part.to_dict("records")))
        audit.append(
            {
                "dataset_id": dataset,
                "human_group_id": hgid,
                "group_id": gid,
                "status": reason,
                "anchored": not text(row.get("session_id")).lower().startswith("deanchored"),
                "scope_tokens": sorted(tokens),
            }
        )
    # A disputed/incomplete duplicate must not survive through another label
    # attached to the same current group.
    retained -= {a["group_id"] for a in audit if a["status"] != "eligible"}
    for record in audit:
        if record["status"] == "eligible" and record["group_id"] not in retained:
            record["status"] = "related_quarantined_human_label"
    if table.empty:
        return table, audit
    return table[table["group_id"].isin(retained)].copy(), audit


class _Components:
    def __init__(self):
        self.parents: dict[str, str] = {}

    def root(self, token: str) -> str:
        self.parents.setdefault(token, token)
        root = token
        while self.parents[root] != root:
            root = self.parents[root]
        while token != root:
            previous = self.parents[token]
            self.parents[token] = root
            token = previous
        return root

    def join(self, tokens: set[str]):
        if not tokens:
            return
        roots = sorted({self.root(t) for t in tokens})
        for root in roots[1:]:
            self.parents[root] = roots[0]


def freeze_partitions(
    gold: pd.DataFrame,
    weak: pd.DataFrame,
    audit: list[dict],
    *,
    n_folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Freeze connected-component folds and exclude ALL human-related weak data.

    Unknowns, overlap conflicts, and below-quorum observations still connect
    lineage. Exclusion is stronger than per-fold exclusion: no weak observation
    touching any existing human scope is admitted in any policy or fold.
    """
    components = _Components()
    gold_tokens = [observation_tokens(r) for r in gold.to_dict("records")]
    weak_tokens = [observation_tokens(r) for r in weak.to_dict("records")]
    for tokens in [*gold_tokens, *weak_tokens]:
        components.join(tokens)
    for record in audit:
        components.join(set(record["scope_tokens"]))
    members = defaultdict(list)
    for token in sorted(components.parents):
        members[components.root(token)].append(token)
    identities = {root: sha256_json(tokens) for root, tokens in members.items()}
    token_index = {t: identities[components.root(t)] for t in sorted(components.parents)}
    human_components = {token_index[t] for record in audit for t in record["scope_tokens"]}
    gold = gold.copy()
    weak = weak.copy()
    gold["component_id"] = [token_index[min(t)] for t in gold_tokens]
    weak["component_id"] = [token_index[min(t)] for t in weak_tokens]
    weak["excluded_human_scope"] = weak["component_id"].isin(human_components)
    counts = gold.groupby("component_id").size().to_dict()
    if n_folds < 2 or len(counts) < n_folds:
        raise ValueError(
            f"Need at least {n_folds} independent human components; found {len(counts)}"
        )
    loads = [0] * n_folds
    folds = {}
    for component in sorted(counts, key=lambda c: (-counts[c], c)):
        fold = min(range(n_folds), key=lambda f: (loads[f], f))
        folds[component] = fold
        loads[fold] += counts[component]
    gold["fold"] = gold["component_id"].map(folds).astype(int)
    return gold, weak, token_index


def connect_frozen_scopes(frame: pd.DataFrame, token_index: dict[str, str], blocked: set[str]):
    """Propagate frozen exclusions and component budgets through new full context."""
    tokens = [observation_tokens(r) for r in frame.to_dict("records")]
    components = _Components()
    for scope in tokens:
        # Frozen component aliases preserve transitive connections through rows
        # that are absent from this particular new wave.
        components.join(scope | {f"frozen:{token_index[t]}" for t in scope if t in token_index})
    bad = {
        components.root(f"frozen:{component}")
        for component in blocked
        if f"frozen:{component}" in components.parents
    }
    members = defaultdict(list)
    for token in sorted(components.parents):
        members[components.root(token)].append(token)
    identities = {root: sha256_json(values) for root, values in members.items()}
    return pd.DataFrame(
        {
            "component_id": [identities[components.root(min(t))] for t in tokens],
            "excluded_human_scope": [components.root(min(t)) in bad for t in tokens],
        },
        index=frame.index,
    )


def exclude_frozen_scopes(frame: pd.DataFrame, token_index: dict[str, str], blocked: set[str]):
    """Conservatively propagate a frozen exclusion through newly added evidence."""
    return connect_frozen_scopes(frame, token_index, blocked)["excluded_human_scope"].astype(bool)


def load_frozen_evaluation(directory: str | Path):
    """Verify the frozen outputs before any experiment can consume their truth."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError("Unsupported evaluation schema")
    expected = {"gold.parquet", "weak_partition.parquet", "audit.json", "components.json"}
    if set(manifest.get("outputs", {})) != expected:
        raise ValueError("Incomplete frozen evaluation manifest")
    for name, digest in manifest["outputs"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Frozen evaluation integrity failure: {name}")
    return (
        manifest,
        pd.read_parquet(directory / "gold.parquet"),
        pd.read_parquet(directory / "weak_partition.parquet"),
        json.loads((directory / "components.json").read_text()),
    )
