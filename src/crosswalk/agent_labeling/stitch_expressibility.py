"""Option-menu EXPRESSIBILITY of the stitching option generator.

Expressibility answers a generator-quality question that is upstream of, and
independent from, the panel: *can the offered option menu even express the
settled answer?* A settled label whose exact edge set matches NO generated
option is unreachable — a unanimous panel can score at most a near-miss, and a
human reviewer would have to hand-construct the answer.

The metric is: fraction of settled (pair-semantics, non-reject-all) stitching
labels whose EXACT selected edge set equals some option generated for the
current sidecar group that best-corresponds to the label. It is measured
without running any provider — it depends only on the option generator
(``generate_top_k_alternatives`` + ``build_stitch_options``) and the sidecar.

Denominator note: a label only counts toward expressibility when its full edge
set is contained in one current sidecar group ("clean-recoverable"). Labels
whose edges have split across / been lost from the current sidecar (component
drift) are a separate data-drift concern, not an option-menu gap, and are
reported separately rather than folded into the rate.

This module reads sidecars and labels READ ONLY.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from ..matching.alternatives import generate_top_k_alternatives
from ..matching.stitch_options import build_stitch_options


def _label_edge_set(raw: str) -> frozenset[tuple[str, str]]:
    """Parse a label's ``selected_edges`` JSON into a frozenset of (ref, tgt)."""
    if not isinstance(raw, str) or not raw:
        return frozenset()
    try:
        edges = json.loads(raw)
    except (ValueError, TypeError):
        return frozenset()
    return frozenset((str(e["ref_id"]), str(e["target_id"])) for e in edges)


def settled_labels(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Filter a stitching-label frame to settled pair-semantics rows.

    Drops reject-all rows (empty ``selected_edges``) and, if a
    ``label_semantics`` column exists, any ``set``-semantics rows (which assert a
    whole-group set rather than specific pairs and so are not option-menu picks).
    """
    df = labels_df
    if "label_semantics" in df.columns:
        df = df[df["label_semantics"].fillna("pair") != "set"]
    mask = df["selected_edges"].map(lambda x: bool(_label_edge_set(x)))
    return df[mask].reset_index(drop=True)


def _option_edge_sets(group: dict, k: int) -> list[frozenset[tuple[str, str]]]:
    """Generate the group's option menu and return each option's edge set."""
    g = dict(group)
    g["alternatives"] = generate_top_k_alternatives(
        group.get("edges", []),
        ref_geoms=group.get("ref_geometries", {}),
        target_geoms=group.get("target_geometries", {}),
        k=k,
    )
    ctx = build_stitch_options(g)
    return [
        frozenset((e["ref_id"], e["target_id"]) for e in opt["edges"]) for opt in ctx["options"]
    ]


def _recover_group(
    edge_index: dict[tuple[str, str], list[str]],
    groups_by_id: dict[str, dict],
    label_es: frozenset,
) -> tuple[dict | None, int, int]:
    """Find the sidecar group holding the most of a label's edges.

    Returns (group, n_matched, n_total). ``group`` is None when no edge of the
    label survives in any current group. The label is "clean-recoverable" iff
    ``n_matched == n_total`` (its whole edge set lives in that one group).
    """
    counts: dict[str, int] = defaultdict(int)
    for e in label_es:
        for gid in edge_index.get(e, ()):
            counts[gid] += 1
    if not counts:
        return None, 0, len(label_es)
    # #367: label_es is a frozenset, so its iteration order (hence insertion
    # order into counts) is hash-seed dependent; sort before max() so a count
    # tie resolves deterministically to the smallest group_id.
    best = max(sorted(counts), key=counts.get)
    return groups_by_id[best], counts[best], len(label_es)


@dataclass
class LabelExpressibility:
    label_group_id: str
    sidecar_group_id: str | None
    match_type: str
    n_label_edges: int
    n_group_edges: int
    recoverable: bool  # whole label edge set contained in one current group
    covered: bool  # exact label edge set equals some generated option
    n_options: int


@dataclass
class ExpressibilityReport:
    dataset: str
    k: int
    n_settled: int
    n_recoverable: int
    n_covered: int
    per_label: list[LabelExpressibility] = field(default_factory=list)

    @property
    def expressibility(self) -> float | None:
        """Covered / clean-recoverable (the option-menu-only population)."""
        return round(self.n_covered / self.n_recoverable, 4) if self.n_recoverable else None

    @property
    def misses(self) -> list[LabelExpressibility]:
        return [r for r in self.per_label if r.recoverable and not r.covered]

    def summary(self) -> dict:
        return {
            "dataset": self.dataset,
            "k": self.k,
            "n_settled": self.n_settled,
            "n_recoverable": self.n_recoverable,
            "n_covered": self.n_covered,
            "expressibility": self.expressibility,
            "n_misses": len(self.misses),
        }


def measure_expressibility(
    dataset: str,
    groups: list[dict],
    labels_df: pd.DataFrame,
    k: int = 8,
) -> ExpressibilityReport:
    """Compute option-menu expressibility of ``groups`` against settled labels.

    Args:
        dataset: dataset id (recorded on the report).
        groups: the current sidecar's ``groups`` list (each with ``edges`` and,
            when available, ``ref_geometries`` / ``target_geometries``).
        labels_df: raw stitching-label frame (``group_id``, ``selected_edges``,
            ``match_type``, optional ``label_semantics``).
        k: top-K organic alternatives to request per group (seeds are appended
            on top by the generator).
    """
    df = settled_labels(labels_df)

    groups_by_id: dict[str, dict] = {}
    edge_index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for g in groups:
        gid = str(g["group_id"])
        groups_by_id[gid] = g
        for e in g.get("edges", []):
            edge_index[(str(e["ref_id"]), str(e["target_id"]))].append(gid)

    per_label: list[LabelExpressibility] = []
    n_recoverable = 0
    n_covered = 0
    for _, row in df.iterrows():
        label_es = _label_edge_set(row["selected_edges"])
        group, matched, total = _recover_group(edge_index, groups_by_id, label_es)
        recoverable = group is not None and matched == total
        covered = False
        n_options = 0
        if recoverable:
            n_recoverable += 1
            option_sets = _option_edge_sets(group, k)
            n_options = len(option_sets)
            covered = any(label_es == s for s in option_sets)
            if covered:
                n_covered += 1
        per_label.append(
            LabelExpressibility(
                label_group_id=str(row["group_id"]),
                sidecar_group_id=str(group["group_id"]) if group is not None else None,
                match_type=str(row.get("match_type", "")),
                n_label_edges=len(label_es),
                n_group_edges=len(group.get("edges", [])) if group is not None else 0,
                recoverable=recoverable,
                covered=covered,
                n_options=n_options,
            )
        )

    return ExpressibilityReport(
        dataset=dataset,
        k=k,
        n_settled=len(df),
        n_recoverable=n_recoverable,
        n_covered=n_covered,
        per_label=per_label,
    )
