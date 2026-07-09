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

SET-semantics labels (``label_semantics == "set"``) carry no edges — they
assert group MEMBERSHIP (``ref_ids`` / ``target_ids``), not specific pairs —
so ``settled_labels()`` drops them (see its docstring). They are NOT excluded
from this module's metric: dropping them silently would leave expressibility
blind to exactly the large-group failure mode where every generated option
uses the FULL ref/target set and none can express a human's exclusions (e.g.
co_bogota_roads group 3c3e6853: the human kept 55/64 refs and 153/191 targets,
but every generated option used all 191 targets, capping best-option boundary
precision at 0.806 with zero visibility from the pair-only metric). SET labels
are scored separately, via ``set_settled_labels()`` + the existing set-label
scoring machinery (``stitch_eval.set_label_metrics``), and reported alongside
the pair-semantics results rather than folded into them — a set label is
scored as membership/boundary/coverage against the BEST generated option, not
an exact edge-set match, since it does not assert which particular edges are
correct.

This module reads sidecars and labels READ ONLY.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from ..matching.alternatives import generate_top_k_alternatives
from ..matching.stitch_options import build_stitch_options
from .stitch_eval import _is_set_label, _parse_id_list, set_label_metrics


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


def set_settled_labels(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Filter a stitching-label frame to settled SET-semantics rows.

    Companion to ``settled_labels()``: where that function DROPS
    ``label_semantics == "set"`` rows (they carry no edges, so they are not
    option-menu picks), this selects exactly those rows instead, so callers can
    score them separately with set-label scoring (membership/boundary/coverage)
    rather than silently losing them from the expressibility metric.

    Drops set rows with empty membership (neither ``ref_ids`` nor
    ``target_ids`` populated) — the set-label analogue of a pair reject-all
    row: nothing was asserted, so there is nothing to score.
    """
    if labels_df.empty:
        return labels_df.iloc[0:0].reset_index(drop=True)
    df = labels_df[labels_df.apply(_is_set_label, axis=1)]
    if df.empty:
        return df.reset_index(drop=True)
    has_members = df.apply(
        lambda row: bool(
            _parse_id_list(row.get("ref_ids")) | _parse_id_list(row.get("target_ids"))
        ),
        axis=1,
    )
    return df[has_members].reset_index(drop=True)


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


def _recover_group_by_members(
    seg_index: dict[str, list[str]],
    groups_by_id: dict[str, dict],
    members: frozenset[str],
) -> tuple[dict | None, int, int]:
    """Find the sidecar group holding the most of a SET label's members.

    Mirrors ``_recover_group``, but a SET label carries no edges (only a
    ``ref_ids`` / ``target_ids`` membership assertion), so recovery keys on
    segment ids — both endpoints of every group edge — rather than edge pairs.

    Returns (group, n_matched, n_total). The label is "clean-recoverable" iff
    ``n_matched == n_total`` (its whole membership lives in one current group).
    """
    counts: dict[str, int] = defaultdict(int)
    for m in members:
        for gid in seg_index.get(m, ()):
            counts[gid] += 1
    if not counts:
        return None, 0, len(members)
    # Same deterministic tie-break as _recover_group (#367): sort before max()
    # so a count tie resolves to the smallest group_id, not hash-seed-dependent
    # iteration order.
    best = max(sorted(counts), key=counts.get)
    return groups_by_id[best], counts[best], len(members)


def _best_set_option_metrics(
    group: dict,
    k: int,
    ref_members: frozenset[str],
    target_members: frozenset[str],
) -> tuple[bool, float, float, int]:
    """Score every generated option against a SET label's membership; keep the best.

    Uses the existing set-label scoring machinery
    (``stitch_eval.set_label_metrics``) rather than reimplementing it — each
    generated option's edge set stands in for the "predicted edges" argument
    that machinery already scores panel/consensus choices against.

    "Best" maximizes ``(membership_exact, boundary_precision, coverage)`` in
    that priority order: an option that nails membership exactly always wins;
    among inexact options, the one with the fewest predicted edges landing
    outside the asserted membership (highest boundary precision) wins, ties
    broken by coverage. This is the direct analogue of exact edge-set
    matching for pair labels, adapted to the fact that a SET label does not
    assert which particular edges are correct — only which segments belong.

    Returns ``(membership_exact, boundary_precision, coverage, n_options)`` for
    the best-scoring option. When the group has no generated options, returns
    ``(False, 0.0, 0.0, 0)`` (nothing to express the membership with).
    """
    option_sets = _option_edge_sets(group, k)
    best: tuple[bool, float, float] = (False, 0.0, 0.0)
    for opt in option_sets:
        scored = set_label_metrics(opt, ref_members, target_members)
        if scored > best:
            best = scored
    return (*best, len(option_sets))


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
class SetLabelExpressibility:
    """Per-label expressibility record for a SET-semantics stitching label.

    Unlike ``LabelExpressibility`` (exact edge-set match), a SET label is
    scored by the BEST generated option's membership/boundary/coverage against
    the label's ``ref_ids`` / ``target_ids`` — see ``_best_set_option_metrics``.
    """

    label_group_id: str
    sidecar_group_id: str | None
    match_type: str
    n_ref_members: int
    n_target_members: int
    recoverable: bool  # whole membership contained in one current group
    covered: bool  # best generated option achieves exact membership match
    best_boundary_precision: float
    best_coverage: float
    n_options: int


@dataclass
class ExpressibilityReport:
    dataset: str
    k: int
    n_settled: int
    n_recoverable: int
    n_covered: int
    per_label: list[LabelExpressibility] = field(default_factory=list)
    # SET-semantics companions to the pair-semantics fields above. Reported
    # alongside (never folded into) the pair numbers, since set labels are
    # scored on membership/boundary/coverage, not exact edge-set match.
    n_set_settled: int = 0
    n_set_recoverable: int = 0
    n_set_covered: int = 0
    per_set_label: list[SetLabelExpressibility] = field(default_factory=list)
    # Rows in the input frame that were neither pair-settled nor set-settled
    # (pair reject-all rows, or set rows with empty membership) — so a reader
    # of ``summary()`` can see dropped-vs-scored instead of that gap being
    # silent, per the bug this module fixes.
    n_dropped: int = 0

    @property
    def expressibility(self) -> float | None:
        """Covered / clean-recoverable (the option-menu-only population)."""
        return round(self.n_covered / self.n_recoverable, 4) if self.n_recoverable else None

    @property
    def misses(self) -> list[LabelExpressibility]:
        return [r for r in self.per_label if r.recoverable and not r.covered]

    @property
    def set_expressibility(self) -> float | None:
        """SET covered / clean-recoverable (mirrors ``expressibility``)."""
        return (
            round(self.n_set_covered / self.n_set_recoverable, 4)
            if self.n_set_recoverable
            else None
        )

    @property
    def set_misses(self) -> list[SetLabelExpressibility]:
        return [r for r in self.per_set_label if r.recoverable and not r.covered]

    @property
    def set_mean_best_boundary_precision(self) -> float | None:
        """Mean of each recoverable SET label's best-option boundary precision.

        The headline diagnostic for the bug this module fixes: even when NO
        option hits exact membership (``set_expressibility`` near 0 for large
        groups), this shows how close the best option gets — e.g. 0.806 for
        co_bogota_roads group 3c3e6853, where every option used all 191
        targets against a human membership that excluded 38 of them.
        """
        rs = [r for r in self.per_set_label if r.recoverable]
        return round(sum(r.best_boundary_precision for r in rs) / len(rs), 4) if rs else None

    @property
    def set_mean_best_coverage(self) -> float | None:
        rs = [r for r in self.per_set_label if r.recoverable]
        return round(sum(r.best_coverage for r in rs) / len(rs), 4) if rs else None

    def summary(self) -> dict:
        return {
            "dataset": self.dataset,
            "k": self.k,
            "n_settled": self.n_settled,
            "n_recoverable": self.n_recoverable,
            "n_covered": self.n_covered,
            "expressibility": self.expressibility,
            "n_misses": len(self.misses),
            "n_set_settled": self.n_set_settled,
            "n_set_recoverable": self.n_set_recoverable,
            "n_set_covered": self.n_set_covered,
            "set_expressibility": self.set_expressibility,
            "n_set_misses": len(self.set_misses),
            "set_mean_best_boundary_precision": self.set_mean_best_boundary_precision,
            "set_mean_best_coverage": self.set_mean_best_coverage,
            "n_dropped": self.n_dropped,
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
    seg_index: dict[str, list[str]] = defaultdict(list)
    for g in groups:
        gid = str(g["group_id"])
        groups_by_id[gid] = g
        seen_segs: set[str] = set()
        for e in g.get("edges", []):
            edge_index[(str(e["ref_id"]), str(e["target_id"]))].append(gid)
            for seg in (str(e["ref_id"]), str(e["target_id"])):
                if seg not in seen_segs:
                    seg_index[seg].append(gid)
                    seen_segs.add(seg)

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

    # SET-semantics labels: scored separately (membership/boundary/coverage
    # against the BEST generated option), not folded into the pair loop above.
    set_df = set_settled_labels(labels_df)

    per_set_label: list[SetLabelExpressibility] = []
    n_set_recoverable = 0
    n_set_covered = 0
    for _, row in set_df.iterrows():
        ref_members = _parse_id_list(row.get("ref_ids"))
        target_members = _parse_id_list(row.get("target_ids"))
        members = ref_members | target_members
        group, matched, total = _recover_group_by_members(seg_index, groups_by_id, members)
        recoverable = group is not None and matched == total
        covered = False
        best_boundary = 0.0
        best_coverage = 0.0
        n_options = 0
        if recoverable:
            n_set_recoverable += 1
            exact, best_boundary, best_coverage, n_options = _best_set_option_metrics(
                group, k, ref_members, target_members
            )
            covered = exact
            if covered:
                n_set_covered += 1
        per_set_label.append(
            SetLabelExpressibility(
                label_group_id=str(row["group_id"]),
                sidecar_group_id=str(group["group_id"]) if group is not None else None,
                match_type=str(row.get("match_type", "")),
                n_ref_members=len(ref_members),
                n_target_members=len(target_members),
                recoverable=recoverable,
                covered=covered,
                best_boundary_precision=best_boundary,
                best_coverage=best_coverage,
                n_options=n_options,
            )
        )

    n_dropped = len(labels_df) - len(df) - len(set_df)

    return ExpressibilityReport(
        dataset=dataset,
        k=k,
        n_settled=len(df),
        n_recoverable=n_recoverable,
        n_covered=n_covered,
        per_label=per_label,
        n_set_settled=len(set_df),
        n_set_recoverable=n_set_recoverable,
        n_set_covered=n_set_covered,
        per_set_label=per_set_label,
        n_dropped=n_dropped,
    )
