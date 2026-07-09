"""Decompose over-backstop M:N stitch groups into panel-sized sub-problems.

The agent stitching panel is accurate at ~30-40 candidate edges (its export
envelope, ``settings.stitch_export_backstop_max_edges``) and structurally
incapable above it: option generation for a monster group produces near-clone
mega-options whose correct answer is not on the menu, unanimity degrades into
shared-heuristic convergence, prompts overflow provider context windows — and
the export backstop means no monster verdict can mint a label anyway. This
module converts one unanswerable question into many answerable ones.

**Splitting algorithm.** The group's candidate edges form a bipartite graph
(ref nodes x target nodes). Biconnected components of that graph partition its
EDGES into blocks: bridge edges are singleton blocks, and tangled clusters are
larger blocks joined at articulation (cut) vertices. Blocks are then
agglomerated along block-cut-tree adjacency (two blocks are adjacent iff they
share an articulation vertex) into connected sub-problems of at most
``max_edges`` edges, via deterministic first-fit union-find passes over the
canonically ordered block-adjacency pairs. A single biconnected block larger
than ``max_edges`` is irreducible: it stays its own sub-problem, flagged
``oversized`` (no evidence pack is generated for it; it routes to human review
exactly as monster groups do today).

**Edge ownership.** Biconnected components partition edges, so every candidate
edge belongs to EXACTLY ONE sub-problem — no edge is ever voted twice, and the
union of sub-problem edge sets equals the group's edge set. An articulation
ref/target may appear in several sub-problems, but only as a shared endpoint;
since votes select EDGES, the recomposed union is well-defined.

**Sub-problem ids.** ``{parent_group_id}__p{sha256(sorted edges)[:10]}`` — a
pure content hash of the sub-problem's sorted ``(ref_id, target_id)`` pairs.
No randomness, no timestamps: the same group decomposed with the same
``max_edges`` yields byte-identical ids across runs and machines, so votes and
labels stay reproducible and re-votes of an unchanged sub-problem supersede
cleanly by id.

**Recomposition (conservative).** A whole-group label may be minted ONLY when
every sub-problem in the roster resolved as a unanimous panel accept; the label
is the union of the sub-selections. Any failed sub-problem (non-unanimous,
unanimous-NONE, cross-mode demotion, ...) or any unvoted/oversized sub-problem
blocks the group label entirely (see :func:`recompose_subproblem_verdicts`).
Mixed human+panel sub-verdict recomposition is deliberately deferred: a human
sub-verdict cannot yet complete a partially-panel-resolved group.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from ..config import settings

#: Separator between a parent group id and the sub-problem content hash.
#: Sidecar group ids are short hex strings, so ``__p`` cannot collide.
SUBPROBLEM_SEPARATOR = "__p"

#: Recomposition statuses (see :func:`recompose_subproblem_verdicts`).
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
STATUS_UNVOTED = "unvoted"

#: Route-reason vocabulary for an oversized (irreducible) sub-problem: it is
#: size-gated out of the panel flow, aligned with the panel-routing code used
#: for groups too large to auto-accept.
REASON_SIZE_GATED = "size_gated"

Pair = tuple[str, str]


@dataclass(frozen=True)
class SubProblem:
    """One panel-sized (or irreducible oversized) unit of a decomposed group."""

    subproblem_id: str
    parent_group_id: str
    edges: tuple[Pair, ...]  # sorted, deduplicated (ref_id, target_id) pairs
    ref_ids: tuple[str, ...]  # sorted distinct ref endpoints
    target_ids: tuple[str, ...]  # sorted distinct target endpoints
    n_blocks: int  # biconnected blocks merged into this sub-problem
    oversized: bool  # True for an irreducible block > max_edges

    @property
    def n_edges(self) -> int:
        return len(self.edges)


@dataclass(frozen=True)
class Decomposition:
    """Result of decomposing one group's candidate-edge graph."""

    parent_group_id: str
    max_edges: int
    n_edges: int  # distinct candidate edges in the parent group
    #: True when the group actually splits into MORE THAN ONE sub-problem.
    #: False both for under-backstop groups (no-op) and for irreducible
    #: monsters (a single biconnected blob that cannot split) — in either case
    #: the caller keeps the original group in the existing flow.
    is_decomposed: bool
    subproblems: tuple[SubProblem, ...]

    @property
    def votable_subproblems(self) -> tuple[SubProblem, ...]:
        return tuple(s for s in self.subproblems if not s.oversized)

    @property
    def oversized_subproblems(self) -> tuple[SubProblem, ...]:
        return tuple(s for s in self.subproblems if s.oversized)


def subproblem_id(parent_group_id: str, edges: Iterable[Pair]) -> str:
    """Deterministic content-hash id for a sub-problem of ``parent_group_id``.

    The id is ``{parent}__p{sha256[:10]}`` over the sorted, deduplicated
    ``(ref_id, target_id)`` pairs — stable across runs, machines, and input
    edge order. Two sub-problems of the same parent share an id iff they have
    identical edge sets (in which case they ARE the same sub-problem).
    """
    canon = sorted({(str(r), str(t)) for r, t in edges})
    digest = hashlib.sha256("\n".join(f"{r}\t{t}" for r, t in canon).encode()).hexdigest()
    return f"{parent_group_id}{SUBPROBLEM_SEPARATOR}{digest[:10]}"


def _canonical_pairs(edges: Iterable) -> list[Pair]:
    """Normalize edges (dicts or pairs) to sorted, deduplicated string pairs."""
    pairs: set[Pair] = set()
    for e in edges:
        if isinstance(e, Mapping):
            pairs.add((str(e["ref_id"]), str(e["target_id"])))
        else:
            r, t = e
            pairs.add((str(r), str(t)))
    return sorted(pairs)


class _UnionFind:
    """Minimal deterministic union-find (representative = smallest index)."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:  # path compression
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, i: int, j: int) -> int:
        ri, rj = self.find(i), self.find(j)
        lo, hi = (ri, rj) if ri < rj else (rj, ri)
        self.parent[hi] = lo
        return lo


def _biconnected_blocks(pairs: list[Pair]) -> list[list[Pair]]:
    """Partition candidate edges into biconnected blocks, canonically ordered.

    Nodes are namespaced (``("R", ref_id)`` / ``("T", target_id)``) so a ref and
    a target sharing a raw id never collapse into one node. The graph is built
    from the sorted pair list, and every returned block is itself sorted with
    blocks ordered by their smallest edge — so the output is independent of
    input edge order and of hash seeds.
    """
    import networkx as nx

    g = nx.Graph()
    for r, t in pairs:  # already sorted -> deterministic insertion order
        g.add_edge(("R", r), ("T", t))

    blocks: list[list[Pair]] = []
    for comp in nx.biconnected_component_edges(g):
        block = []
        for a, b in comp:
            rn, tn = (a, b) if a[0] == "R" else (b, a)
            block.append((rn[1], tn[1]))
        blocks.append(sorted(block))
    # Edges partition across blocks, so each block's first edge is unique ->
    # sorting by it gives a canonical total order.
    blocks.sort(key=lambda b: b[0])
    return blocks


def _merge_blocks(blocks: list[list[Pair]], max_edges: int) -> list[tuple[list[Pair], int]]:
    """Agglomerate adjacent blocks into connected clusters of <= ``max_edges``.

    Two blocks are adjacent iff they share a vertex (necessarily an articulation
    vertex — distinct biconnected components share at most one). First-fit
    union-find passes over the canonically sorted adjacency pairs run to a
    fixpoint; each merge requires the combined cluster to stay within
    ``max_edges``, so cluster connectivity and the size bound hold inductively.
    Oversized irreducible blocks (> ``max_edges`` on their own) are never merged
    with anything (no pair including them can satisfy the bound).

    Returns ``[(sorted_edges, n_blocks), ...]`` canonically ordered.
    """
    uf = _UnionFind(len(blocks))
    size = [len(b) for b in blocks]
    n_blocks = [1] * len(blocks)

    # Block adjacency via shared vertices, iterated deterministically.
    node_blocks: dict[tuple[str, str], list[int]] = {}
    for i, block in enumerate(blocks):
        seen: set[tuple[str, str]] = set()
        for r, t in block:
            for node in (("R", r), ("T", t)):
                if node not in seen:
                    seen.add(node)
                    node_blocks.setdefault(node, []).append(i)
    adj_pairs: set[tuple[int, int]] = set()
    for node in sorted(node_blocks):
        members = node_blocks[node]
        for a_pos in range(len(members)):
            for b_pos in range(a_pos + 1, len(members)):
                adj_pairs.add((members[a_pos], members[b_pos]))
    ordered_pairs = sorted(adj_pairs)

    changed = True
    while changed:
        changed = False
        for i, j in ordered_pairs:
            ri, rj = uf.find(i), uf.find(j)
            if ri == rj:
                continue
            if size[ri] + size[rj] > max_edges:
                continue
            merged_size = size[ri] + size[rj]
            merged_blocks = n_blocks[ri] + n_blocks[rj]
            root = uf.union(ri, rj)
            size[root] = merged_size
            n_blocks[root] = merged_blocks
            changed = True

    clusters: dict[int, list[Pair]] = {}
    for i, block in enumerate(blocks):
        clusters.setdefault(uf.find(i), []).extend(block)
    out = [(sorted(edges), n_blocks[root]) for root, edges in sorted(clusters.items())]
    out.sort(key=lambda c: c[0][0])
    return out


def decompose_candidate_edges(
    parent_group_id: str,
    edges: Iterable,
    max_edges: int | None = None,
) -> Decomposition:
    """Decompose a group's candidate edges into panel-sized sub-problems.

    Pure and deterministic: output depends only on the deduplicated edge SET
    and ``max_edges`` (defaults to ``settings.stitch_export_backstop_max_edges``,
    the panel's export envelope). ``edges`` may be edge dicts (with
    ``ref_id``/``target_id``) or raw pairs.

    * A group with at most ``max_edges`` distinct edges is a no-op
      (``is_decomposed=False``, no sub-problems).
    * A group whose graph is one irreducible biconnected blob (or otherwise
      yields a single sub-problem) is also NOT decomposed — splitting achieved
      nothing, so the caller keeps the original group in the existing
      (human-routed) flow.
    * Otherwise every sub-problem is a connected subgraph of at most
      ``max_edges`` edges, except irreducible blocks, which are flagged
      ``oversized`` and stay human-routed.
    """
    if max_edges is None:
        max_edges = settings.stitch_export_backstop_max_edges
    pairs = _canonical_pairs(edges)

    if len(pairs) <= max_edges:
        return Decomposition(
            parent_group_id=str(parent_group_id),
            max_edges=max_edges,
            n_edges=len(pairs),
            is_decomposed=False,
            subproblems=(),
        )

    blocks = _biconnected_blocks(pairs)
    clusters = _merge_blocks(blocks, max_edges)

    subs = []
    for cluster_edges, cluster_blocks in clusters:
        subs.append(
            SubProblem(
                subproblem_id=subproblem_id(str(parent_group_id), cluster_edges),
                parent_group_id=str(parent_group_id),
                edges=tuple(cluster_edges),
                ref_ids=tuple(sorted({r for r, _ in cluster_edges})),
                target_ids=tuple(sorted({t for _, t in cluster_edges})),
                n_blocks=cluster_blocks,
                oversized=len(cluster_edges) > max_edges,
            )
        )

    is_decomposed = len(subs) > 1
    return Decomposition(
        parent_group_id=str(parent_group_id),
        max_edges=max_edges,
        n_edges=len(pairs),
        is_decomposed=is_decomposed,
        subproblems=tuple(subs) if is_decomposed else (),
    )


def decompose_group(group: Mapping, max_edges: int | None = None) -> Decomposition:
    """Decompose a sidecar/batch group dict (uses its ``edges`` list)."""
    return decompose_candidate_edges(
        str(group.get("group_id", "")), group.get("edges", []) or [], max_edges
    )


#: Keys copied verbatim from a parent group when building a sub-problem group.
_SUBGROUP_ID_MAPS = (
    "ref_geometries",
    "target_geometries",
    "ref_names",
    "target_names",
    "ref_classes",
    "target_classes",
)


def _sub_match_type(n_refs: int, n_targets: int) -> str:
    if n_refs == 1 and n_targets == 1:
        return "1:1"
    if n_refs == 1:
        return "1:N"
    if n_targets == 1:
        return "N:1"
    return "M:N"


def build_subproblem_group(parent: Mapping, sub: SubProblem, n_subproblems: int) -> dict:
    """Derive a self-contained batch group dict for one sub-problem.

    The result flows through the EXISTING machinery unchanged — alternatives
    generation, spatial-context fill, evidence-pack rendering, panel voting —
    exactly like a normal group, with ``group_id`` set to the sub-problem id
    and ``parent_group_id`` recording provenance. Edges are the parent's edge
    dicts filtered to the sub-problem (so confidences, alignment fracs, and
    the #267 per-edge structural fields — computed on the FULL parent graph,
    the more truthful context — carry over verbatim); geometries, names, and
    classes are filtered to the sub-problem's endpoints. Sibling segments are
    NOT copied: the standard spatial-context fill re-adds anything nearby as
    context from the raw datasets, at the sub-problem's own (small) envelope.
    """
    edge_set = set(sub.edges)
    sub_edges = [
        e
        for e in parent.get("edges", []) or []
        if (str(e.get("ref_id")), str(e.get("target_id"))) in edge_set
    ]
    sub_assignment = [
        e
        for e in parent.get("optimizer_assignment", []) or []
        if (str(e.get("ref_id")), str(e.get("target_id"))) in edge_set
    ]
    group: dict = {
        "group_id": sub.subproblem_id,
        "parent_group_id": sub.parent_group_id,
        "n_subproblems": n_subproblems,
        "match_type": _sub_match_type(len(sub.ref_ids), len(sub.target_ids)),
        "ref_ids": list(sub.ref_ids),
        "target_ids": list(sub.target_ids),
        "edges": sub_edges,
        "optimizer_assignment": sub_assignment,
        "n_edges": len(sub.edges),
    }
    if sub.oversized:
        group["subproblem_oversized"] = True
    ref_keep, tgt_keep = set(sub.ref_ids), set(sub.target_ids)
    for key in _SUBGROUP_ID_MAPS:
        src = parent.get(key) or {}
        keep = ref_keep if key.startswith("ref_") else tgt_keep
        group[key] = {str(k): v for k, v in src.items() if str(k) in keep}
    return group


@dataclass
class Recomposition:
    """Outcome of recombining a decomposed group's sub-problem verdicts."""

    parent_group_id: str
    status: str  # STATUS_COMPLETE | STATUS_FAILED | STATUS_UNVOTED
    #: Union of the accepted sub-selections (sorted). Complete for
    #: STATUS_COMPLETE; partial (resolved subset only) otherwise — informational.
    union_edges: list[Pair]
    n_subproblems: int
    n_resolved: int
    failed_subproblems: list[str] = field(default_factory=list)
    unvoted_subproblems: list[str] = field(default_factory=list)


def recompose_subproblem_verdicts(
    parent_group_id: str,
    roster: Iterable[str],
    verdicts: Mapping[str, tuple[str, Iterable[Pair]]],
) -> Recomposition:
    """Conservatively recombine sub-problem verdicts into a group outcome.

    Args:
        parent_group_id: The decomposed group's id.
        roster: EVERY sub-problem id of the decomposition (including oversized
            sub-problems that were never packed/voted). The roster is the
            completeness contract: a verdict for an id outside the roster is
            ignored, and a roster id without a verdict blocks the group.
        verdicts: ``{subproblem_id: (routing, chosen_edge_pairs)}`` — routing is
            the consensus row's value (``auto_accept`` / ``human_review``).

    A whole-group label may be minted ONLY on ``STATUS_COMPLETE``: every roster
    sub-problem has a verdict and every verdict is ``auto_accept``. Any
    non-accept verdict (dissent, NONE, cross-mode demotion, ...) yields
    ``STATUS_FAILED``; a missing verdict (unvoted, or an oversized irreducible
    block) yields ``STATUS_UNVOTED``. Failure trumps unvoted in the status (both
    lists are reported either way). ``union_edges`` deduplicates, though
    sub-problems partition edges so accepted selections can never overlap.
    """
    roster_ids = sorted({str(s) for s in roster})
    failed: list[str] = []
    unvoted: list[str] = []
    union: set[Pair] = set()
    n_resolved = 0
    for sid in roster_ids:
        if sid not in verdicts:
            unvoted.append(sid)
            continue
        routing, pairs = verdicts[sid]
        if str(routing) != "auto_accept":
            failed.append(sid)
            continue
        n_resolved += 1
        union.update((str(r), str(t)) for r, t in pairs)

    if failed:
        status = STATUS_FAILED
    elif unvoted:
        status = STATUS_UNVOTED
    else:
        status = STATUS_COMPLETE
    return Recomposition(
        parent_group_id=str(parent_group_id),
        status=status,
        union_edges=sorted(union),
        n_subproblems=len(roster_ids),
        n_resolved=n_resolved,
        failed_subproblems=failed,
        unvoted_subproblems=unvoted,
    )
