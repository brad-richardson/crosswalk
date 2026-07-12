"""Global match optimization with M:N group detection.

Resolves conflicts using a bipartite connected-components approach
that handles 1:1, 1:N, N:1, and M:N match groups in one pass.

Algorithm:
1. Filter results above min_confidence
2. Build bipartite graph: nodes = (ref_ids ∪ target_ids), edges = match pairs
3. Find connected components via BFS → list of component edge-lists
4. For each component, classify as 1:1, 1:N, N:1, or M:N based on
   contiguity of the multi-side
5. Run greedy 1:1 optimization on unclaimed leftovers
6. Return group_results + optimized_1to1
"""

import hashlib
from collections import defaultdict, deque
from typing import Any

import geopandas as gpd
import numpy as np
from loguru import logger
from scipy.spatial import cKDTree
from shapely import LineString

from ..config import (
    DEFAULT_SNAP_TOLERANCE_M,
    MAX_ALIGNMENT_OVERLAP_M,
    NAMES_COLUMN,
    OPTIMIZER_ALIGNMENT_RESCUE_MAX_ENDPOINT_GAP_M,
    OPTIMIZER_ALIGNMENT_RESCUE_MAX_GAP_M,
    settings,
    sliver_overlap_m,
)
from ..features.semantic import _extract_all_name_variants
from .sliver import sliver_edges_for_match_results
from .types import MatchDecision, MatchResult, MatchType

LOW_CONFIDENCE_ADDITION_REVIEW_FLAG = "low_confidence_addition_review"
PRUNED_SINGLETON_REVIEW_FLAG = "pruned_singleton_review"
COVERAGE_CONFLICT_REVIEW_FLAG = "coverage_conflict"
PARALLEL_SIBLING_REVIEW_FLAG = "parallel_sibling_stub_review"
DECOMPOSED_SINGLETON_REVIEW_FLAG = "decomposed_singleton_review"


def optimizer_review_reason(result: MatchResult) -> str | None:
    """Return the stable sidecar reason for an optimizer REVIEW decision."""
    features = result.features or {}
    for flag, reason in (
        (PARALLEL_SIBLING_REVIEW_FLAG, "parallel_sibling"),
        (LOW_CONFIDENCE_ADDITION_REVIEW_FLAG, "low_confidence_addition"),
        (PRUNED_SINGLETON_REVIEW_FLAG, "pruned_singleton"),
        (COVERAGE_CONFLICT_REVIEW_FLAG, "coverage_conflict"),
        (DECOMPOSED_SINGLETON_REVIEW_FLAG, "decomposed_singleton"),
    ):
        if features.get(flag):
            return reason
    if result.decision == MatchDecision.REVIEW:
        return "group_confidence"
    return None


def _force_review(result: MatchResult, flag: str) -> MatchResult:
    """Clone one result as REVIEW while preserving every scored field."""
    return MatchResult(
        ref_id=result.ref_id,
        target_id=result.target_id,
        decision=MatchDecision.REVIEW,
        confidence=result.confidence,
        score_breakdown=result.score_breakdown,
        features={**result.features, flag: 1.0},
        ref_idx=result.ref_idx,
        target_idx=result.target_idx,
        gers_start_frac=result.gers_start_frac,
        gers_end_frac=result.gers_end_frac,
        local_start_frac=result.local_start_frac,
        local_end_frac=result.local_end_frac,
    )


def _geom_lookup(gdf: gpd.GeoDataFrame, id_column: str) -> dict[Any, LineString]:
    """Build an id -> geometry lookup, falling back to the index if needed."""
    if id_column in gdf.columns:
        return dict(zip(gdf[id_column], gdf.geometry))
    return dict(zip(gdf.index, gdf.geometry))


def _name_lookup(gdf: gpd.GeoDataFrame, id_column: str) -> dict[Any, Any]:
    """Build an id -> name lookup from ``NAMES_COLUMN`` (empty if absent)."""
    if NAMES_COLUMN not in gdf.columns:
        return {}
    keys = gdf[id_column] if id_column in gdf.columns else gdf.index
    return dict(zip(keys, gdf[NAMES_COLUMN]))


def compute_sliver_candidate_edges(
    results: list[MatchResult],
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    ref_id_column: str = "id",
    target_id_column: str = "local_id",
) -> set[tuple[Any, Any]]:
    """Classify candidate edges as junction slivers (shared hybrid rule).

    Thin wrapper over :func:`matching.sliver.sliver_edges_for_match_results`
    that builds the geometry lookups and detects whether the GeoDataFrames are
    in a metric (projected) CRS. Used by both the grouping optimizer and the
    groups-sidecar export so they agree on the exact same sliver set.
    """
    ref_geoms = _geom_lookup(reference, ref_id_column)
    target_geoms = _geom_lookup(target, target_id_column)
    metric = not (reference.crs is not None and reference.crs.is_geographic)
    return sliver_edges_for_match_results(results, ref_geoms, target_geoms, metric=metric)


def compute_group_id(ref_ids: set, target_ids: set) -> str:
    """Compute a deterministic short ID for a match group.

    Hashes sorted ref_ids + target_ids into an 8-character hex string.
    The same set of IDs always produces the same group_id.

    Args:
        ref_ids: Set of reference segment IDs in the group
        target_ids: Set of target segment IDs in the group

    Returns:
        8-character hex string uniquely identifying this group
    """
    key = (
        "|".join(sorted(str(r) for r in ref_ids))
        + "||"
        + "|".join(sorted(str(t) for t in target_ids))
    )
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def _fraction_range(start: Any, end: Any) -> tuple[float, float] | None:
    """Return a finite, ordered fraction range or ``None``."""
    if start is None or end is None:
        return None
    try:
        start_float = float(start)
        end_float = float(end)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(start_float) or not np.isfinite(end_float):
        return None
    return tuple(sorted((start_float, end_float)))


def _balanced_alignment_span(result: MatchResult) -> float:
    """Return conservative two-sided fractional coverage for stable ranking."""
    spans = [
        span[1] - span[0]
        for span in (
            _fraction_range(result.gers_start_frac, result.gers_end_frac),
            _fraction_range(result.local_start_frac, result.local_end_frac),
        )
        if span is not None
    ]
    return min(spans) if spans else 0.0


def _alignment_coverage_rank(result: MatchResult) -> tuple[int, float]:
    """Prefer measured aligned meters, with fractions as a legacy fallback."""
    aligned_length = (result.features or {}).get("aligned_length_m")
    try:
        if aligned_length is not None and np.isfinite(aligned_length):
            return 1, float(aligned_length)
    except TypeError:
        pass
    return 0, _balanced_alignment_span(result)


def _match_result_rank(result: MatchResult, *, name_agrees: bool = False) -> tuple:
    """Canonical best-first rank for selection and logical-pair deduplication."""

    def _stable_number(value: Any) -> tuple[int, float]:
        try:
            if value is not None and np.isfinite(value):
                return 0, float(value)
        except TypeError:
            pass
        return 1, 0.0

    confidence = float(result.confidence) if np.isfinite(result.confidence) else float("-inf")
    coverage_kind, coverage_value = _alignment_coverage_rank(result)
    return (
        -confidence,
        -int(name_agrees),
        -coverage_kind,
        -coverage_value,
        str(result.ref_id),
        str(result.target_id),
        result.ref_idx if result.ref_idx is not None else float("inf"),
        result.target_idx if result.target_idx is not None else float("inf"),
        _stable_number(result.gers_start_frac),
        _stable_number(result.gers_end_frac),
        _stable_number(result.local_start_frac),
        _stable_number(result.local_end_frac),
    )


def optimize_matches_greedy(
    results: list[MatchResult],
    min_confidence: float = 0.5,
    ref_name_lookup: dict[Any, Any] | None = None,
    target_name_lookup: dict[Any, Any] | None = None,
) -> list[MatchResult]:
    """Greedy 1:1 assignment for large datasets.

    Sorts candidates by confidence and greedily assigns matches,
    ensuring each ref and target is matched at most once.

    Time complexity: O(n log n) for sorting
    Space complexity: O(n) where n = number of candidates

    Args:
        results: List of MatchResult objects
        min_confidence: Minimum confidence to consider a match
        ref_name_lookup: Optional reference names used only to break exact
            confidence ties in grouped-optimizer fallbacks.
        target_name_lookup: Optional target names used only to break exact
            confidence ties in grouped-optimizer fallbacks.

    Returns:
        List of greedily-selected MatchResult objects (1:1 assignments)
    """
    import time

    t0 = time.perf_counter()
    logger.info(f"Optimizing {len(results)} match results with greedy algorithm...")

    # Filter by minimum confidence
    valid_results = [r for r in results if r.confidence >= min_confidence]
    logger.info(f"  {len(valid_results)} results above min_confidence={min_confidence}")

    if not valid_results:
        return []

    def _same_name(result: MatchResult) -> bool:
        if not ref_name_lookup or not target_name_lookup:
            return False
        ref_names = _normalized_names_for_range(
            ref_name_lookup.get(result.ref_id),
            _fraction_range(result.gers_start_frac, result.gers_end_frac),
        )
        target_names = _normalized_names_for_range(
            target_name_lookup.get(result.target_id),
            _fraction_range(result.local_start_frac, result.local_end_frac),
        )
        return bool(ref_names & target_names)

    # Confidence remains the primary ordering. Exact ties use non-empty raw-name
    # agreement and then aligned coverage before stable IDs. This is deliberately
    # not a score adjustment: even a tiny genuine confidence lead still wins,
    # while saturated equal scores no longer resolve by UUID alone.
    sorted_results = sorted(
        valid_results,
        key=lambda result: _match_result_rank(result, name_agrees=_same_name(result)),
    )

    assigned_refs: set = set()
    assigned_targets: set = set()
    optimal_matches = []

    for result in sorted_results:
        if result.ref_id not in assigned_refs and result.target_id not in assigned_targets:
            optimal_matches.append(result)
            assigned_refs.add(result.ref_id)
            assigned_targets.add(result.target_id)

    logger.info(
        f"  Found {len(optimal_matches)} greedy 1:1 matches in {time.perf_counter() - t0:.2f}s"
    )

    return optimal_matches


def find_match_components(
    results: list[MatchResult],
    min_confidence: float,
    sliver_edges: set[tuple[Any, Any]] | None = None,
    glue_min_confidence: float | None = None,
) -> list[list[MatchResult]]:
    """Find connected components in the bipartite ref-target match graph.

    Builds an undirected bipartite graph with namespaced nodes
    ("ref", id) / ("target", id) and edges from match pairs.
    Returns one list of MatchResult per connected component.

    Args:
        results: All match results
        min_confidence: Minimum confidence threshold
        sliver_edges: Optional set of ``(ref_id, target_id)`` pairs classified
            as junction slivers (see ``matching.sliver``). Sliver edges
            contribute NO adjacency when building components, so a junction
            kiss can never weld two otherwise-independent groups into one
            monster component. A sliver edge is attached to a component's edge
            list only when BOTH its endpoints already landed in that same
            component via non-sliver edges (it remains an ordinary group
            member there). A sliver whose endpoints end up in different
            components — or whose endpoints have no non-sliver edge at all —
            is dropped from grouping entirely: the only evidence tying those
            segments together is a junction artifact.
        glue_min_confidence: Optional grouping-only confidence prune. Candidate
            edges with ``min_confidence <= confidence < glue_min_confidence``
            are treated like slivers for adjacency (they never weld components
            together) but remain in a component's edge list when their
            endpoints co-land via stronger edges. Defaults to ``min_confidence``
            (no prune).

    Returns:
        List of components, each a list of MatchResult edges
    """
    valid = [r for r in results if r.confidence >= min_confidence]
    if not valid:
        return []

    sliver_edges = sliver_edges or set()
    glue_min = min_confidence if glue_min_confidence is None else glue_min_confidence

    # Split structural (component-building) edges from non-gluing edges. A
    # non-gluing edge is a sliver OR a weak edge below ``glue_min`` (grouping-
    # only confidence prune): neither builds adjacency, both only attach to a
    # component when their endpoints already co-land there via structural
    # edges. All are deduplicated to the highest-confidence result per pair.
    structural: list[MatchResult] = []
    sliver_best: dict[tuple[Any, Any], MatchResult] = {}
    for r in valid:
        pair = (r.ref_id, r.target_id)
        if pair in sliver_edges or r.confidence < glue_min:
            if pair not in sliver_best or _match_result_rank(r) < _match_result_rank(
                sliver_best[pair]
            ):
                sliver_best[pair] = r
        else:
            structural.append(r)

    # Build adjacency list with namespaced nodes (structural edges only)
    adj: dict[tuple[str, Any], set[tuple[str, Any]]] = defaultdict(set)
    # Track which edges (MatchResults) connect each node pair
    edge_lookup: dict[tuple[tuple[str, Any], tuple[str, Any]], MatchResult] = {}

    for r in structural:
        ref_node = ("ref", r.ref_id)
        tgt_node = ("target", r.target_id)
        adj[ref_node].add(tgt_node)
        adj[tgt_node].add(ref_node)
        # Keep the canonical best row if duplicate logical IDs occur.
        key = (ref_node, tgt_node)
        if key not in edge_lookup or _match_result_rank(r) < _match_result_rank(edge_lookup[key]):
            edge_lookup[key] = r

    # BFS to find connected components
    visited: set[tuple[str, Any]] = set()
    components: list[list[MatchResult]] = []
    node_component: dict[tuple[str, Any], int] = {}

    # Canonicalize component discovery itself, not only the edge order inside a
    # component. Otherwise reversing two independent group-producing components
    # reverses their final sidecar order even though every selected pair is the
    # same.
    for start_node in sorted(adj, key=lambda node: (node[0], str(node[1]))):
        if start_node in visited:
            continue

        # BFS from start_node
        component_nodes: set[tuple[str, Any]] = set()
        queue = deque([start_node])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component_nodes.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

        # Collect edges for this component. Sort node/neighbor iteration by id so
        # the emitted edge order is independent of set iteration order (which is
        # hash-seed dependent for str ids). The component's edge LIST order flows
        # downstream into grouping and greedy tie-breaks, so it must be canonical.
        component_edges: list[MatchResult] = []
        ref_nodes = sorted((n for n in component_nodes if n[0] == "ref"), key=lambda n: str(n[1]))
        tgt_node_set = {n for n in component_nodes if n[0] == "target"}

        for rn in ref_nodes:
            for tn in sorted(adj[rn], key=lambda n: str(n[1])):
                if tn in tgt_node_set:
                    key = (rn, tn)
                    if key in edge_lookup:
                        component_edges.append(edge_lookup[key])

        if component_edges:
            for n in component_nodes:
                node_component[n] = len(components)
            components.append(component_edges)

    # Attach sliver edges whose endpoints both landed in the SAME component via
    # structural edges. Cross-component (or unattached) slivers are dropped.
    n_attached = 0
    n_dropped = 0
    for (rid, tid), r in sorted(
        sliver_best.items(),
        key=lambda item: (str(item[0][0]), str(item[0][1])),
    ):
        ci = node_component.get(("ref", rid))
        if ci is not None and ci == node_component.get(("target", tid)):
            components[ci].append(r)
            n_attached += 1
        else:
            n_dropped += 1
    if sliver_best:
        logger.debug(
            f"  Sliver edges: {n_attached} kept as in-component candidates, "
            f"{n_dropped} dropped (would have glued independent components)"
        )

    return components


def _normalize_name(name: Any) -> str:
    """Normalize a segment name to a lowercased comparison key.

    Accepts plain strings and Overture-style name dicts (``{"primary": ...}``).
    Returns ``""`` for missing / unusable names so callers can treat empty as
    "no name evidence" rather than a match.
    """
    if name is None:
        return ""
    if isinstance(name, dict):
        for key in ("primary", "common", "name", "value"):
            v = name.get(key)
            if isinstance(v, str) and v:
                name = v
                break
        else:
            name = next((v for v in name.values() if isinstance(v, str) and v), "")
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def _normalized_names(name: Any) -> set[str]:
    """Return every usable top-level or linear-referenced name in a value."""
    if name is None:
        return set()
    if isinstance(name, str):
        normalized = name.strip().lower()
        return {normalized} if normalized else set()
    if isinstance(name, dict):
        # Keep name parsing consistent with semantic feature computation. In
        # particular, ``common`` is either ``{language: name}`` in target data or
        # an Overture ndarray of ``[language, name]`` pairs. Recursing through the
        # latter admits language codes ("en", "zh", ...) as street names, while
        # ordinary dict-key traversal silently loses the former.
        values = {
            value.strip().lower()
            for value in _extract_all_name_variants(name)
            if isinstance(value, str) and value.strip()
        }
        # Retain compatibility with the small generic name/value dicts accepted
        # by the optimizer even though they are not full Overture name structs.
        for key in ("name", "value"):
            values.update(_normalized_names(name.get(key)))
        return values
    if isinstance(name, (list, tuple, set, np.ndarray)):
        values: set[str] = set()
        for value in name:
            values.update(_normalized_names(value))
        return values
    return set()


def _normalized_names_for_range(
    name: Any,
    span: tuple[float, float] | None,
) -> set[str]:
    """Return names whose linear-reference rules overlap ``span``.

    Overture can store a single geometry whose street name changes partway
    along it. Using only ``names.primary`` misclassifies the other valid span as
    a different street (Bowman/Blackwell and Clifton/Batchelder in Boston).
    When usable rules exist, they therefore take precedence over the top-level
    primary name for a measured span.
    """
    if not isinstance(name, dict) or span is None:
        return _normalized_names(name)
    rules = name.get("rules")
    if not isinstance(rules, (list, tuple, np.ndarray)):
        return _normalized_names(name)

    lo, hi = span
    matched: set[str] = set()
    has_named_rule = False
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rule_names = _normalized_names(rule.get("value"))
        if not rule_names:
            continue
        has_named_rule = True
        between = rule.get("between")
        if between is None:
            matched.update(rule_names)
            continue
        try:
            rule_lo, rule_hi = sorted((float(between[0]), float(between[1])))
        except (IndexError, TypeError, ValueError):
            continue
        if min(hi, rule_hi) > max(lo, rule_lo):
            matched.update(rule_names)

    if has_named_rule:
        # A global common-name map describes translations/aliases of the
        # top-level primary name. It is safe evidence only on a range whose rule
        # includes that primary name. On another named range (for example the
        # Blackwell half of a Bowman/Blackwell geometry), admitting every global
        # alias would erase the range-specific protection this helper provides.
        primary_names = _normalized_names(name.get("primary"))
        if matched.intersection(primary_names):
            matched.update(
                value.strip().lower()
                for value in _extract_all_name_variants({"common": name.get("common")})
                if isinstance(value, str) and value.strip()
            )
        return matched
    return _normalized_names(name)


def _endpoints_are_collinear(
    ep_a: tuple[float, float],
    nb_a: tuple[float, float],
    ep_b: tuple[float, float],
    nb_b: tuple[float, float],
    max_turn_deg: float,
) -> bool:
    """Test whether two segments form a collinear continuation at a shared endpoint.

    ``ep_*`` is the shared (touching) endpoint of each segment and ``nb_*`` its
    adjacent interior vertex, so ``nb_* - ep_*`` is the direction the segment
    *leaves* the shared point. Two collinear continuations leave in nearly
    opposite directions (a straight through-path), so the deflection from
    straight is ``180 - angle(vec_a, vec_b)``. Returns True when that deflection
    is within ``max_turn_deg`` (a genuine corridor) rather than a sharp turn (a
    perpendicular junction kiss between two different streets).
    """
    va = (nb_a[0] - ep_a[0], nb_a[1] - ep_a[1])
    vb = (nb_b[0] - ep_b[0], nb_b[1] - ep_b[1])
    na = (va[0] ** 2 + va[1] ** 2) ** 0.5
    nb = (vb[0] ** 2 + vb[1] ** 2) ** 0.5
    if na == 0.0 or nb == 0.0:
        return False
    cos = (va[0] * vb[0] + va[1] * vb[1]) / (na * nb)
    cos = max(-1.0, min(1.0, cos))
    angle = np.degrees(np.arccos(cos))
    turn = 180.0 - angle
    return turn <= max_turn_deg


def build_contiguity_adjacency(
    ids: list[Any],
    geom_lookup: dict[Any, LineString],
    tolerance: float,
    require_collinear: bool = False,
    max_turn_deg: float = 40.0,
    name_lookup: dict[Any, Any] | None = None,
) -> dict[Any, set[Any]]:
    """Build an endpoint-proximity adjacency map over a set of IDs.

    Two geometries are adjacent if one's endpoint is within ``tolerance`` of the
    other's endpoint. Uses a KD-tree for O(n log n) proximity detection.

    This is the single shared primitive for endpoint-based contiguity: it backs
    both ``_find_contiguous_id_groups`` (connected components) here in the
    optimizer and the contiguous ref-chain enumeration in
    ``matching/alternatives.py`` — so assignment *options* can express the same
    multi-ref spans the optimizer itself can produce.

    Args:
        ids: List of IDs to check (duplicates collapse onto the same key).
        geom_lookup: Dictionary mapping ID to LineString geometry, in the same
            coordinate units as ``tolerance``.
        tolerance: Maximum endpoint distance to consider contiguous.
        require_collinear: When True, a proximity pair only becomes adjacent if
            the two segments are a collinear continuation at the shared endpoint
            (deflection <= ``max_turn_deg``) OR share a normalized name (the
            "same-street rescue" for gently curving named corridors). This is
            the corridor-aware gate used by the M:N branch so perpendicular
            junction kisses do not chain independent corridors together.
        max_turn_deg: Deflection threshold (degrees) for the collinearity gate.
        name_lookup: Optional ID -> name map (string or Overture name dict) for
            the same-name rescue. Ignored when ``require_collinear`` is False.

    Returns:
        Dict mapping each input ID to the set of IDs it is contiguous with.
        Every input ID is a key (empty set if it has no neighbour or has
        missing / degenerate geometry).
    """
    adjacency: dict[Any, set[Any]] = {id_: set() for id_ in ids}

    all_endpoints: list = []
    endpoint_to_id: list = []  # which id does each endpoint belong to
    endpoint_neighbor: list = []  # adjacent interior vertex (for direction)
    for id_ in ids:
        geom = geom_lookup.get(id_)
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        if len(coords) < 2:
            continue
        all_endpoints.append(coords[0][:2])
        endpoint_to_id.append(id_)
        endpoint_neighbor.append(coords[1][:2])
        all_endpoints.append(coords[-1][:2])
        endpoint_to_id.append(id_)
        endpoint_neighbor.append(coords[-2][:2])

    if len(all_endpoints) < 2:
        return adjacency

    name_cache: dict[Any, str] = {}

    def _name(id_: Any) -> str:
        if id_ not in name_cache:
            name_cache[id_] = _normalize_name(name_lookup.get(id_)) if name_lookup else ""
        return name_cache[id_]

    tree = cKDTree(np.array(all_endpoints))
    for ep_i, ep_j in tree.query_pairs(tolerance):
        a = endpoint_to_id[ep_i]
        b = endpoint_to_id[ep_j]
        if a == b:
            continue
        if require_collinear:
            collinear = _endpoints_are_collinear(
                all_endpoints[ep_i],
                endpoint_neighbor[ep_i],
                all_endpoints[ep_j],
                endpoint_neighbor[ep_j],
                max_turn_deg,
            )
            if not collinear:
                na, nb = _name(a), _name(b)
                if not (na and na == nb):
                    continue
        adjacency[a].add(b)
        adjacency[b].add(a)

    return adjacency


def _find_contiguous_id_groups(
    ids: list[Any],
    geom_lookup: dict[Any, LineString],
    tolerance: float,
    require_collinear: bool = False,
    max_turn_deg: float = 40.0,
    name_lookup: dict[Any, Any] | None = None,
) -> list[list[Any]]:
    """Find groups of IDs whose geometries are contiguous.

    Two geometries are contiguous if one's endpoint is within tolerance
    of the other's endpoint. Uses KD-tree for O(n log n) proximity detection.

    Args:
        ids: List of IDs to check
        geom_lookup: Dictionary mapping ID to LineString geometry
        tolerance: Maximum endpoint distance to consider contiguous (meters)
        require_collinear: Gate contiguity on collinear continuation or same
            name (see :func:`build_contiguity_adjacency`).
        max_turn_deg: Deflection threshold (degrees) for the collinearity gate.
        name_lookup: Optional ID -> name map for the same-name rescue.

    Returns:
        List of groups, each a list of contiguous IDs.
        Single-element groups are included for IDs with no contiguous neighbors.
    """
    if len(ids) <= 1:
        return [ids] if ids else []

    adjacency = build_contiguity_adjacency(
        ids,
        geom_lookup,
        tolerance,
        require_collinear=require_collinear,
        max_turn_deg=max_turn_deg,
        name_lookup=name_lookup,
    )

    # Find connected components using BFS.
    visited: set[Any] = set()
    groups: list[list[Any]] = []
    for id_ in ids:
        if id_ in visited:
            continue
        group: list[Any] = []
        queue = deque([id_])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            group.append(node)
            # Sort neighbours by stable id: adjacency values are sets, whose
            # iteration order is hash-seed dependent for str ids. The BFS append
            # order determines each group's member order, which flows into the
            # optimizer's sub-component edge order — so it must be canonical.
            for neighbor in sorted(adjacency[node], key=str):
                if neighbor not in visited:
                    queue.append(neighbor)
        groups.append(group)

    return groups


class _EndpointConnectorIndex:
    """Lazy endpoint index for verifying a real intervening segment."""

    def __init__(self, geom_lookup: dict[Any, LineString], tolerance: float):
        self.geom_lookup = geom_lookup
        self.tolerance = tolerance
        self._tree: cKDTree | None = None
        self._endpoint_ids: list[Any] = []

    def _ensure_tree(self) -> None:
        if self._tree is not None:
            return
        endpoints: list[tuple[float, float]] = []
        endpoint_ids: list[Any] = []
        for id_, geom in self.geom_lookup.items():
            if geom is None or geom.is_empty or len(geom.coords) < 2:
                continue
            endpoints.extend((geom.coords[0][:2], geom.coords[-1][:2]))
            endpoint_ids.extend((id_, id_))
        self._endpoint_ids = endpoint_ids
        self._tree = cKDTree(np.asarray(endpoints)) if endpoints else cKDTree(np.empty((0, 2)))

    def has_short_connector(
        self,
        id_a: Any,
        id_b: Any,
        endpoint_a: tuple[float, float],
        endpoint_b: tuple[float, float],
        max_length_m: float,
    ) -> bool:
        """Return whether one short third segment joins the two endpoints."""
        self._ensure_tree()
        assert self._tree is not None
        near_a = {
            self._endpoint_ids[index]
            for index in self._tree.query_ball_point(endpoint_a, self.tolerance)
        }
        near_b = {
            self._endpoint_ids[index]
            for index in self._tree.query_ball_point(endpoint_b, self.tolerance)
        }
        for connector_id in sorted(near_a.intersection(near_b) - {id_a, id_b}, key=str):
            connector = self.geom_lookup.get(connector_id)
            if (
                connector is None
                or connector.is_empty
                or len(connector.coords) < 2
                or connector.length > max_length_m
            ):
                continue
            start = connector.coords[0]
            end = connector.coords[-1]
            if (
                np.hypot(start[0] - endpoint_a[0], start[1] - endpoint_a[1]) <= self.tolerance
                and np.hypot(end[0] - endpoint_b[0], end[1] - endpoint_b[1]) <= self.tolerance
            ) or (
                np.hypot(end[0] - endpoint_a[0], end[1] - endpoint_a[1]) <= self.tolerance
                and np.hypot(start[0] - endpoint_b[0], start[1] - endpoint_b[1]) <= self.tolerance
            ):
                return True
        return False


def _merge_groups_by_alignment_rescue(
    groups: list[list[Any]],
    results: list[MatchResult],
    *,
    multi_id_attr: str,
    multi_name_lookup: dict[Any, Any] | None,
    multi_geom_lookup: dict[Any, LineString],
    shared_name: Any,
    frac_start_attr: str,
    frac_end_attr: str,
    multi_frac_start_attr: str,
    multi_frac_end_attr: str,
    shared_segment_length_m: float,
    max_gap_m: float = OPTIMIZER_ALIGNMENT_RESCUE_MAX_GAP_M,
    max_endpoint_gap_m: float = OPTIMIZER_ALIGNMENT_RESCUE_MAX_ENDPOINT_GAP_M,
    max_overlap_m: float = MAX_ALIGNMENT_OVERLAP_M,
    endpoint_tolerance_m: float = DEFAULT_SNAP_TOLERANCE_M,
    max_turn_deg: float = 40.0,
    connector_index: _EndpointConnectorIndex | None = None,
    min_rescue_confidence: float | None = None,
) -> tuple[list[list[Any]], set[Any]]:
    """Join physical groups when same-name alignment spans are complementary.

    Endpoint contiguity remains authoritative. This rescue only joins *different*
    physical groups when each multi-side segment agrees with the shared
    segment's name over its own alignment range, their shared-side spans overlap
    by no more than ``max_overlap_m``, and the uncovered gap is at most
    ``max_gap_m``. Range-aware name agreement matters when one shared geometry
    legitimately changes street name. Same-side endpoints must also be within
    ``max_endpoint_gap_m`` and both candidates must clear
    ``min_rescue_confidence``. Beyond normal endpoint tolerance, the two
    same-side segments must additionally share a name, form a collinear
    continuation, and have a real short third segment joining their endpoints.
    Potential joins are accepted by connected component only
    after the whole component passes the overlap/gap checks; this prevents a
    pairwise-valid chain from attaching an ambiguous duplicate span by stable-ID
    order. Missing names, geometries, lengths, or valid fractions fail closed.

    Returns the merged ID groups plus the IDs that participated in an
    alignment-only join.
    """
    if (
        len(groups) < 2
        or shared_segment_length_m <= 0
        or max_gap_m < 0
        or max_endpoint_gap_m < 0
        or endpoint_tolerance_m < 0
    ):
        return groups, set()
    if min_rescue_confidence is None:
        min_rescue_confidence = settings.optimizer_match_threshold

    if not _normalized_names(shared_name) or not multi_name_lookup:
        return groups, set()

    id_to_group: dict[Any, int] = {}
    for group_index, group in enumerate(groups):
        for id_ in group:
            id_to_group[id_] = group_index

    best_by_id: dict[Any, MatchResult] = {}
    for result in results:
        id_ = getattr(result, multi_id_attr)
        current = best_by_id.get(id_)
        if current is None or _match_result_rank(result) < _match_result_rank(current):
            best_by_id[id_] = result

    def _range(
        result: MatchResult,
        start_attr: str,
        end_attr: str,
    ) -> tuple[float, float] | None:
        start = getattr(result, start_attr)
        end = getattr(result, end_attr)
        if start is None or end is None:
            return None
        start = float(start)
        end = float(end)
        if not np.isfinite(start) or not np.isfinite(end):
            return None
        lo, hi = min(start, end), max(start, end)
        if lo < 0.0 or hi > 1.0 or hi <= lo:
            return None
        return lo, hi

    def _endpoint_context(
        id_a: Any,
        id_b: Any,
    ) -> (
        tuple[
            float,
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ]
        | None
    ):
        geom_a = multi_geom_lookup.get(id_a)
        geom_b = multi_geom_lookup.get(id_b)
        if (
            geom_a is None
            or geom_b is None
            or geom_a.is_empty
            or geom_b.is_empty
            or len(geom_a.coords) < 2
            or len(geom_b.coords) < 2
        ):
            return None
        coords_a = list(geom_a.coords)
        coords_b = list(geom_b.coords)
        endpoint_pairs = (
            (coords_a[0][:2], coords_a[1][:2], coords_b[0][:2], coords_b[1][:2]),
            (coords_a[0][:2], coords_a[1][:2], coords_b[-1][:2], coords_b[-2][:2]),
            (coords_a[-1][:2], coords_a[-2][:2], coords_b[0][:2], coords_b[1][:2]),
            (coords_a[-1][:2], coords_a[-2][:2], coords_b[-1][:2], coords_b[-2][:2]),
        )
        best = min(
            endpoint_pairs,
            key=lambda context: float(
                np.hypot(context[0][0] - context[2][0], context[0][1] - context[2][1])
            ),
        )
        return (
            float(np.hypot(best[0][0] - best[2][0], best[0][1] - best[2][1])),
            best[0],
            best[1],
            best[2],
            best[3],
        )

    eligible: dict[Any, tuple[MatchResult, tuple[float, float], set[str]]] = {}
    for id_, result in best_by_id.items():
        span = _range(result, frac_start_attr, frac_end_attr)
        multi_span = _range(result, multi_frac_start_attr, multi_frac_end_attr)
        names = _normalized_names_for_range(multi_name_lookup.get(id_), multi_span)
        shared_names = _normalized_names_for_range(shared_name, span)
        if (
            result.confidence >= min_rescue_confidence
            and span is not None
            and multi_span is not None
            and names.intersection(shared_names)
        ):
            eligible[id_] = (result, span, names)

    # Build all pairwise-plausible joins between the ORIGINAL physical groups.
    # We intentionally do not union as we iterate: doing so makes an ambiguous
    # duplicate attach to whichever group happens to sort first.
    group_adjacency: dict[int, set[int]] = {index: set() for index in range(len(groups))}
    connecting_ids: dict[tuple[int, int], set[Any]] = defaultdict(set)
    ids = sorted(eligible, key=str)
    for position, id_a in enumerate(ids):
        group_a = id_to_group.get(id_a)
        if group_a is None:
            continue
        _, range_a, names_a = eligible[id_a]
        for id_b in ids[position + 1 :]:
            group_b = id_to_group.get(id_b)
            if group_b is None or group_a == group_b:
                continue
            _, range_b, names_b = eligible[id_b]

            overlap_frac = max(0.0, min(range_a[1], range_b[1]) - max(range_a[0], range_b[0]))
            gap_frac = max(0.0, max(range_a[0], range_b[0]) - min(range_a[1], range_b[1]))
            if overlap_frac * shared_segment_length_m > max_overlap_m:
                continue
            if gap_frac * shared_segment_length_m > max_gap_m:
                continue

            endpoint_context = _endpoint_context(id_a, id_b)
            if endpoint_context is None or endpoint_context[0] > max_endpoint_gap_m:
                continue
            if endpoint_context[0] > endpoint_tolerance_m:
                if not names_a.intersection(names_b):
                    continue
                if not _endpoints_are_collinear(
                    endpoint_context[1],
                    endpoint_context[2],
                    endpoint_context[3],
                    endpoint_context[4],
                    max_turn_deg,
                ):
                    continue
                if connector_index is None or not connector_index.has_short_connector(
                    id_a,
                    id_b,
                    endpoint_context[1],
                    endpoint_context[3],
                    max_endpoint_gap_m + 2 * endpoint_tolerance_m,
                ):
                    continue

            group_adjacency[group_a].add(group_b)
            group_adjacency[group_b].add(group_a)
            group_pair = tuple(sorted((group_a, group_b)))
            connecting_ids[group_pair].update((id_a, id_b))

    potential_components: list[list[int]] = []
    visited: set[int] = set()
    for group_index in range(len(groups)):
        if group_index in visited or not group_adjacency[group_index]:
            continue
        component: list[int] = []
        queue = deque([group_index])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            queue.extend(sorted(group_adjacency[current] - visited))
        potential_components.append(sorted(component))

    accepted_components: list[list[int]] = []
    rescued_ids: set[Any] = set()
    for component in potential_components:
        component_set = set(component)
        component_records = [
            (id_, id_to_group[id_], eligible[id_][1])
            for id_ in ids
            if id_to_group.get(id_) in component_set
        ]

        # Any excessive overlap between different physical blocks makes the
        # whole attachment ambiguous. Reject the component instead of choosing
        # one duplicate span by iteration order.
        valid = True
        for position, (_, group_a, range_a) in enumerate(component_records):
            for _, group_b, range_b in component_records[position + 1 :]:
                if group_a == group_b:
                    continue
                overlap_frac = max(
                    0.0,
                    min(range_a[1], range_b[1]) - max(range_a[0], range_b[0]),
                )
                if overlap_frac * shared_segment_length_m > max_overlap_m:
                    valid = False
                    break
            if not valid:
                break

        # Validate the sorted interval union as a whole. This allows a genuine
        # A-B-C corridor while rejecting transitive joins with a hidden large
        # gap that no accepted pair should bridge.
        if valid:
            sorted_ranges = sorted(span for _, _, span in component_records)
            covered_hi = sorted_ranges[0][1]
            for lo, hi in sorted_ranges[1:]:
                if max(0.0, lo - covered_hi) * shared_segment_length_m > max_gap_m:
                    valid = False
                    break
                covered_hi = max(covered_hi, hi)

        if not valid:
            continue
        accepted_components.append(component)
        for group_pair, pair_ids in connecting_ids.items():
            if group_pair[0] in component_set and group_pair[1] in component_set:
                rescued_ids.update(pair_ids)

    if not accepted_components:
        return groups, set()

    root_by_group = {index: index for index in range(len(groups))}
    for component in accepted_components:
        root = component[0]
        for group_index in component:
            root_by_group[group_index] = root
    merged: dict[int, list[Any]] = {}
    for group_index, group in enumerate(groups):
        merged.setdefault(root_by_group[group_index], []).extend(group)
    return list(merged.values()), rescued_ids


def _create_group_results(
    results: list[MatchResult],
    match_type: MatchType,
    review_pairs: set[tuple[Any, Any]] | None = None,
    review_flag: str | None = None,
) -> list[MatchResult]:
    """Tag MatchResult objects with group metadata.

    Preserves original MatchResult objects (including alignment fractions,
    confidence, score_breakdown) and adds group metadata to features.

    Sets group decision: MATCH if avg_confidence >= optimizer_review_threshold,
    else REVIEW.

    Args:
        results: MatchResult objects belonging to this group
        match_type: MatchType value (ONE_TO_N, N_TO_ONE, M_TO_N)
        review_pairs: Optional set of ``(ref_id, target_id)`` pairs whose
            decision is forced to REVIEW regardless of the group decision (the
            #367 anti-crossing demote-to-REVIEW; see
            ``_contested_small_span_review_pairs``). These edges stay in the
            group; only their decision is demoted.
        review_flag: Feature flag attached to forced-review pairs. Defaults to
            ``PARALLEL_SIBLING_REVIEW_FLAG`` for backward compatibility.

    Returns:
        List of tagged MatchResult objects (same objects, mutated features)
    """
    if not results:
        return []

    review_pairs = review_pairs or set()
    if review_pairs and review_flag is None:
        review_flag = PARALLEL_SIBLING_REVIEW_FLAG

    avg_confidence = np.mean([r.confidence for r in results])
    group_decision = (
        MatchDecision.MATCH
        if avg_confidence >= settings.optimizer_review_threshold
        else MatchDecision.REVIEW
    )

    ref_ids = set(r.ref_id for r in results)
    target_ids = set(r.target_id for r in results)
    group_id = compute_group_id(ref_ids, target_ids)

    tagged: list[MatchResult] = []
    for r in results:
        demoted = (r.ref_id, r.target_id) in review_pairs
        decision = MatchDecision.REVIEW if demoted else group_decision
        features = {
            **r.features,
            "match_type": match_type,
            "group_id": group_id,
            "group_size": len(results),
            "group_ref_count": len(ref_ids),
            "group_target_count": len(target_ids),
        }
        if demoted and review_flag:
            features[review_flag] = 1.0
        # Create new MatchResult preserving all original fields
        tagged.append(
            MatchResult(
                ref_id=r.ref_id,
                target_id=r.target_id,
                decision=decision,
                confidence=r.confidence,
                score_breakdown=r.score_breakdown,
                features=features,
                ref_idx=r.ref_idx,
                target_idx=r.target_idx,
                gers_start_frac=r.gers_start_frac,
                gers_end_frac=r.gers_end_frac,
                local_start_frac=r.local_start_frac,
                local_end_frac=r.local_end_frac,
            )
        )

    return tagged


def _expand_greedy_matches(
    greedy_matches: list[MatchResult],
    all_candidates: list[MatchResult],
    ref_geoms: dict[Any, LineString],
    target_geoms: dict[Any, LineString],
    tolerance: float,
    min_confidence: float,
    glue_min_confidence: float | None = None,
    corridor_aware: bool = False,
    max_turn_deg: float = 40.0,
    alignment_rescue_max_gap_m: float = OPTIMIZER_ALIGNMENT_RESCUE_MAX_GAP_M,
    ref_name_lookup: dict[Any, Any] | None = None,
    target_name_lookup: dict[Any, Any] | None = None,
    ref_connector_index: _EndpointConnectorIndex | None = None,
    target_connector_index: _EndpointConnectorIndex | None = None,
) -> list[MatchResult]:
    """Expand greedy 1:1 matches to 1:N/N:1 groups where contiguous candidates exist.

    After greedy 1:1 optimization, some refs/targets have additional high-confidence
    candidates for contiguous segments that were excluded by the 1:1 constraint.
    This post-processing step adds those contiguous candidates, expanding 1:1 matches
    to 1:N (one ref → multiple contiguous targets) or N:1 (multiple contiguous refs
    → one target) groups.

    This is safe because it only ADDS matches — existing assignments are never removed.

    Args:
        greedy_matches: 1:1 matches from greedy optimization
        all_candidates: All candidate MatchResults (pre-optimization pool)
        ref_geoms: Reference geometry lookup
        target_geoms: Target geometry lookup
        tolerance: Contiguity tolerance in meters
        min_confidence: Minimum confidence for expansion candidates
        glue_min_confidence: Grouping-only floor for additions. Defaults to
            ``min_confidence``; candidates below it cannot re-glue anchors.
        corridor_aware: Gate contiguity on collinear continuation or same name.
        max_turn_deg: Deflection threshold for corridor-aware contiguity.
        alignment_rescue_max_gap_m: Maximum shared-side gap for the strict
            complementary same-name alignment rescue.
        ref_name_lookup: Optional reference ID -> name map.
        target_name_lookup: Optional target ID -> name map.

    Returns:
        Expanded list of MatchResult objects (original 1:1 + new group members)
    """
    if not greedy_matches:
        return []
    expansion_min_confidence = max(min_confidence, glue_min_confidence or min_confidence)

    # Track which (ref, target) pairs already exist in greedy matches
    existing_pairs = {(r.ref_id, r.target_id) for r in greedy_matches}

    # Build candidate lookup from ALL candidates (not just greedy winners).
    # A scored input can contain duplicate rows for one logical ID pair (for
    # example, repeated source geometries sharing an external ID). Deduplicate
    # before expansion so input order cannot make a lower-confidence duplicate
    # claim ``expanded_pairs`` ahead of the best row.
    best_candidates: dict[tuple[Any, Any], MatchResult] = {}
    for candidate in all_candidates:
        if not np.isfinite(candidate.confidence) or candidate.confidence < expansion_min_confidence:
            continue
        pair = (candidate.ref_id, candidate.target_id)
        current = best_candidates.get(pair)
        if current is None or _match_result_rank(candidate) < _match_result_rank(current):
            best_candidates[pair] = candidate

    candidates_by_ref: dict[Any, list[MatchResult]] = defaultdict(list)
    candidates_by_target: dict[Any, list[MatchResult]] = defaultdict(list)
    for pair in sorted(best_candidates, key=lambda item: (str(item[0]), str(item[1]))):
        candidate = best_candidates[pair]
        candidates_by_ref[candidate.ref_id].append(candidate)
        candidates_by_target[candidate.target_id].append(candidate)

    expanded: list[MatchResult] = []
    expanded_pairs: set = set()
    addition_review_pairs: set[tuple[Any, Any]] = set()

    # 1:N expansion: for each assigned ref, find contiguous candidate targets.
    # Targets already assigned to OTHER refs are allowed — this creates
    # legitimate N:1/M:N patterns where a target is covered by multiple refs.
    for match in greedy_matches:
        ref_id = match.ref_id
        assigned_target = match.target_id

        # Find other high-confidence candidates for this ref (any target, even claimed)
        other_targets = [
            c.target_id
            for c in candidates_by_ref[ref_id]
            if c.target_id != assigned_target
            and (ref_id, c.target_id) not in existing_pairs
            and (ref_id, c.target_id) not in expanded_pairs
        ]

        if not other_targets:
            continue

        # Check which are contiguous with the assigned target
        all_target_ids = [assigned_target] + other_targets
        target_groups = _find_contiguous_id_groups(
            all_target_ids,
            target_geoms,
            tolerance,
            require_collinear=corridor_aware,
            max_turn_deg=max_turn_deg,
            name_lookup=target_name_lookup,
        )
        target_groups, _rescued_target_ids = _merge_groups_by_alignment_rescue(
            target_groups,
            candidates_by_ref[ref_id],
            multi_id_attr="target_id",
            multi_name_lookup=target_name_lookup,
            multi_geom_lookup=target_geoms,
            shared_name=ref_name_lookup.get(ref_id) if ref_name_lookup else None,
            frac_start_attr="gers_start_frac",
            frac_end_attr="gers_end_frac",
            multi_frac_start_attr="local_start_frac",
            multi_frac_end_attr="local_end_frac",
            shared_segment_length_m=ref_geoms.get(ref_id).length
            if ref_geoms.get(ref_id) is not None
            else 0.0,
            max_gap_m=alignment_rescue_max_gap_m,
            endpoint_tolerance_m=tolerance,
            max_turn_deg=max_turn_deg,
            connector_index=target_connector_index,
        )
        # Find the group containing the assigned target
        for tg in target_groups:
            if assigned_target in tg and len(tg) > 1:
                new_target_ids = set(tg) - {assigned_target}
                for c in candidates_by_ref[ref_id]:
                    if (
                        c.target_id in new_target_ids
                        and (ref_id, c.target_id) not in expanded_pairs
                    ):
                        expanded.append(c)
                        expanded_pairs.add((ref_id, c.target_id))
                        if c.confidence < settings.optimizer_match_threshold:
                            addition_review_pairs.add((ref_id, c.target_id))

    # N:1 expansion: for each assigned target, find contiguous candidate refs.
    # Refs already assigned to OTHER targets are allowed.
    for match in greedy_matches:
        target_id = match.target_id
        assigned_ref = match.ref_id

        other_refs = [
            c.ref_id
            for c in candidates_by_target[target_id]
            if c.ref_id != assigned_ref
            and (c.ref_id, target_id) not in existing_pairs
            and (c.ref_id, target_id) not in expanded_pairs
        ]

        if not other_refs:
            continue

        all_ref_ids = [assigned_ref] + other_refs
        ref_groups = _find_contiguous_id_groups(
            all_ref_ids,
            ref_geoms,
            tolerance,
            require_collinear=corridor_aware,
            max_turn_deg=max_turn_deg,
            name_lookup=ref_name_lookup,
        )
        ref_groups, _rescued_ref_ids = _merge_groups_by_alignment_rescue(
            ref_groups,
            candidates_by_target[target_id],
            multi_id_attr="ref_id",
            multi_name_lookup=ref_name_lookup,
            multi_geom_lookup=ref_geoms,
            shared_name=target_name_lookup.get(target_id) if target_name_lookup else None,
            frac_start_attr="local_start_frac",
            frac_end_attr="local_end_frac",
            multi_frac_start_attr="gers_start_frac",
            multi_frac_end_attr="gers_end_frac",
            shared_segment_length_m=target_geoms.get(target_id).length
            if target_geoms.get(target_id) is not None
            else 0.0,
            max_gap_m=alignment_rescue_max_gap_m,
            endpoint_tolerance_m=tolerance,
            max_turn_deg=max_turn_deg,
            connector_index=ref_connector_index,
        )
        for rg in ref_groups:
            if assigned_ref in rg and len(rg) > 1:
                new_ref_ids = set(rg) - {assigned_ref}
                for c in candidates_by_target[target_id]:
                    if c.ref_id in new_ref_ids and (c.ref_id, target_id) not in expanded_pairs:
                        expanded.append(c)
                        expanded_pairs.add((c.ref_id, target_id))
                        if c.confidence < settings.optimizer_match_threshold:
                            addition_review_pairs.add((c.ref_id, target_id))

    if not expanded:
        return greedy_matches

    # Recompose the selected graph after both expansion directions. An added
    # edge can connect the ref-side expansion of one greedy anchor to the
    # target-side expansion of another; the former implementation emitted that
    # edge once from each anchor, producing duplicate selected pairs that #424's
    # selected-vs-pruned invariant correctly rejected. Connected-component
    # recomposition assigns every pair exactly once and gives cross-linked
    # expansions their actual M:N cardinality.
    components = find_match_components(greedy_matches + expanded, min_confidence)
    result: list[MatchResult] = []
    for component in components:
        if len(component) == 1:
            result.extend(component)
            continue

        n_refs = len({match.ref_id for match in component})
        n_targets = len({match.target_id for match in component})
        if n_refs == 1:
            match_type = MatchType.ONE_TO_N
        elif n_targets == 1:
            match_type = MatchType.N_TO_ONE
        else:
            match_type = MatchType.M_TO_N

        review_pairs = {
            (match.ref_id, match.target_id)
            for match in component
            if (match.ref_id, match.target_id) in addition_review_pairs
        }
        result.extend(
            _create_group_results(
                component,
                match_type,
                review_pairs,
                review_flag=LOW_CONFIDENCE_ADDITION_REVIEW_FLAG,
            )
        )

    return result


# Anti-crossing / mutual-exclusion guard for M:N groups (#367 "Mode A —
# parallel-sibling crossing swap"): two near-parallel refs are each plausible
# for two nearby targets, and a small segmentation-boundary-mismatch edge
# (covering only a sliver of BOTH its ref and its target) can still score
# confidently (~0.97-0.99) and survive the confidence-drop prune, over-merging
# the group with a spurious extra MATCH edge. See
# ``_contested_small_span_review_pairs``.
#
# Disposition is DEMOTE-TO-REVIEW, not drop: a flagged stub stays in the group
# (retained as a candidate edge, never silently removed) but its decision is
# demoted from MATCH to REVIEW — mirroring ``_validate_assignment_coverage`` —
# so it surfaces in /stitching-review for human adjudication instead of being
# auto-accepted into the production MATCH set. This keeps the guard safe on the
# genuine-but-contested N:1 fans it cannot geometrically distinguish from a true
# over-merge: those keep their real edge (as REVIEW), never losing coverage.
#
# The trigger is deliberately narrower than loosening ``SLIVER_SPAN_THRESHOLD``
# (config.py documents that as rejected: it risks dropping legitimate short
# matches). The span test only fires for edges that are ALSO contested — a rival
# edge sharing the same ref or target scores strictly higher — so an edge with no
# competing claim (the case the existing sliver / confidence machinery already
# governs) is never touched. And using max(ref_span, tgt_span) rather than min
# keeps every genuine asymmetric-coverage match (one side's span near 1.0 — e.g.
# a short ref fully consumed by a long target) exempt, no matter how small its
# OTHER side's span is.
MN_CONTESTED_EDGE_MAX_SPAN = 0.3

# Absolute-overlap companion to the fraction test. A low span FRACTION on a long
# segment can still be a large ABSOLUTE overlap — config.py documents exactly this
# false-positive class for a fraction-only rule ("25% of a 1.5 km ref is ~375 m of
# real road, not a stub"). Mirroring is_sliver_edge's hybrid rule, an edge is only
# a demotion candidate when the aligned overlap is ALSO small in meters, so genuine
# long-corridor edges are never routed to review. Missing lengths -> +inf overlap
# (via sliver_overlap_m), so an unmeasurable edge is never demoted.
#
# Calibrated on us_boston_streets: the 7 confirmed over-merge stubs (#367 Mode A)
# have aligned overlaps of 5.3-40.8 m, while the documented false-positive scale is
# ~375 m — so 75 m sits in the gap, keeping every true stub demotable (max 40.8 m)
# while exempting genuine long-corridor edges. A NOTE on why there is no score
# margin on "contested": those real crossing stubs are near-ties by nature (the
# rival often beats them by <0.02), so any epsilon on the contest test silently
# re-admits them — measured directly, a 0.02 margin lost 3 of the 7 true stubs.
MN_CONTESTED_MAX_ABS_OVERLAP_M = 75.0


def _edge_span_fracs(r: MatchResult) -> tuple[float, float]:
    """Alignment span fractions (ref_span, tgt_span) for a MatchResult.

    Mirrors ``matching.sliver.edge_span_fracs`` (which reads the same fractions
    off a serialized edge dict) for the in-memory ``MatchResult`` object.
    Missing fractions default to a full [0, 1] span so an unmeasurable edge is
    never mistaken for a small/contested one.
    """
    ref_span = 1.0
    if r.gers_start_frac is not None and r.gers_end_frac is not None:
        ref_span = abs(r.gers_end_frac - r.gers_start_frac)
    tgt_span = 1.0
    if r.local_start_frac is not None and r.local_end_frac is not None:
        tgt_span = abs(r.local_end_frac - r.local_start_frac)
    return ref_span, tgt_span


def _contested_small_span_review_pairs(
    component_results: list[MatchResult],
    ref_geoms: dict[Any, LineString] | None = None,
    target_geoms: dict[Any, LineString] | None = None,
) -> set[tuple[Any, Any]]:
    """Identify contested small-span M:N edges to demote to REVIEW (module note above).

    An edge is flagged when ALL of:
      - it is small on both alignment axes: ``max(ref_span, tgt_span) <
        MN_CONTESTED_EDGE_MAX_SPAN``; and
      - its aligned overlap is ALSO small in absolute meters (``<
        MN_CONTESTED_MAX_ABS_OVERLAP_M``) — the hybrid companion that exempts a
        genuine long-corridor edge whose low fraction still spans many meters;
        and
      - it is contested: another edge sharing its ``ref_id`` OR its ``target_id``
        has strictly higher confidence. (No score margin: real crossing stubs are
        near-ties, so any epsilon silently re-admits them — see the module note.)

    ``ref_geoms``/``target_geoms`` supply segment lengths for the absolute-overlap
    gate; a missing length yields +inf overlap, so an unmeasurable edge is never
    demoted (matching ``is_sliver_edge``'s fail-safe).

    Orphan-guard: a node (ref or target) is never left with zero MATCH edges. If
    demoting a flagged candidate would orphan its ref or its target (leave it
    with no un-flagged edge), the highest-confidence orphaning candidate is kept
    as MATCH instead (processed highest-confidence first, so at most one edge is
    rescued per orphaned node).

    Returns the set of ``(ref_id, target_id)`` pairs to demote to REVIEW. The
    edges themselves are NOT removed — the caller keeps them in the group and
    flips only their decision.
    """
    ref_geoms = ref_geoms or {}
    target_geoms = target_geoms or {}
    best_by_ref: dict[Any, float] = {}
    best_by_target: dict[Any, float] = {}
    for r in component_results:
        if r.confidence > best_by_ref.get(r.ref_id, -1.0):
            best_by_ref[r.ref_id] = r.confidence
        if r.confidence > best_by_target.get(r.target_id, -1.0):
            best_by_target[r.target_id] = r.confidence

    kept: list[MatchResult] = []
    candidate_demote: list[MatchResult] = []
    for r in component_results:
        ref_span, tgt_span = _edge_span_fracs(r)
        if max(ref_span, tgt_span) >= MN_CONTESTED_EDGE_MAX_SPAN:
            kept.append(r)
            continue
        ref_len = ref_geoms[r.ref_id].length if r.ref_id in ref_geoms else None
        tgt_len = target_geoms[r.target_id].length if r.target_id in target_geoms else None
        if sliver_overlap_m(ref_span, tgt_span, ref_len, tgt_len) >= MN_CONTESTED_MAX_ABS_OVERLAP_M:
            # Small fraction but a large absolute overlap — a genuine edge, not a
            # segmentation stub. Never demote.
            kept.append(r)
            continue
        contested = (
            best_by_ref[r.ref_id] > r.confidence or best_by_target[r.target_id] > r.confidence
        )
        if contested:
            candidate_demote.append(r)
        else:
            kept.append(r)

    if not candidate_demote:
        return set()

    kept_refs = {r.ref_id for r in kept}
    kept_targets = {r.target_id for r in kept}
    demote: set[tuple[Any, Any]] = set()
    for r in sorted(candidate_demote, key=lambda r: -r.confidence):
        if r.ref_id not in kept_refs or r.target_id not in kept_targets:
            # Rescue: demoting this would leave its ref or target with no MATCH
            # edge — keep it MATCH instead.
            kept_refs.add(r.ref_id)
            kept_targets.add(r.target_id)
        else:
            demote.add((r.ref_id, r.target_id))

    return demote


def _classify_and_resolve_component(
    component_results: list[MatchResult],
    ref_geoms: dict[Any, LineString],
    target_geoms: dict[Any, LineString],
    tolerance: float,
    corridor_aware: bool = False,
    max_turn_deg: float = 40.0,
    alignment_rescue_max_gap_m: float = OPTIMIZER_ALIGNMENT_RESCUE_MAX_GAP_M,
    ref_name_lookup: dict[Any, Any] | None = None,
    target_name_lookup: dict[Any, Any] | None = None,
    ref_connector_index: _EndpointConnectorIndex | None = None,
    target_connector_index: _EndpointConnectorIndex | None = None,
) -> tuple[list[MatchResult], list[MatchResult]]:
    """Classify a connected component and resolve it into groups.

    Classification rules:
    - |R|=1, |T|=1  → 1:1 candidate (pass to greedy optimizer)
    - |R|=1, |T|>1  → check target contiguity → contiguous subgroups = 1:N
    - |R|>1, |T|=1  → check ref contiguity → contiguous subgroups = N:1
    - |R|>1, |T|>1  → check BOTH sides fully contiguous → M:N, else decompose
                       into per-ref 1:N and per-target N:1 sub-problems

    Args:
        component_results: MatchResults in this component
        ref_geoms: Reference geometry lookup
        target_geoms: Target geometry lookup
        tolerance: Contiguity tolerance in meters

    Returns:
        Tuple of (group_results, leftover_1to1_candidates)
    """
    # Deduplicate to a CANONICAL order (sorted by stable string id) rather than
    # list(set(...)), whose order is hash-seed dependent for str ids. This id
    # order seeds contiguity grouping and the sub-component decomposition below,
    # so it must be deterministic to keep group membership/selection reproducible.
    ref_ids = sorted({r.ref_id for r in component_results}, key=str)
    target_ids = sorted({r.target_id for r in component_results}, key=str)

    n_refs = len(ref_ids)
    n_targets = len(target_ids)

    # Case 1: 1:1 — single ref, single target
    if n_refs == 1 and n_targets == 1:
        return [], component_results

    # Case 2: 1:N — single ref, multiple targets
    if n_refs == 1 and n_targets > 1:
        target_groups = _find_contiguous_id_groups(
            target_ids,
            target_geoms,
            tolerance,
            require_collinear=corridor_aware,
            max_turn_deg=max_turn_deg,
            name_lookup=target_name_lookup,
        )
        shared_ref_id = ref_ids[0]
        shared_ref_geom = ref_geoms.get(shared_ref_id)
        target_groups, _rescued_target_ids = _merge_groups_by_alignment_rescue(
            target_groups,
            component_results,
            multi_id_attr="target_id",
            multi_name_lookup=target_name_lookup,
            multi_geom_lookup=target_geoms,
            shared_name=ref_name_lookup.get(shared_ref_id) if ref_name_lookup else None,
            frac_start_attr="gers_start_frac",
            frac_end_attr="gers_end_frac",
            multi_frac_start_attr="local_start_frac",
            multi_frac_end_attr="local_end_frac",
            shared_segment_length_m=shared_ref_geom.length if shared_ref_geom is not None else 0.0,
            max_gap_m=alignment_rescue_max_gap_m,
            endpoint_tolerance_m=tolerance,
            max_turn_deg=max_turn_deg,
            connector_index=target_connector_index,
        )

        group_results: list[MatchResult] = []
        leftover: list[MatchResult] = []
        contiguous_match_groups: list[list[MatchResult]] = []
        singleton_matches: list[MatchResult] = []

        for tg in target_groups:
            tg_set = set(tg)
            group_matches = [r for r in component_results if r.target_id in tg_set]

            if len(tg) > 1:
                contiguous_match_groups.append(group_matches)
            else:
                singleton_matches.extend(group_matches)

        for group_matches in contiguous_match_groups:
            group_results.extend(_create_group_results(group_matches, MatchType.ONE_TO_N))
        leftover.extend(singleton_matches)

        return group_results, leftover

    # Case 3: N:1 — multiple refs, single target
    if n_refs > 1 and n_targets == 1:
        ref_groups = _find_contiguous_id_groups(
            ref_ids,
            ref_geoms,
            tolerance,
            require_collinear=corridor_aware,
            max_turn_deg=max_turn_deg,
            name_lookup=ref_name_lookup,
        )
        shared_target_id = target_ids[0]
        shared_target_geom = target_geoms.get(shared_target_id)
        ref_groups, _rescued_ref_ids = _merge_groups_by_alignment_rescue(
            ref_groups,
            component_results,
            multi_id_attr="ref_id",
            multi_name_lookup=ref_name_lookup,
            multi_geom_lookup=ref_geoms,
            shared_name=target_name_lookup.get(shared_target_id) if target_name_lookup else None,
            frac_start_attr="local_start_frac",
            frac_end_attr="local_end_frac",
            multi_frac_start_attr="gers_start_frac",
            multi_frac_end_attr="gers_end_frac",
            shared_segment_length_m=shared_target_geom.length
            if shared_target_geom is not None
            else 0.0,
            max_gap_m=alignment_rescue_max_gap_m,
            endpoint_tolerance_m=tolerance,
            max_turn_deg=max_turn_deg,
            connector_index=ref_connector_index,
        )

        group_results = []
        leftover = []
        contiguous_match_groups: list[list[MatchResult]] = []
        singleton_matches: list[MatchResult] = []

        for rg in ref_groups:
            rg_set = set(rg)
            group_matches = [r for r in component_results if r.ref_id in rg_set]

            if len(rg) > 1:
                contiguous_match_groups.append(group_matches)
            else:
                singleton_matches.extend(group_matches)

        for group_matches in contiguous_match_groups:
            group_results.extend(_create_group_results(group_matches, MatchType.N_TO_ONE))
        leftover.extend(singleton_matches)

        return group_results, leftover

    # Case 4: M:N — multiple refs AND multiple targets
    # Decompose using contiguity groups on both sides, then match sub-components.
    # Corridor-aware: gate contiguity on collinear continuation / same name so a
    # perpendicular junction kiss does not chain two different streets into one
    # over-merged "monster" corridor. Perpendicular corridors then fall into
    # separate contiguity groups and the per-(ref_group x target_group)
    # re-matching below decomposes the monster into per-corridor subgroups.
    ref_groups = _find_contiguous_id_groups(
        ref_ids,
        ref_geoms,
        tolerance,
        require_collinear=corridor_aware,
        max_turn_deg=max_turn_deg,
        name_lookup=ref_name_lookup,
    )
    target_groups = _find_contiguous_id_groups(
        target_ids,
        target_geoms,
        tolerance,
        require_collinear=corridor_aware,
        max_turn_deg=max_turn_deg,
        name_lookup=target_name_lookup,
    )

    refs_fully_contiguous = len(ref_groups) == 1 and len(ref_groups[0]) == n_refs
    targets_fully_contiguous = len(target_groups) == 1 and len(target_groups[0]) == n_targets

    if refs_fully_contiguous and targets_fully_contiguous:
        # Both sides fully contiguous → M:N group. Anti-crossing guard (#367
        # Mode A): demote contested small-span edges to REVIEW so a
        # confident-but-spurious boundary-mismatch edge doesn't over-merge the
        # MATCH set — the edge stays in the group for human adjudication (see
        # ``_contested_small_span_review_pairs``).
        review_pairs = _contested_small_span_review_pairs(
            component_results, ref_geoms, target_geoms
        )
        return _create_group_results(component_results, MatchType.M_TO_N, review_pairs), []

    # Decompose into sub-components by matching contiguity groups on each side.
    # For each (ref_group, target_group) pair, collect connecting edges.
    # Build edge lookup for quick access
    edge_by_pair: dict[tuple, MatchResult] = {}
    for r in component_results:
        edge_by_pair[(r.ref_id, r.target_id)] = r

    group_results: list[MatchResult] = []
    leftover: list[MatchResult] = []
    used_edges: set[tuple] = set()

    for rg in ref_groups:
        rg_set = set(rg)
        for tg in target_groups:
            tg_set = set(tg)
            # Collect edges connecting this ref_group to this target_group
            sub_edges = [
                edge_by_pair[(rid, tid)] for rid in rg for tid in tg if (rid, tid) in edge_by_pair
            ]
            if not sub_edges:
                continue

            sub_ref_ids = {r.ref_id for r in sub_edges}
            sub_target_ids = {r.target_id for r in sub_edges}

            # Classify the sub-component
            if len(sub_ref_ids) == 1 and len(sub_target_ids) == 1:
                # 1:1 sub-component → leftover for greedy
                leftover.extend(sub_edges)
            elif len(sub_ref_ids) == 1 and len(sub_target_ids) > 1:
                # 1:N sub-component
                group_results.extend(_create_group_results(sub_edges, MatchType.ONE_TO_N))
            elif len(sub_ref_ids) > 1 and len(sub_target_ids) == 1:
                # N:1 sub-component
                group_results.extend(_create_group_results(sub_edges, MatchType.N_TO_ONE))
            else:
                # Smaller M:N sub-component. Same anti-crossing guard as the
                # single-corridor M:N case above (#367 Mode A): demote contested
                # small-span edges to REVIEW, keeping them in the group.
                sub_review_pairs = _contested_small_span_review_pairs(
                    sub_edges, ref_geoms, target_geoms
                )
                group_results.extend(
                    _create_group_results(sub_edges, MatchType.M_TO_N, sub_review_pairs)
                )

            for e in sub_edges:
                used_edges.add((e.ref_id, e.target_id))

    # Any edges not assigned to a sub-component go to leftover
    for r in component_results:
        if (r.ref_id, r.target_id) not in used_edges:
            leftover.append(r)

    return group_results, leftover


def _validate_assignment_coverage(
    results: list[MatchResult],
    ref_geoms: dict[Any, LineString],
    target_geoms: dict[Any, LineString],
    max_overlap_m: float = MAX_ALIGNMENT_OVERLAP_M,
) -> list[MatchResult]:
    """Detect conflicting alignment coverage in assigned matches.

    Checks both assignment directions: no two targets should claim overlapping
    portions of one reference, and no two references should claim overlapping
    portions of one target. When either overlap exceeds ``max_overlap_m``, the
    lower-confidence match is demoted to REVIEW.

    Args:
        results: Optimized match results
        ref_geoms: Reference geometries for computing overlap in meters
        target_geoms: Target geometries for computing overlap in meters
        max_overlap_m: Maximum accepted overlap in meters

    Returns:
        Results with conflicting lower-confidence matches demoted to REVIEW
    """
    if not results:
        return results

    demote_sides: dict[int, set[str]] = defaultdict(set)
    accepted_by_ref: dict[Any, list[int]] = defaultdict(list)
    accepted_by_target: dict[Any, list[int]] = defaultdict(list)

    def _conflicts_with_accepted(
        candidate: MatchResult,
        accepted_indices: list[int],
        *,
        segment_id: Any,
        start_attr: str,
        end_attr: str,
        geom_lookup: dict[Any, LineString],
    ) -> bool:
        segment = geom_lookup.get(segment_id)
        segment_length_m = segment.length if segment is not None else 0.0
        if segment_length_m <= 0:
            return False
        candidate_range = _fraction_range(
            getattr(candidate, start_attr),
            getattr(candidate, end_attr),
        )
        if candidate_range is None:
            return False
        for accepted_index in accepted_indices:
            accepted = results[accepted_index]
            accepted_range = _fraction_range(
                getattr(accepted, start_attr),
                getattr(accepted, end_attr),
            )
            if accepted_range is None:
                continue
            overlap_frac = max(
                0.0,
                min(candidate_range[1], accepted_range[1])
                - max(candidate_range[0], accepted_range[0]),
            )
            if overlap_frac * segment_length_m > max_overlap_m:
                return True
        return False

    # Build a maximal, deterministic set of mutually compatible automatic
    # matches. Processing best-first preserves the higher-ranked edge on a
    # conflict. Crucially, an edge demoted on either axis is never admitted on
    # the other axis, so it cannot cascade a REVIEW decision into a compatible
    # lower-ranked MATCH (A overlaps B, B overlaps C, but A does not overlap C).
    # Pre-existing REVIEW edges likewise remain visible without blocking the
    # automatic assignment.
    ranked_match_indices = sorted(
        (index for index, result in enumerate(results) if result.decision == MatchDecision.MATCH),
        key=lambda index: _match_result_rank(results[index]),
    )
    for index in ranked_match_indices:
        result = results[index]
        if _conflicts_with_accepted(
            result,
            accepted_by_ref[result.ref_id],
            segment_id=result.ref_id,
            start_attr="gers_start_frac",
            end_attr="gers_end_frac",
            geom_lookup=ref_geoms,
        ):
            demote_sides[index].add("ref")
        if _conflicts_with_accepted(
            result,
            accepted_by_target[result.target_id],
            segment_id=result.target_id,
            start_attr="local_start_frac",
            end_attr="local_end_frac",
            geom_lookup=target_geoms,
        ):
            demote_sides[index].add("target")
        if index in demote_sides:
            continue
        accepted_by_ref[result.ref_id].append(index)
        accepted_by_target[result.target_id].append(index)

    if not demote_sides:
        return results

    demote_count = len(demote_sides)
    logger.info(f"  Coverage validation: demoting {demote_count} conflicting matches to REVIEW")

    validated: list[MatchResult] = []
    for i, r in enumerate(results):
        if i in demote_sides:
            conflict_features = {COVERAGE_CONFLICT_REVIEW_FLAG: 1.0}
            conflict_features.update({f"{side}_coverage_conflict": 1.0 for side in demote_sides[i]})
            validated.append(
                MatchResult(
                    ref_id=r.ref_id,
                    target_id=r.target_id,
                    decision=MatchDecision.REVIEW,
                    confidence=r.confidence,
                    score_breakdown=r.score_breakdown,
                    features={**r.features, **conflict_features},
                    ref_idx=r.ref_idx,
                    target_idx=r.target_idx,
                    gers_start_frac=r.gers_start_frac,
                    gers_end_frac=r.gers_end_frac,
                    local_start_frac=r.local_start_frac,
                    local_end_frac=r.local_end_frac,
                )
            )
        else:
            validated.append(r)

    return validated


def optimize_matches_with_grouping(
    results: list[MatchResult],
    reference: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    min_confidence: float = 0.5,
    contiguity_tolerance: float = DEFAULT_SNAP_TOLERANCE_M,
    ref_id_column: str = "id",
    target_id_column: str = "local_id",
    glue_min_confidence: float | None = None,
    corridor_aware: bool | None = None,
    corridor_max_turn_deg: float | None = None,
    alignment_rescue_max_gap_m: float = OPTIMIZER_ALIGNMENT_RESCUE_MAX_GAP_M,
) -> list[MatchResult]:
    """Optimize matches with M:N group detection.

    Unified bipartite connected-components approach that handles
    1:1, 1:N, N:1, and M:N match groups in one pass.

    Args:
        results: List of MatchResult objects
        reference: Reference GeoDataFrame (for ref geometry lookup)
        target: Target GeoDataFrame (for target geometry lookup)
        min_confidence: Minimum confidence threshold
        contiguity_tolerance: Max distance for contiguity check (meters)
        ref_id_column: Column name for reference IDs
        target_id_column: Column name for target IDs
        glue_min_confidence: Grouping-only confidence prune (see
            :func:`find_match_components`). Defaults to
            ``settings.optimizer_glue_min_confidence``.
        corridor_aware: Gate optimizer contiguity on collinear continuation / same
            name. Defaults to ``settings.optimizer_corridor_aware``.
        corridor_max_turn_deg: Collinearity deflection threshold (degrees).
            Defaults to ``settings.optimizer_corridor_max_turn_deg``.
        alignment_rescue_max_gap_m: Maximum uncovered shared-side distance for
            joining complementary same-name alignment fragments.

    Returns:
        List of optimized MatchResult objects
    """
    import time

    if glue_min_confidence is None:
        glue_min_confidence = settings.optimizer_glue_min_confidence
    if corridor_aware is None:
        corridor_aware = settings.optimizer_corridor_aware
    if corridor_max_turn_deg is None:
        corridor_max_turn_deg = settings.optimizer_corridor_max_turn_deg
    # The prune only makes sense above the candidate floor.
    glue_min_confidence = max(glue_min_confidence, min_confidence)

    t0 = time.perf_counter()
    logger.info(f"Optimizing {len(results)} results with M:N grouping...")

    # Build geometry lookups
    ref_geoms = _geom_lookup(reference, ref_id_column)
    target_geoms = _geom_lookup(target, target_id_column)

    # Name lookups for the corridor-aware same-name rescue (opt-in). Missing
    # names collapse to "" downstream so they never act as a match.
    ref_name_lookup = _name_lookup(reference, ref_id_column) if corridor_aware else None
    target_name_lookup = _name_lookup(target, target_id_column) if corridor_aware else None
    ref_connector_index = (
        _EndpointConnectorIndex(ref_geoms, contiguity_tolerance) if corridor_aware else None
    )
    target_connector_index = (
        _EndpointConnectorIndex(target_geoms, contiguity_tolerance) if corridor_aware else None
    )

    # Step 0: Classify junction-sliver candidate edges (shared hybrid rule).
    # Slivers never contribute adjacency when building components (a junction
    # kiss must not weld independent groups together) and are never SELECTED
    # into assignments — they only remain visible as ordinary in-group
    # candidates when both endpoints land in the same component anyway.
    metric = not (reference.crs is not None and reference.crs.is_geographic)
    sliver_edges = sliver_edges_for_match_results(results, ref_geoms, target_geoms, metric=metric)
    if sliver_edges:
        logger.info(f"  Classified {len(sliver_edges)} candidate edges as junction slivers")

    # Step 1: Find connected components. Sliver edges and (via the grouping-only
    # prune) weak edges below ``glue_min_confidence`` are excluded from adjacency
    # so they never weld independent groups into a monster.
    components = find_match_components(
        results,
        min_confidence,
        sliver_edges=sliver_edges,
        glue_min_confidence=glue_min_confidence,
    )
    logger.info(f"  Found {len(components)} connected components")

    # Step 2: Classify and resolve each component. Slivers are filtered from
    # the resolution input so they can never be selected into an assignment;
    # they stay in the component itself for group-membership purposes. Weak
    # (pruned) edges are NOT filtered here — they remain scored candidates and
    # can be selected; they simply did not contribute gluing above.
    all_group_results: list[MatchResult] = []
    all_leftover: list[MatchResult] = []
    decomposed_singleton_review_pairs: set[tuple[Any, Any]] = set()
    unattached_weak_by_ref: dict[Any, set[tuple[Any, Any]]] = defaultdict(set)
    unattached_weak_by_target: dict[Any, set[tuple[Any, Any]]] = defaultdict(set)

    for component in components:
        structural = [r for r in component if (r.ref_id, r.target_id) not in sliver_edges]
        if not structural:
            continue
        was_multi_node = (
            len({result.ref_id for result in structural}) > 1
            or len({result.target_id for result in structural}) > 1
        )
        group_results, leftover = _classify_and_resolve_component(
            structural,
            ref_geoms,
            target_geoms,
            contiguity_tolerance,
            corridor_aware=corridor_aware,
            max_turn_deg=corridor_max_turn_deg,
            alignment_rescue_max_gap_m=alignment_rescue_max_gap_m,
            ref_name_lookup=ref_name_lookup,
            target_name_lookup=target_name_lookup,
            ref_connector_index=ref_connector_index,
            target_connector_index=target_connector_index,
        )
        if was_multi_node:
            decomposed_singleton_review_pairs.update(
                (result.ref_id, result.target_id)
                for result in leftover
                if result.confidence < settings.optimizer_match_threshold
            )
        all_group_results.extend(group_results)
        all_leftover.extend(leftover)

    # Step 2b: Recover weak edges that the grouping-only prune left unattached
    # (both endpoints failed to co-land in any component). They are still valid
    # candidates, so feed them to the greedy 1:1 pool — the prune must not
    # reduce match coverage, only stop weak edges from GLUING monsters.
    if glue_min_confidence > min_confidence:
        in_component: set[tuple[Any, Any]] = {
            (r.ref_id, r.target_id) for comp in components for r in comp
        }
        best_unattached: dict[tuple[Any, Any], MatchResult] = {}
        for r in results:
            pair = (r.ref_id, r.target_id)
            if (
                min_confidence <= r.confidence < glue_min_confidence
                and pair not in sliver_edges
                and pair not in in_component
            ):
                if pair not in best_unattached or _match_result_rank(r) < _match_result_rank(
                    best_unattached[pair]
                ):
                    best_unattached[pair] = r
        if best_unattached:
            for pair in best_unattached:
                unattached_weak_by_ref[pair[0]].add(pair)
                unattached_weak_by_target[pair[1]].add(pair)
            all_leftover.extend(best_unattached.values())

    # Step 3: Track claimed refs/targets from groups
    claimed_refs = {r.ref_id for r in all_group_results}
    claimed_targets = {r.target_id for r in all_group_results}

    # Filter leftover to exclude any already claimed
    unclaimed_leftover = [
        r
        for r in all_leftover
        if r.ref_id not in claimed_refs and r.target_id not in claimed_targets
    ]

    # Step 4: Run greedy 1:1 optimization on unclaimed leftovers
    if unclaimed_leftover:
        optimized_1to1 = optimize_matches_greedy(
            unclaimed_leftover,
            min_confidence,
            ref_name_lookup=ref_name_lookup,
            target_name_lookup=target_name_lookup,
        )
    else:
        optimized_1to1 = []

    # Step 5: Post-expansion — expand 1:1 matches to 1:N/N:1 where
    # contiguous candidates exist. Uses ALL candidates (not just unclaimed
    # leftover) so refs can expand to targets claimed by other refs.
    # This only ADDS matches, never removes existing assignments.
    # Sliver candidates are excluded so expansion can never select one.
    if optimized_1to1:
        expandable = [r for r in results if (r.ref_id, r.target_id) not in sliver_edges]
        optimized_1to1 = _expand_greedy_matches(
            optimized_1to1,
            expandable,  # All non-sliver candidates, not just unclaimed
            ref_geoms,
            target_geoms,
            contiguity_tolerance,
            min_confidence,
            glue_min_confidence=glue_min_confidence,
            corridor_aware=corridor_aware,
            max_turn_deg=corridor_max_turn_deg,
            alignment_rescue_max_gap_m=alignment_rescue_max_gap_m,
            ref_name_lookup=ref_name_lookup,
            target_name_lookup=target_name_lookup,
            ref_connector_index=ref_connector_index,
            target_connector_index=target_connector_index,
        )

    for result in optimized_1to1:
        if result.confidence >= settings.optimizer_match_threshold:
            continue
        pair = (result.ref_id, result.target_id)
        weak_alternatives = unattached_weak_by_ref.get(result.ref_id, set()).union(
            unattached_weak_by_target.get(result.target_id, set())
        )
        if any(alternative != pair for alternative in weak_alternatives):
            decomposed_singleton_review_pairs.add(pair)

    if decomposed_singleton_review_pairs:
        optimized_1to1 = [
            _force_review(result, DECOMPOSED_SINGLETON_REVIEW_FLAG)
            if (result.ref_id, result.target_id) in decomposed_singleton_review_pairs
            else result
            for result in optimized_1to1
        ]

    # Combine results and validate alignment coverage
    final = _validate_assignment_coverage(
        all_group_results + optimized_1to1,
        ref_geoms,
        target_geoms,
    )

    # Log summary — count match types across ALL results
    type_counts = defaultdict(int)
    for r in final:
        mt = r.features.get("match_type", MatchType.ONE_TO_ONE)
        type_counts[mt] += 1

    t1 = time.perf_counter()
    logger.info(
        f"  Optimization complete in {t1 - t0:.2f}s: "
        f"{type_counts.get(MatchType.ONE_TO_ONE, 0)} 1:1, "
        f"{type_counts.get(MatchType.ONE_TO_N, 0)} 1:N, "
        f"{type_counts.get(MatchType.N_TO_ONE, 0)} N:1, "
        f"{type_counts.get(MatchType.M_TO_N, 0)} M:N"
    )
    logger.info(f"  Total output: {len(final)} matches")

    return final


def apply_confidence_drop_prune(
    results: list[MatchResult],
    min_confidence: float,
) -> tuple[list[MatchResult], set[tuple[Any, Any]]]:
    """Drop low-confidence SELECTED group edges (M2 / resolver Phase 1).

    Post-optimizer prune: within each M:N / 1:N / N:1 group (a result carrying a
    ``group_id``), drop any selected edge whose confidence is below
    ``min_confidence``. This is the one-parameter confidence filter the #272
    resolver eval validated — an ABSOLUTE threshold, not group-relative
    (``evaluate.py::run_cv`` tuned raw ``confidence >= t`` and that baseline beat
    both keep-all and the learned per-edge model on the clean slice).

    Guarantees:
    - 1:1 matches (no ``group_id``) are never touched — the eval only covered
      M:N group selections.
    - Each group always retains its single highest-confidence edge, so a group is
      never fully emptied (respects the "keep the corridor's backbone edge"
      spirit of the Phase-1 single-corridor exemption). If pruning leaves only
      that edge and it is below ``optimizer_match_threshold``, it is retained as
      REVIEW rather than promoted as an automatic singleton match.
    - After that ordinary prune/demotion, at most one fully named, nearly
      full-coverage edge in the optimizer's REVIEW band is recovered as REVIEW
      when both endpoints would otherwise disappear. This narrow orphan rescue
      preserves an independently valid corridor fragment without re-admitting a
      higher-scoring junction cross-link (Boston Carver Street regression).
    - Identity when nothing qualifies: the input list is returned unchanged and
      ``pruned_pairs`` is empty. When ``min_confidence <= 0`` nothing is dropped.

    Returns ``(kept_results, pruned_pairs)`` where ``pruned_pairs`` is the set of
    dropped ``(ref_id, target_id)`` pairs (raw id types).
    """
    if min_confidence <= 0:
        return results, set()

    by_gid: dict[Any, list[int]] = defaultdict(list)
    for i, r in enumerate(results):
        gid = r.features.get("group_id")
        if gid:
            by_gid[gid].append(i)

    pruned_idx: set[int] = set()
    for _gid, idxs in by_gid.items():
        # The canonical best edge is always retained (never empty the group).
        keep_top = min(idxs, key=lambda index: _match_result_rank(results[index]))
        for i in idxs:
            if i != keep_top and results[i].confidence < min_confidence:
                pruned_idx.add(i)

    if not pruned_idx:
        return results, set()

    singleton_review_idx: set[int] = set()
    for idxs in by_gid.values():
        kept_in_group = [index for index in idxs if index not in pruned_idx]
        if (
            len(kept_in_group) == 1
            and any(index in pruned_idx for index in idxs)
            and results[kept_in_group[0]].confidence < settings.optimizer_match_threshold
        ):
            singleton_review_idx.add(kept_in_group[0])

    orphan_review_idx: set[int] = set()
    exact_name_features = (
        "name_levenshtein",
        "name_jaro_winkler",
        "name_token_sort",
        "name_soundex",
        "name_metaphone",
    )
    for idxs in by_gid.values():
        normal_survivors = [index for index in idxs if index not in pruned_idx]
        survivor_refs = {results[index].ref_id for index in normal_survivors}
        survivor_targets = {results[index].target_id for index in normal_survivors}
        eligible: list[int] = []
        for index in idxs:
            if index not in pruned_idx:
                continue
            result = results[index]
            if not (
                settings.optimizer_review_threshold
                <= result.confidence
                < settings.optimizer_match_threshold
            ):
                continue
            if result.ref_id in survivor_refs or result.target_id in survivor_targets:
                continue
            features = result.features or {}
            try:
                names_are_exact = (
                    features.get("has_name_ref") == 1.0
                    and features.get("has_name_target") == 1.0
                    and features.get("name_is_generic") == 0.0
                    and all(features.get(name) == 1.0 for name in exact_name_features)
                )
            except (TypeError, ValueError):
                names_are_exact = False
            if not names_are_exact:
                continue
            ref_range = _fraction_range(result.gers_start_frac, result.gers_end_frac)
            target_range = _fraction_range(result.local_start_frac, result.local_end_frac)
            if ref_range is None or target_range is None:
                continue
            if not (
                0.0 <= ref_range[0] < ref_range[1] <= 1.0
                and 0.0 <= target_range[0] < target_range[1] <= 1.0
            ):
                continue
            coverages = (ref_range[1] - ref_range[0], target_range[1] - target_range[0])
            if min(coverages) < 0.70 or max(coverages) < 0.99:
                continue
            eligible.append(index)
        if eligible:
            orphan_review_idx.add(
                min(eligible, key=lambda index: _match_result_rank(results[index]))
            )

    pruned_idx.difference_update(orphan_review_idx)

    pruned_pairs = {(results[i].ref_id, results[i].target_id) for i in pruned_idx}
    kept: list[MatchResult] = []
    for index, result in enumerate(results):
        if index in pruned_idx:
            continue
        if index in singleton_review_idx or index in orphan_review_idx:
            result = MatchResult(
                ref_id=result.ref_id,
                target_id=result.target_id,
                decision=MatchDecision.REVIEW,
                confidence=result.confidence,
                score_breakdown=result.score_breakdown,
                features={**result.features, PRUNED_SINGLETON_REVIEW_FLAG: 1.0},
                ref_idx=result.ref_idx,
                target_idx=result.target_idx,
                gers_start_frac=result.gers_start_frac,
                gers_end_frac=result.gers_end_frac,
                local_start_frac=result.local_start_frac,
                local_end_frac=result.local_end_frac,
            )
        kept.append(result)
    return kept, pruned_pairs


def group_is_structurally_simple(
    n_corridors: int,
    n_assignment_components: int,
    n_edges: int,
    max_assignment_components: int,
    soft_max_edges: int,
    backstop_max_edges: int,
) -> bool:
    """Structural export gate: is this group a clean, single-decision unit?

    Replaces a flat edge-count cap. A group is structurally simple when it is a
    single corridor-pair (one street matched to one street — fine even when long,
    e.g. a 30-edge Beacon-St corridor) OR it has few assignment-components and
    stays within a soft edge budget. A hard backstop ceiling blocks anything
    larger regardless — a defence against a structure-detection bug ever
    auto-exporting a monster, NOT the primary gate.

    Args:
        n_corridors: Number of distinct corridor-pairs in the candidate graph.
        n_assignment_components: Connected components of the selected assignment.
        n_edges: Candidate edge count.
        max_assignment_components: Max assignment-components for a simple group.
        soft_max_edges: Soft edge budget for a multi-component simple group.
        backstop_max_edges: Hard ceiling; nothing above this is ever simple.

    Returns:
        True if the group may auto-export under the structural gate.
    """
    if n_edges > backstop_max_edges:
        return False
    if n_corridors <= 1:
        return True
    return n_assignment_components <= max_assignment_components and n_edges <= soft_max_edges


def compute_group_structure(
    edges: list[tuple[Any, Any]],
    ref_ids: list[Any],
    target_ids: list[Any],
    assignment_pairs: set[tuple[Any, Any]],
    sliver_pairs: set[tuple[Any, Any]],
    ref_geoms: dict[Any, LineString],
    target_geoms: dict[Any, LineString],
    tolerance: float = DEFAULT_SNAP_TOLERANCE_M,
    corridor_aware: bool = True,
    max_turn_deg: float = 40.0,
    ref_name_lookup: dict[Any, Any] | None = None,
    target_name_lookup: dict[Any, Any] | None = None,
) -> tuple[dict[tuple[Any, Any], dict], dict]:
    """Compute candidate-graph structure features for a match group.

    Derives per-edge topology (degree, bridge, biconnected block, corridor ids,
    selected, sliver) and per-group counts (edges, corridors,
    assignment-components, largest biconnected block) from data already in the
    sidecar. These are cheap, purely structural signals persisted for the future
    learned group resolver (see the group-splitting design doc §6) and to drive
    the ``oversized_group`` flag and the structural export gate.

    Args:
        edges: All candidate ``(ref_id, target_id)`` pairs in the group.
        ref_ids / target_ids: The group's segment ids.
        assignment_pairs: Pairs selected by the optimizer assignment.
        sliver_pairs: Pairs classified as junction slivers.
        ref_geoms / target_geoms: Projected (metric) geometry lookups.
        tolerance: Contiguity tolerance (meters) for corridor detection.
        corridor_aware / max_turn_deg / *_name_lookup: Corridor gate config,
            matching the optimizer so corridor ids agree with the grouping.

    Returns:
        ``(per_edge, per_group)`` where ``per_edge`` maps each pair to its
        structure dict and ``per_group`` holds the group-level counts.
    """
    import networkx as nx

    # Corridor ids from same-side contiguity (collinear-gated), so a corridor is
    # a maximal collinear/same-name chain of segments.
    def _corridor_index(ids, geoms, names):
        groups = _find_contiguous_id_groups(
            list(ids),
            geoms,
            tolerance,
            require_collinear=corridor_aware,
            max_turn_deg=max_turn_deg,
            name_lookup=names,
        )
        return {sid: i for i, grp in enumerate(groups) for sid in grp}

    ref_corridor = _corridor_index(ref_ids, ref_geoms, ref_name_lookup)
    tgt_corridor = _corridor_index(target_ids, target_geoms, target_name_lookup)

    # Candidate bipartite graph (slivers excluded — they are junction artifacts,
    # not real adjacency).
    g = nx.Graph()
    g.add_nodes_from(("ref", r) for r in ref_ids)
    g.add_nodes_from(("target", t) for t in target_ids)
    structural_edges = [e for e in edges if e not in sliver_pairs]
    for rid, tid in structural_edges:
        g.add_edge(("ref", rid), ("target", tid))

    degree = dict(g.degree())
    bridge_set = set()
    block_of_edge: dict[frozenset, int] = {}
    largest_block = 0
    if g.number_of_edges() > 0:
        for a, b in nx.bridges(g):
            bridge_set.add(frozenset((a, b)))
        for i, block in enumerate(nx.biconnected_component_edges(g)):
            block = list(block)
            for a, b in block:
                block_of_edge[frozenset((a, b))] = i
            largest_block = max(largest_block, len(block))

    # Assignment-components: connected components of the selected subgraph.
    ga = nx.Graph()
    for rid, tid in assignment_pairs:
        ga.add_edge(("ref", rid), ("target", tid))
    n_assignment_components = nx.number_connected_components(ga) if ga.number_of_edges() else 0

    # Corridor-pairs actually connected by a (non-sliver) edge.
    corridor_pairs = {
        (ref_corridor.get(rid), tgt_corridor.get(tid)) for rid, tid in structural_edges
    }
    n_corridors = len(corridor_pairs) if corridor_pairs else 0

    per_edge: dict[tuple[Any, Any], dict] = {}
    for rid, tid in edges:
        rn, tn = ("ref", rid), ("target", tid)
        key = frozenset((rn, tn))
        is_sliver = (rid, tid) in sliver_pairs
        per_edge[(rid, tid)] = {
            "degree_ref": int(degree.get(rn, 0)),
            "degree_tgt": int(degree.get(tn, 0)),
            "is_bridge": (not is_sliver) and key in bridge_set,
            "biconnected_block": block_of_edge.get(key, -1) if not is_sliver else -1,
            "corridor_ref": ref_corridor.get(rid, -1),
            "corridor_tgt": tgt_corridor.get(tid, -1),
            "selected": (rid, tid) in assignment_pairs,
            "is_sliver": is_sliver,
        }

    per_group = {
        "n_edges": len(edges),
        "n_corridors": n_corridors,
        "n_assignment_components": n_assignment_components,
        "largest_biconnected_block": largest_block,
    }
    return per_edge, per_group
