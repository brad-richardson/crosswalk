"""Top-K assignment alternative generation for M:N match groups.

Enumerates valid assignment combinations for a match group's candidate
edges, filters by contiguity and overlap constraints, and returns the
top K alternatives ranked by total confidence.

Each target's choice set is not limited to a *single* reference segment: it
also includes CONTIGUOUS multi-ref chains (e.g. "T4 spans R3 and R5
end-to-end"), built from the same endpoint-proximity contiguity the optimizer
uses (``optimizer.build_contiguity_adjacency``). Without this, a target that
legitimately covers several reference segments could never appear in any
option, so the correct answer was structurally inexpressible. Options remain
strict subsets of the group's EXISTING edges: a chain ``{R3, R5}`` for target
``T4`` is only offered when both ``(R3, T4)`` and ``(R5, T4)`` edges (each with
an ML confidence) are present in the group.
"""

import itertools
import logging
import math
from collections import defaultdict

from shapely import LineString

from ..config import DEFAULT_SNAP_TOLERANCE_M

logger = logging.getLogger(__name__)

# Enumeration strategy thresholds: groups below these limits use exhaustive
# enumeration (all combos via itertools.product); larger groups fall back to
# greedy assignment + perturbation.
MAX_EXHAUSTIVE_TARGETS = 6  # max target segments for exhaustive 1:N / M:N
MAX_EXHAUSTIVE_COMBOS = 10_000  # max total combos before switching to greedy
MAX_EXHAUSTIVE_N_TO_1_REFS = 10  # max ref segments for exhaustive N:1 (2^N subsets)

# Contiguous multi-ref chain bounds.
#
# A single local target rarely spans more than a couple of reference segments
# end-to-end, so contiguous ref-chains are enumerated only up to this length.
# Keeping the bound small (chains of 2-3 refs) is what keeps the per-target
# option count — and hence the itertools.product blow-up — bounded.
MAX_REF_CHAIN_LEN = 3
# Cap on contiguous multi-ref chains offered per target, keeping the
# highest-total-confidence chains. Blow-up analysis: each target's option count
# is at most (#refs-with-an-edge) singletons + MAX_CHAINS_PER_TARGET chains + 1
# ("unassigned"). The exhaustive product over up to MAX_EXHAUSTIVE_TARGETS
# targets is still gated by MAX_EXHAUSTIVE_COMBOS (groups that exceed it fall
# back to greedy), so worst-case enumeration stays bounded even though the
# per-target choice set grew.
MAX_CHAINS_PER_TARGET = 6

# Contiguity tolerance for chain building, in meters — matches the optimizer's
# endpoint-snap tolerance so options can express the spans the optimizer
# produces. Group geometries arrive as WGS84 (lon/lat) GeoJSON and are
# projected to a local equirectangular meter frame before the check; the
# optimizer checks contiguity on properly projected (ensure_projected_crs)
# geometries, so boundary cases right at the tolerance can differ slightly
# between the two paths.
CHAIN_CONTIGUITY_TOLERANCE_M = DEFAULT_SNAP_TOLERANCE_M

# Beam width for chain construction: at each chain size the frontier keeps
# only the top subsets by total confidence. In dense adjacency graphs (many
# refs meeting one intersection within tolerance) the number of connected
# size-3 subsets is O(n^3); the beam keeps construction itself bounded, not
# just the final per-target option list.
MAX_CHAIN_FRONTIER = 64

# Alignment fraction keys preserved from sidecar edges through to alternatives
_ALIGNMENT_KEYS = ("gers_start_frac", "gers_end_frac", "local_start_frac", "local_end_frac")

# --- "Base set minus flagged edge(s)" seed bounds ------------------------------
#
# Beyond the two whole-group seeds (full candidate set + optimizer selection),
# the generator also emits a TARGETED, bounded set of "base minus a droppable
# edge" seeds, so the settled answer "the optimizer set except that one
# sliver/parallel-sibling/low-confidence edge" is directly expressible instead of
# only appearing when it happens to rank into the confidence-sorted top-K (the
# measured expressibility hole: most no-exact-option NONE ballots).
#
# The additions are bounded so the menu never balloons into a blind power set
# (menu bloat + the pixel-identical-image problem): at most
# ``MAX_FLAGGED_SINGLE_DROPS`` single-edge removals per base, plus one combined
# removal of the top few flagged edges, all deduped. An edge is a drop candidate
# only when it carries a real droppability SIGNAL (never flagged blindly by
# count); the cap then keeps the *most* droppable ones.
MAX_FLAGGED_SINGLE_DROPS = 6  # max single-edge removals emitted per base set
MAX_COMBINED_DROP = 3  # max edges removed together in the one combined seed
# An edge's confidence is "low" (a drop signal) if it is below this absolute
# floor, OR this far below the base set's strongest edge (relative gap).
LOW_CONF_ABS = 0.85
LOW_CONF_REL_MARGIN = 0.10


def generate_top_k_alternatives(
    component_edges: list[dict],
    ref_geoms: dict[str, dict] | None = None,
    target_geoms: dict[str, dict] | None = None,
    k: int = 5,
    include_seed_options: bool = True,
) -> list[dict]:
    """Generate top-K assignment alternatives for a match group.

    For each target, enumerates which ref(s) it could be assigned to: any
    single ref that has an edge to it, any CONTIGUOUS chain of up to
    ``MAX_REF_CHAIN_LEN`` such refs, or "unassigned". Ranks by total confidence
    and returns the top K.

    For groups with <= MAX_EXHAUSTIVE_TARGETS targets (and a product under
    MAX_EXHAUSTIVE_COMBOS), uses exhaustive enumeration via itertools.product.
    For larger groups, falls back to greedy perturbation (which can also
    propose contiguous multi-ref edges).

    Seed options (see ``include_seed_options``): the per-target enumeration and
    greedy perturbation can structurally fail to express some settled answers —
    e.g. an M:N group whose correct answer is "accept every candidate edge" (a
    target legitimately spanning >MAX_REF_CHAIN_LEN refs), or a large group whose
    optimizer selection ranks below the confidence-sorted top-K. To close that
    measured expressibility gap, whole-group seed alternatives are ALWAYS
    appended (deduped) after the organic top-K, so they survive truncation:
      * the full candidate set (union of all group edges),
      * the optimizer's selected edge set (edges flagged ``selected`` — a
        known-good assignment even when ``optimizer_assignment`` is empty, which
        it is for most giant M:N groups), and
      * a TARGETED, bounded set of "base set minus flagged edge(s)" seeds — each
        base (optimizer selection / full candidate set) with the most-droppable
        edge(s) removed — so "the optimizer set except that one bad edge" is
        directly expressible (see ``_seed_alternatives`` / ``_edge_droppability``).
    The minus-edge seeds are bounded (at most ``MAX_FLAGGED_SINGLE_DROPS`` single
    removals per base plus one combined removal, all deduped), so the menu never
    balloons into a blind power set.

    Args:
        component_edges: List of edge dicts with ref_id, target_id, confidence,
            an optional ``selected`` flag (the optimizer's per-edge selection),
            and optional alignment fracs (gers_start_frac, gers_end_frac,
            local_start_frac, local_end_frac)
        ref_geoms: Optional {ref_id: geometry} used to build contiguous ref
            chains. Geometry may be a GeoJSON-style mapping ({"coordinates": ...})
            or a shapely LineString. When absent, only single-ref options are
            enumerated (multi-ref spans require geometry to establish
            contiguity).
        target_geoms: Reserved for symmetry; unused (the M:N path enumerates per
            target, so only ref-side contiguity is needed, and the N:1 path
            already enumerates the full ref power set).
        k: Number of top organic alternatives to return (before seeds)
        include_seed_options: When True (default), append the full-candidate-set
            and optimizer-selected-set seed options after the top-K. Set False
            to recover the pure top-K-by-confidence behavior.

    Returns:
        List of alternative dicts, each with:
        - option_index: int
        - edges: list of {ref_id, target_id, confidence, ...alignment fracs}
        - total_confidence: float
        - summary: human-readable string
    """
    if not component_edges:
        return []

    # Collect unique refs and targets
    ref_ids = sorted(set(e["ref_id"] for e in component_edges))
    target_ids = sorted(set(e["target_id"] for e in component_edges))

    # Build lookup: (ref_id, target_id) -> full edge data (confidence + alignment fracs).
    # On duplicate keys, keep the edge with highest confidence.
    edge_data: dict[tuple[str, str], dict] = {}
    for e in component_edges:
        key = (e["ref_id"], e["target_id"])
        if key not in edge_data or e["confidence"] > edge_data[key]["confidence"]:
            edge_data[key] = e

    # For N:1 groups (multiple refs, 1 target), enumerate per-ref assignment
    # (each ref independently maps to the target or not — the full ref power
    # set, so a ref-side multi-span is already expressible). For 1:N and M:N,
    # enumerate per-target assignment (each target maps to a ref, a contiguous
    # ref chain, or nothing).
    if len(target_ids) == 1 and len(ref_ids) > 1:
        alternatives = _enumerate_n_to_1(ref_ids, target_ids[0], edge_data, k)
    else:
        # Project ref geometries to a metric frame once, for contiguity.
        ref_metric = _metric_geom_lookup(ref_geoms)

        # Build per-target options: each option is a tuple of ref_ids (a single
        # ref is a 1-tuple; a contiguous chain is a 2- or 3-tuple) or None.
        target_options: dict[str, list[tuple[str, ...] | None]] = {}
        for tid in target_ids:
            refs_for_t = [rid for rid in ref_ids if (rid, tid) in edge_data]
            subsets = _target_ref_subsets(refs_for_t, tid, edge_data, ref_metric)
            subsets.append(None)  # "unassigned" option
            target_options[tid] = subsets

        # Decide enumeration strategy. n_combos reflects the *actual* per-target
        # option counts (singletons + capped chains + None), so the exhaustive
        # product stays gated by MAX_EXHAUSTIVE_COMBOS despite the larger sets.
        n_combos = 1
        for tid in target_ids:
            n_combos *= len(target_options[tid])

        if len(target_ids) <= MAX_EXHAUSTIVE_TARGETS and n_combos <= MAX_EXHAUSTIVE_COMBOS:
            alternatives = _exhaustive_enumeration(target_ids, target_options, edge_data, ref_ids)
        else:
            alternatives = _greedy_perturbation(
                target_ids, target_options, edge_data, ref_ids, k * 3
            )

    # Output invariant: every alternative must be a deduplicated subset of the
    # group's candidate edges. The enumeration paths above already build edges
    # only from ``edge_data``, but enforce it explicitly here so a future path
    # bug (or a caller mutating the group after generation) can never ship an
    # alternative that references edges outside the group.
    valid_keys = set(edge_data.keys())
    alternatives = [_sanitize_alternative(a, valid_keys, ref_ids) for a in alternatives]
    alternatives = [a for a in alternatives if a["edges"]]

    # Sort by total confidence descending
    alternatives.sort(key=lambda a: a["total_confidence"], reverse=True)

    # Deduplicate (same edge set)
    seen: set[frozenset] = set()
    unique = []
    for alt in alternatives:
        edge_key = frozenset((e["ref_id"], e["target_id"]) for e in alt["edges"])
        if edge_key not in seen:
            seen.add(edge_key)
            unique.append(alt)

    # Take top K organic alternatives.
    top_k = unique[:k]

    # Append whole-group seed options (full set + optimizer-selected set) AFTER
    # truncation, so they are guaranteed present even when they rank below the
    # confidence-sorted top-K (large groups) or are structurally inexpressible by
    # the per-target enumeration (e.g. settled answer == accept every edge).
    if include_seed_options:
        # Selected pairs are collected from ALL input edges (not the deduped
        # edge_data survivors), so the flag is not lost when a duplicate pair's
        # flagged edge had lower confidence than the copy kept in edge_data.
        selected_pairs = {
            (e["ref_id"], e["target_id"]) for e in component_edges if e.get("selected")
        }
        selected_keys = [key for key in edge_data if key in selected_pairs]
        seen_keys = {
            frozenset((e["ref_id"], e["target_id"]) for e in alt["edges"]) for alt in top_k
        }
        for seed in _seed_alternatives(edge_data, ref_ids, selected_keys):
            key = frozenset((e["ref_id"], e["target_id"]) for e in seed["edges"])
            if key and key not in seen_keys:
                seen_keys.add(key)
                top_k.append(seed)

    # Assign option indices over the final (organic + seed) list.
    for i, alt in enumerate(top_k):
        alt["option_index"] = i

    return top_k


def _edge_droppability(edge: dict, base_max_conf: float) -> float:
    """Heuristic "how droppable is this edge" score (``> 0`` => a flag candidate).

    Uses only signals already present on the sidecar edge dict — never geometry
    or anything the alternatives path doesn't already carry:

    * ``is_sliver`` — a junction sliver (the optimizer's own sliver classifier).
    * ``pruned`` — an edge the optimizer explicitly pruned.
    * ``decision == "review"`` — flagged for review rather than accepted.
    * ``review_reason`` present (e.g. ``parallel_sibling``, ``coverage_conflict``,
      ``low_confidence_addition``) — a borderline edge the optimizer annotated.
    * ``selected`` is False — the optimizer did not pick it, so within the FULL
      candidate set it is, by the optimizer's own reckoning, a natural drop.
    * low confidence relative to the base's strongest edge (or below an absolute
      floor) — the classic "weak edge" drop signal.

    Returns ``0.0`` for an edge with no droppability signal, so a base set of
    uniformly strong, accepted, non-sliver edges flags nothing (no minus-edge
    seed is emitted for it). The score only RANKS the flagged edges; the caller
    caps how many are dropped.
    """
    conf = float(edge.get("confidence", 0.0) or 0.0)
    score = 0.0
    if edge.get("is_sliver"):
        score += 3.0
    if edge.get("pruned"):
        score += 3.0
    if str(edge.get("decision", "")).strip().lower() == "review":
        score += 2.0
    rr = edge.get("review_reason")
    if rr is not None and str(rr).strip().lower() not in ("", "nan", "none"):
        score += 2.0
    # An edge the optimizer did NOT select is the most droppable member of the
    # full candidate set. Only counts when the flag is present *and* False (a
    # missing ``selected`` key is not evidence either way).
    if "selected" in edge and not edge.get("selected"):
        score += 2.0
    gap = base_max_conf - conf
    if conf < LOW_CONF_ABS or gap >= LOW_CONF_REL_MARGIN:
        # Continuous term so, among low-confidence edges, the weakest ranks
        # first; ``+1`` guarantees a positive (flagged) score.
        score += 1.0 + max(0.0, gap)
    return score


def _flagged_edges(
    base_keys: list[tuple[str, str]],
    edge_data: dict[tuple[str, str], dict],
) -> list[tuple[str, str]]:
    """Ranked, bounded shortlist of the most-droppable edges within a base set.

    Returns at most ``MAX_FLAGGED_SINGLE_DROPS`` keys, most-droppable first
    (tie-break: lowest confidence, then id, for determinism). The sole edge of a
    single-edge base is never flagged (dropping it would empty the seed).
    """
    if len(base_keys) < 2:
        return []
    base_max = max(
        (float(edge_data.get(k, {}).get("confidence", 0.0) or 0.0) for k in base_keys),
        default=0.0,
    )
    scored: list[tuple[float, float, tuple[str, str]]] = []
    for k in base_keys:
        e = edge_data.get(k, {})
        s = _edge_droppability(e, base_max)
        if s > 0.0:
            conf = float(e.get("confidence", 0.0) or 0.0)
            scored.append((s, conf, k))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [k for _, _, k in scored[:MAX_FLAGGED_SINGLE_DROPS]]


def _seed_alternatives(
    edge_data: dict[tuple[str, str], dict],
    ref_ids: list[str],
    selected_keys: list[tuple[str, str]],
) -> list[dict]:
    """Whole-group + "base minus flagged edge(s)" seed alternatives.

    The whole-group seeds are the full candidate set and the optimizer's
    selection (edges flagged ``selected`` — populated even for giant M:N groups
    whose ``optimizer_assignment`` is empty). On top of those, for each base set
    (optimizer selection and full candidate set) with >= 2 edges, this emits
    "base minus the most-droppable edge(s)" seeds so a settled answer of the form
    "the optimizer set except that one bad edge" is directly expressible rather
    than depending on it ranking into the confidence-sorted top-K — the measured
    expressibility hole. See :func:`_edge_droppability` for the flagging signals
    and the module-level bounds for the caps.

    Every emitted seed is a STRICT, non-empty subset of the group's candidate
    edges (built from ``edge_data``), so it satisfies the same output invariant
    as the enumerated alternatives. All seeds are deduped against one another
    here (the caller additionally dedupes them against the organic top-K).

    Every seed is tagged ``is_seed: True`` so downstream consumers that score or
    rank the ORGANIC alternatives (e.g. ``select_stitching_batch``'s
    borderline / low-confidence tiers) can exclude seeds: the full-set seed is a
    superset of every proper assignment and would otherwise always win
    ``max(total_confidence)``, skewing selection.
    """
    seeds: list[dict] = []
    seen: set[frozenset[tuple[str, str]]] = set()

    def _alt(keys: list[tuple[str, str]]) -> dict:
        edges = [_make_edge(edge_data, rid, tid) for rid, tid in keys]
        total = round(sum(e["confidence"] for e in edges), 4)
        return {
            "edges": edges,
            "total_confidence": total,
            "summary": _build_summary(edges, ref_ids),
            "is_seed": True,
        }

    def _emit(keys: list[tuple[str, str]]) -> None:
        fkey = frozenset(keys)
        if not fkey or fkey in seen:
            return
        seen.add(fkey)
        seeds.append(_alt(list(keys)))

    all_keys = list(edge_data.keys())
    if all_keys:
        _emit(all_keys)
    if selected_keys:
        _emit(selected_keys)

    # "Base minus flagged edge(s)" seeds. Bases: the optimizer selection (a
    # correct answer is often the optimizer set minus a stray edge) and the full
    # candidate set (the correct answer is always a subset of it). Both bases are
    # deduped by ``_emit``, so when they coincide the removals are generated once.
    bases: list[list[tuple[str, str]]] = []
    if len(selected_keys) >= 2:
        bases.append(selected_keys)
    if len(all_keys) >= 2:
        bases.append(all_keys)

    for base in bases:
        flagged = _flagged_edges(base, edge_data)
        for fk in flagged:
            _emit([k for k in base if k != fk])
        # One combined seed dropping the top few flagged edges together, for the
        # "optimizer set minus a couple of slivers" answer.
        if len(flagged) >= 2:
            drop = set(flagged[:MAX_COMBINED_DROP])
            _emit([k for k in base if k not in drop])

    return seeds


def prune_group_options_to_edges(
    group: dict,
    ref_geoms: dict | None = None,
    target_geoms: dict | None = None,
) -> None:
    """Re-sync a group's precomputed options to its current edge set (in place).

    After a group's ``edges`` are reduced (e.g. clipped to a 500m display
    envelope by ``_fill_spatial_context``), any previously computed
    ``alternatives`` or ``optimizer_assignment`` may still reference edges that
    no longer exist in the group. This filters ``optimizer_assignment`` to the
    surviving edges and regenerates ``alternatives`` from them, so downstream
    consumers (review UI, evidence packs) only ever see options that are valid
    subsets of the group's own edges.
    """
    surviving = {(e["ref_id"], e["target_id"]) for e in group.get("edges", [])}
    if group.get("optimizer_assignment"):
        group["optimizer_assignment"] = [
            e
            for e in group["optimizer_assignment"]
            if (e.get("ref_id"), e.get("target_id")) in surviving
        ]
    if group.get("alternatives"):
        # Preserve the ORGANIC k only: seed options (is_seed=True) are appended
        # on top of k by the generator, so counting them here would grow k by
        # up to 2 on every re-sync.
        k = len([a for a in group["alternatives"] if not a.get("is_seed")]) or 5
        group["alternatives"] = generate_top_k_alternatives(
            group.get("edges", []),
            ref_geoms=ref_geoms if ref_geoms is not None else group.get("ref_geometries", {}),
            target_geoms=(
                target_geoms if target_geoms is not None else group.get("target_geometries", {})
            ),
            k=k,
        )


# ---------------------------------------------------------------------------
# Contiguous ref-chain construction
# ---------------------------------------------------------------------------


def _extract_coords(geom) -> list[tuple[float, float]] | None:
    """Pull 2D coordinates from a GeoJSON-style mapping or a shapely geometry."""
    if geom is None:
        return None
    coords = None
    if isinstance(geom, dict):
        coords = geom.get("coordinates")
        # Only simple LineStrings participate in contiguity.
        if geom.get("type") not in (None, "LineString"):
            return None
    elif hasattr(geom, "coords"):
        try:
            coords = list(geom.coords)
        except (NotImplementedError, TypeError):
            return None
    if not coords or len(coords) < 2:
        return None
    try:
        return [(float(c[0]), float(c[1])) for c in coords]
    except (TypeError, ValueError, IndexError):
        return None


def _metric_geom_lookup(geoms: dict[str, object] | None) -> dict[str, LineString]:
    """Convert ref geometries to shapely LineStrings in a metric frame.

    Group geometries are stored as WGS84 (lon/lat degrees); a meter-based
    contiguity tolerance is meaningless against degrees. Coordinates that fall
    within geographic bounds are reprojected to a local equirectangular meter
    frame (scaled at the collection's mean latitude); coordinates already in a
    projected (meter) frame are used as-is. This keeps the chain-contiguity
    check unit-consistent with the optimizer's endpoint-snap tolerance.
    """
    if not geoms:
        return {}

    parsed: dict[str, list[tuple[float, float]]] = {}
    lats: list[float] = []
    looks_geographic = True
    for gid, geom in geoms.items():
        coords = _extract_coords(geom)
        if coords is None:
            continue
        parsed[gid] = coords
        for x, y in coords:
            lats.append(y)
            if abs(x) > 180.0 or abs(y) > 90.0:
                looks_geographic = False

    if not parsed:
        return {}

    if looks_geographic and lats:
        lat0 = sum(lats) / len(lats)
        kx = 111_320.0 * math.cos(math.radians(lat0))
        ky = 110_540.0
    else:
        kx = ky = 1.0  # already projected — identity

    out: dict[str, LineString] = {}
    for gid, coords in parsed.items():
        out[gid] = LineString([(x * kx, y * ky) for x, y in coords])
    return out


def _enumerate_contiguous_chains(
    refs: list[str],
    adjacency: dict[str, set[str]],
    max_len: int,
    conf: dict[str, float] | None = None,
    max_frontier: int = MAX_CHAIN_FRONTIER,
) -> list[frozenset[str]]:
    """Enumerate connected ref subsets of size 2..max_len (contiguous chains).

    Grows connected subsets one node at a time from singletons, so every
    returned subset is contiguous under the endpoint-proximity adjacency.
    The frontier at each size is beam-limited to the top ``max_frontier``
    subsets by total confidence (deterministic tie-break by member ids), so
    construction stays bounded in dense adjacency graphs instead of
    materializing all O(n^max_len) connected subsets.
    """
    conf = conf or {}

    def _beam_key(subset: frozenset[str]) -> tuple:
        return (-sum(conf.get(r, 0.0) for r in subset), tuple(sorted(subset)))

    results: set[frozenset[str]] = set()
    # Frontier holds the connected subsets of the current size.
    frontier: list[frozenset[str]] = [frozenset([r]) for r in refs]
    for _size in range(2, max_len + 1):
        next_frontier: list[frozenset[str]] = []
        seen_this_size: set[frozenset[str]] = set()
        for subset in frontier:
            for node in subset:
                for neighbor in adjacency.get(node, ()):
                    if neighbor in subset:
                        continue
                    cand = subset | {neighbor}
                    if cand in results or cand in seen_this_size:
                        continue
                    seen_this_size.add(cand)
                    next_frontier.append(cand)
        next_frontier.sort(key=_beam_key)
        del next_frontier[max_frontier:]
        results.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return list(results)


def _target_ref_subsets(
    refs_for_t: list[str],
    tid: str,
    edge_data: dict[tuple[str, str], dict],
    ref_metric: dict[str, LineString],
) -> list[tuple[str, ...]]:
    """Build the ordered choice set of ref subsets for one target.

    Always includes every single ref that has an edge to ``tid``. When ref
    geometries are available, additionally includes contiguous chains of 2..
    MAX_REF_CHAIN_LEN refs (keeping the top MAX_CHAINS_PER_TARGET by total
    confidence). Every returned subset is a subset of the group's existing
    edges for this target.
    """
    # Import here to avoid a module-load cycle (optimizer imports many deps).
    from .optimizer import build_contiguity_adjacency

    def _conf(rid: str) -> float:
        return edge_data.get((rid, tid), {}).get("confidence", 0.0)

    # Singletons, highest confidence first for deterministic ordering.
    singles = sorted(refs_for_t, key=lambda r: (-_conf(r), r))
    subsets: list[tuple[str, ...]] = [(r,) for r in singles]

    if len(refs_for_t) < 2 or not ref_metric:
        return subsets

    geom_lookup = {r: ref_metric[r] for r in refs_for_t if r in ref_metric}
    if len(geom_lookup) < 2:
        return subsets

    adjacency = build_contiguity_adjacency(
        list(geom_lookup.keys()), geom_lookup, CHAIN_CONTIGUITY_TOLERANCE_M
    )
    chains = _enumerate_contiguous_chains(
        list(geom_lookup.keys()),
        adjacency,
        MAX_REF_CHAIN_LEN,
        conf={r: _conf(r) for r in geom_lookup},
    )

    def _chain_conf(chain: frozenset[str]) -> float:
        return sum(_conf(r) for r in chain)

    # Highest total-confidence chains first, capped to bound the option count.
    chains.sort(key=lambda c: (-_chain_conf(c), tuple(sorted(c))))
    for chain in chains[:MAX_CHAINS_PER_TARGET]:
        subsets.append(tuple(sorted(chain)))

    return subsets


def _make_edge(edge_data: dict[tuple[str, str], dict], rid: str, tid: str) -> dict:
    """Build an output edge dict from the source edge data, preserving alignment fracs."""
    source = edge_data.get((rid, tid), {})
    edge = {
        "ref_id": rid,
        "target_id": tid,
        "confidence": round(source.get("confidence", 0.0), 4),
    }
    for key in _ALIGNMENT_KEYS:
        if key in source:
            edge[key] = source[key]
    return edge


def _edges_from_subset(
    edge_data: dict[tuple[str, str], dict],
    tid: str,
    subset: tuple[str, ...] | None,
) -> tuple[list[dict], float]:
    """Expand a (target, ref-subset) choice into edges + summed confidence."""
    if subset is None:
        return [], 0.0
    edges = []
    total = 0.0
    for rid in subset:
        edge = _make_edge(edge_data, rid, tid)
        edges.append(edge)
        total += edge["confidence"]
    return edges, total


def _exhaustive_enumeration(
    target_ids: list[str],
    target_options: dict[str, list[tuple[str, ...] | None]],
    edge_data: dict[tuple[str, str], dict],
    ref_ids: list[str],
) -> list[dict]:
    """Enumerate all valid assignment combos via itertools.product.

    Each combo entry is a ref subset (a 1-tuple for a single ref, or a longer
    tuple for a contiguous chain) or None ("unassigned").
    """
    option_lists = [target_options[tid] for tid in target_ids]
    alternatives = []

    for combo in itertools.product(*option_lists):
        edges = []
        total_conf = 0.0
        for tid, subset in zip(target_ids, combo):
            sub_edges, sub_conf = _edges_from_subset(edge_data, tid, subset)
            edges.extend(sub_edges)
            total_conf += sub_conf

        # Skip empty assignments
        if not edges:
            continue

        summary = _build_summary(edges, ref_ids)
        alternatives.append(
            {
                "edges": edges,
                "total_confidence": round(total_conf, 4),
                "summary": summary,
            }
        )

    return alternatives


def _enumerate_n_to_1(
    ref_ids: list[str],
    target_id: str,
    edge_data: dict[tuple[str, str], dict],
    k: int,
) -> list[dict]:
    """Enumerate N:1 alternatives: each ref independently maps to the target or not.

    For N refs, there are 2^N - 1 non-empty subsets. Enumerate all for N <= 10,
    otherwise use greedy + perturbation. This path already enumerates the full
    ref power set, so a ref-side multi-span (the mirror of a target spanning
    multiple refs) is already expressible — no chain machinery needed here.
    """
    n = len(ref_ids)
    alternatives = []

    if n <= MAX_EXHAUSTIVE_N_TO_1_REFS:
        # Enumerate all non-empty subsets of refs
        for mask in range(1, 1 << n):
            edges = []
            total_conf = 0.0
            for i, rid in enumerate(ref_ids):
                if mask & (1 << i):
                    edge = _make_edge(edge_data, rid, target_id)
                    edges.append(edge)
                    total_conf += edge["confidence"]
            summary = _build_summary(edges, ref_ids)
            alternatives.append(
                {"edges": edges, "total_confidence": round(total_conf, 4), "summary": summary}
            )
    else:
        # Greedy: include all refs, then generate perturbations by dropping each one
        all_edges = [_make_edge(edge_data, rid, target_id) for rid in ref_ids]
        total = sum(e["confidence"] for e in all_edges)
        alternatives.append(
            {
                "edges": list(all_edges),
                "total_confidence": round(total, 4),
                "summary": _build_summary(all_edges, ref_ids),
            }
        )
        # Drop each ref one at a time
        for i in range(len(ref_ids)):
            subset = [e for j, e in enumerate(all_edges) if j != i]
            sub_conf = sum(e["confidence"] for e in subset)
            alternatives.append(
                {
                    "edges": subset,
                    "total_confidence": round(sub_conf, 4),
                    "summary": _build_summary(subset, ref_ids),
                }
            )

    return alternatives


def _greedy_perturbation(
    target_ids: list[str],
    target_options: dict[str, list[tuple[str, ...] | None]],
    edge_data: dict[tuple[str, str], dict],
    ref_ids: list[str],
    max_alternatives: int = 30,
) -> list[dict]:
    """Generate alternatives via greedy assignment + perturbations.

    Starts with greedy (best single ref per target), then perturbs each
    target's assignment across its full option set — which now includes
    contiguous multi-ref chains — so large groups can also propose multi-ref
    edges rather than staying single-ref blind.
    """
    alternatives = []

    def _best_single(tid: str) -> tuple[str, ...] | None:
        """Highest-confidence single-ref option for a target (or None)."""
        best_rid = None
        best_conf = -1.0
        for opt in target_options[tid]:
            if opt is None or len(opt) != 1:
                continue
            conf = edge_data.get((opt[0], tid), {}).get("confidence", 0.0)
            if conf > best_conf:
                best_conf = conf
                best_rid = opt[0]
        return (best_rid,) if best_rid is not None else None

    # Greedy assignment: for each target, pick highest-confidence single ref.
    greedy_assignment: dict[str, tuple[str, ...] | None] = {
        tid: _best_single(tid) for tid in target_ids
    }

    # Add greedy as first alternative
    _add_alternative(greedy_assignment, target_ids, edge_data, ref_ids, alternatives)

    # Perturb: for each target, try each alternative option (single ref, a
    # contiguous chain, or None).
    for tid in target_ids:
        current = greedy_assignment[tid]
        for alt_opt in target_options[tid]:
            if alt_opt == current:
                continue
            perturbed = dict(greedy_assignment)
            perturbed[tid] = alt_opt
            _add_alternative(perturbed, target_ids, edge_data, ref_ids, alternatives)
            if len(alternatives) >= max_alternatives:
                return alternatives

    # Pairwise perturbation: swap two targets' assignments (only when every
    # edge in the swapped subset exists for its new target).
    for i, tid1 in enumerate(target_ids):
        for tid2 in target_ids[i + 1 :]:
            perturbed = dict(greedy_assignment)
            perturbed[tid1], perturbed[tid2] = perturbed[tid2], perturbed[tid1]
            if _assignment_valid(perturbed[tid1], tid1, edge_data) and _assignment_valid(
                perturbed[tid2], tid2, edge_data
            ):
                _add_alternative(perturbed, target_ids, edge_data, ref_ids, alternatives)
            if len(alternatives) >= max_alternatives:
                return alternatives

    return alternatives


def _assignment_valid(
    subset: tuple[str, ...] | None,
    tid: str,
    edge_data: dict[tuple[str, str], dict],
) -> bool:
    """True if every ref in the subset has an existing edge to the target."""
    if subset is None:
        return True
    return all((rid, tid) in edge_data for rid in subset)


def _add_alternative(
    assignment: dict[str, tuple[str, ...] | None],
    target_ids: list[str],
    edge_data: dict[tuple[str, str], dict],
    ref_ids: list[str],
    alternatives: list[dict],
) -> None:
    """Convert an assignment dict to an alternative and append it."""
    edges = []
    total_conf = 0.0

    for tid in target_ids:
        sub_edges, sub_conf = _edges_from_subset(edge_data, tid, assignment[tid])
        edges.extend(sub_edges)
        total_conf += sub_conf

    if not edges:
        return

    summary = _build_summary(edges, ref_ids)
    alternatives.append(
        {
            "edges": edges,
            "total_confidence": round(total_conf, 4),
            "summary": summary,
        }
    )


def _sanitize_alternative(
    alt: dict,
    valid_keys: set[tuple[str, str]],
    ref_ids: list[str],
) -> dict:
    """Enforce the output invariant on a single alternative.

    Drops any edge whose (ref_id, target_id) is not one of the group's
    candidate edges, removes duplicate edges, and recomputes total_confidence
    and summary from the surviving edge set. A violation (invalid or duplicate
    edge) is logged so it can never be shipped silently.
    """
    clean_edges: list[dict] = []
    seen: set[tuple[str, str]] = set()
    dropped = 0
    for e in alt.get("edges", []):
        key = (e.get("ref_id"), e.get("target_id"))
        if key not in valid_keys or key in seen:
            dropped += 1
            continue
        seen.add(key)
        clean_edges.append(e)

    if dropped:
        logger.warning(
            "generate_top_k_alternatives: dropped %d invalid/duplicate edge(s) "
            "from an alternative (kept %d valid subset edges)",
            dropped,
            len(clean_edges),
        )

    total = round(sum(e.get("confidence", 0.0) for e in clean_edges), 4)
    return {
        "edges": clean_edges,
        "total_confidence": total,
        "summary": _build_summary(clean_edges, ref_ids) if clean_edges else "",
    }


def _build_summary(edges: list[dict], ref_ids: list[str]) -> str:
    """Build a human-readable summary of an assignment.

    Format: "ref_A -> tgt_1+tgt_2, ref_B -> tgt_3"
    """
    by_ref: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        by_ref[e["ref_id"]].append(e["target_id"])

    parts = []
    for rid in sorted(by_ref.keys()):
        tids = sorted(by_ref[rid])
        # Shorten IDs for readability
        rid_short = _shorten_id(rid)
        tid_parts = "+".join(_shorten_id(t) for t in tids)
        parts.append(f"{rid_short} -> {tid_parts}")

    return ", ".join(parts)


def _shorten_id(id_str: str) -> str:
    """Shorten an ID for display.

    UUIDs (ref IDs): show first 8 chars (e.g., "c6c8d93d...")
    Target IDs (dataset_numeric_h3): strip dataset prefix, show from first
    numeric segment (e.g., "4696_882a...")
    """
    if len(id_str) <= 12:
        return id_str
    # UUID pattern: 8-4-4-4-12 hex with dashes
    if len(id_str) == 36 and id_str.count("-") == 4:
        return id_str[:8] + "..."
    # Target ID: find first segment starting with a digit
    parts = id_str.split("_")
    for i, part in enumerate(parts):
        if part and part[0].isdigit():
            suffix = "_".join(parts[i:])
            if len(suffix) > 12:
                return suffix[:10] + "..."
            return suffix
    # Fallback: prefix
    return id_str[:10] + "..."
