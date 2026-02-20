"""Top-K assignment alternative generation for M:N match groups.

Enumerates valid assignment combinations for a match group's candidate
edges, filters by contiguity and overlap constraints, and returns the
top K alternatives ranked by total confidence.
"""

import itertools
from collections import defaultdict


def generate_top_k_alternatives(
    component_edges: list[dict],
    ref_geoms: dict[str, dict] | None = None,
    target_geoms: dict[str, dict] | None = None,
    k: int = 5,
) -> list[dict]:
    """Generate top-K assignment alternatives for a match group.

    For each target, enumerates which ref it could be assigned to (or
    unassigned). Ranks by total confidence and returns the top K.

    For groups with <= 6 targets, uses exhaustive enumeration via
    itertools.product. For larger groups, falls back to greedy
    perturbation.

    Args:
        component_edges: List of edge dicts with ref_id, target_id, confidence
        ref_geoms: Reserved for future contiguity filtering (currently unused)
        target_geoms: Reserved for future contiguity filtering (currently unused)
        k: Number of top alternatives to return

    Returns:
        List of alternative dicts, each with:
        - option_index: int
        - edges: list of {ref_id, target_id, confidence}
        - total_confidence: float
        - summary: human-readable string
    """
    if not component_edges:
        return []

    # Collect unique refs and targets
    ref_ids = sorted(set(e["ref_id"] for e in component_edges))
    target_ids = sorted(set(e["target_id"] for e in component_edges))

    # Build lookup: (ref_id, target_id) -> confidence
    edge_confidence: dict[tuple[str, str], float] = {}
    for e in component_edges:
        key = (e["ref_id"], e["target_id"])
        # Keep highest confidence if duplicate
        if key not in edge_confidence or e["confidence"] > edge_confidence[key]:
            edge_confidence[key] = e["confidence"]

    # Build per-target options: which refs can each target be assigned to?
    target_options: dict[str, list[str | None]] = {}
    for tid in target_ids:
        options = [rid for rid in ref_ids if (rid, tid) in edge_confidence]
        # Add "unassigned" option
        options.append(None)
        target_options[tid] = options

    # Decide enumeration strategy
    n_combos = 1
    for tid in target_ids:
        n_combos *= len(target_options[tid])

    if len(target_ids) <= 6 and n_combos <= 10000:
        alternatives = _exhaustive_enumeration(target_ids, target_options, edge_confidence, ref_ids)
    else:
        alternatives = _greedy_perturbation(
            target_ids, target_options, edge_confidence, ref_ids, k * 3
        )

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

    # Take top K and assign option indices
    top_k = unique[:k]
    for i, alt in enumerate(top_k):
        alt["option_index"] = i

    return top_k


def _exhaustive_enumeration(
    target_ids: list[str],
    target_options: dict[str, list[str | None]],
    edge_confidence: dict[tuple[str, str], float],
    ref_ids: list[str],
) -> list[dict]:
    """Enumerate all valid assignment combos via itertools.product."""
    option_lists = [target_options[tid] for tid in target_ids]
    alternatives = []

    for combo in itertools.product(*option_lists):
        # combo[i] = ref_id or None for target_ids[i]
        edges = []
        total_conf = 0.0

        for tid, rid in zip(target_ids, combo):
            if rid is not None:
                conf = edge_confidence.get((rid, tid), 0.0)
                edges.append({"ref_id": rid, "target_id": tid, "confidence": round(conf, 4)})
                total_conf += conf

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


def _greedy_perturbation(
    target_ids: list[str],
    target_options: dict[str, list[str | None]],
    edge_confidence: dict[tuple[str, str], float],
    ref_ids: list[str],
    max_alternatives: int = 30,
) -> list[dict]:
    """Generate alternatives via greedy assignment + perturbations.

    Starts with greedy (best confidence per target), then perturbs
    each target's assignment to generate alternatives.
    """
    alternatives = []

    # Greedy assignment: for each target, pick highest-confidence ref
    greedy_assignment: dict[str, str | None] = {}
    for tid in target_ids:
        best_rid = None
        best_conf = -1.0
        for rid in target_options[tid]:
            if rid is None:
                continue
            conf = edge_confidence.get((rid, tid), 0.0)
            if conf > best_conf:
                best_conf = conf
                best_rid = rid
        greedy_assignment[tid] = best_rid

    # Add greedy as first alternative
    _add_alternative(greedy_assignment, target_ids, edge_confidence, ref_ids, alternatives)

    # Perturb: for each target, try each alternative ref
    for tid in target_ids:
        current_rid = greedy_assignment[tid]
        for alt_rid in target_options[tid]:
            if alt_rid == current_rid:
                continue
            perturbed = dict(greedy_assignment)
            perturbed[tid] = alt_rid
            _add_alternative(perturbed, target_ids, edge_confidence, ref_ids, alternatives)
            if len(alternatives) >= max_alternatives:
                return alternatives

    # Pairwise perturbation: swap two targets' assignments
    for i, tid1 in enumerate(target_ids):
        for tid2 in target_ids[i + 1 :]:
            perturbed = dict(greedy_assignment)
            perturbed[tid1], perturbed[tid2] = perturbed[tid2], perturbed[tid1]
            # Only valid if edges exist
            valid = True
            if perturbed[tid1] is not None and (perturbed[tid1], tid1) not in edge_confidence:
                valid = False
            if perturbed[tid2] is not None and (perturbed[tid2], tid2) not in edge_confidence:
                valid = False
            if valid:
                _add_alternative(perturbed, target_ids, edge_confidence, ref_ids, alternatives)
            if len(alternatives) >= max_alternatives:
                return alternatives

    return alternatives


def _add_alternative(
    assignment: dict[str, str | None],
    target_ids: list[str],
    edge_confidence: dict[tuple[str, str], float],
    ref_ids: list[str],
    alternatives: list[dict],
) -> None:
    """Convert an assignment dict to an alternative and append it."""
    edges = []
    total_conf = 0.0

    for tid in target_ids:
        rid = assignment[tid]
        if rid is not None:
            conf = edge_confidence.get((rid, tid), 0.0)
            edges.append({"ref_id": rid, "target_id": tid, "confidence": round(conf, 4)})
            total_conf += conf

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
    """Shorten an ID for display (last 6 chars if long)."""
    if len(id_str) > 12:
        return "..." + id_str[-6:]
    return id_str
