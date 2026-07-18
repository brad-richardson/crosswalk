"""Shared assignment-option builder for stitching groups.

A stitching group's optimizer assignment and pre-computed top-K alternatives
are turned into a deduplicated set of one-click "options" ("verify, don't
construct"). Both the web review UI and the agent evidence-pack generator use
this single implementation so humans and agents pick from the identical option
set.

Each option carries a ``letter`` (A, B, C, ...) for stable, provider-agnostic
referencing, plus its exact edge set (enriched with per-edge confidence and
alignment fractions from the group's own edges).
"""

import string

from ..utils.physical import physical_is_informative

_ALIGN_KEYS = ("gers_start_frac", "gers_end_frac", "local_start_frac", "local_end_frac")

# Per-edge structural features (#267) persisted in every groups sidecar. Passed
# through verbatim so evidence packs can surface them (a degree-4 junction
# endpoint / graph cut-edge / corridor membership explains a junction-kiss far better
# than a bare confidence number). Older sidecar vintages may lack some or all of
# these; missing keys are simply omitted (never defaulted), so downstream
# display degrades gracefully rather than fabricating structure.
_STRUCT_KEYS = (
    "degree_ref",
    "degree_tgt",
    "candidate_graph_bridge",
    "biconnected_block",
    "corridor_ref",
    "corridor_tgt",
    "ref_physical",
    "target_physical",
    # Per-edge lateral-offset evidence (MI-4 physical-separation trigger).
    "lateral_offset_m",
    "lateral_offset_p95_m",
    "offset_over_expected_halfwidth",
)

# Decision provenance is audit context, not an input to option construction.
# Preserve it on the exact displayed edge so the committed panel evidence can
# distinguish a production MATCH from a conservative REVIEW/demotion later.
_DECISION_KEYS = (
    "selected",
    "decision",
    "review_reason",
    "optimizer_decision",
    "decision_reason",
    "pruned",
    "selected_elsewhere",
)


def build_stitch_options(group: dict) -> dict:
    """Build the assignment option picker + optimizer pre-seed for a group.

    The optimizer's own proposed assignment and the pre-computed top-K
    alternatives are turned into one-click options, deduplicated by exact edge
    set (optimizer first, then alternatives).

    Returns a context dict with:
    - options: list of option dicts. Each has ``letter`` (A, B, ...), ``key``,
      ``label``, ``is_optimizer``, ``edges`` (list of enriched
      {ref_id, target_id, confidence, *fracs, *#267 structural fields}),
      ``edge_count``,
      ``total_confidence``, ``mean_confidence``, ``active_refs``,
      ``active_targets``.
    - optimizer_letter: the letter of the optimizer's option, or None.
    - preseed_active_refs / preseed_active_targets / preseed_inactive_ids /
      preseed_edges / has_preseed: UI pre-seed state derived from the optimizer
      assignment (None / empty when no optimizer assignment is present).
    """
    group_edges = group.get("edges", []) or []
    # Full edge lookup: (ref_id, target_id) -> edge dict (confidence + fracs).
    edge_lookup: dict[tuple[str, str], dict] = {}
    for e in group_edges:
        edge_lookup[(e["ref_id"], e["target_id"])] = e
    group_edge_set = set(edge_lookup.keys())

    optimizer = group.get("optimizer_assignment") or []
    alternatives = group.get("alternatives") or []

    def _valid_edges(edges: list[dict]) -> list[dict]:
        """Keep only edges in the group, deduped; enrich with confidence+fracs."""
        out = []
        seen = set()
        for e in edges:
            key = (e.get("ref_id"), e.get("target_id"))
            if key in group_edge_set and key not in seen:
                seen.add(key)
                src = edge_lookup[key]
                enriched = {
                    "ref_id": key[0],
                    "target_id": key[1],
                    "confidence": round(float(src.get("confidence", 0.0)), 4),
                }
                for ak in _ALIGN_KEYS:
                    if ak in src:
                        enriched[ak] = src[ak]
                # Pass through #267 structural features when present; omit missing
                # ones so older sidecar vintages degrade gracefully.
                for sk in _STRUCT_KEYS:
                    if sk in src:
                        enriched[sk] = src[sk]
                # Historical sidecars used the ambiguous name ``is_bridge`` for
                # a graph-theory cut edge. Canonicalize it while reading; never
                # pass that name into new human/agent evidence.
                if "candidate_graph_bridge" not in enriched and "is_bridge" in src:
                    enriched["candidate_graph_bridge"] = bool(src["is_bridge"])
                # Keep neutral ground/no-flag rules in the sidecar audit map, but
                # do not repeat them on every option edge unless the opposite
                # side makes the physical comparison informative.
                if not (
                    physical_is_informative(enriched.get("ref_physical"))
                    or physical_is_informative(enriched.get("target_physical"))
                ):
                    enriched.pop("ref_physical", None)
                    enriched.pop("target_physical", None)
                for dk in _DECISION_KEYS:
                    if dk in src:
                        enriched[dk] = src[dk]
                out.append(enriched)
        return out

    def _edge_key(edges: list[dict]) -> frozenset:
        return frozenset((e["ref_id"], e["target_id"]) for e in edges)

    def _confidences(valid_edges: list[dict]) -> tuple[float, float]:
        # Compute over the already-validated, deduplicated edge set (the exact
        # edges displayed). Summing over raw edges would double-count any
        # duplicated or out-of-group edge and inflate the displayed confidence.
        confs = [e.get("confidence", 0.0) for e in valid_edges]
        total = round(sum(confs), 4)
        mean = round(total / len(confs), 4) if confs else 0.0
        return total, mean

    def _make_option(key: str, label: str, is_optimizer: bool, raw_edges: list[dict]) -> dict:
        edges = _valid_edges(raw_edges)
        total, mean = _confidences(edges)
        return {
            "key": key,
            "label": label,
            "is_optimizer": is_optimizer,
            "edges": edges,
            "edge_count": len(edges),
            "total_confidence": total,
            "mean_confidence": mean,
            "active_refs": sorted({e["ref_id"] for e in edges}),
            "active_targets": sorted({e["target_id"] for e in edges}),
        }

    options: list[dict] = []
    seen: set[frozenset] = set()

    if optimizer:
        opt = _make_option("optimizer", "Optimizer", True, optimizer)
        if opt["edges"]:
            options.append(opt)
            seen.add(_edge_key(opt["edges"]))

    alt_num = 0
    for alt in alternatives:
        edges = _valid_edges(alt.get("edges", []))
        if not edges:
            continue
        key = _edge_key(edges)
        if key in seen:
            continue
        seen.add(key)
        alt_num += 1
        # total/mean are computed inside _make_option from the validated,
        # deduplicated edge set. We deliberately do NOT trust a stored
        # ``alt["total_confidence"]`` here: if an alternative's edge list was
        # ever built against a different (e.g. pre-clip) group, its stored total
        # would be inflated relative to the edges actually displayed.
        opt = _make_option(f"alt{alt_num}", f"Alt {alt_num}", False, alt.get("edges", []))
        options.append(opt)

    # Assign stable letters and record the optimizer's letter.
    letters = list(string.ascii_uppercase)
    optimizer_letter = None
    for i, opt in enumerate(options):
        letter = letters[i] if i < len(letters) else f"O{i}"
        opt["letter"] = letter
        if opt["is_optimizer"]:
            optimizer_letter = letter

    # Pre-seed pill active-state from the optimizer assignment.
    preseed_refs = None
    preseed_targets = None
    preseed_inactive_ids: list[str] = []
    preseed_valid = _valid_edges(optimizer)
    if preseed_valid:
        preseed_refs = sorted({e["ref_id"] for e in preseed_valid})
        preseed_targets = sorted({e["target_id"] for e in preseed_valid})
        active_ids = set(preseed_refs) | set(preseed_targets)
        for sid in group.get("ref_ids", []) + group.get("target_ids", []):
            if sid not in active_ids:
                preseed_inactive_ids.append(sid)

    preseed_edges = options[0]["edges"] if (options and options[0]["is_optimizer"]) else []

    return {
        "options": options,
        "optimizer_letter": optimizer_letter,
        "preseed_active_refs": preseed_refs,
        "preseed_active_targets": preseed_targets,
        "preseed_inactive_ids": preseed_inactive_ids,
        "preseed_edges": preseed_edges,
        "has_preseed": bool(preseed_refs) or bool(preseed_targets),
    }
