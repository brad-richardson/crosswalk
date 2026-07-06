"""Batch selection logic for stitching review.

Scores and selects a curated batch of M:N groups for human review,
prioritizing large groups, borderline cases where the optimizer's
assignment is uncertain, and low-confidence groups where the best
assignment is likely wrong.
"""

from loguru import logger

# Minimum edge count to qualify as a "large" group.
LARGE_GROUP_MIN_EDGES = 10

# Tier fractions control batch composition.
# 30% large: groups with 10+ edges (complex M:N merges)
# 30% borderline: groups where top-2 alternatives are close in confidence
# 20% low confidence: groups where best alternative has low avg confidence
# 20% clear winner: calibration groups where one alternative clearly dominates
TIER_LARGE_FRAC = 0.3
TIER_BORDERLINE_FRAC = 0.3
TIER_LOW_CONF_FRAC = 0.2
TIER_CLEAR_FRAC = 0.2

# A group only qualifies for the *borderline* tier when its distinguishing edge
# is genuinely uncertain. The borderline score is the per-edge contestedness of
# the most-contested edge in the symmetric difference of the top-2 alternatives
# (see ``compute_borderline_score``). A score of 0.2 corresponds to a
# distinguishing edge whose optimizer confidence lies inside ~[0.1, 0.9] — below
# that the optimizer is effectively certain about the edge that separates the two
# structures, so the choice is NOT a coin flip and does not belong in the
# borderline tier. This is the guard that keeps clean long 1:N chains (whose
# runner-up alternative differs by a single ~0.99-confidence edge) out of the
# human queue.
BORDERLINE_MIN_SCORE = 0.2


def _pair_confidence_map(alternative: dict) -> dict[tuple, float]:
    """Map each ``(ref_id, target_id)`` edge of an alternative to its confidence."""
    out: dict[tuple, float] = {}
    for e in alternative.get("edges", []):
        out[(e.get("ref_id"), e.get("target_id"))] = float(e.get("confidence", 0.0))
    return out


def compute_borderline_score(alternatives: list[dict]) -> float:
    """Per-edge contestedness of the choice between the top-2 alternatives.

    The old metric scored ``1 - |top1 - top2| / top1`` over the *summed*
    ``total_confidence`` of the two best alternatives. For a clean chain of N
    high-confidence edges the runner-up is the same chain minus one edge, so the
    score was ~``1 - 1/N`` — longer chains looked more borderline regardless of
    how certain every edge was. That length bias sampled crystal-clear 1:N chains
    into the human queue as if they were coin flips.

    This metric instead looks at the SYMMETRIC DIFFERENCE of the top-2
    alternatives' edge sets — the edges the two structures actually disagree
    about — and asks how uncertain the optimizer is about each. An edge with
    confidence ``c`` has contestedness ``1 - |2c - 1|``: it peaks at 1.0 when
    ``c = 0.5`` (a true coin flip) and falls to 0.0 as ``c`` approaches 0 or 1
    (the optimizer is sure the edge does / does not belong). We take the MAX over
    the differing edges — "is there a genuinely contested edge that distinguishes
    the two structures?" — rather than a sum or mean, because summing would
    re-introduce the length bias (a chain differing by five certain edges would
    outscore a single 0.5-confidence swap) and a mean would dilute one real coin
    flip among many certain edges.

    Returns 0.0 when there are fewer than two alternatives or the top two share
    an identical edge set (nothing to be borderline about).
    """
    if len(alternatives) < 2:
        return 0.0
    sorted_alts = sorted(alternatives, key=lambda a: a["total_confidence"], reverse=True)
    m1 = _pair_confidence_map(sorted_alts[0])
    m2 = _pair_confidence_map(sorted_alts[1])
    sym_diff = set(m1) ^ set(m2)
    if not sym_diff:
        return 0.0
    best = 0.0
    for key in sym_diff:
        conf = m1[key] if key in m1 else m2[key]
        contestedness = 1.0 - abs(2.0 * conf - 1.0)
        if contestedness > best:
            best = contestedness
    return best


def select_stitching_batch(
    groups: list[dict],
    reviewed_group_ids: set[str],
    k: int = 15,
    candidate_group_ids: set[str] | None = None,
) -> list[dict]:
    """Select a curated batch of groups for stitching review.

    Scores each unreviewed group and selects with tier-based balancing:
    - ~30% (large): groups with 10+ edges (complex M:N merges)
    - ~30% (borderline): groups where the top-2 alternatives disagree about a
      genuinely contested edge (see ``compute_borderline_score``)
    - ~20% (low_confidence): groups where best alternative has low avg confidence
    - ~20% (clear_winner): groups where one alternative clearly dominates

    Args:
        groups: List of group dicts from the groups sidecar, each must have
            pre-computed 'alternatives' (from generate_top_k_alternatives)
        reviewed_group_ids: Set of group_ids already reviewed
        k: Target batch size
        candidate_group_ids: When provided, only groups whose ``group_id`` is in
            this set are eligible for selection (already-reviewed groups are still
            excluded on top of this). Used to gate the human review queue to the
            groups that FAILED the agent panel (routed to ``human_review``); pass
            ``None`` to score every unreviewed group (the panel-feed / legacy
            sampling behavior).

    Returns:
        List of group dicts selected for review, ordered by review value,
        each annotated with 'review_tier' and 'review_score'
    """
    # Score each unreviewed group
    scored_groups: list[dict] = []
    for group in groups:
        gid = group.get("group_id", "")
        if gid in reviewed_group_ids:
            continue
        # Human-queue gating: when a candidate allow-list is supplied, only those
        # groups (e.g. the ones the agent panel routed to human_review) are
        # eligible. Panel-feed / legacy sampling passes None to consider every
        # unreviewed group.
        if candidate_group_ids is not None and gid not in candidate_group_ids:
            continue

        # Score on ORGANIC alternatives only: whole-group seed options
        # (is_seed=True, e.g. the full candidate set) are supersets of proper
        # assignments, so their summed total_confidence would always win
        # max()/top-2 and skew the borderline / low-confidence tiers. Fall back
        # to all alternatives if only seeds exist (defensive; the generator
        # always emits >=1 organic alternative when the group has edges).
        all_alternatives = group.get("alternatives", [])
        alternatives = [a for a in all_alternatives if not a.get("is_seed")] or all_alternatives
        edges = group.get("edges", [])
        n_edges = len(edges)

        # Borderline score: per-edge contestedness of the top-2 alternatives'
        # disagreement (symmetric difference of their edge sets), NOT the old
        # length-biased summed-confidence ratio. See compute_borderline_score.
        borderline_score = compute_borderline_score(alternatives)

        # Low confidence score: how weak the best alternative is.
        # Uses average per-edge confidence of the top alternative.
        # Score is inverted (1.0 = lowest confidence = most worth reviewing).
        if alternatives:
            best_alt = max(alternatives, key=lambda a: a["total_confidence"])
            n_alt_edges = len(best_alt.get("edges", []))
            avg_edge_conf = best_alt["total_confidence"] / n_alt_edges if n_alt_edges > 0 else 0.0
            low_conf_score = 1.0 - avg_edge_conf
        else:
            low_conf_score = 1.0  # no alternatives at all = very suspect

        # Composite review value (used for sorting within tiers)
        review_value = borderline_score + low_conf_score

        scored_groups.append(
            {
                **group,
                "_n_edges": n_edges,
                "_borderline_score": borderline_score,
                "_low_conf_score": low_conf_score,
                "_review_value": review_value,
            }
        )

    if not scored_groups:
        return []

    # Tier-based selection
    n_large = max(1, int(k * TIER_LARGE_FRAC))
    n_borderline = max(1, int(k * TIER_BORDERLINE_FRAC))
    n_low_conf = max(1, int(k * TIER_LOW_CONF_FRAC))
    n_clear = max(0, k - n_large - n_borderline - n_low_conf)

    selected: list[dict] = []
    used_ids: set[str] = set()

    # Tier 1: large groups — always include the single largest group,
    # then fill remaining slots from groups with 10+ edges
    by_size = sorted(scored_groups, key=lambda g: g["_n_edges"], reverse=True)
    large_filled = 0

    # Always pick the largest group
    if by_size:
        g = by_size[0]
        g["review_tier"] = "large"
        g["review_score"] = round(g["_review_value"], 3)
        selected.append(g)
        used_ids.add(g["group_id"])
        large_filled += 1

    # Fill remaining large slots from 10+ edge groups
    for g in by_size[1:]:
        if large_filled >= n_large:
            break
        gid = g["group_id"]
        if gid not in used_ids and g["_n_edges"] >= LARGE_GROUP_MIN_EDGES:
            g["review_tier"] = "large"
            g["review_score"] = round(g["_review_value"], 3)
            selected.append(g)
            used_ids.add(gid)
            large_filled += 1

    # Redistribute unfilled large slots to borderline
    n_borderline += n_large - large_filled

    # Tier 2: highest borderline score (from remaining). Only groups whose
    # distinguishing edge is genuinely contested (score >= BORDERLINE_MIN_SCORE)
    # are eligible — a clean chain whose runner-up differs by one near-certain
    # edge must never be sampled here just to fill the tier. When too few groups
    # clear the bar the tier simply stays short (a smaller, higher-signal queue),
    # which is the intended behavior.
    by_borderline = sorted(scored_groups, key=lambda g: g["_borderline_score"], reverse=True)
    count = 0
    for g in by_borderline:
        if count >= n_borderline:
            break
        if g["_borderline_score"] < BORDERLINE_MIN_SCORE:
            break
        gid = g["group_id"]
        if gid not in used_ids:
            g["review_tier"] = "borderline"
            g["review_score"] = round(g["_review_value"], 3)
            selected.append(g)
            used_ids.add(gid)
            count += 1

    # Tier 3: highest low-confidence score (from remaining)
    by_low_conf = sorted(scored_groups, key=lambda g: g["_low_conf_score"], reverse=True)
    count = 0
    for g in by_low_conf:
        if count >= n_low_conf:
            break
        gid = g["group_id"]
        if gid not in used_ids:
            g["review_tier"] = "low_confidence"
            g["review_score"] = round(g["_review_value"], 3)
            selected.append(g)
            used_ids.add(gid)
            count += 1

    # Tier 4: clear winners (lowest borderline score, for calibration)
    by_clear = sorted(scored_groups, key=lambda g: g["_borderline_score"])
    count = 0
    for g in by_clear:
        if count >= n_clear:
            break
        gid = g["group_id"]
        if gid not in used_ids:
            g["review_tier"] = "clear_winner"
            g["review_score"] = round(g["_review_value"], 3)
            selected.append(g)
            used_ids.add(gid)
            count += 1

    # Backfill to k. The borderline tier now refuses non-contested groups
    # (BORDERLINE_MIN_SCORE), so when few groups are genuinely borderline its
    # slots go unfilled and the queue would fall short of k even though other
    # reviewable groups remain. Top up from the highest-review-value leftovers so
    # the queue still reaches k whenever enough material exists. These are the
    # uncontested remainder, so they are tagged "clear_winner" (calibration).
    # Nothing is padded beyond the eligible pool, so a gated (panel-failure) pool
    # smaller than k still yields a correspondingly short queue.
    if len(selected) < k:
        by_value = sorted(scored_groups, key=lambda g: g["_review_value"], reverse=True)
        for g in by_value:
            if len(selected) >= k:
                break
            gid = g["group_id"]
            if gid not in used_ids:
                g["review_tier"] = "clear_winner"
                g["review_score"] = round(g["_review_value"], 3)
                selected.append(g)
                used_ids.add(gid)

    # Clean up internal scoring keys
    for g in selected:
        g.pop("_n_edges", None)
        g.pop("_borderline_score", None)
        g.pop("_low_conf_score", None)
        g.pop("_review_value", None)

    tier_counts: dict[str, int] = {}
    for g in selected:
        tier_counts[g["review_tier"]] = tier_counts.get(g["review_tier"], 0) + 1
    logger.info(
        f"Selected {len(selected)} groups for review "
        f"(large={tier_counts.get('large', 0)}, "
        f"borderline={tier_counts.get('borderline', 0)}, "
        f"low_conf={tier_counts.get('low_confidence', 0)}, "
        f"clear={tier_counts.get('clear_winner', 0)})"
    )

    return selected
