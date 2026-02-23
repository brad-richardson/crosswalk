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


def select_stitching_batch(
    groups: list[dict],
    reviewed_group_ids: set[str],
    k: int = 15,
) -> list[dict]:
    """Select a curated batch of groups for stitching review.

    Scores each unreviewed group and selects with tier-based balancing:
    - ~30% (large): groups with 10+ edges (complex M:N merges)
    - ~30% (borderline): groups where top-2 alternatives are close in confidence
    - ~20% (low_confidence): groups where best alternative has low avg confidence
    - ~20% (clear_winner): groups where one alternative clearly dominates

    Args:
        groups: List of group dicts from the groups sidecar, each must have
            pre-computed 'alternatives' (from generate_top_k_alternatives)
        reviewed_group_ids: Set of group_ids already reviewed
        k: Target batch size

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

        alternatives = group.get("alternatives", [])
        edges = group.get("edges", [])
        n_edges = len(edges)

        # Borderline score: how close top-2 alternatives are in confidence
        if len(alternatives) >= 2:
            sorted_alts = sorted(alternatives, key=lambda a: a["total_confidence"], reverse=True)
            top1 = sorted_alts[0]["total_confidence"]
            top2 = sorted_alts[1]["total_confidence"]
            if top1 > 0:
                borderline_score = 1.0 - abs(top1 - top2) / top1
            else:
                borderline_score = 0.0
        elif len(alternatives) == 1:
            borderline_score = 0.0
        else:
            borderline_score = 0.0

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

    # Tier 2: highest borderline score (from remaining)
    by_borderline = sorted(scored_groups, key=lambda g: g["_borderline_score"], reverse=True)
    count = 0
    for g in by_borderline:
        if count >= n_borderline:
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

    # Clean up internal scoring keys
    for g in selected:
        g.pop("_n_edges", None)
        g.pop("_borderline_score", None)
        g.pop("_low_conf_score", None)
        g.pop("_review_value", None)

    logger.info(
        f"Selected {len(selected)} groups for review "
        f"(large={large_filled}, borderline={n_borderline}, "
        f"low_conf={n_low_conf}, clear={n_clear})"
    )

    return selected
