"""Batch selection logic for stitching review.

Scores and selects a curated batch of M:N groups for human review,
prioritizing groups that overlap with existing human labels,
borderline cases where the optimizer's assignment is uncertain,
and low-confidence groups where the best assignment is likely wrong.
"""

from loguru import logger

# Scoring weights for composite review value.
# Label overlap is weighted higher because groups with existing human pair
# labels give us direct comparison data for evaluating optimizer accuracy.
LABEL_OVERLAP_WEIGHT = 2.0
BORDERLINE_WEIGHT = 1.0

# Tier fractions control batch composition.
# 30% label overlap: groups that overlap with existing human pair labels
# 30% borderline: groups where top-2 alternatives are close in confidence
# 30% low confidence: groups where best alternative has low avg confidence
# 10% clear winner: calibration groups where one alternative clearly dominates
TIER_OVERLAP_FRAC = 0.3
TIER_BORDERLINE_FRAC = 0.3
TIER_LOW_CONF_FRAC = 0.3


def select_stitching_batch(
    groups: list[dict],
    existing_labels_df,
    reviewed_group_ids: set[str],
    k: int = 20,
) -> list[dict]:
    """Select a curated batch of groups for stitching review.

    Scores each unreviewed group by "review value" and selects with
    tier-based balancing:
    - ~30% (label_overlap): groups whose edges overlap with human pair labels
    - ~30% (borderline): groups where top-2 alternatives are close in confidence
    - ~30% (low_confidence): groups where best alternative has low avg confidence
    - ~10% (clear_winner): groups where one alternative clearly dominates

    Args:
        groups: List of group dicts from the groups sidecar, each must have
            pre-computed 'alternatives' (from generate_top_k_alternatives)
        existing_labels_df: DataFrame of existing human pair labels with
            'gers_id' and 'target_id' columns (can be empty)
        reviewed_group_ids: Set of group_ids already reviewed
        k: Target batch size

    Returns:
        List of group dicts selected for review, ordered by review value,
        each annotated with 'review_tier' and 'review_score'
    """
    # Build set of existing labeled pairs for overlap check
    labeled_pairs: set[tuple[str, str]] = set()
    if existing_labels_df is not None and len(existing_labels_df) > 0:
        for _, row in existing_labels_df.iterrows():
            ref_id = str(row.get("gers_id", row.get("ref_id", "")))
            target_id = str(row.get("target_id", ""))
            if ref_id and target_id:
                labeled_pairs.add((ref_id, target_id))

    # Score each unreviewed group
    scored_groups: list[dict] = []
    for group in groups:
        gid = group.get("group_id", "")
        if gid in reviewed_group_ids:
            continue

        alternatives = group.get("alternatives", [])

        # Label overlap score: fraction of edges that have human labels
        edges = group.get("edges", [])
        if edges:
            overlap_count = sum(
                1 for e in edges if (str(e["ref_id"]), str(e["target_id"])) in labeled_pairs
            )
            label_overlap_score = overlap_count / len(edges)
        else:
            label_overlap_score = 0.0

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
            n_edges = len(best_alt.get("edges", []))
            avg_edge_conf = best_alt["total_confidence"] / n_edges if n_edges > 0 else 0.0
            low_conf_score = 1.0 - avg_edge_conf
        else:
            low_conf_score = 1.0  # no alternatives at all = very suspect

        # Composite review value
        review_value = (
            label_overlap_score * LABEL_OVERLAP_WEIGHT + borderline_score * BORDERLINE_WEIGHT
        )

        scored_groups.append(
            {
                **group,
                "_label_overlap_score": label_overlap_score,
                "_borderline_score": borderline_score,
                "_low_conf_score": low_conf_score,
                "_review_value": review_value,
            }
        )

    if not scored_groups:
        return []

    # Tier-based selection (clamp so sizes are non-negative and sum to k)
    n_overlap = max(1, int(k * TIER_OVERLAP_FRAC))
    n_borderline = max(1, int(k * TIER_BORDERLINE_FRAC))
    n_low_conf = max(1, int(k * TIER_LOW_CONF_FRAC))
    n_clear = max(0, k - n_overlap - n_borderline - n_low_conf)
    # If tier minimums exceed k, scale back to fit
    total_tiers = n_overlap + n_borderline + n_low_conf + n_clear
    if total_tiers > k:
        n_overlap = max(1, k // 3)
        n_borderline = max(1, k // 3)
        n_low_conf = max(0, k - n_overlap - n_borderline)
        n_clear = 0

    selected: list[dict] = []
    used_ids: set[str] = set()

    # Tier 1: highest label-overlap score
    by_overlap = sorted(scored_groups, key=lambda g: g["_label_overlap_score"], reverse=True)
    for g in by_overlap:
        if len(selected) >= n_overlap:
            break
        gid = g["group_id"]
        if gid not in used_ids:
            g["review_tier"] = "label_overlap"
            g["review_score"] = round(g["_review_value"], 3)
            selected.append(g)
            used_ids.add(gid)

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
        g.pop("_label_overlap_score", None)
        g.pop("_borderline_score", None)
        g.pop("_low_conf_score", None)
        g.pop("_review_value", None)

    logger.info(
        f"Selected {len(selected)} groups for review "
        f"(overlap={n_overlap}, borderline={n_borderline}, "
        f"low_conf={n_low_conf}, clear={n_clear})"
    )

    return selected
