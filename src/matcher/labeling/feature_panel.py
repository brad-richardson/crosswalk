"""Feature display components for the labeling UI."""

import html

import streamlit as st

from ..config import ALIGNMENT_FULL_TOLERANCE
from .data_loader import CandidatePairView

# Feature display configuration - top features by XGBoost importance
# These are the features that most influence the ML model's predictions
# Ordered by importance (descending) - updated 2026-01-23 after alignment-aware topology fix
# Top 10 features by importance from latest model (2026-01-23)
# Geometric features dominate (58% total), especially distance/overlap metrics
TOP_FEATURES = [
    ("centroid_distance_m", "Centroid Dist", 0.230),
    ("buffer_iou_5m", "Buffer IoU 5m", 0.101),
    ("buffer_iou_15m", "Buffer IoU 15m", 0.100),
    ("target_coverage", "Target Cov", 0.052),
    ("length_ratio", "Length Ratio", 0.049),
    ("min_coverage", "Min Coverage", 0.040),
    ("hausdorff_p95_m", "Hausdorff P95", 0.036),
    ("hausdorff_distance_m", "Hausdorff Dist", 0.032),
    ("class_similarity", "Class", 0.030),
    ("lateral_offset_m", "Lateral Offset", 0.027),
]

RAW_FEATURE_UNITS = {
    "hausdorff_distance_m": "m",
    "mean_hausdorff_distance_m": "m",
    "hausdorff_p95_m": "m",
    "projection_distance_m": "m",
    "centroid_distance_m": "m",
    "min_endpoint_proximity_m": "m",
    "max_endpoint_proximity_m": "m",
    "heading_delta": "°",
    "lateral_offset_m": "m",
    "buffer_iou_5m": "",
    "buffer_iou_15m": "",
    "length_ratio": "",
    "name_levenshtein": "",
    "name_jaro_winkler": "",
    "name_token_sort": "",
    "class_similarity": "",
    "from_degree_target": "",
    "from_degree_ref": "",
    "to_degree_target": "",
    "to_degree_ref": "",
}


def render_confidence_badge(pair: CandidatePairView) -> None:
    """Render the overall confidence score with decision badge - compact."""
    confidence_pct = pair.confidence * 100
    if pair.confidence >= 0.75:
        color = "#4CAF50"
    elif pair.confidence >= 0.5:
        color = "#FF9800"
    else:
        color = "#F44336"

    decision = pair.decision.upper()
    if decision == "MATCH":
        badge_bg = "#1B5E20"
    elif decision == "REVIEW":
        badge_bg = "#E65100"
    else:
        badge_bg = "#B71C1C"

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
            <span style="font-size: 36px; font-weight: bold; color: {color};">{confidence_pct:.0f}%</span>
            <span style="background: {badge_bg}; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; font-size: 14px;">{decision}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_segment_comparison(pair: CandidatePairView) -> None:
    """Render side-by-side segment info comparison - compact."""
    ref_id_short = pair.ref_id[:16] + "..." if len(pair.ref_id) > 16 else pair.ref_id

    # Escape user-provided data for XSS prevention
    ref_id_escaped = html.escape(ref_id_short)
    target_id_escaped = html.escape(str(pair.target_id))
    ref_name_escaped = html.escape(pair.ref_name or "N/A")
    target_name_escaped = html.escape(pair.target_name or "N/A")
    ref_class_escaped = html.escape(pair.ref_class or "N/A")
    target_class_escaped = html.escape(pair.target_class or "N/A")

    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
            <div>
                <div style="display: flex; align-items: center; margin-bottom: 6px;">
                    <span style="display: inline-block; width: 16px; height: 3px; background: #2196F3; margin-right: 6px;"></span>
                    <strong style="font-size: 14px;">Reference</strong>
                </div>
                <div style="color: #888; font-size: 11px;">ID: {ref_id_escaped}</div>
                <div style="font-size: 16px;">Name: {ref_name_escaped}</div>
                <div style="font-size: 15px;">Class: {ref_class_escaped}</div>
            </div>
            <div>
                <div style="display: flex; align-items: center; margin-bottom: 6px;">
                    <span style="display: inline-block; width: 16px; height: 3px; background: #F44336; margin-right: 6px;"></span>
                    <strong style="font-size: 14px;">Target</strong>
                </div>
                <div style="color: #888; font-size: 11px;">ID: {target_id_escaped}</div>
                <div style="font-size: 16px;">Name: {target_name_escaped}</div>
                <div style="font-size: 15px;">Class: {target_class_escaped}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _normalize_feature_for_display(feature_key: str, value: float | None) -> float:
    """Normalize a raw feature value to 0-1 for bar display (higher = better match)."""
    if value is None:
        return 0.0

    # Distance features: lower is better, normalize with reasonable max
    if feature_key in (
        "centroid_distance_m",
        "mean_hausdorff_distance_m",
        "hausdorff_distance_m",
        "hausdorff_p95_m",
    ):
        return max(0.0, 1.0 - value / 50.0)  # 50m = 0 score
    if feature_key == "lateral_offset_m":
        return max(0.0, 1.0 - value / 30.0)  # 30m = 0 score
    if feature_key in ("min_endpoint_proximity_m", "max_endpoint_proximity_m"):
        return max(0.0, 1.0 - value / 20.0)  # 20m = 0 score

    # Heading: 0-90 degrees, lower is better
    if feature_key == "heading_delta":
        return max(0.0, 1.0 - value / 90.0)

    # Length ratio: 1.0 = perfect match, penalize deviation from 1.0
    if feature_key == "length_ratio":
        # Ratio of 0.5 or 2.0 = 50% score, 0.25 or 4.0 = 0% score
        deviation = abs(1.0 - value)
        return max(0.0, 1.0 - deviation / 0.75)

    # Degree features: normalize to 0-1 based on typical values (1-4)
    # Higher degrees generally indicate more connected intersections
    # Handle degree=0 (isolated/invalid nodes) as no connectivity
    if feature_key in (
        "from_degree_target",
        "from_degree_ref",
        "to_degree_target",
        "to_degree_ref",
    ):
        if value <= 0:
            return 0.0
        return min(1.0, value / 4.0)  # 4+ = max

    # collinear_gap_ratio is already 0-1 where higher = better match
    # (handled by default case below)

    # Boolean/binary features
    if feature_key in ("intersection_match", "dead_end_match", "has_name_target", "has_name_ref"):
        return 1.0 if value else 0.0

    # Already 0-1 scores (similarities, IoU, etc.)
    return max(0.0, min(1.0, value))


def render_score_breakdown(pair: CandidatePairView) -> None:
    """Render top ML features as compact inline bars, ordered by importance."""
    scores_html = '<div style="font-size: 11px;">'
    scores_html += '<div style="color: #888; margin-bottom: 4px; font-size: 10px;">Top features by model importance:</div>'

    for feature_key, label, _importance in TOP_FEATURES:
        raw_value = pair.features.get(feature_key)
        score = _normalize_feature_for_display(feature_key, raw_value)
        bar_width = int(score * 100)

        if score >= 0.7:
            bar_color = "#4CAF50"
        elif score >= 0.4:
            bar_color = "#FF9800"
        else:
            bar_color = "#F44336"

        # Show raw value in tooltip-style format
        if raw_value is not None:
            # Handle booleans explicitly to avoid float-formatting TypeError
            if isinstance(raw_value, bool):
                raw_str = "Yes" if raw_value else "No"
            else:
                unit = RAW_FEATURE_UNITS.get(feature_key, "")
                if unit:
                    raw_str = f"{raw_value:.1f}{unit}"
                else:
                    raw_str = f"{raw_value:.2f}"
        else:
            raw_str = "N/A"

        scores_html += f"""
        <div style="display: flex; align-items: center; margin-bottom: 2px;">
            <span style="width: 90px; flex-shrink: 0;">{label}</span>
            <div style="flex-grow: 1; background: #333; border-radius: 2px; height: 6px; margin: 0 6px;">
                <div style="background: {bar_color}; width: {bar_width}%; height: 100%; border-radius: 2px;"></div>
            </div>
            <span style="width: 45px; text-align: right; font-size: 10px; color: #aaa;">{raw_str}</span>
        </div>"""

    scores_html += "</div>"
    st.markdown(scores_html, unsafe_allow_html=True)


def render_raw_features(pair: CandidatePairView) -> None:
    """Render raw feature values in a collapsible section, organized by category."""
    # Organize features by category for better readability
    FEATURE_CATEGORIES = {
        "Geometric": [
            "hausdorff_distance_m",
            "mean_hausdorff_distance_m",
            "hausdorff_p95_m",
            "buffer_iou_5m",
            "buffer_iou_15m",
            "heading_delta",
            "length_ratio",
            "projection_distance_m",
            "centroid_distance_m",
            "collinear_gap_ratio",
        ],
        "Name Similarity": [
            "name_levenshtein",
            "name_jaro_winkler",
            "name_token_sort",
            "name_soundex",
            "name_metaphone",
            "has_name_ref",
            "has_name_target",
            "name_is_generic",
            "cardinal_direction_mismatch",
        ],
        "Class": [
            "class_similarity",
        ],
        "Endpoint/Connectivity": [
            "min_endpoint_proximity_m",
            "max_endpoint_proximity_m",
            "shared_endpoint_count",
        ],
        "Lateral Offset": [
            "lateral_offset_m",
            "lateral_offset_iqr_m",
            "lateral_offset_p95_m",
        ],
        "Topology": [
            "from_degree_ref",
            "to_degree_ref",
            "from_degree_target",
            "to_degree_target",
            "degree_match_score",
            "degree_signature_similarity",
            "is_dead_end_ref",
            "is_dead_end_target",
            "dead_end_match",
            "is_intersection_ref",
            "is_intersection_target",
            "intersection_match",
        ],
        "Alignment Coverage": [
            "ref_coverage",
            "target_coverage",
            "min_coverage",
            "coverage_ratio",
        ],
        "Graphlet": [
            "graphlet_similarity",
            "endpoint_degree_similarity",
        ],
    }

    with st.expander("All Features (ML)"):
        for category, feature_keys in FEATURE_CATEGORIES.items():
            # Check if any features in this category exist
            category_features = {
                k: pair.features.get(k) for k in feature_keys if k in pair.features
            }
            if not category_features:
                continue

            st.markdown(f"**{category}**")
            for key, value in category_features.items():
                if value is None:
                    display = "N/A"
                elif isinstance(value, bool):
                    display = "Yes" if value else "No"
                elif isinstance(value, float):
                    unit = RAW_FEATURE_UNITS.get(key, "")
                    if unit:
                        display = f"{value:.2f} {unit}"
                    else:
                        display = f"{value:.3f}"
                else:
                    display = str(value)
                st.text(f"  {key}: {display}")
            st.markdown("")  # Spacer


def render_feature_panel(pair: CandidatePairView) -> None:
    """Render the complete feature panel - compact version."""
    render_confidence_badge(pair)
    st.markdown("<div style='margin: 8px 0;'></div>", unsafe_allow_html=True)
    render_segment_comparison(pair)
    st.markdown("<div style='margin: 8px 0;'></div>", unsafe_allow_html=True)
    render_score_breakdown(pair)
    st.markdown("<div style='margin: 4px 0;'></div>", unsafe_allow_html=True)
    render_raw_features(pair)


def render_minimal_feature_panel(pair: CandidatePairView) -> None:
    """Render a minimal feature panel for quick/mobile mode.

    Shows only the most essential information:
    - Confidence score with decision badge
    - Name comparison (reference vs target)
    - Road class comparison
    """
    # Confidence and decision in a compact format
    confidence_pct = pair.confidence * 100
    if pair.confidence >= 0.75:
        color = "#4CAF50"
    elif pair.confidence >= 0.5:
        color = "#FF9800"
    else:
        color = "#F44336"

    decision = pair.decision.upper()
    if decision == "MATCH":
        badge_bg = "#1B5E20"
    elif decision == "REVIEW":
        badge_bg = "#E65100"
    else:
        badge_bg = "#B71C1C"

    # Single compact row with all key info - escape for XSS prevention
    ref_name = html.escape(pair.ref_name or "No name")
    target_name = html.escape(pair.target_name or "No name")
    ref_class = html.escape(pair.ref_class or "-")
    target_class = html.escape(pair.target_class or "-")

    st.markdown(
        f"""
        <div style="background: #1E1E1E; border-radius: 8px; padding: 12px; margin: 8px 0;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-size: 28px; font-weight: bold; color: {color};">{confidence_pct:.0f}%</span>
                <span style="background: {badge_bg}; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold;">{decision}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 14px;">
                <div style="flex: 1;">
                    <div style="color: #2196F3; font-weight: bold;">Reference</div>
                    <div style="color: #EEE;">{ref_name}</div>
                    <div style="color: #888; font-size: 12px;">{ref_class}</div>
                </div>
                <div style="flex: 0 0 30px; display: flex; align-items: center; justify-content: center;">
                    <span style="color: #666;">↔</span>
                </div>
                <div style="flex: 1; text-align: right;">
                    <div style="color: #F44336; font-weight: bold;">Target</div>
                    <div style="color: #EEE;">{target_name}</div>
                    <div style="color: #888; font-size: 12px;">{target_class}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alignment_info(pair: CandidatePairView) -> None:
    """Render alignment coverage information for the pair.

    Shows what percentage of each segment is aligned/matched. This is computed
    automatically by the alignment algorithm - no manual adjustment needed.

    Args:
        pair: The candidate pair being displayed
    """
    # Check if alignment is partial (use centralized tolerance)
    tol = ALIGNMENT_FULL_TOLERANCE
    is_partial = (
        pair.ref_start_frac > tol
        or pair.ref_end_frac < (1.0 - tol)
        or pair.target_start_frac > tol
        or pair.target_end_frac < (1.0 - tol)
    )

    if not is_partial:
        return  # Full alignment - nothing special to show

    ref_coverage = pair.ref_end_frac - pair.ref_start_frac
    target_coverage = pair.target_end_frac - pair.target_start_frac

    st.markdown(
        f"""
        <div style="background: #1E1E1E; border-radius: 4px; padding: 8px; margin: 4px 0; font-size: 12px;">
            <div style="font-weight: bold; color: #888; margin-bottom: 4px;">Alignment Coverage</div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #2196F3;">Reference: {ref_coverage * 100:.0f}% ({pair.ref_start_frac * 100:.0f}%-{pair.ref_end_frac * 100:.0f}%)</span>
                <span style="color: #F44336;">Target: {target_coverage * 100:.0f}% ({pair.target_start_frac * 100:.0f}%-{pair.target_end_frac * 100:.0f}%)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
