"""Feature display components for the labeling UI."""

import html

import streamlit as st

from ..config import ALIGNMENT_FULL_TOLERANCE
from .data_loader import CandidatePairView

# Feature display configuration
# All scores are normalized 0-1 where higher = better match
FEATURE_LABELS = {
    "hausdorff_norm": "Hausdorff",
    "mean_hausdorff_norm": "Mean Haus.",
    "buffer_iou": "Buffer IoU",
    "overlap_ratio": "Overlap",
    "heading_norm": "Heading",
    "length_ratio": "Length",
    "projection_norm": "Proximity",
    "name_similarity": "Name",
    "class_similarity": "Class",
}

RAW_FEATURE_UNITS = {
    "hausdorff_distance": "m",
    "mean_hausdorff_distance": "m",
    "projection_distance": "m",
    "centroid_distance": "m",
    "heading_delta": "deg",
    "buffer_iou": "",
    "overlap_ratio": "",
    "length_ratio": "",
    "name_levenshtein": "",
    "name_jaro_winkler": "",
    "name_token_sort": "",
    "class_similarity": "",
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


def render_score_breakdown(pair: CandidatePairView) -> None:
    """Render the weighted score breakdown as compact inline bars."""
    # Build all scores in one HTML block for compactness
    scores_html = '<div style="font-size: 11px;">'

    for key, label in FEATURE_LABELS.items():
        score = pair.score_breakdown.get(key, 0.0)
        bar_width = int(score * 100)
        if score >= 0.7:
            bar_color = "#4CAF50"
        elif score >= 0.4:
            bar_color = "#FF9800"
        else:
            bar_color = "#F44336"

        scores_html += f"""
        <div style="display: flex; align-items: center; margin-bottom: 2px;">
            <span style="width: 90px; flex-shrink: 0;">{label}</span>
            <div style="flex-grow: 1; background: #333; border-radius: 2px; height: 6px; margin: 0 6px;">
                <div style="background: {bar_color}; width: {bar_width}%; height: 100%; border-radius: 2px;"></div>
            </div>
            <span style="width: 32px; text-align: right; font-weight: bold;">{score:.2f}</span>
        </div>"""

    scores_html += "</div>"
    st.markdown(scores_html, unsafe_allow_html=True)


def render_raw_features(pair: CandidatePairView) -> None:
    """Render raw feature values in a collapsible section."""
    with st.expander("Raw Features"):
        for key, value in pair.features.items():
            unit = RAW_FEATURE_UNITS.get(key, "")
            if unit:
                st.text(f"{key}: {value:.2f} {unit}")
            else:
                st.text(f"{key}: {value:.3f}")


def render_feature_panel(pair: CandidatePairView) -> None:
    """Render the complete feature panel - compact version."""
    render_confidence_badge(pair)
    st.markdown("<div style='margin: 8px 0;'></div>", unsafe_allow_html=True)
    render_segment_comparison(pair)
    st.markdown("<div style='margin: 8px 0;'></div>", unsafe_allow_html=True)
    render_score_breakdown(pair)


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
