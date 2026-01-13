"""Feature display components for the labeling UI."""

import streamlit as st

from .data_loader import CandidatePairView


# Feature display configuration
# All scores are normalized 0-1 where higher = better match
FEATURE_LABELS = {
    "hausdorff_norm": "Hausdorff",
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
        badge_color = "#4CAF50"
        badge_bg = "#1B5E20"
    elif decision == "REVIEW":
        badge_color = "#FF9800"
        badge_bg = "#E65100"
    else:
        badge_color = "#F44336"
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

    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
            <div>
                <div style="display: flex; align-items: center; margin-bottom: 6px;">
                    <span style="display: inline-block; width: 16px; height: 3px; background: #2196F3; margin-right: 6px;"></span>
                    <strong style="font-size: 14px;">Reference</strong>
                </div>
                <div style="color: #888; font-size: 11px;">ID: {ref_id_short}</div>
                <div style="font-size: 16px;">Name: {pair.ref_name or 'N/A'}</div>
                <div style="font-size: 15px;">Class: {pair.ref_class or 'N/A'}</div>
            </div>
            <div>
                <div style="display: flex; align-items: center; margin-bottom: 6px;">
                    <span style="display: inline-block; width: 16px; height: 3px; background: #F44336; margin-right: 6px;"></span>
                    <strong style="font-size: 14px;">Target</strong>
                </div>
                <div style="color: #888; font-size: 11px;">ID: {pair.target_id}</div>
                <div style="font-size: 16px;">Name: {pair.target_name or 'N/A'}</div>
                <div style="font-size: 15px;">Class: {pair.target_class or 'N/A'}</div>
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

        scores_html += f'''
        <div style="display: flex; align-items: center; margin-bottom: 2px;">
            <span style="width: 90px; flex-shrink: 0;">{label}</span>
            <div style="flex-grow: 1; background: #333; border-radius: 2px; height: 6px; margin: 0 6px;">
                <div style="background: {bar_color}; width: {bar_width}%; height: 100%; border-radius: 2px;"></div>
            </div>
            <span style="width: 32px; text-align: right; font-weight: bold;">{score:.2f}</span>
        </div>'''

    scores_html += '</div>'
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
