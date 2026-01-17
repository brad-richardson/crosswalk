"""Feature display components for the labeling UI."""

from typing import Optional

import streamlit as st

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
                <div style="font-size: 16px;">Name: {pair.ref_name or "N/A"}</div>
                <div style="font-size: 15px;">Class: {pair.ref_class or "N/A"}</div>
            </div>
            <div>
                <div style="display: flex; align-items: center; margin-bottom: 6px;">
                    <span style="display: inline-block; width: 16px; height: 3px; background: #F44336; margin-right: 6px;"></span>
                    <strong style="font-size: 14px;">Target</strong>
                </div>
                <div style="color: #888; font-size: 11px;">ID: {pair.target_id}</div>
                <div style="font-size: 16px;">Name: {pair.target_name or "N/A"}</div>
                <div style="font-size: 15px;">Class: {pair.target_class or "N/A"}</div>
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


def render_subsegment_controls(
    pair: CandidatePairView,
    estimated_subsegment: Optional[dict[str, float]] = None,
) -> tuple[float, float, float, float, bool]:
    """Render compact horizontal sub-segment selection bar.

    Args:
        pair: The candidate pair being labeled
        estimated_subsegment: Precomputed overlap estimate (optional)

    Returns:
        Tuple of (ref_start, ref_end, target_start, target_end, is_active)
        where is_active indicates if sub-segment mode is enabled
    """
    _init_subseg_state()
    slider_key_suffix = f"_v{st.session_state.subseg_version}"

    # Single row: checkbox + presets + sliders
    cols = st.columns([1, 1, 1, 1, 1, 3, 3])

    with cols[0]:
        is_active = st.checkbox(
            "Subsegment",
            value=st.session_state.subseg_active,
            key=f"subseg_toggle{slider_key_suffix}",
            help="Label only portions of each segment",
        )
        st.session_state.subseg_active = is_active

    if not is_active:
        return (0.0, 1.0, 0.0, 1.0, False)

    # Preset buttons
    with cols[1]:
        if estimated_subsegment and st.button("Estimate", help="Use estimated overlap"):
            _set_subseg_values(
                round(estimated_subsegment["ref_start_pct"] * 100),
                round(estimated_subsegment["ref_end_pct"] * 100),
                round(estimated_subsegment["target_start_pct"] * 100),
                round(estimated_subsegment["target_end_pct"] * 100),
            )
            st.rerun()

    with cols[2]:
        if st.button("Full", help="Reset to 0-100%"):
            _set_subseg_values(0, 100, 0, 100)
            st.rerun()

    with cols[3]:
        if st.button("1st½", help="First 50%"):
            _set_subseg_values(0, 50, 0, 50)
            st.rerun()

    with cols[4]:
        if st.button("2nd½", help="Second 50%"):
            _set_subseg_values(50, 100, 50, 100)
            st.rerun()

    # Reference slider
    with cols[5]:
        ref_start, ref_end = st.slider(
            "Ref",
            min_value=0,
            max_value=100,
            value=(st.session_state.subseg_ref_start, st.session_state.subseg_ref_end),
            step=1,
            format="%d%%",
            key=f"ref_range_slider{slider_key_suffix}",
        )
        st.session_state.subseg_ref_start = ref_start
        st.session_state.subseg_ref_end = ref_end

    # Target slider
    with cols[6]:
        target_start, target_end = st.slider(
            "Target",
            min_value=0,
            max_value=100,
            value=(st.session_state.subseg_target_start, st.session_state.subseg_target_end),
            step=1,
            format="%d%%",
            key=f"target_range_slider{slider_key_suffix}",
        )
        st.session_state.subseg_target_start = target_start
        st.session_state.subseg_target_end = target_end

    return (
        ref_start / 100.0,
        ref_end / 100.0,
        target_start / 100.0,
        target_end / 100.0,
        True,
    )


def get_subseg_state(
    estimated_subsegment: Optional[dict[str, float]] = None,
) -> tuple[float, float, float, float, bool]:
    """Get current subsegment state values, auto-applying estimate if needed.

    This should be called before the map renders. It handles auto-apply logic
    so the map shows correct values on the first render.

    Args:
        estimated_subsegment: Precomputed overlap estimate for auto-apply

    Returns:
        Tuple of (ref_start, ref_end, target_start, target_end, is_active)
    """
    _init_subseg_state()
    is_active = st.session_state.subseg_active
    if not is_active:
        return (0.0, 1.0, 0.0, 1.0, False)

    # Auto-apply estimate if active and values are at defaults (new pair)
    values_at_defaults = (
        st.session_state.subseg_ref_start == 0
        and st.session_state.subseg_ref_end == 100
        and st.session_state.subseg_target_start == 0
        and st.session_state.subseg_target_end == 100
    )
    if estimated_subsegment and values_at_defaults:
        # Apply estimate immediately (no rerun needed, just update state)
        st.session_state.subseg_ref_start = round(estimated_subsegment["ref_start_pct"] * 100)
        st.session_state.subseg_ref_end = round(estimated_subsegment["ref_end_pct"] * 100)
        st.session_state.subseg_target_start = round(estimated_subsegment["target_start_pct"] * 100)
        st.session_state.subseg_target_end = round(estimated_subsegment["target_end_pct"] * 100)
        st.session_state.subseg_version += 1

    return (
        st.session_state.subseg_ref_start / 100.0,
        st.session_state.subseg_ref_end / 100.0,
        st.session_state.subseg_target_start / 100.0,
        st.session_state.subseg_target_end / 100.0,
        True,
    )


def _init_subseg_state() -> None:
    """Initialize session state for sub-segment selection."""
    if "subseg_active" not in st.session_state:
        st.session_state.subseg_active = False
    if "subseg_version" not in st.session_state:
        st.session_state.subseg_version = 0
    if "subseg_ref_start" not in st.session_state:
        st.session_state.subseg_ref_start = 0
    if "subseg_ref_end" not in st.session_state:
        st.session_state.subseg_ref_end = 100
    if "subseg_target_start" not in st.session_state:
        st.session_state.subseg_target_start = 0
    if "subseg_target_end" not in st.session_state:
        st.session_state.subseg_target_end = 100


def _set_subseg_values(ref_start: int, ref_end: int, target_start: int, target_end: int) -> None:
    """Set subsegment values and increment version to force slider update."""
    st.session_state.subseg_ref_start = ref_start
    st.session_state.subseg_ref_end = ref_end
    st.session_state.subseg_target_start = target_start
    st.session_state.subseg_target_end = target_end
    st.session_state.subseg_version += 1


def reset_subsegment_state() -> None:
    """Reset sub-segment slider values for a new pair.

    Preserves the toggle state so users can keep labeling in subsegment mode.
    The slider values reset to defaults - if subsegment mode is active,
    the estimate will be auto-applied on the next render.
    """
    # Ensure state is initialized before modifying
    _init_subseg_state()
    # Preserve subseg_active - don't reset it
    st.session_state.subseg_ref_start = 0
    st.session_state.subseg_ref_end = 100
    st.session_state.subseg_target_start = 0
    st.session_state.subseg_target_end = 100
    st.session_state.subseg_version = st.session_state.get("subseg_version", 0) + 1
