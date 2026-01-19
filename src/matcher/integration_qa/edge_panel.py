"""Edge details panel for integration QA app."""

from typing import Any

import streamlit as st


def render_edge_details(edge: dict[str, Any], is_orphan: bool = True) -> None:
    """Render edge details in sidebar."""
    st.subheader("Edge Details")

    # Basic info
    st.markdown(f"**Edge ID:** {edge.get('edge_id', 'N/A')}")
    st.markdown(f"**Original ID:** {edge.get('_original_id', edge.get('original_id', 'N/A'))}")
    st.markdown(
        f"**Source Dataset:** {edge.get('_source_dataset', edge.get('source_dataset', 'N/A'))}"
    )
    st.markdown(f"**Source Type:** {edge.get('_source', edge.get('source_type', 'N/A'))}")

    # Geometry info
    geom = edge.get("geometry")
    if geom and hasattr(geom, "length"):
        st.markdown(f"**Length:** {geom.length:.1f}m")

    # Orphan info
    if is_orphan:
        st.divider()
        st.markdown("**Orphan Status**")

        # Show why it's an orphan
        if "unmatched_reason" in edge and edge["unmatched_reason"]:
            reason = str(edge["unmatched_reason"]).replace("_", " ").title()
            st.markdown(f"Reason: {reason}")

        if "component_status" in edge and edge["component_status"]:
            st.markdown(f"Status: {edge['component_status']}")

        # Distance to reference network
        if "nearest_ref_distance" in edge and edge["nearest_ref_distance"]:
            dist = edge["nearest_ref_distance"]
            if dist is not None:
                st.markdown(f"Distance to Reference: {dist:.1f}m")

        # Distance to main network (connected edges)
        if "nearest_main_distance" in edge and edge["nearest_main_distance"]:
            dist = edge["nearest_main_distance"]
            if dist is not None and dist > 0:
                st.markdown(f"Distance to Main Network: {dist:.1f}m")

        # Priority
        if "qa_priority" in edge and edge["qa_priority"]:
            priority = edge["qa_priority"]
            priority_colors = {"high": "red", "medium": "orange", "low": "gray"}
            color = priority_colors.get(priority, "gray")
            st.markdown(f"Priority: :{color}[**{priority.upper()}**]")

        # Component info (if available from older data)
        if "component_id" in edge and edge.get("component_id"):
            st.markdown(f"Component ID: {edge['component_id']}")
        if "component_size" in edge and edge.get("component_size"):
            st.markdown(f"Component Size: {edge['component_size']} edges")

    # Match info (for merged edges)
    if not is_orphan:
        if edge.get("_match_ref_id"):
            st.divider()
            st.markdown("**Match Info**")
            st.markdown(f"Matched to: {edge['_match_ref_id']}")
            if edge.get("_match_confidence"):
                st.markdown(f"Confidence: {edge['_match_confidence']:.2f}")

    # Road attributes
    st.divider()
    st.markdown("**Attributes**")
    for col in ["name", "road_class", "highway"]:
        if col in edge and edge[col]:
            st.markdown(f"{col.title()}: {edge[col]}")


def render_decision_buttons(
    is_orphan: bool = True,
    on_decision: callable = None,
) -> None:
    """Render decision buttons."""
    st.divider()

    if is_orphan:
        st.subheader("Decision")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Keep (K)", type="primary", use_container_width=True):
                if on_decision:
                    on_decision("keep", "")

        with col2:
            if st.button("Discard (D)", type="secondary", use_container_width=True):
                if on_decision:
                    on_decision("discard", "")

        # Reason selection
        _reason = st.selectbox(  # noqa: F841 - UI element, value used by Streamlit
            "Reason (optional)",
            ["", "legitimate_new", "data_error", "out_of_scope", "duplicate"],
            key="orphan_reason",
        )
    else:
        st.subheader("Decision")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Correct (C)", type="primary", use_container_width=True):
                if on_decision:
                    on_decision("correct", "")

        with col2:
            if st.button("Incorrect (I)", type="secondary", use_container_width=True):
                if on_decision:
                    on_decision("incorrect", "")

        # Reason selection
        _reason = st.selectbox(  # noqa: F841 - UI element, value used by Streamlit
            "Reason (optional)",
            ["", "matching_error", "duplicate", "wrong_source"],
            key="merged_reason",
        )


def render_stats(orphan_stats: dict, merged_stats: dict) -> None:
    """Render statistics."""
    st.subheader("Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Orphan Decisions**")
        st.metric("Total", orphan_stats.get("total", 0))
        st.metric("Keep", orphan_stats.get("keep", 0))
        st.metric("Discard", orphan_stats.get("discard", 0))

    with col2:
        st.markdown("**Merged Decisions**")
        st.metric("Total", merged_stats.get("total", 0))
        st.metric("Correct", merged_stats.get("correct", 0))
        st.metric("Incorrect", merged_stats.get("incorrect", 0))


def render_map_legend() -> None:
    """Render map legend explaining layer colors."""
    st.markdown(
        """
        **Map Legend**

        🔵 **Reference** - Overture base road network

        🟢 **Matched** - Your data matched to a reference edge

        🟠 **To Merge** - Your data connected to network but no match (will be added)

        🔴 **Orphan** - Disconnected from network (needs review)

        🟣 **Selected** - Currently reviewing this edge

        ---

        **What to do:**
        - **Keep**: Edge is valid, should be in final network
        - **Discard**: Edge is invalid (data error, duplicate, etc.)
        """
    )
