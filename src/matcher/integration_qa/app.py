"""Streamlit app for integration QA."""

import os
import sys
from pathlib import Path

import geopandas as gpd
import streamlit as st
from streamlit_folium import st_folium

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from matcher.integration.output import load_integration_result
from matcher.integration_qa.decision_store import MergedDecisionStore, OrphanDecisionStore
from matcher.integration_qa.edge_panel import (
    render_decision_buttons,
    render_edge_details,
    render_map_legend,
    render_stats,
)
from matcher.integration_qa.map_view import create_integration_map
from matcher.integration_qa.state import QASession, load_reviewer_name, save_reviewer_name


def main():
    """Main app entry point."""
    st.set_page_config(
        page_title="Integration QA",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("Integration QA")

    # Initialize session state
    if "session" not in st.session_state:
        st.session_state.session = QASession()
        st.session_state.session.reviewer_name = load_reviewer_name()

    session = st.session_state.session

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        # Integration output directory
        integration_dir = st.text_input(
            "Integration Output Directory",
            value=os.environ.get("INTEGRATION_DIR", "data/integrated"),
            help="Directory containing edges.parquet, orphans.parquet, etc.",
        )

        # Reviewer name
        reviewer = st.text_input(
            "Reviewer Name",
            value=session.reviewer_name,
            help="Your name for tracking decisions",
        )
        if reviewer != session.reviewer_name:
            session.reviewer_name = reviewer
            save_reviewer_name(reviewer)

        # Load data button
        if st.button("Load Data", type="primary"):
            st.session_state.data_loaded = False

        st.divider()

        # View selection
        view = st.radio(
            "View",
            ["Orphans", "Merged Edges"],
            index=0 if session.current_view == "orphans" else 1,
        )
        session.current_view = "orphans" if view == "Orphans" else "merged"

    # Load data
    integration_path = Path(integration_dir)
    if not integration_path.exists():
        st.error(f"Integration directory not found: {integration_dir}")
        st.info("Run the integration pipeline first: `matcher integrate ...`")
        return

    # Load integration result
    try:
        result = load_integration_result(integration_path)
        edges = result.edges
        orphan_edges = result.orphan_edges
    except Exception as e:
        st.error(f"Error loading integration result: {e}")
        return

    # Initialize decision stores
    labels_dir = Path("data/labels")
    orphan_store = OrphanDecisionStore(labels_dir / "integration_orphans.parquet")
    merged_store = MergedDecisionStore(labels_dir / "integration_merged.parquet")

    # Filter already-reviewed edges
    reviewed_orphan_ids = orphan_store.get_reviewed_edges(session.reviewer_name)
    reviewed_merged_ids = merged_store.get_reviewed_edges(session.reviewer_name)

    # Main content area
    col_map, col_details = st.columns([3, 1])

    with col_map:
        # Apply filters and get current data
        if session.current_view == "orphans":
            filtered_edges = orphan_edges.copy() if orphan_edges is not None else gpd.GeoDataFrame()

            # Use _original_id as edge identifier
            id_col = "edge_id" if "edge_id" in filtered_edges.columns else "_original_id"
            if (
                not session.show_reviewed
                and len(filtered_edges) > 0
                and id_col in filtered_edges.columns
            ):
                filtered_edges = filtered_edges[~filtered_edges[id_col].isin(reviewed_orphan_ids)]

            if (
                session.filter_by_priority
                and len(filtered_edges) > 0
                and "qa_priority" in filtered_edges.columns
            ):
                filtered_edges = filtered_edges[
                    filtered_edges["qa_priority"] == session.filter_by_priority
                ]

            if (
                session.filter_by_component is not None
                and len(filtered_edges) > 0
                and "component_id" in filtered_edges.columns
            ):
                filtered_edges = filtered_edges[
                    filtered_edges["component_id"] == session.filter_by_component
                ]

            display_edges = filtered_edges
            _current_store = orphan_store  # noqa: F841 - reserved for future use
            is_orphan = True
        else:
            # Filter to non-reference edges
            filtered_edges = edges.copy() if edges is not None else gpd.GeoDataFrame()
            if len(filtered_edges) > 0:
                filtered_edges = filtered_edges[filtered_edges["_source"] != "reference"]

            # Use _original_id as edge identifier
            id_col = "edge_id" if "edge_id" in filtered_edges.columns else "_original_id"
            if (
                not session.show_reviewed
                and len(filtered_edges) > 0
                and id_col in filtered_edges.columns
            ):
                filtered_edges = filtered_edges[~filtered_edges[id_col].isin(reviewed_merged_ids)]

            if session.filter_by_source and len(filtered_edges) > 0:
                filtered_edges = filtered_edges[
                    filtered_edges["_source_dataset"] == session.filter_by_source
                ]

            display_edges = filtered_edges
            _current_store = merged_store  # noqa: F841 - reserved for future use
            is_orphan = False

        # Navigation
        if len(display_edges) > 0:
            st.subheader(
                f"{'Orphan' if is_orphan else 'Merged'} Edges ({len(display_edges)} remaining)"
            )

            col_prev, col_idx, col_next = st.columns([1, 2, 1])

            with col_prev:
                if st.button("← Previous"):
                    session.current_index = max(0, session.current_index - 1)

            with col_idx:
                session.current_index = st.number_input(
                    "Index",
                    min_value=0,
                    max_value=len(display_edges) - 1,
                    value=min(session.current_index, len(display_edges) - 1),
                    label_visibility="collapsed",
                )

            with col_next:
                if st.button("Next →"):
                    session.current_index = min(len(display_edges) - 1, session.current_index + 1)

            # Get current edge
            current_edge = display_edges.iloc[session.current_index]
            current_edge_dict = current_edge.to_dict()

            # Render map with click handling
            selected_id = current_edge.get("edge_id", current_edge.get("_original_id"))
            m = create_integration_map(
                edges=edges,
                orphan_edges=orphan_edges if is_orphan else None,
                selected_edge_id=selected_id,
            )
            map_data = st_folium(m, width=None, height=500, returned_objects=["last_clicked"])

            # Handle map clicks - find nearest edge and select it
            if map_data and map_data.get("last_clicked"):
                click_lat = map_data["last_clicked"]["lat"]
                click_lon = map_data["last_clicked"]["lng"]
                click_key = f"{click_lat:.6f},{click_lon:.6f}"

                # Only process if this is a new click (not the same as last processed)
                last_click = st.session_state.get("last_processed_click")
                if click_key != last_click:
                    st.session_state.last_processed_click = click_key

                    # Find nearest edge in display_edges
                    from shapely.geometry import Point

                    click_point = Point(click_lon, click_lat)
                    min_dist = float("inf")
                    nearest_pos = None

                    for pos, (_idx, row) in enumerate(display_edges.iterrows()):
                        if row.geometry is not None:
                            dist = row.geometry.distance(click_point)
                            if dist < min_dist:
                                min_dist = dist
                                nearest_pos = pos

                    # If click is reasonably close to an edge (within ~0.001 degrees ≈ 100m)
                    if nearest_pos is not None and min_dist < 0.001:
                        if nearest_pos != session.current_index:
                            session.current_index = nearest_pos
                            st.rerun()

            # Decision buttons and edge details in right panel
            with col_details:
                # Decision callback (defined first so it can be used by buttons)
                def on_decision(decision: str, reason: str):
                    # Use _original_id if edge_id not available
                    edge_id_val = current_edge.get("edge_id", current_edge.get("_original_id", 0))
                    try:
                        edge_id_int = int(edge_id_val) if edge_id_val else 0
                    except (ValueError, TypeError):
                        edge_id_int = hash(str(edge_id_val)) % (
                            10**9
                        )  # Create numeric ID from string

                    if is_orphan:
                        orphan_store.add_decision(
                            edge_id=edge_id_int,
                            original_id=str(current_edge.get("_original_id", "")),
                            source_dataset=str(current_edge.get("_source_dataset", "")),
                            component_id=int(current_edge.get("component_id", 0))
                            if current_edge.get("component_id")
                            else 0,
                            decision=decision,
                            reason=reason or st.session_state.get("orphan_reason", ""),
                            reviewer=session.reviewer_name,
                            session_id=session.session_id,
                            length_m=float(current_edge.geometry.length)
                            if current_edge.geometry
                            else 0,
                            road_class=str(
                                current_edge.get("class", current_edge.get("road_class", ""))
                            ),
                            nearest_main_dist_m=float(
                                current_edge.get(
                                    "nearest_main_distance",
                                    current_edge.get("nearest_ref_distance", 0),
                                )
                                or 0
                            ),
                            component_size=int(current_edge.get("component_size", 0))
                            if current_edge.get("component_size")
                            else 0,
                        )
                    else:
                        merged_store.add_decision(
                            edge_id=edge_id_int,
                            original_id=str(current_edge.get("_original_id", "")),
                            source_dataset=str(current_edge.get("_source_dataset", "")),
                            source_type=str(current_edge.get("_source", "")),
                            match_ref_id=str(current_edge.get("_match_ref_id", ""))
                            if current_edge.get("_match_ref_id")
                            else None,
                            decision=decision,
                            reason=reason or st.session_state.get("merged_reason", ""),
                            reviewer=session.reviewer_name,
                            session_id=session.session_id,
                            match_confidence=float(current_edge.get("_match_confidence", 0) or 0),
                            length_m=float(current_edge.geometry.length)
                            if current_edge.geometry
                            else 0,
                            road_class=str(current_edge.get("road_class", "")),
                        )

                    # Save undo action
                    session.push_undo(
                        {
                            "type": "orphan" if is_orphan else "merged",
                            "edge_id": edge_id_int,
                        }
                    )

                    # Move to next
                    session.current_index = min(len(display_edges) - 1, session.current_index + 1)
                    st.rerun()

                # Decision buttons FIRST (most important)
                render_decision_buttons(is_orphan, on_decision)

                # Undo button
                if st.button("Undo (Z)"):
                    undo_action = session.pop_undo()
                    if undo_action:
                        if undo_action["type"] == "orphan":
                            orphan_store.remove_last()
                        else:
                            merged_store.remove_last()
                        st.rerun()

                # Edge details below decision
                render_edge_details(current_edge_dict, is_orphan)

                # Filters in expander
                with st.expander("Filters", expanded=False):
                    new_show_reviewed = st.checkbox(
                        "Show reviewed", value=session.show_reviewed, key="filter_show_reviewed"
                    )
                    if new_show_reviewed != session.show_reviewed:
                        session.show_reviewed = new_show_reviewed
                        st.rerun()

                    if is_orphan:
                        priority_filter = st.selectbox(
                            "Priority",
                            ["All", "High", "Medium", "Low"],
                            index=0,
                            key="filter_priority",
                        )
                        new_priority = None if priority_filter == "All" else priority_filter.lower()
                        if new_priority != session.filter_by_priority:
                            session.filter_by_priority = new_priority
                            st.rerun()

                        if (
                            orphan_edges is not None
                            and len(orphan_edges) > 0
                            and "component_id" in orphan_edges.columns
                        ):
                            component_ids = sorted(orphan_edges["component_id"].dropna().unique())
                            component_filter = st.selectbox(
                                "Component",
                                ["All"] + [str(c) for c in component_ids],
                                index=0,
                                key="filter_component",
                            )
                            new_component = (
                                None if component_filter == "All" else int(component_filter)
                            )
                            if new_component != session.filter_by_component:
                                session.filter_by_component = new_component
                                st.rerun()
                    else:
                        if edges is not None and len(edges) > 0:
                            datasets = sorted(edges["_source_dataset"].dropna().unique())
                            dataset_filter = st.selectbox(
                                "Source Dataset",
                                ["All"] + list(datasets),
                                index=0,
                                key="filter_dataset",
                            )
                            new_source = None if dataset_filter == "All" else dataset_filter
                            if new_source != session.filter_by_source:
                                session.filter_by_source = new_source
                                st.rerun()

                # Statistics in expander
                with st.expander("Statistics", expanded=False):
                    render_stats(orphan_store.get_stats(), merged_store.get_stats())

                # Map legend / help
                with st.expander("Help / Legend", expanded=False):
                    render_map_legend()

        else:
            st.info("No edges to review. All done or adjust filters.")

    # Keyboard shortcuts (via JavaScript)
    st.markdown(
        """
        <script>
        document.addEventListener('keydown', function(e) {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            switch(e.key.toLowerCase()) {
                case 'k':
                    document.querySelector('button[kind="primary"]')?.click();
                    break;
                case 'd':
                    document.querySelector('button[kind="secondary"]')?.click();
                    break;
                case 'c':
                    document.querySelector('button[kind="primary"]')?.click();
                    break;
                case 'i':
                    document.querySelector('button[kind="secondary"]')?.click();
                    break;
                case 'z':
                    // Find undo button by text
                    const buttons = document.querySelectorAll('button');
                    buttons.forEach(b => {
                        if (b.textContent.includes('Undo')) b.click();
                    });
                    break;
            }
        });
        </script>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
