"""Main Streamlit application for labeling road segment matches."""

import json
import os
import sys
from pathlib import Path

# Add src to path for direct streamlit execution
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import streamlit as st
import streamlit.components.v1 as components


# Config file for persisting settings like labeler name
CONFIG_FILE = Path.home() / ".matcher_labeler_config.json"


def load_config() -> dict:
    """Load config from file."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_config(config: dict) -> None:
    """Save config to file."""
    CONFIG_FILE.write_text(json.dumps(config))

from matcher.labeling.data_loader import (
    CandidatePairView,
    filter_candidates,
    generate_scored_candidates,
    load_geodataframe,
)
from matcher.labeling.comparison_view import render_comparison_view
from matcher.labeling.feature_panel import render_feature_panel
from matcher.labeling.label_store import LabelStore
from matcher.labeling.map_view import create_comparison_map
from matcher.labeling.state import (
    advance_to_next,
    get_session,
    go_to_previous,
    init_session_state,
    pop_undo,
    push_undo,
    set_decision_filter,
    set_labeler_name,
)


def main():
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="Road Segment Labeling",
        page_icon="🛣️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Reduce margins for compact layout
    st.markdown("""
        <style>
        .block-container {
            padding-top: 0.5rem;
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .element-container {
            margin-bottom: 0.3rem;
        }
        div[data-testid="stVerticalBlock"] > div {
            gap: 0.3rem;
        }
        iframe {
            min-height: 50vh;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize state
    init_session_state()

    # Get paths from environment or use defaults
    reference_path = Path(os.environ.get(
        "MATCHER_REFERENCE_PATH",
        "data/raw/overture_segments.parquet"
    ))
    target_path = Path(os.environ.get(
        "MATCHER_TARGET_PATH",
        "data/raw/boston_streets.parquet"
    ))
    labels_path = Path(os.environ.get(
        "MATCHER_LABELS_PATH",
        "data/labels/labels.parquet"
    ))

    # Initialize label store
    if st.session_state.label_store is None:
        st.session_state.label_store = LabelStore(labels_path)

    # Render UI
    render_sidebar(reference_path, target_path)
    render_main_content()


def render_sidebar(reference_path: Path, target_path: Path) -> None:
    """Render the sidebar with controls and stats."""
    session = get_session()

    with st.sidebar:
        st.title("🛣️ Road Labeling")

        # Session info
        st.subheader("Session")

        # Load saved name from config
        config = load_config()
        default_name = session.labeler_name or config.get("labeler_name", "")

        labeler = st.text_input(
            "Your name",
            value=default_name,
            placeholder="Enter your name",
        )
        if labeler != session.labeler_name:
            set_labeler_name(labeler)
            # Save to config for persistence
            config["labeler_name"] = labeler
            save_config(config)

        st.text(f"Session ID: {session.session_id}")

        st.divider()

        # Data loading
        st.subheader("Data")
        if not st.session_state.data_loaded:
            st.text(f"Reference: {reference_path.name}")
            st.text(f"Target: {target_path.name}")

            if st.button("Load Data", type="primary"):
                with st.spinner("Loading and scoring candidates..."):
                    load_data(reference_path, target_path)
                st.rerun()
        else:
            st.success(f"Loaded {len(st.session_state.candidates)} candidates")

            # Map style
            st.subheader("Map")
            tile_options = ["Light", "Satellite", "OpenStreetMap"]
            current_tile = st.session_state.get("tile_layer_choice", "Light")
            tile_index = tile_options.index(current_tile) if current_tile in tile_options else 0
            tile_layer = st.selectbox(
                "Base map",
                tile_options,
                index=tile_index,
                key="tile_layer",
            )
            # Store the choice separately so it persists across reruns
            st.session_state.tile_layer_choice = tile_layer

            # Filter controls
            st.subheader("Filter")
            filter_options = ["All", "Review", "Match", "No Match"]
            current_filter = session.decision_filter
            if current_filter:
                default_idx = filter_options.index(current_filter.title())
            else:
                default_idx = 0

            selected = st.radio(
                "Show decisions",
                filter_options,
                index=default_idx,
                horizontal=True,
            )

            new_filter = None if selected == "All" else selected.lower().replace(" ", "_")
            if new_filter != session.decision_filter:
                set_decision_filter(new_filter)
                st.rerun()

        st.divider()

        # Progress stats
        st.subheader("Progress")
        label_store = st.session_state.label_store
        stats = label_store.get_stats()

        st.metric("Total Labeled", stats["total"])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Match", stats["match"])
        with col2:
            st.metric("No Match", stats["no_match"])
        with col3:
            st.metric("Associated", stats.get("associated", 0))

        if stats.get("unsure", 0) > 0 or stats.get("maybe", 0) > 0 or stats.get("skip", 0) > 0:
            st.metric("Unsure", stats.get("unsure", 0) + stats.get("maybe", 0) + stats.get("skip", 0))

        st.divider()

        # Mode toggle
        st.subheader("Mode")
        if "show_comparison" not in st.session_state:
            st.session_state.show_comparison = False

        mode = st.radio(
            "View",
            ["Labeling", "Compare Labelers"],
            index=1 if st.session_state.show_comparison else 0,
            horizontal=True,
            label_visibility="collapsed",
        )
        if (mode == "Compare Labelers") != st.session_state.show_comparison:
            st.session_state.show_comparison = (mode == "Compare Labelers")
            st.rerun()

        st.divider()

        # Export
        if stats["total"] > 0:
            if st.button("📥 Download Labels"):
                # Trigger download
                df = label_store.df
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="labels.csv",
                    mime="text/csv",
                )


def render_main_content() -> None:
    """Render the main content area with map and features."""
    # Check if in comparison mode
    if st.session_state.get("show_comparison", False):
        render_comparison_view(st.session_state.label_store)
        return

    if not st.session_state.data_loaded:
        st.info("👈 Click 'Load Data' in the sidebar to get started")
        return

    session = get_session()
    label_store = st.session_state.label_store

    # Check if we're in 1:N mode
    if "one_to_n_mode" not in st.session_state:
        st.session_state.one_to_n_mode = False
    if "selected_refs" not in st.session_state:
        st.session_state.selected_refs = set()

    # Get filtered candidates - filter out pairs this labeler already labeled
    current_labeler = session.labeler_name or ""
    filtered = filter_candidates(
        st.session_state.candidates,
        decision_filter=session.decision_filter,
        labeled_pairs=label_store.get_labeled_pairs(labeler=current_labeler) if current_labeler else set(),
        show_labeled=False,
    )

    if not filtered:
        if session.decision_filter:
            st.success(f"All '{session.decision_filter}' pairs have been labeled!")
        else:
            st.success("All pairs have been labeled!")
        return

    # Ensure index is valid
    if session.current_index >= len(filtered):
        session.current_index = 0

    # Get current pair
    pair = filtered[session.current_index]

    # Progress indicator
    st.progress(
        session.current_index / len(filtered),
        text=f"Pair {session.current_index + 1} of {len(filtered)}",
    )

    # Check if in 1:N mode
    if st.session_state.one_to_n_mode:
        render_one_to_n_mode(pair, filtered, label_store)
    else:
        render_single_pair_mode(pair, filtered, label_store, session)


def _add_keyboard_shortcuts():
    """Add keyboard shortcut support via JavaScript."""
    import streamlit.components.v1 as components

    # JavaScript to capture keypresses and trigger Streamlit buttons
    js_code = """
    <script>
    const doc = window.parent.document;

    // Only add listener once
    if (!doc.keyboardShortcutsAdded) {
        doc.keyboardShortcutsAdded = true;
        doc.addEventListener('keydown', function(e) {
            // Ignore if typing in an input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }

            const key = e.key.toLowerCase();
            let buttonText = null;

            if (key === 'm') buttonText = 'Match';
            else if (key === 'n') buttonText = 'No Match';
            else if (key === 'i') buttonText = 'Associated';
            else if (key === 'u') buttonText = 'Unsure';
            else if (key === 'z') buttonText = 'Undo';
            else if (key === 'arrowleft') buttonText = '←';
            else if (key === 'arrowright') buttonText = '→';

            if (buttonText) {
                // Find button containing the text
                const buttons = doc.querySelectorAll('button');
                for (const btn of buttons) {
                    if (btn.innerText.includes(buttonText) && !btn.disabled) {
                        btn.click();
                        e.preventDefault();
                        break;
                    }
                }
            }
        });
    }
    </script>
    """
    components.html(js_code, height=0)


def render_single_pair_mode(pair, filtered, label_store, session):
    """Render the standard single-pair labeling mode."""
    # Add keyboard shortcut handler via JavaScript
    _add_keyboard_shortcuts()

    # Track session labels count
    if "session_label_count" not in st.session_state:
        st.session_state.session_label_count = 0

    # Compact header with shortcuts and navigation
    col_shortcuts, col_nav = st.columns([2, 1])
    with col_shortcuts:
        labeled_this_session = st.session_state.session_label_count
        st.markdown(f"**Keys:** M N I U Z ←/→ &nbsp;|&nbsp; **Labeled:** {labeled_this_session} &nbsp;|&nbsp; **Remaining:** {len(filtered)}")
    with col_nav:
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        with nav_col1:
            if st.button("←", disabled=session.current_index == 0, key="prev_top", help="Previous (Left Arrow)"):
                go_to_previous()
                st.rerun()
        with nav_col2:
            # Use dynamic key to force update after labeling
            new_idx = st.number_input(
                "Go to #",
                min_value=1,
                max_value=len(filtered),
                value=session.current_index + 1,
                key=f"jump_to_{len(filtered)}_{session.current_index}",
                label_visibility="collapsed",
            )
            if new_idx - 1 != session.current_index:
                session.current_index = new_idx - 1
                st.rerun()
        with nav_col3:
            if st.button("→", disabled=session.current_index >= len(filtered) - 1, key="next_top", help="Next (Right Arrow)"):
                advance_to_next()
                st.rerun()

    # Main layout: map on left, features on right
    col_map, col_features = st.columns([2, 1])

    with col_map:
        # Map view
        tile_layer = st.session_state.get("tile_layer_choice", "Light")
        m = create_comparison_map(pair, tile_layer=tile_layer)
        # Render map as static HTML - more reliable than st_folium for automated browsers
        # Use get_root().render() to avoid "Trust Notebook" message
        map_html = m.get_root().render()
        components.html(map_html, height=550)

    with col_features:
        # Feature panel
        render_feature_panel(pair)

        # 1:N mode button (compact)
        related_count = len([c for c in st.session_state.candidates if c.target_id == pair.target_id])
        if related_count > 1:
            if st.button(f"🔗 {related_count} candidates", help="View all candidates for this target"):
                st.session_state.one_to_n_mode = True
                st.session_state.selected_refs = {pair.ref_id}
                st.rerun()

    # Action buttons - compact row with keyboard shortcuts shown
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("✅ Match (M)", type="primary", use_container_width=True):
            record_label(pair, "match", label_store)
            st.rerun()

    with col2:
        if st.button("❌ No Match (N)", use_container_width=True):
            record_label(pair, "no_match", label_store)
            st.rerun()

    with col3:
        if st.button("🔗 Associated (I)", use_container_width=True, help="Sidewalk of road or road of sidewalk"):
            record_label(pair, "associated", label_store)
            st.rerun()

    with col4:
        if st.button("🤔 Unsure (U)", use_container_width=True):
            record_label(pair, "unsure", label_store)
            st.rerun()

    with col5:
        if st.button("↩️ Undo (Z)", disabled=len(session.undo_stack) == 0, use_container_width=True):
            undo_last_label(label_store)
            st.rerun()


def render_one_to_n_mode(pair, filtered, label_store):
    """Render the 1:N labeling mode showing all candidates for a target."""
    from matcher.labeling.map_view import create_multi_reference_map

    # Find all candidates for this target
    target_id = pair.target_id
    related_candidates = [c for c in st.session_state.candidates if c.target_id == target_id]

    st.subheader(f"1:N Mode - {len(related_candidates)} reference segments for target {target_id}")

    # Main layout
    col_map, col_list = st.columns([2, 1])

    with col_map:
        # Map showing target + all related references
        tile_layer = st.session_state.get("tile_layer_choice", "Light")
        m = create_multi_reference_map(pair.target_geometry, pair.target_name, related_candidates, st.session_state.selected_refs, tile_layer=tile_layer)
        # Render map as static HTML
        map_html = m.get_root().render()
        components.html(map_html, height=500)

    with col_list:
        st.markdown("**Select reference segments that match this target:**")

        for cand in sorted(related_candidates, key=lambda c: -c.confidence):
            is_selected = cand.ref_id in st.session_state.selected_refs
            label = f"{cand.ref_name or cand.ref_id} ({cand.confidence:.0%})"

            if st.checkbox(label, value=is_selected, key=f"ref_{cand.ref_id}"):
                st.session_state.selected_refs.add(cand.ref_id)
            else:
                st.session_state.selected_refs.discard(cand.ref_id)

        st.divider()
        st.markdown(f"**Selected:** {len(st.session_state.selected_refs)} segments")

    # Action buttons
    st.divider()
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        if st.button("✅ Label as 1:N Match", type="primary", disabled=len(st.session_state.selected_refs) == 0):
            record_one_to_n_label(target_id, related_candidates, label_store)
            st.session_state.one_to_n_mode = False
            st.session_state.selected_refs = set()
            st.rerun()

    with col2:
        if st.button("❌ No Match (entire target)"):
            record_label(pair, "no_match", label_store)
            st.session_state.one_to_n_mode = False
            st.session_state.selected_refs = set()
            st.rerun()

    with col3:
        if st.button("🤔 Unsure"):
            record_label(pair, "unsure", label_store)
            st.session_state.one_to_n_mode = False
            st.session_state.selected_refs = set()
            st.rerun()

    with col4:
        if st.button("🔙 Back to single mode"):
            st.session_state.one_to_n_mode = False
            st.session_state.selected_refs = set()
            st.rerun()


def record_one_to_n_label(target_id: str, related_candidates: list, label_store: LabelStore) -> None:
    """Record a 1:N match where multiple ref segments match one target."""
    session = get_session()

    if not session.labeler_name:
        st.error("Please enter your name in the sidebar first!")
        return

    # Record each selected ref as a match
    for cand in related_candidates:
        if cand.ref_id in st.session_state.selected_refs:
            label_store.add(
                ref_id=cand.ref_id,
                target_id=cand.target_id,
                label="match_1n",  # Special label for 1:N matches
                labeler=session.labeler_name,
                session_id=session.session_id,
                original_decision=cand.decision,
                original_confidence=cand.confidence,
                features=cand.features,
            )
            push_undo(cand.ref_id, cand.target_id, "match_1n")


def load_data(reference_path: Path, target_path: Path) -> None:
    """Load reference and target data and generate candidates."""
    reference = load_geodataframe(reference_path)
    target = load_geodataframe(target_path)

    candidates = generate_scored_candidates(
        reference=reference,
        target=target,
        buffer_distance=50.0,
        ref_id_column="id",
        target_id_column="id",
        ref_name_column="names",
        target_name_column="names",
        ref_class_column="class",
        target_class_column="class",
    )

    st.session_state.candidates = candidates
    st.session_state.data_loaded = True


def record_label(
    pair: CandidatePairView,
    label: str,
    label_store: LabelStore,
) -> None:
    """Record a label for the current pair."""
    session = get_session()

    if not session.labeler_name:
        st.error("Please enter your name in the sidebar first!")
        return

    # Add to label store
    label_store.add(
        ref_id=pair.ref_id,
        target_id=pair.target_id,
        label=label,
        labeler=session.labeler_name,
        session_id=session.session_id,
        original_decision=pair.decision,
        original_confidence=pair.confidence,
        features=pair.features,
    )

    # Push to undo stack
    push_undo(pair.ref_id, pair.target_id, label)

    # Increment session label count
    if "session_label_count" not in st.session_state:
        st.session_state.session_label_count = 0
    st.session_state.session_label_count += 1


def undo_last_label(label_store: LabelStore) -> None:
    """Undo the last labeling action."""
    undo_action = pop_undo()
    if undo_action:
        label_store.remove_last()
        # Decrement session label count
        if st.session_state.get("session_label_count", 0) > 0:
            st.session_state.session_label_count -= 1


def handle_keyboard_shortcuts(
    pair: CandidatePairView,
    label_store: LabelStore,
) -> None:
    """Handle keyboard shortcuts using a hidden text input."""
    # This is a workaround since Streamlit doesn't have native keyboard support
    # Users can focus on this input and use keyboard shortcuts
    pass  # TODO: Add keyboard shortcut support via custom component or JS


if __name__ == "__main__":
    main()
