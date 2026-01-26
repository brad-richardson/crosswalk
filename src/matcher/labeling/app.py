"""Main Streamlit application for labeling road segment matches."""

import json
import logging
import os
import sys
from pathlib import Path

# Configure logging to show in terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add src to path for direct streamlit execution
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import streamlit as st
import streamlit.components.v1 as components

# Config file for persisting settings like labeler name
CONFIG_FILE = Path.home() / ".matcher_labeler_config.json"

# Project root for resolving data paths (src/matcher/labeling -> project root)
PROJECT_ROOT = Path(__file__).parents[3]

# Default dataset to use when none is selected
DEFAULT_DATASET = "us_boston_streets"


def _find_overture_reference(dataset_name: str, raw_dir: Path) -> str | None:
    """Find the Overture reference file for a dataset.

    Uses the centralized find_overture_segments() which tries progressively
    shorter region prefixes with versioned filenames.

    Examples:
        us_boston_streets -> us_boston_overture_segments_v1.0.parquet
        us_fort_collins_streets -> us_fort_collins_overture_segments_v1.0.parquet
    """
    from matcher.filenames import find_overture_segments

    path = find_overture_segments(raw_dir, dataset_name)
    return path.name if path else None


def _discover_datasets_from_yaml() -> dict[str, tuple[str, str]]:
    """Auto-discover datasets from yaml config files in datasets/ directory.

    Returns:
        Dict mapping dataset_name to (target_file, reference_file)
    """
    from matcher.datasets.schema import list_dataset_configs
    from matcher.filenames import find_target_file

    datasets = {}
    raw_dir = PROJECT_ROOT / "data/raw"

    for dataset_name in list_dataset_configs():
        # Find versioned target file
        target_path = find_target_file(raw_dir, dataset_name)
        if not target_path:
            continue

        # Find corresponding Overture reference file
        reference_file = _find_overture_reference(dataset_name, raw_dir)
        if reference_file:
            datasets[dataset_name] = (target_path.name, reference_file)

    return datasets


def _discover_osm_datasets() -> dict[str, tuple[str, str]]:
    """Auto-discover OSM datasets from data/raw/ directory.

    Looks for versioned files matching pattern: {region}_osm_segments_v*.parquet
    Maps them to corresponding Overture reference files.
    """
    from matcher.filenames import extract_version_from_filename

    raw_dir = PROJECT_ROOT / "data/raw"
    if not raw_dir.exists():
        return {}

    osm_datasets = {}

    # Find all *_osm_segments*.parquet files
    for osm_file in raw_dir.glob("*_osm_segments*.parquet"):
        # Only process versioned files
        version = extract_version_from_filename(osm_file)
        if version is None:
            continue

        filename = osm_file.stem  # e.g., "us_boston_streets_osm_segments_v1.0"

        # Extract region from filename by removing version suffix
        base_name = filename.rsplit("_v", 1)[0]  # "us_boston_streets_osm_segments"

        if not base_name.endswith("_osm_segments"):
            continue

        region = base_name.replace("_osm_segments", "")  # "us_boston_streets"
        dataset_id = f"{region}_osm"  # "us_boston_streets_osm"

        # Find corresponding Overture reference
        overture_ref = _find_overture_reference(region, raw_dir)
        if overture_ref:
            osm_datasets[dataset_id] = (osm_file.name, overture_ref)

    return osm_datasets


def _get_dataset_config() -> dict[str, tuple[str, str]]:
    """Get combined dataset config from yaml configs and auto-discovered OSM datasets."""
    # Start with datasets discovered from yaml configs
    config = _discover_datasets_from_yaml()
    # Add auto-discovered OSM datasets
    config.update(_discover_osm_datasets())
    return config


# Dynamic dataset config that includes auto-discovered OSM datasets
# Re-computed on each access to pick up newly fetched datasets
def get_dataset_config() -> dict[str, tuple[str, str]]:
    """Get current dataset config, refreshing OSM discoveries."""
    return _get_dataset_config()


# Initial config for module-level access
DATASET_CONFIG = _get_dataset_config()


def get_dataset_raw_files() -> dict[str, str]:
    """Get mapping of dataset_id to raw file, refreshing discoveries."""
    config = get_dataset_config()
    return {k: v[0] for k, v in config.items()}


# For backwards compatibility
DATASET_RAW_FILES = {k: v[0] for k, v in DATASET_CONFIG.items()}


def load_config() -> dict:
    """Load config from file with backup recovery.

    Returns:
        Config dict, or empty dict if no config exists
    """
    backup_file = CONFIG_FILE.with_suffix(".json.bak")

    # Try primary config first
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception as e:
            # Config is non-critical, just print warning
            print(f"Warning: Failed to load config from {CONFIG_FILE}: {e}")

            # Try backup
            if backup_file.exists():
                try:
                    print(f"Recovering config from backup: {backup_file}")
                    return json.loads(backup_file.read_text())
                except Exception as backup_e:
                    print(f"Warning: Backup config also failed: {backup_e}")

    # Try backup as fallback if primary doesn't exist
    if backup_file.exists():
        try:
            return json.loads(backup_file.read_text())
        except Exception:
            pass

    return {}


def save_config(config: dict) -> None:
    """Save config to file atomically with backup.

    Uses write-to-temp-then-replace pattern to prevent corruption.
    """
    temp_file = CONFIG_FILE.with_suffix(".json.tmp")
    backup_file = CONFIG_FILE.with_suffix(".json.bak")

    # Write to temp file first
    temp_file.write_text(json.dumps(config))

    # Backup existing file (replace() is cross-platform atomic)
    if CONFIG_FILE.exists():
        CONFIG_FILE.replace(backup_file)

    # Atomic replace
    temp_file.replace(CONFIG_FILE)


from matcher.config import settings
from matcher.labeling.comparison_view import render_comparison_view
from matcher.labeling.data_loader import (
    CandidatePairView,
    filter_by_confidence_band,
    filter_candidates,
    generate_scored_candidates_with_cache,
    get_cache_info,
    load_cached_candidates,
    load_geodataframe,
    save_candidates_to_cache,
)
from matcher.labeling.dataset_registry import DatasetRegistry
from matcher.labeling.feature_panel import (
    render_alignment_info,
    render_feature_panel,
)
from matcher.labeling.label_store import LabelLoadError, LabelStore, get_data_version
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


def get_data_paths() -> tuple[Path, Path, str]:
    """Get reference, target paths and dataset_id with env var precedence.

    Returns:
        Tuple of (reference_path, target_path, dataset_id)
    """
    # Get selected dataset from session state or query params
    default_dataset = st.query_params.get("dataset", DEFAULT_DATASET)
    selected = st.session_state.get("selected_dataset", default_dataset)

    # Validate selection (refresh config to pick up newly fetched datasets)
    current_config = get_dataset_config()
    if selected not in current_config:
        # Fall back to first available dataset, or error
        if current_config:
            selected = next(iter(current_config))
        else:
            raise ValueError("No datasets configured - add YAML files to datasets/ directory")

    target_filename, reference_filename = current_config[selected]

    # Env vars override dropdown selection (for CLI compatibility)
    reference_path = Path(
        os.environ.get(
            "MATCHER_REFERENCE_PATH", str(PROJECT_ROOT / "data/raw" / reference_filename)
        )
    )
    target_path = Path(
        os.environ.get("MATCHER_TARGET_PATH", str(PROJECT_ROOT / "data/raw" / target_filename))
    )
    # Return dataset_id instead of labels_path - LabelStore uses partitions
    return reference_path, target_path, selected


def check_dataset_change() -> bool:
    """Detect and handle dataset switching.

    Returns:
        True if dataset changed and state was reset, False otherwise
    """
    current = st.session_state.get("selected_dataset", DEFAULT_DATASET)
    previous = st.session_state.get("_last_dataset")

    # Prevent dataset change while loading
    if st.session_state.get("is_loading", False) and previous is not None and previous != current:
        st.session_state.selected_dataset = previous
        return False

    if previous is not None and previous != current:
        # Clear data cache
        st.session_state.pop("candidates", None)
        st.session_state.pop("reference", None)
        st.session_state.pop("target", None)
        st.session_state.data_loaded = False

        # Reset label store (CRITICAL: prevents cross-dataset label pollution)
        st.session_state.label_store = None

        # Reset navigation
        st.session_state.current_index = 0

        # Reset undo stack
        st.session_state.undo_stack = []

        st.session_state._last_dataset = current
        return True

    st.session_state._last_dataset = current
    return False


def main():
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="Road Segment Labeling",
        page_icon="🛣️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Reduce margins for compact layout
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
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
        /* Mobile optimizations */
        @media (max-width: 768px) {
            .block-container {
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
                padding-top: 1rem !important;
            }
            div[data-testid="stVerticalBlock"] > div {
                gap: 0.2rem !important;
            }
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Initialize state
    init_session_state()

    # Check for dataset change and reset state if needed
    check_dataset_change()

    # Get paths (respects env vars and dataset selection)
    reference_path, target_path, dataset_id = get_data_paths()

    # Initialize label store with dataset_id (uses Hive partitioning)
    if st.session_state.label_store is None:
        try:
            st.session_state.label_store = LabelStore(dataset_id)
            # Access .df to trigger load and catch errors early
            _ = st.session_state.label_store.df
        except LabelLoadError as e:
            st.error(
                f"**Failed to load labels for {dataset_id}**\n\n"
                f"{e}\n\n"
                "Check the label files in the `labels/` directory for corruption."
            )
            st.stop()

    # Render UI
    render_sidebar(reference_path, target_path, dataset_id)
    render_main_content()


def render_sidebar(reference_path: Path, target_path: Path, dataset_id: str) -> None:
    """Render the sidebar with controls and stats."""
    session = get_session()

    # Get dataset registry for metadata
    registry = DatasetRegistry()

    with st.sidebar:
        st.title("🛣️ Road Labeling")

        # Dataset selector
        st.subheader("Dataset")

        # Get default from query params for persistence across refreshes
        # Refresh config to pick up newly fetched datasets
        current_raw_files = get_dataset_raw_files()
        default_dataset = st.query_params.get("dataset", DEFAULT_DATASET)
        if default_dataset not in current_raw_files:
            default_dataset = (
                next(iter(current_raw_files)) if current_raw_files else DEFAULT_DATASET
            )

        dataset_keys = sorted(current_raw_files.keys())

        def get_dataset_display_name(key: str) -> str:
            """Get display name from registry or fallback to key.

            Always includes country code prefix (e.g., "US Boston Streets").
            """
            # Extract country code from key (e.g., "us" from "us_boston_streets")
            parts = key.replace("_osm", "").split("_")
            country_prefix = ""
            if len(parts) >= 2 and len(parts[0]) == 2:
                country_prefix = parts[0].upper() + " "

            # Handle auto-discovered OSM datasets (e.g., "us_boston_streets_osm")
            is_osm = key.endswith("_osm")
            base_key = key[:-4] if is_osm else key

            ds = registry.get(base_key)
            if ds:
                name = f"{country_prefix}{ds.name}"
                return f"{name} (OSM)" if is_osm else name

            # Fallback: format nicely from key
            rest = " ".join(p.title() for p in parts[1:])
            name = f"{country_prefix}{rest}".strip()
            return f"{name} (OSM)" if is_osm else name

        selected_dataset = st.selectbox(
            "Target Dataset",
            options=dataset_keys,
            format_func=get_dataset_display_name,
            index=dataset_keys.index(default_dataset),
            key="selected_dataset",
            disabled=st.session_state.is_loading,
        )

        # Update query params for persistence
        st.query_params["dataset"] = selected_dataset

        # Show env var override indicator
        if os.environ.get("MATCHER_TARGET_PATH"):
            st.info("Using env var override")

        # Warnings for file issues
        raw_file = current_raw_files.get(selected_dataset, "")
        raw_path = PROJECT_ROOT / "data/raw" / raw_file
        if raw_file and not raw_path.exists():
            st.warning(f"Dataset not found: {raw_file}")

        legacy_path = PROJECT_ROOT / "data/labels/labels.parquet"
        if legacy_path.exists():
            st.warning("Found legacy labels.parquet - rename to labels_boston_streets.parquet")

        # Labeler name (compact)
        config = load_config()
        default_name = session.labeler_name or config.get("labeler_name", "")
        labeler = st.text_input(
            "Labeler",
            value=default_name,
            placeholder="Your name",
            label_visibility="collapsed",
        )
        if labeler != session.labeler_name:
            set_labeler_name(labeler)
            config["labeler_name"] = labeler
            save_config(config)

        st.divider()

        # Data loading
        if not st.session_state.data_loaded:
            # Check for cache
            cache_info = get_cache_info(dataset_id, reference_path, target_path)

            # Initialize preferences
            if "use_cache" not in st.session_state:
                st.session_state.use_cache = True
            if "review_only" not in st.session_state:
                st.session_state.review_only = True

            # Auto-load if cache exists
            if cache_info["exists"]:
                st.session_state.is_loading = True
                try:
                    with st.spinner("Loading cached candidates..."):
                        load_data(
                            reference_path,
                            target_path,
                            dataset_id,
                            use_cache=True,
                            review_only=st.session_state.review_only,
                        )
                finally:
                    st.session_state.is_loading = False
                st.rerun()

            # No cache - show manual load UI
            st.caption(f"Ref: {reference_path.name}")
            st.caption(f"Target: {target_path.name}")

            # Confidence band filter
            lower_bound = settings.optimizer_review_threshold - 0.05
            upper_bound = settings.optimizer_match_threshold + 0.05
            review_only = st.checkbox(
                f"Review band ({lower_bound:.2f}-{upper_bound:.2f})",
                value=st.session_state.review_only,
                key="review_only_checkbox",
                help="Focus on candidates near decision boundaries",
            )
            st.session_state.review_only = review_only

            if st.button("Load Data", type="primary", disabled=st.session_state.is_loading):
                st.session_state.is_loading = True
                try:
                    with st.spinner("Loading and scoring candidates..."):
                        load_data(
                            reference_path,
                            target_path,
                            dataset_id,
                            use_cache=True,
                            review_only=review_only,
                        )
                finally:
                    st.session_state.is_loading = False
                st.rerun()
        else:
            # Show load status with filter indicator
            loaded_count = len(st.session_state.candidates)
            if st.session_state.get("candidates_filtered", False):
                full_count = st.session_state.get("candidates_full_count", loaded_count)
                st.caption(f"✓ {loaded_count:,}/{full_count:,} candidates (review band)")
            else:
                st.caption(f"✓ {loaded_count:,} candidates")

            # Filter controls
            filter_options = ["All", "Review", "Match", "No Match"]
            current_filter = session.decision_filter
            if current_filter:
                display_filter = current_filter.replace("_", " ").title()
                default_idx = (
                    filter_options.index(display_filter) if display_filter in filter_options else 0
                )
            else:
                default_idx = 0

            selected = st.radio(
                "Filter",
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
        label_store = st.session_state.label_store
        stats = label_store.get_stats()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Match", stats["match"])
        with col2:
            st.metric("No Match", stats["no_match"])
        with col3:
            unsure_count = stats.get("unsure", 0) + stats.get("maybe", 0) + stats.get("skip", 0)
            st.metric("Unsure", unsure_count)

        # Mode toggle
        if "show_comparison" not in st.session_state:
            st.session_state.show_comparison = False

        mode = st.radio(
            "Mode",
            ["Labeling", "Compare"],
            index=1 if st.session_state.show_comparison else 0,
            horizontal=True,
        )
        if (mode == "Compare") != st.session_state.show_comparison:
            st.session_state.show_comparison = mode == "Compare"
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
        st.info("Click 'Load Data' in the sidebar to get started")
        return

    session = get_session()
    label_store = st.session_state.label_store

    # Check if we're in 1:N mode
    if "one_to_n_mode" not in st.session_state:
        st.session_state.one_to_n_mode = False
    if "selected_refs" not in st.session_state:
        st.session_state.selected_refs = set()

    # Check if we're reviewing specific disagreements from comparison view
    review_disagreements = st.session_state.get("review_disagreements", None)

    # Show notice if in disagreement review mode
    if review_disagreements:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"📋 Reviewing {len(review_disagreements)} disagreement pairs")
        with col2:
            if st.button("✕ Exit Review Mode"):
                st.session_state.review_disagreements = None
                st.rerun()

    # Get filtered candidates - filter out pairs this labeler already labeled
    current_labeler = session.labeler_name or ""
    filtered = filter_candidates(
        st.session_state.candidates,
        decision_filter=session.decision_filter,
        labeled_pairs=label_store.get_labeled_pairs(labeler=current_labeler)
        if current_labeler
        else set(),
        show_labeled=False,
        specific_pairs=review_disagreements,
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
    # Uses multiple strategies to find the parent document for better desktop compatibility
    js_code = """
    <script>
    (function() {
        // Find the top-level Streamlit document
        let doc = document;
        try {
            // Try to access parent document (works when not blocked by CORS)
            if (window.parent && window.parent.document) {
                doc = window.parent.document;
            }
            // Try top-level if available
            if (window.top && window.top.document && window.top !== window.parent) {
                doc = window.top.document;
            }
        } catch (e) {
            // Cross-origin restriction, use current document
            doc = document;
        }

        // Clear any existing listener to avoid duplicates on rerun
        if (doc._matcherKeyHandler) {
            doc.removeEventListener('keydown', doc._matcherKeyHandler);
        }

        doc._matcherKeyHandler = function(e) {
            // Ignore if typing in an input, textarea, or contenteditable
            const tag = e.target.tagName.toUpperCase();
            if (tag === 'INPUT' || tag === 'TEXTAREA' || e.target.isContentEditable) {
                return;
            }

            // Ignore if modifier keys are pressed (except shift for arrows)
            if (e.ctrlKey || e.metaKey || e.altKey) {
                return;
            }

            const key = e.key.toLowerCase();
            // Match button by text content - use patterns that avoid ambiguity
            // (e.g., 'Match' would match both 'Match' and 'No Match' with includes())
            let shortcutMatch = null;

            if (key === 'm') shortcutMatch = 'Match';
            else if (key === 'n') shortcutMatch = 'No Match';
            else if (key === 'u') shortcutMatch = 'Unsure';
            else if (key === 'z') shortcutMatch = 'Undo';
            else if (key === 'arrowleft') shortcutMatch = '←';
            else if (key === 'arrowright') shortcutMatch = '→';

            if (shortcutMatch) {
                // Find button matching the text
                // Use endsWith to avoid 'Match' matching 'No Match' (buttons have emoji prefix)
                const buttons = doc.querySelectorAll('button[kind="secondary"], button[kind="primary"], button');
                for (const btn of buttons) {
                    const text = (btn.innerText || btn.textContent || '').trim();
                    if (text.endsWith(shortcutMatch) && !btn.disabled) {
                        btn.click();
                        e.preventDefault();
                        e.stopPropagation();
                        break;
                    }
                }
            }
        };

        doc.addEventListener('keydown', doc._matcherKeyHandler);
    })();
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
        st.markdown(
            f"**Keys:** M N U Z ←/→ &nbsp;|&nbsp; **Labeled:** {labeled_this_session} &nbsp;|&nbsp; **Remaining:** {len(filtered)}"
        )
    with col_nav:
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        with nav_col1:
            if st.button(
                "←",
                disabled=session.current_index == 0,
                key="prev_top",
                help="Previous (Left Arrow)",
            ):
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
            if st.button(
                "→",
                disabled=session.current_index >= len(filtered) - 1,
                key="next_top",
                help="Next (Right Arrow)",
            ):
                advance_to_next()
                st.rerun()

    # Pass alignment fractions from the pair (computed by pipeline)
    alignment_kwargs = {
        "ref_start_pct": pair.ref_start_frac,
        "ref_end_pct": pair.ref_end_frac,
        "target_start_pct": pair.target_start_frac,
        "target_end_pct": pair.target_end_frac,
    }

    # Main layout: map + buttons on left, features on right
    col_map, col_features = st.columns([2, 1])

    with col_map:
        # Basemap selector - compact horizontal radio
        tile_options = ["Light", "Satellite", "OpenStreetMap"]
        current_tile = st.session_state.get("tile_layer_choice", "Light")
        tile_index = tile_options.index(current_tile) if current_tile in tile_options else 0
        tile_layer = st.radio(
            "Base map",
            tile_options,
            index=tile_index,
            horizontal=True,
            label_visibility="collapsed",
            key="tile_layer",
        )
        st.session_state.tile_layer_choice = tile_layer

        # Map view - automatically shows alignment if partial
        m = create_comparison_map(pair, tile_layer=tile_layer)
        # Render map as static HTML - more reliable than st_folium for automated browsers
        # Use get_root().render() to avoid "Trust Notebook" message
        map_html = m.get_root().render()
        components.html(map_html, height=550)

        # Action buttons - four across under the map
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
        with btn_col1:
            if st.button("✅ Match", type="primary", use_container_width=True, key="btn_match"):
                record_label(pair, "match", label_store, **alignment_kwargs)
                st.rerun()
        with btn_col2:
            if st.button("❌ No Match", use_container_width=True, key="btn_no_match"):
                record_label(pair, "no_match", label_store, **alignment_kwargs)
                st.rerun()
        with btn_col3:
            if st.button("🤔 Unsure", use_container_width=True, key="btn_unsure"):
                record_label(pair, "unsure", label_store, **alignment_kwargs)
                st.rerun()
        with btn_col4:
            if st.button(
                "↩️ Undo",
                disabled=len(session.undo_stack) == 0,
                use_container_width=True,
                key="btn_undo",
            ):
                undo_last_label(label_store)
                st.rerun()

    with col_features:
        # Feature panel
        render_feature_panel(pair)

        # Show alignment info if partial alignment
        render_alignment_info(pair)

        # 1:N mode button (compact)
        related_count = len(
            [c for c in st.session_state.candidates if c.target_id == pair.target_id]
        )
        if related_count > 1:
            if st.button(
                f"🔗 {related_count} candidates", help="View all candidates for this target"
            ):
                st.session_state.one_to_n_mode = True
                st.session_state.selected_refs = {pair.ref_id}
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
        m = create_multi_reference_map(
            pair.target_geometry,
            pair.target_name,
            related_candidates,
            st.session_state.selected_refs,
            tile_layer=tile_layer,
        )
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
        if st.button(
            "✅ Label as 1:N Match",
            type="primary",
            disabled=len(st.session_state.selected_refs) == 0,
        ):
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


def record_one_to_n_label(
    target_id: str, related_candidates: list, label_store: LabelStore
) -> None:
    """Record a 1:N match where multiple ref segments match one target."""
    session = get_session()

    if not session.labeler_name:
        st.error("Please enter your name in the sidebar first!")
        return

    # Record each selected ref as a match
    for cand in related_candidates:
        if cand.ref_id in st.session_state.selected_refs:
            label_store.add(
                gers_id=cand.ref_id,  # ref_id is the Overture GERS ID
                target_id=cand.target_id,
                label="match_1n",  # Special label for 1:N matches
                labeler=session.labeler_name,
                session_id=session.session_id,
                original_decision=cand.decision,
                original_confidence=cand.confidence,
                features=cand.features,
                # Data version tracking
                ref_data_version=st.session_state.get("ref_data_version"),
                target_data_version=st.session_state.get("target_data_version"),
                # Geometry persistence for reproducible backfill
                ref_geometry=cand.ref_geometry,
                target_geometry=cand.target_geometry,
                ref_name_raw=cand.ref_name,
                target_name_raw=cand.target_name,
                ref_class_raw=cand.ref_class,
                target_class_raw=cand.target_class,
                ref_subclass=cand.ref_subclass,
                target_subclass=cand.target_subclass,
            )
            push_undo(cand.ref_id, cand.target_id, "match_1n")


def load_data(
    reference_path: Path,
    target_path: Path,
    dataset_id: str,
    use_cache: bool = True,
    review_only: bool = True,
) -> None:
    """Load reference and target data and generate candidates.

    Args:
        reference_path: Path to reference (Overture) data
        target_path: Path to target (local) data
        dataset_id: Unique identifier for dataset (used for caching)
        use_cache: Whether to use cached candidates if available
        review_only: If True, filter to review band (thresholds ± 0.05)
    """
    candidates = None

    # Capture data versions for version tracking in labels
    st.session_state.ref_data_version = get_data_version(reference_path)
    st.session_state.target_data_version = get_data_version(target_path)

    logger.info(f"Loading data for {dataset_id}...")

    # Try to load from cache first
    if use_cache:
        logger.info("Checking scored cache...")
        candidates = load_cached_candidates(dataset_id)
        if candidates:
            logger.info(f"Scored cache hit: {len(candidates):,} candidates loaded")

    # Generate fresh candidates if cache miss
    if candidates is None:
        logger.info("Scored cache miss - loading from feature cache...")
        logger.info(f"Loading reference: {reference_path}")
        reference = load_geodataframe(reference_path)
        logger.info(f"Loading target: {target_path}")
        target = load_geodataframe(target_path)

        # Use feature cache for faster loading (skips feature computation if cached)
        candidates = generate_scored_candidates_with_cache(
            reference=reference,
            target=target,
            dataset_id=dataset_id,
            buffer_distance_m=settings.buffer_distance_m,
            ref_id_column="id",
            target_id_column="id",
            ref_name_column="names",
            target_name_column="names",
            ref_class_column="class",
            target_class_column="class",
        )

        # Save full candidates to cache for next time
        if candidates:
            logger.info(f"Saving {len(candidates):,} candidates to scored cache...")
            save_candidates_to_cache(dataset_id, candidates)
            logger.info("Scored cache saved - next load will be fast")

    # Track full count before filtering
    full_count = len(candidates) if candidates else 0
    st.session_state.candidates_full_count = full_count

    # Apply confidence band filter if requested and we have candidates
    if review_only and full_count > 0:
        logger.info("Filtering to review band...")
        candidates = filter_by_confidence_band(candidates, review_only=True)
        st.session_state.candidates_filtered = True
        logger.info(f"Filtered to {len(candidates):,} candidates in review band")
    else:
        st.session_state.candidates_filtered = False

    st.session_state.candidates = candidates
    st.session_state.data_loaded = True
    logger.info(f"Data loading complete: {len(candidates):,} candidates ready")


def record_label(
    pair: CandidatePairView,
    label: str,
    label_store: LabelStore,
    ref_start_pct: float = 0.0,
    ref_end_pct: float = 1.0,
    target_start_pct: float = 0.0,
    target_end_pct: float = 1.0,
) -> None:
    """Record a label for the current pair.

    Args:
        pair: The candidate pair being labeled
        label: Label value (match, no_match, unsure)
        label_store: Label storage manager
        ref_start_pct: Start of reference sub-segment (0.0-1.0)
        ref_end_pct: End of reference sub-segment (0.0-1.0)
        target_start_pct: Start of target sub-segment (0.0-1.0)
        target_end_pct: End of target sub-segment (0.0-1.0)
    """
    session = get_session()

    if not session.labeler_name:
        st.error("Please enter your name in the sidebar first!")
        return

    # Add to label store with sub-segment info and version tracking
    label_store.add(
        gers_id=pair.ref_id,  # ref_id is the Overture GERS ID
        target_id=pair.target_id,
        label=label,
        labeler=session.labeler_name,
        session_id=session.session_id,
        original_decision=pair.decision,
        original_confidence=pair.confidence,
        features=pair.features,
        ref_start_pct=ref_start_pct,
        ref_end_pct=ref_end_pct,
        target_start_pct=target_start_pct,
        target_end_pct=target_end_pct,
        # Data version tracking (captured when data was loaded)
        ref_data_version=st.session_state.get("ref_data_version"),
        target_data_version=st.session_state.get("target_data_version"),
        # Geometry persistence for reproducible backfill
        ref_geometry=pair.ref_geometry,
        target_geometry=pair.target_geometry,
        ref_name_raw=pair.ref_name,
        target_name_raw=pair.target_name,
        ref_class_raw=pair.ref_class,
        target_class_raw=pair.target_class,
        ref_subclass=pair.ref_subclass,
        target_subclass=pair.target_subclass,
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


if __name__ == "__main__":
    main()
