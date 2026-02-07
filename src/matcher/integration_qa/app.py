"""Streamlit app for integration QA."""

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import streamlit as st
from streamlit_folium import st_folium

from matcher.datasets.schema import get_dataset_config, list_dataset_configs
from matcher.filenames import (
    PROJECT_ROOT,
    bridge_filename,
    find_overture_segments,
    find_target_file,
    integration_cache_dir,
    unmatched_filename,
)
from matcher.integration.output import load_integration_result
from matcher.integration_qa.browse_view import render_browse_view
from matcher.integration_qa.decision_store import MergedDecisionStore, OrphanDecisionStore
from matcher.integration_qa.edge_panel import (
    render_decision_buttons,
    render_edge_details,
    render_map_legend,
    render_stats,
)
from matcher.integration_qa.map_view import create_integration_map
from matcher.integration_qa.state import QASession, load_reviewer_name, save_reviewer_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background pipeline runner — survives Streamlit reruns/reconnections
# ---------------------------------------------------------------------------


@dataclass
class PipelineJob:
    """Tracks a running or completed pipeline job."""

    dataset_name: str
    stage: str = "queued"  # queued → matching → integrating → done / error
    error: str | None = None
    future: Future | None = field(default=None, repr=False)


class _PipelineManager:
    """Thread-pool backed pipeline manager.

    Held in a @st.cache_resource so it persists across reruns.
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._jobs: dict[str, PipelineJob] = {}

    def submit(self, dataset_name: str, fn, *args) -> PipelineJob:
        job = PipelineJob(dataset_name=dataset_name)
        job.future = self._executor.submit(self._run, job, fn, *args)
        self._jobs[dataset_name] = job
        return job

    @staticmethod
    def _run(job: PipelineJob, fn, *args):
        """Wrapper that updates job.stage as the callable progresses."""
        try:
            fn(job, *args)
            job.stage = "done"
        except Exception as e:
            job.stage = "error"
            job.error = str(e)
            logger.exception(f"Pipeline failed for {job.dataset_name}")

    def get_job(self, dataset_name: str) -> PipelineJob | None:
        return self._jobs.get(dataset_name)

    def is_running(self, dataset_name: str) -> bool:
        job = self._jobs.get(dataset_name)
        return job is not None and job.future is not None and not job.future.done()


@st.cache_resource
def _get_pipeline_manager() -> _PipelineManager:
    return _PipelineManager()


def _pipeline_task(
    job: PipelineJob,
    dataset_name: str,
    include_matching: bool,
):
    """Run match (if needed) + integration in a background thread."""
    # Resolve inputs
    inputs = None if include_matching else _find_integration_inputs(dataset_name)

    if inputs is None or isinstance(inputs, str):
        match_inputs = _find_matching_inputs(dataset_name)
        if isinstance(match_inputs, str):
            raise RuntimeError(match_inputs)
        ref, target = match_inputs
        job.stage = "matching"
        bridge, unmatched = _run_matching(dataset_name, ref, target)
        inputs = (ref, bridge, unmatched, target)

    ref, bridge, unmatched, target = inputs
    job.stage = "integrating"
    _run_integration(dataset_name, ref, bridge, unmatched, target)


def _list_datasets() -> list[tuple[str, str]]:
    """List available datasets with display names.

    Returns:
        List of (dataset_name, display_label) tuples, sorted by display_label.
    """
    names = list_dataset_configs()
    result = []
    for name in names:
        config = get_dataset_config(name)
        if config and config.display_name:
            label = f"{config.display_name} ({name})"
        else:
            label = name
        result.append((name, label))
    result.sort(key=lambda x: x[1])
    return result


def _find_integration_inputs(
    dataset_name: str,
) -> tuple[Path, Path, Path, Path | None] | str:
    """Find reference, bridge, unmatched, and target files for a dataset.

    Searches for bridge/unmatched in two locations:
    1. data/output/{name}_bridge.parquet + {name}_unmatched.parquet
    2. data/output/{name}/bridge.parquet + unmatched.parquet (subdir convention)

    Also locates the full target file so the integration pipeline can
    extract matched geometries from the bridge.

    Returns:
        (reference_path, bridge_path, unmatched_path, target_path) or error message string.
        target_path may be None if the raw target file doesn't exist.
    """
    raw_dir = PROJECT_ROOT / "data" / "raw"
    output_dir = PROJECT_ROOT / "data" / "output"

    ref = find_overture_segments(raw_dir, dataset_name)
    if not ref:
        return f"Reference (Overture) segments not found for {dataset_name}"

    # Locate full target file (optional but important for matched edge extraction)
    target = find_target_file(raw_dir, dataset_name)

    # Try flat layout: data/output/{name}_bridge.parquet
    bridge = output_dir / bridge_filename(dataset_name)
    if bridge.exists():
        unmatched = output_dir / unmatched_filename(dataset_name)
        if not unmatched.exists():
            # Also check generic unmatched sibling
            unmatched = output_dir / "unmatched.parquet"
        if unmatched.exists():
            return ref, bridge, unmatched, target

    # Try subdir layout: data/output/{name}/bridge.parquet
    subdir = output_dir / dataset_name
    bridge_sub = subdir / "bridge.parquet"
    if bridge_sub.exists():
        unmatched_sub = subdir / "unmatched.parquet"
        if unmatched_sub.exists():
            return ref, bridge_sub, unmatched_sub, target

    return (
        f"Bridge file not found for {dataset_name}. Click **Run Match + Integrate** to generate it."
    )


def _find_matching_inputs(dataset_name: str) -> tuple[Path, Path] | str:
    """Find reference and target raw files for running the matching pipeline.

    Returns:
        (reference_path, target_path) or error message string.
    """
    raw_dir = PROJECT_ROOT / "data" / "raw"

    ref = find_overture_segments(raw_dir, dataset_name)
    if not ref:
        return f"Reference (Overture) segments not found in {raw_dir}"

    target = find_target_file(raw_dir, dataset_name)
    if not target:
        return f"Target data file not found for {dataset_name} in {raw_dir}"

    return ref, target


def _run_matching(dataset_name: str, ref: Path, target: Path) -> tuple[Path, Path]:
    """Run the matching pipeline to produce bridge and unmatched files.

    Outputs to data/output/{dataset_name}/ to avoid filename collisions.

    Returns:
        (bridge_path, unmatched_path)
    """
    from matcher.pipeline.runner import run_pipeline

    output_dir = PROJECT_ROOT / "data" / "output" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    bridge_path = output_dir / "bridge.parquet"
    run_pipeline(
        reference_path=ref,
        target_path=target,
        output_path=bridge_path,
        method="xgboost",
    )

    unmatched_path = output_dir / "unmatched.parquet"
    return bridge_path, unmatched_path


def _run_integration(
    dataset_name: str,
    ref: Path,
    bridge: Path,
    unmatched: Path,
    target: Path | None = None,
) -> bool:
    """Run integration pipeline, store results in cache.

    Args:
        dataset_name: Dataset identifier
        ref: Reference (Overture) segments path
        bridge: Bridge file path (match results)
        unmatched: Unmatched segments path
        target: Full target data path (needed to extract matched geometries)

    Returns:
        True on success, raises on failure.
    """
    from matcher.integration import TargetConfig, run_integration_pipeline

    output_dir = integration_cache_dir(dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TargetConfig(
        name=dataset_name,
        bridge_path=bridge,
        unmatched_path=unmatched,
        priority=1,
        target_path=target,
    )
    run_integration_pipeline(
        reference_path=ref,
        target_configs=[config],
        output_dir=output_dir,
        target_id_column="id",
    )
    return True


def _get_dataset_status(dataset_name: str) -> str:
    """Get integration status for a dataset.

    Returns:
        "cached" if integration results exist,
        "ready" if integration inputs (bridge/unmatched) are available,
        "needs_matching" if raw data exists but matching hasn't been run,
        "missing" if raw data files are missing.
    """
    cache_dir = integration_cache_dir(dataset_name)
    if (cache_dir / "edges.parquet").exists():
        return "cached"

    integration_inputs = _find_integration_inputs(dataset_name)
    if not isinstance(integration_inputs, str):
        return "ready"

    matching_inputs = _find_matching_inputs(dataset_name)
    if not isinstance(matching_inputs, str):
        return "needs_matching"

    return "missing"


def render_integration_qa_sidebar() -> tuple[str, "QASession"]:
    """Render integration QA sidebar controls.

    Returns:
        Tuple of (integration_dir, session)
    """
    # Initialize session state for QA
    if "qa_session" not in st.session_state:
        st.session_state.qa_session = QASession()
        st.session_state.qa_session.reviewer_name = load_reviewer_name()

    session = st.session_state.qa_session

    # Dataset selector
    datasets = _list_datasets()
    if not datasets:
        st.warning("No datasets found in datasets/*.yaml")
        return "", session

    dataset_labels = [label for _, label in datasets]
    dataset_names = [name for name, _ in datasets]

    # Restore selected index from URL param, then session state
    selected_idx = 0
    url_dataset = st.query_params.get("dataset")
    if url_dataset and url_dataset in dataset_names:
        selected_idx = dataset_names.index(url_dataset)
    elif "qa_selected_dataset" in st.session_state:
        prev = st.session_state.qa_selected_dataset
        if prev in dataset_names:
            selected_idx = dataset_names.index(prev)

    chosen_idx = st.selectbox(
        "Dataset",
        range(len(dataset_labels)),
        index=selected_idx,
        format_func=lambda i: dataset_labels[i],
        key="qa_dataset_selector",
    )
    dataset_name = dataset_names[chosen_idx]
    st.session_state.qa_selected_dataset = dataset_name
    st.query_params["dataset"] = dataset_name

    # Status indicator
    status = _get_dataset_status(dataset_name)
    status_labels = {
        "cached": "Cached",
        "ready": "Ready to integrate",
        "needs_matching": "Needs matching",
        "missing": "Missing raw data",
    }
    st.caption(f"Status: {status_labels.get(status, status)}")

    # Integration directory (derived from cache)
    integration_dir = str(integration_cache_dir(dataset_name))

    # Action buttons based on status
    mgr = _get_pipeline_manager()
    job = mgr.get_job(dataset_name)

    if status == "missing":
        match_inputs = _find_matching_inputs(dataset_name)
        if isinstance(match_inputs, str):
            st.info(match_inputs)
    elif mgr.is_running(dataset_name):
        # Pipeline running in background — show progress and auto-refresh
        stage_labels = {
            "queued": "Queued...",
            "matching": "Running matching pipeline...",
            "integrating": "Running integration pipeline...",
        }
        st.info(stage_labels.get(job.stage, f"Running ({job.stage})..."))
        time.sleep(2)
        st.rerun()
    elif job is not None and job.stage == "done":
        # Just finished — show success and clear
        st.success("Pipeline complete!")
        # Show re-run controls
        include_matching = st.checkbox("Include matching", key="qa_include_matching")
        if st.button("Re-run Integration", key="qa_run_integration"):
            mgr.submit(dataset_name, _pipeline_task, dataset_name, include_matching)
            st.rerun()
    elif job is not None and job.stage == "error":
        st.error(f"Pipeline failed: {job.error}")
        include_matching = st.checkbox("Include matching", key="qa_include_matching")
        if st.button("Retry", type="primary", key="qa_run_integration"):
            mgr.submit(dataset_name, _pipeline_task, dataset_name, include_matching)
            st.rerun()
    else:
        # Normal idle state
        button_label = "Re-run Integration" if status == "cached" else "Run Integration"
        button_type = "secondary" if status == "cached" else "primary"
        include_matching = st.checkbox("Include matching", key="qa_include_matching")
        if st.button(button_label, type=button_type, key="qa_run_integration"):
            mgr.submit(dataset_name, _pipeline_task, dataset_name, include_matching)
            st.rerun()

    st.divider()

    # Basemap selector
    basemap = st.radio(
        "Basemap",
        ["Light", "Satellite", "OpenStreetMap"],
        index=["Light", "Satellite", "OpenStreetMap"].index(session.basemap),
        horizontal=True,
        key="qa_basemap_selector",
    )
    if basemap != session.basemap:
        session.basemap = basemap

    # Reviewer name
    reviewer = st.text_input(
        "Reviewer Name",
        value=session.reviewer_name,
        help="Your name for tracking decisions",
        key="qa_reviewer_name",
    )
    if reviewer != session.reviewer_name:
        session.reviewer_name = reviewer
        save_reviewer_name(reviewer)

    return integration_dir, session


def render_integration_qa_content(integration_dir: str, session: "QASession") -> None:
    """Render the main integration QA content area.

    Args:
        integration_dir: Path to integration output directory
        session: QA session state
    """
    integration_path = Path(integration_dir)
    if not integration_path.exists() or not (integration_path / "edges.parquet").exists():
        st.info(
            "No integration results for this dataset. "
            "Select a dataset and click **Run Integration** in the sidebar."
        )
        return

    # Load integration result
    try:
        result = load_integration_result(integration_path)
        edges = result.edges
        disconnected_edges = result.disconnected_edges
        filtered_edges = result.filtered_edges
        net_new_edges = result.net_new_edges
        bridge_edges = result.bridge_edges
    except Exception as e:
        st.error(f"Error loading integration result: {e}")
        return

    # Summary metrics bar — compute from actual loaded data (pipeline stats
    # reflect pre-orphan-detection counts which can be misleading)
    n_connected = (
        int((edges["_source"] == "target_new").sum())
        if edges is not None and "_source" in edges.columns
        else 0
    )
    n_net_new = len(net_new_edges) if net_new_edges is not None else 0
    n_disconnected = len(disconnected_edges) if disconnected_edges is not None else 0
    n_filtered = len(filtered_edges) if filtered_edges is not None else 0
    n_bridges = len(bridge_edges) if bridge_edges is not None else 0

    cols = st.columns(7)
    cols[0].metric("Net New", n_net_new)
    cols[1].metric("To Merge", n_connected)
    cols[2].metric("Bridges", n_bridges)
    cols[3].metric("Disconnected", n_disconnected)
    cols[4].metric("Filtered", n_filtered)
    cols[5].metric(
        "Reference",
        len(edges[edges["_source"] == "reference"])
        if edges is not None and "_source" in edges.columns
        else 0,
    )
    cols[6].metric("Total", (len(edges) if edges is not None else 0) + n_disconnected + n_filtered)

    # Two tabs: Browse Map (default) and Edge Review
    tab_browse, tab_review = st.tabs(["Browse Map", "Edge Review"])

    with tab_browse:
        render_browse_view(
            edges=edges,
            disconnected_edges=disconnected_edges,
            net_new_edges=net_new_edges,
            bridge_edges=bridge_edges,
            basemap=session.basemap,
        )

    with tab_review:
        _render_review_view(edges, disconnected_edges, filtered_edges, net_new_edges, session)


def _render_review_view(
    edges: gpd.GeoDataFrame,
    disconnected_edges: gpd.GeoDataFrame,
    filtered_edges: gpd.GeoDataFrame,
    net_new_edges: gpd.GeoDataFrame | None,
    session: "QASession",
) -> None:
    """Render the existing edge-by-edge review view.

    This is the original review flow, now in a tab.
    """
    # Combine disconnected + filtered for orphan review (they share the same QA flow)
    import pandas as pd

    if disconnected_edges is not None and filtered_edges is not None:
        orphan_edges = (
            gpd.GeoDataFrame(
                pd.concat([disconnected_edges, filtered_edges], ignore_index=True),
                crs=disconnected_edges.crs if disconnected_edges.crs else filtered_edges.crs,
            )
            if len(disconnected_edges) > 0 or len(filtered_edges) > 0
            else gpd.GeoDataFrame()
        )
    elif disconnected_edges is not None:
        orphan_edges = disconnected_edges
    elif filtered_edges is not None:
        orphan_edges = filtered_edges
    else:
        orphan_edges = gpd.GeoDataFrame()
    # Initialize decision stores
    labels_dir = Path("data/labels")
    orphan_store = OrphanDecisionStore(labels_dir / "integration_orphans.parquet")
    merged_store = MergedDecisionStore(labels_dir / "integration_merged.parquet")

    # Filter already-reviewed edges
    reviewed_orphan_ids = orphan_store.get_reviewed_edges(session.reviewer_name)
    reviewed_merged_ids = merged_store.get_reviewed_edges(session.reviewer_name)

    # View selection
    view = st.radio(
        "Review Mode",
        ["Orphans", "Merged Edges"],
        index=0 if session.current_view == "orphans" else 1,
        horizontal=True,
        key="qa_review_view_selector",
    )
    new_view = "orphans" if view == "Orphans" else "merged"
    if new_view != session.current_view:
        session.current_view = new_view
        session.current_index = 0
        st.session_state.pop("qa_last_processed_click", None)

    # Main content area
    col_map, col_details = st.columns([3, 1])

    with col_map:
        # Apply filters and get current data
        if session.current_view == "orphans":
            filtered_edges = orphan_edges.copy() if orphan_edges is not None else gpd.GeoDataFrame()

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
            is_orphan = True
        else:
            filtered_edges = edges.copy() if edges is not None else gpd.GeoDataFrame()
            if len(filtered_edges) > 0:
                filtered_edges = filtered_edges[filtered_edges["_source"] != "reference"]

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
            is_orphan = False

        # Navigation
        if len(display_edges) > 0:
            st.subheader(
                f"{'Orphan' if is_orphan else 'Merged'} Edges ({len(display_edges)} remaining)"
            )

            session.current_index = min(session.current_index, len(display_edges) - 1)

            current_edge = display_edges.iloc[session.current_index]
            current_edge_dict = current_edge.to_dict()

            # Render map with click handling
            selected_id = current_edge.get("edge_id", current_edge.get("_original_id"))
            m = create_integration_map(
                edges=edges,
                net_new_edges=net_new_edges,
                selected_edge_id=selected_id,
                disconnected_edges=disconnected_edges if is_orphan else None,
                filtered_edges=filtered_edges if is_orphan else None,
            )
            map_key = f"qa_map_{session.current_view}"
            map_data = st_folium(
                m, width=None, height=500, returned_objects=["last_clicked"], key=map_key
            )

            # Handle map clicks
            if map_data and map_data.get("last_clicked"):
                click_lat = map_data["last_clicked"]["lat"]
                click_lon = map_data["last_clicked"]["lng"]
                click_key = f"{click_lat:.6f},{click_lon:.6f}"

                last_click = st.session_state.get("qa_last_processed_click")
                if click_key != last_click:
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

                    if nearest_pos is not None and min_dist < 0.001:
                        st.session_state.qa_last_processed_click = click_key
                        if nearest_pos != session.current_index:
                            session.current_index = nearest_pos
                            st.rerun()
                    else:
                        st.session_state.qa_last_processed_click = click_key

            # Navigation controls
            col_prev, col_idx, col_next = st.columns([1, 2, 1])

            at_start = session.current_index == 0
            at_end = session.current_index >= len(display_edges) - 1

            with col_prev:
                if st.button("← Previous", disabled=at_start, key="qa_prev"):
                    session.current_index = max(0, session.current_index - 1)
                    st.rerun()

            with col_idx:
                st.caption(f"Edge {session.current_index + 1} of {len(display_edges)}")
                new_index = st.number_input(
                    "Index",
                    min_value=0,
                    max_value=len(display_edges) - 1,
                    value=session.current_index,
                    label_visibility="collapsed",
                    key="qa_edge_index_input",
                )
                if new_index != session.current_index:
                    session.current_index = new_index
                    st.rerun()

            with col_next:
                if st.button("Next →", disabled=at_end, key="qa_next"):
                    session.current_index = min(len(display_edges) - 1, session.current_index + 1)
                    st.rerun()

            if at_end:
                st.info("Last edge in filtered list. Apply different filters to see more.")

            # Decision buttons and edge details in right panel
            with col_details:

                def on_decision(decision: str, reason: str):
                    edge_id_val = current_edge.get("edge_id", current_edge.get("_original_id", 0))
                    try:
                        edge_id_int = int(edge_id_val) if edge_id_val else 0
                    except (ValueError, TypeError):
                        edge_id_int = hash(str(edge_id_val)) % (10**9)

                    if is_orphan:
                        orphan_store.add_decision(
                            edge_id=edge_id_int,
                            original_id=str(current_edge.get("_original_id", "")),
                            dataset_id=str(current_edge.get("_source_dataset", "")),
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
                            dataset_id=str(current_edge.get("_source_dataset", "")),
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

                    session.push_undo(
                        {
                            "type": "orphan" if is_orphan else "merged",
                            "edge_id": edge_id_int,
                        }
                    )

                    session.current_index = min(len(display_edges) - 1, session.current_index + 1)
                    st.rerun()

                render_decision_buttons(is_orphan, on_decision)

                if st.button("Undo (Z)", key="qa_undo"):
                    undo_action = session.pop_undo()
                    if undo_action:
                        if undo_action["type"] == "orphan":
                            orphan_store.remove_last()
                        else:
                            merged_store.remove_last()
                        st.rerun()

                render_edge_details(current_edge_dict, is_orphan)

                with st.expander("Filters", expanded=False):
                    new_show_reviewed = st.checkbox(
                        "Show reviewed",
                        value=session.show_reviewed,
                        key="qa_filter_show_reviewed",
                    )
                    if new_show_reviewed != session.show_reviewed:
                        session.show_reviewed = new_show_reviewed
                        session.current_index = 0
                        st.session_state.pop("qa_last_processed_click", None)
                        st.rerun()

                    if is_orphan:
                        priority_filter = st.selectbox(
                            "Priority",
                            ["All", "High", "Medium", "Low"],
                            index=0,
                            key="qa_filter_priority",
                        )
                        new_priority = None if priority_filter == "All" else priority_filter.lower()
                        if new_priority != session.filter_by_priority:
                            session.filter_by_priority = new_priority
                            session.current_index = 0
                            st.session_state.pop("qa_last_processed_click", None)
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
                                key="qa_filter_component",
                            )
                            new_component = (
                                None if component_filter == "All" else int(component_filter)
                            )
                            if new_component != session.filter_by_component:
                                session.filter_by_component = new_component
                                session.current_index = 0
                                st.session_state.pop("qa_last_processed_click", None)
                                st.rerun()
                    else:
                        if edges is not None and len(edges) > 0:
                            all_datasets = sorted(edges["_source_dataset"].dropna().unique())
                            dataset_filter = st.selectbox(
                                "Source Dataset",
                                ["All"] + list(all_datasets),
                                index=0,
                                key="qa_filter_dataset",
                            )
                            new_source = None if dataset_filter == "All" else dataset_filter
                            if new_source != session.filter_by_source:
                                session.filter_by_source = new_source
                                session.current_index = 0
                                st.session_state.pop("qa_last_processed_click", None)
                                st.rerun()

                with st.expander("Statistics", expanded=False):
                    render_stats(orphan_store.get_stats(), merged_store.get_stats())

                with st.expander("Help / Legend", expanded=False):
                    render_map_legend()

        else:
            st.info("No edges to review. All done or adjust filters.")

    # Keyboard shortcuts
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


def main():
    """Main app entry point (standalone mode)."""
    st.set_page_config(
        page_title="Integration QA",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Reduce padding/margins for more map space
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar configuration
    with st.sidebar:
        st.title("Integration QA")
        st.header("Configuration")
        integration_dir, session = render_integration_qa_sidebar()

    # Render main content
    render_integration_qa_content(integration_dir, session)


if __name__ == "__main__":
    # Add parent to path for standalone execution
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    main()
