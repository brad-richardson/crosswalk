"""Label Review mode for reviewing and managing existing labels."""

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ..config import FEATURE_COLUMNS
from ..config import settings as default_settings
from ..matching.ml import MLMatcher
from .data_loader import CandidatePairView
from .data_store import DataStore
from .label_store import LabelStore


def get_settings():
    """Get settings instance."""
    return default_settings


def load_labeled_pairs_with_predictions(
    dataset_id: str,
    labels_dir: Path,
) -> pd.DataFrame:
    """Load labeled pairs with FRESH ML predictions (no caching).

    Uses the same pattern as evaluate_by_dataset() in ml.py:
    1. Load labels + features from disk via LabelStore.load_all()
    2. Extract features with MLMatcher._extract_features_and_labels()
    3. Impute with MLMatcher._impute_missing()
    4. Run fresh inference with model.predict_proba()

    Args:
        dataset_id: Dataset identifier to filter labels
        labels_dir: Path to labels directory

    Returns:
        DataFrame with: gers_id, target_id, label, labeler, labeled_at,
        ml_confidence, ml_decision, and all features.
    """
    # Load all labels + features for this dataset (fresh from disk)
    all_labels = LabelStore.load_all(labels_dir)

    if len(all_labels) == 0:
        return pd.DataFrame()

    # Filter to requested dataset
    df = all_labels[all_labels["dataset"] == dataset_id].copy()

    if len(df) == 0:
        return pd.DataFrame()

    # Load model and run fresh inference (same pattern as evaluate_by_dataset)
    try:
        settings = get_settings()
        model_path = settings.model_path

        if not model_path.exists():
            st.warning(f"ML model not found at {model_path}")
            df["ml_confidence"] = 0.5
            df["ml_decision"] = "unknown"
            df["ml_pred_class"] = -1
            return df

        matcher = MLMatcher(auto_select=True)
        matcher.load_model(str(model_path))

        # Extract features - reuses existing method
        X, y = matcher._extract_features_and_labels(df, binary=True)

        # Impute missing values - reuses existing method
        X = matcher._impute_missing(X)

        # Get probabilities (fresh inference, not cached)
        probs = matcher.model.predict_proba(X)

        # Get match class probability
        match_class = matcher.label_encoder.get("match", 1)
        class_indices = list(matcher.model.classes_)
        if match_class in class_indices:
            match_idx = class_indices.index(match_class)
        else:
            match_idx = 1

        df["ml_confidence"] = probs[:, match_idx]

        # Determine ML decision based on thresholds
        df["ml_decision"] = df["ml_confidence"].apply(
            lambda p: (
                "match"
                if p >= settings.optimizer_match_threshold
                else "review"
                if p >= settings.optimizer_review_threshold
                else "no_match"
            )
        )

        # Also get the predicted class for disagreement comparison
        y_pred = matcher.model.predict(X)
        df["ml_pred_class"] = y_pred  # 1=match, 0=no_match

    except Exception as e:
        st.warning(f"Could not load ML model: {e}")
        df["ml_confidence"] = 0.5
        df["ml_decision"] = "unknown"
        df["ml_pred_class"] = -1

    return df


def filter_disagreements(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to pairs where human label disagrees with ML prediction.

    Uses binary comparison (same as evaluate_by_dataset):
    - Human match: match -> 1
    - Human no_match: no_match -> 0
    - ML prediction: ml_pred_class (1=match, 0=no_match)

    Args:
        df: DataFrame with labels and ml_pred_class column

    Returns:
        Filtered DataFrame with only disagreement pairs.
    """
    if df.empty:
        return df
    # Normalize human labels to binary (same as _extract_features_and_labels)
    human_match = (df["label"] == "match").astype(int)
    ml_pred = df["ml_pred_class"]
    return df[human_match != ml_pred]


def filter_low_confidence(df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """Filter to pairs where ML confidence is uncertain (close to 0.5).

    Args:
        df: DataFrame with ml_confidence column
        threshold: Maximum distance from 0.5 to consider uncertain

    Returns:
        Filtered DataFrame with only low-confidence pairs.
    """
    if df.empty:
        return df
    distance_from_decision = (df["ml_confidence"] - 0.5).abs()
    return df[distance_from_decision <= threshold]


def render_label_review_sidebar(labels_dir: Path) -> tuple[str | None, str]:
    """Render sidebar for Label Review mode.

    Args:
        labels_dir: Path to labels directory

    Returns:
        Tuple of (selected_dataset_id, filter_type)
    """
    st.sidebar.subheader("Label Review")

    # Find datasets with labels
    human_dir = labels_dir / "human"
    available_datasets = []
    if human_dir.exists():
        for p in human_dir.iterdir():
            if p.is_dir() and p.name.startswith("dataset="):
                dataset_id = p.name.replace("dataset=", "")
                available_datasets.append(dataset_id)

    if not available_datasets:
        st.sidebar.warning("No labeled datasets found")
        return None, "all"

    available_datasets.sort()

    # Get default from query params (persists across reloads)
    # Use same 'dataset' param as labeling mode for consistency
    default_dataset = st.query_params.get("dataset", "")
    if default_dataset not in available_datasets:
        default_dataset = available_datasets[0]
    default_index = available_datasets.index(default_dataset)

    # Track previous dataset for change detection
    prev_dataset = st.session_state.get("review_prev_dataset")

    # Dataset selector
    dataset = st.sidebar.selectbox(
        "Dataset", available_datasets, index=default_index, key="review_dataset"
    )

    # Persist dataset to query params
    st.query_params["dataset"] = dataset

    # Detect dataset change to trigger data reload
    if prev_dataset != dataset:
        st.session_state.review_dataset_changed = True
        st.session_state.review_prev_dataset = dataset
        st.session_state.review_selected_pair = None  # Clear selection on dataset change

    # Get filter from query params
    filter_map = {
        "all": "All",
        "disagreements": "ML Disagreements",
        "low_confidence": "Low Confidence",
    }
    reverse_filter_map = {v: k for k, v in filter_map.items()}
    default_filter = st.query_params.get("filter", "all")
    default_filter_display = filter_map.get(default_filter, "All")

    filter_options = ["All", "ML Disagreements", "Low Confidence"]
    default_filter_index = (
        filter_options.index(default_filter_display)
        if default_filter_display in filter_options
        else 0
    )

    # Filter selector
    filter_type_display = st.sidebar.radio(
        "Filter",
        filter_options,
        index=default_filter_index,
        key="review_filter",
    )

    # Persist filter to query params
    filter_type = reverse_filter_map[filter_type_display]
    st.query_params["filter"] = filter_type

    return dataset, filter_type


def _get_label_color(label: str) -> str:
    """Get Streamlit named color for a label (for markdown syntax)."""
    if label == "match":
        return "green"
    elif label == "no_match":
        return "red"
    return "orange"


def _get_label_badge_style(label: str) -> tuple[str, str]:
    """Get badge color (hex) and display text for a label.

    Args:
        label: Label string (match, no_match, etc.)

    Returns:
        Tuple of (color hex, display text).
    """
    if label == "match":
        return "#4CAF50", "MATCH"
    elif label == "no_match":
        return "#F44336", "NO MATCH"
    return "#FF9800", label.upper()


def _get_confidence_color(confidence: float) -> str:
    """Get Streamlit named color for confidence value (for markdown syntax)."""
    if confidence >= 0.75:
        return "green"
    elif confidence >= 0.5:
        return "orange"
    return "red"


def render_label_list(df: pd.DataFrame) -> tuple[str, str] | None:
    """Render list of labeled pairs with mobile-friendly layout.

    Args:
        df: DataFrame with labels and ML predictions

    Returns:
        Selected (gers_id, target_id) tuple or None if no selection.
    """
    if df.empty:
        st.info("No labels match the current filter")
        return None

    # Prepare display dataframe
    display_df = df[
        ["gers_id", "target_id", "label", "labeler", "ml_confidence", "ml_decision"]
    ].copy()

    # Stats summary
    total = len(df)
    matches = len(df[df["label"] == "match"])
    no_matches = len(df[df["label"] == "no_match"])

    # Disagreement count
    human_match = (df["label"] == "match").astype(int)
    ml_match = (df["ml_decision"] == "match").astype(int)
    disagreements = (human_match != ml_match).sum()

    # Stats row using native Streamlit columns
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", total)
    c2.metric("Match", matches)
    c3.metric("No Match", no_matches)
    c4.metric("Disagree", disagreements)

    st.divider()

    # Render each label as a card using Streamlit containers
    for idx, row in display_df.iterrows():
        gers_id = row["gers_id"]
        target_id = str(row["target_id"])
        label = row["label"]
        ml_decision = row["ml_decision"]
        ml_confidence = row["ml_confidence"]
        labeler = row.get("labeler", "unknown")

        # Check for disagreement
        human_is_match = label == "match"
        ml_is_match = ml_decision == "match"
        is_disagree = human_is_match != ml_is_match

        # Truncate IDs for display
        gers_short = gers_id[:24] + "..." if len(gers_id) > 24 else gers_id
        target_short = target_id[:20] if len(target_id) > 20 else target_id

        # Use container for card-like grouping
        with st.container():
            # Header row: label + disagree indicator
            label_color = _get_label_color(label)
            conf_color = _get_confidence_color(ml_confidence)

            header = f"**:{label_color}[{label.upper()}]**"
            if is_disagree:
                header += "  :orange[DISAGREE]"

            st.markdown(header)

            # Info in compact format
            st.caption(
                f"**Ref:** `{gers_short}`  \n"
                f"**Target:** `{target_short}`  \n"
                f"**ML:** :{conf_color}[{ml_decision} ({ml_confidence * 100:.0f}%)]  |  "
                f"**By:** {labeler}"
            )

            if st.button("View", key=f"view_{idx}", use_container_width=True):
                return gers_id, target_id

            st.divider()

    return None


def build_candidate_pair_view(
    gers_id: str,
    target_id: str,
    label_row: pd.Series,
    features: dict,
    pair_data: dict,
    ml_confidence: float,
) -> CandidatePairView:
    """Build a CandidatePairView from stored label/feature/geometry data.

    Reuses the existing CandidatePairView class from data_loader.py so we can
    reuse create_comparison_map() and render_feature_panel().

    Args:
        gers_id: Reference segment ID
        target_id: Target segment ID
        label_row: Row from labels DataFrame
        features: Feature dict for the pair
        pair_data: Geometry and attribute data from DataStore
        ml_confidence: ML model confidence score

    Returns:
        CandidatePairView instance for rendering.
    """
    from shapely.ops import substring

    ref_geom = pair_data.get("ref_geometry")
    target_geom = pair_data.get("target_geometry")

    # Get alignment fractions from label (default to full segment)
    ref_start = label_row.get("ref_start_pct", 0.0) or 0.0
    ref_end = label_row.get("ref_end_pct", 1.0) or 1.0
    target_start = label_row.get("target_start_pct", 0.0) or 0.0
    target_end = label_row.get("target_end_pct", 1.0) or 1.0

    # Create aligned sublines if partial alignment
    ref_aligned = None
    target_aligned = None
    if ref_geom and (ref_start > 0.01 or ref_end < 0.99):
        ref_aligned = substring(ref_geom, ref_start, ref_end, normalized=True)
    if target_geom and (target_start > 0.01 or target_end < 0.99):
        target_aligned = substring(target_geom, target_start, target_end, normalized=True)

    # Determine decision from confidence
    settings = get_settings()
    if ml_confidence >= settings.optimizer_match_threshold:
        decision = "match"
    elif ml_confidence >= settings.optimizer_review_threshold:
        decision = "review"
    else:
        decision = "no_match"

    return CandidatePairView(
        ref_id=gers_id,
        target_id=target_id,
        ref_geometry=ref_geom,
        target_geometry=target_geom,
        ref_name=pair_data.get("ref_name"),
        target_name=pair_data.get("target_name"),
        ref_class=pair_data.get("ref_class"),
        target_class=pair_data.get("target_class"),
        ref_subclass=pair_data.get("ref_subclass"),
        target_subclass=pair_data.get("target_subclass"),
        decision=decision,
        confidence=ml_confidence,
        features=features or {},
        ref_aligned_geometry=ref_aligned,
        target_aligned_geometry=target_aligned,
        ref_start_frac=ref_start,
        ref_end_frac=ref_end,
        target_start_frac=target_start,
        target_end_frac=target_end,
    )


def render_label_detail(
    gers_id: str,
    target_id: str,
    dataset_id: str,
    labels_dir: Path,
    labeler_name: str,
    all_data_df: pd.DataFrame,
) -> bool:
    """Render detail view for a single pair with mobile-friendly layout.

    Reuses existing components:
    - create_comparison_map() from map_view.py
    - render_feature_panel() from feature_panel.py

    Args:
        gers_id: Reference segment ID
        target_id: Target segment ID
        dataset_id: Dataset identifier
        labels_dir: Path to labels directory
        labeler_name: Name of current labeler (for updates)
        all_data_df: DataFrame with all loaded data including ML predictions

    Returns:
        True if user clicked back button.
    """
    from .feature_panel import render_feature_panel
    from .map_view import create_comparison_map

    # Back button at top
    if st.button("< Back to List", use_container_width=True):
        return True

    label_store = LabelStore(dataset_id, labels_dir)
    ds = DataStore(dataset_id, data_dir=labels_dir / "data")

    # Get the row from already-loaded data (has ML predictions)
    mask = (all_data_df["gers_id"] == gers_id) & (all_data_df["target_id"] == str(target_id))
    if not mask.any():
        st.error(f"Label not found: {gers_id} / {target_id}")
        return False

    row = all_data_df[mask].iloc[-1]

    # Get geometry from DataStore
    pair_data = ds.get_pair(gers_id, target_id)
    if not pair_data:
        st.warning("Geometry data not available for this pair")
        pair_data = {}

    # Extract features from the row (all FEATURE_COLUMNS)
    features = {col: row.get(col, 0) for col in FEATURE_COLUMNS if col in row.index}

    # Build CandidatePairView for reusing map/feature panel
    ml_conf = row.get("ml_confidence", 0.5)
    pair_view = build_candidate_pair_view(gers_id, target_id, row, features, pair_data, ml_conf)

    # Current label summary card at top (mobile-friendly)
    human_label = row["label"]
    ml_decision = row.get("ml_decision", "unknown")
    labeler = row.get("labeler", "unknown")

    # Check for disagreement
    human_is_match = human_label == "match"
    ml_is_match = ml_decision == "match"
    is_disagree = human_is_match != ml_is_match

    badge_color, badge_text = _get_label_badge_style(human_label)
    conf_color = _get_confidence_color(ml_conf)
    disagree_html = (
        '<div style="background: #FF9800; color: #000; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; margin-top: 8px;">DISAGREE WITH ML</div>'
        if is_disagree
        else ""
    )

    st.markdown(
        f"""
        <style>
        .detail-summary {{
            background: #1E1E1E;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }}
        .detail-summary-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            align-items: center;
        }}
        .detail-summary-item {{
            flex: 1;
            min-width: 120px;
        }}
        .detail-summary-label {{
            font-size: 11px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 4px;
        }}
        .detail-summary-value {{
            font-size: 16px;
            color: #fff;
        }}
        .detail-badge {{
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: bold;
            color: white;
            display: inline-block;
        }}
        @media (max-width: 768px) {{
            .detail-summary-item {{
                min-width: 100px;
            }}
            .detail-summary-value {{
                font-size: 14px;
            }}
        }}
        </style>
        <div class="detail-summary">
            <div class="detail-summary-row">
                <div class="detail-summary-item">
                    <div class="detail-summary-label">Human Label</div>
                    <div class="detail-summary-value">
                        <span class="detail-badge" style="background: {badge_color}">{badge_text}</span>
                    </div>
                </div>
                <div class="detail-summary-item">
                    <div class="detail-summary-label">ML Prediction</div>
                    <div class="detail-summary-value" style="color: {conf_color}">{ml_decision} ({ml_conf * 100:.0f}%)</div>
                </div>
                <div class="detail-summary-item">
                    <div class="detail-summary-label">Labeled By</div>
                    <div class="detail-summary-value">{labeler}</div>
                </div>
            </div>
            {disagree_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Map - full width for better mobile viewing
    if pair_data.get("ref_geometry") and pair_data.get("target_geometry"):
        m = create_comparison_map(pair_view)
        map_html = m.get_root().render()
        components.html(map_html, height=400)
    else:
        st.warning("Geometry data not available")

    # Helper to invalidate cache after label changes
    def _invalidate_cache():
        cache_key = f"review_data_{dataset_id}"
        if cache_key in st.session_state:
            del st.session_state[cache_key]

    # Action buttons - prominent, full width on mobile
    st.markdown("### Update Label")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Match", key="update_match", use_container_width=True, type="primary"):
            label_store.update_label(gers_id, target_id, "match", labeler_name)
            _invalidate_cache()
            st.success("Updated to match")
            st.rerun()
    with c2:
        if st.button("No Match", key="update_no_match", use_container_width=True):
            label_store.update_label(gers_id, target_id, "no_match", labeler_name)
            _invalidate_cache()
            st.success("Updated to no_match")
            st.rerun()
    with c3:
        if st.button("Unsure", key="update_unsure", use_container_width=True):
            label_store.update_label(gers_id, target_id, "unsure", labeler_name)
            _invalidate_cache()
            st.success("Updated to unsure")
            st.rerun()

    # Feature details in expander (less prominent on mobile)
    with st.expander("Feature Details", expanded=False):
        render_feature_panel(pair_view)

    # Delete in expander with confirmation
    with st.expander("Delete Label", expanded=False):
        st.warning("This action cannot be undone.")
        if st.checkbox("I understand, delete this label", key="confirm_delete"):
            if st.button(
                "Delete Label Permanently",
                type="primary",
                key="delete_label",
                use_container_width=True,
            ):
                label_store.delete_label(gers_id, target_id)
                _invalidate_cache()
                st.success("Label deleted")
                return True  # Go back to list

    return False
