"""Comparison view for evaluating labeler agreement."""

from typing import Optional

import pandas as pd
import streamlit as st

from .label_store import LabelStore


def render_comparison_view(label_store: LabelStore) -> None:
    """Render the comparison view showing labeler agreement."""
    st.header("Labeler Comparison")

    df = label_store.df
    if df.empty:
        st.warning("No labels yet. Label some pairs first!")
        return

    # Get unique labelers
    labelers = sorted(df["labeler"].unique())

    if len(labelers) < 2:
        st.info(f"Only one labeler found ({labelers[0]}). Need at least 2 labelers to compare.")
        return

    # Labeler selection
    col1, col2 = st.columns(2)
    with col1:
        labeler_a = st.selectbox("Labeler A", labelers, index=0)
    with col2:
        remaining = [l for l in labelers if l != labeler_a]
        labeler_b = st.selectbox("Labeler B", remaining, index=0) if remaining else None

    if not labeler_b:
        st.warning("Select two different labelers")
        return

    # Filter to selected labelers, keep only most recent label per pair
    df_a = df[df["labeler"] == labeler_a].sort_values("labeled_at").drop_duplicates(
        subset=["ref_id", "target_id"], keep="last"
    ).set_index(["ref_id", "target_id"])
    df_b = df[df["labeler"] == labeler_b].sort_values("labeled_at").drop_duplicates(
        subset=["ref_id", "target_id"], keep="last"
    ).set_index(["ref_id", "target_id"])

    # Find common pairs
    common_pairs = df_a.index.intersection(df_b.index)

    st.divider()

    # Summary stats
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(f"{labeler_a} labels", len(df_a))
    with col2:
        st.metric(f"{labeler_b} labels", len(df_b))
    with col3:
        st.metric("Common pairs", len(common_pairs))

    if len(common_pairs) == 0:
        st.info("No pairs labeled by both labelers yet.")
        return

    # Calculate agreement
    agreements = []
    disagreements = []

    for pair in common_pairs:
        label_a = str(df_a.loc[pair, "label"])
        label_b = str(df_b.loc[pair, "label"])

        # Normalize labels for comparison (treat match_1n as match)
        norm_a = "match" if "match" in label_a else label_a
        norm_b = "match" if "match" in label_b else label_b

        if norm_a == norm_b:
            agreements.append((pair, label_a, label_b))
        else:
            disagreements.append((pair, label_a, label_b))

    agreement_rate = len(agreements) / len(common_pairs) * 100

    # Display agreement rate
    st.subheader("Agreement")
    col1, col2, col3 = st.columns(3)

    with col1:
        color = "#4CAF50" if agreement_rate >= 80 else "#FF9800" if agreement_rate >= 60 else "#F44336"
        st.markdown(
            f"""
            <div style="text-align: center;">
                <span style="font-size: 48px; font-weight: bold; color: {color};">
                    {agreement_rate:.0f}%
                </span>
                <br>
                <span style="color: #666;">Agreement Rate</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.metric("Agree", len(agreements))

    with col3:
        st.metric("Disagree", len(disagreements))

    st.divider()

    # Show disagreements
    if disagreements:
        st.subheader(f"Disagreements ({len(disagreements)})")

        # Create disagreement dataframe
        disagree_data = []
        for (ref_id, target_id), label_a, label_b in disagreements:
            row_a = df_a.loc[(ref_id, target_id)]
            confidence = row_a["original_confidence"] if "original_confidence" in row_a.index else 0
            original = row_a["original_decision"] if "original_decision" in row_a.index else ""
            disagree_data.append({
                "ref_id": ref_id[:12] + "..." if len(ref_id) > 15 else ref_id,
                "target_id": str(target_id)[:12] + "..." if len(str(target_id)) > 15 else target_id,
                labeler_a: label_a,
                labeler_b: label_b,
                "confidence": f"{confidence:.0%}" if isinstance(confidence, (int, float)) else str(confidence),
                "original": str(original),
            })

        disagree_df = pd.DataFrame(disagree_data)
        st.dataframe(disagree_df, use_container_width=True, hide_index=True)

        # Option to review disagreements
        st.markdown("---")
        if st.button("Review Disagreements in Labeling UI"):
            # Store disagreement pairs for filtering
            st.session_state.review_disagreements = [
                (ref_id, target_id) for (ref_id, target_id), _, _ in disagreements
            ]
            st.session_state.show_comparison = False
            st.rerun()

    # Show agreement breakdown by label type
    st.subheader("Agreement by Label Type")

    # Create confusion matrix
    label_types = ["match", "no_match", "unsure", "maybe", "skip"]
    matrix_data = {la: {lb: 0 for lb in label_types} for la in label_types}

    for pair in common_pairs:
        label_a = str(df_a.loc[pair, "label"])
        label_b = str(df_b.loc[pair, "label"])
        # Normalize match_1n to match
        if "match" in label_a:
            label_a = "match"
        if "match" in label_b:
            label_b = "match"
        if label_a in matrix_data and label_b in label_types:
            matrix_data[label_a][label_b] += 1

    # Filter to only show labels that exist
    used_labels = set()
    for pair in common_pairs:
        la = str(df_a.loc[pair, "label"])
        lb = str(df_b.loc[pair, "label"])
        used_labels.add("match" if "match" in la else la)
        used_labels.add("match" if "match" in lb else lb)

    used_labels = sorted(used_labels)

    if used_labels:
        matrix_df = pd.DataFrame(
            [[matrix_data[la][lb] for lb in used_labels] for la in used_labels],
            index=[f"{labeler_a}: {l}" for l in used_labels],
            columns=[f"{labeler_b}: {l}" for l in used_labels],
        )
        st.dataframe(matrix_df, use_container_width=True)

    # Label distribution per labeler
    st.subheader("Label Distribution")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{labeler_a}**")
        dist_a = df_a["label"].value_counts()
        for label, count in dist_a.items():
            st.text(f"  {label}: {count}")

    with col2:
        st.markdown(f"**{labeler_b}**")
        dist_b = df_b["label"].value_counts()
        for label, count in dist_b.items():
            st.text(f"  {label}: {count}")
