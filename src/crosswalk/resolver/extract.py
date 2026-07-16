"""Extract a per-edge keep/drop training table from group sidecars + labels.

Data reality (verified 2026-07, see the prototype writeup + the round-3 design
``research/learned_optimizer_design.md`` §3):

* Since PR #344 the ``*_groups.json`` sidecar carries, per group, a
  **``candidate_edges``** list: every floor-passing candidate pair in the
  group's connected component, **uncapped**, each as
  ``{ref_id, target_id, confidence, selected[, selected_elsewhere][, pruned]}``
  (``pipeline/runner.py::_compute_candidate_graph_by_group``). This is the
  canonical candidate universe the design's stage-1 persistence contract
  defines, and it is what makes UNDER-selection learnable at the pair level:
  ``selected=False`` candidates that were *not* selected elsewhere are the
  optimizer-drop negatives that the selected-only sidecar could never expose.
  This module consumes ``candidate_edges`` as the universe when present.
* Legacy fallback: the older M2 sidecar persisted only the selected assignment
  (``edges``, ~99.9% ``selected=True``) plus a **capped** (64/group)
  ``rejected_edges`` list. Sidecars predating #344 (or written with
  ``stitch_persist_candidate_graph=False``, which emits an empty list) have no
  usable ``candidate_edges``; for them the table is built from ``edges`` +
  ``rejected_edges`` as before, with a logged warning that the uncapped
  under-selection universe is unavailable (the 64-cap and the exclusion of
  pairs selected in other groups mean under-selection labels are only partially
  observable on that path).
* ``candidate_edges`` carries only topology + confidence + selection flags — it
  is the design's *stage-1* layer. The per-edge structural layer (degree,
  bridge, corridor ids, alignment fracs) and the 78 typed pair features are the
  *stage-2* join target (design §3.2), keyed by ``(group_id, ref_id,
  target_id)``. Until that parquet lands, this builder enriches each candidate
  edge with the structural fields already present on the group's ``edges`` /
  ``rejected_edges`` records (matched by ``(ref_id, target_id)``); genuinely new
  candidates — visible only in the uncapped ``candidate_edges`` — get NaN /
  default structural values (honest: stage 2 fills them via FeatureStore).
* The 78 pairwise ML features live in ``labels/features/`` keyed by
  ``(gers_id, target_id)`` but only for *pair*-labeled pairs — coverage of
  group edges is ~5%. So per-edge features today come from the sidecar itself.
* Ground truth = curated ``labels/stitching/`` ``selected_edges``. Labels map
  to current sidecar groups by edge overlap (group_id churns on any component
  shift) via ``stitch_eval.recover_labeled_groups`` — reused here verbatim.
* **Empty-set (reject-all) labels** — pair-semantics rows with
  ``selected_edges == []`` (human-confirmed now; historical panel-NONE rows
  remain readable) — carry no edges to overlap on, so they map by
  verbatim ``group_id`` only (same rule as ``stitch_eval.
  recover_empty_reject_all``). On the candidate-graph path they emit every
  (rule-5-filtered, non-``selected_elsewhere``) candidate edge with ``keep=0``
  — the "select nothing" group shape the cross-mode defect needs. On the
  legacy path they emit ZERO rows: the label asserts "nothing in the full
  candidate set should be kept", and the capped ``edges``+``rejected_edges``
  view cannot represent that full set faithfully, so emitting partial
  negatives would understate the group's reject-all semantics silently.

The row key is ``(group_id, ref_id, target_id)`` — the same key the design's
stage-2 ``candidates.parquet`` feature columns join on, so no competing scheme
is introduced.

Provenance is preserved on every row so the eval can slice by dataset,
labeler, and clean-vs-split-vs-empty mapping. Multiple historical labels can
recover onto the same current group. Equal per-edge labels are deduplicated
across distinct historical labels while their human group ids remain in the
``historical_human_group_ids`` audit column. A repeated candidate inside one
historical label is malformed and fails closed. If any current candidate key
receives contradictory ``keep`` values, the entire current group is
quarantined from training and evaluation. Each build logs a one-line stats
summary (rows, label collisions, rule-5 drops, ``selected_elsewhere``
exclusions, and legacy-known omission occurrences) and attaches the same dict
as ``df.attrs["build_stats"]``. The exact collision and omission identities are
attached separately as the deterministic, JSON-serializable
``df.attrs["build_audit"]`` so deferred adjudication does not require
intercepting private helpers or reconstructing discarded intermediate frames.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from crosswalk.agent_labeling.stitch_eval import recover_labeled_groups
from crosswalk.config import FEATURE_COLUMNS as PAIRWISE_FEATURE_COLUMNS

EDGE_LABEL_COL = "keep"

# Schema version for the per-edge training TABLE produced by
# :func:`build_edge_table` (its column set + how columns are populated). This is
# distinct from the ``build_audit`` dict's own ``schema_version`` (that versions
# the audit payload) and from the resolver FEATURE version (that versions the
# model's feature contract). Bump when the emitted column set or a column's
# meaning changes. A per-build column-set hash is recorded alongside it so silent
# column accretion (e.g. via the candidate-parquet join, see
# CANDIDATE_EXCLUDE_FROM_JOIN) is detectable in the audit even between bumps.
RESOLVER_TABLE_SCHEMA_VERSION = 1

# Row key = the design's stage-2 feature-parquet join key (§3.2). Kept stable so
# FeatureStore columns can be merged later without introducing a competing
# scheme.
KEY_COLUMNS = ("dataset_id", "group_id", "ref_id", "target_id")

# Columns from the typed candidates parquet that are safe to bring into the
# training table. Excludes ground-truth label columns.
CANDIDATE_EXCLUDE_FROM_JOIN = {
    "human_group_id",
    "labeler",
    "provenance",
    EDGE_LABEL_COL,
}

# Structural / provenance columns that exist in both the JSON sidecar row
# and the parquet; parquet is authoritative when present (runtime parity).
STRUCTURAL_OVERLAP_TO_ENRICH = {
    "confidence",
    "degree_ref",
    "degree_tgt",
    "is_bridge",
    "candidate_graph_bridge",
    "is_sliver",
    "biconnected_block",
    "corridor_ref",
    "corridor_tgt",
    "gers_start_frac",
    "gers_end_frac",
    "local_start_frac",
    "local_end_frac",
    "n_edges",
    "n_candidate_edges",
    "n_corridors",
    "n_assignment_components",
    "largest_biconnected_block",
    "oversized_group",
}

PARQUET_JOIN_KEYS = ("group_id", "ref_id", "target_id")

# Candidate-parquet columns we KNOW about and expect to see joined onto the
# training table (design §3.2): the 78 typed pairwise features + signed lateral
# offset + class/length/optimizer-decision context, plus the structural overlap
# fields. This is NOT a hard allowlist — any parquet column outside this set is
# still joined (the exclusion-list mechanism in CANDIDATE_EXCLUDE_FROM_JOIN is
# preserved). It is a KNOWN set used only to emit a loud warning when a new,
# unrecognized column silently enters the table, so accretion becomes visible.
EXPECTED_CANDIDATE_JOIN_COLUMNS = (
    set(PAIRWISE_FEATURE_COLUMNS)
    | set(STRUCTURAL_OVERLAP_TO_ENRICH)
    | {
        "lateral_offset_signed_m",
        "ref_class",
        "target_class",
        "ref_length_m",
        "target_length_m",
        "optimizer_decision",
        "optimizer_reason",
    }
)


def _human_edge_set(selected_edges_raw) -> frozenset[tuple[str, str]]:
    """Parse a stitching label's ``selected_edges`` JSON to an edge frozenset.

    Local copy of the tiny parser (rather than importing the private
    ``stitch_eval._human_edge_set``) so this research harness does not depend on
    an internal API that may change without notice.
    """
    try:
        edges = json.loads(selected_edges_raw)
    except (ValueError, TypeError):
        return frozenset()
    return frozenset((str(e["ref_id"]), str(e["target_id"])) for e in edges)


def load_sidecar_groups(path: str | Path) -> list[dict]:
    """Load the ``groups`` list from a ``*_groups.json`` sidecar."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        return data.get("groups", [])
    return data


def load_stitching_labels(path: str | Path) -> pd.DataFrame:
    """Load a curated stitching label CSV (group_id kept as str)."""
    return pd.read_csv(path, dtype={"group_id": str})


def _edge_key(edge: dict) -> tuple[str, str]:
    return (str(edge["ref_id"]), str(edge["target_id"]))


def _group_has_candidate_graph(group: dict) -> bool:
    """True iff the group carries a usable #344 ``candidate_edges`` universe.

    The key is present-but-empty when ``stitch_persist_candidate_graph=False``
    (the flag emits ``candidate_edges: []``); every real group has >= 1 edge, so
    an empty list is treated as "feature disabled" and routed to the legacy
    fallback.
    """
    return bool(group.get("candidate_edges"))


def discover_candidates_parquet(groups_path: str | Path) -> Path | None:
    """Find the typed ``*_candidates.parquet`` for a given groups.json.

    Tries, in order:
    1. Factory layout: ``<dataset_dir>/candidates.parquet`` (same dir as groups.json)
    2. Output layout: ``<stem>_groups.json → <stem>_candidates.parquet``
    3. Via ``candidates_sidecar_path`` from a reconstructed bridge path.

    Returns None if no file exists.
    """
    groups_path = Path(groups_path)

    # 1. Factory: data/factory/release=.../dataset=<name>/groups.json -> candidates.parquet
    factory_candidate = groups_path.parent / "candidates.parquet"
    if factory_candidate.exists():
        return factory_candidate

    # 2. Standard output: us_boston_streets_groups.json -> us_boston_streets_candidates.parquet
    stem = groups_path.stem
    if stem.endswith("_groups"):
        cand_stem = stem[: -len("_groups")] + "_candidates"
    else:
        cand_stem = stem + "_candidates"
    candidate = groups_path.parent / f"{cand_stem}.parquet"
    if candidate.exists():
        return candidate

    # 3. Reconstruct bridge path and use the canonical helper
    try:
        from ..filenames import candidates_sidecar_path

        if stem.endswith("_groups"):
            bridge_stem = stem[: -len("_groups")] + "_bridge"
        else:
            bridge_stem = stem
        bridge_path = groups_path.parent / f"{bridge_stem}.parquet"
        cand_via_bridge = candidates_sidecar_path(bridge_path)
        if cand_via_bridge.exists():
            return cand_via_bridge
    except Exception:
        pass

    return None


def _normalize_candidate_keys(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Validate and string-normalize candidate join keys without mutating input."""
    missing = [key for key in PARQUET_JOIN_KEYS if key not in df.columns]
    if missing:
        raise ValueError(f"{source} missing required key column(s): {', '.join(missing)}")

    null_counts = {key: int(df[key].isna().sum()) for key in PARQUET_JOIN_KEYS}
    null_counts = {key: count for key, count in null_counts.items() if count}
    if null_counts:
        raise ValueError(f"{source} has null join keys: {null_counts}")

    normalized = df.copy()
    for key in PARQUET_JOIN_KEYS:
        normalized[key] = normalized[key].astype(str)

    duplicate_count = int(normalized.duplicated(subset=list(PARQUET_JOIN_KEYS)).sum())
    if duplicate_count:
        raise ValueError(f"{source} has {duplicate_count} duplicate key(s) on {PARQUET_JOIN_KEYS}")
    return normalized


def load_candidates_parquet(path: str | Path) -> pd.DataFrame:
    """Load a typed candidates parquet (83 features + signed lateral offset).

    Validates basic expectations: non-empty, has join keys, unique keys.
    Returns the raw frame with string-typed join keys for safe merging.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"candidates parquet not found: {path}")

    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"candidates parquet {path} is empty")
    return _normalize_candidate_keys(df, f"candidates parquet {path}")


def _enrich_with_candidate_parquet(
    df: pd.DataFrame,
    candidates_df: pd.DataFrame | None,
    stats: dict,
) -> pd.DataFrame:
    """Left-join typed candidate features onto the per-edge training table.

    Design §3.2: parquet is the canonical typed substrate with 83 FEATURE_COLUMNS
    + lateral_offset_signed_m + structural context + provenance. This joins on
    (group_id, ref_id, target_id) and:
    * adds any new columns (e.g. 83 FEATURE_COLUMNS, lateral_offset_signed_m,
      ref_class, target_class, optimizer_decision, etc.)
    * for overlapping structural columns (confidence, degree_*, is_bridge, etc.)
      fills NaN/defaults from parquet where authoritative.

    Returns enriched df; updates stats with enrichment counters.
    """
    # pandas merge/copy operations do not provide a sufficiently stable attrs
    # preservation contract for audit metadata. Keep it explicitly so callers
    # can attach collision/build stats before enrichment without losing them.
    input_attrs = dict(df.attrs)
    if df.empty or candidates_df is None or candidates_df.empty:
        stats["candidate_parquet_rows"] = len(candidates_df) if candidates_df is not None else 0
        stats["candidate_parquet_enriched"] = 0
        stats["candidate_parquet_missing_keys"] = 0
        stats["candidate_parquet_unexpected_columns"] = []
        df.attrs = input_attrs
        return df

    candidates_df = _normalize_candidate_keys(candidates_df, "candidates_df")
    df = df.copy()
    for key in PARQUET_JOIN_KEYS:
        df[key] = df[key].astype(str)

    # Columns we never overwrite from parquet (ground truth / provenance)
    exclude_from_join = CANDIDATE_EXCLUDE_FROM_JOIN | {"dataset_id"}

    # Split candidate columns into new vs overlapping
    new_cols = [
        c
        for c in candidates_df.columns
        if c not in df.columns and c not in exclude_from_join and c not in PARQUET_JOIN_KEYS
    ]

    # Make silent column accretion visible: the join adds ANY new parquet column
    # (exclusion-list mechanism, not an allowlist), so a schema drift upstream
    # can quietly widen the training table. Warn loudly + record the offenders in
    # stats when a joined column is not in the known/expected set.
    unexpected_cols = sorted(c for c in new_cols if c not in EXPECTED_CANDIDATE_JOIN_COLUMNS)
    stats["candidate_parquet_unexpected_columns"] = unexpected_cols
    if unexpected_cols:
        logger.warning(
            "_enrich_with_candidate_parquet: {} unrecognized candidate-parquet "
            "column(s) joined into the training table (not in "
            "EXPECTED_CANDIDATE_JOIN_COLUMNS): {}. If intended, add them to the "
            "expected set (and bump RESOLVER_TABLE_SCHEMA_VERSION); otherwise this "
            "is silent schema accretion.",
            len(unexpected_cols),
            unexpected_cols,
        )
    overlapping_authoritative = [
        c
        for c in candidates_df.columns
        if c in df.columns
        and c not in PARQUET_JOIN_KEYS
        and c not in exclude_from_join
        and c in STRUCTURAL_OVERLAP_TO_ENRICH
    ]

    # Track how many keys in df had no match in parquet
    df_keys = set(zip(df["group_id"], df["ref_id"], df["target_id"]))
    cand_keys = set(
        zip(
            candidates_df["group_id"].astype(str),
            candidates_df["ref_id"].astype(str),
            candidates_df["target_id"].astype(str),
        )
    )
    stats["candidate_parquet_rows"] = len(candidates_df)
    stats["candidate_parquet_missing_keys"] = len(df_keys - cand_keys)

    # 1. Add new columns via left merge
    if new_cols:
        merge_cols = list(PARQUET_JOIN_KEYS) + new_cols
        df = df.merge(candidates_df[merge_cols], on=list(PARQUET_JOIN_KEYS), how="left")

    # 2. Enrich overlapping structural columns from parquet (parquet authoritative for runtime parity)
    if overlapping_authoritative:
        merge_cols = list(PARQUET_JOIN_KEYS) + overlapping_authoritative
        df = df.merge(
            candidates_df[merge_cols],
            on=list(PARQUET_JOIN_KEYS),
            how="left",
            suffixes=("", "_pq"),
        )
        for col in overlapping_authoritative:
            pq_col = f"{col}_pq"
            if pq_col not in df.columns:
                continue
            # Parquet is runtime-authoritative (computed on full candidate graph),
            # so prefer it whenever present (not just placeholder fill).
            # This closes the gap where JSON sidecar had default 0/-1 for genuinely new candidates.
            df[col] = (
                df[pq_col].where(df[pq_col].notna(), df[col]) if col in df.columns else df[pq_col]
            )
            df = df.drop(columns=[pq_col])

    stats["candidate_parquet_enriched"] = len(df_keys & cand_keys)
    df.attrs = input_attrs

    return df


def _resolve_duplicate_label_collisions(
    df: pd.DataFrame,
    stats: dict[str, int],
) -> tuple[pd.DataFrame, set[tuple[str, str]], list[dict]]:
    """Fail closed when historical labels disagree on a current group.

    Group-id churn can cause several historical labels to recover onto the same
    current sidecar group. That emits repeated :data:`KEY_COLUMNS` rows. A
    repeated key with one unambiguous ``keep`` value is safe to stable-dedupe:
    its candidate/runtime fields all came from the same current group edge.
    If a repeated key has both keep values, however, choosing a label or unioning
    their selected edges would manufacture ground truth. Quarantine every row
    for that current ``(dataset_id, group_id)`` instead.

    ``duplicate_rows`` counts surplus raw rows beyond one per duplicated key;
    ``quarantined_rows`` counts all raw rows removed with conflicting groups.
    Duplicate keys emitted by the *same* human label are not historical-label
    collisions: they mean the candidate source itself is malformed, so they
    raise instead of being silently collapsed.
    """
    collision_defaults = {
        "duplicate_rows": 0,
        "duplicate_keys": 0,
        "conflicting_keys": 0,
        "quarantined_groups": 0,
        "quarantined_rows": 0,
        "deduplicated_rows": 0,
    }
    stats.update(collision_defaults)
    if df.empty:
        return df, set(), []

    key_columns = list(KEY_COLUMNS)
    source_columns = [*key_columns, "human_group_id"]
    within_source_duplicate = df.duplicated(subset=source_columns, keep=False)
    if within_source_duplicate.any():
        examples = (
            df.loc[within_source_duplicate, source_columns]
            .drop_duplicates()
            .head(3)
            .to_dict("records")
        )
        raise ValueError(
            "candidate rows repeat within one historical human_group_id; "
            f"refusing malformed candidate universe (examples={examples})"
        )

    lineage_by_key = (
        df.groupby(key_columns, sort=False, dropna=False)["human_group_id"]
        .agg(lambda values: sorted({str(value) for value in values}))
        .to_dict()
    )
    duplicated = df.duplicated(subset=key_columns, keep=False)
    if not duplicated.any():
        df = df.copy()
        df["historical_human_group_ids"] = df.apply(
            lambda row: json.dumps(lineage_by_key[tuple(row[column] for column in key_columns)]),
            axis=1,
        )
        return df, set(), []

    duplicate_sizes = df.loc[duplicated].groupby(key_columns, sort=False, dropna=False).size()
    stats["duplicate_keys"] = len(duplicate_sizes)
    stats["duplicate_rows"] = int(duplicate_sizes.sum() - len(duplicate_sizes))

    keep_counts = (
        df.loc[duplicated]
        .groupby(key_columns, sort=False, dropna=False)[EDGE_LABEL_COL]
        .nunique(dropna=False)
    )
    conflicting = keep_counts[keep_counts > 1].reset_index()[key_columns]
    stats["conflicting_keys"] = len(conflicting)

    quarantined_group_keys: set[tuple[str, str]] = set()
    collision_audit: list[dict] = []
    if not conflicting.empty:
        group_columns = ["dataset_id", "group_id"]
        quarantined_groups = conflicting[group_columns].drop_duplicates()
        quarantined_group_keys = {
            (str(row.dataset_id), str(row.group_id))
            for row in quarantined_groups.itertuples(index=False)
        }
        group_index = pd.MultiIndex.from_frame(df[group_columns])
        quarantined_index = pd.MultiIndex.from_frame(quarantined_groups)
        quarantine_mask = group_index.isin(quarantined_index)
        stats["quarantined_groups"] = len(quarantined_groups)
        stats["quarantined_rows"] = int(quarantine_mask.sum())

        # Preserve the exact contradictory claims before removing the group.
        # All values are converted to JSON-native scalars and every collection
        # is sorted, so this audit is byte-stable across input row order.
        for dataset_value, group_value in sorted(quarantined_group_keys):
            group_rows = df.loc[
                (df["dataset_id"].astype(str) == dataset_value)
                & (df["group_id"].astype(str) == group_value)
            ]
            historical_ids = sorted({str(value) for value in group_rows["human_group_id"]})
            group_conflicts = conflicting.loc[
                (conflicting["dataset_id"].astype(str) == dataset_value)
                & (conflicting["group_id"].astype(str) == group_value)
            ]
            conflict_records: list[dict] = []
            for conflict_row in group_conflicts.sort_values(
                ["ref_id", "target_id"], kind="stable"
            ).itertuples(index=False):
                ref_id = str(conflict_row.ref_id)
                target_id = str(conflict_row.target_id)
                claims_df = group_rows.loc[
                    (group_rows["ref_id"].astype(str) == ref_id)
                    & (group_rows["target_id"].astype(str) == target_id)
                ]
                claims = sorted(
                    (
                        {
                            "human_group_id": str(claim.human_group_id),
                            "provenance": str(claim.provenance),
                            "keep": int(claim.keep),
                        }
                        for claim in claims_df.itertuples(index=False)
                    ),
                    key=lambda claim: (
                        claim["human_group_id"],
                        claim["provenance"],
                        claim["keep"],
                    ),
                )
                conflict_records.append(
                    {
                        "ref_id": ref_id,
                        "target_id": target_id,
                        "claims": claims,
                    }
                )
            collision_audit.append(
                {
                    "case_id": (f"label_collision:{dataset_value}:" + "+".join(historical_ids)),
                    "dataset_id": dataset_value,
                    "current_group_id": group_value,
                    "historical_human_group_ids": historical_ids,
                    "raw_row_occurrences": int(len(group_rows)),
                    "conflicting_key_count": int(len(conflict_records)),
                    "conflicting_edges": conflict_records,
                }
            )
        df = df.loc[~quarantine_mask].copy()
        logger.warning(
            "build_edge_table: quarantined {} current group(s) / {} raw row(s) "
            "because {} duplicate candidate key(s) have conflicting keep labels",
            stats["quarantined_groups"],
            stats["quarantined_rows"],
            stats["conflicting_keys"],
        )

    before_dedupe = len(df)
    df = df.drop_duplicates(subset=key_columns, keep="first").copy()
    stats["deduplicated_rows"] = before_dedupe - len(df)
    df["historical_human_group_ids"] = df.apply(
        lambda row: json.dumps(lineage_by_key[tuple(row[column] for column in key_columns)]),
        axis=1,
    )
    return df, quarantined_group_keys, collision_audit


def _audit_text(value) -> str:
    """Return a JSON-safe string for optional label/sidecar metadata."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    return str(value)


def _complete_collision_audit(
    collision_audit: list[dict],
    human_by: dict[str, pd.Series],
    mapped: list[tuple[str, str, str]],
) -> list[dict]:
    """Add stable historical-label metadata to collision records."""
    provenance_by_mapping = {
        (str(hgid), str(group_id)): provenance for hgid, group_id, provenance in mapped
    }
    completed: list[dict] = []
    for record in collision_audit:
        group_id = str(record["current_group_id"])
        labels: list[dict] = []
        for human_group_id in record["historical_human_group_ids"]:
            hrow = human_by[str(human_group_id)]
            selected_edges = [
                {"ref_id": ref_id, "target_id": target_id}
                for ref_id, target_id in sorted(_human_edge_set(hrow.get("selected_edges", "[]")))
            ]
            labels.append(
                {
                    "human_group_id": str(human_group_id),
                    "provenance": str(
                        provenance_by_mapping.get((str(human_group_id), group_id), "")
                    ),
                    "labeler": _audit_text(hrow.get("labeler", "")),
                    "labeled_at": _audit_text(hrow.get("labeled_at", "")),
                    "label_semantics": _audit_text(hrow.get("label_semantics", "pair")) or "pair",
                    "selected_edges": selected_edges,
                }
            )
        completed_record = dict(record)
        completed_record["historical_labels"] = labels
        completed.append(completed_record)
    return completed


def _build_legacy_known_omission_audit(
    events: list[tuple[str, str, str, str, str]],
    dataset_id: str,
    quarantined_group_keys: set[tuple[str, str]],
    groups_by_id: dict[str, dict],
) -> list[dict]:
    """Collapse omission occurrences to deterministic current-group/edge records."""
    events_by_key: dict[tuple[str, str, str], list[tuple[str, str, str, str, str]]] = {}
    for event in events:
        events_by_key.setdefault((event[1], event[3], event[4]), []).append(event)

    audit: list[dict] = []
    for (group_id, ref_id, target_id), key_events in sorted(events_by_key.items()):
        occurrences = sorted(
            (
                {
                    "human_group_id": str(event[2]),
                    "provenance": str(event[0]),
                }
                for event in key_events
            ),
            key=lambda occurrence: (
                occurrence["human_group_id"],
                occurrence["provenance"],
            ),
        )
        historical_ids = sorted({occurrence["human_group_id"] for occurrence in occurrences})
        group = groups_by_id.get(group_id, {})
        audit.append(
            {
                "case_id": (
                    f"legacy_known_omission:{dataset_id}:{ref_id}:{target_id}:"
                    + "+".join(historical_ids)
                ),
                "dataset_id": str(dataset_id),
                "current_group_id": str(group_id),
                "ref_id": str(ref_id),
                "target_id": str(target_id),
                "ref_name": _audit_text(group.get("ref_names", {}).get(ref_id, "")),
                "target_name": _audit_text(group.get("target_names", {}).get(target_id, "")),
                "retained_after_quarantine": (str(dataset_id), str(group_id))
                not in quarantined_group_keys,
                "occurrence_count": int(len(key_events)),
                "occurrences": occurrences,
            }
        )
    return audit


def _edge_row(
    edge: dict,
    group: dict,
    n_edges: int,
    dataset_id: str,
    hgid: str,
    hrow,
    provenance: str,
    human_es: frozenset[tuple[str, str]],
) -> dict:
    """Build one per-edge row from a (possibly enriched) edge dict.

    Shared by both the candidate-graph and the legacy paths. ``edge`` must carry
    ``ref_id`` / ``target_id`` / ``selected`` / ``confidence``; every structural
    field is read with a NaN/default fallback, so a bare stage-1
    ``candidate_edges`` record (topology + confidence only) yields a valid row
    with default structure, while a legacy ``edges`` / ``rejected_edges`` record
    (or an enriched candidate edge) fills the structural layer.
    """
    key = _edge_key(edge)
    return {
        "dataset_id": dataset_id,
        "group_id": group["group_id"],
        "human_group_id": hgid,
        "labeler": hrow.get("labeler", ""),
        "provenance": provenance,
        "match_type": group.get("match_type", ""),
        "ref_id": key[0],
        "target_id": key[1],
        EDGE_LABEL_COL: int(key in human_es),
        # Legacy `edges` records omit `selected` (they are all selected), so the
        # default is True; candidate_edges and rejected_edges always set it.
        "selected": bool(edge.get("selected", True)),
        # True iff the confidence-drop prune (#284) removed this edge from the
        # selection. Provenance only — NOT a model feature (deterministic in
        # `confidence`). Absent from bare candidate_edges records -> False.
        "pruned": bool(edge.get("pruned", False)),
        # raw per-edge sidecar fields (NaN/default when only stage-1 topology is
        # available; the design's stage-2 parquet fills these via FeatureStore).
        "confidence": float(edge.get("confidence", float("nan"))),
        "degree_ref": int(edge.get("degree_ref", 0)),
        "degree_tgt": int(edge.get("degree_tgt", 0)),
        "is_bridge": bool(edge.get("candidate_graph_bridge", edge.get("is_bridge", False))),
        "is_sliver": bool(edge.get("is_sliver", False)),
        "biconnected_block": int(edge.get("biconnected_block", -1)),
        "corridor_ref": int(edge.get("corridor_ref", -1)),
        "corridor_tgt": int(edge.get("corridor_tgt", -1)),
        "gers_start_frac": float(edge.get("gers_start_frac", float("nan"))),
        "gers_end_frac": float(edge.get("gers_end_frac", float("nan"))),
        "local_start_frac": float(edge.get("local_start_frac", float("nan"))),
        "local_end_frac": float(edge.get("local_end_frac", float("nan"))),
        # per-group structural fields
        "n_edges": int(group.get("n_edges", n_edges)),
        "n_corridors": int(group.get("n_corridors", 1)),
        "n_assignment_components": int(group.get("n_assignment_components", 1)),
        "largest_biconnected_block": int(group.get("largest_biconnected_block", 1)),
        "oversized_group": bool(group.get("oversized_group", False)),
        "num_refs": len(group.get("ref_ids", [])),
        "num_targets": len(group.get("target_ids", [])),
    }


def _rows_from_candidate_graph(
    group: dict,
    dataset_id: str,
    hgid: str,
    hrow,
    provenance: str,
    human_es: frozenset[tuple[str, str]],
    filter_rule5: bool,
    stats: dict[str, int],
    legacy_known_omissions: list[tuple[str, str, str, str, str]],
) -> list[dict]:
    """Emit rows over the group's #344 ``candidate_edges`` universe.

    Label mapping (design §2.2, and the #344 reconciliation note):

    * ``selected=True``  -> optimizer keep for THIS group (positive candidate).
    * ``selected=False`` & not ``selected_elsewhere`` -> the previously
      unlearnable under-selection NEGATIVE (optimizer drop).
    * ``selected_elsewhere=True`` -> the pair IS an optimizer selection, just in
      another group / as a 1:1. It is NOT a drop for this group, so it is
      excluded (it appears as ``selected=True`` in its owning group's universe).

    The ``keep`` label itself is still ground truth (edge in the human's
    ``selected_edges``), independent of the optimizer's ``selected`` flag — that
    is exactly how an under-selection positive (``keep=1, selected=False``)
    becomes observable.

    Rule-5 filter (the #344 review's required guard): ``candidate_edges``
    attribution rule 5 (``_compute_candidate_graph_by_group``) can attach a
    component edge to a group containing NEITHER endpoint. Such an edge is not a
    within-group decision for this group, so — following the design's
    endpoint-membership recommendation — it is dropped when neither ``ref_id``
    is in the group's ``ref_ids`` nor ``target_id`` in its ``target_ids``. An
    explicitly owned ``pruned: true`` edge is exempt: its endpoints may both
    have left the post-prune group, but its pre-prune ownership is authoritative.

    Recall accounting: a human-selected edge known to the group's legacy view
    (``edges``/``rejected_edges``) but NOT emitted here — below the candidate
    floor, glue/sliver-pruned out of the reconstructed component, attributed to
    another group, or rule-5 filtered — is recorded as a *legacy-known omission*.
    This deliberately does not claim to be a universal candidate-recall count:
    a selected edge absent from both the candidate graph and the mapped group's
    legacy view cannot be classified here, and split labels commonly contain
    valid edges belonging to a different current group. Occurrence, unique raw,
    and unique retained-group counts are finalized after label-collision
    quarantine.
    """
    grp_refs = {str(x) for x in group.get("ref_ids", [])}
    grp_tgts = {str(x) for x in group.get("target_ids", [])}

    # Structural-layer lookup: candidate_edges is stage-1 (topology only), so
    # enrich from the group's `edges` + `rejected_edges` which carry the full
    # structural layer. Candidate-edge fields (confidence / selected) win.
    struct_lookup: dict[tuple[str, str], dict] = {}
    for e in group.get("edges", []):
        struct_lookup.setdefault(_edge_key(e), e)
    for e in group.get("rejected_edges", []):
        struct_lookup.setdefault(_edge_key(e), e)

    candidate_edges = group.get("candidate_edges", [])
    n_edges = len(candidate_edges)
    rows: list[dict] = []
    emitted: set[tuple[str, str]] = set()
    for cand in candidate_edges:
        key = _edge_key(cand)
        stats["candidate_seen"] += 1
        owned_pruned = cand.get("pruned") is True
        if filter_rule5 and not owned_pruned and key[0] not in grp_refs and key[1] not in grp_tgts:
            stats["rule5_filtered"] += 1
            continue
        if not cand.get("selected", False) and cand.get("selected_elsewhere", False):
            stats["selected_elsewhere_excluded"] += 1
            continue
        # Enrich stage-1 topology with the structural layer where available; the
        # candidate record's confidence/selected/selected_elsewhere take priority.
        merged = {**struct_lookup.get(key, {}), **cand}
        if key not in struct_lookup:
            stats["new_negatives"] += int(not cand.get("selected", False))
        emitted.add(key)
        rows.append(_edge_row(merged, group, n_edges, dataset_id, hgid, hrow, provenance, human_es))

    # Legacy-known omissions vs the emitted universe (see docstring above).
    for ref_id, target_id in human_es:
        if (ref_id, target_id) in struct_lookup and (ref_id, target_id) not in emitted:
            legacy_known_omissions.append(
                (provenance, str(group["group_id"]), hgid, ref_id, target_id)
            )
    return rows


def _rows_from_legacy_edges(
    group: dict,
    dataset_id: str,
    hgid: str,
    hrow,
    provenance: str,
    human_es: frozenset[tuple[str, str]],
    include_rejected: bool,
) -> list[dict]:
    """Emit rows the pre-#344 way: ``edges`` + capped ``rejected_edges``.

    Kept for sidecars without a usable ``candidate_edges`` universe. Under
    selection is only partially observable here (see the module docstring).
    """
    edges = list(group.get("edges", []))
    if include_rejected:
        seen = {_edge_key(e) for e in edges}
        for rej in group.get("rejected_edges", []):
            if _edge_key(rej) not in seen:
                edges.append(rej)
    n_edges = len(edges)
    return [
        _edge_row(edge, group, n_edges, dataset_id, hgid, hrow, provenance, human_es)
        for edge in edges
    ]


def build_edge_table(
    groups: list[dict],
    human_df: pd.DataFrame,
    dataset_id: str,
    include_split: bool = True,
    include_rejected: bool = True,
    prefer_candidate_graph: bool = True,
    filter_rule5: bool = True,
    include_empty: bool = True,
    candidates_df: pd.DataFrame | None = None,
    candidates_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build a per-edge training table for one dataset.

    Each row is one candidate edge inside a *labeled* group, with the raw
    sidecar fields, group-context columns, the ``keep`` label (1 iff the edge is
    in the human's selected set), the ``selected`` optimizer baseline, and
    provenance columns. The row key is :data:`KEY_COLUMNS`.

    Candidate universe, per group:

    * **``candidate_edges`` present (post-#344)** and ``prefer_candidate_graph``:
      the uncapped candidate graph is the universe (design §2.2). Non-selected
      candidates are the under-selection negatives; ``selected_elsewhere`` edges
      are excluded (they are selected in another group); rule-5 attribution noise
      is filtered by endpoint membership. Structural columns are enriched from
      ``edges`` / ``rejected_edges`` where present, else default.
    * **otherwise (legacy)**: ``edges`` + (capped) ``rejected_edges``, exactly as
      before. A single warning is logged noting the uncapped under-selection
      universe is unavailable for those groups.

    Empty-set (reject-all) labels — pair rows with ``selected_edges == []`` —
    map by verbatim ``group_id`` (no edges to overlap on) and emit the group's
    full candidate universe with ``keep=0`` and ``provenance="empty"``: the
    design §2.4a "select nothing" shape. Legacy groups cannot express the full
    universe, so an empty label on a legacy group emits zero rows and is
    counted + warned (``empty_legacy_skipped``).

    Stage-2 typed candidate parquet (P1, PR #414): when provided via
    ``candidates_df`` or ``candidates_path`` (or discovered adjacent to the
    groups.json), the 83 FEATURE_COLUMNS + signed lateral offset + class/length +
    optimizer decision/reason are left-joined on (group_id, ref_id, target_id)
    (design §3.2). This fills the ~95% gap where pair features were missing and
    replaces placeholder structural values with authoritative runtime values.

    Args:
        groups: sidecar groups (from :func:`load_sidecar_groups`).
        human_df: curated stitching labels for this dataset.
        dataset_id: dataset identifier stored on every row.
        include_split: if False, only ``clean`` labels (all selected edges in
            one group) are emitted; ``split`` labels (human edge set spans
            multiple groups, so the within-group keep set is partial and the
            drop label is noisy) are dropped.
        include_rejected: legacy-path only — fold the capped ``rejected_edges``
            list into the table as ``selected=False`` rows. Ignored when the
            candidate graph is used (it already contains every candidate).
        prefer_candidate_graph: if True (default), consume ``candidate_edges``
            when a group carries a non-empty one; set False to force the legacy
            path (e.g. for A/B comparison against the pre-#344 behavior).
        filter_rule5: candidate-graph path only — drop rule-5 attribution noise
            (candidate edges with neither endpoint in the group), except an
            explicitly owner-attributed ``pruned`` edge. Default True.
        include_empty: if True (default), emit all-``keep=0`` rows for
            reject-all labels whose group_id still exists (candidate-graph
            groups only; see above).
        candidates_df: optional pre-loaded typed candidates parquet DataFrame.
            When provided, its columns are joined onto the edge table.
        candidates_path: optional path to a typed candidates parquet file.
            If both df and path are None, no parquet join occurs.

    Returns:
        DataFrame with one row per (group, edge); empty if no labels map. The
        per-build counters, including duplicate-label collision quarantine, are
        logged once and attached as ``df.attrs["build_stats"]``. Exact,
        deterministically sorted collision and legacy-known omission identities
        are attached as ``df.attrs["build_audit"]``.
    """
    # Resolve candidates parquet if a path was given but no df
    if candidates_df is None and candidates_path is not None:
        candidates_df = load_candidates_parquet(candidates_path)

    rec = recover_labeled_groups(groups, human_df)
    gmap = {g["group_id"]: g for g in groups}
    human_by = {str(r["group_id"]): r for _, r in human_df.iterrows()}

    mapped: list[tuple[str, str, str]] = [(hgid, bg, "clean") for hgid, bg in rec["clean"]]
    if include_split:
        mapped += [(hgid, bg, "split") for hgid, bg, _, _ in rec["split"]]
    # Empty-set (reject-all) labels carry no edges, so edge-overlap recovery is
    # impossible; they survive only on a verbatim group_id match — the same rule
    # as stitch_eval.recover_empty_reject_all.
    n_empty_unrecovered = 0
    if include_empty:
        for hgid in rec["empty"]:
            if hgid in gmap:
                mapped.append((hgid, hgid, "empty"))
            else:
                n_empty_unrecovered += 1

    stats: dict[str, int] = {
        "rows": 0,
        "positives": 0,
        "negatives": 0,
        "candidate_seen": 0,
        "rule5_filtered": 0,
        "selected_elsewhere_excluded": 0,
        "new_negatives": 0,
        "human_selected_outside_candidate_graph": 0,
        "human_selected_outside_candidate_graph_clean": 0,
        "human_selected_outside_candidate_graph_split": 0,
        "legacy_known_omission_occurrences": 0,
        "legacy_known_omission_occurrences_clean": 0,
        "legacy_known_omission_occurrences_split": 0,
        "legacy_known_omission_unique_raw_keys": 0,
        "legacy_known_omission_unique_raw_keys_clean": 0,
        "legacy_known_omission_unique_raw_keys_split": 0,
        "legacy_known_omission_unique_retained_keys": 0,
        "legacy_known_omission_unique_retained_keys_clean": 0,
        "legacy_known_omission_unique_retained_keys_split": 0,
        "empty_rows": 0,
        "empty_legacy_skipped": 0,
        "empty_unrecovered": n_empty_unrecovered,
        "legacy_groups": 0,
        "candidate_groups": 0,
    }
    rows: list[dict] = []
    legacy_known_omissions: list[tuple[str, str, str, str, str]] = []
    for hgid, bg, provenance in mapped:
        group = gmap.get(bg)
        if group is None:
            continue
        hrow = human_by[hgid]
        human_es = _human_edge_set(hrow["selected_edges"])
        if prefer_candidate_graph and _group_has_candidate_graph(group):
            stats["candidate_groups"] += 1
            new_rows = _rows_from_candidate_graph(
                group,
                dataset_id,
                hgid,
                hrow,
                provenance,
                human_es,
                filter_rule5,
                stats,
                legacy_known_omissions,
            )
            if provenance == "empty":
                stats["empty_rows"] += len(new_rows)
            rows.extend(new_rows)
        elif provenance == "empty":
            # A reject-all label asserts "keep nothing from the FULL candidate
            # set"; the capped legacy view cannot represent that set, so partial
            # keep=0 rows would silently understate the label. Emit nothing.
            stats["empty_legacy_skipped"] += 1
        else:
            stats["legacy_groups"] += 1
            rows.extend(
                _rows_from_legacy_edges(
                    group, dataset_id, hgid, hrow, provenance, human_es, include_rejected
                )
            )

    df = pd.DataFrame(rows)
    stats["raw_rows"] = len(df)
    df, quarantined_group_keys, collision_audit = _resolve_duplicate_label_collisions(df, stats)
    collision_audit = _complete_collision_audit(collision_audit, human_by, mapped)

    # This is an audit of legacy-known omission events, not a universal recall
    # claim. An event is one historical-label occurrence. A unique key is scoped
    # to current group + candidate pair, so the same omission recovered from two
    # historical labels or both provenance slices is counted once in the total.
    # Slice counters filter provenance first. Retained counts exclude current
    # groups quarantined for contradictory truth.
    for provenance in (None, "clean", "split"):
        suffix = f"_{provenance}" if provenance else ""
        events = [
            event
            for event in legacy_known_omissions
            if provenance is None or event[0] == provenance
        ]
        raw_keys = {(event[1], event[3], event[4]) for event in events}
        retained_keys = {
            (event[1], event[3], event[4])
            for event in events
            if (dataset_id, event[1]) not in quarantined_group_keys
        }
        stats[f"legacy_known_omission_occurrences{suffix}"] = len(events)
        stats[f"legacy_known_omission_unique_raw_keys{suffix}"] = len(raw_keys)
        stats[f"legacy_known_omission_unique_retained_keys{suffix}"] = len(retained_keys)

    build_audit = {
        "schema_version": 1,
        # Table-level version + a content hash of the emitted column set. The
        # hash travels with the audit and shifts whenever columns are added or
        # dropped (e.g. a new candidate-parquet column silently joining), giving
        # a detectable fingerprint of the training table's shape between explicit
        # RESOLVER_TABLE_SCHEMA_VERSION bumps. Populated after any parquet join
        # below (see _finalize_table_schema_audit).
        "table_schema_version": RESOLVER_TABLE_SCHEMA_VERSION,
        "quarantined_groups": collision_audit,
        "legacy_known_omissions": _build_legacy_known_omission_audit(
            legacy_known_omissions,
            dataset_id,
            quarantined_group_keys,
            gmap,
        ),
    }

    # Backward-compatible aliases for research scripts written before the
    # occurrence-vs-key distinction. New reports must use the accurately named
    # counters above.
    stats["human_selected_outside_candidate_graph"] = stats["legacy_known_omission_occurrences"]
    stats["human_selected_outside_candidate_graph_clean"] = stats[
        "legacy_known_omission_occurrences_clean"
    ]
    stats["human_selected_outside_candidate_graph_split"] = stats[
        "legacy_known_omission_occurrences_split"
    ]
    stats["rows"] = len(df)
    if len(df):
        stats["positives"] = int(df[EDGE_LABEL_COL].sum())
        stats["negatives"] = int((df[EDGE_LABEL_COL] == 0).sum())

    # Stage-2 parquet join (P1): enrich with typed candidate features if provided.
    # This is the R1 step that closes the ~95% pair-feature gap and provides
    # signed lateral offset, class/length, optimizer decision/reason, and authoritative
    # structural values. Keeps row count identical — only adds columns / fills defaults.
    df.attrs["build_stats"] = stats
    df.attrs["build_audit"] = build_audit
    if candidates_df is not None:
        df = _enrich_with_candidate_parquet(df, candidates_df, stats)
    # Update row counts after enrichment (count unchanged, but recompute for safety if needed)
    stats["rows"] = len(df)
    if len(df):
        stats["positives"] = int(df[EDGE_LABEL_COL].sum())
        stats["negatives"] = int((df[EDGE_LABEL_COL] == 0).sum())

    if stats["legacy_groups"]:
        logger.warning(
            "build_edge_table[{}]: {}/{} labeled groups have no candidate_edges "
            "(pre-#344 or stitch_persist_candidate_graph=False); falling back to "
            "edges+rejected_edges — the uncapped under-selection universe is "
            "unavailable for those groups.",
            dataset_id,
            stats["legacy_groups"],
            len(mapped),
        )
    if stats["empty_legacy_skipped"]:
        logger.warning(
            "build_edge_table[{}]: {} reject-all (empty-set) labels map to groups "
            "without candidate_edges; the legacy view cannot express the full "
            "candidate universe, so they emit no rows.",
            dataset_id,
            stats["empty_legacy_skipped"],
        )
    # Record the final column-set fingerprint (after any parquet join) so the
    # audit captures the table's actual shape, including silently-accreted
    # columns. Deterministic over the column NAMES (order-independent).
    columns_sorted = sorted(map(str, df.columns))
    build_audit["table_columns"] = columns_sorted
    build_audit["table_columns_sha256"] = hashlib.sha256(
        "\x1f".join(columns_sorted).encode("utf-8")
    ).hexdigest()

    logger.info("build_edge_table[{}]: {}", dataset_id, stats)
    df.attrs["build_stats"] = stats
    df.attrs["build_audit"] = build_audit
    df.attrs["table_schema_version"] = RESOLVER_TABLE_SCHEMA_VERSION
    return df


def build_multi_dataset_table(
    specs: list[
        tuple[str, str | Path, str | Path] | tuple[str, str | Path, str | Path, str | Path]
    ],
    include_split: bool = True,
    auto_discover_candidates: bool = True,
) -> pd.DataFrame:
    """Build and concatenate per-edge tables for several datasets.

    Supports both legacy 3-tuples ``(dataset_id, groups_json_path, labels_csv_path)``
    and extended 4-tuples with an explicit candidates parquet path. When
    ``auto_discover_candidates`` is True (default), a candidates parquet adjacent
    to the groups.json is auto-discovered via :func:`discover_candidates_parquet`
    (factory ``candidates.parquet`` or output ``*_candidates.parquet``) and joined.

    Args:
        specs: list of 3- or 4-tuples.
        include_split: passed through to :func:`build_edge_table`.
        auto_discover_candidates: if True, try to find and join a typed candidates
            parquet beside each groups.json when no explicit path is given.
    """
    frames = []
    for spec in specs:
        if len(spec) == 4:
            dataset_id, groups_path, labels_path, candidates_path = spec
        else:
            dataset_id, groups_path, labels_path = spec
            candidates_path = None
            if auto_discover_candidates:
                try:
                    discovered = discover_candidates_parquet(groups_path)
                    if discovered is not None:
                        candidates_path = discovered
                        logger.info(
                            f"build_multi_dataset_table[{dataset_id}]: auto-discovered "
                            f"candidates parquet {discovered}"
                        )
                except Exception as exc:
                    logger.debug(
                        f"build_multi_dataset_table[{dataset_id}]: candidate discovery failed: {exc}"
                    )

        groups = load_sidecar_groups(groups_path)
        human_df = load_stitching_labels(labels_path)
        candidates_df = (
            load_candidates_parquet(candidates_path) if candidates_path is not None else None
        )

        frames.append(
            build_edge_table(
                groups,
                human_df,
                dataset_id,
                include_split=include_split,
                candidates_df=candidates_df,
            )
        )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
