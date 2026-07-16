"""Tests for stage-2 typed candidate parquet join (R1, PR #414 follow-up).

Validates discovery, loading, and enrichment of the edge table with the 83
FEATURE_COLUMNS + signed lateral offset + structural context.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from crosswalk.resolver.extract import (
    build_edge_table,
    build_multi_dataset_table,
    discover_candidates_parquet,
    load_candidates_parquet,
)


def _edge(ref, tgt, conf, selected=True, **kw):
    d = {
        "ref_id": ref,
        "target_id": tgt,
        "confidence": conf,
        "selected": selected,
        "degree_ref": 1,
        "degree_tgt": 1,
        "is_bridge": False,
        "is_sliver": False,
        "biconnected_block": 0,
        "corridor_ref": 0,
        "corridor_tgt": 0,
        "gers_start_frac": 0.0,
        "gers_end_frac": 1.0,
        "local_start_frac": 0.0,
        "local_end_frac": 1.0,
    }
    d.update(kw)
    return d


def _cand(ref, tgt, conf, selected, selected_elsewhere=False):
    d = {"ref_id": ref, "target_id": tgt, "confidence": conf, "selected": selected}
    if selected_elsewhere:
        d["selected_elsewhere"] = True
    return d


def _group_cg(gid, candidate_edges, ref_ids, target_ids, edges=None):
    if edges is None:
        edges = [
            _edge(c["ref_id"], c["target_id"], c["confidence"])
            for c in candidate_edges
            if c.get("selected")
        ]
    return {
        "group_id": gid,
        "match_type": "M:N",
        "edges": edges,
        "candidate_edges": candidate_edges,
        "n_candidate_edges": len(candidate_edges),
        "ref_ids": ref_ids,
        "target_ids": target_ids,
        "n_edges": len(candidate_edges),
        "n_corridors": 1,
        "n_assignment_components": 1,
        "largest_biconnected_block": 1,
        "oversized_group": False,
    }


def _label_df():
    return pd.DataFrame(
        [
            {
                "group_id": "hg1",
                "dataset_id": "ds",
                "selected_edges": json.dumps([{"ref_id": "A", "target_id": "T"}]),
                "match_type": "M:N",
                "num_refs": 1,
                "num_targets": 1,
                "labeler": "brad",
                "labeled_at": "2026-01-01",
                "session_id": "x",
            }
        ]
    )


def test_discover_candidates_parquet_output_layout(tmp_path: Path):
    # Output layout: <ds>_groups.json -> <ds>_candidates.parquet
    groups_path = tmp_path / "us_boston_streets_groups.json"
    groups_path.write_text(json.dumps({"groups": []}))
    cand_path = tmp_path / "us_boston_streets_candidates.parquet"
    pd.DataFrame(
        [{"group_id": "g1", "ref_id": "A", "target_id": "T", "confidence": 0.9}]
    ).to_parquet(cand_path)

    discovered = discover_candidates_parquet(groups_path)
    assert discovered == cand_path


def test_discover_candidates_parquet_factory_layout(tmp_path: Path):
    # Factory layout: dataset=.../groups.json -> candidates.parquet
    dataset_dir = tmp_path / "dataset=us_boston_streets"
    dataset_dir.mkdir()
    groups_path = dataset_dir / "groups.json"
    groups_path.write_text(json.dumps({"groups": []}))
    cand_path = dataset_dir / "candidates.parquet"
    pd.DataFrame(
        [{"group_id": "g1", "ref_id": "A", "target_id": "T", "confidence": 0.9}]
    ).to_parquet(cand_path)

    discovered = discover_candidates_parquet(groups_path)
    assert discovered == cand_path


def test_load_candidates_parquet_validates_keys(tmp_path: Path):
    cand_path = tmp_path / "cands.parquet"
    df = pd.DataFrame(
        [
            {"group_id": "g1", "ref_id": "A", "target_id": "T", "hausdorff_distance_m": 1.0},
            {"group_id": "g1", "ref_id": "B", "target_id": "T", "hausdorff_distance_m": 2.0},
        ]
    )
    df.to_parquet(cand_path)

    loaded = load_candidates_parquet(cand_path)
    assert len(loaded) == 2
    assert "hausdorff_distance_m" in loaded.columns


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "is empty"),
        (
            [
                {"group_id": "g1", "ref_id": "A", "target_id": "T"},
                {"group_id": "g1", "ref_id": "A", "target_id": "T"},
            ],
            "duplicate key",
        ),
        (
            [{"group_id": "g1", "ref_id": None, "target_id": "T"}],
            "null join keys",
        ),
    ],
)
def test_load_candidates_parquet_rejects_unsafe_keys(
    tmp_path: Path, rows: list[dict], message: str
):
    cand_path = tmp_path / "cands.parquet"
    columns = ["group_id", "ref_id", "target_id"] if not rows else None
    pd.DataFrame(rows, columns=columns).to_parquet(cand_path)

    with pytest.raises(ValueError, match=message):
        load_candidates_parquet(cand_path)


def test_build_edge_table_rejects_duplicate_direct_candidate_frame():
    grp = _group_cg(
        "g1",
        [_cand("A", "T", 0.99, True)],
        ref_ids=["A"],
        target_ids=["T"],
    )
    duplicate_candidates = pd.DataFrame(
        [
            {"group_id": "g1", "ref_id": "A", "target_id": "T", "feature": 1.0},
            {"group_id": "g1", "ref_id": "A", "target_id": "T", "feature": 2.0},
        ]
    )

    with pytest.raises(ValueError, match="duplicate key"):
        build_edge_table([grp], _label_df(), "ds", candidates_df=duplicate_candidates)


def test_build_edge_table_enriches_with_parquet(tmp_path: Path):
    grp = _group_cg(
        "g1",
        [_cand("A", "T", 0.99, True), _cand("B", "T", 0.4, False)],
        ref_ids=["A", "B"],
        target_ids=["T"],
    )
    human = _label_df()

    # Base table without parquet has no typed pair features
    base = build_edge_table([grp], human, "ds")
    assert "hausdorff_distance_m" not in base.columns
    assert "lateral_offset_signed_m" not in base.columns

    # Parquet with 83-feature-like columns + extras
    cand_parquet = pd.DataFrame(
        [
            {
                "group_id": "g1",
                "ref_id": "A",
                "target_id": "T",
                "confidence": 0.99,
                "degree_ref": 5,
                "degree_tgt": 7,
                "is_bridge": True,
                "lateral_offset_signed_m": 2.5,
                "ref_class": "residential",
                "target_class": "cycleway",
                "ref_length_m": 10.0,
                "target_length_m": 12.0,
                "hausdorff_distance_m": 1.2,
                "name_levenshtein": 0.8,
                "optimizer_decision": "selected",
            },
            {
                "group_id": "g1",
                "ref_id": "B",
                "target_id": "T",
                "confidence": 0.4,
                "degree_ref": 3,
                "degree_tgt": 2,
                "is_bridge": False,
                "lateral_offset_signed_m": -1.2,
                "ref_class": "primary",
                "target_class": "primary",
                "ref_length_m": 15.0,
                "target_length_m": 12.0,
                "hausdorff_distance_m": 10.5,
                "name_levenshtein": 0.2,
                "optimizer_decision": "rejected",
            },
        ]
    )

    enriched = build_edge_table([grp], human, "ds", candidates_df=cand_parquet)
    # New columns from parquet should appear
    for col in [
        "hausdorff_distance_m",
        "name_levenshtein",
        "lateral_offset_signed_m",
        "ref_class",
        "target_class",
        "ref_length_m",
        "optimizer_decision",
    ]:
        assert col in enriched.columns, f"missing enriched col {col}"

    # Structural enrichment: parquet authoritative
    assert list(enriched.sort_values("ref_id")["degree_ref"]) == [5, 3]
    assert list(enriched.sort_values("ref_id")["lateral_offset_signed_m"]) == [2.5, -1.2]

    # Row count unchanged, keep label preserved
    assert len(enriched) == len(base) == 2
    assert enriched.attrs["build_stats"]["candidate_parquet_rows"] == 2
    assert enriched.attrs["build_stats"]["candidate_parquet_enriched"] == 2

    # Keep column still ground truth, not overwritten
    keep_by_ref = dict(zip(enriched["ref_id"], enriched["keep"]))
    assert keep_by_ref["A"] == 1 and keep_by_ref["B"] == 0


def test_build_multi_dataset_table_auto_discovers(tmp_path: Path):
    # Dataset 1 with factory layout
    ds1_dir = tmp_path / "dataset=ds1"
    ds1_dir.mkdir()
    grp1 = _group_cg("g1", [_cand("A", "T", 0.9, True)], ["A"], ["T"])
    groups1_path = ds1_dir / "groups.json"
    groups1_path.write_text(json.dumps({"n_groups": 1, "groups": [grp1]}))
    cand1_path = ds1_dir / "candidates.parquet"
    pd.DataFrame(
        [{"group_id": "g1", "ref_id": "A", "target_id": "T", "hausdorff_distance_m": 1.0}]
    ).to_parquet(cand1_path)
    labels1_path = tmp_path / "labels1.csv"
    pd.DataFrame(
        [
            {
                "group_id": "g1",
                "dataset_id": "ds1",
                "selected_edges": json.dumps([{"ref_id": "A", "target_id": "T"}]),
                "match_type": "M:N",
                "num_refs": 1,
                "num_targets": 1,
                "labeler": "brad",
                "labeled_at": "2026-01-01",
                "session_id": "x",
            }
        ]
    ).to_csv(labels1_path, index=False)

    # Build via multi-dataset helper with auto-discovery
    df = build_multi_dataset_table(
        [("ds1", str(groups1_path), str(labels1_path))],
        auto_discover_candidates=True,
    )
    assert "hausdorff_distance_m" in df.columns
    assert df.attrs if hasattr(df, "attrs") else True


def test_build_edge_table_stamps_table_schema_version_and_columns_hash():
    from crosswalk.resolver.extract import RESOLVER_TABLE_SCHEMA_VERSION

    grp = _group_cg(
        "g1",
        [_cand("A", "T", 0.99, True), _cand("B", "T", 0.4, False)],
        ref_ids=["A", "B"],
        target_ids=["T"],
    )
    df = build_edge_table([grp], _label_df(), "ds")

    # Version travels with the artifact via attrs and the audit dict.
    assert df.attrs["table_schema_version"] == RESOLVER_TABLE_SCHEMA_VERSION
    audit = df.attrs["build_audit"]
    assert audit["table_schema_version"] == RESOLVER_TABLE_SCHEMA_VERSION
    # Column-set hash is present, deterministic, and matches the emitted columns.
    assert audit["table_columns"] == sorted(map(str, df.columns))
    import hashlib

    expected = hashlib.sha256("\x1f".join(audit["table_columns"]).encode("utf-8")).hexdigest()
    assert audit["table_columns_sha256"] == expected


def test_column_hash_shifts_when_parquet_adds_a_column():
    grp = _group_cg(
        "g1",
        [_cand("A", "T", 0.99, True), _cand("B", "T", 0.4, False)],
        ref_ids=["A", "B"],
        target_ids=["T"],
    )
    base = build_edge_table([grp], _label_df(), "ds")
    cand = pd.DataFrame(
        [
            {"group_id": "g1", "ref_id": "A", "target_id": "T", "hausdorff_distance_m": 1.0},
            {"group_id": "g1", "ref_id": "B", "target_id": "T", "hausdorff_distance_m": 9.0},
        ]
    )
    enriched = build_edge_table([grp], _label_df(), "ds", candidates_df=cand)

    assert (
        base.attrs["build_audit"]["table_columns_sha256"]
        != enriched.attrs["build_audit"]["table_columns_sha256"]
    )
    assert "hausdorff_distance_m" in enriched.attrs["build_audit"]["table_columns"]


def _capture_loguru_warnings():
    """Context-manager-like helper: returns (messages list, remove fn).

    loguru does not route to pytest's caplog, so capture via a temporary sink.
    """
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    return messages, lambda: logger.remove(sink_id)


def test_expected_candidate_columns_do_not_warn():
    grp = _group_cg(
        "g1",
        [_cand("A", "T", 0.99, True), _cand("B", "T", 0.4, False)],
        ref_ids=["A", "B"],
        target_ids=["T"],
    )
    # All joined columns are in EXPECTED_CANDIDATE_JOIN_COLUMNS.
    cand = pd.DataFrame(
        [
            {
                "group_id": "g1",
                "ref_id": "A",
                "target_id": "T",
                "hausdorff_distance_m": 1.0,
                "lateral_offset_signed_m": 2.5,
                "ref_class": "residential",
            },
            {
                "group_id": "g1",
                "ref_id": "B",
                "target_id": "T",
                "hausdorff_distance_m": 9.0,
                "lateral_offset_signed_m": -1.0,
                "ref_class": "primary",
            },
        ]
    )
    messages, remove = _capture_loguru_warnings()
    try:
        df = build_edge_table([grp], _label_df(), "ds", candidates_df=cand)
    finally:
        remove()

    assert df.attrs["build_stats"]["candidate_parquet_unexpected_columns"] == []
    assert not any("unrecognized candidate-parquet" in m for m in messages)


def test_unexpected_candidate_column_warns_and_is_recorded():
    grp = _group_cg(
        "g1",
        [_cand("A", "T", 0.99, True), _cand("B", "T", 0.4, False)],
        ref_ids=["A", "B"],
        target_ids=["T"],
    )
    cand = pd.DataFrame(
        [
            {
                "group_id": "g1",
                "ref_id": "A",
                "target_id": "T",
                "hausdorff_distance_m": 1.0,
                "some_new_mystery_column": 7,
            },
            {
                "group_id": "g1",
                "ref_id": "B",
                "target_id": "T",
                "hausdorff_distance_m": 9.0,
                "some_new_mystery_column": 8,
            },
        ]
    )
    messages, remove = _capture_loguru_warnings()
    try:
        df = build_edge_table([grp], _label_df(), "ds", candidates_df=cand)
    finally:
        remove()

    unexpected = df.attrs["build_stats"]["candidate_parquet_unexpected_columns"]
    assert unexpected == ["some_new_mystery_column"]
    # Column still joined (exclusion-list mechanism preserved, not a hard allowlist).
    assert "some_new_mystery_column" in df.columns
    assert any("unrecognized candidate-parquet" in m for m in messages)
