"""Unit tests for the resolver eval provenance slices.

Covers the two rollups added alongside the anchoring-provenance plumbing:

* ``_labeler_era`` — collapse ``panel_*_vN`` labelers to a coarse rubric era.
* ``slice_report`` — emit per-era and anchored/de-anchored slices in addition to
  the flat per-labeler ones.

Synthetic tables only (no dependency on data/), so these run in CI.
"""

from __future__ import annotations

import json

import pandas as pd

from crosswalk.resolver.evaluate import _labeler_era, slice_report


def test_labeler_era_maps_panel_versions_and_humans():
    assert _labeler_era("panel_unanimous_v1") == "v1"
    assert _labeler_era("panel_unanimous_v7") == "v7"
    assert _labeler_era("panel_quorum_v3") == "v3"
    # Human labelers and anything else roll up to "human".
    assert _labeler_era("brad") == "human"
    assert _labeler_era("panel_unanimous") == "human"  # no version suffix
    assert _labeler_era("agent_batch_v2") == "human"  # not a panel labeler
    assert _labeler_era("") == "human"


def _edge(ref, tgt, conf):
    return {
        "ref_id": ref,
        "target_id": tgt,
        "confidence": conf,
        "selected": True,
        "degree_ref": 1,
        "degree_tgt": 1,
        "is_bridge": True,
        "is_sliver": False,
        "biconnected_block": 0,
        "corridor_ref": 0,
        "corridor_tgt": 0,
        "gers_start_frac": 0.0,
        "gers_end_frac": 1.0,
        "local_start_frac": 0.0,
        "local_end_frac": 1.0,
    }


def _group(gid, edges):
    return {
        "group_id": gid,
        "match_type": "N:1",
        "edges": edges,
        "ref_ids": sorted({e["ref_id"] for e in edges}),
        "target_ids": sorted({e["target_id"] for e in edges}),
        "n_edges": len(edges),
        "n_corridors": 1,
        "n_assignment_components": 1,
        "largest_biconnected_block": 1,
        "oversized_group": False,
    }


def _label_row(gid, edges, labeler, session_id):
    return {
        "group_id": gid,
        "dataset_id": "ds",
        "selected_edges": json.dumps([{"ref_id": r, "target_id": t} for r, t in edges]),
        "match_type": "N:1",
        "num_refs": 1,
        "num_targets": 1,
        "labeler": labeler,
        "labeled_at": "2026-01-01",
        "session_id": session_id,
    }


def _synthetic_edge_table():
    """A table with enough groups per era/anchor slice for grouped CV to run.

    Each group has one kept and one dropped edge (both classes present). Two
    cohorts: v1/anchored panel votes and human/de-anchored labels.
    """
    from crosswalk.resolver.extract import build_edge_table

    groups = []
    labels = []
    cohorts = [
        ("panel_unanimous_v1", "ds_batch_optionmenu", "P"),  # anchored era v1
        ("brad", "deanchored_v1", "H"),  # de-anchored human
    ]
    for labeler, session_id, prefix in cohorts:
        for i in range(6):
            gid = f"{prefix}{i}"
            keep = (f"{prefix}k{i}", f"{prefix}kt{i}")
            drop = (f"{prefix}d{i}", f"{prefix}dt{i}")
            groups.append(_group(gid, [_edge(*keep, 0.95), _edge(*drop, 0.30)]))
            labels.append(_label_row(gid, [keep], labeler, session_id))
    return build_edge_table(groups, pd.DataFrame(labels), "ds")


def test_slice_report_emits_era_and_anchored_slices():
    df = _synthetic_edge_table()
    # Sanity: provenance columns survived extraction.
    assert set(df["anchored"].unique()) == {True, False}

    report = slice_report(df)
    slices = set(report["slice"])
    # Per-era rollup (panel_unanimous_v1 -> v1; brad -> human).
    assert "era=v1" in slices
    assert "era=human" in slices
    # Anchored vs de-anchored rollup.
    assert "anchored" in slices
    assert "deanchored" in slices
    # Flat per-labeler slices are still present.
    assert "labeler=brad" in slices
