"""Tests for stitch-level evaluation metrics."""

import json

import pandas as pd
import pytest

from mbench.eval.stitch_metrics import StitchEvalResult, evaluate_stitch_groups


@pytest.fixture
def bridge_df():
    """Bridge with edges for two groups."""
    return pd.DataFrame(
        {
            "ref_id": ["r1", "r1", "r2", "r3", "r3", "r4"],
            "target_id": ["t1", "t2", "t1", "t3", "t4", "t5"],
            "confidence": [0.9, 0.8, 0.7, 0.95, 0.6, 0.85],
        }
    )


@pytest.fixture
def stitch_labels():
    """Two labeled groups."""
    return pd.DataFrame(
        {
            "group_id": ["aaa", "bbb"],
            "dataset_id": ["test", "test"],
            "selected_edges": [
                json.dumps(
                    [
                        {"ref_id": "r1", "target_id": "t1"},
                        {"ref_id": "r2", "target_id": "t1"},
                    ]
                ),
                json.dumps(
                    [
                        {"ref_id": "r3", "target_id": "t3"},
                    ]
                ),
            ],
            "match_type": ["N:1", "1:1"],
            "num_refs": [2, 1],
            "num_targets": [1, 1],
        }
    )


def test_evaluate_stitch_groups_basic(bridge_df, stitch_labels):
    """Curated edges found in bridge, extra edges detected."""
    result = evaluate_stitch_groups(bridge_df, stitch_labels)

    assert result.groups_evaluated == 2
    # Group aaa: curated {r1-t1, r2-t1}, bridge has {r1-t1, r1-t2, r2-t1}
    #   recall = 2/2 = 1.0, precision = 2/3 = 0.667
    # Group bbb: curated {r3-t3}, bridge has {r3-t3, r3-t4}
    #   recall = 1/1 = 1.0, precision = 1/2 = 0.5
    assert result.recall == pytest.approx(1.0)  # macro avg
    assert result.precision == pytest.approx((2 / 3 + 0.5) / 2, abs=0.01)
    assert result.total_curated_edges == 3
    assert result.total_extra_edges == 2  # r1-t2 and r3-t4


def test_evaluate_stitch_groups_missing_edges(bridge_df, stitch_labels):
    """When bridge is missing curated edges, recall drops."""
    # Remove r2-t1 from bridge
    bridge_missing = bridge_df[~((bridge_df["ref_id"] == "r2") & (bridge_df["target_id"] == "t1"))]
    result = evaluate_stitch_groups(bridge_missing, stitch_labels)

    # Group aaa: curated {r1-t1, r2-t1}, bridge has {r1-t1, r1-t2}
    #   recall = 1/2 = 0.5, precision = 1/2 = 0.5
    assert result.recall == pytest.approx(0.75)  # macro avg (0.5 + 1.0) / 2


def test_evaluate_stitch_groups_empty_labels():
    """No stitch labels -> zero metrics."""
    bridge = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "confidence": [0.9]})
    labels = pd.DataFrame(
        columns=[
            "group_id",
            "dataset_id",
            "selected_edges",
            "match_type",
            "num_refs",
            "num_targets",
        ]
    )
    result = evaluate_stitch_groups(bridge, labels)
    assert result.groups_evaluated == 0
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_stitch_eval_result_to_dict():
    r = StitchEvalResult(
        groups_evaluated=5,
        precision=0.8,
        recall=0.9,
        f1=0.847,
        exact_match_rate=0.6,
        total_curated_edges=10,
        total_extra_edges=2,
        precision_filtered=0.85,
        recall_filtered=0.92,
        f1_filtered=0.884,
        exact_match_rate_filtered=0.7,
        groups_sliver_affected=1,
    )
    d = r.to_dict()
    assert d["groups_evaluated"] == 5
    assert d["stitch_f1"] == 0.847
    assert d["stitch_exact_match_rate"] == 0.6
    assert d["stitch_f1_filtered"] == 0.884
    assert d["stitch_groups_sliver_affected"] == 1


def _line(coords):
    return {"type": "LineString", "coordinates": coords}


@pytest.fixture
def groups_sidecar():
    """A single M:N group with real geometries and one junction-sliver edge.

    r1-t1 is a full-length overlap (legit). r1-t2 is a tiny clip at the far end
    of t1's line -> a sliver.
    """
    # ~111 m per 0.001 deg lat. r1/t1 run ~110 m; the sliver edge barely overlaps.
    return [
        {
            "group_id": "g_current",
            "match_type": "N:1",
            "edges": [
                {
                    "ref_id": "r1",
                    "target_id": "t1",
                    "gers_start_frac": 0.0,
                    "gers_end_frac": 1.0,
                    "local_start_frac": 0.0,
                    "local_end_frac": 1.0,
                },
                {
                    "ref_id": "r1",
                    "target_id": "t2",
                    # tiny overlap: 2% span of a ~110 m segment -> ~2.2 m < 5 m
                    "gers_start_frac": 0.0,
                    "gers_end_frac": 0.02,
                    "local_start_frac": 0.98,
                    "local_end_frac": 1.0,
                },
            ],
            "ref_geometries": {
                "r1": _line([[-71.0, 42.0], [-71.0, 42.001]]),
            },
            "target_geometries": {
                "t1": _line([[-71.0, 42.0], [-71.0, 42.001]]),
                "t2": _line([[-71.0, 42.0009], [-71.0, 42.002]]),
            },
        }
    ]


def test_groups_based_mapping_and_sliver(groups_sidecar):
    """Label maps by edge-overlap to the current group; sliver drops from both."""
    bridge = pd.DataFrame(
        {
            "ref_id": ["r1", "r1"],
            "target_id": ["t1", "t2"],
            "confidence": [0.9, 0.2],
        }
    )
    # Human curated only the legit edge; group_id is a STALE hash.
    labels = pd.DataFrame(
        {
            "group_id": ["stale_hash"],
            "selected_edges": [json.dumps([{"ref_id": "r1", "target_id": "t1"}])],
            "labeler": ["brad"],
        }
    )
    result = evaluate_stitch_groups(bridge, labels, groups=groups_sidecar)

    assert result.groups_evaluated == 1
    # Raw: pred = {r1-t1, r1-t2}, curated = {r1-t1} -> P=0.5 R=1.0, not exact.
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(1.0)
    assert result.exact_match_rate == pytest.approx(0.0)
    # Filtered: the sliver r1-t2 is removed from pred -> exact match.
    assert result.groups_sliver_affected == 1
    assert result.precision_filtered == pytest.approx(1.0)
    assert result.exact_match_rate_filtered == pytest.approx(1.0)
    assert result.f1_filtered >= result.f1


def test_labeler_breakdown(groups_sidecar):
    """Counts and metrics are split human vs panel."""
    bridge = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "confidence": [0.9]})
    labels = pd.DataFrame(
        {
            "group_id": ["g_current", "g_current"],
            "selected_edges": [
                json.dumps([{"ref_id": "r1", "target_id": "t1"}]),
                json.dumps([{"ref_id": "r1", "target_id": "t1"}]),
            ],
            "labeler": ["brad", "panel_unanimous_v1"],
        }
    )
    result = evaluate_stitch_groups(bridge, labels, groups=groups_sidecar)
    assert result.label_counts_by_labeler == {"brad": 1, "panel_unanimous_v1": 1}
    assert set(result.metrics_by_labeler) == {"human", "panel"}
    assert result.metrics_by_labeler["human"]["n"] == 1
    assert result.metrics_by_labeler["panel"]["n"] == 1


def test_label_lost_when_edges_absent(groups_sidecar):
    """A label whose edges no longer exist in any group is dropped, not crashed."""
    bridge = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "confidence": [0.9]})
    labels = pd.DataFrame(
        {
            "group_id": ["gone"],
            "selected_edges": [json.dumps([{"ref_id": "zzz", "target_id": "qqq"}])],
            "labeler": ["brad"],
        }
    )
    result = evaluate_stitch_groups(bridge, labels, groups=groups_sidecar)
    assert result.groups_evaluated == 0


def test_reject_all_label_retained_on_group_id_match(groups_sidecar):
    """Empty-edge (reject-all) labels survive via verbatim group_id match.

    They cannot be recovered by edge-overlap (no edges), so they map only when
    the original group_id still exists in the sidecar, and they contribute a real
    'no edges' ground-truth case to exact-match / F1.
    """
    # Bridge selected NOTHING for g_current -> matches the reject-all label.
    bridge = pd.DataFrame({"ref_id": ["rX"], "target_id": ["tX"], "confidence": [0.9]})
    labels = pd.DataFrame(
        {
            "group_id": ["g_current"],
            "selected_edges": [json.dumps([])],
            "labeler": ["brad"],
        }
    )
    result = evaluate_stitch_groups(bridge, labels, groups=groups_sidecar)
    assert result.groups_evaluated == 1
    # pred empty, curated empty -> perfect exact match.
    assert result.exact_match_rate == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)


def test_reject_all_label_dropped_when_group_id_gone(groups_sidecar):
    """A reject-all label whose group_id no longer exists is unrecoverable."""
    bridge = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "confidence": [0.9]})
    labels = pd.DataFrame(
        {
            "group_id": ["stale_hash"],
            "selected_edges": [json.dumps([])],
            "labeler": ["brad"],
        }
    )
    result = evaluate_stitch_groups(bridge, labels, groups=groups_sidecar)
    assert result.groups_evaluated == 0


# ---------------------------------------------------------------------------
# SET-semantics metrics (membership / boundary / coverage)
# ---------------------------------------------------------------------------
def _set_group():
    """An M:N group with three candidate edges over {r1,r2} x {t1,t2}."""
    return [
        {
            "group_id": "gset",
            "match_type": "M:N",
            "edges": [
                {"ref_id": "r1", "target_id": "t1"},
                {"ref_id": "r2", "target_id": "t2"},
                {"ref_id": "r1", "target_id": "t2"},
            ],
            "ref_geometries": {},
            "target_geometries": {},
        }
    ]


def _set_label(ref_ids, target_ids, group_id="gset", labeler="brad"):
    return {
        "group_id": group_id,
        "selected_edges": "[]",
        "label_semantics": "set",
        "ref_ids": json.dumps(ref_ids),
        "target_ids": json.dumps(target_ids),
        "labeler": labeler,
    }


def test_set_label_excluded_from_edge_pool_and_scored_separately():
    bridge = pd.DataFrame(
        {"ref_id": ["r1", "r2"], "target_id": ["t1", "t2"], "confidence": [0.9, 0.9]}
    )
    labels = pd.DataFrame([_set_label(["r1", "r2"], ["t1", "t2"])])
    result = evaluate_stitch_groups(bridge, labels, groups=_set_group())
    # Set label does NOT enter the edge-F1 pool.
    assert result.groups_evaluated == 0
    # Scored on set components: optimizer connects exactly the membership.
    assert result.set_groups_evaluated == 1
    assert result.set_membership_exact_rate == pytest.approx(1.0)
    assert result.set_boundary_precision == pytest.approx(1.0)
    assert result.set_coverage == pytest.approx(1.0)
    assert result.set_metrics_by_labeler["human"]["n"] == 1


def test_set_coverage_penalizes_uncovered_member():
    # Optimizer only connects r1-t1; membership claims r2/t2 too.
    bridge = pd.DataFrame({"ref_id": ["r1"], "target_id": ["t1"], "confidence": [0.9]})
    labels = pd.DataFrame([_set_label(["r1", "r2"], ["t1", "t2"])])
    result = evaluate_stitch_groups(bridge, labels, groups=_set_group())
    assert result.set_groups_evaluated == 1
    assert result.set_membership_exact_rate == pytest.approx(0.0)  # r2/t2 not predicted
    assert result.set_boundary_precision == pytest.approx(1.0)  # the one edge is within
    # 2 of 4 members covered (r1, t1).
    assert result.set_coverage == pytest.approx(0.5)


def test_set_boundary_precision_penalizes_cross_edge():
    # Membership is only {r1}/{t1} but the optimizer also keeps r1-t2 (t2 not a
    # member) -> a boundary-crossing edge.
    bridge = pd.DataFrame(
        {"ref_id": ["r1", "r1"], "target_id": ["t1", "t2"], "confidence": [0.9, 0.9]}
    )
    labels = pd.DataFrame([_set_label(["r1"], ["t1"])])
    result = evaluate_stitch_groups(bridge, labels, groups=_set_group())
    assert result.set_groups_evaluated == 1
    assert result.set_membership_exact_rate == pytest.approx(0.0)  # t2 leaked in
    assert result.set_boundary_precision == pytest.approx(0.5)  # 1 of 2 edges within
    assert result.set_coverage == pytest.approx(1.0)


def test_mixed_pair_and_set_labels_split_pools():
    bridge = pd.DataFrame(
        {"ref_id": ["r1", "r2"], "target_id": ["t1", "t2"], "confidence": [0.9, 0.9]}
    )
    labels = pd.DataFrame(
        [
            {
                "group_id": "gset",
                "selected_edges": json.dumps([{"ref_id": "r1", "target_id": "t1"}]),
                "label_semantics": "pair",
                "ref_ids": "",
                "target_ids": "",
                "labeler": "panel_unanimous_v2",
            },
            _set_label(["r1", "r2"], ["t1", "t2"]),
        ]
    )
    result = evaluate_stitch_groups(bridge, labels, groups=_set_group())
    assert result.groups_evaluated == 1  # only the pair label
    assert result.set_groups_evaluated == 1  # only the set label
    # to_dict surfaces both metric families.
    d = result.to_dict()
    assert "stitch_set_membership_exact_rate" in d
    assert d["stitch_set_groups_evaluated"] == 1
