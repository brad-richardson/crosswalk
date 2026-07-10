"""stitch-eval reporting for decomposed groups (#367 Mode B, #388 follow-up).

Before the fix, ``evaluate_batch`` compared every sub-problem row against the
WHOLE-group human label — a category error that emitted N misleading rows per
monster and double-counted the label. Now sub-problem rows are consumed and the
decomposed parent is evaluated ONCE on the recomposed union.
"""

from __future__ import annotations

import json

import pandas as pd

from crosswalk.agent_labeling.stitch_eval import evaluate_batch, summarize

from .test_stitch_export_decomposed import PARENT, make_decomposed_batch


def _human_df(pairs, group_id="hlabel"):
    selected = json.dumps([{"ref_id": r, "target_id": t} for r, t in pairs])
    return pd.DataFrame(
        [{"group_id": group_id, "selected_edges": selected, "label_semantics": "pair"}]
    )


def test_decomposed_group_evaluated_once_on_union(tmp_path):
    b, sub_ids = make_decomposed_batch(
        tmp_path / "b1",
        [
            {"edges": [("r1", "t1")]},
            {"edges": [("r2", "t2")]},
        ],
    )
    human = _human_df([("r1", "t1"), ("r2", "t2")])
    results = evaluate_batch(b, human)

    # Exactly ONE row (the recomposed parent) — no per-sub-problem noise, and no
    # sub-problem id ever appears in the results.
    assert len(results) == 1
    row = results[0]
    assert row.group_id == PARENT
    assert row.is_recomposed is True
    assert row.consensus == "decomposed"
    assert {r.group_id for r in results}.isdisjoint(set(sub_ids))
    # Recomposed union == human label -> exact match.
    assert row.panel_edge_set == frozenset({("r1", "t1"), ("r2", "t2")})
    assert row.exact_match is True
    assert row.f1 == 1.0


def test_no_double_count_in_summary(tmp_path):
    b, sub_ids = make_decomposed_batch(
        tmp_path / "b1",
        [
            {"edges": [("r1", "t1")]},
            {"edges": [("r2", "t2")]},
            {"edges": [("r3", "t3")]},
        ],
    )
    human = _human_df([("r1", "t1"), ("r2", "t2"), ("r3", "t3")])
    summary = summarize(evaluate_batch(b, human))

    # One group, not three: the three sub-problems collapse to one parent.
    assert summary["n_groups"] == 1
    assert summary["panel_exact_rate"] == 1.0
    assert "decomposed" in summary["by_consensus"]
    assert summary["by_consensus"]["decomposed"]["n"] == 1
    assert summary["decomposition"]["recomposed_parents"] == 1
    # Recomposed rows are excluded from the option-coverage denominator (no
    # whole-group option menu) rather than always scored as a gap. n_opt (the
    # denominator) is therefore 0 here, distinct from n_groups == 1.
    assert summary["option_coverage"] == {"n_opt": 0, "covered": 0, "gap": 0, "gap_rate": 0.0}


def test_failed_subproblem_recomposes_partial_union(tmp_path):
    # A failed sub-problem still yields ONE parent row (routing human_review),
    # whose panel edge set is the union of the ACCEPTED sub-selections only.
    b, sub_ids = make_decomposed_batch(
        tmp_path / "b1",
        [
            {"edges": [("r1", "t1")]},
            {"edges": [("r2", "t2")], "routing": "human_review", "chosen": []},
        ],
    )
    human = _human_df([("r1", "t1"), ("r2", "t2")])
    results = evaluate_batch(b, human)

    assert len(results) == 1
    row = results[0]
    assert row.is_recomposed is True
    assert row.routing == "human_review"
    assert row.panel_edge_set == frozenset({("r1", "t1")})  # accepted sub only


def test_option_coverage_denominator_is_n_opt():
    # summarize exposes n_opt (the option-menu group count) separately from
    # n_groups, and the gap RATE is computed against n_opt — a recomposed parent
    # (no whole-group option menu) inflates n_groups but not n_opt, so the CLI
    # must display gap/n_opt, never gap/n_groups.
    from crosswalk.agent_labeling.stitch_eval import GroupEval, summarize

    def _ge(gid, covered, is_recomposed):
        es = frozenset({("r1", "t1")})
        return GroupEval(
            group_id=gid,
            human_group_id=gid,
            match_type="M:N",
            human_edge_set=es,
            consensus="decomposed" if is_recomposed else "unanimous",
            routing="human_review",
            panel_choice="A",
            panel_edge_set=es,
            exact_match=True,
            f1=1.0,
            option_covered=covered,
            is_recomposed=is_recomposed,
        )

    # One real option-menu group (uncovered -> a gap) plus one recomposed parent.
    results = [_ge("plain", covered=False, is_recomposed=False), _ge("parent", False, True)]
    summary = summarize(results)
    oc = summary["option_coverage"]
    assert summary["n_groups"] == 2  # both rows are groups
    assert oc["n_opt"] == 1  # only the non-recomposed row has an option menu
    assert oc["gap"] == 1
    # 1/1 == 1.0, NOT 1/2 == 0.5 — the denominator is n_opt, not n_groups.
    assert oc["gap_rate"] == 1.0


def test_parent_with_direct_row_and_subs_counted_once(tmp_path):
    # Defensive double-count guard (#405): a parent with BOTH a direct
    # whole-group consensus row AND voted sub-problems must be evaluated ONCE
    # (the recomposed union preferred), never twice.
    import csv

    from .test_stitch_export import CONSENSUS_COLUMNS

    b, sub_ids = make_decomposed_batch(
        tmp_path / "b1",
        [{"edges": [("r1", "t1")]}, {"edges": [("r2", "t2")]}],
    )
    # Inject a DIRECT consensus row for the parent (as if an early wave voted the
    # whole group before it was decomposed), with the same union edge set.
    rows = list(csv.DictReader((b / "consensus.csv").open()))
    rows.append(
        {
            "group_id": PARENT,
            "consensus": "unanimous",
            "choice": "A",
            "edge_set": json.dumps([["r1", "t1"], ["r2", "t2"]]),
            "routing": "auto_accept",
            "n_votes": 4,
            "n_valid": 4,
            "minority": "",
            "mean_confidence": 0.9,
            "route_reason": "unanimous",
        }
    )
    with (b / "consensus.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CONSENSUS_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    human = _human_df([("r1", "t1"), ("r2", "t2")])
    results = evaluate_batch(b, human)

    # Exactly ONE row for the parent — the recomposed union, not the direct row.
    parent_rows = [r for r in results if r.group_id == PARENT]
    assert len(parent_rows) == 1
    assert parent_rows[0].is_recomposed is True
    assert summarize(results)["n_groups"] == 1


def test_non_decomposed_batch_unaffected(tmp_path):
    # A batch with no decomposition must behave exactly as before: no synthetic
    # parent rows, no "decomposed" tier, no decomposition summary key.
    import csv

    from .test_stitch_export import _V4_VOTERS, CONSENSUS_COLUMNS, _edge, _line

    batch_dir = tmp_path / "plain"
    batch_dir.mkdir(parents=True)
    # votes.csv WITH edge_set: the whole-group eval path reads per-provider
    # edge sets (the shared _write_votes_csv helper omits that column).
    with (batch_dir / "votes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "provider", "model", "choice", "edge_set"])
        for p, m in _V4_VOTERS:
            w.writerow(["g1", p, m, "A", json.dumps([["r1", "t1"]])])
    edges = [("r1", "t1")]
    groups = [
        {
            "group_id": "g1",
            "match_type": "1:1",
            "ref_ids": ["r1"],
            "target_ids": ["t1"],
            "edges": [_edge("r1", "t1")],
            "ref_geometries": {"r1": _line(42.28, 0.0005)},
            "target_geometries": {"t1": _line(42.29, 0.0005)},
        }
    ]
    (batch_dir / "batch.json").write_text(json.dumps({"dataset_id": "d", "groups": groups}))
    gdir = batch_dir / "g1"
    gdir.mkdir()
    import yaml

    (gdir / "metadata.yaml").write_text(
        yaml.dump(
            {
                "group_id": "g1",
                "match_type": "1:1",
                "segments": {
                    "reference": [{"id": "r1", "label": "R1"}],
                    "target": [{"id": "t1", "label": "T1"}],
                },
                "options": [{"letter": "A", "edges": [{"ref": "R1", "target": "T1"}]}],
            }
        )
    )
    with (batch_dir / "consensus.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CONSENSUS_COLUMNS)
        w.writeheader()
        w.writerow(
            {
                "group_id": "g1",
                "consensus": "unanimous",
                "choice": "A",
                "edge_set": json.dumps([["r1", "t1"]]),
                "routing": "auto_accept",
                "n_votes": 3,
                "n_valid": 3,
                "minority": "",
                "mean_confidence": 0.9,
                "route_reason": "unanimous",
            }
        )
    human = _human_df(edges, group_id="hg1")
    summary = summarize(evaluate_batch(batch_dir, human))
    assert summary["n_groups"] == 1
    assert "decomposed" not in summary["by_consensus"]
    assert "decomposition" not in summary
