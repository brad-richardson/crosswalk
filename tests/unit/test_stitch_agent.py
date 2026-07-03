"""Unit tests for the agent stitching-label pipeline (evidence, runner, eval).

No live CLI calls: subprocess invocation is mocked for runner tests. Synthetic
group fixtures exercise option letter<->edge-set mapping, vote parsing,
consensus rules, evidence metadata, and eval matching.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from matcher.agent_labeling import stitch_runner as sr
from matcher.agent_labeling.stitch_eval import (
    edge_prf,
    evaluate_batch,
    recover_empty_reject_all,
    recover_labeled_groups,
    summarize,
)
from matcher.agent_labeling.stitch_evidence import build_metadata, generate_group_evidence
from matcher.matching.stitch_options import build_stitch_options

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

R1, R2 = "ref-1", "ref-2"
T1, T2 = "tgt_1_88a", "tgt_2_88b"


def _line(coords):
    return {"type": "LineString", "coordinates": coords}


def make_group() -> dict:
    """A 2x2 M:N group with an optimizer assignment and two alternatives."""
    edges = [
        {
            "ref_id": R1,
            "target_id": T1,
            "confidence": 0.9,
            "gers_start_frac": 0.0,
            "gers_end_frac": 1.0,
            "local_start_frac": 0.0,
            "local_end_frac": 1.0,
        },
        {"ref_id": R1, "target_id": T2, "confidence": 0.4},
        {"ref_id": R2, "target_id": T2, "confidence": 0.8},
    ]
    return {
        "group_id": "grp001",
        "match_type": "M:N",
        "ref_ids": [R1, R2],
        "target_ids": [T1, T2],
        "edges": edges,
        "optimizer_assignment": [
            {"ref_id": R1, "target_id": T1},
            {"ref_id": R2, "target_id": T2},
        ],
        "alternatives": [
            {
                "edges": [{"ref_id": R1, "target_id": T1}, {"ref_id": R2, "target_id": T2}],
                "total_confidence": 1.7,
            },
            {
                "edges": [{"ref_id": R1, "target_id": T1}, {"ref_id": R1, "target_id": T2}],
                "total_confidence": 1.3,
            },
        ],
        "ref_geometries": {R1: _line([[0, 0], [1, 1]]), R2: _line([[1, 1], [2, 2]])},
        "target_geometries": {T1: _line([[0, 0.1], [1, 1.1]]), T2: _line([[1, 1.1], [2, 2.1]])},
        "ref_names": {R1: "Main St", R2: "Main St"},
        "target_names": {T1: "MAIN STREET", T2: "MAIN STREET"},
        "ref_classes": {R1: "residential", R2: "residential"},
        "target_classes": {T1: "local", T2: "local"},
    }


# ---------------------------------------------------------------------------
# Option letter <-> edge-set mapping
# ---------------------------------------------------------------------------


def test_build_options_letters_and_optimizer():
    ctx = build_stitch_options(make_group())
    letters = [o["letter"] for o in ctx["options"]]
    # A = optimizer, then deduped alternatives.
    assert letters[0] == "A"
    assert ctx["optimizer_letter"] == "A"
    assert ctx["options"][0]["is_optimizer"] is True
    # Optimizer edge set equals first alternative -> deduped, so B is the 2nd alt.
    edge_sets = {
        o["letter"]: {(e["ref_id"], e["target_id"]) for e in o["edges"]} for o in ctx["options"]
    }
    assert edge_sets["A"] == {(R1, T1), (R2, T2)}
    assert edge_sets["B"] == {(R1, T1), (R1, T2)}


def test_build_options_edges_enriched_with_confidence():
    ctx = build_stitch_options(make_group())
    opt_a = ctx["options"][0]
    e = next(x for x in opt_a["edges"] if x["target_id"] == T1)
    assert e["confidence"] == 0.9
    assert e["gers_start_frac"] == 0.0


def test_build_options_drops_non_group_edges():
    g = make_group()
    g["alternatives"].append({"edges": [{"ref_id": "phantom", "target_id": "ghost"}]})
    ctx = build_stitch_options(g)
    for o in ctx["options"]:
        for e in o["edges"]:
            assert (e["ref_id"], e["target_id"]) in {(R1, T1), (R1, T2), (R2, T2)}


def test_build_options_multiref_alternative_gets_stable_letter():
    """A multi-ref span alternative (T2 spans R1+R2) is a distinct, lettered option.

    Guards the UI / evidence-pack contract: letters stay sequential (A, B, C...)
    and a target-spans-two-refs alternative surfaces as its own option.
    """
    g = make_group()
    # T2 spans both refs R1 and R2 (both edges already exist in the group).
    g["alternatives"].append(
        {
            "edges": [
                {"ref_id": R1, "target_id": T1},
                {"ref_id": R1, "target_id": T2},
                {"ref_id": R2, "target_id": T2},
            ],
            "total_confidence": 2.1,
        }
    )
    ctx = build_stitch_options(g)
    letters = [o["letter"] for o in ctx["options"]]
    assert letters == list("ABC")[: len(letters)]  # sequential, no gaps
    edge_sets = {
        o["letter"]: {(e["ref_id"], e["target_id"]) for e in o["edges"]} for o in ctx["options"]
    }
    # The multi-ref span is present as exactly one option.
    multiref = {(R1, T1), (R1, T2), (R2, T2)}
    assert sum(1 for s in edge_sets.values() if s == multiref) == 1


def test_choice_to_edge_set():
    obl = {"A": [(R1, T1), (R2, T2)], "B": [(R1, T1)]}
    assert sr.choice_to_edge_set("A", obl) == frozenset({(R1, T1), (R2, T2)})
    assert sr.choice_to_edge_set("NONE", obl) == frozenset()
    assert sr.choice_to_edge_set("Z", obl) == frozenset()


# ---------------------------------------------------------------------------
# Vote parsing / validation
# ---------------------------------------------------------------------------


def test_parse_vote_plain_json():
    raw = '{"choice": "A", "confidence": 0.9, "reasoning": "clear"}'
    choice, conf, reason = sr.parse_vote(raw, {"A", "B"})
    assert (choice, conf, reason) == ("A", 0.9, "clear")


def test_parse_vote_fenced_json():
    raw = 'Here you go:\n```json\n{"choice":"B","confidence":0.7,"reasoning":"x"}\n```\n'
    assert sr.parse_vote(raw, {"A", "B"})[0] == "B"


def test_parse_vote_embedded_in_prose():
    raw = 'I think {"choice": "NONE", "confidence": 0.2, "reasoning": "none fit"} is right.'
    assert sr.parse_vote(raw, {"A", "B"})[0] == "NONE"


def test_parse_vote_option_prefix_and_period():
    raw = '{"choice": "Option A.", "confidence": 1, "reasoning": "y"}'
    assert sr.parse_vote(raw, {"A"})[0] == "A"


def test_parse_vote_confidence_clamped():
    raw = '{"choice": "A", "confidence": 5, "reasoning": "y"}'
    assert sr.parse_vote(raw, {"A"})[1] == 1.0


def test_parse_vote_rejects_invalid_letter():
    with pytest.raises(ValueError):
        sr.parse_vote('{"choice": "Q", "confidence": 1, "reasoning": "y"}', {"A", "B"})


def test_parse_vote_rejects_garbage():
    with pytest.raises(ValueError):
        sr.parse_vote("total garbage no json here", {"A"})


def test_parse_vote_missing_choice():
    with pytest.raises(ValueError):
        sr.parse_vote('{"confidence": 1, "reasoning": "y"}', {"A"})


def test_parse_vote_reasoning_containing_braces():
    # A greedy {.*} regex would over-capture; non-greedy {.*?} would truncate
    # at the } inside the reasoning string. raw_decode handles both.
    raw = '{"choice": "A", "confidence": 0.8, "reasoning": "edge set {R1->T1} wins"}'
    choice, conf, reason = sr.parse_vote(raw, {"A"})
    assert choice == "A"
    assert reason == "edge set {R1->T1} wins"


def test_parse_vote_two_concatenated_objects_takes_first():
    raw = (
        '{"choice": "B", "confidence": 0.6, "reasoning": "first"}'
        '{"choice": "A", "confidence": 0.9, "reasoning": "second"}'
    )
    choice, _conf, reason = sr.parse_vote(raw, {"A", "B"})
    assert choice == "B"
    assert reason == "first"


def test_parse_vote_fenced_object_in_prose_with_braces():
    raw = (
        "Considering the options {A, B}:\n"
        "```json\n"
        '{"choice": "B", "confidence": 0.7, "reasoning": "map {x} shows overlap"}\n'
        "```\n"
        "Done."
    )
    choice, _conf, reason = sr.parse_vote(raw, {"A", "B"})
    assert choice == "B"
    assert reason == "map {x} shows overlap"


def test_extract_json_skips_unparseable_brace_runs():
    raw = '{not json} but later {"choice": "A", "confidence": 1, "reasoning": "ok"}'
    assert sr.parse_vote(raw, {"A"})[0] == "A"


# ---------------------------------------------------------------------------
# Consensus rules
# ---------------------------------------------------------------------------


def _vote(provider, choice, es=frozenset()):
    return sr.Vote(
        group_id="g",
        provider=provider,
        model="m",
        choice=choice,
        confidence=0.9,
        reasoning="",
        edge_set=es,
    )


def test_consensus_unanimous_auto_accept():
    es = frozenset({(R1, T1)})
    votes = [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "A", es)]
    c = sr.compute_consensus(votes)
    assert c.consensus == "unanimous"
    assert c.routing == "auto_accept"
    assert c.choice == "A"
    assert c.edge_set == es


def test_consensus_unanimous_none_routes_to_human():
    votes = [_vote("claude", "NONE"), _vote("codex", "NONE"), _vote("agy", "NONE")]
    c = sr.compute_consensus(votes)
    assert c.consensus == "unanimous"
    assert c.routing == "human_review"  # NONE never auto-accepts


def test_consensus_majority():
    votes = [_vote("claude", "A"), _vote("codex", "A"), _vote("agy", "B")]
    c = sr.compute_consensus(votes)
    assert c.consensus == "majority"
    assert c.routing == "human_review"
    assert c.choice == "A"
    assert "agy=B" in c.minority


def test_consensus_none_all_differ():
    votes = [_vote("claude", "A"), _vote("codex", "B"), _vote("agy", "NONE")]
    c = sr.compute_consensus(votes)
    assert c.consensus == "none"
    assert c.routing == "human_review"


def test_consensus_abstention_breaks_unanimity():
    # 2 agree + 1 abstain -> not unanimous (needs all 3 valid), so majority.
    votes = [_vote("claude", "A"), _vote("codex", "A"), _vote("agy", "ABSTAIN")]
    c = sr.compute_consensus(votes)
    assert c.consensus == "majority"
    assert c.n_valid == 2


def test_consensus_all_abstain():
    votes = [_vote("claude", "ABSTAIN"), _vote("codex", "ABSTAIN"), _vote("agy", "ABSTAIN")]
    c = sr.compute_consensus(votes)
    assert c.consensus == "none"
    assert c.n_valid == 0


# ---------------------------------------------------------------------------
# Class-consistency gate
# ---------------------------------------------------------------------------


def test_mode_sets_documented_and_disjoint():
    # The gate's correctness depends on the two mode sets being disjoint and
    # non-empty; a class in both would make is_cross_mode_edge ill-defined.
    assert sr.PEDESTRIAN_CLASSES and sr.VEHICULAR_CLASSES
    assert sr.PEDESTRIAN_CLASSES.isdisjoint(sr.VEHICULAR_CLASSES)
    # Spot-check the canonical members of each mode.
    assert {"footway", "sidewalk", "path"} <= sr.PEDESTRIAN_CLASSES
    assert {"residential", "primary", "service"} <= sr.VEHICULAR_CLASSES


def test_road_class_mode_classification():
    assert sr.road_class_mode("footway") == "pedestrian"
    assert sr.road_class_mode("PRIMARY") == "vehicular"  # case-insensitive
    assert sr.road_class_mode("residential") == "vehicular"
    # Ambiguous / unknown / missing -> neutral (never gates).
    assert sr.road_class_mode("cycleway") == "neutral"
    assert sr.road_class_mode("track") == "neutral"
    assert sr.road_class_mode("unknown") == "neutral"
    assert sr.road_class_mode("") == "neutral"
    assert sr.road_class_mode(None) == "neutral"


def test_is_cross_mode_edge():
    # Pedestrian vs vehicular, either orientation -> cross-mode.
    assert sr.is_cross_mode_edge("footway", "residential")
    assert sr.is_cross_mode_edge("primary", "sidewalk")
    # Same-mode pairs are not cross-mode.
    assert not sr.is_cross_mode_edge("residential", "primary")
    assert not sr.is_cross_mode_edge("footway", "path")
    # Any neutral/missing side passes (do not over-gate on absent data).
    assert not sr.is_cross_mode_edge("cycleway", "residential")
    assert not sr.is_cross_mode_edge("footway", "")
    assert not sr.is_cross_mode_edge("primary", None)


def test_class_gate_demotes_cross_mode_auto_accept():
    es = frozenset({(R1, T1)})
    votes = [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "A", es)]
    # footway ref matched to a residential target -> cross-mode -> demoted.
    c = sr.compute_consensus(votes, edge_classes=[("footway", "residential")])
    assert c.consensus == "unanimous"  # the panel still agreed
    assert c.routing == "human_review"
    assert c.route_reason == "class-mismatch"


def test_class_gate_passes_same_mode_auto_accept():
    es = frozenset({(R1, T1)})
    votes = [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "A", es)]
    c = sr.compute_consensus(votes, edge_classes=[("residential", "primary")])
    assert c.routing == "auto_accept"
    assert c.route_reason == ""


def test_class_gate_passes_missing_or_neutral_class():
    es = frozenset({(R1, T1)})
    votes = [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "A", es)]
    # Missing target class -> pass.
    c = sr.compute_consensus(votes, edge_classes=[("footway", "")])
    assert c.routing == "auto_accept"
    # Neutral (cycleway) vs vehicular -> pass.
    c2 = sr.compute_consensus(votes, edge_classes=[("cycleway", "residential")])
    assert c2.routing == "auto_accept"


def test_class_gate_only_affects_auto_accept():
    # A majority (non-auto-accept) verdict with a cross-mode chosen edge is NOT
    # relabeled "class-mismatch" — it was already routed to human review.
    es = frozenset({(R1, T1)})
    votes = [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "B")]
    c = sr.compute_consensus(votes, edge_classes=[("footway", "residential")])
    assert c.routing == "human_review"
    assert c.route_reason == ""  # reason reserved for the class-gate demotion


def test_class_gate_disabled_without_edge_classes():
    # Backward-compat: no edge_classes -> gate is a no-op.
    es = frozenset({(R1, T1)})
    votes = [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "A", es)]
    c = sr.compute_consensus(votes)
    assert c.routing == "auto_accept"
    assert c.route_reason == ""


# ---------------------------------------------------------------------------
# Runner: retry-once + abstention (mocked subprocess)
# ---------------------------------------------------------------------------


def test_run_provider_retries_once_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_invoker(prompt, group_dir, letters, model, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return "garbage no json"
        return '{"choice": "A", "confidence": 0.8, "reasoning": "ok"}'

    monkeypatch.setitem(sr._INVOKERS, "claude", fake_invoker)
    vote = sr.run_provider_on_group(
        sr.ProviderSpec("claude", "m"),
        "g",
        None,
        "prompt",
        ["A", "B"],
        {"A": [(R1, T1)]},
    )
    assert calls["n"] == 2
    assert vote.choice == "A"
    assert vote.edge_set == frozenset({(R1, T1)})


def test_run_provider_abstains_after_retries(monkeypatch):
    def fake_invoker(prompt, group_dir, letters, model, timeout):
        return "still garbage"

    monkeypatch.setitem(sr._INVOKERS, "codex", fake_invoker)
    vote = sr.run_provider_on_group(
        sr.ProviderSpec("codex", "m"),
        "g",
        None,
        "prompt",
        ["A"],
        {"A": [(R1, T1)]},
    )
    assert vote.choice == "ABSTAIN"
    assert vote.error


def test_run_provider_cli_failure_records_stderr(monkeypatch):
    """Non-zero CLI exit is an invocation failure, not a parse error."""

    def failing_invoker(prompt, group_dir, letters, model, timeout):
        raise RuntimeError("codex exited with code 2: auth expired")

    monkeypatch.setitem(sr._INVOKERS, "codex", failing_invoker)
    vote = sr.run_provider_on_group(
        sr.ProviderSpec("codex", "m"),
        "g",
        None,
        "prompt",
        ["A"],
        {"A": [(R1, T1)]},
    )
    assert vote.choice == "ABSTAIN"
    assert "invocation error" in vote.error
    assert "auth expired" in vote.error


def test_check_exit_raises_with_truncated_stderr():
    import subprocess as sp

    result = sp.CompletedProcess(args=["x"], returncode=3, stdout="", stderr="boom " * 200)
    with pytest.raises(RuntimeError) as exc:
        sr._check_exit("agy", result)
    msg = str(exc.value)
    assert "agy exited with code 3" in msg
    assert len(msg) < 600  # stderr truncated


def test_check_exit_passes_on_zero():
    import subprocess as sp

    result = sp.CompletedProcess(args=["x"], returncode=0, stdout="ok", stderr="")
    sr._check_exit("claude", result)  # no raise


# ---------------------------------------------------------------------------
# Evidence metadata correctness
# ---------------------------------------------------------------------------


def test_build_metadata_option_table():
    g = make_group()
    ctx = build_stitch_options(g)
    meta = build_metadata(g, ctx)
    assert meta["group_id"] == "grp001"
    assert meta["match_type"] == "M:N"
    assert meta["optimizer_letter"] == "A"
    assert meta["n_ref_segments"] == 2
    # Option A is optimizer with 2 edges.
    opt_a = next(o for o in meta["options"] if o["letter"] == "A")
    assert opt_a["is_optimizer"] is True
    assert opt_a["edge_count"] == 2
    # Edge label uses R#/T# and carries confidence.
    e = next(x for x in opt_a["edges"] if x["target"] == "T1")
    assert e["edge"] == "R1->T1"
    assert e["confidence"] == 0.9
    assert e["ref_aligned_frac"] == 1.0
    # Segment names/classes present.
    assert meta["segments"]["reference"][0]["name"] == "Main St"


def test_generate_group_evidence_writes_files(tmp_path):
    g = make_group()
    meta = generate_group_evidence(g, tmp_path / g["group_id"])
    d = tmp_path / g["group_id"]
    assert (d / "overview.png").exists()
    assert (d / "option_A.png").exists()
    assert (d / "option_B.png").exists()
    assert (d / "metadata.yaml").exists()
    assert (d / "prompt.txt").exists()
    prompt = (d / "prompt.txt").read_text()
    assert '"choice"' in prompt
    assert "NONE" in prompt
    assert meta is not None


def test_generate_group_evidence_no_options_returns_none(tmp_path):
    g = make_group()
    g["optimizer_assignment"] = []
    g["alternatives"] = []
    assert generate_group_evidence(g, tmp_path / "x") is None


# ---------------------------------------------------------------------------
# Eval matching logic (exact + F1)
# ---------------------------------------------------------------------------


def test_edge_prf_exact():
    a = frozenset({(R1, T1), (R2, T2)})
    assert edge_prf(a, a) == (1.0, 1.0, 1.0)


def test_edge_prf_partial():
    pred = frozenset({(R1, T1), (R1, T2)})
    truth = frozenset({(R1, T1), (R2, T2)})
    prec, rec, f1 = edge_prf(pred, truth)
    assert prec == 0.5
    assert rec == 0.5
    assert f1 == 0.5


def test_edge_prf_both_empty_is_perfect():
    assert edge_prf(frozenset(), frozenset()) == (1.0, 1.0, 1.0)


def test_edge_prf_empty_pred_nonempty_truth():
    prec, rec, f1 = edge_prf(frozenset(), frozenset({(R1, T1)}))
    assert f1 == 0.0


def test_recover_labeled_groups_classifies():
    groups = [
        {
            "group_id": "gA",
            "edges": [{"ref_id": R1, "target_id": T1}, {"ref_id": R2, "target_id": T2}],
        },
        {"group_id": "gB", "edges": [{"ref_id": "x", "target_id": "y"}]},
    ]
    human_df = pd.DataFrame(
        [
            # clean: both edges in gA
            {
                "group_id": "h_clean",
                "selected_edges": json.dumps(
                    [{"ref_id": R1, "target_id": T1}, {"ref_id": R2, "target_id": T2}]
                ),
            },
            # empty selection -> NONE
            {"group_id": "h_empty", "selected_edges": "[]"},
            # lost: edge not in any group
            {
                "group_id": "h_lost",
                "selected_edges": json.dumps([{"ref_id": "gone", "target_id": "nowhere"}]),
            },
        ]
    )
    rec = recover_labeled_groups(groups, human_df)
    assert ("h_clean", "gA") in rec["clean"]
    assert "h_empty" in rec["empty"]
    assert "h_lost" in rec["lost"]
    assert rec["target_group_ids"] == ["gA"]


def test_recover_empty_reject_all():
    # gA survives verbatim; the reject-all label keyed on gA is recoverable,
    # the one keyed on a vanished group_id is not. Non-empty labels are ignored.
    groups = [{"group_id": "gA", "edges": [{"ref_id": R1, "target_id": T1}]}]
    human_df = pd.DataFrame(
        [
            {"group_id": "gA", "selected_edges": "[]"},  # reject-all, survives
            {"group_id": "gone", "selected_edges": "[]"},  # reject-all, lost
            {
                "group_id": "gA",  # non-empty label -> not a reject-all
                "selected_edges": json.dumps([{"ref_id": R1, "target_id": T1}]),
            },
        ]
    )
    rec = recover_empty_reject_all(groups, human_df)
    assert rec["recovered"] == ["gA"]
    assert rec["unrecoverable"] == ["gone"]


def test_evaluate_batch_end_to_end(tmp_path):
    """Generate a pack, fake a consensus/votes, run the eval."""
    g = make_group()
    batch_dir = tmp_path / "batch"
    generate_group_evidence(g, batch_dir / g["group_id"])

    # Human label picks the optimizer's edge set (A).
    human_es = [{"ref_id": R1, "target_id": T1}, {"ref_id": R2, "target_id": T2}]
    human_df = pd.DataFrame([{"group_id": "hg", "selected_edges": json.dumps(human_es)}])

    # Panel unanimously chose A (matches human).
    edge_set_str = json.dumps(sorted([[R1, T1], [R2, T2]]))
    pd.DataFrame(
        [
            {
                "group_id": "grp001",
                "consensus": "unanimous",
                "choice": "A",
                "edge_set": edge_set_str,
                "routing": "auto_accept",
                "n_votes": 3,
                "n_valid": 3,
                "minority": "",
                "mean_confidence": 0.9,
            }
        ]
    ).to_csv(batch_dir / "consensus.csv", index=False)
    pd.DataFrame(
        [
            {
                "group_id": "grp001",
                "provider": p,
                "model": "m",
                "choice": "A",
                "confidence": 0.9,
                "reasoning": "",
                "edge_set": edge_set_str,
                "latency_s": 1.0,
                "timestamp": "",
                "error": "",
            }
            for p in ("claude", "codex", "agy")
        ]
    ).to_csv(batch_dir / "votes.csv", index=False)

    results = evaluate_batch(batch_dir, human_df)
    assert len(results) == 1
    r = results[0]
    assert r.exact_match is True
    assert r.f1 == 1.0
    assert r.option_covered is True

    summary = summarize(results)
    assert summary["panel_exact_rate"] == 1.0
    assert summary["by_provider"]["claude"]["exact_rate"] == 1.0
    assert summary["by_consensus"]["unanimous"]["n"] == 1


def test_evaluate_batch_disagreement(tmp_path):
    g = make_group()
    batch_dir = tmp_path / "batch"
    generate_group_evidence(g, batch_dir / g["group_id"])

    # Human picked a DIFFERENT edge set than the panel.
    human_es = [{"ref_id": R1, "target_id": T2}]
    human_df = pd.DataFrame([{"group_id": "hg", "selected_edges": json.dumps(human_es)}])
    panel_str = json.dumps(sorted([[R1, T1], [R2, T2]]))
    pd.DataFrame(
        [
            {
                "group_id": "grp001",
                "consensus": "unanimous",
                "choice": "A",
                "edge_set": panel_str,
                "routing": "auto_accept",
                "n_votes": 3,
                "n_valid": 3,
                "minority": "",
                "mean_confidence": 0.9,
            }
        ]
    ).to_csv(batch_dir / "consensus.csv", index=False)
    pd.DataFrame(
        [
            {
                "group_id": "grp001",
                "provider": "claude",
                "model": "m",
                "choice": "A",
                "confidence": 0.9,
                "reasoning": "",
                "edge_set": panel_str,
                "latency_s": 1.0,
                "timestamp": "",
                "error": "",
            }
        ]
    ).to_csv(batch_dir / "votes.csv", index=False)

    results = evaluate_batch(batch_dir, human_df)
    assert len(results) == 1
    assert results[0].exact_match is False


# ---------------------------------------------------------------------------
# Human-label -> group mapping (edge-level overlap preferred)
# ---------------------------------------------------------------------------


def _mapping_fixtures():
    from matcher.agent_labeling.stitch_evidence import build_metadata

    g = make_group()
    ctx = build_stitch_options(g)
    metas = {g["group_id"]: build_metadata(g, ctx)}
    cand = {g["group_id"]: frozenset({(R1, T1), (R1, T2), (R2, T2)})}
    return metas, cand


def test_mapping_requires_edge_overlap_when_candidates_known():
    """A label whose segments are in the group but whose edges never existed
    as candidate edges must NOT map (would skew coverage/agreement metrics)."""
    from matcher.agent_labeling.stitch_eval import map_human_labels_to_groups

    metas, cand = _mapping_fixtures()
    # (R2, T1) uses group segments but is not a candidate edge.
    human_df = pd.DataFrame(
        [
            {
                "group_id": "h_phantom",
                "selected_edges": json.dumps([{"ref_id": R2, "target_id": T1}]),
            }
        ]
    )
    assert map_human_labels_to_groups(human_df, metas, cand) == {}


def test_mapping_prefers_max_edge_overlap():
    from matcher.agent_labeling.stitch_eval import map_human_labels_to_groups

    metas, cand = _mapping_fixtures()
    human_df = pd.DataFrame(
        [
            {
                "group_id": "h_one_edge",
                "selected_edges": json.dumps([{"ref_id": R1, "target_id": T1}]),
            },
            {
                "group_id": "h_two_edges",
                "selected_edges": json.dumps(
                    [
                        {"ref_id": R1, "target_id": T1},
                        {"ref_id": R2, "target_id": T2},
                    ]
                ),
            },
        ]
    )
    mapping = map_human_labels_to_groups(human_df, metas, cand)
    assert mapping == {"grp001": "h_two_edges"}


def test_mapping_falls_back_to_segment_membership_without_candidates():
    from matcher.agent_labeling.stitch_eval import map_human_labels_to_groups

    metas, _cand = _mapping_fixtures()
    # Same phantom edge; without candidate edges the segment fallback applies.
    human_df = pd.DataFrame(
        [
            {
                "group_id": "h_phantom",
                "selected_edges": json.dumps([{"ref_id": R2, "target_id": T1}]),
            }
        ]
    )
    mapping = map_human_labels_to_groups(human_df, metas, None)
    assert mapping == {"grp001": "h_phantom"}


def test_evaluate_batch_uses_batch_json_candidate_edges(tmp_path):
    """With batch.json present, a phantom-edge label must not be mapped."""
    g = make_group()
    batch_dir = tmp_path / "batch"
    generate_group_evidence(g, batch_dir / g["group_id"])
    (batch_dir / "batch.json").write_text(json.dumps({"groups": [g]}))

    human_df = pd.DataFrame(
        [
            {
                "group_id": "h_phantom",
                "selected_edges": json.dumps([{"ref_id": R2, "target_id": T1}]),
            }
        ]
    )
    panel_str = json.dumps(sorted([[R1, T1], [R2, T2]]))
    pd.DataFrame(
        [
            {
                "group_id": "grp001",
                "consensus": "unanimous",
                "choice": "A",
                "edge_set": panel_str,
                "routing": "auto_accept",
                "n_votes": 3,
                "n_valid": 3,
                "minority": "",
                "mean_confidence": 0.9,
            }
        ]
    ).to_csv(batch_dir / "consensus.csv", index=False)
    pd.DataFrame(
        [
            {
                "group_id": "grp001",
                "provider": "claude",
                "model": "m",
                "choice": "A",
                "confidence": 0.9,
                "reasoning": "",
                "edge_set": panel_str,
                "latency_s": 1.0,
                "timestamp": "",
                "error": "",
            }
        ]
    ).to_csv(batch_dir / "votes.csv", index=False)

    assert evaluate_batch(batch_dir, human_df) == []
