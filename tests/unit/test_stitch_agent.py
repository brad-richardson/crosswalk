"""Unit tests for the agent stitching-label pipeline (evidence, runner, eval).

No live CLI calls: subprocess invocation is mocked for runner tests. Synthetic
group fixtures exercise option letter<->edge-set mapping, vote parsing,
consensus rules, evidence metadata, and eval matching.
"""

from __future__ import annotations

import copy
import errno
import json
import math
import subprocess

import pandas as pd
import pytest

from crosswalk.agent_labeling import stitch_runner as sr
from crosswalk.agent_labeling.stitch_eval import (
    edge_prf,
    evaluate_batch,
    recover_empty_reject_all,
    recover_labeled_groups,
    summarize,
)
from crosswalk.agent_labeling.stitch_evidence import (
    build_metadata,
    build_prompt,
    generate_group_evidence,
    prune_options_for_panel,
)
from crosswalk.matching.stitch_options import build_stitch_options

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


def test_build_options_duplicated_raw_edges_do_not_inflate_confidence():
    """Duplicated / out-of-group raw edges must not inflate the displayed confidence.

    Regression for the option-generation defect: an alternative whose edge list
    contained many duplicated / cross-group edges (and an inflated stored
    ``total_confidence``) rendered as a nonsensical mean confidence (e.g.
    "3242%"). The displayed total/mean must be computed over the validated,
    deduplicated edge set actually shown.
    """
    g = make_group()
    g["alternatives"] = [
        {
            # Same in-group edge repeated 30x plus a phantom edge, with a wildly
            # inflated stored total_confidence.
            "edges": [{"ref_id": R1, "target_id": T1}] * 30
            + [{"ref_id": "phantom", "target_id": "ghost"}],
            "total_confidence": 227.0,
        }
    ]
    # Drop the optimizer so the alternative is option A.
    g["optimizer_assignment"] = []
    ctx = build_stitch_options(g)
    opt = ctx["options"][0]
    assert opt["edge_count"] == 1  # only the single distinct in-group edge shows
    assert opt["total_confidence"] == pytest.approx(0.9)  # R1->T1 confidence, not 227
    assert opt["mean_confidence"] == pytest.approx(0.9)


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
    assert c.route_reason == "unanimous"


def test_consensus_unanimous_none_routes_to_human():
    votes = [_vote("claude", "NONE"), _vote("codex", "NONE"), _vote("agy", "NONE")]
    c = sr.compute_consensus(votes)
    assert c.consensus == "unanimous"
    assert c.routing == "human_review"  # NONE never auto-accepts
    assert c.route_reason == "unanimous_none"


def test_consensus_majority():
    votes = [_vote("claude", "A"), _vote("codex", "A"), _vote("agy", "B")]
    c = sr.compute_consensus(votes)
    assert c.consensus == "majority"
    assert c.routing == "human_review"
    assert c.choice == "A"
    assert "agy=B" in c.minority
    assert c.route_reason == "dissent:agy=B"


def test_consensus_none_all_differ():
    votes = [_vote("claude", "A"), _vote("codex", "B"), _vote("agy", "NONE")]
    c = sr.compute_consensus(votes)
    assert c.consensus == "none"
    assert c.routing == "human_review"
    assert c.route_reason == "no_majority"


def test_consensus_abstention_breaks_unanimity():
    # 2 agree + 1 abstain -> not unanimous (needs all 3 valid), so majority.
    votes = [_vote("claude", "A"), _vote("codex", "A"), _vote("agy", "ABSTAIN")]
    c = sr.compute_consensus(votes)
    assert c.consensus == "majority"
    assert c.n_valid == 2
    assert c.route_reason == "below_quorum:2"


def test_consensus_all_abstain():
    votes = [_vote("claude", "ABSTAIN"), _vote("codex", "ABSTAIN"), _vote("agy", "ABSTAIN")]
    c = sr.compute_consensus(votes)
    assert c.consensus == "none"
    assert c.n_valid == 0
    assert c.route_reason == "all_abstained"


# ---------------------------------------------------------------------------
# Class-consistency gate
# ---------------------------------------------------------------------------


def test_mode_sets_documented_and_disjoint():
    # The gate's correctness depends on the mode sets being pairwise disjoint and
    # non-empty; a class in two sets would make is_cross_mode_edge ill-defined.
    assert sr.PEDESTRIAN_CLASSES and sr.VEHICULAR_CLASSES and sr.CYCLEWAY_CLASSES
    assert sr.PEDESTRIAN_CLASSES.isdisjoint(sr.VEHICULAR_CLASSES)
    assert sr.PEDESTRIAN_CLASSES.isdisjoint(sr.CYCLEWAY_CLASSES)
    assert sr.VEHICULAR_CLASSES.isdisjoint(sr.CYCLEWAY_CLASSES)
    # Spot-check the canonical members of each mode.
    assert {"footway", "sidewalk", "path"} <= sr.PEDESTRIAN_CLASSES
    assert {"residential", "primary", "service"} <= sr.VEHICULAR_CLASSES
    assert {"cycleway"} <= sr.CYCLEWAY_CLASSES


def test_road_class_mode_classification():
    assert sr.road_class_mode("footway") == "pedestrian"
    assert sr.road_class_mode("PRIMARY") == "vehicular"  # case-insensitive
    assert sr.road_class_mode("residential") == "vehicular"
    # cycleway is its own bike mode (no longer neutral).
    assert sr.road_class_mode("cycleway") == "bike"
    assert sr.road_class_mode("CYCLEWAY") == "bike"  # case-insensitive
    # Ambiguous / unknown / missing -> neutral (never gates).
    assert sr.road_class_mode("track") == "neutral"
    assert sr.road_class_mode("unknown") == "neutral"
    assert sr.road_class_mode("") == "neutral"
    assert sr.road_class_mode(None) == "neutral"


def test_is_cross_mode_edge():
    # Any two DIFFERENT non-neutral modes, either orientation -> cross-mode.
    assert sr.is_cross_mode_edge("footway", "residential")  # pedestrian<->vehicular
    assert sr.is_cross_mode_edge("primary", "sidewalk")
    # road<->cycleway is cross-mode (Brad's 2026-07-05 decision).
    assert sr.is_cross_mode_edge("cycleway", "residential")
    assert sr.is_cross_mode_edge("primary", "cycleway")
    # pedestrian<->cycleway is cross-mode (conservative default; see PR body).
    assert sr.is_cross_mode_edge("footway", "cycleway")
    assert sr.is_cross_mode_edge("cycleway", "sidewalk")
    # Same-mode pairs are not cross-mode.
    assert not sr.is_cross_mode_edge("residential", "primary")
    assert not sr.is_cross_mode_edge("footway", "path")
    assert not sr.is_cross_mode_edge("cycleway", "cycleway")  # bike<->bike stays same-mode
    # Any neutral/missing side passes (do not over-gate on absent data).
    assert not sr.is_cross_mode_edge("track", "residential")
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
    assert c.route_reason == "unanimous"


def test_class_gate_passes_missing_or_neutral_class():
    es = frozenset({(R1, T1)})
    votes = [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "A", es)]
    # Missing target class -> pass.
    c = sr.compute_consensus(votes, edge_classes=[("footway", "")])
    assert c.routing == "auto_accept"
    # Neutral (track) vs vehicular -> pass.
    c2 = sr.compute_consensus(votes, edge_classes=[("track", "residential")])
    assert c2.routing == "auto_accept"


def test_class_gate_demotes_road_cycleway_auto_accept():
    # road<->cycleway (co_bogota_bike_network shape: primary ref, cycleway target)
    # is cross-mode and must route to human review, not auto-accept.
    es = frozenset({(R1, T1)})
    votes = [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "A", es)]
    c = sr.compute_consensus(votes, edge_classes=[("primary", "cycleway")])
    assert c.consensus == "unanimous"
    assert c.routing == "human_review"
    assert c.route_reason == "class-mismatch"


def test_class_gate_passes_cycleway_cycleway_auto_accept():
    # cycleway<->cycleway is same-mode and stays auto-acceptable.
    es = frozenset({(R1, T1)})
    votes = [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "A", es)]
    c = sr.compute_consensus(votes, edge_classes=[("cycleway", "cycleway")])
    assert c.routing == "auto_accept"
    assert c.route_reason == "unanimous"


def test_class_gate_only_affects_auto_accept():
    # A majority (non-auto-accept) verdict with a cross-mode chosen edge is NOT
    # relabeled "class-mismatch" — it was already routed to human review, and
    # its reason reflects the vote outcome (the dissent), not the class gate.
    es = frozenset({(R1, T1)})
    votes = [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "B")]
    c = sr.compute_consensus(votes, edge_classes=[("footway", "residential")])
    assert c.routing == "human_review"
    assert c.route_reason == "dissent:agy=B"


def test_class_gate_disabled_without_edge_classes():
    # Backward-compat: no edge_classes -> gate is a no-op.
    es = frozenset({(R1, T1)})
    votes = [_vote("claude", "A", es), _vote("codex", "A", es), _vote("agy", "A", es)]
    c = sr.compute_consensus(votes)
    assert c.routing == "auto_accept"
    assert c.route_reason == "unanimous"


# ---------------------------------------------------------------------------
# Runner: retry-once + abstention (mocked subprocess)
# ---------------------------------------------------------------------------


def test_run_provider_retries_once_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
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
    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
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


def test_run_provider_cli_failure_hard_fails(monkeypatch):
    """A non-zero CLI exit is a provider-down signal -> hard-fail, not abstain.

    (Previously this abstained; the panel now halts on invocation/quota errors
    rather than silently dropping the voter. budget=0 hard-fails immediately.)
    """

    def failing_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        raise RuntimeError("codex exited with code 2: auth expired")

    monkeypatch.setitem(sr._INVOKERS, "codex", failing_invoker)
    with pytest.raises(sr.ProviderInvocationError, match="auth expired"):
        sr.run_provider_on_group(
            sr.ProviderSpec("codex", "m"),
            "g",
            None,
            "prompt",
            ["A"],
            {"A": [(R1, T1)]},
            invocation_budget_s=0.0,
        )


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


def test_image_paths_include_junction_zoom_crops(tmp_path):
    """codex only sees images attached via -i, so zoom_*.png must be listed.

    Regression for the enriched_ab1 wave: #302 packs referenced junction zoom
    crops in the prompt, but _image_paths omitted them, leaving codex blind to
    the crops that claude/agy read by path.
    """
    (tmp_path / "overview.png").write_bytes(b"png")
    (tmp_path / "option_A.png").write_bytes(b"png")
    (tmp_path / "zoom_R3_T8.png").write_bytes(b"png")
    (tmp_path / "zoom_R1_T2.png").write_bytes(b"png")

    imgs = sr._image_paths(tmp_path, ["A"])

    assert imgs[0].endswith("overview.png")
    assert imgs[1].endswith("option_A.png")
    assert [p for p in imgs if "zoom_" in p] == [
        str(tmp_path / "zoom_R1_T2.png"),
        str(tmp_path / "zoom_R3_T8.png"),
    ]


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


def test_prompt_dedupes_edge_descriptors_across_options(tmp_path):
    """Each distinct edge is described ONCE in an EDGES legend; options reference
    short ids. A shared edge is no longer re-printed in longhand per option.

    make_group() has options A={R1->T1, R2->T2} and B={R1->T1, R1->T2}, so R1->T1
    is shared -> its descriptor (`conf=`) line must appear exactly once.
    """
    g = make_group()
    ctx = build_stitch_options(g)
    meta = build_metadata(g, ctx)
    prompt = build_prompt(tmp_path, meta, ctx)

    # Legend header present; options no longer print `conf=` lines themselves.
    assert "EDGES (" in prompt
    lines = prompt.splitlines()
    edges_i = next(i for i, ln in enumerate(lines) if ln.startswith("EDGES ("))
    options_i = next(i for i, ln in enumerate(lines) if ln.startswith("OPTIONS:"))
    legend = lines[edges_i:options_i]
    options_block = lines[options_i:]

    # Distinct edge universe = {R1->T1, R2->T2, R1->T2} -> exactly 3 legend rows,
    # each carrying full detail once (the shared R1->T1 appears exactly once).
    legend_rows = [ln for ln in legend if "conf=" in ln]
    assert len(legend_rows) == 3
    assert sum("R1->T1" in ln for ln in legend_rows) == 1
    # No descriptor `conf=` lines leak into the OPTIONS section.
    assert not any("conf=" in ln for ln in options_block if ln.startswith("      "))

    # Build the short-id -> R#->T# map from the legend, then confirm each option's
    # `edges:` line reconstructs exactly the option's true edge set (unambiguous).
    id_to_edge = {}
    for ln in legend_rows:
        eid, rest = ln.strip().split(":", 1)
        id_to_edge[eid] = rest.strip().split()[0]
    # Options are emitted in order, each with one `edges:` line listing its short ids.
    edge_lines = [
        ln.strip()[len("edges: ") :] for ln in options_block if ln.strip().startswith("edges:")
    ]
    assert len(edge_lines) == len(meta["options"])
    for opt, refs in zip(meta["options"], edge_lines):
        got = {id_to_edge[r.strip()] for r in refs.split(",")}
        want = {e["edge"] for e in opt["edges"]}
        assert got == want


def test_generate_group_evidence_no_options_returns_none(tmp_path):
    g = make_group()
    g["optimizer_assignment"] = []
    g["alternatives"] = []
    assert generate_group_evidence(g, tmp_path / "x") is None


# ---------------------------------------------------------------------------
# #267 structural pass-through + junction-sliver display (packs enrichment)
# ---------------------------------------------------------------------------


def _line_len(length_m, lat=42.36, lon=-71.06):
    deg = length_m / (111000.0 * math.cos(math.radians(lat)))
    return {"type": "LineString", "coordinates": [[lon, lat], [lon + deg, lat]]}


def make_struct_group() -> dict:
    """A 1x2 group carrying #267 structural fields and a BORDERLINE junction edge.

    R1->T1 is a full-coverage continuation; R1->T2 shares only 2.9% of a 200 m
    ref (~5.8 m absolute overlap) -> fails the sliver test only on the 5 m floor
    -> BORDERLINE.
    """
    edges = [
        {
            "ref_id": R1,
            "target_id": T1,
            "confidence": 0.99,
            "gers_start_frac": 0.0,
            "gers_end_frac": 1.0,
            "local_start_frac": 0.0,
            "local_end_frac": 1.0,
            "degree_ref": 2,
            "degree_tgt": 1,
            "is_bridge": True,
            "biconnected_block": 3,
            "corridor_ref": 0,
            "corridor_tgt": 0,
        },
        {
            "ref_id": R1,
            "target_id": T2,
            "confidence": 0.88,
            "gers_start_frac": 0.0,
            "gers_end_frac": 0.029,
            "local_start_frac": 0.0,
            "local_end_frac": 0.025,
            "degree_ref": 4,
            "degree_tgt": 2,
            "is_bridge": False,
            "biconnected_block": 3,
            "corridor_ref": 0,
            "corridor_tgt": 1,
        },
    ]
    return {
        "group_id": "grp_struct",
        "match_type": "M:N",
        "ref_ids": [R1],
        "target_ids": [T1, T2],
        "edges": edges,
        "optimizer_assignment": [{"ref_id": R1, "target_id": T1}, {"ref_id": R1, "target_id": T2}],
        "alternatives": [{"edges": [{"ref_id": R1, "target_id": T1}]}],
        "ref_geometries": {R1: _line_len(200.0)},
        "target_geometries": {T1: _line_len(200.0), T2: _line_len(50.0)},
        "ref_names": {R1: "Main St"},
        "target_names": {T1: "Main", T2: "Side"},
        "ref_classes": {R1: "residential"},
        "target_classes": {T1: "residential", T2: "residential"},
        "n_corridors": 2,
        "n_assignment_components": 1,
        "largest_biconnected_block": 3,
        "oversized_group": False,
    }


def test_valid_edges_passes_through_structural_fields():
    """build_stitch_options must keep the six #267 per-edge fields."""
    ctx = build_stitch_options(make_struct_group())
    opt = ctx["options"][0]
    e = next(x for x in opt["edges"] if x["target_id"] == T2)
    for k in (
        "degree_ref",
        "degree_tgt",
        "is_bridge",
        "biconnected_block",
        "corridor_ref",
        "corridor_tgt",
    ):
        assert k in e, f"{k} stripped from enriched edge"
    assert e["degree_ref"] == 4
    assert e["corridor_tgt"] == 1


def test_valid_edges_omits_missing_structural_fields():
    """Older sidecars without structural fields degrade gracefully (omitted)."""
    g = make_group()  # fixture with no #267 fields
    ctx = build_stitch_options(g)
    for opt in ctx["options"]:
        for e in opt["edges"]:
            assert "degree_ref" not in e
            assert "is_bridge" not in e


def test_build_metadata_surfaces_overlap_tag_and_structure():
    g = make_struct_group()
    ctx = build_stitch_options(g)
    meta = build_metadata(g, ctx)
    # Group-level structure summary present.
    assert meta["structure"]["n_corridors"] == 2
    assert meta["structure"]["oversized_group"] is False
    opt_a = next(o for o in meta["options"] if o["is_optimizer"])
    assert opt_a["borderline_edge_count"] == 1
    # The junction-kiss edge carries overlap + BORDERLINE tag + struct fields.
    e = next(x for x in opt_a["edges"] if x["target"] == "T2")
    assert e["tag"] == "BORDERLINE"
    assert e["overlap_m"] == pytest.approx(5.8, abs=0.2)
    assert e["degree_ref"] == 4
    assert e["is_sliver"] is False
    # The full-coverage continuation is untagged.
    e1 = next(x for x in opt_a["edges"] if x["target"] == "T1")
    assert "tag" not in e1


def test_build_metadata_graceful_without_structure():
    """No #267 fields anywhere -> empty structure dict, no per-edge struct keys."""
    g = make_group()
    meta = build_metadata(g, build_stitch_options(g))
    assert meta["structure"] == {}
    for opt in meta["options"]:
        for e in opt["edges"]:
            assert "degree_ref" not in e


def test_generate_evidence_renders_junction_zoom_and_prompt(tmp_path):
    g = make_struct_group()
    d = tmp_path / g["group_id"]
    meta = generate_group_evidence(g, d)
    # A zoom crop was rendered for the BORDERLINE edge and referenced in metadata.
    assert meta["zoom_crops"] == ["zoom_R1_T2.png"]
    assert (d / "zoom_R1_T2.png").exists()
    prompt = (d / "prompt.txt").read_text()
    assert "BORDERLINE" in prompt
    assert "overlap~" in prompt
    assert "deg R4/T2" in prompt
    assert "Group structure:" in prompt
    assert "junction zoom:" in prompt
    assert str(d.resolve() / "zoom_R1_T2.png") in prompt


def test_no_zoom_crop_for_edge_absent_from_all_options(tmp_path):
    """A flagged group edge that no option displays gets no crop (no orphan files)."""
    g = make_struct_group()
    # A second junction-kiss edge, present in the group's candidate edges but in
    # NO option (not in the optimizer assignment nor any alternative).
    t3 = "tgt_3_88c"
    g["target_ids"].append(t3)
    g["target_geometries"][t3] = _line_len(50.0)
    g["target_names"][t3] = "Orphan"
    g["target_classes"][t3] = "residential"
    g["edges"].append(
        {
            "ref_id": R1,
            "target_id": t3,
            "confidence": 0.5,
            "gers_start_frac": 0.0,
            "gers_end_frac": 0.029,
            "local_start_frac": 0.0,
            "local_end_frac": 0.025,
        }
    )
    d = tmp_path / g["group_id"]
    meta = generate_group_evidence(g, d)
    # Only the in-option BORDERLINE edge (R1->T2) gets a crop; the orphan does not.
    assert meta["zoom_crops"] == ["zoom_R1_T2.png"]
    assert not (d / "zoom_R1_T3.png").exists()
    # Every crop file on disk is referenced by the prompt.
    prompt = (d / "prompt.txt").read_text()
    for z in d.glob("zoom_*.png"):
        assert str(z.resolve()) in prompt


def test_pack_size_and_zoom_crop_cap_bounded(tmp_path):
    """Pack stays in the measured order of magnitude and zoom crops are capped."""
    from crosswalk.agent_labeling import stitch_evidence as se

    g = make_struct_group()
    d = tmp_path / g["group_id"]
    generate_group_evidence(g, d)
    pngs = list(d.glob("*.png"))
    zooms = list(d.glob("zoom_*.png"))
    assert len(zooms) <= se.MAX_ZOOM_CROPS
    total = sum(p.stat().st_size for p in d.glob("*") if p.is_file())
    # Hardest measured packs were ~620 KB; keep the same order of magnitude with
    # a generous ceiling so a runaway crop count would trip this.
    assert total < 1_500_000, f"pack too large: {total} bytes across {len(pngs)} PNGs"


# ---------------------------------------------------------------------------
# Panel option pruning (diverse subset for monster groups)
# ---------------------------------------------------------------------------


def _pairs(spec: list[int]) -> list[tuple[str, str]]:
    return [(f"r{i}", f"t{i}") for i in spec]


def _mk_option(key: str, pairs: list[tuple[str, str]], conf: float, is_optimizer=False) -> dict:
    """Hand-build an option dict shaped like build_stitch_options output."""
    edges = [{"ref_id": r, "target_id": t, "confidence": conf} for r, t in pairs]
    total = round(conf * len(edges), 4)
    return {
        "key": key,
        "label": key,
        "is_optimizer": is_optimizer,
        "edges": edges,
        "edge_count": len(edges),
        "total_confidence": total,
        "mean_confidence": conf,
        "active_refs": sorted({r for r, _ in pairs}),
        "active_targets": sorted({t for _, t in pairs}),
    }


def _mk_ctx(options: list[dict]) -> dict:
    for i, o in enumerate(options):
        o["letter"] = chr(ord("A") + i)
    opt_letter = next((o["letter"] for o in options if o["is_optimizer"]), None)
    return {"options": options, "optimizer_letter": opt_letter}


def _edge_sets(ctx: dict) -> list[frozenset]:
    return [frozenset((e["ref_id"], e["target_id"]) for e in o["edges"]) for o in ctx["options"]]


def test_prune_noop_below_option_count():
    """<= max_options is a no-op: ctx untouched, no provenance (byte-identical packs)."""
    g = make_group()
    ctx = build_stitch_options(g)
    snapshot = copy.deepcopy(ctx)
    assert prune_options_for_panel(ctx, g) is None  # settings defaults
    assert ctx == snapshot


def test_prune_noop_below_distinct_edge_trigger():
    """Many options over FEW distinct edges is a no-op: small groups keep everything."""
    g = make_group()  # 3 distinct candidate edges
    ctx = build_stitch_options(g)
    snapshot = copy.deepcopy(ctx)
    assert prune_options_for_panel(ctx, g, max_options=1, min_distinct_edges_trigger=5) is None
    assert ctx == snapshot


def test_prune_max_min_diversity_beats_confidence():
    """Greedy fill must pick the most DISTINCT option, not the most confident.

    The near-duplicate of the optimizer (symmetric difference 1) has much higher
    total confidence than the fully disjoint option (symmetric difference 10);
    greedy-by-confidence would keep the near-duplicate, diversity keeps the
    disjoint one.
    """
    opt = _mk_option("optimizer", _pairs(list(range(5))), 0.9, is_optimizer=True)
    near_dup = _mk_option("alt1", _pairs(list(range(4))), 0.9)  # optimizer minus one edge
    disjoint = _mk_option("alt2", _pairs(list(range(5, 10))), 0.4)
    ctx = _mk_ctx([opt, near_dup, disjoint])
    group = {
        "edges": [
            {"ref_id": r, "target_id": t, "confidence": 0.5} for r, t in _pairs(list(range(10)))
        ]
    }

    info = prune_options_for_panel(ctx, group, max_options=2, min_distinct_edges_trigger=5)

    assert info == {"n_before": 3, "n_after": 2, "dropped_keys": ["alt1"]}
    assert _edge_sets(ctx) == [
        frozenset(_pairs(list(range(5)))),
        frozenset(_pairs(list(range(5, 10)))),
    ]
    # Survivors re-lettered A, B; optimizer letter tracks the survivor.
    assert [o["letter"] for o in ctx["options"]] == ["A", "B"]
    assert ctx["optimizer_letter"] == "A"


def test_prune_keeps_optimizer_and_seed_options():
    """Optimizer + whole-group seeds (full set / optimizer-selected set) always survive.

    Seed identity does not survive build_stitch_options (no is_seed on options),
    so the pruner re-identifies them by edge set from the group's edges.
    """
    all_pairs = _pairs(list(range(6)))
    selected = all_pairs[:2]
    g = {
        "group_id": "grp_big",
        "match_type": "M:N",
        "ref_ids": [r for r, _ in all_pairs],
        "target_ids": [t for _, t in all_pairs],
        "edges": [
            {"ref_id": r, "target_id": t, "confidence": 0.9, "selected": (r, t) in selected}
            for r, t in all_pairs
        ],
        "optimizer_assignment": [{"ref_id": all_pairs[0][0], "target_id": all_pairs[0][1]}],
        "alternatives": [
            # Seeds as generate_top_k_alternatives appends them (identity lost).
            {"edges": [{"ref_id": r, "target_id": t} for r, t in all_pairs]},
            {"edges": [{"ref_id": r, "target_id": t} for r, t in selected]},
            # High-confidence near-duplicate fillers that diversity should drop.
            {"edges": [{"ref_id": r, "target_id": t} for r, t in [all_pairs[0], all_pairs[2]]]},
            {"edges": [{"ref_id": r, "target_id": t} for r, t in [all_pairs[0], all_pairs[3]]]},
            {"edges": [{"ref_id": r, "target_id": t} for r, t in [all_pairs[0], all_pairs[4]]]},
        ],
    }
    ctx = build_stitch_options(g)
    assert len(ctx["options"]) == 6

    info = prune_options_for_panel(ctx, g, max_options=3, min_distinct_edges_trigger=5)

    assert info is not None and info["n_after"] == 3
    kept = _edge_sets(ctx)
    assert frozenset(all_pairs[:1]) in kept  # optimizer proposal
    assert frozenset(all_pairs) in kept  # full-candidate-set seed
    assert frozenset(selected) in kept  # optimizer-selected-set seed
    assert ctx["optimizer_letter"] == "A"


def test_prune_without_protected_starts_from_highest_confidence():
    """No optimizer / seed option present: greedy seeds from max total_confidence."""
    a = _mk_option("alt1", _pairs(list(range(3))), 0.5)
    b = _mk_option("alt2", _pairs(list(range(3, 6))), 0.9)
    c = _mk_option("alt3", _pairs(list(range(6, 9))), 0.4)
    ctx = _mk_ctx([a, b, c])
    group = {
        "edges": [
            {"ref_id": r, "target_id": t, "confidence": 0.5} for r, t in _pairs(list(range(9)))
        ]
    }

    info = prune_options_for_panel(ctx, group, max_options=1, min_distinct_edges_trigger=5)

    assert info == {"n_before": 3, "n_after": 1, "dropped_keys": ["alt1", "alt3"]}
    assert _edge_sets(ctx) == [frozenset(_pairs(list(range(3, 6))))]
    assert [o["letter"] for o in ctx["options"]] == ["A"]
    assert ctx["optimizer_letter"] is None


def test_generate_evidence_prunes_and_reletters_consistently(tmp_path, monkeypatch):
    """End-to-end: pruning at the metadata level keeps letters, images, prompt,
    and provenance consistent (vote parsing is metadata-letter driven)."""
    from crosswalk.config import settings

    monkeypatch.setattr(settings, "stitch_panel_max_options", 2)
    monkeypatch.setattr(settings, "stitch_panel_prune_min_distinct_edges", 2)

    g = make_group()
    # alt1 = near-duplicate of the optimizer (dropped); alt2 = distinct (kept,
    # re-lettered from C to B).
    g["alternatives"] = [
        {"edges": [{"ref_id": R1, "target_id": T1}]},
        {"edges": [{"ref_id": R1, "target_id": T2}]},
    ]
    d = tmp_path / g["group_id"]
    meta = generate_group_evidence(g, d)

    assert meta["options_pruned"] == {"n_before": 3, "n_after": 2, "dropped_keys": ["alt1"]}
    assert [o["letter"] for o in meta["options"]] == ["A", "B"]
    assert meta["optimizer_letter"] == "A"
    # The survivor re-lettered to B is the distinct alternative, not the near-dup.
    opt_b = next(o for o in meta["options"] if o["letter"] == "B")
    assert {(e["ref"], e["target"]) for e in opt_b["edges"]} == {("R1", "T2")}
    # Images exist exactly for surviving letters.
    assert (d / "option_A.png").exists()
    assert (d / "option_B.png").exists()
    assert not (d / "option_C.png").exists()
    # Prompt agrees with metadata: pruned letters only, pruned choice string.
    prompt = (d / "prompt.txt").read_text()
    assert "Option A (optimizer):" in prompt
    assert "Option B:" in prompt
    assert "Option C" not in prompt
    assert '"<A|B|NONE>"' in prompt
    # Runner-side letter -> edge-set mapping loads the pruned metadata cleanly.
    letters, options_by_letter, _ = sr._load_group_context(d)
    assert letters == ["A", "B"]
    assert set(options_by_letter["B"]) == {(R1, T2)}


def test_generate_evidence_below_threshold_records_no_pruning(tmp_path, monkeypatch):
    """Below-threshold groups: no options_pruned key, full option set intact."""
    from crosswalk.config import settings

    monkeypatch.setattr(settings, "stitch_panel_max_options", 8)
    monkeypatch.setattr(settings, "stitch_panel_prune_min_distinct_edges", 200)

    g = make_group()
    meta = generate_group_evidence(g, tmp_path / g["group_id"])
    assert "options_pruned" not in meta
    assert len(meta["options"]) == 2


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


def test_recover_labeled_groups_set_label_not_misread_as_empty():
    """A SET label (empty selected_edges but membership in ref_ids/target_ids)
    is a MATCH assertion: it must recover by membership overlap into the 'set'
    bucket and contribute its group to target_group_ids — never land in
    'empty' (reject-all/NONE)."""
    groups = [
        {
            "group_id": "gA",
            "edges": [{"ref_id": R1, "target_id": T1}, {"ref_id": R2, "target_id": T2}],
        },
    ]
    human_df = pd.DataFrame(
        [
            {
                "group_id": "h_set",
                "selected_edges": "[]",
                "label_semantics": "set",
                "ref_ids": json.dumps([R1, R2]),
                "target_ids": json.dumps([T1]),
            },
            {
                "group_id": "h_set_lost",
                "selected_edges": "[]",
                "label_semantics": "set",
                "ref_ids": json.dumps(["vanished_ref"]),
                "target_ids": json.dumps(["vanished_tgt"]),
            },
        ]
    )
    rec = recover_labeled_groups(groups, human_df)
    assert rec["empty"] == []  # neither set row is a reject-all
    assert ("h_set", "gA") in rec["set"]
    assert rec["set_lost"] == ["h_set_lost"]
    assert rec["target_group_ids"] == ["gA"]


def test_recover_empty_reject_all_skips_set_labels():
    """A SET label with a verbatim-surviving group_id must NOT be recovered as
    a reject-all — it asserts a match, not 'no edges'."""
    groups = [{"group_id": "gA", "edges": [{"ref_id": R1, "target_id": T1}]}]
    human_df = pd.DataFrame(
        [
            {
                "group_id": "gA",
                "selected_edges": "[]",
                "label_semantics": "set",
                "ref_ids": json.dumps([R1]),
                "target_ids": json.dumps([T1]),
            },
        ]
    )
    rec = recover_empty_reject_all(groups, human_df)
    assert rec["recovered"] == []
    assert rec["unrecoverable"] == []


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


def test_evaluate_batch_sliver_filtered(tmp_path):
    """Panel includes a junction sliver the human omitted.

    Raw comparison disagrees; the sliver-filtered comparison (slivers removed
    from BOTH sides using batch.json geometries) agrees.
    """
    import math

    def _line_len(length_m, lat=42.36, lon=-71.06):
        deg = length_m / (111000.0 * math.cos(math.radians(lat)))
        return {"type": "LineString", "coordinates": [[lon, lat], [lon + deg, lat]]}

    group = {
        "group_id": "grp001",
        "match_type": "M:N",
        "ref_ids": [R1],
        "target_ids": [T1, T2],
        "edges": [
            {
                "ref_id": R1,
                "target_id": T1,
                "confidence": 0.9,
                "gers_start_frac": 0.0,
                "gers_end_frac": 1.0,
                "local_start_frac": 0.0,
                "local_end_frac": 1.0,
            },
            {  # junction sliver: 1 m of a 50 m ref, 0.5 m of a 10 m target
                "ref_id": R1,
                "target_id": T2,
                "confidence": 0.3,
                "gers_start_frac": 0.0,
                "gers_end_frac": 0.02,
                "local_start_frac": 0.0,
                "local_end_frac": 0.05,
            },
        ],
        "optimizer_assignment": [
            {"ref_id": R1, "target_id": T1},
            {"ref_id": R1, "target_id": T2},
        ],
        "alternatives": [],
        "ref_geometries": {R1: _line_len(50.0)},
        "target_geometries": {T1: _line_len(50.0), T2: _line_len(10.0)},
        "ref_names": {R1: "Main St"},
        "target_names": {T1: "Main", T2: "Side"},
        "ref_classes": {R1: "residential"},
        "target_classes": {T1: "residential", T2: "residential"},
    }

    batch_dir = tmp_path / "batch"
    generate_group_evidence(group, batch_dir / group["group_id"])
    (batch_dir / "batch.json").write_text(json.dumps({"groups": [group]}))

    # Human omitted the sliver edge; panel included it.
    human_es = [{"ref_id": R1, "target_id": T1}]
    human_df = pd.DataFrame([{"group_id": "hg", "selected_edges": json.dumps(human_es)}])
    panel_str = json.dumps(sorted([[R1, T1], [R1, T2]]))
    pd.DataFrame(
        [
            {
                "group_id": "grp001",
                "consensus": "unanimous",
                "choice": "A",
                "edge_set": panel_str,
                "routing": "auto_accept",
                "n_votes": 1,
                "n_valid": 1,
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
    r = results[0]
    assert r.exact_match is False  # raw: panel has the sliver, human doesn't
    assert r.exact_match_filtered is True  # filtered: sliver removed from both
    assert r.f1_filtered == 1.0

    summary = summarize(results)
    assert summary["panel_exact_rate"] == 0.0
    assert summary["panel_exact_rate_filtered"] == 1.0
    assert summary["n_groups_sliver_affected"] == 1


# ---------------------------------------------------------------------------
# Human-label -> group mapping (edge-level overlap preferred)
# ---------------------------------------------------------------------------


def _mapping_fixtures():
    from crosswalk.agent_labeling.stitch_evidence import build_metadata

    g = make_group()
    ctx = build_stitch_options(g)
    metas = {g["group_id"]: build_metadata(g, ctx)}
    cand = {g["group_id"]: frozenset({(R1, T1), (R1, T2), (R2, T2)})}
    return metas, cand


def test_mapping_requires_edge_overlap_when_candidates_known():
    """A label whose segments are in the group but whose edges never existed
    as candidate edges must NOT map (would skew coverage/agreement metrics)."""
    from crosswalk.agent_labeling.stitch_eval import map_human_labels_to_groups

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
    from crosswalk.agent_labeling.stitch_eval import map_human_labels_to_groups

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
    from crosswalk.agent_labeling.stitch_eval import map_human_labels_to_groups

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


# ---------------------------------------------------------------------------
# Diagnostic pack-feedback instrumentation (wave-local flag; default OFF)
# ---------------------------------------------------------------------------


def test_pack_feedback_column_present_in_votes_schema():
    # The votes.csv schema must carry the diagnostic self-report column.
    assert "pack_feedback" in sr.VOTES_COLUMNS


def test_augment_prompt_only_adds_feedback_request():
    base = 'respond with {"choice": "A", ...}'
    aug = sr.augment_prompt_with_feedback(base)
    assert aug.startswith(base)
    assert "pack_feedback" in aug
    assert "missing_info" in aug and "ambiguities" in aug and "confidence_basis" in aug
    # Default prompt is untouched by the augmentation helper.
    assert "pack_feedback" not in base


def test_extract_pack_feedback_pulls_nested_object():
    raw = json.dumps(
        {
            "choice": "A",
            "confidence": 0.8,
            "reasoning": "clear",
            "pack_feedback": {
                "missing_info": ["no name on T1"],
                "ambiguities": [],
                "confidence_basis": "parallel geometry",
            },
        }
    )
    fb = sr._extract_pack_feedback(raw)
    parsed = json.loads(fb)
    assert parsed["missing_info"] == ["no name on T1"]
    assert parsed["confidence_basis"] == "parallel geometry"


def test_extract_pack_feedback_absent_returns_empty():
    raw = '{"choice": "A", "confidence": 0.8, "reasoning": "clear"}'
    assert sr._extract_pack_feedback(raw) == ""
    assert sr._extract_pack_feedback("total garbage no json") == ""


def test_extract_pack_feedback_empty_object_returns_empty():
    raw = '{"choice": "A", "confidence": 0.8, "reasoning": "x", "pack_feedback": {}}'
    assert sr._extract_pack_feedback(raw) == ""


def test_parse_vote_ignores_pack_feedback_key():
    # A vote carrying the diagnostic key still parses to a valid choice.
    raw = json.dumps(
        {
            "choice": "B",
            "confidence": 0.7,
            "reasoning": "ok",
            "pack_feedback": {"ambiguities": ["z"]},
        }
    )
    choice, conf, _ = sr.parse_vote(raw, {"A", "B"})
    assert choice == "B" and conf == 0.7


# ---------------------------------------------------------------------------
# Fourth voter: opencode invoker + named panel config (default OFF)
# ---------------------------------------------------------------------------


def test_opencode_registered_and_v3_panel_composition():
    """The 4th voter is wired into the invoker registry and the opt-in panel."""
    assert "opencode" in sr._INVOKERS
    # DEFAULT_PANEL (production) is unchanged: still the 3 incumbents.
    assert [p.name for p in sr.DEFAULT_PANEL] == ["claude", "codex", "agy"]
    # v3-candidate adds opencode as a distinct 4th, decorrelated model family.
    v3 = sr.get_panel("v3-candidate")
    assert [p.name for p in v3] == ["claude", "codex", "agy", "opencode"]
    assert v3[3].model == "openrouter/qwen/qwen3-vl-235b-a22b-instruct"
    # Named-panel resolution: default/v2 == 3 voters; unknown/empty -> default.
    assert [p.name for p in sr.get_panel("default")] == ["claude", "codex", "agy"]
    assert [p.name for p in sr.get_panel("v2")] == ["claude", "codex", "agy"]
    assert sr.get_panel("nonexistent") is sr.DEFAULT_PANEL
    assert sr.get_panel(None) is sr.DEFAULT_PANEL


def test_invoke_opencode_arg_construction(monkeypatch, tmp_path):
    """Prompt goes via stdin (not argv); -m model and one -f per attached image.

    The prompt used to be a positional arg, but a large group's prompt exceeds
    the OS single-arg limit (E2BIG), so it is now piped via stdin.
    """
    import subprocess as sp

    gdir = tmp_path / "grp"
    gdir.mkdir()
    (gdir / "overview.png").write_bytes(b"\x89PNG")
    (gdir / "option_A.png").write_bytes(b"\x89PNG")
    (gdir / "option_B.png").write_bytes(b"\x89PNG")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return sp.CompletedProcess(
            cmd, 0, stdout='{"choice":"A","confidence":1,"reasoning":"x"}', stderr=""
        )

    monkeypatch.setattr(sr.subprocess, "run", fake_run)
    out = sr.invoke_opencode("PROMPT TEXT", gdir, ["A", "B"], "openrouter/qwen/x", timeout=99)

    cmd = captured["cmd"]
    assert cmd[:2] == ["opencode", "run"]
    # The prompt is piped via stdin, never placed on argv.
    assert captured["kwargs"]["input"] == "PROMPT TEXT"
    assert "PROMPT TEXT" not in cmd
    # -m model appears.
    assert "-m" in cmd and cmd[cmd.index("-m") + 1] == "openrouter/qwen/x"
    # One -f per image: overview + option_A + option_B = 3.
    assert cmd.count("-f") == 3
    f_args = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-f"]
    assert f_args[0].endswith("overview.png")
    assert any(a.endswith("option_A.png") for a in f_args)
    assert any(a.endswith("option_B.png") for a in f_args)
    assert captured["kwargs"]["timeout"] == 99
    assert '"choice":"A"' in out


def test_invoke_opencode_raises_on_nonzero_exit(monkeypatch, tmp_path):
    import subprocess as sp

    gdir = tmp_path / "grp"
    gdir.mkdir()
    (gdir / "overview.png").write_bytes(b"\x89PNG")

    def fake_run(cmd, **kwargs):
        return sp.CompletedProcess(cmd, 1, stdout="", stderr="quota exceeded")

    monkeypatch.setattr(sr.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as exc:
        sr.invoke_opencode("P", gdir, [], "m")
    assert "opencode exited with code 1" in str(exc.value)
    assert "quota exceeded" in str(exc.value)


def test_opencode_vote_parses_through_runner(monkeypatch):
    """A valid opencode-style JSON answer parses to a real vote."""

    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        return '{"choice": "B", "confidence": 0.66, "reasoning": "qwen picks B"}'

    monkeypatch.setitem(sr._INVOKERS, "opencode", fake_invoker)
    vote = sr.run_provider_on_group(
        sr.OPENCODE_QWEN, "g", None, "prompt", ["A", "B"], {"B": [(R1, T1)]}
    )
    assert vote.provider == "opencode"
    assert vote.choice == "B"
    assert vote.edge_set == frozenset({(R1, T1)})


def test_opencode_hard_fails_on_invocation_error(monkeypatch):
    """opencode quota exhaustion halts the run (was: abstain). budget=0 = immediate."""

    def failing(prompt, group_dir, letters, model, timeout, effort=""):
        raise RuntimeError("opencode exited with code 1: quota exceeded")

    monkeypatch.setitem(sr._INVOKERS, "opencode", failing)
    with pytest.raises(sr.ProviderInvocationError, match="quota exceeded"):
        sr.run_provider_on_group(
            sr.OPENCODE_QWEN,
            "g",
            None,
            "prompt",
            ["A"],
            {"A": [(R1, T1)]},
            invocation_budget_s=0.0,
        )


def test_four_voter_consensus_unanimous_needs_all_four():
    """With a 4-voter panel, unanimity requires all four agreeing (3/4 is majority)."""
    es = frozenset({(R1, T1)})
    four_agree = [
        _vote("claude", "A", es),
        _vote("codex", "A", es),
        _vote("agy", "A", es),
        _vote("opencode", "A", es),
    ]
    c = sr.compute_consensus(four_agree)
    assert c.consensus == "unanimous"
    assert c.routing == "auto_accept"
    # 3/4 with a lone dissenter is a majority -> human review (never auto-accept).
    three_one = [
        _vote("claude", "A", es),
        _vote("codex", "A", es),
        _vote("agy", "A", es),
        _vote("opencode", "B"),
    ]
    c2 = sr.compute_consensus(three_one)
    assert c2.consensus == "majority"
    assert c2.routing == "human_review"
    assert "opencode=B" in c2.minority
    assert c2.route_reason == "dissent:opencode=B"
    # 3 agree + 1 abstain: quorum met (>=3 valid) but the abstention blocked
    # full unanimity — stamped "abstention", not below_quorum.
    three_abstain = [
        _vote("claude", "A", es),
        _vote("codex", "A", es),
        _vote("agy", "A", es),
        _vote("opencode", "ABSTAIN"),
    ]
    c3 = sr.compute_consensus(three_abstain)
    assert c3.consensus == "majority"
    assert c3.routing == "human_review"
    assert c3.route_reason == "abstention"


# ---------------------------------------------------------------------------
# Resumable per-group batch driver
# ---------------------------------------------------------------------------


def _write_min_pack(batch_dir, gid):
    """Write a minimal evidence pack (metadata + prompt) for run_batch."""
    import yaml

    g = make_group()
    ctx = build_stitch_options(g)
    meta = build_metadata(g, ctx)
    d = batch_dir / gid
    d.mkdir(parents=True)
    (d / "metadata.yaml").write_text(yaml.safe_dump(meta))
    (d / "prompt.txt").write_text('respond {"choice": "A"}')
    return d


def test_run_batch_resume_skips_completed_groups(tmp_path, monkeypatch):
    """resume=True reloads partials and does NOT re-invoke completed groups."""
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    _write_min_pack(batch_dir, "g1")
    _write_min_pack(batch_dir, "g2")

    # Each group runs in an isolated scratch copy, so the invoker sees a scratch
    # dir, not the group name — count invocations (3 providers per group) instead.
    calls = {"n": 0}

    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        calls["n"] += 1
        return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'

    for name in ("claude", "codex", "agy"):
        monkeypatch.setitem(sr._INVOKERS, name, fake_invoker)

    panel = [
        sr.ProviderSpec("claude", "m"),
        sr.ProviderSpec("codex", "m"),
        sr.ProviderSpec("agy", "m"),
    ]

    # First run does g1 only, writing partials (3 invocations: one per provider).
    sr.run_batch(batch_dir, panel=panel, group_ids=["g1"])
    assert (batch_dir / "votes.partial.csv").exists()
    assert calls["n"] == 3

    calls["n"] = 0
    # Resume over both groups: g1 already recorded -> only g2 runs (3 more calls).
    votes_df, cons_df = sr.run_batch(batch_dir, panel=panel, resume=True)
    assert calls["n"] == 3  # g1 skipped; only g2 invoked
    # Final output carries BOTH groups.
    assert set(cons_df["group_id"]) == {"g1", "g2"}
    assert set(votes_df["group_id"]) == {"g1", "g2"}


def _breaker_panel_and_invokers(monkeypatch, failing_invoker):
    """3-provider panel where claude/codex vote A and agy uses failing_invoker."""

    def ok_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'

    monkeypatch.setitem(sr._INVOKERS, "claude", ok_invoker)
    monkeypatch.setitem(sr._INVOKERS, "codex", ok_invoker)
    monkeypatch.setitem(sr._INVOKERS, "agy", failing_invoker)
    return [
        sr.ProviderSpec("claude", "m"),
        sr.ProviderSpec("codex", "m"),
        sr.ProviderSpec("agy", "m"),
    ]


def test_run_batch_timeout_breaker_halts_after_consecutive(tmp_path, monkeypatch):
    """N consecutive timeout-abstains from ONE provider promote to provider-down halt.

    Completed groups must be flushed to the partials before the raise so the run
    stays resumable.
    """
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    for gid in ("g1", "g2", "g3", "g4"):
        _write_min_pack(batch_dir, gid)

    def always_timeout(prompt, group_dir, letters, model, timeout, effort=""):
        raise subprocess.TimeoutExpired(cmd="agy", timeout=timeout)

    panel = _breaker_panel_and_invokers(monkeypatch, always_timeout)
    with pytest.raises(sr.ProviderInvocationError, match="consecutive"):
        sr.run_batch(batch_dir, panel=panel)
    # Groups before the breaker group were flushed and are resumable.
    votes = pd.read_csv(batch_dir / "votes.partial.csv", dtype={"group_id": str})
    assert set(votes["group_id"]) == {"g1", "g2"}


def test_run_batch_timeout_breaker_resets_on_success(tmp_path, monkeypatch):
    """A successful vote resets the consecutive-timeout count — no false halt."""
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    for gid in ("g1", "g2", "g3", "g4", "g5"):
        _write_min_pack(batch_dir, gid)

    calls = {"n": 0}

    def intermittent(prompt, group_dir, letters, model, timeout, effort=""):
        calls["n"] += 1
        if calls["n"] == 3:  # succeed on the 3rd group only
            return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'
        raise subprocess.TimeoutExpired(cmd="agy", timeout=timeout)

    panel = _breaker_panel_and_invokers(monkeypatch, intermittent)
    votes_df, cons_df = sr.run_batch(batch_dir, panel=panel)  # must NOT raise
    assert set(cons_df["group_id"]) == {"g1", "g2", "g3", "g4", "g5"}
    agy = votes_df[votes_df["provider"] == "agy"]
    assert (agy["choice"] == "ABSTAIN").sum() == 4


def test_run_batch_overflow_abstains_do_not_trip_breaker(tmp_path, monkeypatch):
    """Context overflow is a group property: monsters in a row never halt the run."""
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    for gid in ("g1", "g2", "g3", "g4"):
        _write_min_pack(batch_dir, gid)

    def always_overflow(prompt, group_dir, letters, model, timeout, effort=""):
        raise sr.GroupScopedProviderError(
            "agy context overflow: prompt exceeds the model's context window"
        )

    panel = _breaker_panel_and_invokers(monkeypatch, always_overflow)
    votes_df, cons_df = sr.run_batch(batch_dir, panel=panel)  # must NOT raise
    assert set(cons_df["group_id"]) == {"g1", "g2", "g3", "g4"}
    agy = votes_df[votes_df["provider"] == "agy"]
    assert (agy["choice"] == "ABSTAIN").all()


def test_quota_and_rate_limit_messages_not_misclassified_as_group_scoped():
    """Marker-set breadth guard: provider-down bodies must stay provider-down.

    Broadening _CONTEXT_OVERFLOW_MARKERS to something colliding with a realistic
    quota/rate-limit message would silently convert #334 halts into abstains.
    """
    provider_down_bodies = [
        "429 rate limit reached for gpt-5.5: limit 30000 tokens per min",
        "insufficient_quota: you exceeded your current quota",
        "401 unauthorized: invalid api key",
        "upstream connect error or disconnect/reset before headers",
        "billing hard limit has been reached",
    ]
    for body in provider_down_bodies:
        result = subprocess.CompletedProcess(args=["codex"], returncode=1, stdout="", stderr=body)
        with pytest.raises(RuntimeError) as exc_info:
            sr._check_exit("codex", result)
        assert not isinstance(exc_info.value, sr.GroupScopedProviderError), body


def test_run_batch_resume_rejects_mismatched_panel(tmp_path, monkeypatch):
    """Partials written by a DIFFERENT panel are ignored: every group re-runs.

    Guards the silent-cache trap: 3-voter partials satisfying a --panel
    v3-candidate --resume run would return cached 3-voter votes with the 4th
    voter never invoked.
    """
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    _write_min_pack(batch_dir, "g1")

    calls = {"n": 0, "providers": set()}

    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        calls["n"] += 1
        return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'

    for name in ("claude", "codex", "agy", "opencode"):
        monkeypatch.setitem(sr._INVOKERS, name, fake_invoker)

    panel3 = [
        sr.ProviderSpec("claude", "m"),
        sr.ProviderSpec("codex", "m"),
        sr.ProviderSpec("agy", "m"),
    ]
    panel4 = [*panel3, sr.ProviderSpec("opencode", "m")]

    # Complete a 3-voter run: partials now record every group with 3 providers.
    sr.run_batch(batch_dir, panel=panel3)
    assert calls["n"] == 3

    calls["n"] = 0
    # Resume with the 4-voter panel: provider-set mismatch -> partials ignored,
    # g1 re-runs with all FOUR voters (not returned from the 3-voter cache).
    votes_df, _cons_df = sr.run_batch(batch_dir, panel=panel4, resume=True)
    assert calls["n"] == 4
    assert set(votes_df["provider"]) == {"claude", "codex", "agy", "opencode"}


def test_run_batch_resume_respects_group_selection(tmp_path, monkeypatch):
    """A filtered resume must not leak previously-done, unrequested groups."""
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    _write_min_pack(batch_dir, "g1")
    _write_min_pack(batch_dir, "g2")

    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'

    for name in ("claude", "codex", "agy"):
        monkeypatch.setitem(sr._INVOKERS, name, fake_invoker)

    panel = [
        sr.ProviderSpec("claude", "m"),
        sr.ProviderSpec("codex", "m"),
        sr.ProviderSpec("agy", "m"),
    ]

    # Run everything once (partials record g1 + g2).
    sr.run_batch(batch_dir, panel=panel)
    # Filtered resume asking ONLY for g2: g1 must not appear in the output.
    votes_df, cons_df = sr.run_batch(batch_dir, panel=panel, group_ids=["g2"], resume=True)
    assert set(votes_df["group_id"]) == {"g2"}
    assert set(cons_df["group_id"]) == {"g2"}


def test_run_provider_collect_feedback_toggle(monkeypatch):
    """collect_feedback=True captures the self-report; False leaves it empty."""
    raw = json.dumps(
        {
            "choice": "A",
            "confidence": 0.9,
            "reasoning": "ok",
            "pack_feedback": {"missing_info": ["x"], "ambiguities": [], "confidence_basis": "y"},
        }
    )

    def fake_invoker(prompt, group_dir, letters, model, timeout, effort=""):
        return raw

    monkeypatch.setitem(sr._INVOKERS, "claude", fake_invoker)
    args = (sr.ProviderSpec("claude", "m"), "g", None, "prompt", ["A"], {"A": [(R1, T1)]})

    vote_off = sr.run_provider_on_group(*args)
    assert vote_off.choice == "A"
    assert vote_off.pack_feedback == ""

    vote_on = sr.run_provider_on_group(*args, collect_feedback=True)
    assert vote_on.choice == "A"
    assert json.loads(vote_on.pack_feedback)["missing_info"] == ["x"]


# --- invocation hard-fail vs parse abstain (quota handling) -------------------


def _attempt(invoker, retries=1, budget=300.0):
    """Drive _attempt_provider with a fake invoker, group_dir=None (unit mode)."""
    return sr._attempt_provider(
        sr.ProviderSpec(name="claude", model="test-model"),
        "g1",
        None,  # group_dir -> skip scratch pack
        "prompt",
        ["A"],
        {"A": [("r1", "t1")]},
        {"A"},
        invoker,
        5,  # timeout
        retries,
        invocation_budget_s=budget,
    )


class _FakeClock:
    """Deterministic monotonic clock: only sleep() advances time.

    Lets the backoff/deadline logic run through many iterations with zero real
    wall-clock, so budget exhaustion and exponential backoff are actually
    exercised — a no-op sleep would freeze ``remaining`` and never exhaust the
    budget, leaving the core mechanism untested.
    """

    def __init__(self):
        self.t = 0.0
        self.sleeps: list[float] = []

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def _install_fake_clock(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(sr.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(sr.time, "sleep", clock.sleep)
    return clock


def test_persistent_invocation_error_hard_fails(monkeypatch):
    """A provider that keeps failing (e.g. quota) raises, not abstains."""
    _install_fake_clock(monkeypatch)

    def always_fail(*_a, **_k):
        raise RuntimeError("opencode exited with code 1: 429 insufficient_quota")

    with pytest.raises(sr.ProviderInvocationError, match="insufficient_quota"):
        _attempt(always_fail, budget=0.0)


def test_budget_exhaustion_backs_off_exponentially(monkeypatch):
    """Persistent failure retries with doubling backoff until the budget is spent."""
    clock = _install_fake_clock(monkeypatch)

    def always_fail(*_a, **_k):
        raise RuntimeError("exited with code 1: network unreachable")

    with pytest.raises(sr.ProviderInvocationError):
        _attempt(always_fail, budget=300.0)
    # Exponential backoff (5,10,20,40,...), capped at 60, final sleep clamped to
    # the remaining budget; the clock only advances via sleep, so the sleeps sum
    # to exactly the budget.
    assert clock.sleeps[:4] == [5.0, 10.0, 20.0, 40.0]
    assert max(clock.sleeps) == 60.0
    assert abs(sum(clock.sleeps) - 300.0) < 1e-6


def test_timeout_abstains_immediately_without_retry(monkeypatch):
    """A timeout is group-scoped: abstain on ONE attempt (no backoff), run continues.

    Retrying a timeout just burns another full timeout window, and halting lets
    one oversized group kill a whole wave; a hung *provider* still halts via the
    generic nonzero-exit path.
    """
    clock = _install_fake_clock(monkeypatch)
    calls = {"n": 0}

    def always_timeout(*_a, **_k):
        calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd="claude", timeout=5)

    vote = _attempt(always_timeout, budget=300.0)
    assert vote.choice == "ABSTAIN"
    assert "timeout after" in vote.error
    assert calls["n"] == 1  # single attempt, no retry
    assert clock.sleeps == []  # and no backoff sleeps


def test_context_overflow_abstains_and_continues(monkeypatch):
    """A context-window overflow is deterministic per group -> abstain, not halt."""
    clock = _install_fake_clock(monkeypatch)
    calls = {"n": 0}

    def overflowing(*_a, **_k):
        calls["n"] += 1
        raise sr.GroupScopedProviderError(
            "codex context overflow: prompt exceeds the model's context window"
        )

    vote = _attempt(overflowing, budget=300.0)
    assert vote.choice == "ABSTAIN"
    assert "context overflow" in vote.error
    assert calls["n"] == 1
    assert clock.sleeps == []


def test_check_exit_classifies_context_overflow_beyond_snippet():
    """Overflow markers are scanned in the FULL output, not the 500-char snippet.

    codex prints a long banner + echoed prompt to stderr before its overflow
    line; truncating first would misclassify the overflow as provider-down.
    """
    banner = "OpenAI Codex v0.142.5\n" + ("x" * 600) + "\n"
    result = subprocess.CompletedProcess(
        args=["codex"],
        returncode=1,
        stdout="",
        stderr=banner + "ERROR: Codex ran out of room in the model's context window.",
    )
    with pytest.raises(sr.GroupScopedProviderError, match="context overflow"):
        sr._check_exit("codex", result)


def test_check_exit_classifies_cli_internal_timeout():
    """agy's own 'timeout waiting for response' is group-scoped, not provider-down."""
    result = subprocess.CompletedProcess(
        args=["agy"], returncode=1, stdout="", stderr="Error: timeout waiting for response"
    )
    with pytest.raises(sr.GroupScopedProviderError, match="response timeout"):
        sr._check_exit("agy", result)


def test_check_exit_generic_failure_still_runtime_error():
    """Non-group-scoped nonzero exits keep the provider-down classification."""
    result = subprocess.CompletedProcess(
        args=["codex"], returncode=1, stdout="", stderr="401 unauthorized"
    )
    with pytest.raises(RuntimeError) as exc_info:
        sr._check_exit("codex", result)
    assert not isinstance(exc_info.value, sr.GroupScopedProviderError)


def test_invoke_agy_print_timeout_derived_from_timeout(monkeypatch, tmp_path):
    """agy's internal response timer follows the caller's timeout (was: 2m hardcoded)."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(sr.subprocess, "run", fake_run)
    sr.invoke_agy("PROMPT", tmp_path, ["A"], "Gemini 3.5 Flash (Medium)", timeout=240)
    assert "--print-timeout=225s" in captured["cmd"]
    # Floor: never below 30s even with a tiny caller timeout.
    sr.invoke_agy("PROMPT", tmp_path, ["A"], "Gemini 3.5 Flash (Medium)", timeout=20)
    assert "--print-timeout=30s" in captured["cmd"]


def test_invocation_error_recovers_within_budget(monkeypatch):
    """A transient failure that clears on retry yields a normal vote (no raise)."""
    _install_fake_clock(monkeypatch)
    calls = {"n": 0}

    def flaky(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("temporary 429 rate limit")
        return '{"choice": "A", "confidence": 0.9, "reasoning": "ok"}'

    vote = _attempt(flaky, budget=60.0)
    assert vote.choice == "A" and vote.provider == "claude"
    assert calls["n"] == 2  # failed once, succeeded on retry


def test_unexpected_exception_propagates_not_retried(monkeypatch):
    """A programming bug in an invoker must fail fast, not masquerade as quota."""
    _install_fake_clock(monkeypatch)
    calls = {"n": 0}

    def buggy(*_a, **_k):
        calls["n"] += 1
        raise AttributeError("'NoneType' object has no attribute 'foo'")

    with pytest.raises(AttributeError):
        _attempt(buggy, budget=300.0)
    assert calls["n"] == 1  # not caught, not retried across the budget


def test_parse_error_still_abstains_not_hard_fail():
    """Malformed output abstains (soft) — it must NOT halt the run."""

    def garbage(*_a, **_k):
        return "no json here, just prose"

    vote = _attempt(garbage, retries=1)
    assert vote.choice == "ABSTAIN"
    assert "parse/validation" in vote.error


def test_hard_fail_propagates_through_run_provider_on_group(monkeypatch):
    """run_provider_on_group must not swallow ProviderInvocationError."""
    _install_fake_clock(monkeypatch)

    def always_fail(*_a, **_k):
        raise RuntimeError("exited with code 1: quota exceeded")

    monkeypatch.setitem(sr._INVOKERS, "claude", always_fail)
    with pytest.raises(sr.ProviderInvocationError):
        sr.run_provider_on_group(
            sr.ProviderSpec(name="claude", model="m"),
            "g1",
            None,  # group_dir None -> skip scratch pack
            "prompt",
            ["A"],
            {"A": [("r1", "t1")]},
            invocation_budget_s=0.0,
        )


def test_deterministic_oserror_fast_fails_without_backoff(monkeypatch):
    """An E2BIG-class OSError hard-fails immediately (no wasted backoff budget)."""
    clock = _install_fake_clock(monkeypatch)

    def arg_too_long(*_a, **_k):
        raise OSError(errno.E2BIG, "Argument list too long")

    with pytest.raises(sr.ProviderInvocationError, match="not retryable"):
        _attempt(arg_too_long, budget=300.0)
    assert clock.sleeps == []  # failed on the first attempt, never slept


def test_missing_binary_oserror_fast_fails(monkeypatch):
    """A missing-CLI (ENOENT) failure is deterministic -> immediate hard-fail."""
    _install_fake_clock(monkeypatch)

    def no_binary(*_a, **_k):
        raise FileNotFoundError(errno.ENOENT, "No such file or directory")

    with pytest.raises(sr.ProviderInvocationError, match="not retryable"):
        _attempt(no_binary, budget=300.0)


def test_transient_oserror_still_backs_off(monkeypatch):
    """A non-fatal OSError (e.g. ECONNREFUSED) keeps the backoff-then-hardfail path."""
    clock = _install_fake_clock(monkeypatch)

    def conn_refused(*_a, **_k):
        raise ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused")

    with pytest.raises(sr.ProviderInvocationError):
        _attempt(conn_refused, budget=300.0)
    assert clock.sleeps  # backed off (not a fatal errno), then gave up on budget


# --- invoker transport contract: prompt must NOT go on argv (E2BIG regression) ---


def _capture_subprocess_run(monkeypatch):
    """Patch subprocess.run to capture cmd/input/cwd and return a clean result."""
    cap = {}

    def fake_run(cmd, input=None, **kw):
        cap["cmd"] = list(cmd)
        cap["input"] = input
        cap["cwd"] = kw.get("cwd")
        return subprocess.CompletedProcess(cmd, 0, stdout='{"choice": "A"}', stderr="")

    monkeypatch.setattr(sr.subprocess, "run", fake_run)
    return cap


# A prompt larger than the Linux single-arg limit (MAX_ARG_STRLEN ~128 KB): if any
# invoker still put this on argv, a real exec would raise E2BIG.
_BIG_PROMPT = "X" * 200_000


def test_invoke_codex_prompt_via_stdin_not_argv(tmp_path, monkeypatch):
    cap = _capture_subprocess_run(monkeypatch)
    sr.invoke_codex(_BIG_PROMPT, tmp_path, ["A"], "gpt-5.5")
    assert cap["input"] == _BIG_PROMPT  # prompt piped on stdin
    assert _BIG_PROMPT not in cap["cmd"]  # never an argv element
    assert "-" in cap["cmd"]  # stdin sentinel present


def test_invoke_opencode_prompt_via_stdin_not_argv(tmp_path, monkeypatch):
    cap = _capture_subprocess_run(monkeypatch)
    sr.invoke_opencode(_BIG_PROMPT, tmp_path, ["A"], "openrouter/qwen/x")
    assert cap["input"] == _BIG_PROMPT
    assert _BIG_PROMPT not in cap["cmd"]
    assert cap["cmd"][:2] == ["opencode", "run"]


def test_invoke_agy_prompt_via_file_not_argv(tmp_path, monkeypatch):
    cap = _capture_subprocess_run(monkeypatch)
    sr.invoke_agy(_BIG_PROMPT, tmp_path, ["A"], "Gemini 3.5 Flash (Medium)")
    assert _BIG_PROMPT not in cap["cmd"]  # not on argv
    assert cap["cwd"] == str(tmp_path)  # agy runs in the group dir
    pf = tmp_path / "panel_prompt.txt"
    assert pf.exists() and pf.read_text() == _BIG_PROMPT  # prompt written to file
    assert "--dangerously-skip-permissions" in cap["cmd"]
