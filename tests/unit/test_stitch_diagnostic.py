"""Isolation, resume, provenance, and analysis tests for Codex diagnostics."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from crosswalk.agent_labeling import stitch_diagnostic as diagnostic
from crosswalk.agent_labeling.stitch_runner import Vote, get_panel, panel_descriptor
from crosswalk.agent_labeling.wave_manifest import WaveManifest


def _evidence(group_id: str) -> dict:
    return {
        "schema_version": 1,
        "group_id": group_id,
        "evidence_id": f"evidence-{group_id}",
        "evidence_pack_sha256": f"pack-{group_id}",
        "evidence": {
            "schema_version": 1,
            "group_id": group_id,
            "evidence_id": f"evidence-{group_id}",
            "evidence_pack_sha256": f"pack-{group_id}",
            "displayed_candidate_universe_sha256": f"universe-{group_id}",
            "option_menu_sha256": f"menu-{group_id}",
            "selectable_choices": ["A", "B", "NONE"],
            "displayed_edges": [
                {"ref_id": "ref-1", "target_id": "target-1"},
                {"ref_id": "ref-1", "target_id": "target-2"},
            ],
            "option_menu": [
                {
                    "letter": "A",
                    "option_id": "option-a",
                    "edges": [{"ref_id": "ref-1", "target_id": "target-1"}],
                },
                {
                    "letter": "B",
                    "option_id": "option-b",
                    "edges": [{"ref_id": "ref-1", "target_id": "target-2"}],
                },
            ],
        },
    }


def _write_group(batch_dir: Path, group_id: str, dataset: str, variant: str) -> None:
    group_dir = batch_dir / group_id
    group_dir.mkdir(parents=True)
    (group_dir / "prompt.txt").write_text(f"Vote on {dataset} {group_id} {variant}\n")
    (group_dir / "metadata.yaml").write_text(
        f"""group_id: {group_id}
segments:
  reference:
    - {{label: R1, id: ref-1}}
  target:
    - {{label: T1, id: target-1}}
    - {{label: T2, id: target-2}}
options: []
"""
    )
    (group_dir / "evidence.json").write_text("{}\n")


def _wave(tmp_path: Path, *, ordinary_groups: int = 4) -> tuple[Path, list[dict]]:
    schedule = []
    run_index = 1
    # Ordinary groups split across two datasets.
    for index in range(ordinary_groups):
        dataset = "ds_a" if index % 2 == 0 else "ds_b"
        group_id = f"g{index}"
        batch_dir = tmp_path / f"batch_{run_index}"
        _write_group(batch_dir, group_id, dataset, "enriched")
        (batch_dir / "batch.json").write_text(
            json.dumps(
                {
                    "dataset_id": dataset,
                    "experiment": {"variant": "enriched"},
                    "groups": [{"group_id": group_id}],
                }
            )
        )
        schedule.append(
            {
                "run_index": run_index,
                "dataset_id": dataset,
                "group_id": group_id,
                "variant": "enriched",
                "batch_dir": str(batch_dir),
            }
        )
        run_index += 1

    # One factorial group with two variants in distinct immutable batch dirs.
    for variant in ("enriched", "minimal"):
        batch_dir = tmp_path / f"batch_{run_index}"
        _write_group(batch_dir, "factorial", "ds_a", variant)
        (batch_dir / "batch.json").write_text(
            json.dumps(
                {
                    "dataset_id": "ds_a",
                    "experiment": {"variant": variant},
                    "groups": [{"group_id": "factorial"}],
                }
            )
        )
        schedule.append(
            {
                "run_index": run_index,
                "dataset_id": "ds_a",
                "group_id": "factorial",
                "variant": variant,
                "batch_dir": str(batch_dir),
            }
        )
        run_index += 1

    manifest_path = tmp_path / "wave.json"
    WaveManifest(
        {
            "wave": "test-wave",
            "panel": "v7-candidate",
            "required_panel": panel_descriptor(get_panel("v7-candidate")),
            "total_pack_count": len(schedule),
            "batch_dirs": sorted({row["batch_dir"] for row in schedule}),
            "run_schedule": schedule,
        }
    ).write(manifest_path)
    return manifest_path, schedule


@pytest.fixture
def fake_evidence(monkeypatch: pytest.MonkeyPatch):
    def load(group_dir: Path, *, allow_legacy: bool = True):
        return _evidence(Path(group_dir).name)

    monkeypatch.setattr(diagnostic, "load_evidence_manifest", load)
    monkeypatch.setattr(diagnostic, "validate_manifest_against_batch", lambda *args: None)
    return load


def _plan(
    tmp_path: Path,
    fake_evidence,
    *,
    ordinary_groups: int = 4,
    canonical_draws: int = 3,
) -> tuple[Path, dict, list[dict]]:
    manifest_path, schedule = _wave(tmp_path, ordinary_groups=ordinary_groups)
    plan = diagnostic.build_diagnostic_plan(
        manifest_path,
        tmp_path / "diagnostics" / "run",
        run_id="test-run",
        canonical_draws=canonical_draws,
        holdout_groups=2 if ordinary_groups >= 3 else 1,
        holdout_factorial_groups=1,
        seed=17,
        timeout=30,
        invocation_budget_s=10,
    )
    plan_path = diagnostic.write_diagnostic_plan(plan)
    return plan_path, plan, schedule


def _audit_feedback() -> str:
    return json.dumps(
        {
            "none_reason": None,
            "desired_edges": [],
            "closest_option": "A",
            "disputed_edges": [],
            "missing_info": [],
            "ambiguities": [],
            "rubric_gaps": [],
            "strongest_counterargument": "Option B is geometrically close.",
            "confidence_basis": "Exact continuation.",
        }
    )


def _fake_invoke(calls: list[dict]):
    def invoke(
        provider,
        group_id,
        group_dir,
        prompt,
        letters,
        options,
        **kwargs,
    ):
        calls.append(
            {
                "group_id": group_id,
                "prompt": prompt,
                "collect_feedback": kwargs["collect_feedback"],
            }
        )
        return Vote(
            group_id=group_id,
            provider=provider.name,
            model=provider.model,
            choice="A",
            confidence=0.9,
            reasoning="A is the exact physical continuation.",
            edge_set=frozenset(options["A"]),
            latency_s=0.01,
            timestamp=datetime.now(UTC).isoformat(),
            raw='{"choice":"A"}',
            pack_feedback=_audit_feedback() if kwargs["collect_feedback"] else "",
            evidence_delivery="{}",
        )

    return invoke


def test_plan_is_stratified_group_level_and_export_quarantined(
    tmp_path: Path, fake_evidence
) -> None:
    plan_path, plan, _ = _plan(tmp_path, fake_evidence)

    assert len(plan["packs"]) == 6
    assert plan["source_manifest"]["unique_group_count"] == 5
    assert plan["planned_call_count"] == 24
    assignments = {
        (item["dataset_id"], item["group_id"]): item["cohort"]
        for item in plan["cohort_policy"]["assignments"]
    }
    assert sum(value == "holdout" for value in assignments.values()) == 2
    factorial_cohorts = {
        pack["cohort"] for pack in plan["packs"] if pack["group_id"] == "factorial"
    }
    assert factorial_cohorts == {"holdout"}
    assert len(plan["smoke_pack_keys"]) == 2
    assert (plan_path.parent / ".no-export").is_file()
    assert not list(plan_path.parent.rglob("votes.csv"))
    assert not list(plan_path.parent.rglob("consensus.csv"))
    diagnostic.validate_diagnostic_plan(plan_path)


def test_output_inside_source_batch_is_rejected(tmp_path: Path, fake_evidence) -> None:
    manifest_path, schedule = _wave(tmp_path)
    with pytest.raises(ValueError, match="must not be inside source batch"):
        diagnostic.build_diagnostic_plan(
            manifest_path,
            Path(schedule[0]["batch_dir"]) / "diagnostic",
            run_id="bad",
            holdout_groups=2,
            holdout_factorial_groups=1,
        )


def test_smoke_full_resume_is_atomic_and_does_not_touch_panel_files(
    tmp_path: Path, fake_evidence
) -> None:
    plan_path, plan, schedule = _plan(tmp_path, fake_evidence, ordinary_groups=3, canonical_draws=2)
    sentinel = Path(schedule[0]["batch_dir"]) / "votes.partial.csv"
    sentinel.write_bytes(b"user-panel-state\n")
    before = sentinel.read_bytes()
    calls: list[dict] = []
    progress: list[str] = []

    smoke = diagnostic.run_diagnostic(
        plan_path,
        workers=1,
        smoke=True,
        invoke=_fake_invoke(calls),
        progress=progress.append,
    )
    assert smoke == {"selected": 6, "completed": 6, "skipped": 0}
    assert sum(call["collect_feedback"] for call in calls) == 2
    assert all(
        (diagnostic.AUDIT_INSTRUCTION in call["prompt"]) == call["collect_feedback"]
        for call in calls
    )
    assert any("choice=SEALED" in line for line in progress)
    assert all("choice=A" not in line for line in progress if "factorial" in line)

    full = diagnostic.run_diagnostic(
        plan_path,
        workers=2,
        invoke=_fake_invoke(calls),
        progress=lambda _: None,
    )
    assert full == {"selected": 15, "completed": 9, "skipped": 6}
    result_files = sorted(Path(plan["output_dir"]).glob("*/results/*/*.json"))
    frozen_bytes = {path: path.read_bytes() for path in result_files}

    resumed = diagnostic.run_diagnostic(
        plan_path,
        workers=2,
        invoke=_fake_invoke(calls),
        progress=lambda _: None,
    )
    assert resumed == {"selected": 15, "completed": 0, "skipped": 15}
    assert {path: path.read_bytes() for path in result_files} == frozen_bytes
    assert sentinel.read_bytes() == before
    assert not list(Path(plan["output_dir"]).rglob("*.tmp"))
    assert diagnostic.diagnostic_status(plan_path)["complete"] == 15


def test_source_prompt_drift_invalidates_resume(tmp_path: Path, fake_evidence) -> None:
    plan_path, plan, _ = _plan(tmp_path, fake_evidence)
    prompt = Path(plan["packs"][0]["group_dir"]) / "prompt.txt"
    prompt.write_text(prompt.read_text() + "tampered\n")
    with pytest.raises(ValueError, match="source prompt changed"):
        diagnostic.validate_diagnostic_plan(plan_path)


def test_source_batch_drift_invalidates_resume(tmp_path: Path, fake_evidence) -> None:
    plan_path, plan, _ = _plan(tmp_path, fake_evidence)
    batch_path = Path(plan["packs"][0]["batch_dir"]) / "batch.json"
    batch_path.write_text(batch_path.read_text() + "\n")
    with pytest.raises(ValueError, match="source batch changed"):
        diagnostic.validate_diagnostic_plan(plan_path)


def test_feedback_validation_rejects_false_no_exact_option(tmp_path: Path, fake_evidence) -> None:
    plan_path, plan, _ = _plan(tmp_path, fake_evidence)
    pack = plan["packs"][0]
    feedback = json.loads(_audit_feedback())
    feedback["none_reason"] = "no_exact_option"
    feedback["desired_edges"] = [{"ref_id": "R1", "target_id": "T1"}]
    status, normalized, errors = diagnostic.validate_pack_feedback(
        json.dumps(feedback),
        choice="NONE",
        pack=pack,
        evidence_manifest=_evidence(pack["group_id"]),
    )
    assert status == "invalid"
    assert normalized["desired_edges_normalized"] == [{"ref_id": "ref-1", "target_id": "target-1"}]
    assert any("exactly matches" in error for error in errors)
    assert plan_path.exists()


def test_feedback_validation_contains_malformed_field_types(tmp_path: Path, fake_evidence) -> None:
    _, plan, _ = _plan(tmp_path, fake_evidence)
    pack = plan["packs"][0]
    feedback = json.loads(_audit_feedback())
    feedback.update(
        {
            "none_reason": ["not", "hashable"],
            "closest_option": {"also": "not hashable"},
            "disputed_edges": [
                {
                    "ref_id": "R1",
                    "target_id": "T1",
                    "verdict": ["include"],
                    "missing_evidence": [],
                }
            ],
        }
    )
    status, normalized, errors = diagnostic.validate_pack_feedback(
        json.dumps(feedback),
        choice="A",
        pack=pack,
        evidence_manifest=_evidence(pack["group_id"]),
    )
    assert status == "invalid"
    assert normalized["none_reason"] is None
    assert len(errors) == 3


def test_three_way_split_has_no_arbitrary_modal_result(tmp_path: Path, fake_evidence) -> None:
    _, plan, _ = _plan(tmp_path, fake_evidence)
    pack = plan["packs"][0]

    def result(pass_id: str, choice: str, edge_set: list[list[str]]) -> dict:
        return {
            "pass_id": pass_id,
            "choice": choice,
            "confidence": 0.5,
            "edge_set": edge_set,
            "reasoning": "test",
            "abstain_reason": "",
            "error": "",
        }

    results = {
        "canonical-1": result("canonical-1", "A", [["ref-1", "target-1"]]),
        "canonical-2": result("canonical-2", "B", [["ref-1", "target-2"]]),
        "canonical-3": result("canonical-3", "NONE", []),
    }
    row = diagnostic._analyze_pack(plan, pack, results, human=None)
    assert row["stability"] == "split"
    assert row["modal_choice"] is None
    assert row["modal_edge_set"] == []
    assert row["modal_exact_human"] is None


def test_human_menu_expressible_counts_the_selectable_empty_set() -> None:
    letter_a = frozenset({("ref-1", "target-1")})
    letter_b = frozenset({("ref-1", "target-2")})
    option_sets = {letter_a, letter_b}

    # 1. empty-set (reject-all) label + NONE selectable -> expressible
    assert diagnostic._human_menu_expressible(frozenset(), option_sets, True) is True
    # 2. empty-set label + NONE NOT selectable -> inexpressible
    assert diagnostic._human_menu_expressible(frozenset(), option_sets, False) is False
    # 3. nonempty label equal to a lettered option -> expressible
    assert diagnostic._human_menu_expressible(letter_a, option_sets, False) is True
    # 4. nonempty label absent from the menu -> inexpressible, and NONE never rescues it
    orphan = frozenset({("ref-9", "target-9")})
    assert diagnostic._human_menu_expressible(orphan, option_sets, False) is False
    assert diagnostic._human_menu_expressible(orphan, option_sets, True) is False


def test_analysis_defaults_to_development_and_holdout_requires_freeze(
    tmp_path: Path, fake_evidence
) -> None:
    plan_path, plan, _ = _plan(tmp_path, fake_evidence, ordinary_groups=3, canonical_draws=3)
    diagnostic.run_diagnostic(
        plan_path,
        workers=2,
        invoke=_fake_invoke([]),
        progress=lambda _: None,
    )
    summary = diagnostic.analyze_diagnostic(plan_path, labels_root=tmp_path / "no-labels")
    expected_development = sum(pack["cohort"] == "development" for pack in plan["packs"])
    assert summary["cohort"] == "development"
    assert summary["packs_analyzed"] == expected_development
    assert summary["stability_counts"] == {"stable": expected_development}
    assert summary["valid_audit_feedback"] == expected_development

    with pytest.raises(ValueError, match="fix-frozen"):
        diagnostic.analyze_diagnostic(plan_path, include_holdout=True)
    marker = tmp_path / "fix-frozen.json"
    marker.write_text(json.dumps({"fix_id": "rubric-v2-frozen"}))
    holdout = diagnostic.analyze_diagnostic(
        plan_path,
        include_holdout=True,
        fix_frozen_marker=marker,
        labels_root=tmp_path / "no-labels",
    )
    assert holdout["cohort"] == "holdout"
    assert holdout["fix_frozen"]["fix_id"] == "rubric-v2-frozen"
    assert holdout["factorial_groups"][0]["modal_contrast"] == "consistent"
    assert len(holdout["factorial_groups"][0]["variants"]) == 2
    assert holdout["per_pack"][0]["canonical_votes"][0]["reasoning"]
    assert holdout["per_pack"][0]["audit"]["feedback_status"] == "valid"


def test_worker_limit_is_fail_fast(tmp_path: Path, fake_evidence) -> None:
    plan_path, _, _ = _plan(tmp_path, fake_evidence)
    with pytest.raises(ValueError, match="workers must be"):
        diagnostic.run_diagnostic(plan_path, workers=diagnostic.MAX_WORKERS + 1)
