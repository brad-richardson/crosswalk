"""Integrity tests for the targeted physical/coincidence stitch-wave builder."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

SCRIPT = Path(__file__).parents[2] / "scripts" / "build_physical_stitch_wave.py"
SPEC = importlib.util.spec_from_file_location("build_physical_stitch_wave", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
wave = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = wave
SPEC.loader.exec_module(wave)

RUNNER_SCRIPT = Path(__file__).parents[2] / "scripts" / "run_physical_stitch_wave.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_physical_stitch_wave", RUNNER_SCRIPT)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
wave_runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = wave_runner
RUNNER_SPEC.loader.exec_module(wave_runner)


def _line(coords: list[list[float]]) -> dict:
    return {"type": "LineString", "coordinates": coords}


def _physical(level: int = 0, *flags: str) -> dict:
    return {
        "level_lr": [{"between": [0.0, 1.0], "value": level}],
        "road_flags_lr": [{"between": [0.0, 1.0], "value": list(flags)}],
    }


def _group() -> dict:
    ref_a, ref_b = "ref-a", "ref-b"
    target_a, target_b = "target-a", "target-b"
    edges = [
        {
            "ref_id": ref_a,
            "target_id": target_a,
            "confidence": 0.9,
            "gers_start_frac": 0.0,
            "gers_end_frac": 1.0,
            "local_start_frac": 0.0,
            "local_end_frac": 1.0,
            "ref_physical": _physical(1, "is_bridge"),
            "target_physical": _physical(1, "is_bridge"),
        },
        {
            "ref_id": ref_a,
            "target_id": target_b,
            "confidence": 0.45,
            "ref_physical": _physical(1, "is_bridge"),
            "target_physical": _physical(0),
        },
        {
            "ref_id": ref_b,
            "target_id": target_b,
            "confidence": 0.85,
            "ref_physical": _physical(0),
            "target_physical": _physical(0),
        },
    ]
    return {
        "group_id": "synthetic-group",
        "match_type": "M:N",
        "n_candidate_edges": len(edges),
        "ref_ids": [ref_a, ref_b],
        "target_ids": [target_a, target_b],
        "edges": edges,
        "optimizer_assignment": [
            {"ref_id": ref_a, "target_id": target_a},
            {"ref_id": ref_b, "target_id": target_b},
        ],
        "ref_geometries": {
            ref_a: _line([[6.0, 46.0], [6.001, 46.0]]),
            ref_b: _line([[6.0002, 46.00001], [6.0008, 46.00001]]),
        },
        "target_geometries": {
            target_a: _line([[6.0, 46.0001], [6.0005, 46.0001]]),
            target_b: _line([[6.0005, 46.0001], [6.001, 46.0001]]),
        },
        "ref_names": {ref_a: "Bridge", ref_b: "Surface road"},
        "target_names": {target_a: "Bridge", target_b: "Surface road"},
        "ref_classes": {ref_a: "trunk", ref_b: "cycleway"},
        "target_classes": {target_a: "trunk", target_b: "cycleway"},
        "ref_physical": {
            ref_a: _physical(1, "is_bridge"),
            ref_b: _physical(0),
        },
        "target_physical": {
            target_a: _physical(1, "is_bridge"),
            target_b: _physical(0),
        },
    }


def test_target_capabilities_and_sanitization_remove_unsupported_domains() -> None:
    sydney = wave._target_capabilities("au_sydney_roads")
    assert sydney.has_level is False
    assert sydney.flag_domains == frozenset({"is_bridge", "is_tunnel"})

    group = _group()
    group["target_physical"]["target-a"]["road_flags_lr"][0]["value"].append("is_covered")
    wave._sanitize_group_target_physical(group, sydney)

    assert "level_lr" not in group["target_physical"]["target-a"]
    assert group["target_physical"]["target-a"]["road_flags_lr"][0]["value"] == ["is_bridge"]
    assert "level_lr" not in group["edges"][0]["target_physical"]

    london = wave._target_capabilities("gb_london_roads")
    assert london.has_level is False
    assert london.flag_domains == frozenset({"is_tunnel"})
    geneva = wave._target_capabilities("ch_grand_geneva_cycle_schema")
    assert geneva.has_level is False
    assert geneva.flag_domains == frozenset()


def test_regression_pairs_must_be_in_the_offered_edge_universe() -> None:
    group = _group()
    rejected = {"ref_id": "known-ref", "target_id": "known-target"}
    group["rejected_edges"] = [rejected]
    capabilities = wave._target_capabilities("fi_helsinki_roads")

    ranked = wave._rank_group(
        group,
        manual_pairs=set(),
        known_none=set(),
        capabilities=capabilities,
        required_pairs={("known-ref", "known-target")},
        forced_ref_ids=set(),
    )

    assert ranked is not None
    assert ranked.audit["required_pair_hits"] == []


def test_streaming_selection_forces_required_offered_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    required = _group()
    ordinary = _group()
    ordinary["group_id"] = "ordinary-group"
    ordinary["edges"][0]["confidence"] = 0.99
    ordinary["edges"][1]["target_id"] = "ordinary-target"
    sidecar = tmp_path / "groups.json"
    sidecar.write_text(json.dumps({"groups": [ordinary, required]}))
    monkeypatch.setattr(wave, "_known_none_group_ids", lambda dataset: set())

    pair = (required["edges"][1]["ref_id"], required["edges"][1]["target_id"])
    selected = wave.select_dataset_groups(
        "fi_helsinki_roads",
        sidecar,
        quota=1,
        manual_pairs=set(),
        required_pairs={pair},
        forced_ref_ids=set(),
    )

    assert [row.group_id for row in selected] == [required["group_id"]]
    assert selected[0].audit["required_pair_hits"] == [list(pair)]


@pytest.mark.parametrize(
    "dataset_id",
    [
        "au_sydney_roads",
        "gb_london_roads",
        "ch_grand_geneva_cycle_schema",
    ],
)
def test_unsupported_target_layer_never_reaches_pack_display(
    dataset_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wave, "_fill_spatial_context", lambda groups, dataset: None)
    sidecars = tmp_path / "sidecars"
    output = tmp_path / "batches"
    sidecars.mkdir()
    for suffix in ("bridge.parquet", "candidates.parquet", "groups.json"):
        (sidecars / f"{dataset_id}_{suffix}").write_text("fixture")
    group = _group()
    ranked = wave.RankedGroup(1.0, group["group_id"], group, (), {})

    batch_dir = wave._write_batch(
        dataset_id,
        [ranked],
        sidecar_root=sidecars,
        output_root=output,
        wave_name="no_unsupported_layer",
        variant="enriched",
        k_alternatives=4,
    )

    group_dir = batch_dir / group["group_id"]
    prompt = (group_dir / "prompt.txt").read_text()
    metadata = yaml.safe_load((group_dir / "metadata.yaml").read_text())
    assert "T physical: layer" not in prompt
    assert all("level_lr" not in segment["physical"] for segment in metadata["segments"]["target"])


def test_factorial_packs_keep_menu_fixed_and_toggle_only_requested_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wave, "_fill_spatial_context", lambda groups, dataset: None)
    sidecars = tmp_path / "sidecars"
    output = tmp_path / "batches"
    sidecars.mkdir()
    for suffix in ("bridge.parquet", "candidates.parquet", "groups.json"):
        (sidecars / f"fi_helsinki_roads_{suffix}").write_text("fixture")

    group = _group()
    ranked = wave.RankedGroup(
        score=1.0,
        group_id=group["group_id"],
        group=group,
        tags=("coincidence", "physical_agreement"),
        audit={},
    )
    dirs = {
        variant: wave._write_batch(
            "fi_helsinki_roads",
            [ranked],
            sidecar_root=sidecars,
            output_root=output,
            wave_name="test_wave",
            variant=variant,
            k_alternatives=4,
        )
        for variant in wave.VARIANTS
    }

    evidence = {
        variant: json.loads((batch_dir / group["group_id"] / "evidence.json").read_text())[
            "evidence"
        ]
        for variant, batch_dir in dirs.items()
    }
    assert len({row["option_menu_sha256"] for row in evidence.values()}) == 1

    prompts = {
        variant: (batch_dir / group["group_id"] / "prompt.txt").read_text()
        for variant, batch_dir in dirs.items()
    }
    assert "R physical:" in prompts["enriched"]
    assert "Same-side coincidence" in prompts["enriched"]
    assert "R physical:" not in prompts["no_physical"]
    assert "Same-side coincidence" in prompts["no_physical"]
    assert "R physical:" in prompts["no_coincidence"]
    assert "Same-side coincidence" not in prompts["no_coincidence"]
    assert "R physical:" not in prompts["minimal"]
    assert "Same-side coincidence" not in prompts["minimal"]

    selections = {"fi_helsinki_roads": [ranked]}
    batch_dirs = {("fi_helsinki_roads", variant): batch_dir for variant, batch_dir in dirs.items()}
    required_pair = (
        group["edges"][0]["ref_id"],
        group["edges"][0]["target_id"],
    )
    wave._assert_required_pairs_in_generated_menus(
        selections,
        {"fi_helsinki_roads": {required_pair}},
        batch_dirs,
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        wave._write_batch(
            "fi_helsinki_roads",
            [ranked],
            sidecar_root=sidecars,
            output_root=output,
            wave_name="test_wave",
            variant="enriched",
            k_alternatives=4,
        )


def test_wave_preflight_rejects_any_output_collision_before_writes(
    tmp_path: Path,
) -> None:
    occupied = tmp_path / "dataset_wave"
    occupied.mkdir()
    planned = {("dataset", "enriched"): occupied}

    with pytest.raises(FileExistsError, match="partial/overwritten wave"):
        wave._assert_output_paths_available(planned, tmp_path / "manifest.json")


def test_schedule_runner_validates_exact_panel_and_pack_roster(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    group_dir = batch_dir / "group-1"
    group_dir.mkdir(parents=True)
    (batch_dir / "batch.json").write_text("{}")
    (group_dir / "evidence.json").write_text("{}")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "panel": "v7-candidate",
                "required_panel": list(wave.REQUIRED_PANEL),
                "total_pack_count": 1,
                "run_schedule": [
                    {
                        "run_index": 1,
                        "batch_dir": str(batch_dir),
                        "group_id": "group-1",
                        "dataset_id": "dataset",
                        "variant": "enriched",
                    }
                ],
            }
        )
    )

    manifest, panel = wave_runner.load_and_validate_manifest(manifest_path)

    assert manifest["total_pack_count"] == 1
    assert wave_runner._panel_descriptor(panel) == list(wave.REQUIRED_PANEL)

    manifest["required_panel"][1]["model"] = "gpt-5.6-terra"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Panel drift"):
        wave_runner.load_and_validate_manifest(manifest_path)


def test_schedule_runner_halts_on_wave_level_consecutive_timeouts() -> None:
    streaks: dict[str, int] = {}

    def votes(choice: str, reason: str = "") -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "provider": "codex",
                    "choice": choice,
                    "abstain_reason": reason,
                }
            ]
        )

    wave_runner._record_wave_timeout_streaks(votes("ABSTAIN", "timeout"), streaks, "group-1")
    wave_runner._record_wave_timeout_streaks(votes("ABSTAIN", "timeout"), streaks, "group-2")
    with pytest.raises(wave_runner.ProviderInvocationError, match="3 consecutive scheduled groups"):
        wave_runner._record_wave_timeout_streaks(votes("ABSTAIN", "timeout"), streaks, "group-3")

    wave_runner._record_wave_timeout_streaks(votes("A"), streaks, "group-4")
    assert streaks["codex"] == 0


def test_schedule_retry_drops_timeout_partials_and_reinvokes_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    panel = wave_runner.get_panel("v7-candidate")
    vote_rows = []
    for group_id in ("timed-out", "successful"):
        for spec in panel:
            timed_out = group_id == "timed-out" and spec.name == "codex"
            vote_rows.append(
                {
                    "group_id": group_id,
                    "provider": spec.name,
                    "model": spec.model,
                    "choice": "ABSTAIN" if timed_out else "A",
                    "abstain_reason": "timeout" if timed_out else "",
                }
            )
    pd.DataFrame(vote_rows).to_csv(batch_dir / "votes.partial.csv", index=False)
    pd.DataFrame([{"group_id": "timed-out"}, {"group_id": "successful"}]).to_csv(
        batch_dir / "consensus.partial.csv", index=False
    )

    schedule = [
        {
            "run_index": index,
            "batch_dir": str(batch_dir),
            "group_id": group_id,
            "dataset_id": "dataset",
            "variant": "enriched",
        }
        for index, group_id in enumerate(("timed-out", "successful"), start=1)
    ]
    calls: list[list[str]] = []

    def fake_run_batch(_batch_dir, *, group_ids, panel, **_kwargs):
        calls.append(list(group_ids))
        if group_ids == ["timed-out"]:
            retained = pd.read_csv(batch_dir / "votes.partial.csv")
            assert "timed-out" not in set(retained["group_id"])
            assert "successful" in set(retained["group_id"])
        votes = pd.DataFrame(
            [
                {
                    "provider": spec.name,
                    "model": spec.model,
                    "choice": "A",
                    "abstain_reason": "",
                }
                for _group_id in group_ids
                for spec in panel
            ]
        )
        consensus = pd.DataFrame([{"group_id": group_id} for group_id in group_ids])
        return votes, consensus

    monkeypatch.setattr(wave_runner, "run_batch", fake_run_batch)
    wave_runner.execute_schedule(
        {"run_schedule": schedule},
        panel,
        timeout=600,
        invocation_budget=600,
        retry_timeouts=True,
    )

    assert calls == [["timed-out"], ["successful"], ["timed-out", "successful"]]
