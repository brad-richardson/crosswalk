"""Integrity tests for the targeted physical/coincidence stitch-wave builder."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from crosswalk.agent_labeling.stitch_runner import get_panel, panel_descriptor
from crosswalk.agent_labeling.wave_manifest import WaveManifest

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
    expected_panel = panel_descriptor(get_panel("v7-candidate"))
    manifest_path = tmp_path / "manifest.json"
    content = {
        "panel": "v7-candidate",
        "required_panel": expected_panel,
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
    WaveManifest(content).write(manifest_path)

    manifest, panel = wave_runner.load_and_validate_manifest(manifest_path)

    assert manifest["total_pack_count"] == 1
    assert panel_descriptor(panel) == expected_panel

    # Tamper the panel and re-stamp the digest so the drift check (not the
    # integrity digest) is what fails.
    content["required_panel"][1]["model"] = "gpt-5.6-terra"
    WaveManifest(content).write(manifest_path)
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
    panel = get_panel("v7-candidate")
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


def _schedule_row(index: int, batch_dir: Path, group_id: str) -> dict:
    return {
        "run_index": index,
        "batch_dir": str(batch_dir),
        "group_id": group_id,
        "dataset_id": "dataset",
        "variant": "enriched",
    }


def _panel_votes(panel, group_ids, choice: str = "A", abstain_reason: str = "") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "provider": spec.name,
                "model": spec.model,
                "choice": choice,
                "abstain_reason": abstain_reason,
            }
            for _group_id in group_ids
            for spec in panel
        ]
    )


def test_parallel_schedule_serializes_same_batch_dir_and_completes_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    import time

    panel = get_panel("v7-candidate")
    batch_a = tmp_path / "batch-a"
    batch_b = tmp_path / "batch-b"
    # Contiguous-per-dir order mirrors the real manifests: a dispatcher must
    # skip ahead past a busy dir (a-2) to reach b-1 for any parallelism.
    schedule = [
        _schedule_row(1, batch_a, "a-1"),
        _schedule_row(2, batch_a, "a-2"),
        _schedule_row(3, batch_b, "b-1"),
        _schedule_row(4, batch_b, "b-2"),
    ]

    state_lock = threading.Lock()
    active: dict[Path, int] = {}
    max_active_per_dir: dict[Path, int] = {}
    max_active_total = 0
    completed: set[str] = set()
    spans: dict[str, tuple[float, float]] = {}

    def fake_run_batch(batch_dir, *, group_ids, panel, **_kwargs):
        nonlocal max_active_total
        batch_dir = Path(batch_dir)
        begin = time.monotonic()
        with state_lock:
            active[batch_dir] = active.get(batch_dir, 0) + 1
            max_active_per_dir[batch_dir] = max(
                max_active_per_dir.get(batch_dir, 0), active[batch_dir]
            )
            max_active_total = max(max_active_total, sum(active.values()))
        time.sleep(0.05)
        with state_lock:
            active[batch_dir] -= 1
            completed.update(group_ids)
            if len(group_ids) == 1:
                # Per-pack spans only: the final consolidation pass re-calls
                # run_batch with each dir's full roster and must not overwrite
                # the scheduled pack's timing.
                spans[group_ids[0]] = (begin, time.monotonic())
        votes = _panel_votes(panel, group_ids)
        consensus = pd.DataFrame([{"group_id": group_id} for group_id in group_ids])
        return votes, consensus

    monkeypatch.setattr(wave_runner, "run_batch", fake_run_batch)
    wave_runner.execute_schedule(
        {"run_schedule": schedule},
        panel,
        timeout=600,
        invocation_budget=600,
        group_workers=3,
    )

    # The partial-CSV read-modify-write means a batch dir must never see two
    # concurrent run_batch calls, while distinct batch dirs may overlap.
    assert max(max_active_per_dir.values()) == 1
    assert completed == {"a-1", "a-2", "b-1", "b-2"}
    # Real parallelism happened: the dispatcher skipped past busy-dir a-2 and
    # ran b-1 while a-1 was still in flight.
    assert max_active_total >= 2
    assert spans["b-1"][0] < spans["a-1"][1]
    # Manifest order within a dir is preserved.
    assert spans["a-2"][0] >= spans["a-1"][1]
    assert spans["b-2"][0] >= spans["b-1"][1]


def test_parallel_schedule_halts_all_lanes_after_first_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    panel = get_panel("v7-candidate")
    # Lane 1 fails on its only pack; lane 2 has a pack in flight while the
    # failure happens plus a follow-up pack; a third dir is still queued.
    schedule = [
        _schedule_row(1, tmp_path / "batch-fail", "fail-1"),
        _schedule_row(2, tmp_path / "batch-slow", "slow-1"),
        _schedule_row(3, tmp_path / "batch-slow", "slow-2"),
        _schedule_row(4, tmp_path / "batch-queued", "queued-1"),
    ]
    slow_started = threading.Event()
    failure_raised = threading.Event()
    started: list[str] = []
    started_lock = threading.Lock()

    def fake_run_batch(_batch_dir, *, group_ids, panel, **_kwargs):
        with started_lock:
            started.extend(group_ids)
        if group_ids == ["fail-1"]:
            # Fail only once the other lane's pack is in flight, so the test
            # exercises "in-flight pack finishes, its lane then stops".
            assert slow_started.wait(timeout=10)
            failure_raised.set()
            raise wave_runner.ProviderInvocationError("claude: quota symptom")
        if group_ids == ["slow-1"]:
            slow_started.set()
            # Hold this pack in flight until the other worker has failed, so
            # the stop event is set before this worker considers slow-2. The
            # short sleep gives the failing worker's `stop.set()` (which runs
            # a few frames after failure_raised.set()) a wide margin over this
            # worker's post-return bookkeeping.
            assert failure_raised.wait(timeout=10)
            import time

            time.sleep(0.05)
        votes = _panel_votes(panel, group_ids)
        consensus = pd.DataFrame([{"group_id": group_id} for group_id in group_ids])
        return votes, consensus

    monkeypatch.setattr(wave_runner, "run_batch", fake_run_batch)
    with pytest.raises(wave_runner.ProviderInvocationError, match="quota symptom"):
        wave_runner.execute_schedule(
            {"run_schedule": schedule},
            panel,
            timeout=600,
            invocation_budget=600,
            group_workers=2,
        )

    assert "fail-1" in started
    # The in-flight pack finishes (its partial flushes for resume)...
    assert "slow-1" in started
    # ...but its lane stops before the next pack, and the never-started lane
    # is cancelled or exits on the stop event without invoking the panel.
    assert "slow-2" not in started
    assert "queued-1" not in started


@pytest.mark.parametrize("group_workers", [1, 2])
def test_pause_drains_in_flight_pack_and_skips_consolidation(
    group_workers: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    panel = get_panel("v7-candidate")
    # Two packs in one dir plus a second dir: pausing during the very first
    # pack must prevent every later pack AND the final consolidation pass
    # (which would otherwise invoke the panel on not-yet-voted groups).
    schedule = [
        _schedule_row(1, tmp_path / "batch-a", "a-1"),
        _schedule_row(2, tmp_path / "batch-a", "a-2"),
        _schedule_row(3, tmp_path / "batch-b", "b-1"),
    ]
    pause = threading.Event()
    calls: list[list[str]] = []
    calls_lock = threading.Lock()

    def fake_run_batch(_batch_dir, *, group_ids, panel, **_kwargs):
        with calls_lock:
            calls.append(list(group_ids))
        if group_ids == ["a-1"]:
            pause.set()
        votes = _panel_votes(panel, group_ids)
        consensus = pd.DataFrame([{"group_id": group_id} for group_id in group_ids])
        return votes, consensus

    monkeypatch.setattr(wave_runner, "run_batch", fake_run_batch)
    paused = wave_runner.execute_schedule(
        {"run_schedule": schedule},
        panel,
        timeout=600,
        invocation_budget=600,
        group_workers=group_workers,
        pause=pause,
    )

    assert paused is True
    # a-2 never starts; b-1 may only have started before the pause landed
    # (parallel mode); no multi-group consolidation call ever happens.
    flat = [group_id for call in calls for group_id in call]
    assert "a-1" in flat
    assert "a-2" not in flat
    assert all(len(call) == 1 for call in calls)


def test_pause_handler_debounces_duplicate_signals_then_aborts() -> None:
    import threading

    pause = threading.Event()
    monotonic = {"now": 100.0}
    handler = {}

    def fake_signal(signum, fn):
        handler[signum] = fn

    original_signal, original_monotonic = wave_runner.signal.signal, wave_runner.time.monotonic
    wave_runner.signal.signal = fake_signal
    wave_runner.time.monotonic = lambda: monotonic["now"]
    try:
        wave_runner._install_pause_handler(pause)
        sigint = handler[wave_runner.signal.SIGINT]

        sigint(wave_runner.signal.SIGINT, None)
        assert pause.is_set()
        # uv run forwards the same Ctrl-C to the child: a duplicate inside the
        # debounce window must NOT abort the drain.
        monotonic["now"] = 100.5
        sigint(wave_runner.signal.SIGINT, None)
        # A deliberate second signal after the window force-aborts.
        monotonic["now"] = 103.0
        with pytest.raises(KeyboardInterrupt):
            sigint(wave_runner.signal.SIGINT, None)
    finally:
        wave_runner.signal.signal = original_signal
        wave_runner.time.monotonic = original_monotonic


def test_parallel_schedule_applies_wave_timeout_breaker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = get_panel("v7-candidate")
    schedule = [_schedule_row(i, tmp_path / f"batch-{i}", f"group-{i}") for i in range(1, 5)]

    def fake_run_batch(_batch_dir, *, group_ids, panel, **_kwargs):
        votes = _panel_votes(panel, group_ids)
        votes.loc[votes["provider"] == "codex", ["choice", "abstain_reason"]] = [
            "ABSTAIN",
            "timeout",
        ]
        consensus = pd.DataFrame([{"group_id": group_id} for group_id in group_ids])
        return votes, consensus

    monkeypatch.setattr(wave_runner, "run_batch", fake_run_batch)
    with pytest.raises(wave_runner.ProviderInvocationError, match="3 consecutive scheduled groups"):
        wave_runner.execute_schedule(
            {"run_schedule": schedule},
            panel,
            timeout=600,
            invocation_budget=600,
            group_workers=2,
        )
