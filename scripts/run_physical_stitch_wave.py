#!/usr/bin/env python3
"""Execute a physical/coincidence stitch wave in its counterbalanced order."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from crosswalk.agent_labeling.stitch_runner import (
    AbstainReason,
    ProviderInvocationError,
    get_panel,
    run_batch,
)

WAVE_TIMEOUT_BREAKER_N = 3


def _panel_descriptor(panel: list[Any]) -> list[dict[str, str | None]]:
    return [
        {"provider": spec.name, "model": spec.model, "effort": spec.effort}
        for spec in panel
    ]


def load_and_validate_manifest(path: Path) -> tuple[dict, list[Any]]:
    manifest = json.loads(path.read_text())
    panel_name = str(manifest.get("panel", ""))
    panel = get_panel(panel_name)
    actual_panel = _panel_descriptor(panel)
    if actual_panel != manifest.get("required_panel"):
        raise ValueError(
            f"Panel drift for {panel_name}: expected {manifest.get('required_panel')}, "
            f"got {actual_panel}"
        )

    schedule = manifest.get("run_schedule") or []
    if len(schedule) != int(manifest.get("total_pack_count", -1)):
        raise ValueError("run_schedule length does not match total_pack_count")
    expected_indices = list(range(1, len(schedule) + 1))
    if [row.get("run_index") for row in schedule] != expected_indices:
        raise ValueError("run_schedule indices are not contiguous and ordered")

    seen: set[tuple[str, str]] = set()
    for row in schedule:
        batch_dir = Path(row["batch_dir"])
        group_id = str(row["group_id"])
        key = (str(batch_dir), group_id)
        if key in seen:
            raise ValueError(f"duplicate scheduled pack: {key}")
        seen.add(key)
        if not (batch_dir / "batch.json").is_file():
            raise FileNotFoundError(batch_dir / "batch.json")
        if not (batch_dir / group_id / "evidence.json").is_file():
            raise FileNotFoundError(batch_dir / group_id / "evidence.json")
    return manifest, panel


def _validate_group_votes(votes, panel: list[Any], group_id: str) -> None:
    expected = {(spec.name, spec.model) for spec in panel}
    actual = set(zip(votes["provider"], votes["model"], strict=False))
    if len(votes) != len(panel) or actual != expected:
        raise RuntimeError(
            f"{group_id}: recorded voters {sorted(actual)} do not match "
            f"required panel {sorted(expected)}"
        )


def _record_wave_timeout_streaks(
    votes,
    consecutive_timeouts: dict[str, int],
    group_id: str,
    *,
    threshold: int = WAVE_TIMEOUT_BREAKER_N,
) -> None:
    """Preserve run_batch's provider-down timeout breaker across row-wise calls."""
    for row in votes.to_dict("records"):
        provider = str(row["provider"])
        timed_out = (
            str(row.get("choice", "")) == "ABSTAIN"
            and str(row.get("abstain_reason", "")) == str(AbstainReason.TIMEOUT)
        )
        consecutive_timeouts[provider] = (
            consecutive_timeouts.get(provider, 0) + 1 if timed_out else 0
        )
        if consecutive_timeouts[provider] >= threshold:
            raise ProviderInvocationError(
                f"{provider}: timed out on {threshold} consecutive scheduled groups "
                f"(last: {group_id}) — treating as provider-down and halting. "
                "Completed groups are preserved in partial vote files; fix the "
                "provider and rerun this schedule with --retry-timeouts."
            )


def _drop_timeout_groups_from_partials(batch_dirs: set[Path]) -> set[tuple[Path, str]]:
    """Forget only timeout-affected groups so resume reinvokes their full panel."""
    dropped: set[tuple[Path, str]] = set()
    for batch_dir in batch_dirs:
        votes_path = batch_dir / "votes.partial.csv"
        consensus_path = batch_dir / "consensus.partial.csv"
        if not votes_path.is_file() or not consensus_path.is_file():
            continue
        votes = pd.read_csv(votes_path, dtype={"group_id": str})
        consensus = pd.read_csv(consensus_path, dtype={"group_id": str})
        if "abstain_reason" not in votes:
            continue
        timeout_rows = votes[
            (votes["choice"].astype(str) == "ABSTAIN")
            & (votes["abstain_reason"].fillna("").astype(str) == str(AbstainReason.TIMEOUT))
        ]
        timeout_ids = set(timeout_rows["group_id"].astype(str))
        if not timeout_ids:
            continue
        votes[~votes["group_id"].astype(str).isin(timeout_ids)].to_csv(
            votes_path, index=False
        )
        consensus[~consensus["group_id"].astype(str).isin(timeout_ids)].to_csv(
            consensus_path, index=False
        )
        dropped.update((batch_dir, group_id) for group_id in timeout_ids)
    return dropped


def execute_schedule(
    manifest: dict,
    panel: list[Any],
    *,
    timeout: int,
    invocation_budget: float,
    retry_timeouts: bool = False,
) -> None:
    schedule = manifest["run_schedule"]
    scheduled_by_batch: dict[Path, list[str]] = defaultdict(list)
    for row in schedule:
        scheduled_by_batch[Path(row["batch_dir"])].append(str(row["group_id"]))

    if retry_timeouts:
        dropped = _drop_timeout_groups_from_partials(set(scheduled_by_batch))
        print(
            f"Retrying {len(dropped)} timeout-affected packs; successful partials retained",
            flush=True,
        )

    consecutive_timeouts: dict[str, int] = {}
    for position, row in enumerate(schedule, start=1):
        batch_dir = Path(row["batch_dir"])
        group_id = str(row["group_id"])
        print(
            f"[{position}/{len(schedule)}] {row['dataset_id']} {group_id} "
            f"variant={row['variant']}",
            flush=True,
        )
        votes, consensus = run_batch(
            batch_dir,
            panel=panel,
            group_ids=[group_id],
            timeout=timeout,
            collect_feedback=True,
            resume=True,
            invocation_budget_s=invocation_budget,
        )
        _validate_group_votes(votes, panel, group_id)
        _record_wave_timeout_streaks(votes, consecutive_timeouts, group_id)
        if len(consensus) != 1:
            raise RuntimeError(f"{group_id}: expected one consensus row")

    # Row-wise runs intentionally use partial files for crash-safe accumulation.
    # Consolidate every multi-group batch once at the end so votes.csv and
    # consensus.csv contain its complete scheduled roster rather than the last row.
    for batch_dir, group_ids in scheduled_by_batch.items():
        votes, consensus = run_batch(
            batch_dir,
            panel=panel,
            group_ids=group_ids,
            timeout=timeout,
            collect_feedback=True,
            resume=True,
            invocation_budget_s=invocation_budget,
        )
        if len(votes) != len(group_ids) * len(panel) or len(consensus) != len(group_ids):
            raise RuntimeError(
                f"{batch_dir}: final consolidation is incomplete "
                f"({len(votes)} votes, {len(consensus)} consensus rows)"
            )
    print(f"Completed and consolidated {len(schedule)} scheduled packs", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--invocation-budget", type=float, default=600.0)
    parser.add_argument(
        "--retry-timeouts",
        action="store_true",
        help="Drop timeout-affected partial groups and rerun their full panel.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate packs, panel, and schedule without invoking voters.",
    )
    args = parser.parse_args()

    manifest, panel = load_and_validate_manifest(args.manifest)
    schedule = manifest["run_schedule"]
    print(
        f"Validated {len(schedule)} scheduled packs with panel "
        + ", ".join(
            f"{spec.name}={spec.model}/{spec.effort}" for spec in panel
        ),
        flush=True,
    )
    if args.validate_only:
        return

    execute_schedule(
        manifest,
        panel,
        timeout=args.timeout,
        invocation_budget=args.invocation_budget,
        retry_timeouts=args.retry_timeouts,
    )


if __name__ == "__main__":
    main()
