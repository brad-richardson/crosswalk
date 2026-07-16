#!/usr/bin/env python3
"""Execute a physical/coincidence stitch wave in its counterbalanced order."""

from __future__ import annotations

import argparse
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from crosswalk.agent_labeling.stitch_runner import (
    AbstainReason,
    ProviderInvocationError,
    run_batch,
)
from crosswalk.agent_labeling.wave_manifest import WaveManifest

WAVE_TIMEOUT_BREAKER_N = 3


def load_and_validate_manifest(path: Path) -> tuple[dict, list[Any]]:
    """Load, integrity-check, and structurally validate the wave manifest.

    Thin wrapper over :meth:`WaveManifest.load_validated`, which owns the
    contract (field names, schema version, integrity digest, panel-drift check,
    schedule invariants, and cwd-independent ``batch_dir`` resolution). The
    returned schedule rows carry manifest-resolved ``batch_dir`` paths.
    """
    manifest = WaveManifest.load_validated(path)
    return manifest.content, manifest.panel


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
        timed_out = str(row.get("choice", "")) == "ABSTAIN" and str(
            row.get("abstain_reason", "")
        ) == str(AbstainReason.TIMEOUT)
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
        votes[~votes["group_id"].astype(str).isin(timeout_ids)].to_csv(votes_path, index=False)
        consensus[~consensus["group_id"].astype(str).isin(timeout_ids)].to_csv(
            consensus_path, index=False
        )
        dropped.update((batch_dir, group_id) for group_id in timeout_ids)
    return dropped


def _run_scheduled_pack(
    position: int,
    total: int,
    row: dict,
    panel: list[Any],
    *,
    timeout: int,
    invocation_budget: float,
) -> Any:
    """Invoke the full panel on one scheduled pack and validate its ballots."""
    batch_dir = Path(row["batch_dir"])
    group_id = str(row["group_id"])
    print(
        f"[{position}/{total}] {row['dataset_id']} {group_id} variant={row['variant']}",
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
    if len(consensus) != 1:
        raise RuntimeError(f"{group_id}: expected one consensus row")
    return votes


def _execute_rows_parallel(
    schedule: list[dict],
    panel: list[Any],
    *,
    timeout: int,
    invocation_budget: float,
    group_workers: int,
) -> None:
    """Run scheduled packs on a bounded worker pool, one lane per batch dir.

    Concurrency contract:

    - ``run_batch`` rewrites a batch dir's ``votes.partial.csv`` /
      ``consensus.partial.csv`` WHOLE on every per-group flush, so two
      concurrent ``run_batch`` calls on the same batch dir would race the
      read-modify-write and silently drop each other's rows. The unit of
      parallelism is therefore the BATCH DIR: each pool task owns one dir
      exclusively and runs that dir's scheduled packs in schedule order.
      (Per-row tasks with per-dir locks would degrade to sequential execution:
      the schedule lists each dir's packs contiguously, so a FIFO pool's
      workers would all queue on the first dir's lock.)
    - The wave timeout breaker counts consecutive timed-out packs per provider
      in COMPLETION order (the sequential path counts in schedule order, which
      is undefined under concurrency). Same threshold and halt semantics.
    - On the first failure (provider-down, quota symptom, ballot validation),
      a shared stop event makes every other lane halt before starting its next
      pack, and batch dirs not yet started are cancelled. Only packs already
      in flight run to completion (their partials flush for resume) before the
      error re-raises as the wave-level stop.
    """
    rows_by_batch: dict[Path, list[tuple[int, dict]]] = defaultdict(list)
    for position, row in enumerate(schedule, start=1):
        rows_by_batch[Path(row["batch_dir"])].append((position, row))

    streak_lock = threading.Lock()
    consecutive_timeouts: dict[str, int] = {}
    stop = threading.Event()

    def _run_batch_lane(rows: list[tuple[int, dict]]) -> None:
        for position, row in rows:
            if stop.is_set():
                return
            try:
                votes = _run_scheduled_pack(
                    position,
                    len(schedule),
                    row,
                    panel,
                    timeout=timeout,
                    invocation_budget=invocation_budget,
                )
                with streak_lock:
                    _record_wave_timeout_streaks(votes, consecutive_timeouts, str(row["group_id"]))
            except BaseException:
                stop.set()
                raise

    with ThreadPoolExecutor(max_workers=group_workers) as ex:
        futures = [ex.submit(_run_batch_lane, rows) for rows in rows_by_batch.values()]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise


def execute_schedule(
    manifest: dict,
    panel: list[Any],
    *,
    timeout: int,
    invocation_budget: float,
    retry_timeouts: bool = False,
    group_workers: int = 1,
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

    if group_workers > 1:
        _execute_rows_parallel(
            schedule,
            panel,
            timeout=timeout,
            invocation_budget=invocation_budget,
            group_workers=group_workers,
        )
    else:
        consecutive_timeouts: dict[str, int] = {}
        for position, row in enumerate(schedule, start=1):
            votes = _run_scheduled_pack(
                position,
                len(schedule),
                row,
                panel,
                timeout=timeout,
                invocation_budget=invocation_budget,
            )
            _record_wave_timeout_streaks(votes, consecutive_timeouts, str(row["group_id"]))

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
    parser.add_argument(
        "--group-workers",
        type=int,
        default=1,
        help=(
            "Run up to N scheduled packs concurrently (each pack still fans its "
            "panel seats out in parallel, so provider concurrency is N per seat). "
            "Packs sharing a batch dir stay serialized to protect the partial-CSV "
            "read-modify-write. Default 1 preserves the strict schedule order."
        ),
    )
    args = parser.parse_args()
    if args.group_workers < 1:
        parser.error("--group-workers must be >= 1")

    manifest, panel = load_and_validate_manifest(args.manifest)
    schedule = manifest["run_schedule"]
    print(
        f"Validated {len(schedule)} scheduled packs with panel "
        + ", ".join(f"{spec.name}={spec.model}/{spec.effort}" for spec in panel),
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
        group_workers=args.group_workers,
    )


if __name__ == "__main__":
    main()
