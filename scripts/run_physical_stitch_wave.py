#!/usr/bin/env python3
"""Execute a physical/coincidence stitch wave in its counterbalanced order."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
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
    record_streaks: Any,
) -> None:
    """Invoke the full panel on one scheduled pack and validate its ballots.

    ``record_streaks(votes, group_id)`` runs between ballot validation and the
    consensus-row check, preserving the pre-parallelism sequential order (the
    timeout breaker fires before a malformed-consensus RuntimeError would).
    """
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
    record_streaks(votes, group_id)
    if len(consensus) != 1:
        raise RuntimeError(f"{group_id}: expected one consensus row")


def _execute_rows_parallel(
    schedule: list[dict],
    panel: list[Any],
    *,
    timeout: int,
    invocation_budget: float,
    group_workers: int,
    pause: threading.Event,
) -> None:
    """Run scheduled packs on a worker pool via an order-preserving dispatcher.

    Concurrency contract:

    - ``run_batch`` rewrites a batch dir's ``votes.partial.csv`` /
      ``consensus.partial.csv`` WHOLE on every per-group flush, so two
      concurrent ``run_batch`` calls on the same batch dir would race the
      read-modify-write and silently drop each other's rows. Workers therefore
      always claim the LOWEST-index unclaimed schedule row whose batch dir has
      no pack in flight. A busy dir's next pack is deferred and later rows of
      other dirs may start first, but start order otherwise tracks the
      manifest's counterbalanced schedule within a ~``group_workers`` window.
      This matters: the schedule rotates factorial variants across rounds so
      temporal/provider drift cannot confound the 2x2 contrasts, and a pause
      leaves an approximately balanced prefix — coarser regroupings (e.g. one
      lane per batch dir) would collapse that rotation. Voter invocations are
      stateless one-shot subprocesses, so ordering never affects an individual
      ballot, only these aggregate properties.
    - The wave timeout breaker counts consecutive timed-out packs per provider
      in COMPLETION order — with the dispatcher this stays close to schedule
      order. A success from any concurrent pack resets the provider's count
      (proof of life, deliberately): the breaker targets a provider that is
      DOWN, which under parallelism still trips within about
      ``group_workers + threshold`` completions.
    - On the first failure (provider-down, quota symptom, ballot validation),
      a shared stop event makes every worker halt before claiming another
      pack. Only packs already in flight run to completion (their partials
      flush for resume) before the error re-raises as the wave-level stop.
    """
    streak_lock = threading.Lock()
    consecutive_timeouts: dict[str, int] = {}
    stop = threading.Event()

    dispatch = threading.Condition()
    claimed: set[int] = set()
    busy_dirs: set[Path] = set()
    rows = list(enumerate(schedule, start=1))

    def _locked_record_streaks(votes: Any, group_id: str) -> None:
        with streak_lock:
            _record_wave_timeout_streaks(votes, consecutive_timeouts, group_id)

    def _claim_next() -> tuple[int, dict] | None:
        """Claim the first runnable schedule row, or None when the wave is over.

        Blocks (with a poll interval, so a signal-handler pause is noticed)
        while unclaimed rows exist but all of their dirs are busy.
        """
        with dispatch:
            while True:
                if stop.is_set() or pause.is_set():
                    return None
                runnable = next(
                    (
                        (position, row)
                        for position, row in rows
                        if position not in claimed and Path(row["batch_dir"]) not in busy_dirs
                    ),
                    None,
                )
                if runnable is not None:
                    claimed.add(runnable[0])
                    busy_dirs.add(Path(runnable[1]["batch_dir"]))
                    return runnable
                if len(claimed) == len(rows):
                    return None
                dispatch.wait(timeout=1.0)

    def _release(row: dict) -> None:
        with dispatch:
            busy_dirs.discard(Path(row["batch_dir"]))
            dispatch.notify_all()

    def _work() -> None:
        while True:
            runnable = _claim_next()
            if runnable is None:
                return
            position, row = runnable
            try:
                _run_scheduled_pack(
                    position,
                    len(schedule),
                    row,
                    panel,
                    timeout=timeout,
                    invocation_budget=invocation_budget,
                    record_streaks=_locked_record_streaks,
                )
            except BaseException:
                stop.set()
                raise
            finally:
                _release(row)

    with ThreadPoolExecutor(max_workers=group_workers) as ex:
        futures = [ex.submit(_work) for _ in range(group_workers)]
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
    pause: threading.Event | None = None,
) -> bool:
    """Run the wave schedule; return True if it was PAUSED before completing.

    ``pause`` is a cooperative drain switch (set by the signal handler in
    ``main``, or by a caller/test): once set, no new pack starts — in-flight
    packs finish and flush their partials — and final consolidation is
    SKIPPED, because consolidating a partially-voted batch dir would invoke
    the panel on its missing groups rather than replaying from partials.
    Rerunning the same command resumes from the partials and consolidates at
    the end of the completed run.
    """
    pause = pause if pause is not None else threading.Event()
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
            pause=pause,
        )
    else:
        consecutive_timeouts: dict[str, int] = {}

        def _record_streaks(votes: Any, group_id: str) -> None:
            _record_wave_timeout_streaks(votes, consecutive_timeouts, group_id)

        for position, row in enumerate(schedule, start=1):
            if pause.is_set():
                break
            _run_scheduled_pack(
                position,
                len(schedule),
                row,
                panel,
                timeout=timeout,
                invocation_budget=invocation_budget,
                record_streaks=_record_streaks,
            )

    if pause.is_set():
        print(
            "Wave paused: in-flight packs flushed to partial files; rerun the same "
            "command to resume and consolidate.",
            flush=True,
        )
        return True

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
    return False


def _install_pause_handler(pause: threading.Event) -> None:
    """Make SIGINT/SIGTERM request a graceful pause instead of an abort.

    First signal: set ``pause`` so lanes drain (in-flight packs finish and
    flush; nothing new starts). Signals arriving within the next 2 seconds are
    treated as DUPLICATES of the first — ``uv run`` forwards its own copy of a
    process-group signal to the child python, so a single Ctrl-C or ``pkill -f``
    otherwise lands twice and the second would abort the drain (observed on the
    2026-07-16 wave: the abort orphaned in-flight provider subprocesses, whose
    retry loops kept spending quota). A deliberate second signal after the
    debounce window force-aborts via KeyboardInterrupt.
    """
    state = {"first_signal_monotonic": None}

    def _handle(signum: int, frame: Any) -> None:
        now = time.monotonic()
        first = state["first_signal_monotonic"]
        if first is not None and now - first > 2.0:
            raise KeyboardInterrupt
        if first is None:
            state["first_signal_monotonic"] = now
            pause.set()
            print(
                "pause requested: finishing in-flight packs, then stopping "
                "(send the signal again after 2s to abort immediately)",
                flush=True,
            )

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


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
            "Run up to N scheduled packs concurrently. Workers claim rows in "
            "manifest order, deferring a row while its batch dir has a pack in "
            "flight (protects the partial-CSV read-modify-write), so parallelism "
            "requires the window to span multiple batch dirs. Each pack fans its "
            "panel seats out in parallel, so provider concurrency is N per seat — "
            "quota/rate-limit halts become more likely at higher N (the wave "
            "halts safely and resumes). Default 1 preserves the exact schedule "
            "order."
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

    pause = threading.Event()
    _install_pause_handler(pause)
    paused = execute_schedule(
        manifest,
        panel,
        timeout=args.timeout,
        invocation_budget=args.invocation_budget,
        retry_timeouts=args.retry_timeouts,
        group_workers=args.group_workers,
        pause=pause,
    )
    if paused:
        sys.exit(130)


if __name__ == "__main__":
    main()
