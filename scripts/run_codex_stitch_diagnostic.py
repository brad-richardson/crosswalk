#!/usr/bin/env python3
"""Plan, run, and analyze isolated Codex stitching diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from crosswalk.agent_labeling.stitch_diagnostic import (
    DEFAULT_CANONICAL_DRAWS,
    DEFAULT_WORKERS,
    analyze_diagnostic,
    build_diagnostic_plan,
    diagnostic_status,
    run_diagnostic,
    validate_diagnostic_plan,
    write_diagnostic_plan,
)


def _plan(args: argparse.Namespace) -> None:
    plan = build_diagnostic_plan(
        args.source_manifest,
        args.output,
        run_id=args.run_id,
        canonical_draws=args.canonical_draws,
        holdout_groups=args.holdout_groups,
        holdout_factorial_groups=args.holdout_factorial_groups,
        seed=args.seed,
        timeout=args.timeout,
        invocation_budget_s=args.invocation_budget,
    )
    print(
        f"Validated {len(plan['packs'])} packs / "
        f"{plan['source_manifest']['unique_group_count']} unique groups / "
        f"{plan['planned_call_count']} planned Codex calls"
    )
    print(
        f"Provider: {plan['provider']['model']}/{plan['provider']['effort']}; "
        f"smoke packs: {', '.join(plan['smoke_pack_keys'])}"
    )
    if args.validate_only:
        print("Validation only: wrote nothing")
        return
    path = write_diagnostic_plan(plan)
    print(f"Wrote diagnostic plan: {path}")
    print("Diagnostic artifacts are .no-export and separate from panel batches.")


def _run(args: argparse.Namespace) -> None:
    status = run_diagnostic(
        args.plan,
        workers=args.workers,
        resume=not args.no_resume,
        smoke=args.smoke,
        pass_ids=set(args.pass_id) if args.pass_id else None,
    )
    print(json.dumps(status, indent=2))


def _analyze(args: argparse.Namespace) -> None:
    summary = analyze_diagnostic(
        args.plan,
        labels_root=args.labels_root,
        include_holdout=args.include_holdout,
        fix_frozen_marker=args.fix_frozen,
    )
    print(
        f"Analyzed {summary['packs_analyzed']} {summary['cohort']} packs; "
        f"results {summary['results_present']}/{summary['results_planned']}"
    )


def _status(args: argparse.Namespace) -> None:
    print(json.dumps(diagnostic_status(args.plan), indent=2))


def _validate(args: argparse.Namespace) -> None:
    plan = validate_diagnostic_plan(args.plan)
    print(
        f"Validated diagnostic plan {plan['run_id']}: {len(plan['packs'])} packs, "
        f"{plan['planned_call_count']} calls"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repeated Codex draws over immutable stitching packs. This is diagnostic-only: "
            "no consensus and no label export."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="validate inputs and write a diagnostic manifest")
    plan.add_argument("source_manifest", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--canonical-draws", type=int, default=DEFAULT_CANONICAL_DRAWS)
    plan.add_argument("--holdout-groups", type=int, default=15)
    plan.add_argument("--holdout-factorial-groups", type=int, default=2)
    plan.add_argument("--seed", type=int, default=20260716)
    plan.add_argument("--timeout", type=int, default=600)
    plan.add_argument("--invocation-budget", type=float, default=300.0)
    plan.add_argument("--validate-only", action="store_true")
    plan.set_defaults(func=_plan)

    run = sub.add_parser("run", help="execute missing calls with atomic resume")
    run.add_argument("plan", type=Path)
    run.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    run.add_argument("--smoke", action="store_true", help="run the bound ordinary+factorial pair")
    run.add_argument("--pass-id", action="append", help="restrict to a pass; repeatable")
    run.add_argument("--no-resume", action="store_true")
    run.set_defaults(func=_run)

    analyze = sub.add_parser("analyze", help="aggregate development results by default")
    analyze.add_argument("plan", type=Path)
    analyze.add_argument("--labels-root", type=Path, default=Path("labels/stitching"))
    analyze.add_argument("--include-holdout", action="store_true")
    analyze.add_argument("--fix-frozen", type=Path)
    analyze.set_defaults(func=_analyze)

    status = sub.add_parser("status", help="validate and count completed result files")
    status.add_argument("plan", type=Path)
    status.set_defaults(func=_status)

    validate = sub.add_parser("validate", help="revalidate plan, runtime, packs, and results")
    validate.add_argument("plan", type=Path)
    validate.set_defaults(func=_validate)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
