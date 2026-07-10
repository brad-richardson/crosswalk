"""Shared run logic for mbench — single-dataset execution."""

from __future__ import annotations

import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from mbench.adapters.base import ToolAdapter
from mbench.eval.labels import load_labels, load_stitch_labels
from mbench.eval.metrics import EvalResult, evaluate
from mbench.eval.stitch_metrics import StitchEvalResult, evaluate_stitch_groups
from mbench.provenance import collect_provenance
from mbench.results.store import BenchmarkResult, create_result, save_result


@dataclass
class ResourceStats:
    """Resource usage from a tool run."""

    wall_time_s: float
    cpu_user_s: float
    cpu_system_s: float
    peak_rss_mb: float

    def to_dict(self) -> dict:
        return {
            "wall_time_s": round(self.wall_time_s, 2),
            "cpu_user_s": round(self.cpu_user_s, 2),
            "cpu_system_s": round(self.cpu_system_s, 2),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
        }


def _measure_children() -> tuple[float, float, float]:
    """Snapshot child-process CPU time and peak RSS."""
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    # ru_maxrss is KB on Linux, bytes on macOS/BSD
    rss_divisor = 1024 if sys.platform == "linux" else 1024 * 1024
    return usage.ru_utime, usage.ru_stime, usage.ru_maxrss / rss_divisor


def _resolve_stitch_dir(stitch_labels_dir: Path | None, labels_dir: Path) -> Path | None:
    """Resolve the stitch labels directory, defaulting when not given.

    Stitch eval is default-on: when ``--stitch-labels`` / a config default is not
    supplied, fall back to the ``stitching`` directory that sits next to the
    labels dir (e.g. ``labels/human`` -> ``labels/stitching``). Returns None only
    when no candidate directory exists.
    """
    if stitch_labels_dir is not None:
        return stitch_labels_dir
    default = labels_dir.parent / "stitching"
    return default if default.exists() else None


def _maybe_evaluate_stitch(
    dataset: str,
    tool_output,
    stitch_labels_dir: Path | None,
    labels_dir: Path,
) -> StitchEvalResult | None:
    """Run stitch-level eval if labels exist. Never raises (non-blocking).

    Stitch-level eval is decision-AGNOSTIC: it scores the optimizer's edge
    selection within M:N groups, which happens upstream of the match/review
    publication decision, and the armed gate floors were calibrated on the full
    selection. Filtering to the accepted view here would silently consume a
    large share of the calibrated floor margins with no matching-logic change.
    """
    try:
        resolved = _resolve_stitch_dir(stitch_labels_dir, labels_dir)
        if resolved is None:
            logger.info("Stitch eval skipped: no stitching labels directory found")
            return None
        stitch_labels = load_stitch_labels(resolved, dataset)
        if stitch_labels is None:
            logger.info(f"Stitch eval skipped: no stitching labels for '{dataset}'")
            return None
        groups = getattr(tool_output, "groups", None)
        if not groups:
            logger.info("Stitch eval: no groups sidecar; using legacy segment-id mapping")
        result = evaluate_stitch_groups(tool_output.matches, stitch_labels, groups=groups)
        logger.info(
            f"Stitch eval: P={result.precision:.3f} R={result.recall:.3f} "
            f"F1={result.f1:.3f} exact={result.exact_match_rate:.3f} "
            f"({result.groups_evaluated} groups; "
            f"filtered F1={result.f1_filtered:.3f}, "
            f"{result.groups_sliver_affected} sliver-affected)"
        )
        if result.metrics_by_labeler:
            for cls, m in result.metrics_by_labeler.items():
                logger.info(
                    f"  [{cls}] n={m['n']} F1={m['f1']:.3f} exact={m['exact_match_rate']:.3f}"
                )
        return result
    except Exception as exc:  # non-blocking: never fail a run on stitch eval
        logger.warning(f"Stitch eval failed (non-blocking, skipping): {exc}")
        return None


@dataclass
class RunResult:
    """Result from a single benchmark run."""

    tool: str
    dataset: str
    eval_result: EvalResult
    bench_result: BenchmarkResult
    resource_stats: ResourceStats | None = None
    stitch_result: StitchEvalResult | None = None
    decision_results: dict[str, EvalResult] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


ALLOWED_MATCH_DECISIONS = frozenset({"match", "review", "no_match"})


def _decision_views(matches, *, decision_aware: bool = False) -> tuple[object, dict[str, object]]:
    """Return the headline predictions and optional decision-aware views.

    Decision-aware evaluation is an adapter CAPABILITY, not a column sniff:
    adapters that do not declare ``decision_aware`` keep their historical
    combined behavior even when their output happens to carry a
    ``match_decision`` column (base.py promises third-party adapters remain
    compatible). For decision-aware adapters, only explicit ``match`` rows form
    the production headline; ``review`` and the combined proposal queue remain
    visible.
    """
    if not decision_aware:
        return matches, {}
    if "match_decision" not in matches.columns:
        raise ValueError("decision-aware adapter output missing match_decision column")

    raw_decisions = matches["match_decision"]
    if raw_decisions.isna().any():
        raise ValueError("match_decision contains null values")
    decisions = raw_decisions.astype("string").str.lower().str.strip()
    unknown = sorted(set(decisions) - ALLOWED_MATCH_DECISIONS)
    if unknown:
        raise ValueError(f"match_decision contains unknown values: {unknown}")
    accepted = matches[decisions == "match"]
    review = matches[decisions == "review"]
    # Explicit no_match rows are intentionally excluded from every prediction
    # view; they are decisions not to publish/propose an edge.
    proposal = matches[decisions.isin(["match", "review"])]
    return accepted, {"accepted": accepted, "review": review, "proposal": proposal}


def run_single(
    adapter: ToolAdapter,
    dataset: str,
    reference: Path,
    target: Path,
    labels_dir: Path,
    output_dir: Path,
    results_file: Path | None = None,
    stitch_labels_dir: Path | None = None,
    match_level: str = "target",
    **kwargs,
) -> RunResult:
    """Run a tool on a single dataset and evaluate against ground truth.

    Args:
        adapter: Tool adapter instance.
        dataset: Dataset name (must have labels).
        reference: Path to reference parquet file.
        target: Path to target parquet file.
        labels_dir: Path to labels directory.
        output_dir: Output directory for tool artifacts.
        results_file: If set, append result to this JSONL file.
        stitch_labels_dir: If set, path to stitching labels directory for
            group-level evaluation.
        match_level: Evaluation level, "target" (default) or "pair".
        **kwargs: Tool-specific options passed to adapter.run().

    Returns:
        RunResult with evaluation metrics and benchmark result.

    Raises:
        FileNotFoundError: If reference, target, or labels not found.
        RuntimeError: If the tool fails to run.
    """
    for path, desc in [
        (reference, "Reference file"),
        (target, "Target file"),
        (labels_dir, "Labels directory"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{desc} not found: {path}")

    tool_output_dir = output_dir / adapter.name / dataset

    # Measure resource usage around tool execution
    logger.info(f"Running {adapter.name} on {dataset}...")
    cpu_user_before, cpu_sys_before, _ = _measure_children()
    t0 = time.monotonic()

    # Pass the dataset identity through to the adapter. Some tools key behavior
    # on the dataset NAME rather than the file paths — crosswalk's resolver-prune
    # allowlist has been dataset-identity-keyed since #350, so a path-only stitch
    # invocation runs with the prune OFF and evaluates a different (unpruned) row
    # set than production, ~5pt below the calibrated gate floor (#372). ``dataset``
    # is a named run_single parameter, so it never collides with ``**kwargs``;
    # setdefault is just defensive. Adapters that don't read it ignore it.
    run_kwargs = dict(kwargs)
    run_kwargs.setdefault("dataset", dataset)
    output_path = adapter.run(reference, target, tool_output_dir, **run_kwargs)

    wall_time = time.monotonic() - t0
    cpu_user_after, cpu_sys_after, peak_rss_mb = _measure_children()
    resource_stats = ResourceStats(
        wall_time_s=wall_time,
        cpu_user_s=cpu_user_after - cpu_user_before,
        cpu_system_s=cpu_sys_after - cpu_sys_before,
        peak_rss_mb=peak_rss_mb,
    )
    logger.info(
        f"Completed in {wall_time:.1f}s "
        f"(CPU: {resource_stats.cpu_user_s:.1f}u+{resource_stats.cpu_system_s:.1f}s, "
        f"peak RSS: {resource_stats.peak_rss_mb:.0f} MB)"
    )

    logger.info("Parsing output...")
    tool_output = adapter.parse_output(output_path)
    logger.info(f"Found {len(tool_output.matches)} match predictions")

    headline_predictions, decision_views = _decision_views(
        tool_output.matches,
        decision_aware=bool(getattr(adapter, "decision_aware", False)),
    )
    if decision_views:
        logger.info(
            "Decision-aware output: "
            f"{len(decision_views['accepted'])} accepted, "
            f"{len(decision_views['review'])} review, "
            f"{len(decision_views['proposal'])} proposed"
        )

    logger.info(f"Evaluating against ground truth (match_level={match_level})...")
    ground_truth = load_labels(labels_dir, dataset)
    eval_result = evaluate(headline_predictions, ground_truth, match_level=match_level)

    metrics = eval_result.to_dict()
    metrics.update(resource_stats.to_dict())
    decision_results: dict[str, EvalResult] = {}
    if decision_views:
        decision_results = {
            name: evaluate(view, ground_truth, match_level=match_level)
            for name, view in decision_views.items()
        }
        # Counts derive from the already-validated views; no_match is whatever
        # was excluded from the proposal (match + review) queue.
        no_match_count = len(tool_output.matches) - len(decision_views["proposal"])
        metrics["decision_metrics"] = {
            "available": True,
            "headline": "accepted",
            "counts": {
                "match": len(decision_views["accepted"]),
                "review": len(decision_views["review"]),
                "no_match": no_match_count,
            },
            "excluded_no_match_count": no_match_count,
            **{name: result.to_dict() for name, result in decision_results.items()},
        }

    # Stitch-level evaluation — NON-BLOCKING and default-on. If stitch labels
    # exist for this dataset they are computed and reported alongside the pair
    # metrics; otherwise skipped silently. Any error here is logged and swallowed
    # so it can never fail a benchmark run.
    stitch_result = _maybe_evaluate_stitch(
        dataset=dataset,
        tool_output=tool_output,
        stitch_labels_dir=stitch_labels_dir,
        labels_dir=labels_dir,
    )
    if stitch_result is not None:
        metrics.update(stitch_result.to_dict())

    resolved_stitch_dir = _resolve_stitch_dir(stitch_labels_dir, labels_dir)
    metadata = dict(tool_output.metadata)
    try:
        metadata["provenance"] = collect_provenance(
            tool=adapter.name,
            dataset=dataset,
            reference=reference,
            target=target,
            labels_dir=labels_dir,
            stitch_labels_dir=resolved_stitch_dir,
            match_level=match_level,
            run_options=run_kwargs,
            adapter_metadata=tool_output.metadata,
        )
    except Exception as exc:  # best-effort: never discard a completed run
        logger.warning(f"Provenance collection failed (non-blocking): {exc}")
        metadata["provenance"] = {"schema_version": 1, "error": str(exc)}

    bench_result = create_result(
        tool=adapter.name,
        dataset=dataset,
        metrics=metrics,
        metadata=metadata,
        prediction_view="accepted" if decision_views else "combined",
    )

    if results_file is not None:
        save_result(bench_result, results_file)
        logger.info(f"Result saved to {results_file}")

    return RunResult(
        tool=adapter.name,
        dataset=dataset,
        eval_result=eval_result,
        bench_result=bench_result,
        resource_stats=resource_stats,
        stitch_result=stitch_result,
        decision_results=decision_results,
        metadata=metadata,
    )
