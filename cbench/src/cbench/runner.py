"""Shared run logic for cbench — single-dataset execution."""

from __future__ import annotations

import resource
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from cbench.adapters.base import ToolAdapter
from cbench.eval.labels import load_labels
from cbench.eval.metrics import EvalResult, evaluate
from cbench.results.store import BenchmarkResult, create_result, save_result


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
    return usage.ru_utime, usage.ru_stime, usage.ru_maxrss / 1024  # KB -> MB on Linux


@dataclass
class RunResult:
    """Result from a single benchmark run."""

    tool: str
    dataset: str
    eval_result: EvalResult
    bench_result: BenchmarkResult
    resource_stats: ResourceStats | None = None
    metadata: dict = field(default_factory=dict)


def run_single(
    adapter: ToolAdapter,
    dataset: str,
    reference: Path,
    target: Path,
    labels_dir: Path,
    output_dir: Path,
    results_file: Path | None = None,
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

    output_path = adapter.run(reference, target, tool_output_dir, **kwargs)

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

    logger.info("Evaluating against ground truth...")
    ground_truth = load_labels(labels_dir, dataset)
    eval_result = evaluate(tool_output.matches, ground_truth)

    metrics = eval_result.to_dict()
    metrics.update(resource_stats.to_dict())

    bench_result = create_result(
        tool=adapter.name,
        dataset=dataset,
        metrics=metrics,
        metadata=tool_output.metadata,
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
        metadata=tool_output.metadata,
    )
