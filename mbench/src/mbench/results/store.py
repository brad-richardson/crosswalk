"""JSONL results storage and comparison."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

METRIC_SCHEMA_VERSION = 2


@dataclass
class BenchmarkResult:
    """A single benchmark run result."""

    tool: str
    dataset: str
    timestamp: str
    metrics: dict
    metadata: dict = field(default_factory=dict)
    # Defaults identify historical records that predate explicit view tracking.
    metric_schema_version: int = 1
    prediction_view: str = "legacy_combined"

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, line: str) -> BenchmarkResult:
        data = json.loads(line)
        return cls(**data)


def save_result(result: BenchmarkResult, path: Path) -> None:
    """Append a benchmark result to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(result.to_json() + "\n")


def load_results(path: Path) -> list[BenchmarkResult]:
    """Load all benchmark results from a JSONL file."""
    if not path.exists():
        return []
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(BenchmarkResult.from_json(line))
    return results


def create_result(
    tool: str,
    dataset: str,
    metrics: dict,
    metadata: dict | None = None,
    prediction_view: str = "combined",
) -> BenchmarkResult:
    """Create a new BenchmarkResult with current timestamp."""
    return BenchmarkResult(
        tool=tool,
        dataset=dataset,
        timestamp=datetime.now(UTC).isoformat(),
        metrics=metrics,
        metadata=metadata or {},
        metric_schema_version=METRIC_SCHEMA_VERSION,
        prediction_view=prediction_view,
    )


def compare_results(results: list[BenchmarkResult], console: Console | None = None) -> Table:
    """Build a Rich table comparing benchmark results.

    Args:
        results: List of benchmark results to compare.
        console: Optional console to print to. If None, just returns the table.

    Returns:
        Rich Table object.
    """
    table = Table(title="Benchmark Comparison")
    table.add_column("Tool", style="cyan")
    table.add_column("Dataset", style="green")
    table.add_column("Level")
    table.add_column("View")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right", style="bold")
    table.add_column("TP", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("FN", justify="right")
    table.add_column("Unlabeled", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Peak RSS", justify="right")
    table.add_column("Timestamp")

    for r in results:
        m = r.metrics
        wall = m.get("wall_time_s")
        time_str = f"{wall:.1f}s" if wall is not None else "-"
        rss = m.get("peak_rss_mb")
        rss_str = f"{rss:.0f} MB" if rss is not None else "-"
        coverage = m.get("labeled_coverage")
        coverage_str = f"{coverage:.4f}" if coverage is not None else "-"
        table.add_row(
            r.tool,
            r.dataset,
            m.get("match_level", "-"),
            r.prediction_view,
            f"{m.get('precision', 0):.4f}",
            f"{m.get('recall', 0):.4f}",
            f"{m.get('f1', 0):.4f}",
            str(m.get("true_positives", 0)),
            str(m.get("false_positives", 0)),
            str(m.get("false_negatives", 0)),
            str(m.get("unlabeled_predictions", 0)),
            coverage_str,
            time_str,
            rss_str,
            r.timestamp[:19],
        )

    if console is not None:
        console.print(table)
    return table
