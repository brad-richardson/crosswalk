#!/usr/bin/env python
"""Evaluate optimizer prune heuristics without changing production behavior.

The harness discovers curated stitching labels, prefers fresh ``data/output``
sidecars, and falls back to the newest factory snapshot per dataset.  It emits
two reports:

* ``coverage.csv``: which labels and positive edges were actually recoverable;
* ``metrics.csv``: fixed-universe policy metrics by provenance, match type, and
  dataset, including deltas from the persisted production selection.

Coverage is a first-class result: never compare grouping/glue runs on F1 alone
when their recovered label or candidate universes differ.
"""

from __future__ import annotations

import argparse
import json
from contextlib import suppress
from pathlib import Path

import pandas as pd

from crosswalk.resolver.extract import (
    build_edge_table,
    load_sidecar_groups,
    load_stitching_labels,
)
from crosswalk.resolver.heuristic_ablation import PrunePolicy, evaluate_policies

HANDOFF_THRESHOLDS = {"1:N": 0.92, "N:1": 0.94, "M:N": 0.96}


def _human_positive_count(labels: pd.DataFrame) -> int:
    total = 0
    for raw in labels["selected_edges"]:
        with suppress(TypeError, ValueError):
            total += len(json.loads(raw))
    return total


def _find_sidecar(data_root: Path, dataset: str) -> tuple[Path | None, str]:
    fresh = data_root / "data" / "output" / f"{dataset}_groups.json"
    if fresh.exists():
        return fresh, "fresh"
    snapshots = sorted(
        (data_root / "data" / "factory").glob(f"release=*/dataset={dataset}/groups.json"),
        reverse=True,
    )
    if snapshots:
        return snapshots[0], "factory"
    return None, "missing"


def _load_inputs(data_root: Path, datasets: list[str] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels_root = data_root / "labels" / "stitching"
    label_paths = sorted(labels_root.glob("dataset=*/data.csv"))
    if datasets:
        wanted = set(datasets)
        label_paths = [p for p in label_paths if p.parent.name.removeprefix("dataset=") in wanted]

    frames: list[pd.DataFrame] = []
    coverage: list[dict[str, object]] = []
    for labels_path in label_paths:
        dataset = labels_path.parent.name.removeprefix("dataset=")
        labels = load_stitching_labels(labels_path)
        sidecar, source_kind = _find_sidecar(data_root, dataset)
        row: dict[str, object] = {
            "dataset": dataset,
            "sidecar_kind": source_kind,
            "sidecar": "" if sidecar is None else str(sidecar.relative_to(data_root)),
            "label_rows": len(labels),
            "human_positive_edges": _human_positive_count(labels),
            "mapped_label_groups": 0,
            "edge_rows": 0,
            "represented_positive_rows": 0,
        }
        if sidecar is None:
            coverage.append(row)
            continue

        table = build_edge_table(load_sidecar_groups(sidecar), labels, dataset)
        stats = table.attrs.get("build_stats", {})
        row.update(
            {
                "mapped_label_groups": int(table["human_group_id"].nunique()) if len(table) else 0,
                "edge_rows": len(table),
                "represented_positive_rows": int(table["keep"].sum()) if len(table) else 0,
                "candidate_groups": stats.get("candidate_groups", 0),
                "legacy_groups": stats.get("legacy_groups", 0),
                "human_selected_outside_candidate_graph": stats.get(
                    "human_selected_outside_candidate_graph", 0
                ),
                "human_selected_outside_candidate_graph_clean": stats.get(
                    "human_selected_outside_candidate_graph_clean", 0
                ),
                "human_selected_outside_candidate_graph_split": stats.get(
                    "human_selected_outside_candidate_graph_split", 0
                ),
                "empty_unrecovered": stats.get("empty_unrecovered", 0),
                "empty_legacy_skipped": stats.get("empty_legacy_skipped", 0),
            }
        )
        coverage.append(row)
        if len(table):
            frames.append(table)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, pd.DataFrame(coverage)


def _policies(thresholds: list[float], margins: list[float]) -> list[PrunePolicy]:
    policies = [PrunePolicy("preprune")]
    policies.extend(PrunePolicy(f"scalar_{value:.2f}", threshold=value) for value in thresholds)
    policies.extend(PrunePolicy(f"margin_{value:.2f}", margin=value) for value in margins)
    policies.extend(
        [
            PrunePolicy("handoff_per_type", per_type_thresholds=HANDOFF_THRESHOLDS),
            PrunePolicy(
                "handoff_per_type_margin_0.05",
                per_type_thresholds=HANDOFF_THRESHOLDS,
                margin=0.05,
            ),
            PrunePolicy(
                "handoff_per_type_margin_0.05_bridge_guard",
                per_type_thresholds=HANDOFF_THRESHOLDS,
                margin=0.05,
                preserve_bridge_backbone=True,
            ),
        ]
    )
    return policies


def _write_summary(path: Path, coverage: pd.DataFrame, metrics: pd.DataFrame) -> None:
    pooled = metrics[metrics["slice"].isin(["all", "provenance=clean", "provenance=split"])]
    best = (
        pooled.sort_values(["slice", "f1", "group_exact"], ascending=[True, False, False])
        .groupby("slice", sort=False)
        .head(1)
    )
    missing = coverage[coverage["sidecar_kind"] == "missing"]["dataset"].tolist()
    fresh = int((coverage["sidecar_kind"] == "fresh").sum())
    mapped = int(coverage["mapped_label_groups"].sum())
    labels = int(coverage["label_rows"].sum())
    summary_columns = [
        "slice",
        "policy",
        "edges",
        "groups",
        "f1",
        "group_exact",
        "f1_delta",
        "group_exact_delta",
    ]
    table_lines = [
        "| " + " | ".join(summary_columns) + " |",
        "| " + " | ".join("---" for _ in summary_columns) + " |",
    ]
    for _, row in best.iterrows():
        values = []
        for column in summary_columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        table_lines.append("| " + " | ".join(values) + " |")
    lines = [
        "# Optimizer heuristic ablation",
        "",
        "All policy metrics below use one fixed recovered edge table. Grouping/glue runs must",
        "match its coverage before their scores are compared; otherwise label attrition can look",
        "like an optimizer improvement.",
        "",
        f"- Fresh sidecars: {fresh}/{len(coverage)} datasets",
        f"- Mapped label groups: {mapped}/{labels} label rows",
        f"- Missing sidecars: {', '.join(missing) if missing else 'none'}",
        "",
        "## Best observed fixed-universe policy by primary slice",
        "",
        *table_lines,
        "",
        "See `coverage.csv` and `metrics.csv` for the auditable full results.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--threshold", action="append", type=float, dest="thresholds")
    parser.add_argument("--margin", action="append", type=float, dest="margins")
    parser.add_argument("--output", type=Path, default=Path("/tmp/optimizer-heuristic-ablation"))
    args = parser.parse_args()

    thresholds = args.thresholds or [0.90, 0.92, 0.94, 0.96, 0.98]
    margins = args.margins or [0.02, 0.05, 0.08]
    table, coverage = _load_inputs(args.data_root.resolve(), args.datasets)
    if table.empty:
        raise SystemExit("No labeled edges were recovered; inspect coverage inputs.")
    metrics = evaluate_policies(table, _policies(thresholds, margins))

    args.output.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.output / "coverage.csv", index=False)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    _write_summary(args.output / "README.md", coverage, metrics)

    print(coverage.to_string(index=False))
    print("\nPooled metrics:")
    print(metrics[metrics["slice"] == "all"].to_string(index=False, float_format="%.4f"))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
