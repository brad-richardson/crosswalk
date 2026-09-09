#!/usr/bin/env python
"""Freeze an audited resolver benchmark from existing human labels only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from crosswalk.agent_labeling.stitch_provenance import sha256_file
from crosswalk.resolver.evaluation_set import (
    EVALUATION_SCHEMA_VERSION,
    audit_existing_labels,
    freeze_partitions,
)
from crosswalk.resolver.extract import (
    discover_candidates_parquet,
    load_candidates_parquet,
    load_sidecar_groups,
    load_stitching_labels,
)
from crosswalk.resolver.features import RESOLVER_FEATURE_VERSION
from crosswalk.resolver.round2 import EXTENDED_FEATURE_COLUMNS, featurize_extended
from crosswalk.resolver.train import _discover_specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument(
        "--exclusions", type=Path, default=Path("research/stitch_evaluation_exclusions.json")
    )
    parser.add_argument(
        "--sources", type=Path, help="Optional explicit dataset/groups/labels path manifest"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.out / "manifest.json").exists():
        raise ValueError(
            "Evaluation is already frozen; use a new output directory for a new version"
        )
    policy = json.loads(args.exclusions.read_text())
    exclusions = {(e["dataset_id"], e["group_id"]): e["reason"] for e in policy["exclusions"]}
    specs = (
        [
            (s["dataset_id"], Path(s["groups"]), Path(s["labels"]))
            for s in json.loads(args.sources.read_text())
        ]
        if args.sources
        else _discover_specs(args.root, args.root / "labels/stitching")
    )
    tables, audit, sources = [], [], []
    inputs = {str(p): sha256_file(p) for p in [args.observations, args.exclusions]}
    if args.sources:
        inputs[str(args.sources)] = sha256_file(args.sources)
    for source in policy["exclusions"]:
        path = args.root / source["source"]
        inputs[str(path)] = sha256_file(path)
    for dataset, group_path, label_path in specs:
        labels = load_stitching_labels(label_path)
        if not (labels["labeler"] == "brad").any():
            continue
        inputs[str(label_path)] = sha256_file(label_path)
        groups = load_sidecar_groups(group_path) if group_path.exists() else []
        candidates_path = discover_candidates_parquet(group_path) if groups else None
        candidates = load_candidates_parquet(candidates_path) if candidates_path else None
        for path in [group_path, candidates_path]:
            if path and path.exists():
                inputs[str(path)] = sha256_file(path)
        table, records = audit_existing_labels(
            dataset, groups, labels, exclusions=exclusions, candidates_df=candidates
        )
        audit.extend(records)
        sources.append(
            {
                "dataset_id": dataset,
                "groups": str(group_path),
                "labels": str(label_path),
                "candidates": str(candidates_path or ""),
            }
        )
        if len(table):
            tables.append(featurize_extended(table))
        print(
            f"{dataset}: {len(table)} eligible edges; {len(records)} human records audited",
            flush=True,
        )
        del groups, candidates
    if not tables:
        raise ValueError("No existing exact human labels pass the audit")
    gold, weak, components = freeze_partitions(
        pd.concat(tables, ignore_index=True), pd.read_parquet(args.observations), audit
    )
    args.out.mkdir(parents=True, exist_ok=True)
    gold.to_parquet(args.out / "gold.parquet", index=False)
    weak.to_parquet(args.out / "weak_partition.parquet", index=False)
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (args.out / "components.json").write_text(
        json.dumps(components, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "feature_version": RESOLVER_FEATURE_VERSION,
        "inputs": inputs,
        "sources": sources,
        "outputs": {
            name: sha256_file(args.out / name)
            for name in ["gold.parquet", "weak_partition.parquet", "audit.json", "components.json"]
        },
        "human_audit": pd.Series([r["status"] for r in audit]).value_counts().to_dict(),
        "gold": {
            "rows": len(gold),
            "positives": int(gold["keep"].sum()),
            "groups": len(gold[["dataset_id", "group_id"]].drop_duplicates()),
            "components": gold["component_id"].nunique(),
            "deanchored_groups": len(
                gold[~gold["anchored"]][["dataset_id", "group_id"]].drop_duplicates()
            ),
            "datasets": gold.groupby("dataset_id").size().to_dict(),
            "fold_edges": gold.groupby("fold").size().to_dict(),
        },
        "weak": {
            "rows": len(weak),
            "excluded_human_scope": int(weak["excluded_human_scope"].sum()),
            "remaining_tiers": weak[~weak["excluded_human_scope"]]["vote_tier"]
            .value_counts()
            .to_dict(),
        },
        "experiment": {
            "features": EXTENDED_FEATURE_COLUMNS,
            "seeds": [17, 29, 43],
            "selector": "ef1",
            "objective": "reg:logistic",
            "policies": ["human_only", "human_unanimous", "human_majority"],
            "unanimous_weight": 0.35,
            "majority_weight": 0.15,
            "weak_component_weight_cap": 4.0,
            "hyperparameter_search": False,
            "promotion": False,
        },
        "limitations": [
            "Existing exact labels remain mostly optimizer-anchored and geographically concentrated; this is an exploratory paired resolver comparison.",
            "Pair scores are fixed inputs from the existing matcher, not an independently held-out end-to-end matcher evaluation.",
            "Identity exclusions use recorded parent lineage and segment IDs. Unrecorded simultaneous rekeys of both endpoints cannot be reconstructed.",
            "The historical human UI did not hash its candidate universe. Unchanged group identity or explicit membership coverage plus complete retained positives is the available scope check.",
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in ["human_audit", "gold", "weak"]}, indent=2))


if __name__ == "__main__":
    main()
