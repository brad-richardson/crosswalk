#!/usr/bin/env python
"""Recover scoped weak observations from existing votes; never export human labels.

uv run python scripts/recover_stitch_observations.py --out data/experiments/stitch-supervision/batch1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crosswalk.agent_labeling.stitch_provenance import sha256_file
from crosswalk.resolver.observations import (
    OBSERVATION_SCHEMA_VERSION,
    build_vote_observations,
    reconcile_observations,
)
from crosswalk.resolver.votes import (
    default_evidence_paths,
    default_votes_paths,
    load_archived_label_maps,
    load_evidence,
    load_votes,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--votes-root", type=Path, default=Path("labels/votes"))
    parser.add_argument("--batches-root", type=Path, default=Path("data/agents/stitching/batches"))
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = default_votes_paths(args.votes_root)
    evidence_paths = default_evidence_paths(args.votes_root)
    if args.dataset:
        paths = [p for p in paths if p.parent.name.removeprefix("dataset=") in args.dataset]
        evidence_paths = [
            p for p in evidence_paths if p.parent.name.removeprefix("dataset=") in args.dataset
        ]
    votes = load_votes(paths)
    evidence = load_evidence(evidence_paths)
    label_maps = load_archived_label_maps(votes, args.batches_root)
    observations = build_vote_observations(votes, evidence, label_maps=label_maps)
    reconciled = reconcile_observations(observations)
    args.out.mkdir(parents=True, exist_ok=True)
    observations.to_parquet(args.out / "observations.parquet", index=False)
    reconciled.to_parquet(args.out / "reconciled.parquet", index=False)
    report = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "audit": reconciled.attrs.get("observation_audit", {}),
        "vote_rows": len(votes),
        "datasets": sorted(votes["dataset_id"].unique().tolist()),
        "inputs": {str(p): sha256_file(p) for p in sorted([*paths, *evidence_paths])},
        "outputs": {
            name: sha256_file(args.out / name)
            for name in ["observations.parquet", "reconciled.parquet"]
        },
        "scope": "Weak resolution observations only. Unknowns and conflicts are masked; no human labels or models are changed.",
    }
    if len(observations):
        children = observations[observations["group_id"] != observations["parent_group_id"]]
        report["child_observation_groups"] = int(children["observation_id"].nunique())
        report["child_parents"] = int(
            children[["dataset_id", "parent_group_id"]].drop_duplicates().shape[0]
        )
        report["tiers"] = reconciled["vote_tier"].value_counts().to_dict()
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ["inputs", "outputs"]}, indent=2))


if __name__ == "__main__":
    main()
