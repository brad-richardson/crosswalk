#!/usr/bin/env python3
"""Regenerate the durable stitching collision/omission audit snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from crosswalk.resolver.extract import (
    build_edge_table,
    load_sidecar_groups,
    load_stitching_labels,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "research/stitching_deferred_audit.json"
SOURCE_COMMIT = "9adfdefa7a5893dea207bbdd17db828013d6644f"
SOURCE_TREE = "00418f8e72b8f60409bd2ad79cf12f5b98de40ff"

INPUTS = {
    "us_boston_streets": {
        "groups": {
            "path": "data/output/us_boston_streets_groups.json",
            "sha256": "5eaeadc4ac7ad1d0d061c14c58fbdecbe501ce07144ff3686ac5a77218807f28",
        },
        "labels": {
            "path": "labels/stitching/dataset=us_boston_streets/data.csv",
            "sha256": "16014f8b5734932ffa29fcb7ba447e397f872d1cd3a99dab4b3f604af5fbf470",
        },
    },
    "us_seattle_sidewalks": {
        "groups": {
            "path": "data/output/us_seattle_sidewalks_groups.json",
            "sha256": "6310d7c4180b0d212d67eae757c57a8d4dbb290847d2c9a1017c3448d8983b00",
        },
        "labels": {
            "path": "labels/stitching/dataset=us_seattle_sidewalks/data.csv",
            "sha256": "c7dbd3bc8fb67ccef2ab480594b5202dea4228abf4dc00ebff9115acfdc0593b",
        },
    },
}

BUILD_INVOCATION = {
    "include_split": True,
    "include_rejected": True,
    "prefer_candidate_graph": True,
    "filter_rule5": True,
    "include_empty": True,
    "candidates": None,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_snapshot() -> dict:
    datasets = {}
    for dataset, inputs in INPUTS.items():
        resolved = {kind: PROJECT_ROOT / item["path"] for kind, item in inputs.items()}
        for kind, path in resolved.items():
            actual = _sha256(path)
            expected = inputs[kind]["sha256"]
            if actual != expected:
                raise ValueError(
                    f"{dataset} {kind} hash changed: expected {expected}, got {actual}; "
                    "adjudicate the new source snapshot before updating this audit"
                )

        groups = load_sidecar_groups(resolved["groups"])
        labels = load_stitching_labels(resolved["labels"])
        edge_table = build_edge_table(
            groups,
            labels,
            dataset,
            include_split=True,
            include_rejected=True,
            prefer_candidate_graph=True,
            filter_rule5=True,
            include_empty=True,
        )
        datasets[dataset] = {
            "inputs": inputs,
            "stats": edge_table.attrs["build_stats"],
            "audit": edge_table.attrs["build_audit"],
        }

    return {
        "schema_version": 1,
        "artifact_source": {"commit": SOURCE_COMMIT, "tree": SOURCE_TREE},
        "build_invocation": BUILD_INVOCATION,
        "datasets": datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the snapshot is stale")
    args = parser.parse_args()
    rendered = json.dumps(build_snapshot(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != rendered:
            raise SystemExit(f"stale deferred audit: regenerate {OUTPUT}")
        return
    OUTPUT.write_text(rendered)


if __name__ == "__main__":
    main()
