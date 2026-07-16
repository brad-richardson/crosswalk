#!/usr/bin/env python
"""Build the learned-group-resolver per-edge dataset and run the prototype eval.

EXPERIMENTAL research harness — not wired into the pipeline. Reads the group
sidecars + curated stitching labels (+ optional panel votes), builds a per-edge
keep/drop table, and runs grouped-CV of the prototype classifier against the
optimizer baseline.

Example:

    uv run python scripts/build_resolver_dataset.py \
        --data-root /Users/bradrichardson/dev/matcher \
        --dataset us_boston_streets --dataset us_seattle_sidewalks \
        --out /tmp/resolver_edges.parquet --with-votes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crosswalk.resolver.evaluate import feature_importances, run_cv, slice_report
from crosswalk.resolver.extract import (
    COMBINED_AUDIT_ATTR,
    COMBINED_STATS_ATTR,
    build_multi_dataset_table,
    load_sidecar_groups,
    write_edge_table_parquet,
)
from crosswalk.resolver.features import featurize
from crosswalk.resolver.votes import (
    default_votes_paths,
    edge_soft_labels,
    load_votes,
)


def _specs(data_root: Path, datasets: list[str]) -> list[tuple[str, Path, Path]]:
    specs = []
    for ds in datasets:
        groups = data_root / "data" / "output" / f"{ds}_groups.json"
        labels = data_root / "labels" / "stitching" / f"dataset={ds}" / "data.csv"
        specs.append((ds, groups, labels))
    return specs


def _panel_soft_extra(
    data_root: Path, datasets: list[str], curated_group_ids: set[str]
) -> pd.DataFrame:
    """Featurized panel-soft edges for groups NOT already in the curated table."""
    frames = []
    batches_root = data_root / "data" / "agents" / "stitching" / "batches"
    for ds in datasets:
        groups = load_sidecar_groups(data_root / "data" / "output" / f"{ds}_groups.json")
        votes = load_votes(default_votes_paths(batches_root))
        soft = edge_soft_labels(groups, votes)
        if soft.empty:
            continue
        soft = soft[~soft["group_id"].isin(curated_group_ids)]
        if soft.empty:
            continue
        # attach the sidecar edge fields + group context so featurize works
        gmap = {g["group_id"]: g for g in groups}
        rows = []
        for _, r in soft.iterrows():
            g = gmap[r["group_id"]]
            edge = next(
                (
                    e
                    for e in g["edges"]
                    if str(e["ref_id"]) == r["ref_id"] and str(e["target_id"]) == r["target_id"]
                ),
                None,
            )
            if edge is None:
                continue
            rows.append(
                {
                    "dataset_id": ds,
                    "group_id": r["group_id"],
                    "match_type": g.get("match_type", ""),
                    "ref_id": r["ref_id"],
                    "target_id": r["target_id"],
                    "soft_keep": r["soft_keep"],
                    "confidence": float(edge.get("confidence", float("nan"))),
                    "degree_ref": int(edge.get("degree_ref", 0)),
                    "degree_tgt": int(edge.get("degree_tgt", 0)),
                    "is_bridge": bool(edge.get("is_bridge", False)),
                    "is_sliver": bool(edge.get("is_sliver", False)),
                    "biconnected_block": int(edge.get("biconnected_block", -1)),
                    "corridor_ref": int(edge.get("corridor_ref", -1)),
                    "corridor_tgt": int(edge.get("corridor_tgt", -1)),
                    "gers_start_frac": float(edge.get("gers_start_frac", float("nan"))),
                    "gers_end_frac": float(edge.get("gers_end_frac", float("nan"))),
                    "local_start_frac": float(edge.get("local_start_frac", float("nan"))),
                    "local_end_frac": float(edge.get("local_end_frac", float("nan"))),
                    "n_edges": int(g.get("n_edges", len(g["edges"]))),
                    "n_corridors": int(g.get("n_corridors", 1)),
                    "n_assignment_components": int(g.get("n_assignment_components", 1)),
                    "largest_biconnected_block": int(g.get("largest_biconnected_block", 1)),
                    "oversized_group": bool(g.get("oversized_group", False)),
                    "num_refs": len(g.get("ref_ids", [])),
                    "num_targets": len(g.get("target_ids", [])),
                }
            )
        if rows:
            frames.append(featurize(pd.DataFrame(rows)))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path.cwd())
    ap.add_argument("--dataset", action="append", dest="datasets", required=True)
    ap.add_argument("--out", type=Path, default=None, help="write edge table parquet")
    ap.add_argument("--no-split", action="store_true", help="clean labels only")
    ap.add_argument("--with-votes", action="store_true", help="add panel soft labels")
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    specs = _specs(args.data_root, args.datasets)
    df = build_multi_dataset_table(specs, include_split=not args.no_split)
    if df.empty:
        print("No labeled edges recovered — check paths.")
        return
    feat = featurize(df)
    # featurize returns a reshaped frame that need not carry df.attrs; re-attach
    # the per-dataset build audit that concat preserved on the combined table so
    # write_edge_table_parquet can embed it (pandas to_parquet drops attrs).
    feat.attrs[COMBINED_AUDIT_ATTR] = df.attrs.get(COMBINED_AUDIT_ATTR, {})
    feat.attrs[COMBINED_STATS_ATTR] = df.attrs.get(COMBINED_STATS_ATTR, {})

    print("\n=== per-edge dataset ===")
    print(
        f"edges={len(feat)} groups={feat['group_id'].nunique()} "
        f"keep={int(feat['keep'].sum())} drop={int((feat['keep'] == 0).sum())}"
    )
    print(feat.groupby(["dataset_id", "provenance"]).size().to_string())

    if args.out:
        write_edge_table_parquet(feat, args.out, index=False)
        print(f"wrote {args.out}")

    soft_extra = None
    if args.with_votes:
        curated = set(feat["group_id"].unique())
        soft_extra = _panel_soft_extra(args.data_root, args.datasets, curated)
        print(
            f"\npanel-soft extra groups (not in curated): "
            f"{0 if soft_extra is None or soft_extra.empty else soft_extra['group_id'].nunique()}"
        )

    print("\n=== grouped-CV: model vs baselines (all mapped labels) ===")
    res = run_cv(feat, n_splits=args.n_splits)
    for key in ("baseline_keepall", "baseline_conf", "model"):
        print(res[key].row())
    print("model per-fold F1:", res["fold_f1"])

    clean = feat[feat["provenance"] == "clean"]
    if clean["group_id"].nunique() >= 2:
        print("\n=== grouped-CV: CLEAN labels only ===")
        resc = run_cv(clean, n_splits=args.n_splits)
        for key in ("baseline_keepall", "baseline_conf", "model"):
            print(resc[key].row())
        print("model per-fold F1:", resc["fold_f1"])

    if soft_extra is not None and not soft_extra.empty:
        print("\n=== grouped-CV: CLEAN + panel-soft extra training groups ===")
        resv = run_cv(clean, n_splits=args.n_splits, soft_extra=soft_extra)
        for key in ("model",):
            print(resv[key].row())
        print("model per-fold F1:", resv["fold_f1"])

    print("\n=== slice report ===")
    print(slice_report(feat, n_splits=args.n_splits).to_string(index=False))

    print("\n=== feature importances (gain, full-data fit) ===")
    print(feature_importances(feat).head(12).round(3).to_string())


if __name__ == "__main__":
    main()
