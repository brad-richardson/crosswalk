"""Learned stitcher round-2 experiment runner.

Builds the per-edge table from the CURRENT sidecars (which persist rejected +
pruned candidates, post #282/#284), reproduces the round-1 per-edge model, and
runs the architecture matrix (extended features x structured selection x panel
soft labels). Prints tidy result tables; used to produce
research/learned_stitcher_round2.md.

Usage (PYTHONHASHSEED=0 for bit-stable split-label mapping):
    PYTHONHASHSEED=0 uv run python scripts/run_stitcher_round2.py \
        --data-root /path/to/matcher
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crosswalk.resolver.evaluate import run_cv
from crosswalk.resolver.extract import build_multi_dataset_table, load_sidecar_groups
from crosswalk.resolver.features import FEATURE_COLUMNS, featurize
from crosswalk.resolver.round2 import EXTENDED_FEATURE_COLUMNS, featurize_extended, run_cv2
from crosswalk.resolver.votes import default_votes_paths, edge_soft_labels, load_votes


def _rows(res: dict, slice_name: str, config: str) -> list[dict]:
    out = []
    for key, r in res.items():
        if key in ("oof_proba", "fold_f1"):
            continue
        d = r.row()
        d["slice"] = slice_name
        d["config"] = config
        out.append(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/Users/bradrichardson/dev/matcher")
    args = ap.parse_args()
    root = Path(args.data_root)

    specs = [
        (
            "us_boston_streets",
            root / "data/output/us_boston_streets_groups.json",
            root / "labels/stitching/dataset=us_boston_streets/data.csv",
        ),
        (
            "us_seattle_sidewalks",
            root / "data/output/us_seattle_sidewalks_groups.json",
            root / "labels/stitching/dataset=us_seattle_sidewalks/data.csv",
        ),
    ]
    raw = build_multi_dataset_table(specs)
    print(f"table: {len(raw)} edges / {raw.group_id.nunique()} groups")
    print(raw.groupby(["dataset_id", "provenance"]).group_id.nunique())
    print("over-selection negatives:", int(((raw.selected) & (raw.keep == 0)).sum()))
    print("under-selection positives:", int(((~raw.selected) & (raw.keep == 1)).sum()))
    print("  of which pruned:", int(((raw.pruned) & (raw.keep == 1)).sum()))

    r1 = featurize(raw)
    ext = featurize_extended(raw)

    # Panel soft labels (groups NOT in the curated table), featurized both ways.
    votes = load_votes(default_votes_paths(root / "data/agents/stitching/batches"))
    soft_frames = []
    for ds, groups_path, _ in specs:
        groups = load_sidecar_groups(groups_path)
        soft = edge_soft_labels(groups, votes)
        if len(soft):
            soft["dataset_id"] = ds
            gmap = {g["group_id"]: g for g in groups}
            meta = []
            for gid, sub in soft.groupby("group_id"):
                g = gmap[gid]
                edges = {(str(e["ref_id"]), str(e["target_id"])): e for e in g.get("edges", [])}
                for _, r in sub.iterrows():
                    e = edges.get((r["ref_id"], r["target_id"]), {})
                    meta.append(
                        {
                            **{
                                k: e.get(k)
                                for k in (
                                    "confidence",
                                    "degree_ref",
                                    "degree_tgt",
                                    "is_bridge",
                                    "is_sliver",
                                    "biconnected_block",
                                    "corridor_ref",
                                    "corridor_tgt",
                                    "gers_start_frac",
                                    "gers_end_frac",
                                    "local_start_frac",
                                    "local_end_frac",
                                    "selected",
                                )
                            },
                            "group_id": gid,
                            "ref_id": r["ref_id"],
                            "target_id": r["target_id"],
                            "soft_keep": r["soft_keep"],
                            "match_type": g.get("match_type", ""),
                            "n_edges": g.get("n_edges", 0),
                            "n_corridors": g.get("n_corridors", 1),
                            "n_assignment_components": g.get("n_assignment_components", 1),
                            "largest_biconnected_block": g.get("largest_biconnected_block", 1),
                            "oversized_group": g.get("oversized_group", False),
                            "num_refs": len(g.get("ref_ids", [])),
                            "num_targets": len(g.get("target_ids", [])),
                        }
                    )
            soft_frames.append(pd.DataFrame(meta))
    soft_all = pd.concat(soft_frames, ignore_index=True) if soft_frames else pd.DataFrame()
    # exclude curated-table groups from the soft-label extra set
    soft_all = soft_all[~soft_all.group_id.isin(set(raw.group_id))].reset_index(drop=True)
    soft_all["selected"] = soft_all["selected"].fillna(True)
    soft_all["confidence"] = soft_all["confidence"].astype(float)
    print(f"soft-label extra: {len(soft_all)} edges / {soft_all.group_id.nunique()} groups")
    soft_r1 = featurize(soft_all) if len(soft_all) else None
    soft_ext = featurize_extended(soft_all) if len(soft_all) else None

    slices = {
        "all": lambda d: d,
        "clean": lambda d: d[d.provenance == "clean"],
        "boston": lambda d: d[d.dataset_id == "us_boston_streets"],
        "seattle": lambda d: d[d.dataset_id == "us_seattle_sidewalks"],
    }

    rows: list[dict] = []
    for sname, fn in slices.items():
        sub_r1 = fn(r1).reset_index(drop=True)
        sub_ext = fn(ext).reset_index(drop=True)
        if sub_r1.group_id.nunique() < 5 or sub_r1.keep.nunique() < 2:
            continue
        # 1) round-1 reproduction (original harness, original features)
        res = run_cv(sub_r1)
        for key in ("model", "baseline_keepall", "baseline_conf"):
            d = res[key].row()
            d["slice"], d["config"] = sname, "round1-repro"
            rows.append(d)
        # 2) architecture matrix
        for cfg, (sub, cols, sel, soft) in {
            "r1feats+ef1": (sub_r1, FEATURE_COLUMNS, "ef1", None),
            "extfeats+thr": (sub_ext, EXTENDED_FEATURE_COLUMNS, "threshold", None),
            "extfeats+ef1": (sub_ext, EXTENDED_FEATURE_COLUMNS, "ef1", None),
            "extfeats+ef1+soft": (sub_ext, EXTENDED_FEATURE_COLUMNS, "ef1", soft_ext),
            "r1feats+thr+soft": (sub_r1, FEATURE_COLUMNS, "threshold", soft_r1),
        }.items():
            res2 = run_cv2(sub, cols, selector=sel, soft_extra=soft)
            rows.extend(_rows(res2, sname, cfg))

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for sname in out["slice"].unique():
        print(f"\n===== slice: {sname} =====")
        print(
            out[out["slice"] == sname][
                ["config", "model", "edges", "groups", "P", "R", "F1", "grp_exact", "F1_sliverfilt"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
