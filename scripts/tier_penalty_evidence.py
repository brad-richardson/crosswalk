#!/usr/bin/env python
"""Check every hand-coded constant in compute_class_similarity() against labels.

`crosswalk.features.semantic.compute_class_similarity` scores a class pair with
hand-picked constants:

    cross-tier   TIER_PENALTIES: vehicle<->pedestrian 0.1,
                 vehicle<->bicycle 0.7, bicycle<->pedestrian 0.5
    same class   1.0 (no subclass or subclass agrees)
                 0.9 (one side has a subclass, the other does not)
                 0.85 (subclasses differ)
    same tier    max(0, 1.0 - 0.2 * |rank_a - rank_b|)   [ROAD_CLASS_HIERARCHY]
    neutral/unknown tier or "unknown" class -> NaN

This script measures P(match) over every adjudicated label for each of those
branches, so the constants can be checked against observed behaviour instead of
intuition.

Usage:
  uv run python scripts/tier_penalty_evidence.py [--include-agent] [--min-n 3]

CAVEAT, stated up front: labeled pairs are adjudicated *candidates*, not a
random sample of all pairs. They are enriched for hard/ambiguous cases, so the
absolute P(match) is not a population rate. Comparisons ACROSS branches are
still meaningful because every group is drawn from the same candidate process.
A branch's P(match) is therefore evidence about relative ordering and rough
magnitude, not a value to copy verbatim into the table.
"""

from __future__ import annotations

import argparse
import glob

import pandas as pd
import pyarrow.parquet as pq

from crosswalk.features.semantic import (
    ROAD_CLASS_HIERARCHY,
    TIER_PENALTIES,
    get_traffic_tier,
)

RAW = "data/raw"


def _read(path: str, cols: list[str]) -> pd.DataFrame | None:
    try:
        have = set(pq.read_schema(path).names)
        cols = [c for c in cols if c in have]
        if "id" not in cols or "class" not in cols:
            return None
        return pq.read_table(path, columns=cols).to_pandas().drop_duplicates("id")
    except Exception:
        return None


def load_labels(include_agent: bool) -> pd.DataFrame:
    pats = [("human", "labels/human/dataset=*/data.csv")]
    if include_agent:
        pats.append(("agent", "labels/agent/dataset=*/data.csv"))
    frames = []
    for src, pat in pats:
        for f in sorted(glob.glob(pat)):
            ds = f.split("dataset=")[1].split("/")[0]
            t = _read(f"{RAW}/{ds}_v1.0.parquet", ["id", "class", "subclass"])
            r = _read(f"{RAW}/{ds}_overture_segments_v1.0.parquet", ["id", "class", "subclass"])
            if t is None or r is None:
                continue
            h = pd.read_csv(f)
            if "label" not in h.columns or "target_id" not in h.columns:
                continue
            t = t.set_index("id")
            r = r.set_index("id")
            h["tgt_class"] = h["target_id"].map(t["class"])
            h["ref_class"] = h["gers_id"].map(r["class"])
            h["tgt_sub"] = h["target_id"].map(t["subclass"]) if "subclass" in t else None
            h["ref_sub"] = h["gers_id"].map(r["subclass"]) if "subclass" in r else None
            h["dataset"] = ds
            h["source"] = src
            frames.append(
                h[["dataset", "source", "label", "ref_class", "tgt_class", "ref_sub", "tgt_sub"]]
            )
    return pd.concat(frames, ignore_index=True)


def rate(df: pd.DataFrame, by) -> pd.DataFrame:
    g = df.groupby(by)["label"].agg(n="size", matches=lambda s: int((s == "match").sum()))
    g["P(match)"] = (g["matches"] / g["n"]).round(3)
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-agent", action="store_true")
    ap.add_argument("--min-n", type=int, default=3)
    args = ap.parse_args()

    d = load_labels(args.include_agent)
    d = d[d["label"].isin(["match", "no_match"])].dropna(subset=["ref_class", "tgt_class"])
    d = d[(d.ref_class != "unknown") & (d.tgt_class != "unknown")]
    d["ta"] = d.ref_class.map(get_traffic_tier)
    d["tb"] = d.tgt_class.map(get_traffic_tier)
    d = d.dropna(subset=["ta", "tb"])
    d = d[(d.ta != "neutral") & (d.tb != "neutral")]

    base = "human + agent" if args.include_agent else "human only"
    print(f"LABEL BASE: {base} — {len(d):,} adjudicated pairs with both classes resolved")
    print(f"overall P(match) = {(d.label == 'match').mean():.3f}\n")

    # --- 1. cross-tier: the TIER_PENALTIES table -------------------------------
    cross = d[d.ta != d.tb].copy()
    cross["pair"] = [tuple(sorted([a, b])) for a, b in zip(cross.ta, cross.tb)]
    g = rate(cross, "pair")
    g["coded"] = [TIER_PENALTIES.get(p, TIER_PENALTIES.get(p[::-1])) for p in g.index]
    g["datasets"] = cross.groupby("pair")["dataset"].nunique()
    print("[1] CROSS-TIER — the TIER_PENALTIES constants")
    print(g.sort_values("n", ascending=False).to_string(), "\n")

    # --- 2. same-tier rank decay: 1.0 - 0.2 * rank_diff ------------------------
    same = d[d.ta == d.tb].copy()
    same["rank_diff"] = [
        abs(ROAD_CLASS_HIERARCHY.get(a, 6) - ROAD_CLASS_HIERARCHY.get(b, 6))
        for a, b in zip(same.ref_class, same.tgt_class)
    ]
    same["exact"] = same.ref_class == same.tgt_class
    diff_cls = same[~same.exact].copy()
    g2 = rate(diff_cls, "rank_diff")
    g2["coded"] = [max(0.0, 1.0 - r * 0.2) for r in g2.index]
    print("[2] SAME-TIER, DIFFERENT CLASS — the rank-decay rule")
    print(g2[g2.n >= args.min_n].to_string(), "\n")

    print("[2a] same-tier rank decay, split by tier")
    for tier in ("vehicle", "pedestrian", "bicycle"):
        sub = diff_cls[diff_cls.ta == tier]
        if len(sub) < args.min_n:
            continue
        gt = rate(sub, "rank_diff")
        gt["coded"] = [max(0.0, 1.0 - r * 0.2) for r in gt.index]
        print(f"  --- {tier} ---")
        print("  " + gt[gt.n >= args.min_n].to_string().replace("\n", "\n  "))
    print()

    # --- 3. exact class match: the 1.0 / 0.9 / 0.85 subclass constants ---------
    ex = same[same.exact].copy()

    def sub_branch(row) -> str:
        a = row.ref_sub if isinstance(row.ref_sub, str) and row.ref_sub.strip() else None
        b = row.tgt_sub if isinstance(row.tgt_sub, str) and row.tgt_sub.strip() else None
        if not a and not b:
            return "neither subclass -> 1.0"
        if a and b:
            return "subclass equal -> 1.0" if a.lower() == b.lower() else "subclass differs -> 0.85"
        return "one subclass only -> 0.9"

    ex["branch"] = ex.apply(sub_branch, axis=1)
    print("[3] EXACT CLASS MATCH — the subclass constants")
    print(rate(ex, "branch").sort_values("n", ascending=False).to_string(), "\n")

    # --- 4. biggest individual class pairs, for eyeballing ---------------------
    d["cls_pair"] = d.ref_class + " -> " + d.tgt_class
    g4 = rate(d, "cls_pair")
    print("[4] TOP CLASS PAIRS BY VOLUME (ref -> target)")
    print(g4[g4.n >= 20].sort_values("n", ascending=False).head(18).to_string())


if __name__ == "__main__":
    main()
