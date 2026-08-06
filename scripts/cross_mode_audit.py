#!/usr/bin/env python
"""Cross-mode match audit over factory bridge outputs.

For every dataset in a factory release, join matched pairs (bridge.parquet)
to reference (Overture) and target class values from the raw source parquets,
and flag cross-mode matches: a pedestrian/bike-class feature on one side
matched to a vehicular road class on the other (or any two differing
non-neutral modes).

The mode taxonomy is NOT invented here -- it is imported from the panel
routing's deterministic class-consistency gate
(crosswalk.agent_labeling.stitch_runner):
  PEDESTRIAN_CLASSES, VEHICULAR_CLASSES, CYCLEWAY_CLASSES,
  road_class_mode(), is_cross_mode_edge().

Usage:
  uv run python scripts/cross_mode_audit.py [--release 2026-06-17.0] \
      [--data-root .] [--conf-threshold 0.8] \
      [--examples 5] [--json OUT.json]

The 2026-08-06 run of this script over release 2026-06-17.0 is written up in
research/cross_mode_audit_2026-08-06.md. Re-run it after any change that alters
a dataset's class vocabulary (e.g. a fetch.class_mapping fix + re-fetch) — a
mis-mapped source class shows up here as a cross-mode defect that is really a
labelling artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Reuse the deterministic cross-mode gate taxonomy from the panel runner.
from crosswalk.agent_labeling.stitch_runner import (  # noqa: F401
    CYCLEWAY_CLASSES,
    PEDESTRIAN_CLASSES,
    VEHICULAR_CLASSES,
    is_cross_mode_edge,
    road_class_mode,
)


def load_class_map(path: Path) -> pd.Series | None:
    """Load id -> class from a raw parquet (only the two columns)."""
    if not path.exists():
        return None
    try:
        tbl = pq.read_table(path, columns=["id", "class"])
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  WARN: cannot read {path}: {exc}", file=sys.stderr)
        return None
    df = tbl.to_pandas()
    return df.drop_duplicates("id").set_index("id")["class"]


# Dataset `type` (datasets/<name>.yaml) -> travel mode, used ONLY as a
# fallback when the target's stored class is missing/neutral. A sidewalk
# dataset's targets are pedestrian by construction; a bike dataset's are bike.
# `trail` and `road` are NOT inferred (trails are often mixed-use; roads
# already carry classes).
DATASET_TYPE_MODE = {"sidewalk": "pedestrian", "bike": "bike"}


def dataset_type(datasets_dir: Path, name: str) -> str | None:
    yml = datasets_dir / f"{name}.yaml"
    if not yml.exists():
        return None
    for line in yml.read_text().splitlines():
        if line.startswith("type:"):
            return line.split(":", 1)[1].strip()
    return None


def audit_dataset(
    dataset_dir: Path,
    raw_dir: Path,
    datasets_dir: Path,
    conf_threshold: float,
    n_examples: int,
) -> dict | None:
    name = dataset_dir.name.split("=", 1)[1]
    bridge_path = dataset_dir / "bridge.parquet"
    if not bridge_path.exists():
        return None
    bridge = pd.read_parquet(
        bridge_path,
        columns=["local_id", "gers_id", "confidence", "match_decision", "match_method"],
    )
    matched = bridge[bridge["match_decision"] == "match"].copy()

    ref_classes = load_class_map(raw_dir / f"{name}_overture_segments_v1.0.parquet")
    tgt_classes = load_class_map(raw_dir / f"{name}_v1.0.parquet")

    result: dict = {
        "dataset": name,
        "total_bridge_rows": int(len(bridge)),
        "total_matches": int(len(matched)),
        "ref_class_coverage": None,
        "target_class_coverage": None,
        "cross_mode": None,
        "cross_mode_high_conf": None,
        "cross_mode_pct": None,
        "cross_mode_high_conf_pct": None,
        "cross_mode_veh": None,
        "cross_mode_veh_high_conf": None,
        "cross_mode_veh_pct": None,
        "cross_mode_veh_high_conf_pct": None,
        "mode_pair_counts": {},
        "examples": [],
        "notes": [],
    }
    if ref_classes is None or tgt_classes is None:
        missing = []
        if ref_classes is None:
            missing.append("ref (overture_segments)")
        if tgt_classes is None:
            missing.append("target")
        result["notes"].append(f"missing raw class source: {', '.join(missing)}")
        return result
    if matched.empty:
        result["notes"].append("no matched rows")
        return result

    matched["ref_class"] = matched["gers_id"].map(ref_classes)
    matched["target_class"] = matched["local_id"].map(tgt_classes)
    result["ref_class_coverage"] = float(matched["ref_class"].notna().mean())
    result["target_class_coverage"] = float(matched["target_class"].notna().mean())

    def _norm(v):
        # NaN / non-string (e.g. float nan from the join) -> None, which the
        # gate treats as neutral (never fires).
        return v if isinstance(v, str) else None

    matched["ref_class"] = matched["ref_class"].map(_norm)
    matched["target_class"] = matched["target_class"].map(_norm)
    matched["ref_mode"] = matched["ref_class"].map(road_class_mode)
    matched["target_mode"] = matched["target_class"].map(road_class_mode)

    # Fallback: infer target mode from dataset type when the stored target
    # class is neutral/missing (e.g. us_seattle_sidewalks stores no class).
    ds_type = dataset_type(datasets_dir, name)
    inferred_mode = DATASET_TYPE_MODE.get(ds_type or "")
    if inferred_mode is not None:
        neutral = matched["target_mode"] == "neutral"
        if neutral.any():
            matched.loc[neutral, "target_mode"] = inferred_mode
            matched.loc[neutral, "target_class"] = f"<{ds_type}:inferred>"
            result["notes"].append(
                f"target mode inferred from dataset type={ds_type} for "
                f"{int(neutral.sum())} rows with neutral/missing class"
            )

    matched["cross_mode"] = (
        (matched["ref_mode"] != "neutral")
        & (matched["target_mode"] != "neutral")
        & (matched["ref_mode"] != matched["target_mode"])
    )
    # "Hard" cross-mode: vehicular on exactly one side (the dangerous
    # parallel-road defect). ped<->bike is the gate's conservative extra.
    matched["cross_mode_veh"] = matched["cross_mode"] & (
        (matched["ref_mode"] == "vehicular") | (matched["target_mode"] == "vehicular")
    )

    # Fraction of matched rows whose class maps to "neutral" (unknown/track/
    # missing) — high values mean the audit is BLIND on that side, and a zero
    # cross-mode count is NOT evidence of cleanliness.
    result["ref_neutral_frac"] = round(float((matched["ref_mode"] == "neutral").mean()), 3)
    result["target_neutral_frac"] = round(float((matched["target_mode"] == "neutral").mean()), 3)

    cm = matched[matched["cross_mode"]]
    cm_hi = cm[cm["confidence"] >= conf_threshold]
    cmv = matched[matched["cross_mode_veh"]]
    cmv_hi = cmv[cmv["confidence"] >= conf_threshold]
    n = len(matched)
    result["cross_mode"] = int(len(cm))
    result["cross_mode_high_conf"] = int(len(cm_hi))
    result["cross_mode_pct"] = round(100.0 * len(cm) / n, 2)
    result["cross_mode_high_conf_pct"] = round(100.0 * len(cm_hi) / n, 2)
    result["cross_mode_veh"] = int(len(cmv))
    result["cross_mode_veh_high_conf"] = int(len(cmv_hi))
    result["cross_mode_veh_pct"] = round(100.0 * len(cmv) / n, 2)
    result["cross_mode_veh_high_conf_pct"] = round(100.0 * len(cmv_hi) / n, 2)

    pair_counts = cm.groupby(["ref_mode", "target_mode"]).size().sort_values(ascending=False)
    result["mode_pair_counts"] = {
        f"ref={a}<->target={b}": int(v) for (a, b), v in pair_counts.items()
    }
    # Also class-level detail for the top offending class pairs.
    cls_counts = cm.groupby(["ref_class", "target_class"]).size().sort_values(ascending=False)
    result["class_pair_counts"] = {f"{a}<->{b}": int(v) for (a, b), v in cls_counts.head(8).items()}

    # Examples: highest-confidence cross-mode pairs, de-duplicated by class pair.
    ex_pool = cm.sort_values("confidence", ascending=False)
    seen_pairs: set[tuple] = set()
    examples = []
    for _, row in ex_pool.iterrows():
        key = (row["ref_class"], row["target_class"])
        if key in seen_pairs and len(seen_pairs) < n_examples:
            continue
        seen_pairs.add(key)
        examples.append(
            {
                "local_id": row["local_id"],
                "gers_id": row["gers_id"],
                "ref_class": row["ref_class"],
                "target_class": row["target_class"],
                "confidence": round(float(row["confidence"]), 3),
            }
        )
        if len(examples) >= n_examples:
            break
    result["examples"] = examples

    # Confidence distribution of cross-mode matches.
    if len(cm):
        result["cross_mode_conf"] = {
            "min": round(float(cm["confidence"].min()), 3),
            "median": round(float(cm["confidence"].median()), 3),
            "max": round(float(cm["confidence"].max()), 3),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parent.parent),
        help="Repo root holding data/ and datasets/ (default: this checkout)",
    )
    ap.add_argument("--release", default="2026-06-17.0")
    ap.add_argument("--conf-threshold", type=float, default=0.8)
    ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("--json", help="write full results as JSON to this path")
    args = ap.parse_args()

    root = Path(args.data_root)
    release_dir = root / "data" / "factory" / f"release={args.release}"
    raw_dir = root / "data" / "raw"
    if not release_dir.exists():
        sys.exit(f"release dir not found: {release_dir}")

    datasets_dir = root / "datasets"
    results = []
    for ds_dir in sorted(release_dir.glob("dataset=*")):
        r = audit_dataset(ds_dir, raw_dir, datasets_dir, args.conf_threshold, args.examples)
        if r is not None:
            results.append(r)

    # Table. xveh = cross-mode with a vehicular side (the dangerous
    # parallel-road pattern); xmode additionally counts ped<->bike.
    hdr = (
        f"{'dataset':34} {'matches':>8} {'xmode':>6} {'xhi':>6} "
        f"{'xveh':>6} {'xvehhi':>6} {'xvehhi%':>7}  top mode pairs"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(results, key=lambda x: -(x["cross_mode_veh_high_conf"] or 0)):
        if r["cross_mode"] is None:
            print(
                f"{r['dataset']:34} {r['total_matches']:>8}      -      -      -"
                f"      -       -  {'; '.join(r['notes'])}"
            )
            continue
        pairs = ", ".join(f"{k}:{v}" for k, v in list(r["mode_pair_counts"].items())[:2])
        print(
            f"{r['dataset']:34} {r['total_matches']:>8} {r['cross_mode']:>6} "
            f"{r['cross_mode_high_conf']:>6} {r['cross_mode_veh']:>6} "
            f"{r['cross_mode_veh_high_conf']:>6} {r['cross_mode_veh_high_conf_pct']:>7}  {pairs}"
        )

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nJSON written to {args.json}")


if __name__ == "__main__":
    main()
