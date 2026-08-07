#!/usr/bin/env python
"""Re-key labels whose target_id no longer exists in the current raw target data.

Why
---
Target datasets get re-fetched, and some re-fetches mint a new id scheme
(``sg_singapore_roads`` labels are ``sg_road_None_…`` — the source id column was
null at label time; ``us_utah_slc_roads`` went ``…_201491_…`` -> ``…_854_…``).
Every label keyed to the old scheme is orphaned: ``crosswalk backfill`` skips it
with ``--skip-missing``, so it silently leaves the training set. As of
2026-08-07 that is 511 pairs, ~9% of the labeled base
(research/label_feature_staleness_scope_2026-08-07.md).

``labels/data/`` retains ``target_geometry`` for 100% of them, so the old id can
be mapped to the new one geometrically without re-labeling.

Matching (deliberately conservative — a mis-keyed label poisons training data,
which is strictly worse than a dropped one)
-------------------------------------------------------------------------
  tier 1  exact WKB equality, and exactly one raw row has that WKB
  tier 2  Hausdorff distance <= --tol metres, and exactly ONE raw geometry is
          within tolerance (ambiguous candidates are refused, not guessed)
  else    left unchanged and reported as unmatched

Tier 2 exists because re-fetches commonly rewrite identical geometry with
different WKB bytes (coordinate precision, vertex order, Z stripped): those show
Hausdorff 0.0 while failing byte equality.

Safety
------
  * ``--dry-run`` (default) reports counts and writes nothing.
  * ``--apply`` rewrites ``target_id`` across labels/human, labels/agent,
    labels/data and labels/features, backing each file up to ``.prerekey.bak``.
  * A re-key that would collide with an existing (gers_id, target_id) row in the
    same store is refused and reported.
  * Run ``crosswalk backfill`` afterwards — re-keyed rows still carry features
    computed against the OLD geometry.

Usage
-----
  uv run python scripts/rekey_orphaned_labels.py                  # dry run, all
  uv run python scripts/rekey_orphaned_labels.py -D us_utah_slc_roads
  uv run python scripts/rekey_orphaned_labels.py --tol 0.5 --apply
"""

from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from shapely import wkb as shwkb
from shapely.strtree import STRtree

LABEL_STORES = ("labels/human", "labels/agent")
DATA_STORE = "labels/data"
FEATURE_STORE = "labels/features"


def datasets_with_labels() -> list[str]:
    return sorted(
        p.split("dataset=")[1].split("/")[0]
        for p in glob.glob(f"{DATA_STORE}/dataset=*/data.parquet")
    )


def find_orphans(ds: str) -> tuple[pd.DataFrame, gpd.GeoDataFrame] | None:
    dpath = Path(f"{DATA_STORE}/dataset={ds}/data.parquet")
    rpath = Path(f"data/raw/{ds}_v1.0.parquet")
    if not dpath.exists() or not rpath.exists():
        return None
    cols = pq.ParquetFile(dpath).schema_arrow.names
    if "target_geometry" not in cols:
        return None
    d = pq.ParquetFile(dpath).read(columns=["gers_id", "target_id", "target_geometry"]).to_pandas()
    raw = gpd.read_parquet(rpath, columns=["id", "geometry"])
    orph = d[~d.target_id.isin(set(raw["id"]))].copy()
    orph = orph[orph.target_geometry.notna()]
    return (orph, raw) if len(orph) else None


def build_mapping(orph: pd.DataFrame, raw: gpd.GeoDataFrame, tol: float) -> tuple[dict, dict]:
    """Return (old_id -> new_id, old_id -> reason-for-no-match)."""
    geoms = [shwkb.loads(bytes(b)) for b in orph.target_geometry]

    # tier 1: exact WKB, unique
    wkb_counts: dict[bytes, int] = {}
    wkb_id: dict[bytes, str] = {}
    for g, i in zip(raw.geometry, raw["id"]):
        b = g.wkb
        wkb_counts[b] = wkb_counts.get(b, 0) + 1
        wkb_id[b] = i

    mapping: dict[str, str] = {}
    unresolved: list[int] = []
    for pos, (old_id, g) in enumerate(zip(orph.target_id, geoms)):
        b = g.wkb
        if wkb_counts.get(b) == 1:
            mapping[old_id] = wkb_id[b]
        else:
            unresolved.append(pos)

    reasons: dict[str, str] = {}
    if unresolved and tol >= 0:
        rs = gpd.GeoSeries(raw.geometry.values, crs="EPSG:4326")
        utm = rs.estimate_utm_crs()
        rm = rs.to_crs(utm).values
        om = gpd.GeoSeries([geoms[p] for p in unresolved], crs="EPSG:4326").to_crs(utm).values
        tree = STRtree(rm)
        for k, pos in enumerate(unresolved):
            old_id = orph.target_id.iloc[pos]
            cand = tree.query(om[k].buffer(max(tol, 0.01)))
            if len(cand) == 0:
                reasons[old_id] = "no candidate within tolerance"
                continue
            hd = np.array([shapely.hausdorff_distance(om[k], rm[c]) for c in cand])
            within = cand[hd <= tol]
            if len(within) == 1:
                mapping[old_id] = raw["id"].iloc[within[0]]
            elif len(within) == 0:
                reasons[old_id] = f"nearest hausdorff {hd.min():.1f}m > tol"
            else:
                reasons[old_id] = f"ambiguous: {len(within)} candidates within tol"
    return mapping, reasons


def rewrite(path: Path, mapping: dict, apply: bool, key_cols: list[str]) -> tuple[int, int]:
    """Rewrite target_id in a csv/parquet store. Returns (rows_changed, collisions)."""
    if not path.exists():
        return (0, 0)
    is_csv = path.suffix == ".csv"
    df = pd.read_csv(path) if is_csv else pq.ParquetFile(path).read().to_pandas()
    if "target_id" not in df.columns:
        return (0, 0)
    new = df.target_id.map(lambda t: mapping.get(t, t))
    changed = int((new != df.target_id).sum())
    if not changed:
        return (0, 0)
    probe = df.copy()
    probe["target_id"] = new
    collisions = 0
    if all(c in probe.columns for c in key_cols):
        dup = probe.duplicated(subset=key_cols, keep=False) & (new != df.target_id)
        collisions = int(dup.sum())
        if collisions:
            keep = ~((new != df.target_id) & probe.duplicated(subset=key_cols, keep="first"))
            probe = probe[keep]
    if apply:
        shutil.copy2(path, path.with_suffix(path.suffix + ".prerekey.bak"))
        if is_csv:
            probe.to_csv(path, index=False)
        else:
            probe.to_parquet(path, index=False)
    return (changed, collisions)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", "-D", action="append", help="Limit to these datasets (repeatable)")
    ap.add_argument("--tol", type=float, default=0.5, help="Hausdorff tolerance in metres (tier 2)")
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry run)")
    args = ap.parse_args()

    targets = args.dataset or datasets_with_labels()
    total_orph = total_map = total_coll = 0
    rows = []
    for ds in targets:
        found = find_orphans(ds)
        if not found:
            continue
        orph, raw = found
        mapping, reasons = build_mapping(orph, raw, args.tol)
        # Count ROWS, not distinct ids: a dataset can label the same target
        # several times (different gers_id), so distinct-id counts understate
        # how many label rows a mapping actually recovers.
        n_ids = orph.target_id.nunique()
        n_map_ids = len(mapping)
        n_map = int(orph.target_id.isin(mapping).sum())
        total_orph += len(orph)
        total_map += n_map
        coll = 0
        if mapping:
            for store in LABEL_STORES:
                c, x = rewrite(
                    Path(f"{store}/dataset={ds}/data.csv"),
                    mapping,
                    args.apply,
                    ["gers_id", "target_id"],
                )
                coll += x
            for store in (DATA_STORE, FEATURE_STORE):
                c, x = rewrite(
                    Path(f"{store}/dataset={ds}/data.parquet"),
                    mapping,
                    args.apply,
                    ["gers_id", "target_id"],
                )
                coll += x
        total_coll += coll
        why = pd.Series(list(reasons.values())).value_counts().to_dict() if reasons else {}
        rows.append(
            {
                "dataset": ds,
                "rows": len(orph),
                "ids": n_ids,
                "ids_mapped": n_map_ids,
                "rows_rekeyed": n_map,
                "pct": round(100 * n_map / len(orph), 1),
                "rows_left": len(orph) - n_map,
                "collisions": coll,
            }
        )
        if why:
            top = ", ".join(f"{k} ({v})" for k, v in list(why.items())[:3])
            rows[-1]["why_unmatched"] = top

    if not rows:
        print("No orphaned labels found.")
        return
    df = pd.DataFrame(rows).sort_values("rows", ascending=False)
    print(df.to_string(index=False))
    print(
        f"\n{'APPLIED' if args.apply else 'DRY RUN'}: "
        f"{total_map}/{total_orph} orphaned labels re-keyed "
        f"({100 * total_map / max(total_orph, 1):.1f}%), "
        f"{total_orph - total_map} left unchanged, {total_coll} collisions dropped."
    )
    if not args.apply:
        print("Re-run with --apply to write. Then run `crosswalk backfill`.")


if __name__ == "__main__":
    main()
