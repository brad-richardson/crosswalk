#!/usr/bin/env python
"""Re-key labels whose target_id (or gers_id) no longer exists in current raw data.

Two sides, two very different situations -- see ``--side``.

Why (``--side target``, the default)
------------------------------------
Target datasets get re-fetched, and some re-fetches mint a new id scheme
(``sg_singapore_roads`` labels are ``sg_road_None_…`` — the source id column was
null at label time; ``us_utah_slc_roads`` went ``…_201491_…`` -> ``…_854_…``).
Every label keyed to the old scheme is orphaned: ``crosswalk backfill`` skips it
with ``--skip-missing``, so it silently leaves the training set. As of
2026-08-07 that is 511 pairs, ~9% of the labeled base
(research/label_feature_staleness_scope_2026-08-07.md).

``labels/data/`` retains ``target_geometry`` for 100% of them, so the old id can
be mapped to the new one geometrically without re-labeling.

Why (``--side ref``)
--------------------
Bumping the Overture reference release orphans labels on the *reference* side
too: as of the 2026-07-22.0 bump, 195 labelled GERS ids (199 label rows) are
absent from the current release. These do NOT silently leave the training set —
``crosswalk backfill`` splices their stored ``ref_geometry`` back in via
``stored_ref_overrides`` — but a stale key still costs real quality: because the
spliced segment is not in the current release's connector graph, all 199 rows
come out **100% NaN** on ``graphlet_similarity``, ``endpoint_degree_similarity``,
``clustering_coef_ref`` and ``clustering_coef_delta`` (vs 0.5–17% normally), and
the missingness correlates with dataset. Re-keying restores live topology.

The fates are not what the target side sees. Measured over those 195:

===========  =====  =========================================================
fate         count  what happened
===========  =====  =========================================================
split          134  road still there, re-segmented into N pieces (median 5)
gone            43  no successor within the corridor
reshape         17  one successor, geometry edited
merge            1  absorbed into a longer segment
===========  =====  =========================================================

**Splits are deliberately refused.** A ``match`` label says "this local segment
corresponds to road A"; when A becomes A1..A5, choosing which fragments the
target actually covers is a stitching decision, and inventing it would fan 134
pair labels into ~1,280 synthetic pairs (~23% of the labelled base), unreviewed
and concentrated in whichever datasets happened to churn. Those belong in a
human re-review queue feeding ``labels/stitching/``, not in an automated re-key.

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

``--side ref`` adds a third tier, because a reference release legitimately edits
geometry (vertex densification, alignment fixes) without changing what the road
*is* — such a segment survives with a new id but fails both tiers above:

  tier 3  exactly ONE successor lies mostly inside the old segment's corridor,
          it covers >= --cov of the old segment's length, and total length is
          within --ratio-tol of the old. Two or more successors is a split and
          is refused, not resolved.

Tier 3's bar is tight on purpose: re-keying makes features recompute against
the *new* geometry, which the human never saw, so a loose bar silently mutates
what the label asserts. Of the 18 single-successor candidates in the 2026-07-22.0
bump, only 8 clear ``--cov 0.95 --ratio-tol 0.15``; the rest (down to 70% length
retained) are left alone. All three tiers together re-key 10 GERS ids / 11 label
rows of 203 — 141 of the refusals are splits, i.e. the size of the re-review
queue rather than a failure of the matcher.

Safety
------
  * ``--dry-run`` (default) reports counts and writes nothing.
  * ``--apply`` rewrites the id column across labels/human, labels/agent,
    labels/data and labels/features, backing each file up to ``.prerekey.bak``.
  * A re-key that would collide with an existing (gers_id, target_id) row in the
    same store is refused and reported.
  * Run ``crosswalk backfill`` afterwards — re-keyed rows still carry features
    computed against the OLD geometry. This is a separate step on purpose:
    backfill is a thin wrapper around the shared feature pipeline (CLAUDE.md),
    and mutating label keys is a destructive operation that wants its own
    dry-run and its own review.

Usage
-----
  uv run python scripts/rekey_orphaned_labels.py                  # dry run, all
  uv run python scripts/rekey_orphaned_labels.py -D us_utah_slc_roads
  uv run python scripts/rekey_orphaned_labels.py --tol 0.5 --apply
  uv run python scripts/rekey_orphaned_labels.py --side ref       # after a
                                                  # reference release bump
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


# Per-side wiring: which id column is re-keyed, which stored geometry column
# vouches for it, and where the current raw ids live.
SIDES = {
    "target": {
        "id_col": "target_id",
        "geom_col": "target_geometry",
        "raw": "data/raw/{ds}_v1.0.parquet",
    },
    "ref": {
        "id_col": "gers_id",
        "geom_col": "ref_geometry",
        "raw": "data/raw/{ds}_overture_segments_v1.0.parquet",
    },
}


def find_orphans(ds: str, side: str) -> tuple[pd.DataFrame, gpd.GeoDataFrame] | None:
    spec = SIDES[side]
    id_col, geom_col = spec["id_col"], spec["geom_col"]
    dpath = Path(f"{DATA_STORE}/dataset={ds}/data.parquet")
    rpath = Path(spec["raw"].format(ds=ds))
    if not dpath.exists() or not rpath.exists():
        return None
    cols = pq.ParquetFile(dpath).schema_arrow.names
    if geom_col not in cols:
        return None
    read_cols = sorted({"gers_id", "target_id", geom_col})
    d = pq.ParquetFile(dpath).read(columns=read_cols).to_pandas()
    raw = gpd.read_parquet(rpath, columns=["id", "geometry"])
    raw = raw[raw.geometry.notna()]
    orph = d[~d[id_col].isin(set(raw["id"]))].copy()
    orph = orph[orph[geom_col].notna()]
    return (orph, raw) if len(orph) else None


def _as_geom(value):
    """Stored geometry columns hold WKB bytes; DataStore hands back shapely."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return shwkb.loads(bytes(value))
    return value


def _corridor_successors(old_m, tree: STRtree, raw_m, buf: float, inside_frac: float) -> list[int]:
    """Indices of current segments lying mostly inside the old segment's corridor."""
    corridor = old_m.buffer(buf)
    kept = []
    for i in tree.query(corridor):
        g = raw_m[i]
        if g.length > 0 and g.intersection(corridor).length / g.length >= inside_frac:
            kept.append(int(i))
    return kept


def build_mapping(
    orph: pd.DataFrame,
    raw: gpd.GeoDataFrame,
    tol: float,
    *,
    id_col: str = "target_id",
    geom_col: str = "target_geometry",
    corridor: dict | None = None,
) -> tuple[dict, dict]:
    """Return (old_id -> new_id, old_id -> reason-for-no-match).

    ``corridor`` enables tier 3 (single-successor survival across a re-segmenting
    release); pass ``None`` to keep the strict two-tier identity matching.
    """
    geoms = [_as_geom(v) for v in orph[geom_col]]

    # tier 1: exact WKB, unique
    wkb_counts: dict[bytes, int] = {}
    wkb_id: dict[bytes, str] = {}
    for g, i in zip(raw.geometry, raw["id"]):
        b = g.wkb
        wkb_counts[b] = wkb_counts.get(b, 0) + 1
        wkb_id[b] = i

    mapping: dict[str, str] = {}
    unresolved: list[int] = []
    for pos, (old_id, g) in enumerate(zip(orph[id_col], geoms)):
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
            old_id = orph[id_col].iloc[pos]
            cand = tree.query(om[k].buffer(max(tol, 0.01)))
            hd = np.array([shapely.hausdorff_distance(om[k], rm[c]) for c in cand])
            within = cand[hd <= tol] if len(cand) else cand
            if len(within) == 1:
                mapping[old_id] = raw["id"].iloc[within[0]]
                continue
            if len(within) > 1:
                reasons[old_id] = f"ambiguous: {len(within)} candidates within tol"
                continue
            if corridor is None:
                reasons[old_id] = (
                    "no candidate within tolerance"
                    if len(cand) == 0
                    else f"nearest hausdorff {hd.min():.1f}m > tol"
                )
                continue
            # tier 3: did exactly one successor survive in the old corridor?
            old = om[k]
            if old.length <= 0:
                reasons[old_id] = "degenerate stored geometry"
                continue
            succ = _corridor_successors(
                old, tree, rm, corridor["buffer_m"], corridor["inside_frac"]
            )
            if not succ:
                reasons[old_id] = "gone: no successor in corridor"
                continue
            if len(succ) > 1:
                reasons[old_id] = f"split into {len(succ)} — refused (needs re-review)"
                continue
            new_g = rm[succ[0]]
            cov = old.intersection(new_g.buffer(corridor["buffer_m"])).length / old.length
            ratio = new_g.length / old.length
            if cov < corridor["cov"]:
                reasons[old_id] = f"only {cov:.0%} of old segment covered"
            elif abs(ratio - 1.0) > corridor["ratio_tol"]:
                reasons[old_id] = f"length changed {ratio:.0%} — geometry too different"
            else:
                mapping[old_id] = raw["id"].iloc[succ[0]]
    return mapping, reasons


def rewrite(
    path: Path, mapping: dict, apply: bool, key_cols: list[str], id_col: str = "target_id"
) -> tuple[int, int]:
    """Rewrite ``id_col`` in a csv/parquet store. Returns (rows_changed, collisions)."""
    if not path.exists():
        return (0, 0)
    is_csv = path.suffix == ".csv"
    df = pd.read_csv(path) if is_csv else pq.ParquetFile(path).read().to_pandas()
    if id_col not in df.columns:
        return (0, 0)
    old_vals = df[id_col]
    new = old_vals.map(lambda t: mapping.get(t, t))
    changed = int((new != old_vals).sum())
    if not changed:
        return (0, 0)
    probe = df.copy()
    probe[id_col] = new
    collisions = 0
    if all(c in probe.columns for c in key_cols):
        dup = probe.duplicated(subset=key_cols, keep=False) & (new != old_vals)
        collisions = int(dup.sum())
        if collisions:
            keep = ~((new != old_vals) & probe.duplicated(subset=key_cols, keep="first"))
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
    ap.add_argument(
        "--side",
        choices=sorted(SIDES),
        default="target",
        help="Which id to re-key: 'target' (raw target re-fetch) or 'ref' (Overture bump)",
    )
    ap.add_argument(
        "--cov",
        type=float,
        default=0.95,
        help="tier 3 (--side ref): min fraction of the old segment the successor must cover",
    )
    ap.add_argument(
        "--ratio-tol",
        type=float,
        default=0.15,
        help="tier 3 (--side ref): max fractional length change of the successor",
    )
    args = ap.parse_args()

    spec = SIDES[args.side]
    id_col = spec["id_col"]
    # Tier 3 only applies to the reference side, where a release legitimately
    # re-segments and re-shapes geometry. Target re-fetches keep strict identity.
    corridor = (
        None
        if args.side == "target"
        else {
            "buffer_m": 8.0,
            "inside_frac": 0.60,
            "cov": args.cov,
            "ratio_tol": args.ratio_tol,
        }
    )

    targets = args.dataset or datasets_with_labels()
    total_orph = total_map = total_coll = 0
    rows = []
    all_reasons: list[str] = []
    for ds in targets:
        found = find_orphans(ds, args.side)
        if not found:
            continue
        orph, raw = found
        mapping, reasons = build_mapping(
            orph,
            raw,
            args.tol,
            id_col=id_col,
            geom_col=spec["geom_col"],
            corridor=corridor,
        )
        all_reasons.extend(reasons.values())
        # Count ROWS, not distinct ids: a dataset can label the same target
        # several times (different gers_id), so distinct-id counts understate
        # how many label rows a mapping actually recovers.
        n_ids = orph[id_col].nunique()
        n_map_ids = len(mapping)
        n_map = int(orph[id_col].isin(mapping).sum())
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
                    id_col,
                )
                coll += x
            for store in (DATA_STORE, FEATURE_STORE):
                c, x = rewrite(
                    Path(f"{store}/dataset={ds}/data.parquet"),
                    mapping,
                    args.apply,
                    ["gers_id", "target_id"],
                    id_col,
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
    if all_reasons:
        # Bucket the refusals — on --side ref most of them are splits, and a
        # split count is the size of the human re-review queue, not a failure.
        prefixes = {
            "split into": "split — refused, needs human re-review",
            "only ": "successor covers too little of the old segment",
            "length changed": "successor length changed too much",
            "gone:": "gone — no successor in corridor",
            "ambiguous": "ambiguous — multiple candidates within tolerance",
        }
        buckets: dict[str, int] = {}
        for r in all_reasons:
            key = next((v for p, v in prefixes.items() if r.startswith(p)), r)
            buckets[key] = buckets.get(key, 0) + 1
        print("\nleft unchanged, by reason:")
        for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
            print(f"  {v:5d}  {k}")
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
