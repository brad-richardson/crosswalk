"""Re-key us_seattle_sidewalks labels from ArcGIS OBJECTID to stable SDOT COMPKEY.

Background
----------
Target ``local_id``s embed the upstream id: ``sea_sidewalk_{OBJECTID}_{h3}``.
SDOT reassigns ``OBJECTID`` when the ArcGIS layer is republished (the Seattle
sidewalk demo proved COMPKEY 658573 moved from OBJECTID 27149648 at the
2026-02-08 snapshot to 31932607 live), so every published ``local_id`` rots on
the next fetch. ``COMPKEY`` is the stable asset-management key the city's own
datasets join on.

This script rewrites the ``OBJECTID`` embedded in every Seattle label target id
to the corresponding ``COMPKEY``, deriving the mapping **from the old snapshot**
(``data/raw/us_seattle_sidewalks_v1.0.parquet``, which retains both columns in
``source_tags``). The h3 suffix is derived from the geometry midpoint and the
geometry is unchanged for the same feature, so the stored suffix is preserved
verbatim -- it is NOT recomputed.

Stores rewritten (per dataset):
  * labels/human/dataset=*/data.csv         -- ``target_id`` column
  * labels/agent/dataset=*/data.csv         -- ``target_id`` column (if present)
  * labels/features/dataset=*/data.parquet  -- ``target_id`` column
  * labels/data/dataset=*/data.parquet      -- ``target_id`` column
  * labels/stitching/dataset=*/data.csv     -- ``target_id`` inside ``selected_edges``
                                               JSON, plus the ``target_ids`` JSON array

Safety:
  * ``--dry-run`` (default) reports what would change without writing.
  * ``--apply`` writes, creating ``.bak`` backups first.
  * Any OBJECTID missing from the mapping is reported and left UNCHANGED -- no
    label rows are ever dropped or guessed.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd

# sea_sidewalk_{upstream_id}_{h3suffix}
_ID_RE = re.compile(r"^(?P<prefix>.+)_(?P<uid>\d+)_(?P<h3>[0-9a-fA-F]+)$")


def build_objectid_to_compkey(snapshot_path: Path) -> dict[int, int]:
    """Build the OBJECTID -> COMPKEY mapping from the old snapshot's source_tags."""
    gdf = gpd.read_parquet(snapshot_path)
    tags = pd.DataFrame(list(gdf["source_tags"]))
    for col in ("OBJECTID", "COMPKEY"):
        if col not in tags.columns:
            raise ValueError(f"{snapshot_path} source_tags missing {col}")
    objectid = tags["OBJECTID"].astype("int64")
    compkey = tags["COMPKEY"]
    if compkey.isna().any():
        n = int(compkey.isna().sum())
        raise ValueError(f"{n} rows have a null COMPKEY -- cannot build a clean mapping")
    compkey = compkey.astype("int64")
    mapping = dict(zip(objectid, compkey))
    if len(mapping) != len(objectid):
        raise ValueError("OBJECTID is not unique in the snapshot")
    # COMPKEY must be unique for a lossless re-key
    if compkey.nunique() != len(compkey):
        raise ValueError("COMPKEY is not unique in the snapshot -- re-key would collide")
    return mapping


class Rekeyer:
    def __init__(self, mapping: dict[int, int]):
        self.mapping = mapping
        self.unmapped: dict[int, int] = {}  # objectid -> count of occurrences
        self.remapped = 0
        self.unchanged = 0

    def rekey_id(self, target_id: str) -> str:
        m = _ID_RE.match(str(target_id))
        if not m:
            self.unchanged += 1
            return target_id
        oid = int(m.group("uid"))
        compkey = self.mapping.get(oid)
        if compkey is None:
            self.unmapped[oid] = self.unmapped.get(oid, 0) + 1
            self.unchanged += 1
            return target_id
        self.remapped += 1
        return f"{m.group('prefix')}_{compkey}_{m.group('h3')}"


def _backup_and_write(path: Path, writer, apply: bool) -> None:
    if not apply:
        return
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
    writer(path)


# Whole-token id occurrence in raw text (CSV cells / JSON blobs). The prefix and
# h3 suffix are captured so only the embedded upstream number is rewritten,
# leaving every other byte (float formatting, quoting, ordering) untouched.
_TEXT_ID_RE = re.compile(r"(sea_sidewalk)_(\d+)_([0-9a-fA-F]+)")


def migrate_csv_text(path: Path, rk: Rekeyer, apply: bool) -> None:
    """Rewrite embedded ids in a CSV by raw-text substitution.

    Operating on text (not a pandas round-trip) guarantees the diff is *only*
    the id renumbering -- no float reformatting (``0`` -> ``0.0``), no requoting.
    Applies to every ``sea_sidewalk_*`` token regardless of column, which
    covers ``target_id`` plus the ``selected_edges``/``target_ids`` JSON blobs
    in the stitching store (``ref_ids`` are UUIDs and never match).
    """
    text = path.read_text()
    before = rk.remapped

    def _sub(m: re.Match) -> str:
        oid = int(m.group(2))
        compkey = rk.mapping.get(oid)
        if compkey is None:
            rk.unmapped[oid] = rk.unmapped.get(oid, 0) + 1
            return m.group(0)
        rk.remapped += 1
        return f"{m.group(1)}_{compkey}_{m.group(3)}"

    new_text = _TEXT_ID_RE.sub(_sub, text)
    print(f"  {path.name}: {rk.remapped - before} embedded ids rewritten")
    _backup_and_write(path, lambda p: p.write_text(new_text), apply)


def migrate_parquet_target_id(path: Path, rk: Rekeyer, apply: bool) -> None:
    df = gpd.read_parquet(path) if _is_geo(path) else pd.read_parquet(path)
    if "target_id" not in df.columns:
        print(f"  {path.name}: no target_id column, skipped")
        return
    before = rk.remapped
    df["target_id"] = df["target_id"].map(rk.rekey_id)
    print(f"  {path.name}: {rk.remapped - before} target_id values rewritten ({len(df)} rows)")

    def _write(p: Path) -> None:
        if isinstance(df, gpd.GeoDataFrame):
            df.to_parquet(p, write_covering_bbox=True)
        else:
            df.to_parquet(p)

    _backup_and_write(path, _write, apply)


def _is_geo(path: Path) -> bool:
    try:
        import pyarrow.parquet as pq

        md = pq.read_schema(path).metadata or {}
        return any(b"geo" in k for k in md)
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True, help="worktree root (has labels/)")
    ap.add_argument("--snapshot", type=Path, required=True, help="old v1.0 snapshot parquet")
    ap.add_argument("--dataset", default="us_seattle_sidewalks")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    args = ap.parse_args()

    mapping = build_objectid_to_compkey(args.snapshot)
    print(f"Loaded OBJECTID->COMPKEY mapping: {len(mapping)} entries from {args.snapshot.name}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}\n")

    rk = Rekeyer(mapping)
    labels = args.data_root / "labels"
    ds = args.dataset

    csv_stores = [
        labels / "human" / f"dataset={ds}" / "data.csv",
        labels / "agent" / f"dataset={ds}" / "data.csv",
    ]
    parquet_stores = [
        labels / "features" / f"dataset={ds}" / "data.parquet",
        labels / "data" / f"dataset={ds}" / "data.parquet",
    ]
    stitching = labels / "stitching" / f"dataset={ds}" / "data.csv"

    print("Human/agent CSV stores:")
    for p in csv_stores:
        if p.exists():
            migrate_csv_text(p, rk, args.apply)
        else:
            print(f"  {p} does not exist, skipped")

    print("Feature/data parquet stores:")
    for p in parquet_stores:
        if p.exists():
            migrate_parquet_target_id(p, rk, args.apply)
        else:
            print(f"  {p} does not exist, skipped")

    print("Stitching store (CSV with JSON edge blobs):")
    if stitching.exists():
        migrate_csv_text(stitching, rk, args.apply)
    else:
        print(f"  {stitching} does not exist, skipped")

    print(f"\nTotals: {rk.remapped} ids remapped, {rk.unchanged} left unchanged")
    if rk.unmapped:
        print(f"WARNING: {len(rk.unmapped)} OBJECTIDs had no COMPKEY mapping (left unchanged):")
        for oid, cnt in sorted(rk.unmapped.items()):
            print(f"    OBJECTID {oid}: {cnt} occurrence(s)")
    else:
        print("All referenced OBJECTIDs were mapped to a COMPKEY.")


if __name__ == "__main__":
    main()
