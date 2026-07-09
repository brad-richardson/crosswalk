"""Re-key the us_seattle_sidewalks TARGET parquet from ArcGIS OBJECTID to stable
SDOT COMPKEY, the companion to ``migrate_seattle_compkey.py`` (which re-keys the
label stores).

Why
---
The target ``id`` embeds the upstream key: ``sea_sidewalk_{OBJECTID}_{h3}``.
SDOT reassigns ``OBJECTID`` on every ArcGIS republish, so an OBJECTID-keyed
target rots against any label made against a later vintage. ``COMPKEY`` is the
city's stable asset key. ``datasets/us_seattle_sidewalks.yaml`` already sets
``fetch.id_column: COMPKEY`` so *future* fetches are COMPKEY-keyed; this script
recovers the already-fetched 2026-02-08 snapshot in place, without a re-fetch
(a live re-fetch would pull newer geometry and change the ``h3`` suffix, which
would NOT match the migrated COMPKEY labels — verified: the offline re-key
matches 458/458 COMPKEY label ids exactly).

The mapping is read from the snapshot's own ``source_tags`` (both OBJECTID and
COMPKEY are retained there). The ``h3`` suffix is derived from the geometry
midpoint and the geometry is unchanged, so the suffix is preserved verbatim.

Safety
------
  * ``--dry-run`` (default) reports the re-key counts without writing.
  * ``--apply`` writes, creating a ``.objectid.bak`` backup of the original
    OBJECTID snapshot first (this backup is the OBJECTID->COMPKEY join table
    that ``migrate_seattle_compkey.py --snapshot`` needs, so keep it).
  * Any OBJECTID missing from the mapping is left UNCHANGED and reported.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd

_ID_RE = re.compile(r"^(?P<prefix>sea_sidewalk)_(?P<uid>\d+)_(?P<h3>[0-9a-fA-F]+)$")


def build_objectid_to_compkey(gdf: gpd.GeoDataFrame) -> dict[int, int]:
    tags = pd.DataFrame(list(gdf["source_tags"]))
    for col in ("OBJECTID", "COMPKEY"):
        if col not in tags.columns:
            raise ValueError(f"source_tags missing {col}")
    objectid = tags["OBJECTID"].astype("int64")
    compkey = tags["COMPKEY"]
    if compkey.isna().any():
        raise ValueError(f"{int(compkey.isna().sum())} rows have a null COMPKEY")
    compkey = compkey.astype("int64")
    if objectid.nunique() != len(objectid):
        raise ValueError("OBJECTID is not unique")
    if compkey.nunique() != len(compkey):
        raise ValueError("COMPKEY is not unique -- re-key would collide")
    return dict(zip(objectid, compkey))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--target",
        type=Path,
        default=Path("data/raw/us_seattle_sidewalks_v1.0.parquet"),
        help="OBJECTID-keyed target parquet to re-key in place",
    )
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    args = ap.parse_args()

    gdf = gpd.read_parquet(args.target)
    mapping = build_objectid_to_compkey(gdf)
    print(f"OBJECTID->COMPKEY mapping: {len(mapping)} entries")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}\n")

    remapped = unchanged = 0
    unmapped: dict[int, int] = {}

    def rekey(idv: str) -> str:
        nonlocal remapped, unchanged
        m = _ID_RE.match(str(idv))
        if not m:
            unchanged += 1
            return idv
        c = mapping.get(int(m.group("uid")))
        if c is None:
            unmapped[int(m.group("uid"))] = unmapped.get(int(m.group("uid")), 0) + 1
            unchanged += 1
            return idv
        remapped += 1
        return f"{m.group('prefix')}_{c}_{m.group('h3')}"

    new_id = gdf["id"].map(rekey)
    print(f"{args.target.name}: {remapped} ids re-keyed, {unchanged} unchanged ({len(gdf)} rows)")
    if unmapped:
        print(f"WARNING: {len(unmapped)} OBJECTIDs had no COMPKEY (left unchanged)")

    if args.apply:
        bak = args.target.with_suffix(args.target.suffix + ".objectid.bak")
        if not bak.exists():
            shutil.copy2(args.target, bak)
            print(f"backed up original OBJECTID snapshot -> {bak.name}")
        gdf["id"] = new_id
        gdf.to_parquet(args.target, write_covering_bbox=True)
        print(f"wrote re-keyed target -> {args.target.name}")


if __name__ == "__main__":
    main()
