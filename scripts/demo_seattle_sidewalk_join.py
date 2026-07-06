#!/usr/bin/env python3
"""Killer-demo driver: join SDOT Sidewalk Observations onto Overture via the bridge.

The story (see docs/examples/seattle-sidewalk-join-demo.md): Seattle's sidewalk
defect inspections (trip hazards, obstructions, cross-slopes — the city's ADA
maintenance queue) are keyed by SDOT sidewalk asset IDs and know nothing about
Overture. The crosswalk bridge table makes them joinable to Overture GERS
segment geometry in one SQL query.

This script automates the three inputs the SQL needs, runs the join, and emits
the artifacts:

1. Downloads the *live* SDOT Sidewalk Observations attribute table (no
   geometry) from Seattle's public ArcGIS service
   -> data/demo/seattle_sidewalk_observations.parquet   (cached; --refetch to redo)
2. Builds the ID sidecar (crosswalk local_id -> stable SDOT keys UNITID/COMPKEY)
   from the exact sidewalk snapshot the bridge was built on
   -> data/demo/seattle_sidewalk_ids.parquet
3. Runs docs/examples/seattle/join_observations.sql in DuckDB (reads Overture
   segments live from public S3), prints honest coverage stats, and writes:
   -> data/demo/seattle_sidewalk_hazards_by_gers.parquet  (full result, not committed)
   -> docs/examples/seattle/hazards_sample.csv            (small committed sample)
   -> docs/examples/seattle/map.html                      (self-contained Leaflet map)

Run from the repo root:

    uv run python scripts/demo_seattle_sidewalk_join.py

Requires network access (SDOT ArcGIS + Overture S3) and the local bridge at
data/output/us_seattle_sidewalks_bridge.parquet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OBS_SERVICE = (
    "https://services.arcgis.com/ZOyb2t4B0UYuYNYH/arcgis/rest/services/"
    "Sidewalk_Observation/FeatureServer/0/query"
)
OBS_FIELDS = [
    "OBSERVATION_ID",
    "SIDEWALK_UNITID",
    "INSPECTION_TYPE",
    "INSPECTION_DATE",
    "OBSERVATION_STATUS",
    "OBSERVATION_TYPE",
    "OBSTRUCTION_TYPE",
    "CLEARANCE_IMPACTED",
    "MIN_WIDTH_OBS",
    "HEIGHT_DIFFERENCE_TYPE",
    "UPLIFT_HEIGHT",
    "ISOLATED_CROSS_SLOPE",
    "SURFACE_CONDITION_TYPE",
]
PAGE_SIZE = 2000

REPO_ROOT = Path(__file__).resolve().parent.parent
SQL_PATH = REPO_ROOT / "docs" / "examples" / "seattle" / "join_observations.sql"
DOCS_OUT = REPO_ROOT / "docs" / "examples" / "seattle"


def fetch_observations(out_path: Path, refetch: bool = False) -> None:
    """Page the SDOT Sidewalk Observations attribute table to parquet."""
    import pandas as pd

    if out_path.exists() and not refetch:
        print(f"[fetch] cached: {out_path} (use --refetch to re-download)")
        return
    rows: list[dict] = []
    offset = 0
    while True:
        params = urllib.parse.urlencode(
            {
                "where": "1=1",
                "outFields": ",".join(OBS_FIELDS),
                "returnGeometry": "false",
                "orderByFields": "OBJECTID",
                "resultOffset": offset,
                "resultRecordCount": PAGE_SIZE,
                "f": "json",
            }
        )
        with urllib.request.urlopen(f"{OBS_SERVICE}?{params}", timeout=120) as resp:
            payload = json.load(resp)
        if "error" in payload:
            raise RuntimeError(f"ArcGIS error at offset {offset}: {payload['error']}")
        feats = payload.get("features", [])
        rows.extend(f["attributes"] for f in feats)
        print(f"[fetch] {len(rows)} rows", end="\r", flush=True)
        if not payload.get("exceededTransferLimit") and len(feats) < PAGE_SIZE:
            break
        offset += len(feats)
        time.sleep(0.2)  # be polite to the public endpoint
    print()
    df = pd.DataFrame(rows)
    # Epoch-ms -> timestamp for date fields.
    for col in ("INSPECTION_DATE",):
        df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[fetch] wrote {len(df)} observations -> {out_path}")


def build_id_sidecar(raw_path: Path, out_path: Path) -> None:
    """local_id -> stable SDOT keys, from the snapshot the bridge was built on.

    The bridge's local_id embeds the ArcGIS OBJECTID *at snapshot time*
    (``sea_sidewalk_{OBJECTID}_{h3}``), but SDOT reassigns OBJECTIDs when the
    layer is republished — the stable asset keys are COMPKEY / UNITID, which the
    snapshot carries in source_tags. This sidecar is what makes external
    SDOT-keyed data joinable; see the demo doc's "ID caveat".
    """
    import pandas as pd

    raw = pd.read_parquet(raw_path, columns=["id", "source_tags"])
    sidecar = pd.DataFrame(
        {
            "local_id": raw["id"],
            "unitid": raw["source_tags"].map(lambda t: t.get("UNITID")),
            "compkey": raw["source_tags"].map(lambda t: t.get("COMPKEY")),
            "unitdesc": raw["source_tags"].map(lambda t: t.get("UNITDESC")),
        }
    )
    n_dup = sidecar["unitid"].duplicated().sum()
    if n_dup:
        raise RuntimeError(f"UNITID not unique in snapshot ({n_dup} dups)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar.to_parquet(out_path, index=False)
    print(f"[sidecar] wrote {len(sidecar)} id mappings -> {out_path}")


def run_join(top_n: int) -> None:
    """Execute the demo SQL and emit stats + artifacts."""
    import duckdb

    con = duckdb.connect()
    sql = SQL_PATH.read_text()
    print(f"[join] running {SQL_PATH.relative_to(REPO_ROOT)} (reads Overture from S3)...")
    con.execute(sql)

    # ---- Honest coverage stats -------------------------------------------
    stats = con.execute("""
        WITH open_obs AS (
          SELECT SIDEWALK_UNITID AS unitid FROM read_parquet('data/demo/seattle_sidewalk_observations.parquet')
          WHERE OBSERVATION_STATUS = 'OPEN'
        ),
        ids AS (SELECT * FROM read_parquet('data/demo/seattle_sidewalk_ids.parquet')),
        bridge AS (
          SELECT * FROM read_parquet('data/output/us_seattle_sidewalks_bridge.parquet')
          WHERE match_decision = 'match'
        )
        SELECT
          (SELECT count(*) FROM open_obs)                                            AS open_obs,
          (SELECT count(*) FROM open_obs JOIN ids USING (unitid))                    AS obs_in_snapshot,
          (SELECT count(*) FROM open_obs o JOIN ids i USING (unitid)
             WHERE i.local_id IN (SELECT local_id FROM bridge))                      AS obs_on_matched,
          (SELECT count(*) FROM ids)                                                 AS sidewalk_segments,
          (SELECT count(DISTINCT local_id) FROM bridge)                              AS matched_segments,
          (SELECT count(DISTINCT gers_id) FROM bridge)                               AS bridge_gers_ids,
          (SELECT count(DISTINCT b.gers_id) FROM bridge b JOIN ovt s ON s.id = b.gers_id) AS gers_ids_live,
          (SELECT count(*) FROM hazards_by_gers)                                     AS result_segments
    """).fetchone()
    (
        open_obs,
        obs_in_snapshot,
        obs_on_matched,
        sidewalk_segments,
        matched_segments,
        bridge_gers_ids,
        gers_ids_live,
        result_segments,
    ) = stats
    print(f"""
[stats] open SDOT observations:                {open_obs:>8,}
[stats]   with UNITID in bridge snapshot:      {obs_in_snapshot:>8,} ({100 * obs_in_snapshot / open_obs:.1f}%)
[stats]   on a bridge-matched sidewalk:        {obs_on_matched:>8,} ({100 * obs_on_matched / open_obs:.1f}% of all open obs)
[stats] sidewalk segments in snapshot:         {sidewalk_segments:>8,}
[stats]   bridge-matched (decision='match'):   {matched_segments:>8,} ({100 * matched_segments / sidewalk_segments:.1f}%)
[stats] distinct GERS ids in bridge:           {bridge_gers_ids:>8,}
[stats]   still resolvable in live release:    {gers_ids_live:>8,} ({100 * gers_ids_live / bridge_gers_ids:.1f}%)
[stats] Overture segments carrying hazards:    {result_segments:>8,}
""")

    # ---- Artifacts --------------------------------------------------------
    out_parquet = REPO_ROOT / "data" / "demo" / "seattle_sidewalk_hazards_by_gers.parquet"
    con.execute(
        "COPY (SELECT * EXCLUDE (geojson) FROM hazards_by_gers ORDER BY n_open_obs DESC) TO ? (FORMAT parquet)",
        [str(out_parquet)],
    )
    print(f"[out] full result -> {out_parquet}")

    sample_csv = DOCS_OUT / "hazards_sample.csv"
    con.execute(
        """
        COPY (
          SELECT gers_id, overture_name, example_location, n_sidewalks, n_open_obs,
                 n_trip_hazards, round(max_uplift_in, 2) AS max_uplift_in,
                 n_obstructions, n_surface_defects, n_cross_slope,
                 round(min_confidence, 3) AS min_confidence
          FROM hazards_by_gers ORDER BY n_trip_hazards DESC, n_open_obs DESC LIMIT 25
        ) TO ? (FORMAT csv, HEADER)
        """,
        [str(sample_csv)],
    )
    print(f"[out] sample CSV -> {sample_csv}")

    feats = con.execute(
        """
        SELECT json_object(
                 'type', 'Feature',
                 'geometry', geojson::JSON,
                 'properties', json_object(
                    'gers_id', gers_id, 'name', overture_name,
                    'loc', example_location,
                    'obs', n_open_obs, 'trip', n_trip_hazards,
                    'uplift', round(max_uplift_in, 2), 'obstr', n_obstructions,
                    'surf', n_surface_defects, 'conf', round(min_confidence, 3)))
        FROM hazards_by_gers
        ORDER BY n_open_obs DESC
        LIMIT ?
        """,
        [top_n],
    ).fetchall()
    fc = '{"type":"FeatureCollection","features":[' + ",".join(r[0] for r in feats) + "]}"
    write_map(fc, DOCS_OUT / "map.html", n_features=len(feats))


def write_map(feature_collection_json: str, out_path: Path, n_features: int) -> None:
    """Self-contained Leaflet map with the hazard GeoJSON inlined."""
    template = (REPO_ROOT / "docs" / "examples" / "seattle" / "map_template.html").read_text()
    out_path.write_text(template.replace("/*__GEOJSON__*/null", feature_collection_json))
    size_kb = out_path.stat().st_size / 1024
    print(f"[out] map with top {n_features} segments -> {out_path} ({size_kb:.0f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refetch", action="store_true", help="re-download SDOT observations")
    ap.add_argument("--top-n", type=int, default=1200, help="segments to include in the map")
    ap.add_argument("--skip-join", action="store_true", help="only fetch/build inputs")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)  # the SQL uses repo-root-relative paths
    demo_dir = REPO_ROOT / "data" / "demo"
    fetch_observations(demo_dir / "seattle_sidewalk_observations.parquet", refetch=args.refetch)
    build_id_sidecar(
        REPO_ROOT / "data" / "raw" / "us_seattle_sidewalks_v1.0.parquet",
        demo_dir / "seattle_sidewalk_ids.parquet",
    )
    if not args.skip_join:
        run_join(top_n=args.top_n)


if __name__ == "__main__":
    sys.exit(main())
