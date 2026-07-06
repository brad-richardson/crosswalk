-- Seattle sidewalk observations -> crosswalk bridge -> Overture geometry.
--
-- One query takes a city dataset that knows nothing about Overture (SDOT's
-- open sidewalk defect observations, keyed by SDOT sidewalk asset IDs) and
-- paints it onto Overture Maps segment geometry via the crosswalk bridge
-- table. Runnable in the DuckDB CLI from the repo root, or via
-- scripts/demo_seattle_sidewalk_join.py (which also fetches the inputs).
--
-- See docs/examples/seattle-sidewalk-join-demo.md for the full story,
-- coverage numbers, and caveats.

INSTALL httpfs;
LOAD httpfs;
INSTALL spatial;
LOAD spatial;
SET s3_region = 'us-west-2';

-- 1) The city's operational data: OPEN sidewalk defect observations.
--    Downloaded verbatim from Seattle's public ArcGIS service (see the demo
--    script). Keyed by SIDEWALK_UNITID — an SDOT asset ID, not a map ID.
CREATE OR REPLACE TEMP VIEW obs AS
SELECT
    SIDEWALK_UNITID AS unitid,
    OBSERVATION_TYPE,
    UPLIFT_HEIGHT
FROM read_parquet('data/demo/seattle_sidewalk_observations.parquet')
WHERE OBSERVATION_STATUS = 'OPEN';

-- 2) The ID sidecar: crosswalk local_id -> stable SDOT keys, derived from the
--    exact SDOT sidewalk snapshot the bridge was built on. Needed because the
--    bridge's local_id embeds the snapshot-time ArcGIS OBJECTID, which SDOT
--    reassigns between publishes (see "The ID caveat" in the demo doc).
CREATE OR REPLACE TEMP VIEW ids AS
SELECT local_id, unitid, compkey, unitdesc
FROM read_parquet('data/demo/seattle_sidewalk_ids.parquet');

-- 3) The crosswalk bridge table (ID-only: local_id <-> gers_id + confidence).
--    Local path today; once us_seattle_sidewalks is published this line becomes
--    the R2 URL (note: release= is the BRIDGE release, not the Overture release
--    in the S3 path below), e.g.:
--    read_parquet('https://pub-1960acc8b68148ac82da2fd033be804f.r2.dev/bridges/release=2026-01-21.0/dataset=us_seattle_sidewalks/bridge.parquet')
CREATE OR REPLACE TEMP VIEW bridge AS
SELECT local_id, gers_id, confidence, match_type, gers_start_frac, gers_end_frac
FROM read_parquet('data/output/us_seattle_sidewalks_bridge.parquet')
WHERE match_decision = 'match';

-- 4) The open map: Overture transportation segments, read live from public S3
--    (HTTP range reads; the bbox predicate prunes to Seattle row groups).
--    Overture release: check https://docs.overturemaps.org for the latest —
--    GERS ids are stable across releases, so it need not match the bridge release.
CREATE OR REPLACE TEMP TABLE ovt AS
SELECT id, names.primary AS name, class, geometry
FROM read_parquet(
    's3://overturemaps-us-west-2/release/2026-06-17.0/theme=transportation/type=segment/*',
    hive_partitioning = true)
WHERE bbox.xmin BETWEEN -122.47 AND -122.20
  AND bbox.ymin BETWEEN 47.47 AND 47.80
  AND subtype = 'road';

-- The join: city defects -> stable SDOT key -> local_id -> gers_id -> geometry.
-- Aggregated per Overture segment; geometry clipped to the matched extent via
-- the bridge's gers_*_frac columns (ST_LineSubstring).
CREATE OR REPLACE TEMP TABLE hazards_by_gers AS
WITH obs_per_sidewalk AS (
    SELECT
        unitid,
        count(*)                                            AS n_open_obs,
        count(*) FILTER (WHERE OBSERVATION_TYPE = 'HEIGHTDIFF')   AS n_trip_hazards,
        max(UPLIFT_HEIGHT)                                  AS max_uplift_in,
        count(*) FILTER (WHERE OBSERVATION_TYPE = 'OBSTRUCT')     AS n_obstructions,
        count(*) FILTER (WHERE OBSERVATION_TYPE = 'SURFCOND')     AS n_surface_defects,
        count(*) FILTER (WHERE OBSERVATION_TYPE = 'XSLOPE')       AS n_cross_slope
    FROM obs
    GROUP BY unitid
)
SELECT
    b.gers_id,
    any_value(s.name)                                       AS overture_name,
    any_value(s.class)                                      AS overture_class,
    -- Overture footways are mostly unnamed; keep SDOT's human-readable
    -- location ("X AVE BETWEEN Y ST AND Z ST, NE SIDE") for legibility.
    any_value(i.unitdesc)                                   AS example_location,
    count(DISTINCT i.unitid)                                AS n_sidewalks,
    sum(o.n_open_obs)                                       AS n_open_obs,
    sum(o.n_trip_hazards)                                   AS n_trip_hazards,
    max(o.max_uplift_in)                                    AS max_uplift_in,
    sum(o.n_obstructions)                                   AS n_obstructions,
    sum(o.n_surface_defects)                                AS n_surface_defects,
    sum(o.n_cross_slope)                                    AS n_cross_slope,
    min(b.confidence)                                       AS min_confidence,
    -- Clip to the union of matched extents along the GERS segment.
    ST_AsGeoJSON(ST_LineSubstring(
        any_value(s.geometry),
        least(min(b.gers_start_frac), max(b.gers_end_frac)),
        greatest(min(b.gers_start_frac), max(b.gers_end_frac)))) AS geojson
FROM obs_per_sidewalk o
JOIN ids    i USING (unitid)
JOIN bridge b USING (local_id)
JOIN ovt    s ON s.id = b.gers_id
GROUP BY b.gers_id;

-- A taste of the result: the ten segments with the most open trip hazards.
SELECT gers_id, example_location, n_open_obs, n_trip_hazards, round(max_uplift_in, 2) AS max_uplift_in
FROM hazards_by_gers
ORDER BY n_trip_hazards DESC, n_open_obs DESC
LIMIT 10;
