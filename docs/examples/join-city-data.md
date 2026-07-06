# Worked example: join a city's data to the open map

This is the payoff of a crosswalk **bridge table**. A city keys its operational
data — pavement condition, crash records, permits, curb rules, 311 — to its own
local street IDs. A bridge table maps `local_id → gers_id`, so that locally-keyed
data becomes joinable to [Overture Maps](https://overturemaps.org/) geometry (and
to every *other* city's data on the same GERS ids) in one SQL query.

> **Want the full worked demo with real city data?** See
> [seattle-sidewalk-join-demo.md](seattle-sidewalk-join-demo.md): Seattle's
> live sidewalk defect queue (151,898 open ADA observations) joined through the
> `us_seattle_sidewalks` bridge onto live Overture geometry — runnable script,
> measured coverage, interactive map. This page explains the generic pattern
> and schema.

Everything here runs in [DuckDB](https://duckdb.org/) and reads Overture's public
S3 release directly — no local Overture download.

## The bridge-table schema

A published bridge table is **ID-only** (it carries no geometry). Columns, from
`src/crosswalk/resolution/bridge.py`:

| Column | Type | Meaning |
|--------|------|---------|
| `local_id` | string | The city's own segment ID (the join key into *your* data) |
| `gers_id` | string | Overture GERS segment id (the join key into the open map) |
| `confidence` | float64 | Calibrated P(match) — pick your own precision/recall cutoff |
| `match_type` | string | `1:1`, `1:N`, `N:1`, `M:N` |
| `match_method` | string | `rule`, `xgboost`, `gnn` |
| `match_decision` | string | `match`, `review`, or `no_match` |
| `matched_at` | timestamp | Provenance |
| `pipeline_version` | string | Provenance |
| `gers_start_frac`, `gers_end_frac` | float64 | Where the match starts/ends **along the GERS segment** (0–1) |
| `local_start_frac`, `local_end_frac` | float64 | Where the match starts/ends **along the local segment** (0–1) |

Filter to `match_decision = 'match'` for the headline join; `review` rows are
lower-confidence and shipped for completeness.

## The join, in one query

Given your city dataset keyed by `local_id` (here: an illustrative pavement-condition
table), join through the bridge to Overture geometry. Two things make this fast and
correct, both easy to get wrong:

1. **The bbox filter.** Overture's transportation theme covers the whole planet
   (hundreds of millions of segments), but its files are spatially sorted, so a
   bounding-box filter on the `bbox` struct lets DuckDB skip almost all of the file
   and download only the parts covering your city (predicate pushdown — see
   [Overture's DuckDB guide](https://docs.overturemaps.org/getting-data/duckdb/)).
2. **Two different release strings.** The bridge URL uses the *bridge* release
   (the Overture snapshot the bridge was built on); the S3 path uses a *current*
   Overture release, because Overture's bucket only keeps recent releases. GERS ids
   are stable across releases (98.9% of this bridge's ids resolve five months
   later), so mixing them is fine — never reuse the bridge release in the S3 path.

```sql
INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;
SET s3_region = 'us-west-2';

WITH bridge AS (
  SELECT local_id, gers_id, confidence, gers_start_frac, gers_end_frac
  FROM read_parquet('https://pub-1960acc8b68148ac82da2fd033be804f.r2.dev/bridges/release=2026-01-21.0/dataset=us_montana_missoula/bridge.parquet')
  WHERE match_decision = 'match'
)
SELECT
  city.local_id,
  city.pci,                              -- your local attribute
  s.names.primary        AS overture_name,
  b.gers_id,
  b.confidence,
  ST_AsText(s.geometry)  AS overture_wkt
FROM my_city_pavement city               -- your data, keyed by local_id
JOIN bridge b            USING (local_id)
JOIN read_parquet(
  -- Overture release: check https://docs.overturemaps.org for the latest.
  's3://overturemaps-us-west-2/release/2026-06-17.0/theme=transportation/type=segment/*',
  hive_partitioning = true
) s ON s.id = b.gers_id
-- bbox: Missoula, MT. Swap bbox + dataset together for another city —
-- e.g. Seattle: xmin -122.44 / -122.22, ymin 47.49 / 47.74.
WHERE s.bbox.xmin BETWEEN -114.41 AND -113.68
  AND s.bbox.ymin BETWEEN 46.72 AND 47.14;
```

Published bridge tables live at the host configured in the [live browser](../PUBLISHING.md)
(`site/config.js`); `us_montana_missoula` and `us_usfs_flathead` are published today —
see [`datasets/licenses.toml`](../../datasets/licenses.toml) for the full registry.
Swap the `dataset=` (and the bbox) for yours.

Measured on this dataset (2026-07-06, DuckDB CLI, residential connection): the
bbox-filtered join reads **28,356 of 346,341,579 rows** (0.008%) from the Overture
side and finishes in **19 s**. The same join without the bbox filter scans the
planet-wide theme: **488 s** — 25x slower for identical results.

## A version you can run right now

Everything below is live: the bridge parquet is read from the public R2 bucket, the
geometry from Overture's public S3. Only the `pci` table (the "your city data" side)
is made up — its `local_id`s are real keys from the published bridge.

```sql
INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;
SET s3_region = 'us-west-2';

WITH pci(local_id, pci) AS (             -- illustrative city dataset
  VALUES ('us_montana_missoula_157847_8828985347', 72),  -- Montana Street
         ('us_montana_missoula_158084_8828985347', 58),  -- Idaho Street
         ('us_montana_missoula_155610_8828985341', 81),  -- Dakota Street
         ('us_montana_missoula_167136_88289e260b', 44)   -- Skyline Drive (partial match)
),
bridge AS (
  SELECT local_id, gers_id, confidence, gers_start_frac, gers_end_frac
  FROM read_parquet('https://pub-1960acc8b68148ac82da2fd033be804f.r2.dev/bridges/release=2026-01-21.0/dataset=us_montana_missoula/bridge.parquet')
  WHERE match_decision = 'match'
),
ovt AS (
  SELECT id, names.primary AS name, geometry
  FROM read_parquet(
    -- Overture release: check https://docs.overturemaps.org for the latest.
    's3://overturemaps-us-west-2/release/2026-06-17.0/theme=transportation/type=segment/*',
    hive_partitioning = true)
  -- bbox: Missoula, MT — this filter is why the query touches only a sliver
  -- of the planet-wide file (predicate pushdown on Overture's bbox struct):
  WHERE bbox.xmin BETWEEN -114.41 AND -113.68
    AND bbox.ymin BETWEEN 46.72 AND 47.14
)
SELECT p.local_id, p.pci, s.name AS overture_name, b.gers_id,
  round(b.confidence, 2) AS confidence,
  ST_AsText(ST_LineSubstring(s.geometry, b.gers_start_frac, b.gers_end_frac)) AS matched_wkt
FROM pci p
JOIN bridge b USING (local_id)
JOIN ovt s ON s.id = b.gers_id
ORDER BY p.local_id;
```

Real output (bridge release `2026-01-21.0`, Overture release `2026-06-17.0`; only
`pci` illustrative, everything else live):

```
┌───────────────────────────────────────┬───────┬────────────────────────────────┬──────────────────────────────────────┬────────────┬─────────────────────────────────────────────────┐
│               local_id                │  pci  │         overture_name          │               gers_id                │ confidence │                   matched_wkt                   │
├───────────────────────────────────────┼───────┼────────────────────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────┤
│ us_montana_missoula_155610_8828985341 │    81 │ Dakota Street                  │ d209d333-0d1c-4d68-a88b-1198198ad52f │        1.0 │ LINESTRING (-114.0177103 46.8706429, …)         │
│ us_montana_missoula_157847_8828985347 │    72 │ Montana Street                 │ 9e685568-de2c-40bb-9b41-8945e4406e87 │        1.0 │ LINESTRING (-114.0171259636 46.8729072013, …)   │
│ us_montana_missoula_158084_8828985347 │    58 │ Idaho Street                   │ 9a7afdd7-397f-4ed0-a01e-f4b0b6d30ffa │       0.99 │ LINESTRING (-114.0171131099 46.8738709650, …)   │
│ us_montana_missoula_167136_88289e260b │    44 │ Skyline Drive (North) - 2127-1 │ 0c2720f4-c0bb-41e5-830e-ac94ab2a86ae │        1.0 │ LINESTRING (-113.9249344684 46.8023456729, …)   │
└───────────────────────────────────────┴───────┴────────────────────────────────┴──────────────────────────────────────┴────────────┴─────────────────────────────────────────────────┘
```

(WKT abbreviated with `…` for width; the query returns the full coordinate list.)

## Why the `*_frac` columns matter

Datasets rarely segment the network the same way. A single local segment can
correspond to only *part* of a longer Overture segment, or vice versa — so a raw
`gers_id` join would drape your attribute over geometry that extends past the real
overlap. The `gers_start_frac` / `gers_end_frac` columns are the fix: they give
the matched sub-portion as fractions along the GERS segment, and
`ST_LineSubstring(geometry, start, end)` clips to exactly that stretch. In the run
above, the Skyline Drive row (`0.16`–`0.49`) returns a **clipped** LINESTRING
covering only a third of the Overture segment — which is what you want when
apportioning a local attribute onto a partially-overlapping map segment. Use
`local_start_frac` / `local_end_frac` for the symmetric clip onto *your* geometry.

## Licensing

Overture geometry is ODbL 1.0 (© OpenStreetMap contributors). Published bridge
tables are derived works of both Overture and the local source, and are only
published once the source license is cleared — see
[`datasets/licenses.toml`](../../datasets/licenses.toml) and
[docs/PUBLISHING.md](../PUBLISHING.md).
