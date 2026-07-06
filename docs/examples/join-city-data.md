# Worked example: join a city's data to the open map

This is the payoff of a crosswalk **bridge table**. A city keys its operational
data — pavement condition, crash records, permits, curb rules, 311 — to its own
local street IDs. A bridge table maps `local_id → gers_id`, so that locally-keyed
data becomes joinable to [Overture Maps](https://overturemaps.org/) geometry (and
to every *other* city's data on the same GERS ids) in one SQL query.

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
table), join through the bridge to Overture geometry:

```sql
INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;
SET s3_region = 'us-west-2';

WITH bridge AS (
  SELECT local_id, gers_id, confidence, gers_start_frac, gers_end_frac
  FROM read_parquet('https://<bridge-host>/bridges/release=2026-06-17.0/dataset=us_montana_missoula/bridge.parquet')
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
  's3://overturemaps-us-west-2/release/2026-06-17.0/theme=transportation/type=segment/*',
  hive_partitioning = true
) s ON s.id = b.gers_id;
```

Published bridge tables live at the host configured in the [live browser](../PUBLISHING.md)
(`site/config.js`); `us_montana_missoula` and `us_usfs_flathead` are among the
approved-license (US-PD) datasets and the first two staged for publication — see
[`datasets/licenses.toml`](../../datasets/licenses.toml) for the full registry.
Swap the `<bridge-host>` and `dataset=` for yours.

## A version you can run right now

Published bridge hosting is not live yet, so the block below inlines a tiny
**illustrative** bridge instead of reading the parquet — but it is otherwise a
real, runnable query: the `gers_id`s are **real** Overture ids for those Missoula
streets, and the geometry column is read **live from Overture's public S3**. Only
`local_id` and `pci` (the "city dataset" side) are made up.

```sql
INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;
SET s3_region = 'us-west-2';

WITH bridge(local_id, gers_id, confidence, match_decision, gers_start_frac, gers_end_frac) AS (
  VALUES
    ('MT_MSL_10420', '9e685568-de2c-40bb-9b41-8945e4406e87', 0.98, 'match', 0.00, 1.00),
    ('MT_MSL_10421', '9a7afdd7-397f-4ed0-a01e-f4b0b6d30ffa', 0.95, 'match', 0.12, 0.87),
    ('MT_MSL_10422', 'd209d333-0d1c-4d68-a88b-1198198ad52f', 0.91, 'match', 0.00, 1.00)
),
pci(local_id, pci) AS (                  -- illustrative city dataset
  VALUES ('MT_MSL_10420', 72), ('MT_MSL_10421', 58), ('MT_MSL_10422', 81)
),
ovt AS (
  SELECT id, names.primary AS name, geometry
  FROM read_parquet(
    's3://overturemaps-us-west-2/release/2026-06-17.0/theme=transportation/type=segment/*',
    hive_partitioning = true)
  -- bbox prune so the query touches only Missoula row groups (fast, cheap):
  WHERE bbox.xmin BETWEEN -114.05 AND -113.95
    AND bbox.ymin BETWEEN 46.84 AND 46.90
)
SELECT b.local_id, p.pci, s.name AS overture_name, b.gers_id, b.confidence,
  ST_AsText(ST_LineSubstring(s.geometry, b.gers_start_frac, b.gers_end_frac)) AS matched_wkt
FROM bridge b
JOIN pci p USING (local_id)
JOIN ovt s ON s.id = b.gers_id
ORDER BY b.local_id;
```

Real output (Overture release `2026-06-17.0`; `local_id`/`pci` illustrative,
everything else live):

```
┌──────────────┬───────┬────────────────┬──────────────────────────────────────┬────────────┬────────────────────────────────────────────────────────────┐
│   local_id   │  pci  │ overture_name  │               gers_id                │ confidence │                          matched_wkt                         │
├──────────────┼───────┼────────────────┼──────────────────────────────────────┼────────────┼────────────────────────────────────────────────────────────┤
│ MT_MSL_10420 │    72 │ Montana Street │ 9e685568-de2c-40bb-9b41-8945e4406e87 │       0.98 │ LINESTRING (-114.0171123 46.8729073, …, -114.0185093 46.8729156)               │
│ MT_MSL_10421 │    58 │ Idaho Street   │ 9a7afdd7-397f-4ed0-a01e-f4b0b6d30ffa │       0.95 │ LINESTRING (-114.01726951600182 46.87387057477697, …, -114.01833774044972 46.873867698594005) │
│ MT_MSL_10422 │    81 │ Dakota Street  │ d209d333-0d1c-4d68-a88b-1198198ad52f │       0.91 │ LINESTRING (-114.0177103 46.8706429, …, -114.0185166 46.870644)                │
└──────────────┴───────┴────────────────┴──────────────────────────────────────┴────────────┴────────────────────────────────────────────────────────────┘
```

(WKT abbreviated with `…` for width; the query returns the full coordinate list.)

## Why the `*_frac` columns matter

Datasets rarely segment the network the same way. A single local segment can
correspond to only *part* of a longer Overture segment, or vice versa — so a raw
`gers_id` join would drape your attribute over geometry that extends past the real
overlap. The `gers_start_frac` / `gers_end_frac` columns are the fix: they give
the matched sub-portion as fractions along the GERS segment, and
`ST_LineSubstring(geometry, start, end)` clips to exactly that stretch. In the run
above, `MT_MSL_10421` (Idaho Street, `0.12`–`0.87`) returns a **clipped**
LINESTRING, not the whole Overture segment — which is what you want when
apportioning a local attribute onto a partially-overlapping map segment. Use
`local_start_frac` / `local_end_frac` for the symmetric clip onto *your* geometry.

## Licensing

Overture geometry is ODbL 1.0 (© OpenStreetMap contributors). Published bridge
tables are derived works of both Overture and the local source, and are only
published once the source license is cleared — see
[`datasets/licenses.toml`](../../datasets/licenses.toml) and
[docs/PUBLISHING.md](../PUBLISHING.md).
