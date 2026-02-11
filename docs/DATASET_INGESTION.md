# Dataset Ingestion Guide

This guide describes the current workflow for adding **target datasets** (roads, sidewalks, bike, trails) that will be matched against **Overture** reference data.

## Scope

- Target datasets should be authoritative or reputable third-party sources.
- Avoid using OSM as a target source in this workflow (Overture is the default reference source).
- Configs live in `datasets/*.yaml`.

## Supported Source Types

Set `source.type` in your dataset YAML:

- `arcgis`: ArcGIS FeatureServer/MapServer endpoints
- `download`: direct file download (`shp`, `gpkg`, `gml`, `geojson`, `gdb`)
- `ogc_features`: OGC API Features item endpoint
- `wfs`: WFS endpoint (set `source.where_clause` to `typeName`)
- `os_downloads`: Ordnance Survey Data Hub products
- `manual`: metadata-only entry for datasets that need manual acquisition

## Minimal YAML Template

```yaml
name: xx_example_sidewalks
display_name: Example Sidewalk Network
type: sidewalk

description: Pedestrian centerlines from Example provider.

source:
  type: arcgis
  url: https://example.org/arcgis/rest/services/Sidewalks/FeatureServer/0
  portal_url: https://example.org/open-data

fetch:
  id_prefix: ex_sidewalk
  id_column: OBJECTID
  name_column: NAME
  class_column: TYPE
  class_mapping:
    SIDEWALK: footway
    CROSSWALK: footway
  subclass_column: TYPE
  subclass_mapping:
    SIDEWALK: sidewalk
    CROSSWALK: crosswalk
  bbox: [xmin, ymin, xmax, ymax]
  crs: EPSG:4326
```

## Important Fields

### `name`
- Stable dataset ID used by CLI and filenames.

### `type`
- One of `road`, `sidewalk`, `bike`, `trail`.
- Affects default class fallback behavior:
  - `sidewalk` defaults to `footway` when no class column is provided.
  - `bike` defaults to `cycleway` when no class column is provided.

### `fetch.id_column` (required)
- Must point to a stable upstream identifier.
- Required to preserve label linkage across refreshes.

### `fetch.bbox`
- Used for target fetch filtering (where supported) and for reference fetch extent.
- Format: `[xmin, ymin, xmax, ymax]` in WGS84 coordinates.

### `fetch.source_crs`
- Use when source data CRS metadata is missing/wrong.
- The fetch pipeline reprojects to `EPSG:4326`.

### `fetch.polygon_to_centerline`
- Use when source geometries are polygons but target needs linear network extraction.

### `source.where_clause` for WFS
- For `source.type: wfs`, set `source.where_clause` to the WFS `typeName`.

## End-to-End Workflow

## 1) Add YAML config

Create `datasets/<dataset_name>.yaml` using the template above.

## 2) Verify config

```bash
# verify one dataset
matcher data fetch verify xx_example_sidewalks

# verify everything under a prefix
matcher data fetch verify --prefix xx_

# schema-only check (no URL checks)
matcher data fetch verify xx_example_sidewalks --dry-run
```

## 3) Fetch target data

```bash
matcher data fetch target xx_example_sidewalks

# force refresh
matcher data fetch target xx_example_sidewalks --force

# fetch a set
matcher data fetch target --prefix xx_ --workers 4
```

## 4) Fetch reference data (Overture)

```bash
matcher data fetch reference xx_example_sidewalks -s overture
```

Or fetch target + reference together:

```bash
matcher data fetch all xx_example_sidewalks
```

## 5) Run matching

```bash
matcher match \
  data/raw/xx_example_sidewalks_overture_segments_*.parquet \
  data/raw/xx_example_sidewalks_target_*.parquet \
  -o data/output/xx_example_sidewalks_bridge.parquet
```

Tip: use `matcher data fetch list` and filenames in `data/raw/` to pick exact file paths.

## 6) Discover class mappings

```bash
matcher class discover \
  data/raw/xx_example_sidewalks_target_*.parquet \
  --reference data/raw/xx_example_sidewalks_overture_segments_*.parquet \
  --bridge data/output/xx_example_sidewalks_bridge.parquet
```

This updates/merges classification sections in `datasets/<dataset>.yaml`.

## 7) (Optional) Save quality fingerprint

```bash
matcher data quality xx_example_sidewalks --save-yaml
```

This writes `quality_fingerprint` into the dataset YAML for regression checks.

## Common Pitfalls

### Missing `id_column`
- Fetch fails by design.
- Fix by using a stable upstream ID field.

### CRS mismatch
- Set `fetch.source_crs` explicitly.

### Unexpected geometry types
- Non-line geometries are filtered out.
- Use `polygon_to_centerline: true` when source data is polygon-based pathways.

### WFS typeName errors
- Ensure `source.where_clause` exactly matches WFS `typeName`.

### Huge datasets
- Keep `bbox` focused unless intentionally running national-scale experiments.
- For large downloads, enable:
  - `source.cache_download: true`
  - `source.cache_ttl_hours: <hours>`

## Quick Checklist

- YAML added in `datasets/`
- `source.type` and URL/typeName validated
- Stable `fetch.id_column` set
- `bbox` present and reasonable
- `matcher data fetch verify` passes
- `matcher data fetch target` succeeds
- `matcher data fetch reference -s overture` succeeds
- `matcher match` runs and produces bridge output
