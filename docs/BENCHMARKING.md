# Benchmarking Guide

This guide explains how to benchmark matcher against external tools like Hootenanny using `cbench`.

## cbench CLI

`cbench` is the standalone benchmarking harness in `matcher/cbench/`. It runs conflation tools and evaluates their output against human-labeled ground truth.

### Installation

```bash
cd matcher/cbench
uv pip install -e ".[hootenanny,dev]"
```

### Running Benchmarks

```bash
# Run matcher on a dataset
cbench run matcher us_boston_streets \
    --labels ../labels/human \
    --reference ../data/raw/us_boston_streets_overture_segments_v1.0.parquet \
    --target ../data/raw/us_boston_streets_v1.0.parquet

# Run Hootenanny on the same dataset
cbench run hootenanny us_boston_streets \
    --labels ../labels/human \
    --reference ../data/raw/us_boston_streets_overture_segments_v1.0.parquet \
    --target ../data/raw/us_boston_streets_v1.0.parquet \
    --opt hoot_dir=../../hootenanny

# Compare results
cbench compare cbench_results.jsonl

# List available tool adapters
cbench list-tools
```

### Evaluation Modes

cbench supports two evaluation levels via `--match-level`:

- **target** (default): Target-level matching. A labeled match target is a TP if
  it appears in *any* prediction, regardless of which reference segment was chosen,
  and an FN if it appears in no prediction (TP/FN count unique labeled targets).
  Avoids penalizing tools for picking a different reference segment that covers a
  different subsegment of the same target road. False positives remain pair-exact:
  a `no_match` label asserts that one specific `(ref_id, target_id)` pair is wrong,
  so only a prediction of that exact pair counts as an FP — predicting a *different*
  reference for that target is not evidence of error.

- **pair**: Exact `(ref_id, target_id)` pair matching. More strict — a prediction
  is TP only if the exact pair appears in ground truth.

```bash
# Default: target-level evaluation
cbench run matcher us_boston_streets --labels ../labels/human ...

# Strict pair-level evaluation
cbench run matcher us_boston_streets --labels ../labels/human --match-level pair ...
```

Note that precision only counts errors against labeled ground truth — a
prediction is a false positive only if it hits an explicitly labeled
`no_match` pair, and predictions on unlabeled pairs are reported as
`unlabeled_predictions` and excluded entirely. At target level, TP/FN count
unique labeled targets rather than individual predictions. A low
`labeled_coverage` (the fraction of predictions that touched labeled ground
truth) means the metrics are measured against a small labeled subset and
should be read with caution. Labels marked `unsure` are skipped and reported
as `skipped_unsure`.

## Hootenanny Setup

[Hootenanny](https://github.com/ngageoint/hootenanny) is a vector conflation tool from NGA.

### Docker Compose (Recommended)

```bash
# Clone Hootenanny as a sibling to matcher
cd /path/to/matcher/..
git clone https://github.com/ngageoint/hootenanny.git
cd hootenanny

# Start services (first run builds everything - takes 20-40 min)
make -f Makefile.docker up

# Verify it's working
docker compose exec core-services /var/lib/hootenanny/bin/hoot --version
```

To stop services: `make -f Makefile.docker down`

### OSM Conversion

cbench handles GeoParquet to OSM conversion automatically when running the Hootenanny adapter. The converter:

- Creates OSM `<node>` elements for vertices
- Creates OSM `<way>` elements for each LineString
- Maps the `class` column to `highway=*` tags
- Preserves the `names` column as `name=*` tags
- Adds provenance tags (`matcher_ref_*` / `matcher_tgt_*`) for match extraction

When connectors are provided (via `--opt connectors=path/to/connectors.parquet`), segments sharing the same `connector_id` will reference the same OSM node, preserving network topology.

## Alternative Tools

- **[RoadMatcher](https://github.com/vividsolutions/roadmatcher)** - Java-based open source tool
- **[JOSM Conflation Plugin](https://josm.openstreetmap.de/)** - Semi-automated conflation in JOSM editor
- **[GraphHopper Map Matching](https://github.com/graphhopper/map-matching)** - For GPS trace to road network matching

## Troubleshooting

### Hootenanny Issues

**Hootenanny conflation hangs:**
- Large datasets (>100K ways) can take a long time in the optimization phase
- London (873K ways) takes 60+ minutes; Boston (33K ways) completes in ~5 minutes
- Check process is still alive: `docker compose exec core-services ps aux | grep hoot`

**"No node ID specified for RemoveNodeByEid":**
- Known LinearSnapMerger bug triggered by shared connector nodes
- The cbench adapter skips reference connectors by default to avoid this

### Conversion Issues

**Empty highway tags:**
- Check that `class` column exists in your data
- The converter maps standard road classes to OSM highway tags
