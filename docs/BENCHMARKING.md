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

`cbench` resolves reference/target/labels from `cbench/datasets.toml` by default, so
the common case needs only a tool and a dataset name:

```bash
# Run matcher on a dataset (reference/target/labels come from datasets.toml)
cbench run matcher us_boston_streets -c cbench/datasets.toml

# Run every configured dataset
cbench run-batch matcher -c cbench/datasets.toml

# Explicit paths still override the config
cbench run matcher us_boston_streets \
    --labels ../labels/human \
    --reference ../data/raw/us_boston_streets_overture_segments_v1.0.parquet \
    --target ../data/raw/us_boston_streets_v1.0.parquet

# Run Hootenanny on the same dataset. NOTE: --opt values are explicit overrides
# and are CWD-relative, so `hoot_dir=../../hootenanny` assumes you are in
# `matcher/cbench`. From the repo root use `--opt hoot_dir=../hootenanny`, or
# pass an absolute path to be CWD-independent.
cbench run hootenanny us_boston_streets -c cbench/datasets.toml \
    --opt hoot_dir=../../hootenanny

# Compare results
cbench compare cbench_results.jsonl

# List available tool adapters
cbench list-tools
```

#### Working directory and path resolution

Paths in `datasets.toml` (both the `[defaults]` `data_dir` / `labels_dir` /
`stitch_labels_dir` and the per-dataset filenames) are written **relative to the
config file's directory**, and `cbench` resolves them that way — not relative to
your current working directory. This means `cbench run` / `run-batch` work from
any CWD (e.g. the repo root or the `cbench/` directory) and reliably find pair
labels under `labels/human` and stitching labels under `labels/stitching`.
Explicit `--labels`, `--reference`, `--target`, `--data-dir`, `--labels-dir`
overrides are treated as CWD-relative (normal shell semantics).

#### How the matcher adapter invokes matcher

The matcher adapter shells out with `uv run matcher stitch ...`, executed with
its working directory set to the **matcher repo root** (auto-detected as the
directory containing `src/matcher`). Running via `uv run` from the repo root
means:

- matcher does not need to be on your `PATH`, and
- matcher's relative model path (`data/models/matcher_model_combined.joblib`)
  resolves correctly regardless of where you launched `cbench`.

Override the invocation with `--opt matcher_cmd="matcher"` (to use a binary
already on `PATH`) and/or `--opt repo_root=/path/to/matcher` (to point at a
different checkout).

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

## Stitch-level quality gate

Pair-level P/R/F1 measures the classifier. The **stitch-level** metric measures
the optimizer's final M:N group edge selection against curated stitching labels
(`labels/stitching/`), and it can be enforced as a **gate** with `--gate`:

```bash
# Enforce the gate on one dataset (nonzero exit if it regresses below its floor)
cbench run matcher us_boston_streets -c cbench/datasets.toml --gate

# Enforce across all configured datasets
cbench run-batch matcher -c cbench/datasets.toml --gate
```

With `--gate`, cbench compares each dataset's **sliver-filtered** edge-F1 and
exact-match against the per-dataset floors in `cbench/datasets.toml`
(`[gate.<dataset>]`) and **exits nonzero** if any *armed* dataset falls below.
Without `--gate` the stitch metrics are still computed and printed (non-blocking).

**Why benchmark-time, not CI:** the gate needs live pipeline outputs (a bridge
parquet + its `*_groups.json` sidecar), which don't exist in GitHub Actions
(`data/raw` / `data/output` are untracked). So the gate is part of the
**pre-merge checklist for matching-logic PRs**, run locally against fresh Boston
output. The gate *machinery* itself (mapping, sliver filtering, arming, floor
logic) is unit-tested in CI on a committed miniature fixture
(`cbench/tests/test_gate.py`).

**Auto-arming.** Each `[gate.<dataset>]` block sets `min_mapped_groups`: the
floor is enforced only once at least that many curated labels map to current
pipeline groups (mapping is by edge-overlap, robust to group_id churn). Below
that the dataset reports `skip_unarmed` (non-blocking). This means the gate goes
live automatically as the label base grows — no code change or second PR. As of
2026-07-05, `us_boston_streets` is armed (67 mapped groups, floors F1 0.78 /
exact 0.45, baseline F1 0.8345 / exact 0.5373); `us_seattle_sidewalks` (2 mapped)
is unarmed.

**Adding / updating a floor.** Re-measure the baseline against fresh output
(`cbench run matcher <dataset>` prints the stitch block), then set
`f1_filtered_floor` / `exact_filtered_floor` to baseline − margin (LOO-gate
style: ~0.05 on F1, wider on the noisier exact-match) and `min_mapped_groups` to
~30. Update the block in `cbench/datasets.toml`.

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
