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

# Run the naive geometric baseline (the benchmark floor - pure buffer overlap,
# no learning, no names, no topology). Fast and dependency-light.
cbench run naive us_boston_streets -c cbench/datasets.toml \
    --opt buffer_m=15 --opt min_overlap=0.30 --opt angle_tol_deg=35

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

[Hootenanny](https://github.com/ngageoint/hootenanny) is a vector conflation tool
from NGA. For our purposes we care only about the **matches it finds** (segment ↔
segment correspondences), not its merged geometry — see the scope note in
`BENCHMARK_RESULTS.md`. We run full `hoot conflate` only because hoot's match
decisions are observable only through the conflated output (its
`conflate.match.only` mode discards them); the cbench adapter then extracts
correspondences from the `matcher_ref_*` / `matcher_tgt_*` provenance tags and
review relations and ignores the merged geometry.

### Prebuilt Docker image (Recommended, validated on Apple Silicon)

The fastest path — no source build, no compose stack, no Postgres/Tomcat. Uses a
prebuilt image from Docker Hub that already contains the `hoot` binary. Runs
under x86 emulation on ARM.

```bash
docker pull --platform linux/amd64 hootenanny/run:0.2.41-1
docker run --rm --platform linux/amd64 --entrypoint /usr/bin/hoot \
    hootenanny/run:0.2.41-1 version    # -> Hootenanny 0.2.41

# Run the benchmark (adapter drives `docker run` for you):
uv run cbench run hootenanny us_boston_streets -c cbench/datasets.toml \
    --opt hoot_image=hootenanny/run:0.2.41-1
```

Notes and gotchas (all handled by the adapter, documented here for transparency):

- **Version:** `hootenanny/run:0.2.41-1` is v0.2.41 (2018) — the newest *runnable*
  prebuilt image. The `hootenanny/rpmbuild-hoot-release:latest` (2024) image is a
  build *environment* with **no installed `hoot` binary and no RPMs staged**, so
  it cannot run conflation without a full build. The current 0.2.87 release has
  no runnable prebuilt image.
- **Creator class names differ by version.** 0.2.41 requires fully namespaced
  classes and the highway *merger* is `HighwaySnapMergerCreator` (not
  `HighwayMergerCreator`). The adapter defaults to
  `match.creators=hoot::HighwayMatchCreator` and
  `merger.creators=hoot::HighwaySnapMergerCreator`; override for other versions
  with `--opt match_creators=... --opt merger_creators=...`.
- **Binary path** is `/usr/bin/hoot` in the run image (vs `$HOOT_HOME/bin/hoot`
  in a source build); override with `--opt hoot_bin=...`.
- The repeated `Internal Error: Two nodes were found with the same coordinate`
  log lines during merging are the known non-fatal HighwaySnapMerger warnings —
  conflation still completes.

### Docker Compose (source build — for the current hoot release)

Hootenanny here is a **version-pinned, one-shot frozen baseline** (hoot is out of
active maintenance; the live comparisons going forward are the modern match-stage
options in the landscape section). The numbers in `BENCHMARK_RESULTS.md` come
from the prebuilt-image route above under **x86 emulation**, so their **wall
times are invalid** — only match quality (P/R/F1) is reported from Apple Silicon.

To get valid timing and/or a **current** hoot (0.2.87+) row, do a one-shot run on
**native x86 Linux** — there is no runnable prebuilt image for current hoot, so it
must be built from source. This is a multi-hour, high-risk build under emulation
(EL/CentOS base, GDAL/GEOS/PROJ/v8/node from source); build it on the native box
instead.

> Memory: the compose `core-services`/`postgres` containers are memory-hungry and
> have been observed dying with `Exited (137)` (OOM). Give the run a generous
> memory limit (e.g. Docker Desktop / daemon limit of 16 GB+; conflation of a
> city network peaks high). A box with plenty of RAM (e.g. 64 GB) is comfortable.

```bash
# Clone Hootenanny as a sibling to matcher
cd /path/to/matcher/..
git clone https://github.com/ngageoint/hootenanny.git
cd hootenanny

# Start services (first run builds everything - takes 20-40 min on x86; longer under emulation)
make -f Makefile.docker up

# Verify it's working
docker compose exec core-services /var/lib/hootenanny/bin/hoot --version

# Then point the adapter at the checkout instead of an image:
#   uv run cbench run hootenanny us_boston_streets -c cbench/datasets.toml \
#       --opt hoot_dir=../hootenanny
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

## Valhalla Meili (map-matching)

[Valhalla](https://github.com/valhalla/valhalla) **Meili** is the modern,
actively-maintained match-stage baseline (adapter name `meili`). It treats each
local segment as a synthetic GPS trace and snaps it onto an Overture-derived
routable graph; the matched edge sequence *is* the segment↔GERS correspondence
set. This handles segmentation mismatch natively (a long local segment snaps
across many short Overture segments). Results + analysis:
`docs/BENCHMARK_RESULTS.md` and `research/meili_baseline.md`.

### Pipeline (all handled by the adapter)

1. **Overture → OSM PBF** (`cbench/convert/pbf.py`): each Overture segment becomes
   a way carrying its **GERS id as the OSM `way_id`** (via a JSON sidecar), with
   vertices collapsed by rounded coordinate (≈1 cm) so shared connectors become
   shared nodes → a routable graph. Every way gets a routable `highway=*` tag.
2. **Build tiles + match** with Valhalla's own engine, in-process via the
   `pyvalhalla` wheel: `valhalla_build_tiles` then `valhalla.Actor.trace_attributes`
   with `shape_match=map_snap`. Matched `edges[].way_id` map straight back to GERS
   ids; per target, edges are aggregated with an overlap-length filter to drop
   spuriously-touched edges.

### Install & run

`pyvalhalla` ships **native-ARM** Valhalla binaries (no Docker, no emulation → valid
timing) but requires **Python ≥ 3.12**, so run cbench from a 3.12 env:

```bash
uv pip install -e "cbench[meili]" --python 3.12   # geopandas + pyosmium + pyvalhalla
uv run --python 3.12 cbench run meili us_boston_streets -c cbench/datasets.toml
# sidewalks (Fort Collins / Seattle): pedestrian costing (the default) covers footways
uv run --python 3.12 cbench run meili us_fort_collins_sidewalks -c cbench/datasets.toml
```

Key `--opt`s (defaults in parentheses): `costing` (`pedestrian` — bidirectional,
traverses all local road classes, so it handles roads *and* sidewalks and sidesteps
one-way as the documented Meili failure mode; use `auto` for directional roads-only),
`densify_m` (10), `search_radius` (25), `min_match_frac`/`min_match_m` (0.10 / 8 m,
the overlap threshold that stands in for a no-match abstention), `workers` (8, one
`valhalla.Actor` per thread), `graph_cache_dir` (built tiles are cached per reference
file — keep this OUTSIDE `data/output`), `rebuild` (force a graph rebuild).

### Docker route (design target; blocked on Apple Silicon here)

The intended runtime was the maintained multi-arch Valhalla Docker image. On this
machine both Docker routes were dead ends, hence the in-process `pyvalhalla`
implementation:

- `ghcr.io/gis-ops/docker-valhalla/valhalla` (ARM-native, multi-arch) — the ghcr.io
  blob CDN **stalls at 0 bytes/s** here; the image never finishes pulling.
- `valhalla/valhalla:run-3.3.0` (Docker Hub, amd64-only) — pulls fine but
  **segfaults under qemu emulation** in `valhalla_build_tiles`, even on a 2-way
  graph. This is the amd64-on-arm64 emulation incompatibility, the same class of
  problem that makes Hootenanny's emulated wall time invalid.

`pyvalhalla` is the identical Valhalla engine (v3.7.0) run in-process, so the match
quality is representative and the ARM-native timing is valid. On a machine with a
working multi-arch pull (or native x86 Linux), the Docker recipe is:
`docker run -d -p 8002:8002 -v $PWD/custom_files:/custom_files -e use_tiles_ignore_pbf=False ghcr.io/gis-ops/docker-valhalla/valhalla:latest`
(place the PBF from `convert/pbf.py` in `custom_files/`), then POST each densified
trace to `/trace_attributes`.

## Baseline landscape

Verification of the open-source conflation / map-matching landscape as a source
of benchmark baselines (verified July 2026). "Integration cost" is the effort to
build a headless cbench adapter that takes two road-linework sets (reference =
Overture segments, target = local roads) and emits matches.

**Viability is weighted by MATCH-stage output**: can the tool produce
segment ↔ segment *correspondences* headlessly? Merge/conflation completeness is
irrelevant — `matcher` produces GERS bridge pairs, not merged geometry, so a
baseline only needs to emit "local segment X ↔ Overture segment Y" pairs.

| Tool | Status (Jul 2026) | Headless linework viability | Integration cost |
|------|-------------------|-----------------------------|------------------|
| **Naive geometric** (this repo) | shipped in `cbench` | Native — the benchmark floor | done |
| **Hootenanny** (`ngageoint/hootenanny`) | Active, not archived; latest **v0.2.87 (2024-10)**, ~9-month release gap since; GPL-3.0 | Strong — `hoot conflate` is purpose-built vector-to-vector road conflation | **Medium–High.** No official multi-arch image, but usable **prebuilt amd64 images exist on Docker Hub** (`hootenanny/rpmbuild-hoot-release:latest`, 2024-08; `hootenanny/run:0.2.41-1`, 2018). Runs under emulation on Apple Silicon. The from-source `docker-compose` build (EL/CentOS, GDAL/GEOS/PROJ/v8/node) is the high-cost path; the prebuilt image is the low-cost path. |
| **SharedStreets `shst match`** (`sharedstreets/sharedstreets-js`) | **Effectively abandoned** (last real release v0.15.2, May 2020) | Conceptual fit, but matches to the global SharedStreets reference tiles, not to an arbitrary supplied reference set — impedance mismatch with our ref/target contract | **High.** Needs ancient Node (10–14) in a pinned container; `node-gyp` native deps fail on Node 20/22. Depends on an unmaintained tile backend. **Recommend against.** |
| **Valhalla Meili** (map-matching) | **Actively maintained**, clean multi-arch Docker | Yes — feed each segment as a synthetic GPS trace to `trace_attributes` with `map_snap`. A 2025 arXiv paper conflated 1.78M VDOT segments this way (>98% coverage, 2.5 m median error) | **Medium.** Main work: convert Overture → OSM-PBF → build the routable graph; densify segments into traces; handle one-way/direction; map matched edges back to Overture IDs. Needs a running Valhalla service (single container fine at benchmark scale). |
| **GraphHopper map-matching** | **Actively maintained** (2025 releases); folded into main `graphhopper/graphhopper`, on Maven Central | Same segment-as-trace paradigm as Valhalla; embeddable JVM library (no server needed) | **Medium.** Same Overture→OSM-PBF ingestion tax as Valhalla; shares ~80% of that work. Lower runtime friction (in-process JVM), slightly less conflation-specific precedent. |
| **OpenLR** (`tomtom-international/openlr`) | Maintained (last commit 2025-10; 2025 aarch64 + Java 17 work); released via Maven, no GitHub Releases | **Poor fit** — a location-referencing *codec* (encode on map A, decode on map B), not a conflation engine; decode quality hinges on consistent FRC/FOW attributes local data usually lacks | **High relative to payoff.** Runtime fine; you'd hand-build an encode→decode harness on a codec never meant for bulk conflation. **Low priority.** |

### Overture / GERS ecosystem (2024–2026)

There is **no official Overture drop-in "match two road-linework sets → GERS
links" open-source tool** — Overture's own road conflation is an internal
pipeline whose *outputs* are published, not the matcher. This is arguably the
gap `matcher` fills. Related artifacts:

- **GERS bridge files** (docs.overturemaps.org/gers/bridge-files) — release
  *artifacts* mapping GERS IDs ↔ source IDs for a fixed set of upstream datasets
  (OSM, Esri, Meta/Microsoft, etc.), not a tool. Useful as eval scaffolding for
  OSM-sourced roads, not for arbitrary local data. (GERS IDs are UUIDv4 since the
  June 2025 release.)
- **`OvertureMaps/match-inspector`** — conflates *building footprints*, not
  roads; interesting only as a labeling-UI reference.
- **`OvertureMaps/osm-pbf-parquet`** — a PBF→Parquet transcoder (not a matcher),
  but a useful fast ingestion step if we build an OSM-graph baseline.
- Commercial (not open): TomTom **GEM**, CARTO/Databricks & Wherobots/Sedona
  GERS-aware spatial joins.

### Recommended next baseline (actionable)

After the naive floor and Hootenanny, the best MATCH-stage baseline to build next
is a **map-matcher fed local segments as synthetic GPS traces**, snapping them
onto an Overture-derived routable graph. The matched edge sequence *is* the
segment↔segment correspondence set — exactly the bridge-pair output we score. Two
concrete, maintained options (build one, not both):

**Option A — Valhalla Meili (recommended first).** Strongest precedent: a 2025
arXiv recipe conflated 1.78M road segments this way (>98% coverage, 2.5 m median).

- Runtime: official multi-arch Docker `ghcr.io/valhalla/valhalla:latest` (ARM-native, no emulation).
- Pipeline: (1) convert Overture segments → OSM PBF (reuse `cbench/convert/osm.py`
  geometry logic, emit PBF via `osmium`); (2) `valhalla_build_tiles`; (3) POST each
  target segment to the local `/trace_attributes` endpoint with
  `{"shape":[...], "costing":"auto", "shape_match":"map_snap"}`; (4) read
  `edges[].way_id` back to Overture IDs (carry the GERS id as the OSM way id during
  PBF export so no join is needed).
- Trace-format notes: densify each target LineString to ~10 m spacing before
  sending (Meili expects trace-like point density); set `"search_radius"` ~25 m;
  one-way/directionality mismatches are the documented failure mode — send traces
  in geometric order and allow `"trace_options.turn_penalty_factor":0`.
- Integration cost: **Medium** — the only real work is the Overture→PBF export;
  everything else is HTTP + JSON parsing.

**Option B — GraphHopper map-matching (close second, no server).** Same paradigm
as an embeddable JVM library (`com.graphhopper:graphhopper-map-matching`, 2025
releases on Maven Central) — no service to run, but a JVM subprocess and the same
Overture→PBF import. Pick this if avoiding a running service matters more than
Valhalla's stronger conflation precedent; the Overture→PBF step is shared with
Option A.

**Explicitly deprioritized for MATCH-stage:**

- **SharedStreets `shst match`** — even judged purely on emitting match pairs, it
  emits *SharedStreets* reference IDs, not direct Overture↔local pairs, forcing a
  SharedStreets-tile intermediary; combined with Node-10 `node-gyp` bit-rot
  (won't install on Node 20/22 without pinning Node ≤14 in a container) and an
  unmaintained tile backend, it is not worth the integration cost.
- **OpenLR** — a location-referencing codec, not a matcher; you'd hand-build
  encode→decode and it needs consistent FRC/FOW attributes local data lacks.

Both recommended options impose an **Overture→OSM-PBF graph tax** (the reference
must become a routable graph). Build that converter once — it is the shared
dependency for A and B.

### Other tools (not prioritized)

- **[RoadMatcher](https://github.com/vividsolutions/roadmatcher)** - Java-based open source tool (dormant)
- **[JOSM Conflation Plugin](https://josm.openstreetmap.de/)** - Semi-automated conflation in JOSM editor (interactive, not headless)

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
