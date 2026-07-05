# Benchmark Results

First recorded head-to-head of `matcher` against baselines (naive geometric floor
and Hootenanny), produced with the `cbench` harness on two datasets: the
canonical **`us_boston_streets`** (roads) and **`us_fort_collins_sidewalks`**
(footways), evaluated against human-labeled ground truth.

## Headline

| Dataset | naive F1 | Hootenanny 0.2.41 F1 | matcher F1 |
|---------|----------|----------------------|------------|
| us_boston_streets (roads) | 0.637 | 0.973 | **0.996** |
| us_fort_collins_sidewalks | 0.311 | 0.927 | **0.976** |

matcher wins both (target-level F1). Hootenanny is a strong classical baseline on
both roads and footways; the naive floor collapses on dense parallel sidewalks.

## What is being measured (scope)

The `matcher` product is **match + stitch-resolve only** — it produces GERS
bridge tables (segment ↔ segment correspondences), not merged/conflated
geometry. So this comparison scores **MATCH-stage quality only**: does the tool
find the correct segment correspondences? A tool's *merge* quality (how it fuses
geometry/attributes) is out of scope and deliberately ignored — for Hootenanny
we run full conflation only because it is the sole way to observe hoot's match
decisions (its `conflate.match.only` mode discards results), then we extract the
correspondences from provenance tags and review relations and throw the merged
geometry away.

- **Eval:** target-level precision/recall/F1 (`--match-level target`) vs
  `labels/human/dataset=us_boston_streets` (639 pair labels, 268 match-labeled
  targets), plus non-blocking stitch-level (M:N group) metrics vs
  `labels/stitching` (73 group labels).
- **Machine:** Apple Silicon (arm64), Docker 28.5.1. Hootenanny runs as an x86
  image under emulation.
- **Reproducible:** exact commands in `BENCHMARKING.md`; versions/configs/wall
  times below.

## us_boston_streets (roads) — target-level match quality (primary)

Target = 10,844 local road segments; reference = 125,769 Overture segments.
639 pair labels (268 match-labeled targets).

| Tool | Precision | Recall | F1 | TP | FP | FN | Preds | Wall time | Notes |
|------|-----------|--------|----|----|----|----|-------|-----------|-------|
| naive (geometric floor) | 0.7754 | 0.5410 | 0.6374 | 145 | 42 | 123 | 11,520 | 4.4s | buffer=15m, min_overlap=0.30, angle_tol=35° |
| Hootenanny 0.2.41 | 0.9470 | 1.0000 | 0.9728 | 268 | 15 | 0 | 20,408 | n/a (emulated)* | HighwayMatchCreator/HighwaySnapMergerCreator |
| **matcher (xgboost)** | **0.9963** | **0.9963** | **0.9963** | 267 | 1 | 1 | 15,549 | 85.3s (2965 MB) | full ML stitch pipeline |

### us_boston_streets — stitch-level match quality (M:N groups)

| Tool | Precision | Recall | F1 | Exact-match | Groups | Basis |
|------|-----------|--------|----|-------------|--------|-------|
| naive | 0.393 | 0.299 | 0.340 | 0.055 | 73 | legacy id-map |
| Hootenanny 0.2.41 | 0.771 | 0.879 | 0.821 | 0.384 | 73 | legacy id-map |
| **matcher (xgboost)** | **0.871** | **0.793** | **0.830** | **0.537** | 67 | groups sidecar |

Note: only matcher emits a groups sidecar, so its stitch eval is group-based (67
groups); naive/Hootenanny have no sidecar and fall back to legacy segment-id
mapping (73 groups). Target-level (above) is the clean apples-to-apples metric.

## us_fort_collins_sidewalks (footways) — target-level match quality

Target = 38,714 local sidewalk segments; reference = 42,651 Overture segments.
239 pair labels (212 match-labeled targets). No stitching labels for this
dataset, so target-level only.

| Tool | Precision | Recall | F1 | TP | FP | FN | Preds | Wall time |
|------|-----------|--------|----|----|----|----|-------|-----------|
| naive (geometric floor) | 0.7885 | 0.1934 | 0.3106 | 41 | 11 | 171 | 9,525 | 4.6s |
| Hootenanny 0.2.41 | 0.9289 | 0.9245 | 0.9267 | 196 | 15 | 16 | 33,038 | n/a (emulated)* |
| **matcher (xgboost)** | **1.0000** | **0.9528** | **0.9758** | 202 | 0 | 10 | 19,981 | 68.0s (1825 MB) |

Sidewalks verdict: **Hootenanny CAN match footways** — `HighwayMatchCreator`
handles `highway=footway` about as well as roads (F1 0.927 vs 0.973 on roads), no
special footway match creator needed. The naive baseline, by contrast, **collapses
on sidewalks** (recall 0.19): dense parallel sidewalks on both sides of every
street defeat pure buffer overlap, since one target sidewalk buffer catches many
near-parallel Overture segments. matcher stays robust (F1 0.976, zero false
positives).

## Takeaways

- **matcher beats Hootenanny on match quality.** Target-level F1 0.9963 vs
  0.9728, at far higher precision (1 false positive vs 15). Stitch-level F1 is
  close (0.830 vs 0.821) but matcher's exact-group-match rate is much higher
  (0.537 vs 0.384) — it gets the M:N grouping right, not just the presence of a
  correspondence. (Speed is not compared: Hootenanny ran under x86 emulation
  here, so its wall time is not a valid datapoint — see caveats.)
- **Hootenanny is a strong, legitimate baseline**, not a strawman: perfect
  target-level recall (1.000) and 0.947 precision. It finds essentially every
  correspondence but over-predicts (20,408 pairs) and its M:N grouping is
  coarser. This is exactly the credible classical bar to clear.
- **Both crush the naive floor** (F1 0.637 target / 0.340 stitch on roads),
  confirming the benchmark floor is honest but non-trivial: pure buffer overlap
  already recovers ~54% of road matches at 78% precision.
- **Sidewalks separate the robust from the fragile.** On dense parallel footways
  the naive floor collapses (F1 0.311, recall 0.19) while both Hootenanny (0.927)
  and matcher (0.976) stay strong — matcher again on top with zero false
  positives. Hootenanny needs no special footway creator; `HighwayMatchCreator`
  covers `highway=footway`.

## Reproduce

```bash
# naive floor
uv run cbench run naive us_boston_streets -c cbench/datasets.toml

# Hootenanny via prebuilt image (no source build needed; runs under emulation on ARM)
docker pull --platform linux/amd64 hootenanny/run:0.2.41-1
uv run cbench run hootenanny us_boston_streets -c cbench/datasets.toml \
    --opt hoot_image=hootenanny/run:0.2.41-1

# matcher
uv run cbench run matcher us_boston_streets -c cbench/datasets.toml

# swap us_boston_streets -> us_fort_collins_sidewalks for the footway numbers
uv run cbench compare cbench_results.jsonl
```

## Caveats / honesty notes

- **Hootenanny is a version-pinned, one-shot FROZEN baseline — not a living
  harness.** hoot is out of active maintenance, so ongoing comparative
  investment goes to the modern match-stage options in the landscape section
  (Valhalla Meili / GraphHopper), which are the *live* comparisons going forward.
  The hoot row is recorded with its exact version + config and is not re-run
  routinely.
- **`* wall time is n/a here — Hootenanny ran under x86 emulation on Apple
  Silicon, which makes timing meaningless.** Only the quality columns
  (target-level P/R/F1) are valid from this machine. For real hoot timing, run
  once on native x86 Linux — the user's box still has the full compose-built
  stack (`hootenanny-core-services:latest`, built ~4 months ago). Note: those
  containers have previously died with `Exited (137)` (OOM), so give the run a
  generous memory limit (see BENCHMARKING.md).
- **Version is 0.2.41 (2018), not the current 0.2.87.** This is the newest
  *runnable* prebuilt image; the current release has no runnable image (the
  `hootenanny/rpmbuild-*` images are build-*environments* with no installed
  `hoot` binary) and must be built from source. 0.2.41's `HighwayMatchCreator`
  is the same core matcher; a modern build would likely shift these numbers only
  modestly. An updated native row is a future one-shot on the box above.
- Match quality is the only cross-tool comparison drawn here. naive runs
  in-process (Python; 0 child RSS by design), matcher spawns a child process —
  their wall times are valid but measure different things than hoot's.
- `labeled_coverage` is ~3% for all tools: metrics are measured against the
  labeled subset, so read them as "quality on the audited slice," not population
  estimates.
