# Benchmark Results

> _Harness renamed `cbench` → `mbench` (2026-07-05). Mentions of "cbench" below are historical; the harness is now invoked as `mbench` (a deprecated `cbench` alias still forwards)._

First recorded head-to-head of `matcher` against baselines — the naive geometric
floor, Hootenanny (classical vector conflation), and **Valhalla Meili** (modern
map-matching) — produced with the `cbench` harness and evaluated against
human-labeled ground truth. Primary datasets: the canonical
**`us_boston_streets`** (roads) and **`us_fort_collins_sidewalks`** (footways),
with **`us_seattle_sidewalks`** added for the Meili row.

## Headline

| Dataset | naive F1 | Hootenanny 0.2.41 F1 | Meili (Valhalla) F1 | matcher F1 |
|---------|----------|----------------------|---------------------|------------|
| us_boston_streets (roads) | 0.839 | 0.973 | 0.994 | **0.996** |
| us_fort_collins_sidewalks | 0.365 | 0.927 | 0.962 | **0.976** |
| us_seattle_sidewalks | — | — | 0.944 | — |

> **Naive rows re-measured 2026-07-05** after fixing a shape-sanity guard bug in
> the naive adapter (PR #275 shipped a symmetric-Hausdorff guard that rejected
> legitimate short-reference/long-target pairs — the coverage-asymmetry trap;
> guard removed). Old values (Boston F1 ~~0.637~~, FC F1 ~~0.311~~) are kept
> struck-through in the tables below for provenance. The fix roughly doubled
> naive road recall (0.54 → 0.95); sidewalks still collapse (see below).

matcher wins on both shared datasets (target-level F1). Meili is the strongest
external baseline — it beats Hootenanny on both, with **perfect recall (1.000) on
all three datasets** — but over-predicts (lower precision), which is the
signature of map-matching: it snaps every trace onto *something*, so it never
misses a true match but picks up spurious adjacent/parallel edges. matcher edges
it out on F1 by trading a little recall for much higher precision. The naive floor
collapses on dense parallel sidewalks; Hootenanny is a strong classical baseline
on both roads and footways.

The Meili row also **pilots the path-based-formulation bet** from
`docs/EVAL_ROADMAP.md` (§Architecture assessment #3) — see
[`research/meili_baseline.md`](../research/meili_baseline.md) for what the result
implies (segmentation mismatch handled natively; precision lost to parallel
geometry), with concrete example segment ids.

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
| naive (geometric floor) | 0.7500 | 0.9515 | 0.8388 | 255 | 85 | 13 | 28,664 | 4.0s | buffer=15m, min_overlap=0.30, angle_tol=35°; guard fix 2026-07-05 |
| ~~naive (pre-fix, PR #275 bug)~~ | ~~0.7754~~ | ~~0.5410~~ | ~~0.6374~~ | ~~145~~ | ~~42~~ | ~~123~~ | ~~11,520~~ | ~~4.4s~~ | superseded — symmetric-Hausdorff guard bug |
| Hootenanny 0.2.41 | 0.9470 | 1.0000 | 0.9728 | 268 | 15 | 0 | 20,408 | n/a (emulated)* | HighwayMatchCreator/HighwaySnapMergerCreator |
| Hootenanny 0.2.87 (native x86, tag-only merge) | 0.9884 | 0.6343 | 0.7727 | 170 | 2 | 98 | 10,121 | **5m47s** (3m26s match-only; 1.79 GB)† | HighwayMatchCreator + LinearTagOnlyMerger; snap-merge crashes‡ |
| Meili (Valhalla 3.7.0) | 0.9889 | 1.0000 | 0.9944 | 268 | 3 | 0 | 15,591 | 12.0s (5.0s match-only; 838 MB) | pedestrian costing, densify 10m, search_radius 25m, min_overlap 0.10/8m |
| **matcher (xgboost)** | **0.9963** | **0.9963** | **0.9963** | 267 | 1 | 1 | 15,549 | 85.3s (2965 MB) | full ML stitch pipeline |

† **Native x86 timing is now valid — this closes the "emulated" caveat.** Run
one-shot on the user's box (Intel Core Ultra 7 265K, 20-core, 62 GiB, Ubuntu
24.04, Docker 29.5.2) using **Hootenanny 0.2.87** (`0.2.87_3_g3eb7beb`, current
release family). hoot conflation is **single-threaded** (99% of one core), so the
wall time is honest and the 12-CPU container cap (`docker run --cpus 12`, imposed
because the box also serves media) was non-binding. See
[`research/hoot_native_baseline.md`](../research/hoot_native_baseline.md) for host,
exact command, and logs.

‡ **The faithful snap-merge quality could not be reproduced on 0.2.87**, so this
row's *quality* is not comparable to the 0.2.41 row — the 0.2.41 emulated numbers
(F1 0.973) remain the Boston quality reference. 0.2.87's default/Unifying
`LinearSnapMerger` aborts mid-merge (`No node ID specified for RemoveNodeByEid`,
via `RecursiveElementRemover`) on this synthetic OSM's coincident-coordinate
nodes; the Network algorithm and `AttributeConflation.conf` hit the same abort.
The only completing merge is **`LinearTagOnlyMerger`** (same `HighwayMatchCreator`
matcher — which found 13,121 match sets — but transfers tags instead of snapping
geometry). Because tag-only never splits reference ways, it cannot surface
sub-segment matches, so Boston's short-local-vs-long-Overture segmentation
mismatch depresses recall to 0.634. This is a **merge-representation artifact, not
a matcher regression** (contrast Fort Collins below, where fine 1:1 segmentation
makes tag-only near-faithful at F1 0.940). Per policy we did not rebuild hoot from
source to chase the merger bug.

### us_boston_streets — stitch-level match quality (M:N groups)

| Tool | Precision | Recall | F1 | Exact-match | Groups | Basis |
|------|-----------|--------|----|-------------|--------|-------|
| naive (guard fix, 2026-07-05) | 0.499 | 0.816 | 0.619 | 0.168 | 113 | legacy id-map |
| ~~naive (pre-fix)~~ | ~~0.393~~ | ~~0.299~~ | ~~0.340~~ | ~~0.055~~ | ~~73~~ | legacy id-map |
| Hootenanny 0.2.41 | 0.771 | 0.879 | 0.821 | 0.384 | 73 | legacy id-map |
| Meili (Valhalla 3.7.0) | 0.873 | 0.931 | 0.901 | 0.521 | 73 | legacy id-map |
| **matcher (xgboost)** | **0.871** | **0.793** | **0.830** | **0.537** | 67 | groups sidecar |

Note: only matcher emits a groups sidecar, so its stitch eval is group-based (67
groups); naive/Hootenanny/Meili have no sidecar and fall back to legacy segment-id
mapping (73 groups). **Caveat on the naive stitch row:** it was re-measured after
the guard fix against the *current* stitching-label base, which has since grown
(73 → 113 mapped legacy groups), so its group basis differs from the frozen
Hootenanny/Meili rows — cross-tool stitch numbers are only strictly comparable at
the same label epoch. Target-level (above) is the clean apples-to-apples metric.
Meili's high stitch-edge F1 (0.901) reflects its perfect recall + one-to-many
edge coverage; its exact-group-match (0.521) trails matcher (0.537), i.e. it
recovers the edges but groups them slightly less precisely — consistent with its
target-level over-prediction.

## us_fort_collins_sidewalks (footways) — target-level match quality

Target = 38,714 local sidewalk segments; reference = 42,651 Overture segments.
239 pair labels (212 match-labeled targets). No stitching labels for this
dataset, so target-level only.

| Tool | Precision | Recall | F1 | TP | FP | FN | Preds | Wall time |
|------|-----------|--------|----|----|----|----|-------|-----------|
| naive (geometric floor) | 0.8065 | 0.2358 | 0.3650 | 50 | 12 | 162 | 14,036 | 4.2s |
| ~~naive (pre-fix, PR #275 bug)~~ | ~~0.7885~~ | ~~0.1934~~ | ~~0.3106~~ | ~~41~~ | ~~11~~ | ~~171~~ | ~~9,525~~ | ~~4.6s~~ |
| Hootenanny 0.2.41 | 0.9289 | 0.9245 | 0.9267 | 196 | 15 | 16 | 33,038 | n/a (emulated)* |
| Hootenanny 0.2.87 (native x86, tag-only merge) | 1.0000 | 0.8868 | 0.9400 | 188 | 0 | 24 | 21,876 | **4m50s** (1.48 GB)† |
| Meili (Valhalla 3.7.0) | 0.9258 | 1.0000 | 0.9615 | 212 | 17 | 0 | 41,221 | 15.2s (build+match; 852 MB) |
| **matcher (xgboost)** | **1.0000** | **0.9528** | **0.9758** | 202 | 0 | 10 | 19,981 | 68.0s (1825 MB) |

† Same native x86 / hoot 0.2.87 / tag-only-merge run as the Boston row (see that
footnote and [`research/hoot_native_baseline.md`](../research/hoot_native_baseline.md)).
On sidewalks tag-only merge is **near-faithful** — fine ~1:1 segmentation means few
sub-segment matches are lost — so the native 0.2.87 quality (F1 0.940, perfect
precision) is on par with the 0.2.41 emulated row (F1 0.927) and directly usable.

Sidewalks verdict: **Hootenanny CAN match footways** — `HighwayMatchCreator`
handles `highway=footway` about as well as roads (F1 0.927 vs 0.973 on roads), no
special footway match creator needed. The naive baseline, by contrast, **collapses
on sidewalks** (recall 0.24, F1 0.365 — even after the guard fix that lifted its
road recall to 0.95): dense parallel sidewalks on both sides of every street
defeat pure buffer overlap, since one target sidewalk buffer catches many
near-parallel Overture segments and end-to-end bearing can't separate them.
matcher stays robust (F1 0.976, zero false
positives). **Meili** lands between Hootenanny and matcher (F1 0.962): perfect
recall, but 17 false positives — sidewalks snapped onto the adjacent parallel
*road* centerline (the map-matching precision tax; see path-based section).

## us_seattle_sidewalks (footways) — target-level match quality (Meili baseline)

Target = 46,145 local sidewalk segments; reference = 165,503 Overture segments.
200 pair labels (85 match-labeled targets). Added for the Meili row; naive /
Hootenanny / matcher rows are not (yet) recorded for this dataset.

| Tool | Precision | Recall | F1 | TP | FP | FN | Preds | Wall time |
|------|-----------|--------|----|----|----|----|-------|-----------|
| Meili (Valhalla 3.7.0) | 0.8947 | 1.0000 | 0.9444 | 85 | 10 | 0 | 53,217 | 29.1s (build+match; 1712 MB) |

Same signature as Fort Collins: perfect recall, precision lost to parallel-road
snapping. Non-blocking stitch eval (7 legacy groups): P 0.982 / R 0.964 /
F1 0.973 / exact 0.714.

## Operational complexity

Quality is only half the story: the tools differ enormously in *what it takes to
run them at all*. The rubric below scores the cold-start path for a new user
holding the two input parquets (1 = painful, 5 = effortless) across steps to first
result, dependency weight (pip vs Docker vs x86 emulation), config burden, time to
first result, and maintainability. Full cold-start narratives, the rejected-engine
survey ("is there another engine besides Valhalla?" → GraphHopper), and a ranked
plan to close matcher's DX gap are in
[`research/engine_dx_comparison.md`](../research/engine_dx_comparison.md).

| Engine | Steps | Dep weight | Config | Time | Maint | **Σ/25** |
|--------|:-----:|:----------:|:------:|:----:|:-----:|:--------:|
| naive floor | 5 | 5 | 4 | 5 | 5 | **24** |
| Valhalla Meili | 5 | 4 | 4 | 4 | 4 | **21** |
| **matcher (post top-3 DX fixes)** | 5 | 3 | 4 | 3 | 4 | **19** |
| ~~matcher (pre-fix)~~ | ~~2~~ | ~~3~~ | ~~4~~ | ~~2~~ | ~~3~~ | ~~**14**~~ |
| Hootenanny 0.2.41 (emulated) | 3 | 2 | 3 | 1 | 1 | **10** |
| Hootenanny 0.2.87 (native x86) | 1 | 1 | 2 | 2 | 2 | **8** |

The pre-fix headline was a 7-point **Meili (21) vs matcher (14)** gap at
near-identical quality (F1 0.994 vs 0.996): matcher's fresh-clone path was clone →
heavy install → mandatory **`matcher train`** → mandatory **`matcher data fetch`** →
stitch. The top-3 DX fixes from
[`research/engine_dx_comparison.md`](../research/engine_dx_comparison.md) landed
(2026-07-05) and close most of it:

1. **Pretrained model ships in the package** (466 KB, calibrated) — `stitch` works
   with zero training; a CI lockstep test (`tests/unit/test_shipped_model.py`)
   fails any PR that bumps `FEATURE_VERSION` without reshipping, and model load
   now hard-errors (was: warn) on a version mismatch outside the trusted bundled
   path.
2. **PyPI packaging** — installable `road-matcher` wheel (930 KB incl. model),
   console script `matcher`; heavy tuning/imagery deps moved out of the core
   install.
3. **`matcher fetch-overture --bbox|--clip-target`** — YAML-free Overture
   reference fetch with `--release` pinning and a `.meta.yaml` sidecar.

Measured fresh cold start (throwaway venv, wheel install, 389-segment Boston
slice): install **0.3 s** (warm uv cache) → `fetch-overture --clip-target`
**33 s** (networked) → `stitch` **46 s** → bridge parquet. **Zero training, zero
YAML, zero clone.** The re-scored rubric row: steps now Meili-equal (5); time 3
(stitch itself is ~6× slower than Meili's match); dep weight unchanged (3 — the
pip resolve is still numba/xgboost/geopandas-heavy); maint 4 (the retrain tax on
`FEATURE_VERSION` bumps remains but is CI-enforced rather than silent). The
remaining 2-point gap to Meili (dep weight + time) is inherent to the ML stack,
not setup friction. The naive floor tops the rubric only because it does the
least — it *is* the quality floor (F1 0.839 roads / 0.365 sidewalks).

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
- **All three beat the naive floor, but the floor is honest and non-trivial on
  roads** (naive F1 0.839 target / 0.619 stitch): pure buffer overlap + bearing
  already recovers ~95% of road matches at 75% precision. The gap to matcher on
  roads is now a *precision* gap (0.75 → 0.996), not a recall one — geometry
  alone finds the road correspondences; it just can't reject the near-misses.
- **Sidewalks separate the robust from the fragile.** On dense parallel footways
  the naive floor collapses (F1 0.365, recall 0.24) while both Hootenanny (0.927)
  and matcher (0.976) stay strong — matcher again on top with zero false
  positives. This is where the naive floor stays genuinely weak even after the
  guard fix, and it is the clearest one-number justification for a learned
  approach. Hootenanny needs no special footway creator; `HighwayMatchCreator`
  covers `highway=footway`.
- **Meili is the strongest external baseline, and validates the map-matching
  approach as a real contender.** It beats Hootenanny on every dataset and comes
  within ~0.002–0.014 F1 of matcher, with **perfect recall (1.000) everywhere** —
  path-snapping never misses a true correspondence and handles segmentation
  mismatch natively (a local segment splits cleanly across many Overture GERS
  ids). Its one weakness is **precision**: it snaps sidewalks onto adjacent
  parallel roads and, on roads, occasionally onto a parallel carriageway. This is
  a genuinely different error profile from the pairwise scorer (which is
  precision-first). And because Meili is ARM-native, its wall time **is** a valid
  datapoint: 5–29 s end-to-end, ~6–17× faster than matcher's ML pipeline. See
  [`research/meili_baseline.md`](../research/meili_baseline.md) for the full
  path-based-formulation analysis with example segment ids.

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

# Meili (Valhalla map-matching). Needs the `meili` extra (geopandas + pyosmium +
# pyvalhalla); pyvalhalla ships native-ARM Valhalla binaries and requires Python
# >= 3.12, so run cbench from a 3.12 env, e.g.:
#   uv pip install -e "cbench[meili]" --python 3.12
uv run --python 3.12 cbench run meili us_boston_streets -c cbench/datasets.toml
# footways: costing defaults to `pedestrian` (bidirectional, all classes) — good
# for both roads and sidewalks; swap the dataset name for FC / Seattle.

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
- **`* wall time is n/a in the 0.2.41 rows — that image ran under x86 emulation
  on Apple Silicon, which makes its timing meaningless.** Only the 0.2.41 quality
  columns (target-level P/R/F1) are valid from that machine. **This caveat is now
  closed:** the `†` rows above report **valid native x86 wall times** from a
  one-shot run of current-family **hoot 0.2.87** on the user's 20-core box
  (single-threaded; 5m47s Boston full / 3m26s Boston match-only / 4m50s Fort
  Collins). The from-source snap-merge quality could not be reproduced on 0.2.87
  (`LinearSnapMerger` abort); see the `‡` footnote and
  [`research/hoot_native_baseline.md`](../research/hoot_native_baseline.md).
  (Running the standalone `hoot` CLI container needs neither postgres/tomcat nor
  much memory — the old `Exited (137)` OOMs were the full service stack, not the
  CLI; the conflations peaked at ~1.8 GB.)
- **The headline quality rows are 0.2.41 (2018); the current release is 0.2.87
  (2024).** 0.2.41 is the newest *runnable prebuilt image* (the current release
  has no runnable image — the `hootenanny/rpmbuild-*` images are build
  *environments* with no installed `hoot`). The `†` rows now add a **native
  0.2.87** datapoint from the user's box (source-built
  `hootenanny-core-services:latest`). Findings: 0.2.87's `HighwayMatchCreator`
  matcher is unchanged in spirit (Fort Collins native F1 0.940 ≈ 0.2.41's 0.927),
  but its **merge phase regressed** on synthetic non-topological OSM — the
  `LinearSnapMerger` aborts (`RemoveNodeByEid`), so the faithful snap-merge Boston
  quality stays pinned to the 0.2.41 emulated numbers. Details:
  [`research/hoot_native_baseline.md`](../research/hoot_native_baseline.md).
- Match quality is the only cross-tool comparison drawn here. naive runs
  in-process (Python; 0 child RSS by design), matcher spawns a child process —
  their wall times are valid but measure different things than hoot's.
- `labeled_coverage` is ~3% for all tools: metrics are measured against the
  labeled subset, so read them as "quality on the audited slice," not population
  estimates.
- **Meili timing IS valid (native ARM).** Unlike Hootenanny, Valhalla runs
  ARM-native here (via the `pyvalhalla` wheel, which bundles native-ARM Valhalla
  3.7.0 binaries), so the wall times are real. The Boston number is reported both
  cold (12.0 s, tile build + match) and match-only (5.0 s, cached tiles); FC and
  Seattle are cold. The intended runtime was the maintained multi-arch Valhalla
  *Docker* image, but on this machine both Docker routes were dead ends — the ARM
  image on ghcr.io stalls on blob download (registry CDN, 0 bytes/s), and the
  amd64 image on Docker Hub **segfaults under qemu** during `valhalla_build_tiles`
  (even on a 2-way graph). `pyvalhalla` is the identical engine run in-process and
  is the working ARM-native path; see BENCHMARKING.md for details and the
  Docker-based recipe for machines where it works.
- **Meili has no explicit no-match abstention.** Map-matching snaps every trace
  onto *something*, so it never emits a first-class "no match." We threshold
  instead: a matched Overture way is kept only if its overlap length is ≥ 10 % of
  the target OR ≥ 8 m (`min_match_frac` / `min_match_m`). Below that a target
  simply gets no pair. This is why recall is a perfect 1.000 on the labeled slice
  (every labeled-match target snaps to *some* qualifying edge) and why the
  precision cost shows up instead as parallel-road false positives.
- Meili's Overture→routable-graph conversion collapses vertices by rounded
  coordinate (7 dp ≈ 1 cm) to recover topology, and tags every way with a
  routable `highway=*` (Overture `class`, default `residential`); the GERS id is
  carried as the OSM `way_id`, so matched `way_id`s map straight back with no
  join. `pyvalhalla` requires Python ≥ 3.12 (the abi3 wheel), so the Meili adapter
  is unavailable on the 3.11 cbench CI env — its pure logic is unit-tested there
  instead (`test_meili_adapter.py`, `test_pbf_converter.py`).
