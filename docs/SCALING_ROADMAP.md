# Scaling Roadmap

Written 2026-07-05, synthesizing the seven-workstream scaling wave (PRs #271–#277)
plus the product-scope decisions made the same day. This is the planning document
for scaling the matching + stitching pipeline; `docs/EVAL_ROADMAP.md` remains the
eval-methodology tracker.

## Product north star

**A bridge-table factory**: match many local datasets to Overture, producing
versioned, quality-annotated `local_id ↔ GERS` bridge tables (plus curated
stitching groups), published on Cloudflare R2 for public consumption (free
egress; pattern modeled on the geocoder repo).

Explicitly **in scope**: matching, M:N stitch resolution, per-Overture-release
regeneration, publishing, possibly opinionated *attribute* merging (best-of
name/class keyed by GERS id — a dataframe join, not conflation).

Explicitly **out of scope / deferred**: geometry merging ("Overture-land"),
large Spark jobs (hobby-scale project; the Spark-portable model export stays
healthy because matcher remains the technical source of truth for
tf-data-platform, but it is not a milestone), metro-scale spatial tiling
(unnecessary below ~1M ref segments on the 64 GB box).

Infrastructure reality: solo developer; three personal machines, of which one
always-on 20-core / 64 GB box is the scale workhorse. Everything below is sized
for single-node operation.

## Where things stand (measured, 2026-07-05)

- **Match quality vs the field** (`docs/BENCHMARK_RESULTS.md`): target-level
  match F1 on Boston streets — naive 0.839 / Hootenanny-0.2.41 0.973 / Valhalla
  Meili 0.994 / **matcher 0.996**; Fort Collins sidewalks — 0.365 / 0.927 /
  0.962 / **0.976**. Matcher leads every row. (Naive rows re-measured 2026-07-05
  after fixing a symmetric-Hausdorff guard bug that had halved its road recall.)
  The naive baseline's **sidewalk collapse (recall 0.24, F1 0.365)** remains the
  one-number justification for the learned approach: after the fix the naive
  floor is respectable on roads (recall 0.95, geometry alone finds the
  correspondences — the road gap to matcher is now precision, not recall), so
  the case for ML rests on dense parallel geometry where buffer+bearing can't
  separate near-parallel candidates and recall stays near 0.2.
- **Stitch quality is gated**: mbench `--gate` enforces sliver-filtered edge-F1
  ≥ 0.83 / exact ≥ 0.50 on Boston (2026-07-12 baseline 0.9120 / 0.5714,
  112 mapped pair labels, armed). Seattle arms automatically at 30 mapped pair
  groups and remains unarmed in the latest gate evaluation.
- **Ground truth**: 226 curated stitching labels across 13 datasets (178 pair /
  48 set): Boston 119 (113 pair / 6 set) and Seattle 49 (22 pair / 27 set);
  5,543 pair labels across 34 datasets, all backfillable again after the GERS
  id-churn fix (#273). Panel economics: ~0.68 exportable labels/group on roads,
  ~0.45 on sidewalks at the current 3-provider composition.
- **Throughput** (scale-out audit, #274): Berlin (43k×392k) runs in 7.4 min /
  7.2 GB RSS; Tunis (114k×148k, zero names/classes) matches 88.6%. Corridor
  grouping (#267) generalizes: oversized groups ≤ 0.07% everywhere measured.
  Feature scoring is 84% of wall time (~600 scored pairs/s/process); the
  optimizer is ~2 s.
- **Resolver verdict** (#272): at ~40 clean labeled groups a one-parameter
  confidence prune beats a learned per-edge model; group exact-match doesn't
  move. Under-selection is unlearnable because the sidecar only persists
  optimizer-selected edges. Flip conditions: persist the full candidate graph,
  reach ~150–300 labeled groups.

## Milestones

Ordered by dependency, not strictly by time; M1/M2 are immediate, M4 can start
in parallel, M5 closes the loop.

### M1 — Label factory industrialization

Goal: 150–300 curated stitching labels across ≥4 datasets, produced routinely
rather than by hand-run waves.

- Triage policy (from the audit): panel coverage targets oversized + monster +
  review-band M:N groups, never full coverage (Tunis full coverage ≈ 17k votes
  — infeasible).
- Scheduled waves on the always-on box (provider CLIs are I/O-bound; nightly
  batches). The resumable per-group driver from the #277 wave (partial-CSV
  appends, usage-cap wait-and-retry) becomes the standard runner instead of the
  all-or-nothing one.
- Seattle to 30+ mapped labels arms its stitch-gate floor with zero code change.
- Expand to Berlin (roads, non-US) and one more sidewalk/cycle dataset once
  their batches exist; new-dataset batches now cost one pipeline run + one
  panel wave.
- Dissent routing refinement: codex is the lone-holdout on roads (10/18
  majority-only groups), agy on sidewalks (10/19) — route lone-holdout patterns
  to auto-accept-with-audit-sample rather than human review once the audit
  sample confirms the pattern holds.
- Pair-label trust cascade (ensemble-agreement routing, `agent_weight` > 0)
  stays on the list from EVAL_ROADMAP #4 — it scales the *pair* label base the
  same way the panel scales stitching labels.

### M2 — Candidate-graph persistence + confidence-drop filter (resolver prerequisite)

Goal: make under-selection learnable and bank the cheap optimizer win.

- Persist ALL candidate edges per M:N group in the sidecar (selected +
  rejected, with confidences and structural features), not just selections.
  ~0.5 wk; prerequisite for every learned-resolver step. Also fixes the
  review-UI blind spot the user hit personally (extra plausible edges invisible
  because rejected candidates are discarded). **DONE in two layers**: #282
  added `rejected_edges` (non-selected candidates incident to a group's nodes,
  capped per group, with the structural layer); the follow-up
  `candidate_edges` key persists the FULL per-component floor-passing candidate
  graph per group — uncapped, uniform minimal schema (`confidence`, `selected`,
  `selected_elsewhere`), deterministic single-group attribution — flipping the
  #272 "persist the full candidate graph" condition
  (`stitch_persist_candidate_graph`, default on).
- Ship the flag-gated confidence-drop prune from the #272 eval (the
  one-parameter model that beats the prototype: clean-slice F1 0.872 vs 0.828
  keep-all baseline) behind the stitch gate. **DONE — per-dataset opt-in
  (allowlist)** (#282 shipped it flag-gated; #284 tuned per-dataset thresholds;
  the allowlist cutover made it opt-in). Validated under the #271 gate: Boston
  (117 labels) filtered edge-F1 0.8671→0.8790, group exact 0.5093→0.5833 at
  t=0.96, gate PASS; Seattle (27 labels) 0.8665→0.8913 / 0.40→0.50 at t=0.90
  (0.96 over-prunes the lower-confidence sidewalks). The prune applies ONLY to
  datasets in the `resolver_prune_overrides` **allowlist** (Boston 0.96, Seattle
  0.90); there is no global default floor, so never-tuned datasets are not pruned.
  **To add a dataset: tune its floor via the #284 sweep recipe first** (sweep the
  confidence threshold on that dataset's clean stitching slice under the #271 gate,
  pick the F1/exact-maximizing point, confirm it beats keep-all), THEN add the
  validated `{dataset: threshold}` entry to `resolver_prune_overrides`. See
  ARCHITECTURE.md "Resolver confidence-drop prune".

### M3 — Learned group resolver (armed by M1 + M2)

Goal: replace hand-written edge selection inside groups with a learned model,
once the flip conditions hold.

- Re-run the #272 harness (grouped CV, optimizer baseline, sliver-filtered)
  at 150–300 labels WITH rejected candidates in the table. Panel soft labels
  (reliability-weighted, dissenter down-weighted) were the biggest single
  lever (+0.05 F1) — keep them in.
- Promotion criterion: beat production and the tuned-confidence control on the
  clean slice AND move group exact-match under the stitch gate. Use repeated
  grouped CV plus paired whole-group bootstrap intervals, then require the gain
  to transfer under leave-one-dataset-out evaluation; a single five-fold point
  estimate is not sufficient.
- Hybrid formulation experiment (from the Meili pilot, `research/meili_baseline.md`):
  map-matching as high-recall candidate/path generator (perfect recall on all
  three benchmark datasets, native segmentation-mismatch handling) + learned
  per-edge scorer for precision (parallel-geometry rejection is exactly what
  the feature set is good at). This is now the concrete shape of EVAL_ROADMAP's
  "path-based formulation" long bet, and it is an *evolution* of the current
  architecture, not a rewrite.

### M4 — Bridge-table factory orchestration (the 20-core box)

Goal: `crosswalk factory run` — all stitchable datasets, unattended, restartable.

**Status: orchestration SHIPPED** (`crosswalk factory` command group; see
[FACTORY.md](FACTORY.md)). Landed: the parallel-process work queue with failure
isolation + run-summary table; the versioned `data/factory/release=…/dataset=…/`
layout (bridge + groups + per-dataset manifest + scored cache + log); incremental
skip + automatic resume via a manifest `full_key`; the
re-optimize-without-rescore fast-path (`factory reoptimize`, ~2 s from the scored
cache); and the per-release GERS churn delta report (`factory delta`). The
pipeline was refactored minimally to expose score-then-optimize as separable seams
(`load_and_filter_inputs` / `optimize_and_export`) — the normal `crosswalk stitch`
path is behavior-identical and the stitch tests / mbench gate stay green.
Still open under M4: adopting Boston/Seattle into the factory layout (they stay on
the legacy `data/output/` path for now to keep the review queues stable),
box deployment, and the inventory repair below.

- Work queue over dataset pairs with per-dataset parallel processes (feature
  scoring parallelizes across datasets trivially; full current inventory ≈ one
  overnight run).
- Incremental/resume: re-optimize-without-rescore first (2 s vs 7 min —
  biggest iteration win for grouping changes), then score-cache keyed by
  (dataset, Overture release, FEATURE_VERSION, model hash).
- Overture release lifecycle: per-release regeneration, GERS churn delta
  report per dataset (the #273 stored-geometry fallback is the label-side
  half; the factory needs the output-side half).
- Inventory repair: re-fetch the 10 labeled datasets whose local parquets are
  missing from data/raw; fix the one real class-vocab bug (bogota bike numeric
  codes).
- Memory: Berlin peaks at 7.2 GB → jp_tokyo-scale (~1.26M ref segs) projected
  15–20 GB — fits the 64 GB box without tiling. Tiling stays deferred.

### M5 — Publish: R2-hosted bridge tables

Goal: the public artifact that makes the project useful to others.

**Status: publish tooling SHIPPED** (`crosswalk factory publish`; see
[PUBLISHING.md](PUBLISHING.md)). Landed: the pipeline-free publisher that
license-gates factory outputs and assembles a deterministic staging tree
(`bridges/release=<X>/dataset=<name>/{bridge.parquet, manifest.json}` copied
verbatim, a per-release unified `all_bridges.parquet` sorted by `gers_id` for
reverse lookups, SHA-256 `checksums.txt`, a machine-readable `index.json` with a
`latest_release` pointer, and a self-contained credibility `index.html`); the
license registry `datasets/licenses.toml` (the dataset YAMLs carry no license
field) with an approved/pending-review gate that **never guesses**; immutable
release paths (`--force` to overwrite); `--dry-run` (default) + `--target-dir`
(local, no creds) + the S3-compatible `aws s3 sync` R2 path (creds from `R2_*`
env vars). Validated end-to-end with `--target-dir` against the Mac's `data/factory`
(2 published: `us_usfs_flathead`, `us_montana_missoula`; 3 excluded pending license
review: 2× Singapore + Bogotá bike).

**Awaits the user (credentials):** create the R2 bucket + API token, set the
`R2_*` env vars, run the first `--no-dry-run` R2 upload. Also awaits human license
review to approve more datasets. Recommended to **publish from the always-on box**
(co-located with the full factory sweep) rather than sync `data/factory` back to
the Mac — see PUBLISHING.md "Open operational question".

Design decisions banked in PUBLISHING.md:
- Layout mirrors the factory partitioning; `release=` paths immutable; "latest" is
  a pointer field in `index.json`, not a mutable copy.
- Both per-dataset `bridge.parquet` (primary) and a per-release unified
  `all_bridges.parquet` (reverse lookup; sizes justify it at 34 datasets × 1–50 MB).
- Quality metadata per row is the differentiator: calibrated P(match) (#266),
  match type, review decision, fractional overlap — consumers pick their operating
  point.

Still open under M5:
- The credibility page's stitch-gate column is partial: only `us_boston_streets`
  is armed, and Boston/Seattle are not yet in the factory (legacy `data/output/`),
  so factory datasets show "no stitching labels" until they are adopted.
- Optional layer: opinionated merged attributes (best-of names/class keyed by
  GERS) — post-process join, only after the factory runs clean.

### M6 — Baselines & credibility (mostly done, keep honest)

- Hootenanny: **frozen one-shot baseline** — version-pinned (0.2.41 image),
  config recorded, not re-run routinely. Optional: one native run on the box
  that still has the compose-built stack for valid wall-time numbers (give it
  memory; it OOM'd there once).
- Meili: the **live** external baseline — maintained, ARM-native, re-run as
  matcher improves. GraphHopper map-matching documented as the drop-in
  alternative sharing the PBF conversion.
- Re-measure benchmark rows whenever labels grow materially (they are
  label-relative) and cite `docs/BENCHMARK_RESULTS.md` as the single source.

## Deferred (recorded so they stay deliberate)

- Spark/tf-data-platform consumption of calibration knots — their-side work;
  keep the export manifest current, nothing more.
- Geometry merging / conflated output — Overture-land, revisit only with a
  concrete consumer.
- Spatial tiling / distributed scoring — unnecessary below ~1M ref segments
  per dataset on the 64 GB box.
- Per-dataset-type decision thresholds from calibrated PR curves — data-thin
  until M1 delivers.
