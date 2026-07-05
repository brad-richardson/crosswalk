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
- **Stitch quality is gated**: cbench `--gate` enforces sliver-filtered edge-F1
  ≥ 0.78 / exact ≥ 0.45 on Boston (baseline 0.8345 / 0.537, 67 mapped labels,
  armed). Seattle arms automatically at 30 mapped labels (currently 27).
- **Ground truth**: 140 curated stitching labels (113 Boston / 27 Seattle);
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
  because rejected candidates are discarded).
- Ship the flag-gated confidence-drop prune from the #272 eval (the
  one-parameter model that beats the prototype: clean-slice F1 0.872 vs 0.828
  keep-all baseline) behind the stitch gate.

### M3 — Learned group resolver (armed by M1 + M2)

Goal: replace hand-written edge selection inside groups with a learned model,
once the flip conditions hold.

- Re-run the #272 harness (grouped CV, optimizer baseline, sliver-filtered)
  at 150–300 labels WITH rejected candidates in the table. Panel soft labels
  (reliability-weighted, dissenter down-weighted) were the biggest single
  lever (+0.05 F1) — keep them in.
- Promotion criterion: beat the tuned confidence threshold on the clean slice
  AND move group exact-match, under the stitch gate.
- Hybrid formulation experiment (from the Meili pilot, `research/meili_baseline.md`):
  map-matching as high-recall candidate/path generator (perfect recall on all
  three benchmark datasets, native segmentation-mismatch handling) + learned
  per-edge scorer for precision (parallel-geometry rejection is exactly what
  the feature set is good at). This is now the concrete shape of EVAL_ROADMAP's
  "path-based formulation" long bet, and it is an *evolution* of the current
  architecture, not a rewrite.

### M4 — Bridge-table factory orchestration (the 20-core box)

Goal: `matcher factory run` — all stitchable datasets, unattended, restartable.

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

- Versioned layout: `r2://…/release=<overture-release>/dataset=<name>/bridge.parquet`
  plus a unified long table (`gers_id → [dataset, local_id, confidence,
  match_type, group_id, review_status]`) and a manifest. Queryable directly via
  DuckDB/HTTP range reads; no serving infrastructure.
- Quality metadata per row is the differentiator: calibrated P(match) (#266
  means the number is honest), match type, stitching-group membership, review
  provenance — consumers pick their own precision/recall operating point.
- The credibility page: benchmark table (M6), per-dataset stitch-gate status,
  label provenance stats.
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
