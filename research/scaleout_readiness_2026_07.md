# Scale-Out Readiness Audit — 2026-07

Scope: what it concretely takes to scale stitching + conflation from the current
2-dataset panel (`us_boston_streets`, `us_seattle_sidewalks`) to all available
datasets and beyond, plus a repair probe of the known non-US data-store debt.

Environment for all measurements: 10-core / 16 GB machine, model
`data/models/matcher_model_combined.joblib`, `FEATURE_VERSION 2026-07-04.2`,
79 features / 17 categories. Web UI left running on :8505; Boston/Seattle outputs
and `data/cache/stitch/` untouched.

## TL;DR

- **Inventory**: 24 dataset pairs are stitchable today (22 labeled); **10 labeled
  datasets are blocked** on a missing local target parquet; 9 more are
  Overture-only expansion candidates. 5,487 human labels across 34 datasets.
- **Non-US debt**: root cause was **GERS id churn, not S3** — Overture loads fine;
  backfill skipped exactly the 56 pairs whose gers_id left the current release
  despite stored geometries existing for all of them. **Fixed + repaired in
  PR #273** (584/584 computed, 0 skipped). Retrain is now unblocked (not run).
- **Pipeline validation**: Berlin (dense EU grid) 442 s / 7.2 GB / 91.5% matched,
  Tunis (ML-derived, zero names/classes) 300 s / 6.8 GB / 88.6% matched. #267's
  corridor gate generalizes: oversized rate ≤ 0.07% on both.
- **Scale-out cost**: all 24 stitchable pairs ≈ **3.5–5 h serial** on this machine;
  10x-Boston metro ≈ 25–35 min but **15–20 GB projected RSS** — memory is the first
  hard wall, then lack of incremental/resume, then single-machine feature throughput.
- **Next datasets**: Berlin, Tunis (run here), then geneva_pedestrian, sao_paulo,
  sydney.

---

## Task 1 — Dataset inventory

`data/raw/` holds two kinds of dataset:

- **Full pairs** (local target + Overture segments + connectors) — stitchable now.
- **Overture-only** (segments/connectors present, local target parquet absent) — a
  local re-fetch is required before they can be stitched.

### 1a. Stitchable full pairs (local + Overture present)

| Dataset | Type | Local segs | Overture segs | Connectors | Human labels (m/n/u) | Agent | Pipeline output |
|---|---|---:|---:|---:|---|---:|---|
| us_boston_streets | road (US) | 10,844 | 125,769 | 182,369 | 639 (280/346/13) | 100 | **yes** (panel) |
| us_seattle_sidewalks | sidewalk (US) | 46,145 | 165,503 | 260,542 | 200 (85/115/0) | – | **yes** (panel) |
| de_berlin_roads | road (EU) | 43,369 | 391,918 | 724,281 | 121 (86/35/0) | 84 | **this run** |
| tn_tunis_ml_roads | road (MENA, ML-derived) | 113,566 | 147,523 | 124,491 | 200 (152/48/0) | – | **this run** |
| au_melbourne_roads | road (AU) | 2,927 | 553,250 | 843,888 | 91 (79/12/0) | 62 | no |
| au_sydney_roads | road (AU) | 178,227 | 285,641 | 399,766 | 226 (147/79/0) | 100 | no |
| br_sao_paulo_roads | road (LatAm) | 212,264 | 407,337 | 431,530 | 94 (61/33/0) | – | no |
| ca_toronto_roads | road (CA) | 62,411 | 232,391 | 388,523 | 91 (59/24/8) | – | no |
| ch_geneva_hiking_routes | hiking (EU) | 48 | 193,520 | 314,747 | 50 (23/27/0) | – | no |
| ch_geneva_pedestrian_network | ped/sidewalk (EU) | 1,656 | 110,746 | 186,289 | 50 (34/16/0) | – | no |
| co_bogota_roads | road (LatAm) | 138,516 | 177,344 | 215,164 | 134 (98/36/0) | 87 | no |
| fi_helsinki_roads | road (EU) | 219,106 | 169,748 | 293,326 | 84 (66/18/0) | – | no |
| gb_london_roads | road (EU) | 264,756 | 608,687 | 886,645 | 201 (58/143/0) | 1 | no |
| in_mumbai_streets | road (IN) | 56,797 | 113,349 | 126,982 | 257 (198/58/1) | – | no |
| jp_tokyo_emergency_roads | road (JP) | 977 | 1,052,275 | 1,308,107 | 200 (148/52/0) | – | no |
| us_austin_sidewalks | sidewalk (US) | 11,945 | 207,118 | 330,704 | 94 (79/15/0) | 44 | no |
| us_boston_bike_network | cycle (US) | 3,477 | 125,769 | 182,369 | 86 (38/48/0) | 85 | no |
| us_boston_sidewalks | sidewalk (US) | 110,031 | 125,769 | 182,369 | 281 (76/204/1) | 100 | no |
| us_fort_collins_sidewalks | sidewalk (US) | 38,714 | 42,651 | 68,286 | 238 (212/26/0) | 28 | no |
| us_fort_collins_streets | road (US) | 10,994 | 42,651 | 68,286 | 202 (126/76/0) | 2 | no |
| us_frisco_trails | trail (US) | 529 | 31,241 | 39,510 | 177 (95/82/0) | 100 | no |
| us_montana_helena | road (US) | 8,174 | 22,137 | 34,740 | 192 (76/116/0) | 99 | no |
| us_philadelphia_sidewalks | sidewalk (US) | 204,760 | 189,229 | 283,336 | 345 (203/142/0) | 100 | no |
| us_usfs_lolo | trail (US) | 783 | 33,724 | 55,130 | 107 (97/9/1) | 59 | no |

(24 stitchable, of which 22 have human labels; `ch_grand_geneva_cycle_schema` and
`fr_france_winter_hiking_traces` have a local parquet but no Overture pair fetched.)

### 1b. Labeled but NOT stitchable — local target parquet missing (needs re-fetch)

These 10 datasets have human labels + features + Overture data, but the local
target parquet is absent from `data/raw/`, so `matcher stitch` cannot run until the
local source is re-fetched:

| Dataset | Type | Overture segs | Human labels | class_similarity |
|---|---|---:|---:|---|
| hk_hongkong_roads | road (HK) | 182,808 | 208 | **100% NaN** |
| ke_nairobi_roads | road (KE) | 131,086 | 50 | **100% NaN** |
| co_bogota_bike_network | cycle (LatAm) | 177,344 | 29 | **100% NaN** |
| nl_amsterdam_roads | road (EU) | 94,266 | 200 | 0% |
| sg_singapore_roads | road (SG) | 220,345 | 199 | 4% |
| sg_singapore_footpaths | footpath (SG) | 220,345 | 80 | 0% |
| us_utah_slc_roads | road (US) | 227,018 | 200 | 1% |
| ke_kisumu_roads | road (KE) | 17,257 | 107 | 0% |
| us_usfs_flathead | trail (US) | 24,716 | 50 | 0% |
| us_montana_missoula | road (US) | 25,255 | 4 | 0% |

### 1c. Overture-only, no labels (pure expansion candidates; need local fetch)

`ae_abudhabi_roads`, `co_bogota_sidewalks`, `ke_mombasa_roads`, `kr_seoul_roads`,
`us_ada_county_roads`, `us_fresno_roads`, `us_frisco_roads`, `us_gwinnett_roads`,
`us_montana_bozeman`, plus `us_boston_streets_osm` (a features-only OSM variant).

### 1d. class_similarity 100% NaN — root causes differ

Checked the actual source/stored class values for the four affected datasets:

| Dataset | Local class values | Diagnosis |
|---|---|---|
| tn_tunis_ml_roads | `"unknown"` x 113,566 (100%); names also 100% null | **Honest missing data** — ML-derived source carries no class/name; `semantic.py` maps `"unknown"` → NaN by design. Geometry-only matching. Not a bug. |
| hk_hongkong_roads | `"unknown"` x 208 (stored pairs) | Same — honest missing data. |
| ke_nairobi_roads | `"unknown"` x 50 (stored pairs) | Same — honest missing data. |
| co_bogota_bike_network | numeric codes `'2'`, `'1'`, `'5'` | **Genuine unmapped vocab** — fetch-time `class_mapping` was never applied; a mapping table in the dataset config recovers the feature. The only actionable fix of the four. |

Of these, only **tn_tunis_ml_roads** is currently stitchable (local present) — it is
one of the two datasets run in Task 3 below, deliberately exercising the
no-name/no-class regime end-to-end.

---

## Task 2 — Non-US data-store repair probe: **S3 is NOT the problem; it's GERS id churn**

Ran the backfill for all affected datasets:

```
uv run matcher backfill -D ch_geneva_hiking_routes -D ch_geneva_pedestrian_network \
  -D fi_helsinki_roads -D jp_tokyo_emergency_roads -D tn_tunis_ml_roads
```

Result: **exit 0 in 131.6 s, 6.3 GB peak RSS. Overture data loaded fine for every
dataset — no S3 fetch failure.** But the run reported `528 features computed,
56 skipped`, and the 56 skips break down *exactly* as the known debt:

| Dataset | Pairs | Computed | Skipped | Skipped gers_id absent from current Overture | Stored ref_geometry present for those |
|---|---:|---:|---:|---:|---:|
| fi_helsinki_roads | 84 | 37 | **47** | 47 | 47/47 |
| jp_tokyo_emergency_roads | 200 | 196 | **4** | 4 | 4/4 |
| ch_geneva_pedestrian_network | 50 | 46 | **4** | 4 | 4/4 |
| tn_tunis_ml_roads | 200 | 199 | **1** | 1 | 1/1 |
| ch_geneva_hiking_routes | 50 | 50 | 0 | 0 | – |
| **Total** | | **528** | **56** | **56** | **56/56** |

**Root cause (proven):** the 56 skipped pairs are exactly the ones whose labeled
`gers_id` no longer exists in the *current* Overture release (GERS reference-id
churn across releases). Backfill's Phase 3 resolves the **target** geometry from the
stored `labels/data` DataStore (augmented target GDF), but resolves the
**reference** geometry *only* from the freshly-loaded Overture parquet via
`ref_id_to_idx.get(gers_id)` (`src/matcher/cli/main.py` ~L1624–1636). When the
gers_id is gone from the release, `ref_idx is None` → the pair is skipped — **even
though `labels/data` contains a valid stored `ref_geometry` for all 56** (verified).

Consequences:
- This debt **cannot** be repaired by re-running backfill (with or without network).
  It is a code gap, not an availability problem. The earlier "S3 fetch failed"
  framing is stale — Overture now loads cleanly; the skip is upstream of any fetch.
- The 56 skipped pairs **retain their previous feature rows** (the FeatureStore
  merges: recomputed rows replace, skipped rows are kept), so they carry
  potentially-stale values while still being stamped the current `feature_version`
  — feature_version alone will never surface them.
- The 528 resolvable pairs recompute **deterministically**: diffing this run against
  the pre-run parquets showed 0 cell changes for helsinki / tunis / geneva_ped, and
  only tiny refreshes for jp_tokyo (44 cells) and geneva_hiking (3 cells) that trace
  to source parquets re-fetched today (which are **git-ignored / uncommitted**).

### The fix — implemented and verified (PR #273)

Because rerunning backfill cannot repair this (it is a code gap, not availability),
the fix was implemented in **PR #273** (`fix/backfill-ref-gers-churn`): mirror the
target-side augmentation for the reference side. gers_ids missing from the current
release are appended to the projected reference GDF from stored geometry +
names/class attributes (append-only; refs still in the release keep live data;
explicit `connectors=None` so connector iteration doesn't trip on concat-NaN).
Future GERS churn now self-heals from stored geometry.

Verification:
- Backfill on the 5 affected datasets: **584 computed / 0 skipped** (was 528/56).
- fi_helsinki changes exactly its 47 churned rows (1,384 cells) vs committed state.
- Recompute is deterministic for unaffected pairs (0 cell diffs on repeat runs).
- Small spillover on a few neighbor pairs (e.g. 5 extra rows in ch_geneva_ped) is the
  appended refs restoring the true label-time reference-network context for
  sibling/graphlet features.
- New CLI-level regression test `tests/unit/test_backfill_ref_churn.py` (synthetic
  churned gers_id) + existing parity/label-store/CLI tests green; ruff clean.

The refreshed `labels/features` parquets for all 5 datasets ride in the same PR
(coordinated-backfill pattern of #256), plus a `labels/data` topology write-back for
the repaired geneva_ped pairs. The main checkout was never dirtied (private worktree;
`main` stayed on `main`).

### Retrain status

A retrain remains a coordinated decision and was **not** performed. PR #273 is the
unblocker: once merged, all 584 non-US pairs are honestly computed. Expected impact
mirrors PR #256's analysis: the biggest mover is **jp_tokyo cross-script name pairs**
(honest NaN instead of an exploitable fake-0.0 pattern); the 56 previously-stale pairs
(dominated by helsinki, 47) stop poisoning topology/endpoint and coverage ablations.
Net headline metrics should move little (#256 saw 0.924→0.908 test acc, almost
entirely the newly-honest datasets); expect a correctness/robustness gain rather than
an accuracy jump.

---

## Task 3 — Stitching scale-out: next datasets + pipeline validation

### Recommended next datasets (diversity-optimized)

Ranked for coverage of dataset type, geography, and known gaps:

1. **de_berlin_roads** — dense European road grid; strongest generalization test for
   the #267 corridor-aware M:N grouping outside US data. *(run below)*
2. **tn_tunis_ml_roads** — non-US (MENA), ML-derived road geometry with **no names
   and no classes at all** (the hardest semantic regime the model faces; the
   `class_similarity` 100%-NaN dataset that is actually stitchable). *(run below)*
3. **ch_geneva_pedestrian_network** — non-US pedestrian/sidewalk network; diversifies
   the panel away from US-only sidewalks (Seattle) into a different segmentation style.
4. **br_sao_paulo_roads** — large LatAm road network (212k local / 407k Overture);
   stresses both scale and a distinct name/class vocabulary.
5. **au_sydney_roads** — mature English-language non-US roads with rich labels (226);
   good "road_good" contrast to the Tunis "road_poor" case.

(For the class-NaN gap specifically, `hk_hongkong_roads`, `ke_nairobi_roads`, and
`co_bogota_bike_network` are the other candidates but require a local re-fetch first —
see Task 1b.)

### Measured pipeline runs (TOP 2)

#### de_berlin_roads (dense EU road grid; 43,369 local x 391,918 Overture)

```
uv run matcher stitch de_berlin_roads -m xgboost -o data/output/de_berlin_roads_bridge.parquet
```

| Metric | Value |
|---|---|
| Wall time | **442 s (7.4 min)** — 1,436 s user CPU, 8 workers |
| Peak RSS | **7.2 GB** |
| STRtree candidates | 1,099,725 (25.4 per local segment) |
| Scored candidates (after clip prefilter) | 263,051 |
| Stage split | load 2.4 s · blocking ~1 s · **features 370.7 s (84%)** · XGBoost predict ~4 s · optimizer 1.9 s · group export ~7 s |
| Matched / Review / Unmatched | **39,668 / 1,737 / 1,964** (of 43,369) |
| Bridge rows | 51,031 (match 50,418, review 613) |
| M:N groups | **2,347** (of 10,210 groups: 3,150 1:N, 4,713 N:1) |
| Group size dist (edges) | mean 3.0 · p50 2 · p90 5 · p99 10 · max **177** |
| Size buckets | 2–5: 9,557 · 6–10: 559 · 11–20: 78 · **>20 (monster): 16** |
| oversized_group (export-gated) | **7** (0.07% of groups) |

**Verdict on #267 generalization:** holds up well on a dense non-US grid. Oversized
rate 7/10,210 (Boston reference: 2/3,393; Seattle 0/15,613) and monster (>20-edge)
rate 16/10,210 vs Boston 6/3,393 — same order. The 177-edge max group is the one
artifact worth a manual look, but it is correctly held back by the structural gate
rather than auto-exported.

**Panel cost:** covering all 2,347 M:N groups at prior-wave cadence (3 providers x 60
groups ≈ 180 votes/batch) ≈ **39 batches, ~7,000 votes** — 3.5x Boston's 664 M:N
groups. A triage policy (only oversized + monster + low-confidence M:N) collapses
this to a handful of batches.

#### tn_tunis_ml_roads (ML-derived MENA roads, zero names/classes; 113,566 local x 147,523 Overture)

```
uv run matcher stitch tn_tunis_ml_roads -m xgboost -o data/output/tn_tunis_ml_roads_bridge.parquet
```

| Metric | Value |
|---|---|
| Wall time | **300 s (5.0 min)** — 1,221 s user CPU, 8 workers |
| Peak RSS | **6.8 GB** |
| STRtree candidates | 1,322,811 (11.6 per local segment) |
| Scored candidates (after clip prefilter) | 331,607 |
| Stage split | **features 228.7 s (76%)** · optimizer 3.8 s · group export ~9 s |
| Matched / Review / Unmatched | **100,661 / 3,550 / 9,355** (of 113,566 → 88.6% matched) |
| Bridge rows | 128,340 (match 126,749, review 1,591) |
| M:N groups | **5,751** (of 31,131 groups: 10,392 1:N, 14,988 N:1) |
| Group size dist (edges) | mean 2.7 · p50 2 · p90 4 · p99 9 · max 67 |
| Size buckets | 2–5: 29,587 · 6–10: 1,317 · 11–20: 211 · **>20 (monster): 16** |
| oversized_group (export-gated) | **6** (0.02% of groups) |

**Verdict:** the geometry-only regime works — 88.6% matched with no name/class signal
at all, and the corridor gate stays tight (6 oversized / 31k groups, max group 67
edges vs Berlin's 177). Unmatched rate (8.2%) is the highest of the four measured
datasets, consistent with ML-derived geometry not always having an Overture
counterpart — a useful honest-negative source for labeling.

**Panel cost:** 5,751 M:N groups ≈ **96 batches / ~17,000 votes** at prior cadence —
full coverage is clearly infeasible; Tunis makes the case for triaged panels
(oversized + monster + review-band M:N only ≈ 1–2 batches).

#### Cross-dataset reference table

| Dataset | Local segs | Wall | Peak RSS | Scored pairs | Matched % | M:N groups | Oversized | Monster (>20e) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| us_boston_streets | 10,844 | – (prior) | – | – | 92.4%* | 664 | 2 | 6 |
| us_seattle_sidewalks | 46,145 | – (prior) | – | – | – | 240 | 0 | 2 |
| de_berlin_roads | 43,369 | 442 s | 7.2 GB | 263,051 | 91.5% | 2,347 | 7 | 16 |
| tn_tunis_ml_roads | 113,566 | 300 s | 6.8 GB | 331,607 | 88.6% | 5,751 | 6 | 16 |

*Boston matched% from bridge (10,025 matched + 1,099 review + 19 unmatched per README
workflow table; current output 15,549 bridge rows over 3,393 groups).

### Panel cost extrapolation

Prior waves ran 3 providers × ~60 groups ≈ 180 votes per batch. Panel cost scales with
**M:N group count**, not raw segment count. See the per-dataset M:N counts in the run
tables below and in the Boston/Seattle reference (Boston 664 M:N groups, Seattle 240).

---

## Task 4 — Throughput & bottleneck extrapolation

### Measured throughput (this machine, under parallel-agent load)

| Quantity | Berlin measured | Tunis measured |
|---|---|---|
| Wall time | 442 s | 300 s |
| Scored pairs | 263,051 | 331,607 |
| Effective pipeline rate | ~595 scored pairs/s (features-stage: ~710 pairs/s on 8 workers) | ~1,105 scored pairs/s (features-stage: ~1,450 pairs/s) |
| Peak RSS | 7.2 GB | 6.8 GB |

Feature computation dominates: **84% of wall time** (370.7 s of 442 s in Berlin).
Blocking (STRtree over 392k x 43k) is ~1 s; the optimizer with corridor-aware M:N
grouping is ~2 s; XGBoost prediction ~4 s. The ~53 µs/pair post-#255 figure applies
to the alignment hot loop; end-to-end per-scored-pair cost including all 79 features
is ~1.4 ms wall (8 workers, with CPU contention from concurrent agents — treat
these as conservative).

### (a) Cost of stitching ALL 24 stitchable local pairs

Totals: 1.74 M local segments, ~5.6 M Overture reference segments. Using the
measured 25.4 STRtree candidates and 3–6 scored candidates per local segment
(Berlin 6.1, Tunis 2.9), the full sweep is **~5–9 M scored pairs**. At the measured
~600 scored pairs/s plus per-dataset fixed overhead (load + topology/graphlet
precompute; 30–60 s typical, worst case `jp_tokyo` whose 1.05 M-segment reference
dominates its tiny 977-segment target), a full serial sweep is roughly
**3.5–5 hours on this 10-core machine**, peaking 7–10 GB RSS per large dataset.
That is tolerable as an overnight batch job, but it is a single sequential process
with **no resume**: one crash at dataset 19 restarts from zero unless run per-dataset
by hand.

### (b) Metro-scale run at 10x Boston

10x Boston ≈ 108k local x 1.26 M reference segments. Extrapolating from Berlin
(which is already metro-scale on the reference side): ~0.8–1.2 M scored pairs
→ **~25–35 min wall** for compute. The binding constraint is not time but
**memory**: Berlin peaked at 7.2 GB with a 392k-segment reference; the full-GDF,
in-RAM architecture scales peak RSS roughly with reference size + candidate count,
projecting **15–20 GB** at a 1.26 M-segment reference — over this machine's 16 GB.
`jp_tokyo_emergency_roads` (1.05 M ref segments) is the real-world canary already in
the repo. Without spatial tiling, metro-scale runs need a bigger box or reduced
`--workers` (which trades the only lever that keeps wall time acceptable).

### Top engineering bottlenecks for routine 30+-dataset operation (ranked)

**1. Single-process memory ceiling (no spatial partitioning).** The pipeline holds
the full reference GDF, target GDF, candidate list, per-worker copies, and the
groups sidecar in RAM simultaneously. Measured 7.2 GB peak on Berlin (392k ref);
projected 15–20 GB at 1.26 M ref segments (10x-Boston / jp_tokyo scale). This is the
first hard wall for metro-scale and for running datasets concurrently. Fix: tile the
reference/target spatially (H3 cells are already the ID convention), stream
candidates per tile, and merge groups at tile borders — or at minimum a
`--low-memory` mode that chunks the scored-candidate materialization
(`MatchResult` objects are built for all pairs before optimization).

**2. No incremental/resume execution — and no GERS-churn handling in stored
artifacts.** Every `matcher stitch` recomputes blocking + features from scratch;
a crash mid-sweep loses everything; `matcher stitch --all` is serial. Worse, the
Task 2 finding shows outputs are pinned to an Overture release with no re-map path:
labeled gers_ids already vanished across releases (56 pairs), and every bridge
parquet produced today will face the same churn on the next release. Routine 30+-
dataset operation needs (a) a per-dataset cache keyed on
(input-hash, FEATURE_VERSION, model-hash) so unchanged datasets are skipped,
(b) per-stage checkpoints (scored candidates parquet → optimizer can rerun alone —
the optimizer is 2 s while scoring is 7 min, yet today you cannot re-optimize
without re-scoring), and (c) a stored-geometry fallback for reference ids
(the Task 2 fix) applied consistently in backfill and audit paths.

**3. Feature-stage throughput is single-machine only (84% of wall).** ~600 scored
pairs/s caps a full sweep at hours and a continental run at weeks. The Spark export
path exists for the Overture-side model, but stitching (optimizer + corridor
grouping + group export) has no distributed story. Cheapest wins first: the clip
prefilter already cuts 1.1 M → 263k (Berlin), so (a) push more cheap rejection
before feature computation (coverage/length heuristics), (b) make `--all` run
datasets in parallel processes with a memory budget (small datasets like
us_frisco_trails, usfs_lolo, geneva are nearly free), and (c) only then consider a
distributed feature path reusing the existing `prepare_worker_data` chunking.

**Runner-ups.** (4) *Local-data acquisition debt*: 10 of 34 labeled datasets cannot
be stitched because the local target parquet is missing — re-fetch is scripted per
meta.yaml but not automated or verified in CI; several meta.yamls are missing
entirely (tn_tunis, br_sao_paulo, gb_london locals). (5) *Panel/review scaling*:
M:N group count grows linearly with dataset size (Berlin 2,347), so full-coverage
panels are unaffordable — triage (oversized + monster + low-confidence) must become
the default policy. (6) *Overture S3 fetch reliability*: currently healthy (verified
in Task 2), but auto-fetch happens silently inside backfill with no retry/verify
step; a flaky fetch would surface as silently skipped pairs again.
