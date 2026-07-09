# Unmatched-Segment Export: Design

**Date:** 2026-07-06 (§7 decisions + layout-cohesion review recorded 2026-07-09)
**Status:** Proposed (design only — no implementation in this PR); all §7
questions decided
**Owner:** Brad

## Motivation

The product today is per-dataset bridge tables — ID-only `local_id ↔ gers_id`
mappings published to R2 behind a human license gate
([PUBLISHING.md](../PUBLISHING.md)). The North Star is join-ability between
city data and the open map.

The *complement* of the bridge is just as valuable and currently thrown away at
the publishing boundary: the segments in a license-cleared municipal inventory
that have **no Overture counterpart**. Those rows are:

1. **Raw material for OSM mappers** — a license-vetted, machine-readable "here
   is infrastructure the open map is missing in your city" feed, usable for
   *manual* mapping (armchair + survey), without us doing any geometry imports
   ourselves. This stays deliberately clear of sidewalk-geometry conflation
   into the map — we stage material; mappers and the OSM import process decide.
2. **The intro mechanism for Overture** — a per-release, GERS-adjacent,
   quality-annotated candidate feed that demonstrates exactly what "baking in"
   municipal data would look like, before Overture builds any of it.

This document designs the **unmatched-segments artifact**: definition, schema,
license gating (geometry redistribution is a materially different exposure than
ID-only bridges), pipeline integration, the Overture handoff angle, sizing on
real outputs, and a phased build plan.

## What exists today (grounding)

- **Decisions.** `MatchDecision` is `match` / `review` / `no_match`
  (`src/crosswalk/matching/types.py:12-17`). The optimizer assigns per-group
  MATCH when `avg_confidence >= optimizer_review_threshold` (default 0.5),
  else REVIEW (`src/crosswalk/matching/optimizer.py:496-513`,
  `src/crosswalk/config.py:688-693`); conflicting matches can also be demoted
  to REVIEW (`optimizer.py:1043-1055`). `no_match` rows never reach the bridge
  (`src/crosswalk/resolution/bridge.py:69`), and a per-edge
  `bridge_min_confidence` (default 0.5, `config.py:898-903`) filters further.
- **An unmatched report already exists — but it is impoverished.**
  `generate_unmatched_report()` (`src/crosswalk/resolution/bridge.py:116-174`)
  writes `unmatched.parquet` next to every bridge: targets not in the MATCH
  set, with `unmatched_reason ∈ {no_match_found, low_confidence_review}`. It
  keeps only `["local_id", "name", "road_class"]` *if present* — but ingested
  targets use Overture-normalized columns `id`, `names` (struct with
  `primary`), `class`, `subtype` (verified on `data/raw/*_v1.0.parquet`), so
  **none survive**. Real factory outputs contain only
  `[geometry, unmatched_reason, release, dataset]` — no local_id, no
  attributes, no context. Fixing this is the core of Phase 1.
- **The factory already ships it internally.** Every factory run writes
  `unmatched.parquet` per `release=/dataset=` dir
  (`src/crosswalk/factory/runner.py:34`, [FACTORY.md](../FACTORY.md) "Output
  layout"), and the manifest records `n_unmatched`
  (`src/crosswalk/factory/manifest.py:218`). The bridges publisher copies
  only `bridge.parquet` + `manifest.json`
  (`publish.py::assemble_staging`).
- **"No candidate" is derivable.** `scored_candidates.parquet` caches every
  scored pair with `ref_id, target_id, decision, confidence, …`
  (`src/crosswalk/factory/scored_cache.py:89-101`). A target absent from it had
  **zero candidates within the 75 m buffer**; a present-but-unselected target
  had candidates that scored/optimized out.
- **QA vocabulary already exists.** The `/qa` route
  (`src/crosswalk/web/routes/qa.py`) reviews integration layers (net_new /
  disconnected / filtered) and persists decisions via
  `OrphanDecisionStore` with `decision ∈ {correct, incorrect}` and
  `reason ∈ {legitimate_new, data_error, out_of_scope}` plus context features
  (`length_m`, `road_class`, `nearest_main_dist_m`, `component_size`) —
  `src/crosswalk/integration_qa/decision_store.py:19-34`. That is exactly the
  "genuinely missing from Overture" vs "bad source geometry" signal the
  artifact needs.
- **A noise-screening framework exists.** `crosswalk analyze screen`
  (`src/crosswalk/cli/analyze.py:162-210`) validates unmatched targets against
  external context — `WaterBodyTest`, `BuildingTest`, `LandcoverTest`,
  `FringeTest` (`src/crosswalk/screen/__init__.py`).
- **License gate v1.** `datasets/licenses.toml` has a single `status` field;
  `LicenseRegistry.decision()` is default-deny
  (`src/crosswalk/factory/licenses.py:77-119`). *(Updated 2026-07-09:)* the
  registry now holds 45 entries: **26 `approved`** (the 5 US-PD
  federal/Montana datasets plus the license-panel burndown approvals — CC-BY,
  OGL, KOGL, PDDL, CC0, ODbL, …) and 19 `pending_review`. Registry entries
  also carry `panel_recommendation` / `panel_confidence` /
  `proposed_attribution` fields from the license research panel (#320) — the
  `geometry_status` addition below slots alongside those, and geometry-level
  review can reuse the panel's evidence dossiers. Orthogonal to the license: a
  declarative `quality_hold:` block in `datasets/<name>.yaml` excludes a
  dataset from **both** publish paths (bridges and targets) since #380,
  checked before the license.
- **Target snapshots shipped after this design was drafted (#370,
  2026-07-08).** `crosswalk factory publish --targets` publishes the raw
  ingested target parquets under a sibling
  `targets/dataset=<name>/snapshot=<fetch-date>/` prefix
  (`src/crosswalk/factory/publish_targets.py`), immutable per snapshot, gated
  by the same `status` + quality hold. Crucially, those snapshots are
  **geometry-bearing and carry the full verbatim `source_tags` attribute
  dict** (`fetch/target.py` preserves every original source column there) —
  i.e. the project already redistributes source geometry + attributes under
  the single-field gate this section calls v1. §3's geometry gate must
  therefore cover `targets/` too — see "Layout cohesion" in §4.
- **Publisher layout.** Hive `bridges/release=<R>/dataset=<name>/`, immutable
  releases, per-release `index.json` + `checksums.txt`, top-level `index.json`
  + credibility `index.html`, Pages browser in `site/` (DuckDB-WASM)
  (`src/crosswalk/factory/publish.py:234-460`, [PUBLISHING.md](../PUBLISHING.md)).

---

## 1. Definition: what ships

**Unit = one target (local) segment**, keyed by `local_id` (already
H3-suffixed and stable across releases:
`us_montana_missoula_74045_882898cf47`). One row per unmatched target segment.

### Three reason tiers, all present, reason-coded

| `reason` | Definition | Derivation |
|---|---|---|
| `no_candidate` | Nothing in Overture within the 75 m candidate buffer | `local_id` absent from `scored_candidates.parquet` |
| `below_threshold` | Candidates existed but none survived scoring/optimization (best confidence < floors, or optimizer dropped it) | in scored cache, not in bridge MATCH/REVIEW sets |
| `review` | In the bridge with `match_decision = 'review'` — a *plausible* Overture counterpart exists | bridge rows |

**Decision (2026-07-09, §7 Q1): ship all three tiers in one artifact**, because the split is
the single most useful column for both audiences: mappers should work
`no_candidate` first (highest precision "genuinely missing"), treat
`below_threshold` as "verify against imagery", and generally **not** map
`review` rows (a counterpart likely exists — the row is included so the
artifact partitions the whole dataset and consumers can reconstruct
target-side accounting). The headline "unmatched" count = `no_candidate +
below_threshold`; `review` is surfaced separately, consistent with how the
credibility page already excludes review from match rates
(PUBLISHING.md "Honest caveats"). This replaces today's coarse
`no_match_found` / `low_confidence_review` split (`bridge.py:155-156`), which
conflates "no candidate at all" with "candidates all scored out".

> Accounting quirk to fix while here: the manifest's `n_unmatched`
> (`pipeline/runner.py:1190-1192`) subtracts the full review-id set, while
> `unmatched.parquet` flags only review ids that aren't also matched — e.g.
> `us_montana_missoula` manifest says 103 unmatched but the file holds 272
> `no_match_found` rows. Define published counts **from the artifact**, one
> row per target segment, tiers mutually exclusive (a target that is both
> matched and reviewed elsewhere counts as matched and does not appear).

### Sliver / degenerate filtering

The artifact must not be noise. Filters, in order:

1. **Structural (already applied upstream):** null geometries and
   non-LineStrings never reach matching (`pipeline/runner.py:970-981`).
2. **Hard drop `length_m < 5`:** mirrors the sliver framework's 5 m absolute
   floor (`config.py:79-130`) — a sub-5 m orphan stub is not mappable
   infrastructure. On `sg_singapore_footpaths` this removes 11,570 of 57,582
   rows (20%) at a stroke.
3. **Flag, don't drop, 5–10 m** (`is_short = true`) — mirrors the "borderline"
   display band concept; consumers filter to taste.
4. **Screen flags (Phase 2, additive):** where `analyze screen` context is
   available, carry `screen_flags` (e.g. `in_water`, `in_building`,
   `fringe`) from the existing tests (`src/crosswalk/screen/`). Never
   row-dropping — screening is advisory and context fetches are optional.

No geometric simplification: geometries ship verbatim from the ingested target
parquet (WGS84), same provenance discipline as the bridge's verbatim copy.

---

## 2. Schema + format

### Formats — decision

- **Canonical: GeoParquet** (`unmatched.parquet`), WGS84, one file per
  `release=/dataset=`. Same reader story as the bridges (DuckDB range reads;
  DuckDB spatial reads GeoParquet), Hive-partition friendly, compact.
- **Sidecar: gzipped GeoJSON** (`unmatched.geojson.gz`), generated at publish
  time from the GeoParquet. This is the mapper on-ramp: JOSM opens GeoJSON
  natively, iD/RapiD workflows and umap/kepler/geojson.io all speak it.
  Properties = the full schema below (GeoJSON is lossless for it).
- **PMTiles: NO for now** (deferred, Phase 3 only if demanded). Rationale: we
  have no tippecanoe in the toolchain (the only tiling code is the dynamic
  server-side MVT route, `src/crosswalk/web/routes/tiles.py`, which the static
  Pages site can't use), per-dataset row counts are small enough for a
  bbox-filtered DuckDB-WASM → Leaflet preview (biggest dataset today ≈ 46k
  rows post-sliver-filter), and PMTiles adds a build dependency + artifact for
  a preview nicety. Revisit when a dataset exceeds ~200k unmatched rows or the
  Pages preview proves inadequate.

### Schema (GeoParquet columns)

| column | type | notes |
|---|---|---|
| `local_id` | string | H3-suffixed stable id — **fixes today's missing-id bug** (id column is `id`, not `local_id`; `bridge.py:160`) |
| `geometry` | LineString, WGS84 | verbatim from ingested target |
| `name` | string, nullable | extracted `names.primary` from the Overture-normalized struct |
| `class` | string, nullable | Overture vocab (already normalized at ingestion) |
| `subtype` | string, nullable | Overture vocab (`road` / `footway` / …) |
| `source_tags` | struct/map, nullable | **decided 2026-07-09 (§7 Q2):** verbatim original source attributes, copied unchanged from the ingested target parquet's `source_tags` column — the same bytes the `targets/` snapshots already publish. Per-dataset key sets vary (schema is stable within a dataset file) |
| `length_m` | float64 | computed in UTM at export |
| `is_short` | bool | 5 ≤ length < 10 m |
| `reason` | string | `no_candidate` \| `below_threshold` \| `review` |
| `n_candidates` | int32 | scored candidates within buffer (0 for `no_candidate`) |
| `best_candidate_gers` | string, nullable | argmax-confidence candidate from the scored cache |
| `best_candidate_confidence` | float64, nullable | calibrated P(match) of that candidate |
| `nearest_gers_id` | string, nullable | `sjoin_nearest` against the reference at export time |
| `nearest_gers_distance_m` | float64, nullable | the join-ability context: "closest existing map feature is X m away" |
| `qa_status` | string | Phase 2: `confirmed_missing` \| `bad_source_geometry` \| `out_of_scope` \| `unreviewed` (default) |
| `screen_flags` | list<string>, nullable | Phase 2, from the screen framework |
| `dataset` | string | provenance |
| `release` | string | Overture release the run matched against |
| `pipeline_version` / `feature_version` | string | mirrors bridge/manifest provenance |
| `generated_at` | timestamp (UTC) | run timestamp |

`nearest_gers_*` vs `best_candidate_*` are deliberately distinct: the former is
pure proximity (meaningful even with zero candidates — "nearest mapped road is
430 m away" reads as *genuinely unmapped area*), the latter is the model's best
rejected hypothesis ("there is a segment 8 m away but it scored 0.2" reads as
*geometry disagreement — verify against imagery*).

**`source_tags` ships verbatim** (decided 2026-07-09, §7 Q2) in every
published file — the artifact only publishes `geometry_status = "approved"`
datasets, and that review now includes an attribute/PII glance (§3). The
column name deliberately matches **Overture's own schema property
`source_tags`** — defined in `OvertureMaps/schema` `schema/base/defs.yaml`
(`sourceTags`: *"Any attributes/tags from the original source data that should
be passed through"*, `type: object`) and described in the docs as *"key-value
pairs imported directly from the source data without change"*
(<https://docs.overturemaps.org/schema/reference/base/land/>). It is also
already this codebase's internal name for exactly this data
(`fetch/target.py` writes it at ingestion; the published `targets/` snapshots
carry it). `original_source_tags` was considered and rejected — it matches
neither Overture nor the existing artifacts. Representation: copy the
ingested column as-is (per-dataset struct); if cross-dataset uniformity ever
matters, Overture's Parquet encoding (`map<string,string>`, stringified
values) is the normalization target. The GeoJSON sidecar carries it as a
nested JSON object — lossless.

---

## 3. License gate v2: geometry is a different exposure

A bridge row is an ID cross-reference; courts and data-license terms treat
"substantial extraction" of geometry very differently. Redistributing source
geometry is, in substance, **republishing the dataset**. The gate must reflect
that.

### Registry change (`datasets/licenses.toml`)

Add one field per dataset, orthogonal to the existing `status`:

```toml
[datasets.us_montana_missoula]
status = "approved"            # gates the ID-only bridge (existing, unchanged)
geometry_status = "approved"   # gates geometry-bearing artifacts (NEW)
license = "US-PD"
attribution = "Montana State Library (MSDI Transportation Framework)"
```

- **Vocabulary identical to `status`** (`approved` / `pending_review` /
  absent = pending). Reuses the existing mental model and review workflow —
  one-line human decision, recorded in git.
- **Default-deny, twice over:** a missing `geometry_status` is
  `pending_review` even when `status = "approved"`. Publishing the unmatched
  artifact requires **both** approved. `LicenseRegistry.decision()`
  (`licenses.py:77-119`) grows a `geometry_approved: bool` on
  `LicenseDecision`; the publisher branches on it per artifact.
- Why a separate field rather than a permission list (`artifacts = [...]`):
  two booleans with a strict ordering (geometry ⇒ ids) is the actual decision
  space; a list invites invalid states (`["unmatched"]` without bridge) and
  more parsing. If a third artifact class ever appears, migrate then.
- Review guidance for the human (wording per §7 Q3, decided 2026-07-09):
  `geometry_status = "approved"` requires the source terms to permit
  **redistribution of the data itself** (not merely "use"), with attribution
  obligations we can satisfy in-artifact. The checklist additionally **screens
  for share-alike / derivative-database clauses and assesses
  ODbL-compatibility** — ODbL is the compatibility bar, so
  roughly-ODbL-compatible share-alike terms (e.g. ODbL itself, as with
  `tn_tunis_ml_roads`) are acceptable rather than hard-rejected; the finding
  is recorded in the registry note and surfaced in `unmatched_meta.json`.
  Because §7 Q2 ships `source_tags` verbatim, the review also includes a quick
  PII / attribute-sensitivity glance at the source columns. The ~10
  PD / PD-equivalent datasets (5 US-PD federal/Montana, Boston PDDL ×3,
  Seattle — whose registry note already records geometry-redistribution
  clearance — and Amsterdam CC0) trivially qualify. "Open data licence"
  portals (Singapore LTA, OGL, etc.) need actual reading — exactly what the
  registry's `note`/`likely_license` hints exist for.
- **Scope coherence:** this gate is not unmatched-specific. `targets/`
  (#370) already redistributes full geometry + `source_tags` for every
  `status = "approved"` dataset — a larger exposure than the unmatched
  complement. `geometry_status` therefore gates **every geometry-bearing
  artifact** (`targets/` and `unmatched/`) in the same
  `LicenseRegistry.decision()` change; see "Layout cohesion" (§4), finding 1.

### Attribution ships inside the artifact

Per published dataset dir, alongside the parquet:

- `ATTRIBUTION.txt` — human-readable: source attribution line + the Overture
  attribution block from `[overture]` (nearest-GERS columns are derived from
  Overture, so the artifact is a derivative of both — same reasoning as
  PUBLISHING.md "Licensing & attribution").
- `unmatched_meta.json` — machine-readable license metadata (see below).

### The OSM-import reality check

OSM import requires (a) license compatibility with **ODbL** (waiver/PD/
explicitly compatible), and (b) the **community process**: the Import
Guidelines, wiki documentation, imports@ mailing-list review, and DWG
oversight. We are not, and must not claim to be, any of that.

**Decision: the artifact self-serves the vetting but never claims readiness.**
`unmatched_meta.json` carries:

```jsonc
{
  "dataset": "us_montana_missoula",
  "release": "2026-01-21.0",
  "source_license": "US-PD",              // SPDX-ish, from the registry
  "source_license_share_alike": false,     // from the §7 Q3 geometry-review checklist
  "source_attribution": "Montana State Library (MSDI Transportation Framework)",
  "source_url": "https://gisservicemt.gov/...",
  "geometry_redistribution": "approved",   // what OUR gate verified
  "osm_odbl_compatibility": "unverified",  // NEVER auto-set; human may set "waiver_on_file" etc.
  "import_ready": false,                   // constant. Always false. Not our call.
  "note": "License-cleared for redistribution by this project. OSM import requires independent ODbL-compatibility verification and the OSM community import process (wiki + imports list + DWG). This artifact is source material, not an import."
}
```

The credibility page and README section for the artifact repeat that sentence
verbatim. This keeps us squarely in "staging raw material" territory — the
settled scope — and off the former team's lane (no conflation into the map by
us).

---

## 4. Pipeline integration

### Generation: fix at the shared-pipeline seam, not a new command

The enriched export lands in `optimize_and_export()`
(`src/crosswalk/pipeline/runner.py:997-1210`), replacing today's
`generate_unmatched_report()` call at `runner.py:1177-1184` with an enriched
`export_unmatched()` (new module `src/crosswalk/resolution/unmatched.py`;
`bridge.py`'s version becomes a thin deprecated alias or is removed). It has
everything in scope already: the target GDF (attributes + geometry), the
reference GDF (for `sjoin_nearest`), the full scored `results` list (for
`n_candidates` / `best_candidate_*`), and the optimized bridge rows (for tier
assignment). Both `crosswalk stitch` and the factory flow through this seam
([FACTORY.md](../FACTORY.md): "It does not fork pipeline logic"), so **no new
factory stage and no separate `factory export-unmatched` command for
generation** — the same reasoning as the backfill rule in CLAUDE.md: one code
path, no skew. `factory reoptimize` regenerates the file from the scored cache
in ~2 s per dataset, which is also the cheap backfill path for existing
factory outputs.

Cost check: the only new computation is one `sjoin_nearest` (target-unmatched ×
reference) and a UTM length pass — trivially cheap next to feature scoring
(~84% of wall time per FACTORY.md).

### Publishing: extend `assemble_staging`, same tree

```
r2://<bucket>/
  bridges/release=<R>/dataset=<name>/     (unchanged)
  unmatched/
    release=<R>/                          # same immutability rules
      index.json                          # per-release index for this artifact class
      checksums.txt
      dataset=<name>/
        unmatched.parquet                 # GeoParquet, schema §2
        unmatched.geojson.gz              # mapper on-ramp, generated at publish
        unmatched_meta.json               # machine-readable license metadata (§3)
        ATTRIBUTION.txt
```

- A **sibling top-level prefix (`unmatched/`)**, not files inside
  `bridges/…/dataset=`: releases under `bridges/` are already published and
  immutable (`publish.py` skips existing releases; PUBLISHING.md "Immutable
  release paths"), so adding files into them is prohibited by our own rules.
  A separate prefix also keeps the two license gates physically separate — a
  dataset can be bridge-published but unmatched-withheld.
- **No `all_unmatched.parquet`**: unlike the bridge reverse-lookup ("who
  references this GERS id?"), there is no cross-dataset query that needs one
  object, and geometry makes concatenation heavy. Per-dataset files only.
- `publish.py::assemble_staging` grows an unmatched pass per release: for each
  dataset with `approved + geometry_approved`, copy the parquet verbatim,
  write geojson.gz + meta + attribution, checksum everything into the
  release's `checksums.txt`, and record an `unmatched` block (files, sha256,
  row counts per reason tier) in the per-dataset entry of `index.json`
  (`build_index`, `publish.py:234-251`). Datasets that are bridge-approved but
  geometry-pending appear as `unmatched: {status: "excluded", reason: …}` —
  the same deliberate-exclusion honesty as today's pending list.
- **Pages browser (`site/`):** Stats page gains an "unmatched" column
  (count + % of targets, from `index.json` — no DuckDB needed) and download
  links (parquet + geojson.gz). Browse page: reason-tier filter + table via
  DuckDB-WASM, map preview deferred to Phase 2 (bbox query → GeoJSON →
  Leaflet, reusing the existing DuckDB-WASM plumbing in `browse.js`).
- **Credibility page:** per-dataset unmatched counts + the §3 disclaimer
  paragraph in the licensing block.

### Layout cohesion — align the artifact family BEFORE Phase 1 ships (review 2026-07-09)

`unmatched/` will be the **third artifact family** on the bucket. The first
two (`bridges/`, then `targets/` via #370) grew up two days apart and have
already drifted; because published paths are immutable and the sync is
no-delete, every misalignment becomes permanent the moment
`unmatched/release=…` first syncs. Findings, ranked by what they cost *after*
that point:

1. **Geometry-gate asymmetry (highest cost — it is the legal surface).**
   `targets/` redistributes full geometry + verbatim `source_tags` gated only
   by `status` + quality hold; this design as first drafted put
   `geometry_status` on `unmatched/` alone. The cost is not hypothetical:
   `co_bogota_bike_network`'s target snapshot (`snapshot=2026-07-05`) synced
   to the public bucket on 2026-07-08 — one day before #380 taught the
   targets path about quality holds — and the no-delete design means #380
   only *delists* it from `targets/index.json` at the next publish; the
   parquet itself stays live until deleted by hand. The same shape of mistake
   awaits geometry licensing. **Fix (Phase 1, blocking):** `geometry_status`
   gates every geometry-bearing artifact (`targets/` AND `unmatched/`) in one
   `LicenseRegistry` change; before the first `unmatched/` sync, audit the
   already-live `targets/` objects against the new gate and decide
   delist-vs-delete per dataset (Brad's call — deletion breaks the
   immutability promise, delisting leaves geometry live).
2. **Index-shape divergence.** The root `index.json` is written by the
   *bridges* assembler but is named/positioned as THE bucket index — the
   Pages site reads only it, so `targets/` is invisible to the browser.
   `targets/index.json` diverges from the bridges indexes: no
   `schema_version`, no `generated_at`, no excluded-dataset records (the
   deliberate-exclusion honesty the bridges index already has), no checksums
   — and `generated_from` leaks a local filesystem path into a public
   object. **Fix:** bump the root index to `schema_version: 2` with an
   `artifacts` block pointing at per-family indexes (`bridges`, `targets`,
   `unmatched`); every per-family index carries
   `schema_version`/`generated_at`/exclusion records; the `unmatched/`
   per-release index (tree above) follows the bridges shape from day one;
   drop or relativize `generated_from`.
3. **Metadata-sidecar drift.** Three conventions in flight: bridges publish
   `manifest.json` (verbatim factory provenance, JSON); targets publish
   `meta.yaml` (normalized provenance, YAML); this design proposes
   `unmatched_meta.json` + `ATTRIBUTION.txt`. **Fix:** JSON +
   `ATTRIBUTION.txt` is the convention for new artifacts; retrofit `targets/`
   additively (new snapshots gain `ATTRIBUTION.txt`; existing `meta.yaml`
   stays — snapshots are immutable).
4. **Integrity-manifest gap.** Bridges ship per-release `checksums.txt` +
   per-file sha256 in the index; targets ship neither. `unmatched/` must ship
   both (already specified above); add checksums to future target snapshots
   while touching that code.
5. **Sync-guard duplication.** `publish_sync.py` already holds two
   near-identical immutability guards (`SyncPlan`, release-keyed;
   `TargetSyncPlan`, dataset+snapshot-keyed); `unmatched/` needs a third
   (release-keyed under a different prefix). **Fix:** generalize to one
   prefix-keyed plan *before* adding the third copy, so the immutability unit
   is data, not copy-pasted code.
6. **Storefront-copy drift.** README ("deliberately *not* a geometry-import…
   project"), `site/browse.html` ("the bridge tables are ID-only (no
   geometry)…"), and PUBLISHING.md's bucket layout (bridges-only — `targets/`
   is never mentioned) all predate #370 and become doubly wrong once
   `unmatched/` ships geometry. `datasets/licenses.toml`'s header still says
   the registry gates "the R2 bridge tables". **Fix:** PUBLISHING.md owns the
   full three-prefix bucket layout; one "what we publish" paragraph reused by
   README + credibility page; the registry header says it gates all published
   artifacts.
7. **Staging split — record as deliberate.** Bridges stage to
   `data/publish_staging`, targets to `data/publish_staging_targets`; this
   design puts `unmatched/` in the bridges staging tree. That is the right
   grouping — release-keyed artifacts (`bridges/`, `unmatched/`) share one
   staging root + one sync pass; snapshot-keyed (`targets/`) keeps its own —
   but write it down in PUBLISHING.md so it reads as architecture, not
   accident.

One asymmetry is **correct and should stay**: bridges/unmatched key
`release=` first (Overture-release cadence, whole-release immutability);
targets key `dataset=`/`snapshot=` (per-dataset fetch cadence, snapshot
immutability). Record it as deliberate rather than "aligning" it.

### QA feedback loop (`/qa` → artifact)

The QA decision stores (`integration_qa/decision_store.py`) already record
human judgments keyed by `original_id` + `dataset_id` with exactly the needed
vocabulary. Mapping into the artifact:

| QA `decision`/`reason` (orphan store, `decision_store.py:19-34`) | `qa_status` |
|---|---|
| `correct` / `legitimate_new` | `confirmed_missing` |
| `incorrect` / `data_error` | `bad_source_geometry` |
| `incorrect` or `correct` / `out_of_scope` | `out_of_scope` |
| no row | `unreviewed` |

Join at **publish/staging time** (not factory-run time): QA decisions are
git-tracked CSVs that accrue continuously, while factory outputs are
immutable-ish run products; joining at staging means a re-publish (`--force`)
or the next release picks up new reviews with zero pipeline reruns. The join
key is `original_id == local_id` (verify during implementation — the
integration pipeline derives `original_id` from target ids; if any dataset
predates the H3-suffix migration, run `scripts/migrate_to_h3_ids.py`
equivalence mapping). `confirmed_missing` rows are the artifact's premium
tier — surfaced first in the browser and callable out in per-dataset stats.

---

## 5. Overture handoff angle

What makes this consumable as a future Overture ingestion-candidate feed:

1. **Already-Overture-shaped attributes.** Ingestion normalizes targets to
   Overture transportation vocabulary (`class`, `subtype`, `names.primary` —
   verified in `data/raw/*_v1.0.parquet`). An Overture engineer reads this
   artifact with zero schema translation.
2. **Stable ids + explicit release keying.** `local_id` is H3-suffixed and
   deterministic; every artifact row carries `release`. "Unmatched against
   2026-01-21.0" is a falsifiable, re-checkable claim.
3. **Transition records (Phase 3).** Extend `factory delta`
   (`src/crosswalk/factory/delta.py` — already computes
   same/changed/lost/gained on `match_decision == "match"`) with an
   unmatched-side ledger per dataset ×release-pair:
   - `resolved`: unmatched in R1, matched in R2 → record `resolved_by_gers`
     (the tombstone: "Overture now covers this; stop showing it to mappers").
     This is also the **feedback metric** — if OSM mappers add missing
     sidewalks and Overture's next OSM-derived release picks them up, resolved
     counts measure real-world impact of the artifact.
   - `regressed`: matched in R1, unmatched in R2 (GERS churn / Overture data
     loss — directly useful to Overture QA).
   - `persistent`: unmatched in both (the durable gap backlog).
   Published as `unmatched_delta.json` + `.md` next to the artifact, same
   pattern as the existing delta release-notes.
4. **Quality annotations Overture can trust-but-verify:** calibrated
   `best_candidate_confidence`, `nearest_gers_distance_m`, human `qa_status`,
   and the manifest provenance chain (model fingerprint, `feature_version`,
   buffer) published verbatim — the same credibility mechanics as the bridges.
5. **Per-release immutability** means Overture can diff feeds across releases
   with no phantom churn (the optimizer determinism guarantee, FACTORY.md
   "Determinism note", extends to the unmatched partition since it is a pure
   function of the same inputs).

Explicitly out of scope: merging, geometry harmonization, or producing
Overture-format change files. The feed is *evidence*, not a patch.

---

## 6. Sizing + phasing

### Real numbers (factory outputs on this machine, release 2026-01-21.0)

| dataset | targets (manifest) | unmatched file rows | manifest `n_unmatched`* | `<5 m` (dropped) | median len | post-filter rows |
|---|---|---|---|---|---|---|
| sg_singapore_footpaths | 110,811 | 57,582 | 52,203 | 11,570 | 12 m | ~46,000 |
| co_bogota_bike_network | 6,191 | 1,479 | 902 | 39 | 52 m | ~1,440 |
| sg_singapore_roads | 15,319 | 1,420 | 969 | 0 | 185 m | 1,420 |
| us_montana_missoula | 9,421 | 419 | 103 | 0 | 210 m | 419 |
| us_usfs_flathead | 414 | 98 | 70 | 0 | 248 m | 98 |

\* file rows > manifest `n_unmatched` is the accounting quirk from §1: the
file includes not-matched review-tier rows, the manifest subtracts the whole
review set. The legacy path (not yet in factory) adds us_boston_streets
(small unmatched set) and us_seattle_sidewalks (review-heavy: 1,314 review
edges of 30,624 bridge rows).

Takeaways: (a) most road datasets have small, high-signal unmatched sets
(~100–1,500 rows — very publishable); (b) sidewalk/footpath datasets are the
headline use case *and* the noise risk — Singapore footpaths is ~46k rows
post-filter, ~50% of the inventory, i.e. genuinely large coverage gaps plus
segmentation slivers, which is exactly why the sliver filter and reason tiers
exist; (c) file sizes are trivial (worst case ~10–15 MB GeoParquet, a few MB
gzipped GeoJSON) — no partitioning concerns.

### Phasing

**Phase 1 — minimal shippable artifact (~2–3 days)**
1. `export_unmatched()` in `resolution/unmatched.py`: fix the id/attribute bug,
   reason tiers from scored results + bridge rows, `length_m` + `<5 m` drop +
   `is_short`, `nearest_gers_*`, `best_candidate_*`, `source_tags` verbatim
   (§7 Q2), provenance columns. Wire
   into `optimize_and_export()` (`pipeline/runner.py:1177`) AND into the
   second `generate_unmatched_report` call site at the zero-candidate
   early-return (`pipeline/runner.py:1050`), which must emit the same schema
   (every row `no_candidate` there).
2. `geometry_status` in `licenses.toml` + `LicenseRegistry` (default-deny),
   gating **both** geometry-bearing artifacts — `unmatched/` and the existing
   `targets/` path (§4 "Layout cohesion", finding 1). Set `approved` for the
   PD / PD-equivalent entries as each passes the §3 checklist (5 US-PD
   federal/Montana; Boston PDDL ×3; Seattle — geometry clearance already
   recorded in its registry note; Amsterdam CC0). The Singapore datasets stay
   default-deny (license itself still `pending_review`); Bogotá bike network
   is license-approved but quality-held (#338) — and its pre-#380 target
   snapshot is already live on the bucket, see the cohesion audit item.
3. `assemble_staging()` unmatched pass: `unmatched/release=…` tree, geojson.gz,
   `unmatched_meta.json` (with the non-import disclaimer), `ATTRIBUTION.txt`,
   index.json + checksums extensions; credibility page counts + disclaimer.
4. Stats page column + download links in `site/`.
5. Tests: unit (tier assignment, sliver filter, gate default-deny for
   geometry), publish-tree golden test alongside the existing publish tests.
   Regenerate factory outputs via `factory reoptimize --all` (cache-valid,
   ~2 s/dataset).
6. Layout-cohesion pre-work (§4, blocking the first `unmatched/` sync):
   geometry gate extended to `targets/` + audit of already-live target
   snapshots; generalized prefix-keyed sync immutability guard; root-index
   `artifacts` pointers (`schema_version: 2`).

**Phase 2 — QA feedback + browsing (~2–3 days)**
1. `qa_status` join at staging time from the QA decision stores; surface
   `confirmed_missing` counts in index.json + pages.
2. Browse-page map preview: DuckDB-WASM bbox query on the GeoParquet →
   GeoJSON → Leaflet, reason-tier coloring; row table with tier filter.
3. Optional `screen_flags` when screen context exists for a dataset.
4. `/qa` route: add a mode that reads the factory `unmatched.parquet` directly
   (today it reviews integration-pipeline outputs only), so review can happen
   without running `analyze integrate`.

**Phase 3 — Overture-facing extras (~2–4 days)**
1. `factory delta --unmatched`: resolved/regressed/persistent ledger +
   published `unmatched_delta.{json,md}`; resolved tombstones carry
   `resolved_by_gers`.
2. Cross-release "resolved" metric on the credibility page (the impact story).
3. PMTiles evaluation only if the Phase 2 preview is inadequate at real sizes.
4. Short outreach doc (OSM wiki-style project page draft + Overture-facing
   README section) — written honestly per §3.

---

## 7. Open questions for Brad

**All five decided by Brad, 2026-07-09.** Recorded here with rationale; the
decisions are applied through §§1–6 above.

1. **Review-tier rows: in or out of the public artifact?** —
   **DECIDED: in, reason-coded**, as the design proposed (§1). The artifact
   partitions the whole target set; consumers default-filter on `reason`; the
   headline "unmatched" count stays `no_candidate + below_threshold`; and the
   duplicate-mapping risk is mitigated by the tier docs plus each `review`
   row carrying its `best_candidate_gers` (the likely existing counterpart).
2. **Attribute breadth.** — **DECIDED: ship raw source attributes verbatim**
   for `geometry_status = "approved"` datasets, not just the normalized
   name/class/subtype trio. Column name: **`source_tags`** — exactly
   Overture's schema property (`OvertureMaps/schema`,
   `schema/base/defs.yaml` `sourceTags`: *"Any attributes/tags from the
   original source data that should be passed through"*; docs: *"key-value
   pairs imported directly from the source data without change"*,
   <https://docs.overturemaps.org/schema/reference/base/land/>), and already
   the internal column name at ingestion (`fetch/target.py`) and in the
   published `targets/` snapshots. `original_source_tags` rejected — it
   matches neither Overture nor the existing artifacts. Representation +
   gating applied in §2/§3.
3. **Gate strictness for `geometry_status`.** — **DECIDED: screen, don't
   hard-reject.** The review checklist screens for share-alike /
   derivative-database clauses and **assesses ODbL-compatibility** — ODbL is
   the compatibility bar, so roughly-ODbL-compatible licenses (ODbL itself,
   PD/CC0, attribution-only) are acceptable; the finding is recorded in the
   registry note and surfaced in `unmatched_meta.json` rather than blocking
   publication. Checklist wording updated in §3.
4. **Naming.** — **DECIDED: `unmatched/`** (pipeline-truthful) as the public
   prefix + product name. Cohesion check (2026-07-09) found no
   evidence-backed objection: it matches the internal vocabulary
   (`unmatched.parquet`, manifest `n_unmatched`) and sits naturally beside
   the sibling prefixes (`bridges/`, `targets/`). The one wrinkle —
   `review`-tier rows are not literally unmatched — is already handled by the
   §1 accounting (headline count excludes `review`). The rename window closes
   permanently at first publish (immutable paths).
5. **Overture outreach timing.** — **DECIDED: publish quietly.** No Overture
   community-forum post with Phase 1 — too many questions still outstanding;
   let the artifact accumulate releases first. The Phase 3 outreach doc is
   unaffected.
