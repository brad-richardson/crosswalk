# Unmatched-Segment Export: Design

**Date:** 2026-07-06
**Status:** Proposed (design only — no implementation in this PR)
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
  (`src/crosswalk/factory/manifest.py:218`). The publisher currently copies
  only `bridge.parquet` + `manifest.json`
  (`src/crosswalk/factory/publish.py:357-364`).
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
  (`src/crosswalk/factory/licenses.py:77-119`). 8 datasets are `approved`
  (all US public domain); the rest `pending_review`.
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

**Recommendation: ship all three tiers in one artifact**, because the split is
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

No `source_tags` / raw source attributes in v1 — see Open Questions.

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
- Review guidance for the human: `geometry_status = "approved"` requires the
  source terms to permit **redistribution of the data itself** (not merely
  "use"), with attribution obligations we can satisfy in-artifact. The 8
  current US-PD datasets trivially qualify. "Open data licence" portals
  (Singapore LTA, OGL, etc.) need actual reading — exactly what the registry's
  `note`/`likely_license` hints exist for.

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
   `is_short`, `nearest_gers_*`, `best_candidate_*`, provenance columns. Wire
   into `optimize_and_export()` (`pipeline/runner.py:1177`); manifest counts
   redefined from the artifact (per-tier counts added to `manifest.json`).
2. `geometry_status` in `licenses.toml` + `LicenseRegistry` (default-deny);
   set `approved` for the 5 factory US-PD datasets.
3. `assemble_staging()` unmatched pass: `unmatched/release=…` tree, geojson.gz,
   `unmatched_meta.json` (with the non-import disclaimer), `ATTRIBUTION.txt`,
   index.json + checksums extensions; credibility page counts + disclaimer.
4. Stats page column + download links in `site/`.
5. Tests: unit (tier assignment, sliver filter, gate default-deny for
   geometry), publish-tree golden test alongside the existing publish tests.
   Regenerate factory outputs via `factory reoptimize --all` (cache-valid,
   ~2 s/dataset).

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

1. **Review-tier rows: in or out of the public artifact?** Design says *in,
   reason-coded* (completeness + the browser default-filters them). The
   counterargument: a mapper who ignores the docs and maps `review` rows
   creates OSM duplicates. If that risk feels real, ship `no_candidate` +
   `below_threshold` only and keep `review` in the internal factory file.
2. **Attribute breadth.** v1 ships `name`/`class`/`subtype` only. Municipal
   extras (surface, width, install year — often present in `source_tags`)
   are high-value for mappers *and* higher license/PII exposure, and vocab
   varies wildly per dataset. Ship `source_tags` verbatim for
   `geometry_status=approved` datasets, or hold at the normalized trio?
3. **Gate strictness for `geometry_status`.** Is "source terms permit
   redistribution with attribution" sufficient, or do you also want an
   explicit check for share-alike/derivative-database clauses (which would
   complicate the *consumer's* ODbL story even when redistribution by us is
   fine)? This changes the review checklist wording, not the mechanism.
4. **Naming.** `unmatched/` (pipeline-truthful) vs `gaps/` (audience-truthful)
   as the public prefix + product name. Design assumes `unmatched/` to match
   internal vocabulary; a rename is cheap only before first publish.
5. **Overture outreach timing.** Publish quietly and let the artifact
   accumulate releases first, or pair Phase 1 with a post to the Overture
   community forum? (Affects nothing technical; Phase 3 outreach doc either
   way.)
