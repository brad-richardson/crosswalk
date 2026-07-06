# Dataset registry format: one JSON recipe per city

**Status:** proposal (Phase 0 shipped in this PR — schema + 2 inert examples + CI check)
**Date:** 2026-07-06
**Question (Brad):** "Currently I just have a gigantic yaml with all datasets — maybe
separate git-tracked JSON like some public transit aggregators? How does
OpenAddresses provide that info?"

---

## Recommendation

**Adopt one per-dataset JSON "recipe" file — `datasets/<name>.json` — that merges
the declared fetch config (today `datasets/<name>.yaml`) with the license /
publication metadata (today one entry in `datasets/licenses.toml`), validated
against a checked-in JSON Schema (`datasets/schema/dataset.schema.json`) by a
fast pytest.** Machine-generated run state (`last_fetch`, `quality_fingerprint`)
moves *out* of the recipe into tool-owned state files. `licenses.toml` keeps
working throughout the migration via a dual-read loader; nothing breaks
mid-flight.

This is the OpenAddresses model (one source file = one contribution = one PR,
jsonschema-validated), with the license block upgraded to a
Transitland-DMFR-style structured object (SPDX id + tri-state capability
flags) instead of freeform strings.

### Why not the alternatives

- **(b) Keep two layers, split `licenses.toml` into per-dataset files.** Fixes
  the "gigantic file" symptom (merge conflicts between panel PRs, 528 lines and
  growing ~12/dataset) but leaves a new city as *two* files whose fields
  overlap (`source_url` in the toml duplicates `source.url` in the yaml;
  `display_name` appears in both, and can drift). The North Star explicitly
  wants "an OA-style contribution recipe format so others can add cities" — OA's
  core lesson is that the recipe and its license ride in the *same* reviewable
  file, so the license question is asked at contribution time, not later.
- **(c) Status quo + generated views.** Zero migration cost, but the
  contribution unit stays "edit a 528-line TOML plus add a YAML", CI still can't
  catch a typoed field (TOML parses fine; default-deny silently excludes), and
  the panel keeps writing long prose into single `note` strings because there's
  nowhere structured to put evidence.

### Why merging is safe *here* (the machine-write concern)

Today's `datasets/*.yaml` files are rewritten by tooling: `crosswalk fetch`
updates `last_fetch` and `crosswalk quality fingerprint` writes ~30 metrics
(`src/crosswalk/datasets/schema.py::save_dataset_config`). Naively merging the
license block into that file would put human-reviewed legal decisions in a file
that machines churn — noisy diffs on the one file where diffs are
legally meaningful. The fix is the OA split: **the recipe holds declarations,
the machine holds results.** OA source files declare `conform`; the run results
live in their batch system, which pins each job to a commit-SHA'd raw URL of
the source JSON (`batch.openaddresses.io/api/job/<id>` returns
`"source": "https://raw.githubusercontent.com/openaddresses/openaddresses/<sha>/sources/..."`).
Crosswalk's analogs (`last_fetch`, `quality_fingerprint`) move to
`datasets/state/<name>.yaml` (tool-owned, still git-tracked for provenance) in
Phase 2. Tooling may still touch the recipe for *declared* config it discovers
(`classification` from `discover-classes`), which is human-reviewed before
commit — same as OA bots editing source files.

---

## What OpenAddresses and DMFR actually do (read 2026-07-06)

### OpenAddresses ([github.com/openaddresses/openaddresses](https://github.com/openaddresses/openaddresses))

- **One JSON file per source** in a geographic tree:
  `sources/{ISO-3166-1 country}/{ISO-3166-2 region}/{source}.json`, e.g.
  [`sources/us/wa/city_of_seattle.json`](https://raw.githubusercontent.com/openaddresses/openaddresses/master/sources/us/wa/city_of_seattle.json).
  Contribution = one file, one PR ([CONTRIBUTING.md](https://github.com/openaddresses/openaddresses/blob/master/CONTRIBUTING.md):
  "Comfortable with JSON? Feel free to submit a pull request with the data").
- **Schema v2** ([`schema/source_schema_v2.json`](https://raw.githubusercontent.com/openaddresses/openaddresses/master/schema/source_schema_v2.json),
  draft-07, `additionalProperties: false`): required `schema` + `coverage`;
  data lives in `layers.{addresses|parcels|buildings|centerlines}[]`. The
  version marker is **in-band** (`"schema": 2`); v1→v2 migrated files in place
  and parked the old schema at `schema/retired/`.
- **License is a small structured object per layer entry**
  ([`schema/util/license.json`](https://raw.githubusercontent.com/openaddresses/openaddresses/master/schema/util/license.json)):
  `url`, `text`, `attribution` (bool: attribution required?),
  `"attribution name"` (the text to display — yes, with a literal space),
  `share-alike` (bool). Real example, Austria countrywide
  ([`sources/at/countrywide.json`](https://raw.githubusercontent.com/openaddresses/openaddresses/master/sources/at/countrywide.json)):

  ```json
  "license": {
      "url": "https://creativecommons.org/licenses/by/4.0/",
      "text": "CC BY 4.0 DEED",
      "attribution": true,
      "attribution name": "© Österreichisches Adressregister, data of the record date 01.04.2025"
  }
  ```

  CONTRIBUTING.md also documents a `presumed` flag ("*true* if an OpenAddresses
  community member *interpreted* the license to derive the share-alike or
  attribution booleans") — their version of our panel-vs-human distinction.
- **CI:** [pre-commit.ci](https://raw.githubusercontent.com/openaddresses/openaddresses/master/.pre-commit-config.yaml)
  runs `check-json` + `python-jsonschema/check-jsonschema --schemafile
  schema/source_schema_v2.json` on every PR touching `sources/`; repo tests
  (`test/sources_validator.test.js`, ajv) validate every source file; an
  external batch app actually *runs* changed sources per PR.
- **Downstream:** data is "not relicensed from the original sources… The
  OpenAddresses team does its best to summarize the source licenses in the
  source JSON" ([README](https://github.com/openaddresses/openaddresses/blob/master/README.md));
  consumers read attribution/share-alike straight from the registry file.

### Transitland DMFR ([spec](https://github.com/transitland/distributed-mobility-feed-registry), [atlas](https://github.com/transitland/transitland-atlas))

- JSON files validated against a **versioned JSON Schema**
  ([`dmfr.schema-v0.6.0.json`](https://raw.githubusercontent.com/transitland/distributed-mobility-feed-registry/master/json-schema/dmfr.schema-v0.6.0.json));
  each file names its spec version via `"$schema": "https://dmfr.transit.land/json-schema/dmfr.schema-v0.6.0.json"`.
  Files are per *source domain* (`511.org.dmfr.json` holds 34 feeds), not per
  feed — an aggregation crosswalk does **not** need.
- **The license object is the model to copy** (`definitions.license_description`):
  `spdx_identifier` (validated against the full SPDX enum), `url` (custom
  terms), and tri-state `["yes","no","unknown"]` capability flags —
  `use_without_attribution`, `create_derived_product`, `redistribution_allowed`,
  `commercial_use_allowed`, `share_alike_optional` — plus `attribution_text` /
  `attribution_instructions`. Real feed
  ([`feeds/511.org.dmfr.json`](https://raw.githubusercontent.com/transitland/transitland-atlas/main/feeds/511.org.dmfr.json)):

  ```json
  "license": {
      "url": "http://www.bart.gov/schedules/developers/developer-license-agreement",
      "use_without_attribution": "yes",
      "create_derived_product": "unknown"
  }
  ```

  `"unknown"` as a first-class value is exactly right for license research:
  it distinguishes "we checked and it's silent" from "nobody looked".
- GitHub Actions validates atlas PRs against the schema; moderators review.

### Also glanced at

- **Mobility Database catalogs** ([repo](https://github.com/MobilityData/mobility-database-catalogs)):
  one JSON per source under `catalogs/sources/gtfs/…`, draft-07 schemas in
  `schemas/`, but license is **just a URL string** — the anti-example; it
  pushes every consumer to re-do the legal reading that OA/DMFR encode once.
- **osm-community-index** ([repo](https://github.com/osmlab/osm-community-index)):
  one JSON per resource under `resources/<region>/`, `schema/resource.json`,
  jsonschema validation in CI — the same per-item-JSON + schema-in-repo + CI
  pattern, third independent confirmation it scales socially.

### Adopted / rejected

| From | Adopted | Rejected |
|---|---|---|
| OA | one file = one source = one PR; in-band `schema_version`; strict `additionalProperties: false`; jsonschema in CI; license rides in the recipe; `presumed` idea → our `review` block | **geographic directory tree** — crosswalk's dataset key (`us_boston_streets`) already encodes country and is used verbatim as a filename/partition key across `labels/`, `bridges/dataset=…`, mbench; `us/ma/boston_streets.json` would break the 1:1 name↔file mapping for zero gain at ~50 files. Also OA's space-in-key `"attribution name"` and boolean-only flags |
| DMFR | structured license object: `spdx` + capability flags + attribution text; **tri-state yes/no/unknown**; SPDX where possible, `url`+`name` for custom terms | per-*domain* files aggregating many feeds; `$schema`-URL-carried spec version (we pin with an in-band `schema_version` + repo-local schema — no server to host versioned schemas); negative-polarity flag names (`use_without_attribution`, `share_alike_optional`) — ours are positive (`attribution_required`, `share_alike_required`) to keep review reading natural |
| MobilityData | — | license-as-bare-URL |

---

## The format (schema_version 1)

Schema: [`datasets/schema/dataset.schema.json`](../../datasets/schema/dataset.schema.json)
(JSON Schema draft 2020-12, `additionalProperties: false` throughout).
Examples shipped in this PR: [`datasets/us_boston_streets.json`](../../datasets/us_boston_streets.json),
[`datasets/ch_geneva_pedestrian_network.json`](../../datasets/ch_geneva_pedestrian_network.json)
(nontrivial custom license: SITG terms, no SPDX id, attribution required).

Top-level blocks:

| Block | Replaces | Owner |
|---|---|---|
| `schema_version`, `name`, `display_name`, `type`, `description` | yaml identity + toml `display_name` (dedup) | contributor |
| `source` | yaml `source` + toml `source_url` (dedup) | contributor |
| `fetch`, `matching`, `classification` | yaml (the OA `conform` analog) | contributor / discover tooling |
| `license` | the toml entry's decision + facts | **human reviewer** (gates), panel (facts) |
| `review` | toml `panel_*` + `note` | AI research panel + human |
| *(gone from recipe)* `last_fetch`, `quality_fingerprint` | yaml machine state | fetch/quality tooling → `datasets/state/<name>.yaml` (Phase 2) |

The `license` block, mapped from today's toml fields:

```jsonc
"license": {
  "status": "pending_review",          // toml `status` — the ID-bridge gate, human-only
  "geometry_status": "pending_review", // NEW (#321) — geometry-artifact gate, human-only
  "spdx": "PDDL-1.0",                  // when a real SPDX id exists; null for custom terms
  "name": "…full license name…",       // toml `likely_license`
  "url": "…",                          // toml `license_url`
  "attribution": "…",                  // toml `attribution` (required when approved)
  "proposed_attribution": "…",         // toml `proposed_attribution`
  "capabilities": {                    // DMFR-style tri-state facts, panel-fillable
    "redistribution_allowed": "yes|no|unknown",
    "derivatives_allowed":    "yes|no|unknown",
    "commercial_use_allowed": "yes|no|unknown",
    "attribution_required":   "yes|no|unknown",
    "share_alike_required":   "yes|no|unknown"
  }
}
```

Design points:

- **Decisions vs evidence.** `status`/`geometry_status` stay explicit
  human-flipped enums exactly as `factory/licenses.py` and the #321 design
  define them (default-deny, `approved`/`pending_review`, missing =
  pending). The `capabilities` flags are *evidence supporting* the decision,
  never consumed by the gate — the publisher must keep never guessing. But they
  map directly onto what a `geometry_status` review has to establish
  (`redistribution_allowed` + `derivatives_allowed` + satisfiable attribution),
  so the panel can pre-fill them and the human approves against a checklist
  instead of re-reading prose notes.
- **Conditional validation:** `status: approved` ⇒ `attribution` + `url` +
  (`spdx` or `name`) required, enforced in-schema (`allOf`/`if`) — the invalid
  state that today only surfaces at publish time as a silent exclusion
  (`licenses.py` "approved but missing license/attribution") becomes a PR-time
  schema failure.
- **`review` block** gives the panel structured landing spots:
  `panel_recommendation` / `panel_confidence` / `panel_reviewed_at` (exactly
  today's toml fields), plus `evidence: [{url, quote, fetched_at}]` — the
  burndown dossiers already produce verbatim quotes with URLs; today they get
  squashed into one `note` string. `note` remains for prose.
- **Strictness:** `additionalProperties: false` everywhere (OA-style) so a
  typoed field name fails CI instead of silently default-denying. New fields
  require a schema PR — that is a feature: the schema is the contributor
  documentation.
- **CI:** `tests/unit/test_dataset_registry_schema.py` validates every
  `datasets/*.json` against the schema (jsonschema, dev extra), checks
  `name` == filename stem, and mirrors the approved⇒attribution rule. Fast,
  local, no network — the OA pre-commit.ci equivalent. A pre-commit
  `check-jsonschema` hook can be added later; the pytest is the floor.

## Migration plan (no consumer breaks mid-flight)

**Phase 0 — this PR.** Schema + 2 example recipes + CI test. The `.json` files
are **inert**: `list_dataset_configs()` globs `*.yaml`
(`src/crosswalk/datasets/config.py:317`) and `LicenseRegistry.load()` reads only
`licenses.toml` (`src/crosswalk/factory/licenses.py:58-70`). No consumer change,
no behavior change. (Note: `datasets/licenses.toml` entries for the two examples
remain authoritative; PR #326's approvals land in the toml as usual.)

**Phase 1 — dual-read loaders.**
- `LicenseRegistry.load()` additionally scans `datasets/*.json`; a recipe's
  `license`+`review` blocks are adapted into the same entry dict the toml
  produces today (`decision()` is already `entry.get`-based, so the adapter is
  ~20 lines; `LicenseDecision` unchanged, plus the `geometry_approved` bool
  from the #321 design).
- `load_dataset_config()` reads `<name>.json` if present, else `<name>.yaml`.
- CI rule: a dataset may exist in **either** the json recipe **or**
  (yaml + toml entry), never both — the two example datasets migrate for real
  at the start of this phase. Default-deny semantics are preserved bit-for-bit:
  missing file/entry ⇒ `pending_review`.

**Phase 2 — bulk migration.** A `scripts/migrate_dataset_registry.py` converts
the remaining ~43 datasets mechanically (yaml + toml entry → json; `last_fetch`
/ `quality_fingerprint` → `datasets/state/<name>.yaml`); fetch/quality tooling
writes state files; delete `licenses.toml` (or regenerate it as a build
artifact for one release if anything external reads it — nothing in-repo will).

**Phase 3 — contribution recipe.** CONTRIBUTING section: "add a city = copy a
recipe, fill `source`/`fetch`/`license`, open a PR; CI validates; the license
panel researches `capabilities`+`evidence` on your PR; a human flips `status`."
Optional later: pre-commit hook, publishing the schema at a stable URL, OA-style
"run the changed source" CI.

**Schema evolution:** `schema_version` is in-band (OA); bumps only on breaking
changes; the loader rejects unknown versions; old schema files get parked in
`datasets/schema/retired/` (OA's pattern).

## Open questions

1. `datasets/state/` vs keeping machine state in `data/` (untracked): fetch
   provenance is genuinely useful in git history; leaning tracked.
2. Should `classification` eventually fold into `fetch.class_mapping` entirely?
   Today both exist in the yaml with overlapping content; kept loose in v1.
3. Whether the published `index.json` should surface `capabilities` alongside
   `license`/`attribution` (probably yes — it's exactly what external consumers
   like OSM import vetting want).
