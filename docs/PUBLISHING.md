# Publishing: R2-hosted bridge tables and target snapshots (Milestone M5)

The public artifact of the project — the rosetta stone that makes a city's
locally-keyed data joinable to the open map: queryable **local road/path ID ↔
Overture GERS id** bridge tables plus immutable, normalized source snapshots,
hosted on Cloudflare R2 (free egress). Modeled on the geocoder repo's pattern —
static Parquet + a credibility page, no serving infrastructure. The bridge tooling
is `crosswalk factory publish`; source snapshots use `crosswalk factory publish
--targets` (see [Command surface](#command-surface)).

M5 builds on the factory (M4, [FACTORY.md](FACTORY.md)), which already produces the
right shape locally: `data/factory/release=<overture-release>/dataset=<name>/{bridge.parquet,
manifest.json, groups.json, ...}`, deterministic per PR #296. Publishing is a thin,
pipeline-free step: it reads finished factory outputs, license-gates them, and
assembles + syncs a public tree.

## Status (what landed vs what awaits credentials)

**Landed (this PR):** the `crosswalk factory publish` command; deterministic staging-
tree assembly; the license registry (`datasets/licenses.toml`) + gating; the
per-release unified `all_bridges.parquet`; `index.json` (machine-readable) +
`index.html` (credibility page); SHA-256 checksums + `checksums.txt`; `--dry-run`
(default) and `--target-dir` (local) sync; immutable-release enforcement; and the
`aws s3 sync` R2 code path (untested against live R2). Validated end-to-end with
`--target-dir` against the real `data/factory` outputs on the Mac (2 published, 3
excluded — see [License review](#license-review-outcome)).

**Awaits the user (credentials):** creating the R2 bucket, setting the `R2_*` env
vars, and the first live `--no-dry-run` R2 upload. Also awaits a human license
review to move more datasets from pending to approved.

## Bucket layout

The R2 bucket mirrors the factory's Hive partitioning so `release=` paths are
**immutable** and directly range-queryable over HTTPS:

```
r2://<bucket>/
  index.html                                    # credibility page (human landing)
  index.json                                    # machine-readable index (all releases, latest pointer)
  bridges/
    release=<overture-release>/                 # IMMUTABLE once published
      index.json                                # per-release index (datasets, checksums, stats)
      checksums.txt                             # sha256sum-format manifest for the release
      all_bridges.parquet                       # unified long table (+ `dataset` column), sorted by gers_id
      dataset=<name>/
        bridge.parquet                          # copied verbatim from the factory output
        manifest.json                           # copied verbatim (provenance)
  targets/
    index.json                                  # latest snapshot and source metadata per dataset
    dataset=<name>/
      latest.json                               # mutable pointer to the newest snapshot
      snapshot=<fetch-date>/                    # IMMUTABLE once published
        data.parquet                            # normalized source geometry + attributes
        meta.yaml                               # fetch + license provenance
        ATTRIBUTION.txt                         # artifact-specific source notice
```

**Why mirror the factory partitioning.** The factory already thinks in
`release=/dataset=`; reusing it means publish is a copy + index step with no
reshaping, `factory delta` release-notes line up with published paths, and
consumers get Hive-style partition columns for free if they glob the tree.

**Immutable release paths + a stable "latest" pointer.** A published
`release=<X>/` is never mutated — a new Overture release writes a new partition.
R2/S3 has no symlinks, so the canonical "latest" pointer is the `latest_release`
field in the top-level `index.json` (and each release's own `index.json`).
Consumers read `index.json`, learn the newest release id, and construct URLs.
This is preferred over a mutable `release=latest/` **copy** because a copy doubles
current-release storage and invites the mutable/immutable confusion the layout is
designed to avoid. (A `release=latest/` mirror can be added later if a consumer
genuinely needs a fixed URL; it is deliberately not built now.)

The publisher never overwrites an existing `release=<X>/`: already-published
releases are **skipped** (reported, left byte-identical) and only new releases +
the mutable top-level index files sync — so the routine
`publish --all --no-dry-run` keeps working release after release. `--force`
intentionally re-publishes existing releases too. The R2 existence check **fails
closed**: if the check itself errors (network/auth), the sync aborts rather than
assuming the release is absent and overwriting it.

Within one environment the staging build is deterministic — bridge/manifest
bytes are copied verbatim and `all_bridges.parquet` is written with a stable
sort — so re-assembly reproduces identical checksums. (Caveat: the regenerated
`all_bridges.parquet` embeds the Parquet writer version in its footer, so its
checksum is stable per pyarrow version rather than across environments; the
verbatim-copied `bridge.parquet` files are environment-independent.)

## Public query story

Every table is a plain Parquet file. DuckDB (or any Parquet-over-HTTP reader) does
HTTP range reads, so consumers fetch only the row groups they touch — no download,
no API.

**Two access patterns, two files:**

1. **Per-dataset** `bridge.parquet` — "give me one local dataset's mapping." Small
   (1–50 MB), immutable, the primary object.

   ```sql
   SELECT * FROM read_parquet(
     'https://<bucket-host>/bridges/release=2026-01-21.0/dataset=us_usfs_flathead/bridge.parquet'
   )
   WHERE match_decision = 'match';
   ```

2. **Per-release** `all_bridges.parquet` — the **reverse lookup**: "given a GERS
   id, which local datasets reference it, across everything?" It concatenates every
   published dataset's bridge for the release, adds a `dataset` column, and is
   **sorted by `gers_id`** so `WHERE gers_id = …` prunes row groups.

   ```sql
   SELECT dataset, local_id, confidence, match_type, match_decision
   FROM read_parquet(
     'https://<bucket-host>/bridges/release=2026-01-21.0/all_bridges.parquet'
   )
   WHERE gers_id = '<gers-id>';
   ```

**Why a unified table is worth it (sizing).** 34 datasets at 1–50 MB each ≈ a few
hundred MB to ~1.7 GB per release for `all_bridges.parquet`. That is small enough
to be a single queryable object (DuckDB range-reads it; the `gers_id` sort makes
point/range lookups prune row groups), and it is the *only* way to answer
"who references this GERS id?" without the consumer fetching all 34 per-dataset
files. The per-dataset files stay as the primary artifact; `all_bridges` is the
convenience index. (If a release ever pushed `all_bridges` past a few GB, the
fallback is to partition it — e.g. by GERS-id prefix — but we are far below that.)

**Bridge schema** (as produced by the factory, published verbatim): `local_id,
gers_id, confidence, match_type, match_method, match_decision, matched_at,
pipeline_version, gers_start_frac, gers_end_frac, local_start_frac, local_end_frac`.
`all_bridges.parquet` prepends `dataset`. The **quality columns are the
differentiator**: `confidence` is a calibrated P(match) (#266), `match_type`
(1:1 / 1:N / N:1 / M:N), `match_decision` (`match` vs `review`), and the fractional
overlap columns let consumers reconstruct partial-segment matches.

## Metadata / manifest publication

The per-dataset `manifest.json` (the factory's provenance record —
[FACTORY.md § Manifest schema](FACTORY.md#manifest-schema-manifestjson)) is
published **verbatim alongside** each bridge. It is the provenance story: Overture
release, `feature_version`, model fingerprint, buffer distance, optimizer/prune
settings snapshot, timings, and counts/group stats. Consumers who want to know
"how was this made / is it stale" read the manifest; nothing is recomputed at
publish time.

The generated `index.json` is the machine-readable roll-up across all releases:
`latest_release`, the Overture attribution block, and per-dataset `{status,
license, stats, files{sha256,bytes}, gate}`. `checksums.txt` (sha256sum format)
accompanies each release for integrity verification.

## Versioning / regeneration cadence

**Per Overture release.** When a new Overture transportation release ships:

1. Fetch the new Overture segments/connectors (release id lands in the segments
   `.meta.yaml`).
2. `crosswalk factory run --all` — writes a new `release=<new>/` partition (finished
   datasets from a prior run are skipped via `full_key`).
3. `crosswalk factory delta <dataset> --from <old> --to <new>` — the consumer-facing
   GERS churn release-notes (same/changed/lost/gained).
4. `crosswalk factory publish --all` — assembles + syncs the new release; the old
   release stays untouched (immutable).

Ad-hoc republish of a release (e.g. after a model/feature-version bump that you
*intend* to supersede the prior output) uses `--force`. Otherwise releases are
append-only.

## Credibility page

`index.html` is a self-contained (inline CSS, no external requests), responsive,
light/dark static page generated from the manifests + registry + gate config. It
is deliberately honest — it advertises what is *not* validated as loudly as what
is. Contents:

- **Header** — what the tables are, latest release, generation timestamp.
- **Query it** — copy-pasteable DuckDB examples (per-dataset + reverse lookup),
  with the actual `--site-url` baked in.
- **Published datasets** — per-dataset table from the manifests: type, release,
  target count, matched count, **match rate** (matched / targets, `match` decision
  only), review count, group count, wall time, license, and the **stitch-gate
  floor** where curated stitching labels exist (else "no stitching labels").
- **Excluded — pending license review** — every withheld dataset + the reason,
  so exclusion reads as deliberate, not missing.
- **Honest caveats** — review-band edges excluded from headline match rates;
  `confidence` is calibrated (pick your own operating point); stitch-gate coverage
  is partial; per-dataset validation varies (benchmarked on a few; see
  `BENCHMARK_RESULTS.md`).
- **Licensing & attribution** — the Overture attribution (applies to every table)
  + per-dataset source license; note that unverified-license datasets are excluded
  rather than published under a guess.

Stitch-gate metrics on the page come from `mbench/datasets.toml` `[gate.*]`
floors (the human-curated quality bar), not a live measurement. Today only
`us_boston_streets` is armed (F1 ≥ 0.83 / exact ≥ 0.50), and Boston/Seattle are
**not yet in the factory** (they stay on the legacy `data/output/` review path —
see FACTORY.md "Not yet in the factory"). So the page currently shows "no
stitching labels" for the published factory datasets; when Boston/Seattle are
adopted into the factory their armed floors will surface automatically.

## Licensing & attribution

Published bridge tables are **derived works of both** the local source dataset and
Overture, so every published artifact must carry both attributions:

- **Overture** — the `[overture]` block of `datasets/licenses.toml`. Overture's
  transportation theme is OSM-derived and distributed under **ODbL 1.0**;
  attribution: *"Contains data from the Overture Maps Foundation (overturemaps.org);
  © OpenStreetMap contributors, available under the Open Database License (ODbL)
  1.0."* (Re-confirm against each Overture release's own license notice — flagged
  in the registry `note`.)
- **Per-dataset source license** — the local dataset's own terms.

### The registry (`datasets/licenses.toml`)

The dataset YAMLs carry **no** license field (checked: 0/45), so licensing lives in
a dedicated, human-reviewed registry. The publisher **never guesses**. ID-only
bridges use the base decision:

| `status` | effect |
|---|---|
| `approved` (with `license` + `attribution`) | **published** |
| `pending_review` | **excluded** (excluded-pending-review) |
| no entry | treated as `pending_review` (excluded) |

Geometry-bearing artifacts are a separate redistribution exposure and require a
second, default-deny decision:

| `geometry_status` | effect on `targets/` |
|---|---|
| `approved` (with base `status = "approved"`) | full snapshot published |
| `pending_review` or absent | snapshot excluded; an approved ID bridge may still publish |

`geometry_attribution` optionally replaces the bridge attribution for the snapshot
when full-data redistribution adds obligations such as modification, no-updates,
or warranty notices. Each snapshot carries the resolved value in both `meta.yaml`
and `ATTRIBUTION.txt`.

To publish a bridge, verify its source terms and set `status = "approved"` with a
`license` and `attribution`. To mirror the source snapshot as well, separately
verify redistribution of the complete geometry and attributes, then set
`geometry_status = "approved"`. Both human decisions are recorded in git.

### Quality holds (`quality_hold:` in the dataset YAML)

A license approval is not a quality sign-off. When a dataset's factory output is
**known-defective** (e.g. a systematic matching error), declare a persisted hold in
its `datasets/<name>.yaml`:

```yaml
quality_hold:
  reason: 'cross-mode defect: cycleways matched to parallel road centerlines at
    0.82-0.95 confidence; awaiting optimizer cross-mode gate / learned optimizer'
  since: '2026-07-06'
```

The publisher excludes any held dataset — *even if its license is approved* — with
`reason: "quality hold: …"` in `index.json` and a distinct **quality hold** badge on
the credibility page's on-hold table. The check is deterministic and offline (a YAML
read, same path as display metadata). Fail-safe: any truthy `quality_hold` value
holds, even a malformed block. Remove the block (a reviewed, git-recorded decision)
once the defect is fixed and the output re-verified. Holds beat runtime memory: an
"I'll skip it this sweep" decision made in a session dies with the session — the
YAML block is what keeps the next sweep from shipping the defect.

### License review outcome

Conservative first pass — only unambiguous **public-domain government** sources are
approved; everything else is `pending_review` (excluded) with a `likely_license`
hint for the reviewer. This exercised both paths on the current factory outputs:

- **Approved (published):** `us_usfs_flathead`, `us_montana_missoula` — plus
  `us_usfs_lolo`, `us_montana_helena`, `us_montana_bozeman` in the registry (US
  Forest Service = US federal PD under 17 U.S.C. § 105; Montana State Library MSDI
  = published public domain).
- **Excluded — pending review (present in factory):** `sg_singapore_footpaths`,
  `sg_singapore_roads` (Singapore LTA DataMall — likely Singapore Open Data Licence,
  attribution + redistribution terms to verify), `co_bogota_bike_network` (IDECA
  Bogotá cadastre — terms unclear).
- **All other 40 datasets:** `pending_review` by default, with per-dataset hints
  in the registry — a ready-made review checklist.

## Command surface

```bash
crosswalk factory publish [DATASETS...] [--all] [-D NAME]
    [--release R]              # restrict to release(s); default: all present
    [--target-dir PATH]        # publish to a LOCAL dir (no creds); omit for R2
    [--dry-run/--no-dry-run]   # default: --dry-run (build staging + report, sync nothing)
    [--force]                  # overwrite an already-published (immutable) release
    [--site-url URL]           # public base URL for the credibility-page query examples
    [--staging-dir PATH]       # staging build dir (default: data/publish_staging)
    [--factory-root PATH]      # factory root to publish from (default: data/factory)
```

Behaviour:

- **Always** builds the deterministic staging tree and prints a per-dataset
  published/excluded summary. Safe to run anytime.
- `--dry-run` (default): reports what *would* sync, uploads nothing.
- `--target-dir PATH` + `--no-dry-run`: copies the staging tree to a local dir
  (the offline test/validation path).
- No `--target-dir` (R2) + `--no-dry-run`: `aws s3 sync` to R2. If any `R2_*` env
  var is missing, it **forces a dry run** and tells you which vars to set.
- Immutable release paths: an already-published release is **skipped** (never
  overwritten); only new releases + the top-level index files sync. `--force`
  re-publishes existing releases too.

### R2 sync mechanics

Cloudflare R2 is S3-compatible, so the publisher shells out to the **`aws` CLI**
with `--endpoint-url` (rclone/boto3 not required; `aws` is already installed).
Credentials come from **environment variables only** (never read from disk, never
logged, passed to the CLI via the environment so they don't appear in process
listings):

| env var | meaning |
|---|---|
| `R2_ENDPOINT_URL` | `https://<account-id>.r2.cloudflarestorage.com` |
| `R2_ACCESS_KEY_ID` | R2 API token access key |
| `R2_SECRET_ACCESS_KEY` | R2 API token secret |
| `R2_BUCKET` | target bucket name |

The sync uses no `--delete` (immutable release paths are never removed); re-uploads
overwrite identical bytes idempotently.

### What the user does when ready

1. Create an R2 bucket (e.g. `crosswalk-bridges`) and an R2 API token
   (Object Read & Write). Attach a public custom domain / `r2.dev` URL for HTTPS
   reads.
2. Export the four env vars above in the publishing shell.
3. (Optional) Review + `status = "approved"` more datasets in
   `datasets/licenses.toml`.
4. Dry-run first: `crosswalk factory publish --all --site-url https://<your-host>`
   → inspect the summary + `data/publish_staging/index.html`.
5. Go live: `crosswalk factory publish --all --no-dry-run --site-url https://<your-host>`.
6. Apply the **R2 CORS policy** below (required for the browser data browser to
   range-read the Parquet cross-origin).

### R2 CORS policy (required for the live data browser)

The Pages-hosted [live data browser](#live-data-browser-github-pages) runs
DuckDB-WASM in the visitor's browser and issues cross-origin `GET`/`HEAD` requests
with `Range` headers straight at the R2 objects. R2 must return CORS headers that
allow this, or the browser blocks the reads. Apply this policy to the bucket
(Cloudflare dashboard → R2 → your bucket → **Settings → CORS Policy**, or
`aws s3api put-bucket-cors --endpoint-url $R2_ENDPOINT_URL --bucket $R2_BUCKET
--cors-configuration file://cors.json`):

```json
[
  {
    "AllowedOrigins": [
      "https://<your-github-username>.github.io",
      "http://localhost:8001"
    ],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["Range", "Content-Type"],
    "ExposeHeaders": ["Content-Range", "Content-Length", "Accept-Ranges", "ETag"],
    "MaxAgeSeconds": 3600
  }
]
```

Notes: `AllowedMethods` is read-only (no writes from browsers); `Range` in
`AllowedHeaders` + `Content-Range`/`Accept-Ranges` in `ExposeHeaders` are what make
partial (row-group) reads work. Replace the origin with your Pages URL (custom
domain if you set one); keep `http://localhost:8001` only if you want to test the
Pages site against live R2 from a local static server, otherwise drop it.
`AllowedOrigins: ["*"]` also works and is fine for a fully public dataset.

## Live data browser (GitHub Pages)

A static, backend-free browser for the published tables lives in **`site/`** and
deploys to **GitHub Pages** via `.github/workflows/pages.yml`. It holds **no data** —
it reads `index.json` and the Parquet **at runtime** from R2 (DuckDB-WASM, HTTPS
range reads). Two pages:

- **`index.html` (Stats)** — fetches `index.json` and renders per-dataset coverage
  (match rate, counts, groups, license), aggregate charts (match-rate + size), the
  excluded-pending-review list, and the licensing/attribution block. No DuckDB.
- **`browse.html` (Query & browse)** — DuckDB-WASM over the R2 Parquet:
  - **Reverse GERS-id lookup** against `all_bridges.parquet`;
  - **Per-dataset browse** of `bridge.parquet` with `match_decision` / confidence
    filters, pagination, and direct download links;
  - a **read-only "run your own SQL"** box (SELECT/WITH/… only);
  - the bridge tables are **ID-only** (no geometry), so instead of a map the page
    shows a copy-pasteable DuckDB example that **joins `gers_id` to Overture segment
    geometry** on public S3.

Every page carries an **unofficial/independent** banner — this is a community
project matching local datasets to Overture GERS ids, not an Overture Maps
Foundation product.

### Configuring the data source (one place)

`site/config.js` → `DEFAULT_BASE_URL` points at the **root** of the published tree
(the dir holding `index.json` and `bridges/`). Set it to the R2 public domain
(custom domain or `pub-<hash>.r2.dev`), no trailing slash. Any page also accepts a
`?base=<url>` query-string override for testing against a staging host without
editing the file (`?base` wins over `DEFAULT_BASE_URL`).

### Enabling Pages (one-time, user step)

In the repo: **Settings → Pages → Build and deployment → Source = "GitHub
Actions"**. After that, the `pages.yml` workflow deploys `site/` on every push to
`main` that touches it (path-filtered), and can be run manually via
*workflow_dispatch*. The workflow uses the official `configure-pages` /
`upload-pages-artifact` / `deploy-pages` actions with least-privilege
(`pages: write`, `id-token: write`) permissions.

### Local validation (no R2 needed)

`scripts/serve_bridges_local.py` serves any directory with the **CORS + HTTP Range**
support DuckDB-WASM needs (Python's stock `http.server` has neither). Build a real
staging tree and point the site at it:

```bash
# 1. Build a local staging tree from finished factory outputs.
crosswalk factory publish --all --no-dry-run \
    --target-dir data/publish_staging_local --site-url http://localhost:8000

# 2. Serve the data (CORS + Range) and the site, in two shells.
python scripts/serve_bridges_local.py data/publish_staging_local --port 8000
python scripts/serve_bridges_local.py site --port 8001

# 3. Open the site pointed at the local data source.
open "http://localhost:8001/index.html?base=http://localhost:8000"
```

Exercise: dashboard load, per-dataset browse + pagination, GERS-id reverse lookup,
and the SQL box. (Validated this way on the Mac against the 2-published /
3-excluded staging tree.)

## Open operational question: publish from the box or the Mac?

The scale sweep runs on the always-on 20-core box; the Mac has only a few
validation-run outputs (`data/factory` here has 5 datasets). Publishing must run
where the **complete** `data/factory` tree lives, and today that will be the box
after a full sweep. Two options:

- **Publish from the box (recommended).** The box holds the authoritative full
  sweep; publishing there avoids a large `data/factory` sync back to the Mac and
  keeps "regenerate → publish" a single machine's loop. Put the `R2_*` env vars on
  the box. The publish step is pipeline-free and cheap (copy + checksum + index),
  so it adds negligible load.
- **Publish from the Mac.** Only sensible if `data/factory` is first synced back
  (rsync/tailscale), which is the very transfer publishing-from-the-box avoids.
  Prefer this only for one-off manual releases from Mac-local validation outputs.

**Recommendation: publish from the box**, co-located with the factory sweep. The
tooling is machine-agnostic (`--factory-root`), so the Mac remains fine for
development, dry-runs, and `--target-dir` validation (as done for this PR).
```
