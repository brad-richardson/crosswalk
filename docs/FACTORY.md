# Bridge-Table Factory (`crosswalk factory`)

Milestone **M4** of [SCALING_ROADMAP.md](SCALING_ROADMAP.md): batch, versioned,
resumable stitching of many local datasets to Overture. The factory produces
quality-annotated `local_id ↔ GERS` bridge tables (plus curated M:N stitching
groups) laid out per Overture release, with per-dataset manifests that drive
incremental/resume and a scored-candidate cache for fast re-optimization.

The factory is a thin orchestration layer over the existing stitch pipeline. It
routes every dataset through the same `run_pipeline` seams
(`load_and_filter_inputs → score_candidates_from_geodataframes →
optimize_and_export`), so factory output for a dataset is identical to
`crosswalk stitch <dataset>` (modulo the versioned path + the `matched_at`
timestamp). It does **not** fork pipeline logic.

## Quick start

```bash
# Run all stitchable datasets discovered under data/raw, 4 workers
uv run crosswalk factory run --all --workers 4

# Run specific datasets
uv run crosswalk factory run us_frisco_trails us_usfs_lolo -D de_berlin_roads

# Re-run only grouping/optimization from cached scores (~2 s vs ~7 min)
uv run crosswalk factory reoptimize --all

# GERS churn between two releases (release-notes artifact)
uv run crosswalk factory delta us_frisco_trails --from 2026-01-21.0 --to 2026-06-17.0

# Status of everything produced so far
uv run crosswalk factory status
```

## Command surface

### `crosswalk factory run [DATASETS...] [--all] [-D name ...]`

Runs the full stitch pipeline per dataset in parallel worker **processes** (feature
scoring is CPU-bound; it parallelizes trivially across datasets). Failure of one
dataset never aborts the batch — each returns a status (`done` / `skipped` /
`failed`) rendered in a summary table, and the command exits non-zero iff any
dataset failed.

| Option | Default | Notes |
|---|---|---|
| `--workers, -w` | `min(4, cores/4)` | Concurrent dataset processes. **Keep at `1` on the box** and put the cores into `-j` — see the runbook: `--workers > 1` nests fork-based process pools and crashes on large sweeps. |
| `--jobs-per-dataset, -j` | `1` | Internal scoring parallelism *within* each dataset process. **Use `12` on the box.** |
| `--release` | derived | Override the Overture release (else read from the segments `.meta.yaml`). |
| `--buffer-m, -b` | `75.0` | Candidate search radius (m). Part of `score_key`. |
| `--force, -f` | off | Rerun even if the manifest is current. |
| `--raw-dir` | `data/raw` | Input directory of dataset triples. |
| `--output-dir` | `data/factory` | Factory root. Never `data/output` (that is the live review area). |

Total core budget ≈ `workers × jobs-per-dataset`. **On the box, use `--workers 1
--jobs-per-dataset 12`** (one dataset at a time, scoring across 12 cores) — NOT
`--workers 12 --jobs-per-dataset 1`. The latter reads intuitively (one dataset per
core), but the outer dataset pool is a fork-based `ProcessPoolExecutor` and each
dataset's feature scoring (`features/pipeline.py::compute_features_parallel`) forks
its *own* inner pool. Nested fork pools die with `BrokenProcessPool` at ≥6
concurrent datasets — see the runbook. Datasets then run sequentially, but a single
large dataset still scores in parallel across all 12 cores, so wall time is
comparable while staying stable. Restructuring the multiprocessing so an outer
worker pool is safe (e.g. spawn context, or a single shared pool) is a deferred
follow-up (see "Not yet in the factory").

### `crosswalk factory reoptimize [DATASETS...] [--all]`

Re-runs only optimization / grouping / sidecar export from the cached scored
candidates (`scored_candidates.parquet`). This is the iteration loop for grouping
or optimizer-setting changes: ~2 s instead of a full re-score. Requires a prior
`factory run` whose cache is still valid — i.e. the manifest's `score_key`
(inputs + model + `FEATURE_VERSION` + buffer) still matches. If the score-relevant
inputs changed, `reoptimize` refuses and tells you to `run --force`.

### `crosswalk factory delta DATASET --from RELEASE --to RELEASE`

Reports `local_id`-level GERS match churn between two factory releases of a
dataset: `same` (identical GERS set), `changed` (matched in both, different set),
`lost` (dropped out), `gained` (newly matched). `--format md|csv`, `-o FILE` to
write. Only `match_decision == "match"` bridge rows count as matched (review-band
rows are excluded, matching the pipeline's own matched/unmatched accounting). This
is the consumer-facing release-notes artifact for a new Overture release.

### `crosswalk factory status`

Lists every `release=*/dataset=*` output with its matched count, group count,
wall time, and creation timestamp.

### `crosswalk factory publish [DATASETS...] [--all]`

Assembles the public R2 publication tree from the factory outputs and syncs it
(Milestone **M5**). Pipeline-free: reads finished `release=/dataset=` outputs,
license-gates them via `datasets/licenses.toml`, copies cleared bridge + manifest
files, builds a per-release unified `all_bridges.parquet` + `checksums.txt` +
machine-readable `index.json` + a static credibility `index.html`, then syncs.
Defaults to `--dry-run` (build staging + report, upload nothing); `--target-dir`
publishes to a local dir (no credentials); the R2 path uses the S3-compatible
`aws` CLI with `R2_*` env-var credentials. Immutable release paths (an
already-published release is skipped, never overwritten, unless `--force`).
**Full design + the R2 setup steps: [PUBLISHING.md](PUBLISHING.md).**

## Output layout

Versioned, Hive-partitioned under the factory root (default `data/factory/`):

```
data/factory/
  release=<overture-release>/
    dataset=<name>/
      bridge.parquet              # local_id ↔ gers_id bridge (same schema as crosswalk stitch)
      groups.json                 # M:N/1:N/N:1 stitching groups sidecar (identical to the stitch sidecar)
      manifest.json               # provenance + staleness keys + counts/group stats
      unmatched.parquet           # unmatched + review-band target features
      scored_candidates.parquet   # cached MatchResults for reoptimize (validity-keyed by score_key)
      run.log                     # per-dataset structured log
```

The Overture release identifier is read from the segments file's `.meta.yaml`
(`release:` field); pass `--release` to override or supply one when absent.

## Manifest schema (`manifest.json`)

```jsonc
{
  "dataset": "us_frisco_trails",
  "release": "2026-01-21.0",
  "schema_version": 1,
  "created_at": "<iso8601>",
  "feature_version": "2026-07-04.2",
  "data_version": "v1.0",
  "buffer_distance_m": 75.0,
  "method": "xgboost",

  // Provenance / staleness inputs
  "inputs": { "reference": {name,size,mtime_ns}, "target": {...}, "connectors": {...} },
  "model":  { "name", "present", "size", "mtime_ns" },
  "settings_snapshot": {              // optimizer/prune/export settings + prune allowlist STATE
    "bridge_min_confidence", "enable_calibration", "enable_score_propagation",
    "optimizer_corridor_aware", "optimizer_corridor_max_turn_deg",
    "optimizer_glue_min_confidence", "optimizer_glue_min_confidence_raw",
    "optimizer_review_threshold", "contiguity_tolerance_m",
    "stitch_export_*", "stitch_persist_rejected_edges",
    "stitch_rejected_edges_max_per_group",
    "resolver_prune_enabled", "resolver_prune_overrides"   // the tuned allowlist
  },

  // Staleness keys (see below)
  "score_key": "…", "optimize_key": "…", "full_key": "…",

  // Timings
  "score_wall_s", "optimize_wall_s", "wall_s",

  // Counts
  "n_reference", "n_target", "n_candidates", "n_matched", "n_review", "n_unmatched",

  // Group stats (from groups.json)
  "groups": {
    "n_groups", "n_one_to_n", "n_n_to_one", "n_m_to_n",
    "edge_count_mean", "edge_count_p50", "edge_count_p90", "edge_count_p99", "edge_count_max",
    "n_monster",   // groups with > 20 edges
    "n_oversized"  // export-gated (structurally complex) groups
  },

  "scored_cache": { "path", "n_results", "schema_version" }
}
```

### Staleness keys — how incremental/resume works

- **`score_key`** = hash of everything that changes the *scores* (or their
  on-disk form): input file fingerprints (size + mtime), model fingerprint,
  `FEATURE_VERSION`, `DATA_VERSION`, buffer distance, method, and the scored-cache
  schema version. The `scored_candidates.parquet` cache is valid iff its
  manifest's `score_key` still matches — this is what `reoptimize` checks.
- **`optimize_key`** = hash of the optimizer/prune/export settings snapshot
  (including the resolver-prune allowlist and `optimizer_review_threshold`) plus
  the optimizer `min_confidence` argument. These can change *without* invalidating
  the scores.
- **`full_key`** = hash of both. `factory run` **skips** a dataset whose existing
  manifest has the same `full_key` (unless `--force`).

The manifest is written **last**, atomically. A killed run leaves a dataset
without a complete manifest, so the next `run` re-runs exactly the datasets that
did not finish — resume is automatic.

### Determinism note

The optimizer is **fully deterministic** — its output does not depend on
`PYTHONHASHSEED`. Two *separate* process invocations on identical scores produce
a byte-for-byte identical `groups.json` and an identical bridge parquet (modulo
the `matched_at` timestamp column). `reoptimize` likewise reproduces a full run
exactly. No `PYTHONHASHSEED` workaround is needed, and `factory delta` between
releases reflects only genuine input/model changes — no phantom churn.

This was not always true: the optimizer historically had a hash-seed-dependent
tie-break (grouping / greedy assignment resolved equal-confidence edges via
Python set/dict iteration order), so separate processes could churn a handful of
edge selections (~1% of groups on some datasets), polluting `factory delta` and
breaking bridge-table bit-reproducibility. The source of that nondeterminism —
`list(set(...))` id dedup, set-iteration in BFS neighbour traversal / component
edge collection, and an input-order-dependent greedy tie-break — was replaced
with canonical ordering (sorted stable ids, explicit greedy tie-break key). See
`tests/unit/test_optimizer.py::TestOptimizerDeterminism` (including a slow
cross-`PYTHONHASHSEED` subprocess check).

## Box deployment runbook (the 20-core / 64 GB workhorse)

The scale workhorse is the always-on Linux box: `ssh -p 9292 brad@192.168.1.92`
(reachable over tailscale/LAN; the port is the custom SSH port). It doubles as a
media server, so **cap the factory at 12 cores** to leave headroom.

```bash
# 1. Sync code + install deps (uv is the runner)
cd ~/dev/matcher && git pull
uv pip install -e ".[dev,ml,web]"

# 2. Ensure the model exists (fresh clone) — required before scoring
uv run crosswalk train                 # or copy data/models/matcher_model_combined.joblib

# 3. Cap native BLAS/OpenMP thread pools to 1 BEFORE launching. Scoring already
#    parallelizes at the process level (-j 12), so per-process native threads only
#    oversubscribe the 12 cores and fight the scoring pool. Export these in the
#    same shell that launches the sweep:
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# 4. Full overnight sweep of all stitchable datasets, resumable. Run detached so an
#    SSH drop doesn't kill it; logs stream to factory.out and each dataset also gets
#    data/factory/.../run.log.
#
#    USE --workers 1 --jobs-per-dataset 12 (NOT --workers 12 --jobs-per-dataset 1).
#    --workers > 1 forks an OUTER dataset pool while each dataset's feature scoring
#    forks its OWN inner pool; the nested fork pools die with BrokenProcessPool
#    ~8-50s in at >=6 concurrent datasets (reproduced on this box; NOT OOM / cgroup
#    / segfault / native-thread oversubscription — it is the nested fork itself).
#    So run datasets sequentially and put all 12 cores into intra-dataset scoring.
nohup uv run crosswalk factory run --all --workers 1 --jobs-per-dataset 12 \
    > factory.out 2>&1 &

# 5. If it dies / you kill it, just re-run the same command — finished datasets
#    are skipped via full_key; only unfinished ones re-run.
uv run crosswalk factory run --all --workers 1 --jobs-per-dataset 12   # resumes

# 6. Iterate on grouping/optimizer settings without re-scoring (~2 s/dataset).
#    reoptimize does no feature scoring (reads the cache), so it never spawns the
#    inner pool — but keep --workers 1 for consistency / to avoid the outer pool.
uv run crosswalk factory reoptimize --all --workers 1
```

Sizing: the full ~24-dataset inventory is roughly one overnight run. Feature
scoring is ~84% of wall time (~600 scored pairs/s/process). Because datasets run
sequentially (`--workers 1`), only one reference set is resident at a time, so
memory is bounded by the single largest dataset rather than 12 concurrent ones —
Berlin peaked at 7.2 GB; `jp_tokyo` (~1.26M ref segments) is projected 15–20 GB,
which fits 64 GB comfortably one-at-a-time. Spatial tiling stays deferred
(unnecessary below ~1M ref segments per dataset on this box).

> **Why not parallelize across datasets?** `--workers > 1` is the intuitive way to
> fill 12 cores, but it nests fork-based process pools (outer dataset pool × inner
> per-dataset feature-scoring pool) and reliably crashes with `BrokenProcessPool`
> on large sweeps. Making an outer worker pool safe (spawn context / single shared
> pool) is a deliberate deferred follow-up — see "Not yet in the factory". Until
> then the CLI emits a startup warning when `--workers > 1`.

## Not yet in the factory (deliberately)

- **`us_boston_streets` and `us_seattle_sidewalks`** are still produced by the
  legacy `crosswalk stitch` → `data/output/` path so the panel/review queues stay
  stable. Adopting them into the factory layout is a follow-up.
- **R2 publish (M5)**: SHIPPED as `crosswalk factory publish` — assembles the public
  staging tree (license-gated bridge + manifest per dataset, per-release unified
  `all_bridges.parquet`, checksums, machine-readable `index.json`, credibility
  `index.html`) and syncs to Cloudflare R2 via the S3-compatible `aws` CLI. The
  live R2 upload awaits the user's `R2_*` credentials + bucket; local (`--target-dir`)
  and `--dry-run` paths are validated. See [PUBLISHING.md](PUBLISHING.md).
- **Inventory repair (M4 remainder)**: the 10 labeled datasets missing a local
  target parquet, and the bogota bike class-vocab fix, are prerequisites for
  running those datasets through the factory.
- **Safe cross-dataset parallelism (`--workers > 1`)**: today the outer dataset
  pool nests inside each dataset's fork-based feature-scoring pool and crashes with
  `BrokenProcessPool` on large sweeps, so the box runs `--workers 1
  --jobs-per-dataset 12` (sequential datasets, parallel scoring). Reworking the
  multiprocessing — a `spawn` start method for the outer pool, or a single shared
  pool that both layers draw from — to make `--workers > 1` safe is a deferred
  follow-up. Until then the CLI warns when `--workers > 1`.
