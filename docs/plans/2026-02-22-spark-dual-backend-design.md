# Scaling the Stitch Pipeline: GDF + Spark Dual Backend

> _Harness renamed `cbench` → `mbench` (2026-07-05). The "cbench" mentions below are historical; the harness is now invoked as `mbench`._

## Context

The matcher stitch pipeline currently runs single-node using GeoDataFrames, ProcessPoolExecutor, and Shapely STRtree. This works well for city/region-scale datasets but cannot handle the target scale: Overture (300M segments) vs OSM globally, processed per-dataset. This document explores what needs to change to add a PySpark + Sedona distributed backend while keeping the existing offline path for smaller datasets.

**Constraints**:
- **PySpark + Sedona** for distributed geospatial operations. No alternative frameworks.
- **Python UDFs / `mapPartitions`** for feature computation — keep all feature code in Python.
- **Training stays single-node** — the trained XGBoost model is serialized and broadcast to Spark workers for inference.
- **Dual backend** — GDF path stays as default for small/medium datasets, Spark path for large.

---

## Stage-by-Stage Analysis

### 1. Data Loading

| | GDF (current) | Spark |
|---|---|---|
| API | `gpd.read_parquet()` → in-memory GeoDataFrame | Sedona reads GeoParquet lazily into Spark DataFrame |
| Limitation | Entire dataset must fit in RAM | Partitioned, lazy — handles 100GB+ |
| CRS projection | `ensure_projected_crs()` at load time | `ST_Transform` or pre-projected during ingest |

**What changes**: Minimal. Both read GeoParquet. Spark path adds lazy evaluation.

### 2. Candidate Generation (Blocking)

| | GDF (current) | Spark |
|---|---|---|
| Spatial index | Shapely STRtree (single-node) | Sedona `ST_DWithin` distributed spatial join |
| Output | `CandidateBatch` (numpy arrays with positional indices) | Spark DataFrame with `(ref_id, target_id, ref_geom, target_geom, ...)` |
| Heading/distance | Computed post-join via vectorized numpy | Same math, computed as UDF or post-join |

**Key change**: Current pipeline uses **positional indices** (`ref_idx`, `target_idx`) to index into numpy arrays throughout feature computation. Spark path must use **ID-based** lookups. This is the most fundamental interface difference and would benefit from a refactor even before Spark work begins.

### 3. Feature Computation (72 features) — Hardest Stage

**Current architecture**:
- `prepare_worker_data()` runs ~10 pre-computation steps building global structures (topology Union-Find, graphlet graph, sibling STRtree, spatial context index)
- `compute_features_parallel()` dispatches chunks to ProcessPoolExecutor workers
- Workers access pre-computed data via fork/copy-on-write module globals
- `_compute_feature_chunk()` runs 3-pass architecture per chunk

**Spark approach**: `mapPartitions` calling the same Python feature functions. The existing `_compute_feature_chunk()` is already a self-contained unit of work — both backends would call it.

**The hard part is pre-computed context**:

| Context | Current | Spark approach |
|---|---|---|
| **Topology** (degrees, dead-ends, intersections) | Union-Find over all endpoints, shared via fork/CoW | Separate Spark stage: `groupBy(connector_id).count()` for Overture; Union-Find per spatial partition with halo for target. Materialize as DataFrame, join into candidates. |
| **Graphlets** (3-hop graph features) | Build local road graph from all segments, compute per-segment | Per-partition with halo (same approach as topology). Output per-segment lookup. |
| **Sibling context** (parallel carriageway detection) | STRtree over ALL segments per dataset (not serializable) | Distributed spatial self-join: find all segment pairs within 30m + parallel. Materialize per-segment lookup. |
| **Endpoint features** | `SpatialContextIndex` built from filtered candidates | Per-partition: build local spatial index from partition's target segments |
| **Alignment** (linear referencing) | Per-pair, ThreadPoolExecutor | Per-pair within `mapPartitions` — naturally parallel |

**Halo strategy for boundary correctness**: Partition both datasets by H3 cell (resolution 5, ~8km edge). Each partition includes segments from its cell + a buffer (50m for topology, 30m for siblings) into adjacent cells. Compute context, then deduplicate results by home-cell assignment. This ensures endpoint connectivity and sibling detection are correct at boundaries.

### 4. ML Scoring

| | GDF (current) | Spark |
|---|---|---|
| Model | XGBClassifier loaded from joblib | Same model, broadcast to workers |
| Inference | `model.predict_proba()` on numpy array | Pandas UDF calling `predict_proba` per partition |
| Model size | ~10MB — trivial to broadcast | Same |

**What changes**: Nearly nothing. Broadcast the model bytes, deserialize once per executor.

### 5. M:N Optimization

| | GDF (current) | Spark |
|---|---|---|
| Connected components | In-memory BFS on bipartite graph | **Collect to driver** and run existing optimizer |
| Group resolution | Union-Find + KD-tree contiguity checks | Same code, on driver |
| Input size | All scored match results | ~100-300M results → ~20-60GB. Feasible on large driver node. |

**Alternative if driver memory is tight**: Use GraphFrames `connectedComponents()` for initial component detection, then collect per-component for resolution. Most components are tiny (1-5 edges).

### 6. Output

Trivial. Both paths write the same bridge parquet schema. Spark uses `df.write.parquet()`.

---

## Recommended Abstraction Approach

Use a **lightweight execution contract** — shared types and a `StitchExecutor` protocol, with two implementations:

```
CLI: matcher stitch --execution-mode {gdf,spark}
     Spark-only options: --spark-profile {local,glue}, --partition-scheme {h3,none}, --partition-level <int>

src/matcher/pipeline/execution/
  ├── types.py               ExecutionMode enum, StitchExecutionRequest, StitchExecutionResult
  ├── base.py                StitchExecutor protocol: run(request) -> StitchExecutionResult
  ├── gdf_executor.py        Wraps current runner.py behavior
  └── spark_executor.py      New Spark orchestrator
```

Both executors produce identical output schemas (bridge.parquet + unmatched.parquet + groups.json).

**Shared code** (unchanged):
- All feature math: `features/geometric.py`, `semantic.py`, `alignment.py`, `relational.py`, `_jit_helpers.py`
- `features/compute.py` — `compute_pair_features()`, `assemble_feature_dict()`
- `matching/optimizer.py` — M:N resolution
- `matching/types.py` — MatchResult, MatchDecision, etc.
- `config.py` — feature definitions, thresholds (source of truth across both modes)
- ML model loading/prediction core

**Backend-specific**:
- Candidate generation (STRtree vs Sedona join)
- Worker dispatch (ProcessPoolExecutor vs mapPartitions)
- Pre-computation orchestration (single-node vs distributed stages)
- Output writing

**Packaging**: Spark/Sedona deps live in an optional `[spark]` extra in pyproject.toml. GDF path has zero new dependencies.

**Glue entrypoint**: `src/matcher/pipeline/execution/glue_entrypoint.py` — parses Glue job args, invokes spark executor. Matcher imported as a wheel. Production orchestration via Airflow custom operator calling Glue (operator lives outside this repo).

---

## Config Additions (`src/matcher/config.py`)

New settings for the Spark path (all optional, only relevant when `execution_mode=spark`):
- `execution_mode_default = "gdf"` — default backend
- `spark_partition_scheme = "h3"` — spatial partitioning strategy
- `spark_partition_level = 5` — H3 resolution (tune for throughput)
- `spark_max_rows_per_partition` — upper bound for partition sizing
- `spark_parity_tolerance_confidence = 0.01` — acceptable confidence delta vs GDF
- `spark_output_staging_dir` — staging directory for distributed writes

---

## Pre-Spark Refactoring (Valuable Regardless)

These changes improve the codebase and make the Spark path easier:

1. **ID-based lookups instead of positional indices**: `_compute_feature_chunk()` currently receives `(ref_idx, target_idx)` pairs and indexes into numpy arrays. Changing to ID-based dict lookups would decouple feature computation from GeoDataFrame ordering.

2. **Explicit worker_data parameter**: `_compute_feature_chunk()` reads from module-level `_worker_data` global. Making it accept `worker_data` as a parameter enables both fork/CoW (current) and broadcast/deserialize (Spark) patterns.

3. **Factor pre-computation into independent functions**: `prepare_worker_data()` is monolithic. Splitting topology, graphlets, siblings, alignments into standalone functions that return their results makes them individually replaceable for Spark.

---

## Biggest Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Pre-computed context at partition boundaries | HIGH | H3 partitioning with halo buffer; separate pre-computation stages |
| Sibling STRtree not serializable | HIGH | Pre-compute as spatial self-join, materialize as lookup table |
| Topology Union-Find at boundaries | HIGH | For Overture: trivial (explicit connectors). For target: halo partitions + dedup |
| Per-partition memory pressure | MEDIUM | Careful partition sizing (~10K segments per partition); monitor executor memory |
| Numba JIT cold-start on executors | LOW | `cache=True` already used; ~10-30s one-time per executor |
| M:N optimizer at 300M scale | MEDIUM | Collect to driver (feasible at ~20-60GB); GraphFrames fallback if needed |

---

## Data Flow: Spark Path

```
Reference GeoParquet              Target GeoParquet
       |                                  |
  [H3 partition]                    [H3 partition]
       |                                  |
  [Topology stage]                 [Topology stage]
  (connector groupBy)              (Union-Find + halo)
       |                                  |
  [Graphlet stage]                 [Graphlet stage]
  (local graph + halo)             (local graph + halo)
       |                                  |
  [Sibling stage]                  [Sibling stage]
  (spatial self-join)              (spatial self-join)
       |                                  |
       +--- ref_context ---+--- target_context ---+
                           |
                  [Sedona ST_DWithin Join]
                  (candidate generation)
                           |
                  [Join context DataFrames]
                           |
                  [mapPartitions: feature computation]
                  (calls existing Python feature code)
                           |
                  [Pandas UDF: XGBoost inference]
                  (broadcast model)
                           |
                  [Collect to driver]
                           |
                  [M:N Optimization]
                  (existing optimizer.py, unchanged)
                           |
                  [Write bridge.parquet]
```

## Parity and Validation Strategy

The Spark backend must produce **decision-level parity** (MATCH/REVIEW/NO_MATCH) with the GDF backend on the same inputs. Small confidence differences are acceptable due to floating-point ordering differences in distributed execution.

**Parity testing approach**:
- Run both backends on a golden dataset (e.g., Boston streets), compare decision agreement rate
- Confidence tolerance envelope defined in config (`spark_parity_tolerance_confidence`)
- Output schema and row-level key consistency checks
- Edge case coverage: sparse rural, dense urban grid, divided carriageways, M:N groups

**Scale testing** (Spark-specific):
- Country/global-slice stress runs: memory stability, partition skew, end-to-end runtime
- Output sanity: duplicate key rate, unmatched inflation, group conflict counts at partition boundaries
- Deterministic output on rerun with fixed seed/config

**Acceptance criteria**:
- GDF mode behavior unchanged for all existing tests
- Spark mode passes decision parity threshold on validation datasets
- Spark mode completes country-scale benchmark without driver OOM
- Glue wrapper executes with wheel-based package import

**Note**: cbench stays GDF-only. Spark parity uses a dedicated harness, not cbench.

---

## Implementation Phases (if pursued)

1. **Phase 0 — Pre-Spark refactoring**: ID-based lookups, explicit worker_data param, factored pre-computation. Valuable regardless of Spark adoption.
2. **Phase 1 — Execution contract extraction**: Introduce `StitchExecutor` protocol + `GdfExecutor` wrapping current behavior. No behavior change, just structural.
3. **Phase 2 — Spark candidate generation**: Sedona spatial join as alternative to STRtree. Highest value, simplest Spark piece.
4. **Phase 3 — Distributed pre-computation**: Topology, graphlets, siblings as Spark stages with H3 + halo. Includes partition boundary reconciliation.
5. **Phase 4 — Distributed feature computation + scoring**: `mapPartitions` calling existing feature functions. Broadcast model inference.
6. **Phase 5 — Optimization + output**: Collect-to-driver M:N optimization. Parity validation on golden datasets.
7. **Phase 6 — Glue operationalization**: Glue entrypoint, runtime arg contract, Airflow integration docs.

---

## Non-Goals (Phase 1)

- No Spark in cbench
- No auto backend selection by dataset size
- No rewrite of local internals to abstract dataframe operations
- No distributed training — training stays single-node
- No DuckDB or other alternative backends

---

## What NOT to Do

- **Don't rewrite features in Scala/Java**: 72 features using Shapely, Numba, rapidfuzz, jellyfish — reimplementing in JVM is months of work with numerical divergence risk. `mapPartitions` with Python workers is the right call.
- **Don't distribute the M:N optimizer prematurely**: Its input is much smaller than feature computation. Only optimize if profiling shows it's a bottleneck.
- **Don't use `xgboost.spark`**: It's designed for distributed training, not pre-trained model inference. Broadcasting the model is simpler and correct.
- **Don't force adapter purity**: Current code is tightly GeoDataFrame-coupled. Forced polymorphic adapters everywhere add high refactor risk before scale is proven. Two separate executor implementations sharing core math is the right balance.
