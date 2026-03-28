# Matcher - Claude Development Guide

Road network conflation pipeline for linking local road datasets to Overture Maps GERS identifiers.

For project overview, installation, usage, and CLI reference, see [README.md](README.md).
For ML pipeline architecture and feature details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick Reference

**Use `uv` as the package manager and command runner for this project.**

```bash
# Install with all dependencies
uv pip install -e ".[dev,ml,web]"

# Train ML model (required after fresh clone)
uv run matcher train

# Export model for Spark (XGBoost JSON + manifest)
uv run matcher export-model

# Run tests
uv run pytest tests/

# Format and lint
uv run ruff format src/ tests/ && uv run ruff check src/ tests/

# CLI help
uv run matcher --help
```

## Web UI

```bash
# Install web dependencies
uv pip install -e ".[dev,web]"

# Launch web UI
uv run matcher ui

# Development mode with auto-reload
uv run matcher ui --reload
```

The web UI uses FastAPI + HTMX + Leaflet. Code in `src/matcher/web/`. Modes:

| Route | Purpose |
|-------|---------|
| `/dashboard` | Overview of datasets, label counts, and status |
| `/` (Labeling) | Label candidate pairs as match/no_match/unsure |
| `/review` | Review and correct existing labels |
| `/qa` | Integration QA for reviewing unmatched segments |
| `/audit` | Audit model predictions, feature distributions |
| `/batch` | Agent batch labeling generation and review |
| `/browser` | Browse features and labeled pairs per dataset |
| `/stitching-review` | Review and curate M:N group edge selections |

## Adding a New Feature

**CRITICAL: Features must be added to multiple files to work end-to-end.**

When adding a new ML feature (e.g., a new similarity metric), update ALL of these:

1. **Add to config.py** (single source of truth):
   - Add to `FEATURE_CATEGORIES` dict under the appropriate category
   - `FEATURE_COLUMNS` is derived automatically from `FEATURE_CATEGORIES`
   - Add to `SEMANTIC_FEATURES` if it's a name/class feature

2. **Compute the feature** in `src/matcher/features/` (geometric.py, semantic.py, etc.)
   - See **Performance Requirements** below for vectorization/numba rules

3. **Wire it through compute.py**:
   - Add to `compute_pair_features()` return dict
   - Add to `_get_error_features()` with a sensible default

4. **Backfill existing labels**: Run `matcher backfill` to compute the new feature for all existing labeled pairs

**Automated verification:**
- Run `pytest tests/unit/test_label_store.py` - this test ensures feature parity
- `test_compute_pair_features_returns_all_declared_features` verifies all config features are computed
- `test_ml_feature_columns_match_computed_features` verifies ML uses the same features

**Label storage architecture** (normalized format):
- `labels/human/dataset=*/data.csv` - Human label metadata (no features)
- `labels/agent/dataset=*/data.csv` - Agent label metadata (no features)
- `labels/features/dataset=*/data.parquet` - Computed features (keyed by gers_id, target_id)
- `labels/data/dataset=*/data.parquet` - Raw pair data (geometries, attributes)
- `labels/stitching/dataset=*/data.csv` - Curated M:N group edge selections

Features are stored separately from labels via `FeatureStore` (in `labeling/feature_store.py`).
At training time, `LabelStore.load_all()` joins labels with features automatically.

**Note:** `ml.py` imports `FEATURE_COLUMNS` from `config.py`, so no separate update needed there.

### Performance Requirements for Feature Code

**CRITICAL: Feature computation runs on every candidate pair (100K-1M+ pairs per dataset). All feature code must be optimized.**

Follow this performance hierarchy when implementing features:

1. **Vectorized Shapely** (preferred for geometry operations):
   - Use `shapely.get_point()`, `shapely.get_x()`, `shapely.get_y()` on numpy arrays of geometries
   - Use `shapely.line_interpolate_point(line, distances_array)` for point sampling
   - Use `shapely.distance(points_array, line)` for batch distance computation
   - Use `shapely.get_coordinates(points_array)` to extract coords from point arrays
   - These operate at the C level with no Python per-element overhead

2. **Vectorized numpy** (preferred for numeric computation):
   - Use `np.arctan2`, `np.degrees`, broadcasting, `np.diff` etc. on arrays
   - Avoid Python loops over numpy arrays — use broadcasting instead

3. **Numba `@njit(cache=True)`** (for loops that can't be vectorized):
   - Use for O(N*M) computations, coordinate-level iteration, complex branching
   - Add JIT helpers to `src/matcher/features/_jit_helpers.py`
   - Accept pre-extracted `np.ndarray` coords, not Shapely objects (numba can't handle them)
   - Always add `cache=True` to avoid recompilation across runs
   - Can call other `@njit` functions (e.g., `angle_diff_numba`, `compute_heading_numba`)

4. **Never acceptable:**
   - Python `for` loops over geometry arrays calling Shapely methods per-element
   - Repeated `np.array(line.coords)` — extract coords once, pass as parameter
   - `line.interpolate()` in a loop — use vectorized `line_interpolate_point(line, distances)`

See `compute_crossing_angle_features()` in `geometric.py` for a reference implementation combining all three tiers: vectorized Shapely for sampling, vectorized numpy for heading computation, numba JIT for the O(N*M) angle matrix.

## Feature Computation Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full architecture including:
- 78 features across 17 categories (source of truth: `config.py::FEATURE_COLUMNS`)
- Three computation paths (ML inference, labeling UI, training)
- Pre-computation table (what's pre-computed and why)
- Imputation consistency (training medians)
- Decision thresholds (ML scoring vs optimizer settings)
- Test coverage matrix for consistency

### Coverage Asymmetry

Reference (Overture) and target (local) datasets often have very different segmentation strategies. A short local segment (10m) matched against a long Overture segment (1km) will have `target_coverage ≈ 1.0` but `ref_coverage ≈ 0.01`. When filtering by coverage, prefer `max(ref_coverage, target_coverage)` over `min_coverage` — the latter unfairly penalizes legitimate matches with asymmetric segmentation.

### Backfill Architecture

**The backfill command MUST remain a thin wrapper around the shared pipeline.**

Backfill (`matcher backfill`) recomputes features for existing labeled pairs. It routes
through the same `prepare_worker_data()` → `_compute_feature_chunk()` code path that inference
uses. The only backfill-specific code handles:

1. **Geometry resolution**: Looking up stored geometries from the data store and building an
   augmented target GeoDataFrame that includes both raw data and stored-data segments
2. **Topology override**: Preferring stored topology over computed (3-tier fallback:
   stored > computed > NaN defaults)
3. **Persistence**: Saving results to FeatureStore/DataStore

If you add new features, new worker_data keys, or change the pipeline setup, backfill picks
it up automatically through the shared path. Do NOT add custom feature computation logic to
the backfill command — it defeats the purpose and causes train/test skew.

See `tests/unit/test_backfill_parity.py` for the test that enforces this invariant.

## Change Tracking

### Before/After Comparison for PRs

When making changes to matching logic, feature computation, or optimization, run before/after comparisons on the Boston datasets to track impact:

```bash
# Before changes (on main branch)
git checkout main
uv run matcher stitch data/raw/us_boston_streets_overture_segments_v1.0.parquet \
    data/raw/us_boston_streets_v1.0.parquet \
    -m xgboost -o data/output/before_us_boston_streets_bridge.parquet

# After changes (on feature branch)
git checkout feature-branch
uv run matcher stitch data/raw/us_boston_streets_overture_segments_v1.0.parquet \
    data/raw/us_boston_streets_v1.0.parquet \
    -m xgboost -o data/output/after_us_boston_streets_bridge.parquet
```

Include comparison in PR description:

| Dataset | Metric | Before | After | Delta |
|---------|--------|--------|-------|-------|
| us_boston_streets | Matched | 10025 | 10025 | 0 |
| us_boston_streets | Review | 1099 | 1099 | 0 |
| us_boston_streets | Unmatched | 19 | 19 | 0 |
| us_boston_streets | 1:N groups | 150 | 150 | 0 |

Note: Numbers may not change for code cleanup/edge case fixes - include comparison for transparency.

## Default Development Workflow

For any code changes, follow this workflow:

### 1. Implement and Test Locally

```bash
# Run formatting and linting
uv run ruff format src/ tests/ && uv run ruff check src/ tests/

# Run all tests
uv run pytest tests/ -v

# Run training regression tests (if ML changes)
uv run pytest tests/regression/test_training.py -v
```

### 2. Self-Review Changes

Before committing, review all changes:

```bash
git diff
git status
```

Check for:
- Unused imports or dead code
- Proper formatting and linting
- Test coverage for new functionality
- Clear, descriptive commit messages

### 3. Branch, Commit, Push, and Create PR

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Stage and commit changes (do NOT use --amend unless explicitly requested)
git add .
git commit -m "Add feature description"

# Push to remote
git push -u origin feature/your-feature-name

# Create PR
gh pr create --title "PR title" --body "Description"
```

### 4. Monitor CI and Address Feedback

- Wait for CI checks to complete
- Check for any failing tests or lint issues
- Address feedback with **new commits** (never amend unless explicitly requested)
- Push additional commits as needed

### Commit Rules

- **Never amend commits** unless explicitly requested by the user
- Always add new commits to address feedback or fix issues
- Use descriptive commit messages that explain the "why"
- Keep commits atomic and focused on single changes
