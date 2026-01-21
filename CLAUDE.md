# Matcher - Claude Development Guide

Road network conflation pipeline for linking local road datasets to Overture Maps GERS identifiers.

For installation and usage instructions, see [README.md](README.md).

## Quick Reference

```bash
# Install with all dependencies
pip install -e ".[dev,ml]"

# Train ML model (required after fresh clone)
matcher train --combined

# Run tests
pytest tests/

# Format and lint
ruff format src/ tests/ && ruff check src/ tests/

# CLI help
matcher --help
```

## Project Structure

```
src/matcher/
├── cli.py              # Typer CLI application
├── config.py           # Pydantic settings, FEATURE_COLUMNS (source of truth)
├── fetch/              # Data fetching (Overture, OSM, ArcGIS)
├── features/           # Feature computation
│   ├── compute.py     # Main feature computation interface
│   ├── geometric.py   # Hausdorff, IoU, heading, length features
│   ├── semantic.py    # Name similarity, class matching
│   ├── spatial_context.py  # Topology and endpoint features
│   ├── alignment.py   # Segment alignment and coverage
│   └── relational.py  # Lateral offset features
├── blocking/           # Candidate generation via spatial indexing
├── matching/           # Matching algorithms (rules, ML)
│   ├── ml.py          # XGBoost-based ML matcher (parallelized)
│   ├── rules.py       # Rule-based matcher
│   └── optimizer.py   # Hungarian algorithm for 1:1/1:N optimization
├── pipeline/           # End-to-end pipeline runner
├── resolution/         # Bridge file generation
├── topology/           # Network topology reconstruction
├── labeling/           # Streamlit labeling UI
├── datasets/           # Dataset configuration discovery
├── integration/        # Unmatched segment integration
├── integration_qa/     # QA app for integration review
├── validation/         # Ground-truth validation experiments
├── agent_labeling/     # AI agent labeling batch generation
└── utils/              # Shared utilities
```

## Key Commands

```bash
# Fetch data
matcher fetch --bbox -71.19,42.21,-70.92,42.40 -d overture -d osm

# Fetch local data (see scripts/fetch_*.py for examples)
python scripts/fetch_boston.py

# Run matching with ML model
matcher match data/raw/overture_segments.parquet data/raw/boston_streets.parquet -m xgboost

# Train ML model on labels
matcher train

# Evaluate model on holdout set
matcher eval-model data/models/matcher_model_combined.joblib

# Launch labeling UI
matcher label data/raw/overture_segments.parquet data/raw/boston_streets.parquet

# Integrate unmatched segments into network
matcher integrate data/raw/overture_segments.parquet \
    -t boston_streets:data/output/bridge.parquet:data/output/unmatched.parquet:1

# QA integration results
matcher qa-integration -o data/integrated

# Discover class mappings for new datasets
matcher discover-classes data/raw/new_dataset.parquet \
    --reference data/raw/overture_segments.parquet \
    --bridge data/output/new_dataset_bridge.parquet

# Run validation experiments (ground-truth from Overture provenance)
matcher validate data/raw/overture.parquet --bbox "-71.19,42.21,-70.92,42.40" --strategy random
```

## Labels Directory

Labels are stored in Hive-partitioned format:
```
labels/
├── dataset=boston_streets/data.csv
├── dataset=boston_sidewalks/data.csv
├── dataset=boston_bikes/data.csv
└── ...
```

Each CSV contains: `gers_id`, `target_id`, `label` (match/no_match/unsure), and pre-computed features.

## ML Model

- **Location**: `data/models/matcher_model_combined.joblib`
- **Algorithm**: XGBoost binary classifier
- **Features**: 42 features across 6 categories (defined in `src/matcher/config.py`)
  - Geometric (11): hausdorff_distance_m, buffer_iou_5m/15m, heading_delta, length_ratio, etc.
  - Semantic - Name (8): levenshtein, jaro_winkler, token_sort, soundex, metaphone, presence flags
  - Semantic - Class (1): class_similarity
  - Endpoint/Connectivity (3): min/max_endpoint_proximity_m, shared_endpoint_count
  - Lateral Offset (3): lateral_offset_m, iqr, p95 (robust to outliers)
  - Topology (12): degree features, dead_end/intersection flags
  - Alignment Coverage (4): ref/target/min coverage, coverage_ratio
- **Parallelization**: Uses `ProcessPoolExecutor` with worker initialization for feature computation
- **Auto Model Selection**: When `settings.auto_select_model=True`, automatically uses geometry-only model for datasets with low name coverage

**Note**: The trained model is not committed to git. After cloning, run `matcher train` before using `-m xgboost`.

### Key Thresholds
- `confidence >= 0.5` → MATCH (configurable via `settings.match_threshold`)
- `confidence >= 0.1` → REVIEW (configurable via `settings.review_threshold`)
- `confidence < 0.1` → NO_MATCH
- 1:N groups: `avg_confidence >= settings.review_threshold` → MATCH

### Model Evaluation

**Always use holdout evaluation for unbiased metrics:**

```bash
# Default: 20% holdout with seed=42 (recommended)
matcher eval-model data/models/matcher_model_combined.joblib

# Custom seed for different split
matcher eval-model data/models/matcher_model_combined.joblib --seed 123

# Evaluate on ALL data (may include training data - use with caution)
matcher eval-model data/models/matcher_model_combined.joblib --no-holdout
```

**Why holdout matters:**
- Evaluating on training data gives artificially inflated accuracy (~99%)
- Holdout evaluation gives realistic generalization metrics (~95-96%)
- Use consistent seed (default: 42) for comparable results across experiments
- When comparing models or feature sets, always use the same holdout split

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=matcher

# Run specific test file
pytest tests/unit/test_ml.py -v
```

## Data Files

After fetching, data is stored in:
```
data/
├── raw/                    # Fetched source data
│   ├── overture_segments.parquet
│   ├── overture_connectors.parquet
│   ├── osm_segments.parquet
│   ├── boston_streets.parquet
│   ├── boston_sidewalks.parquet
│   └── boston_bike_network.parquet
├── output/                 # Matching results
│   ├── boston_streets_bridge.parquet
│   └── ...
└── models/                 # Trained ML models
    └── matcher_model_combined.joblib
```

## System Dependencies

- **osmium-tool**: Optional but recommended for fast OSM PBF extraction (`apt install osmium-tool`)
  - If not available, the system falls back to pyosmium (slower but no system deps)
- Python 3.10+

## Common Workflows

### End-to-End Workflow

```
1. DATA ACQUISITION
   └── Identify source → Create fetch script → Fetch local + Overture data

2. INITIAL MATCHING
   └── Generate candidates → Compute features → Score with ML → Optimize 1:N

3. LABELING LOOP (iterate until quality acceptable)
   └── Launch labeling UI → Label pairs → Retrain model → Re-evaluate

4. INTEGRATION & QA
   └── Generate bridge file → Integrate unmatched → QA review orphans
```

### Adding a new dataset
1. **Find data source**: State DOTs, county GIS portals, open data portals
2. **Create fetch script** in `scripts/` (see `fetch_boston.py` as template)
3. **Fetch reference data**: `matcher fetch --bbox <bbox> -d overture`
4. **Run initial matching**: `matcher match ... -m xgboost`
5. **Analyze class mappings**: `matcher discover-classes <dataset> --reference <overture> --bridge <bridge>`
6. **Create YAML config** in `src/matcher/datasets/` if needed
7. **Label samples** to improve model: `matcher label`
8. **Retrain and iterate**: `matcher train && matcher eval-model`

See [docs/DATASET_INGESTION.md](docs/DATASET_INGESTION.md) for detailed instructions.

### Improving match quality
1. Identify weak spots: run `matcher eval-model` and check per-dataset metrics
2. Label more examples in the labeling UI: `matcher label`
3. Retrain model: `matcher train`
4. Evaluate improvement: `matcher eval-model`
5. Repeat until F1 is acceptable (typically >0.95)

### Adding a New Feature

**CRITICAL: Features must be added to multiple files to work end-to-end.**

When adding a new ML feature (e.g., a new similarity metric), update ALL of these:

1. **Add to config.py** (single source of truth):
   - Add to `FEATURE_COLUMNS` list
   - Add to `SEMANTIC_FEATURES` if it's a name/class feature

2. **Compute the feature** in `src/matcher/features/` (geometric.py, semantic.py, etc.)

3. **Wire it through compute.py**:
   - Add to `compute_pair_features()` return dict
   - Add to `_get_error_features()` with a sensible default

4. **Save it in label_store.py**:
   - Add to `LABEL_COLUMNS` list
   - Add to `add()` method with `features.get("feature_name", default)`

**Automated verification:**
- Run `pytest tests/unit/test_label_store.py` - this test ensures feature parity
- The test `test_all_computed_features_are_in_label_columns` will fail if you forget label_store.py

**Why this matters:**
- Features computed but not saved to labels → ML can't use them for training
- Features in labels but not computed → labels have stale/missing values
- The test catches these mismatches automatically

**Note:** `ml.py` imports `FEATURE_COLUMNS` from `config.py`, so no separate update needed there.

## Change Tracking

### Before/After Comparison for PRs

When making changes to matching logic, feature computation, or optimization, run before/after comparisons on the Boston datasets to track impact:

```bash
# Before changes (on main branch)
git checkout main
matcher match data/raw/overture_segments.parquet data/raw/boston_streets.parquet \
    -m xgboost -o data/output/before_boston_streets_bridge.parquet

# After changes (on feature branch)
git checkout feature-branch
matcher match data/raw/overture_segments.parquet data/raw/boston_streets.parquet \
    -m xgboost -o data/output/after_boston_streets_bridge.parquet
```

Include comparison in PR description:

| Dataset | Metric | Before | After | Delta |
|---------|--------|--------|-------|-------|
| boston_streets | Matched | 10025 | 10025 | 0 |
| boston_streets | Review | 1099 | 1099 | 0 |
| boston_streets | Unmatched | 19 | 19 | 0 |
| boston_streets | 1:N groups | 150 | 150 | 0 |

Note: Numbers may not change for code cleanup/edge case fixes - include comparison for transparency.

## Default Development Workflow

For any code changes, follow this workflow:

### 1. Implement and Test Locally

```bash
# Run formatting and linting
ruff format src/ tests/ && ruff check src/ tests/

# Run all tests
pytest tests/ -v

# Run training regression tests (if ML changes)
pytest tests/regression/test_training.py -v
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

- Wait for CI checks to complete (~5 minutes)
- Check for any failing tests or lint issues
- Address feedback with **new commits** (never amend unless explicitly requested)
- Push additional commits as needed

### Commit Rules

- **Never amend commits** unless explicitly requested by the user
- Always add new commits to address feedback or fix issues
- Use descriptive commit messages that explain the "why"
- Keep commits atomic and focused on single changes
