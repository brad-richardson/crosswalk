# Matcher - Claude Development Guide

Road network conflation pipeline for linking local road datasets to Overture Maps GERS identifiers.

For installation and usage instructions, see [README.md](README.md).

## Quick Reference

```bash
# Install with all dependencies
pip install -e ".[dev,ml,label]"

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
# Fetch target data from ArcGIS (reads config from datasets/*.yaml)
matcher fetch target us_boston_streets
matcher fetch target --prefix us_boston  # All Boston datasets
matcher fetch list                        # List available datasets

# Fetch Overture reference data for a configured dataset
matcher fetch reference us_boston_streets
matcher fetch reference us_boston_streets --source osm  # Use OSM instead

# Fetch both target + reference in one command
matcher fetch all us_boston_streets

# Run matching with ML model
matcher match data/raw/us_boston_overture_segments.parquet data/raw/us_boston_streets.parquet -m xgboost

# Train ML model on labels
matcher train

# Evaluate model on holdout set
matcher eval-model data/models/matcher_model_combined.joblib

# Launch labeling UI (auto-discovers datasets with data in data/raw/)
streamlit run src/matcher/labeling/app.py

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
matcher validate-matching data/raw/overture.parquet --bbox "-71.19,42.21,-70.92,42.40" --strategy random
```

## Labeling App

The Streamlit labeling app auto-discovers datasets and provides a UI for creating training labels.

### Running the App

```bash
# Install labeling dependencies
pip install -e ".[label]"

# Launch the app (auto-discovers datasets from data/raw/)
streamlit run src/matcher/labeling/app.py
```

### Prerequisites

Before running the labeling app, ensure data exists in `data/raw/`:

```bash
# Fetch both target and reference data for a dataset
matcher fetch all us_boston_streets

# Or fetch separately:
matcher fetch target --prefix us_boston   # Target data from ArcGIS
matcher fetch reference us_boston_streets  # Overture reference data
```

The app will auto-discover any dataset that has both:
- `data/raw/{dataset_name}.parquet` (target/local data)
- `data/raw/{region}_overture_segments.parquet` (reference data)

## Labels Directory

Labels are stored in Hive-partitioned format:
```
labels/
├── dataset=us_boston_streets/data.csv
├── dataset=us_boston_sidewalks/data.csv
├── dataset=us_boston_bikes/data.csv
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
│   ├── us_boston_streets.parquet
│   ├── us_boston_sidewalks.parquet
│   └── us_boston_bike_network.parquet
├── output/                 # Matching results
│   ├── us_boston_streets_bridge.parquet
│   └── ...
└── models/                 # Trained ML models
    └── matcher_model_combined.joblib
```

## Dataset Configurations

Dataset configs are stored as YAML files in `datasets/` at the repo root.
Names use ISO 3166-1 alpha-2 country code prefix (e.g., `us_`, `co_`, `nl_`):

```
datasets/
├── us_boston_streets.yaml
├── us_boston_sidewalks.yaml
├── us_fort_collins_streets.yaml
├── co_bogota_roads.yaml
└── ...
```

Each YAML file contains:
- **Display info**: name, type (road/bike/sidewalk), description
- **Source config**: URL, portal URL, fetch type
- **Fetch config**: bbox, class_column, class_mapping, name_column
- **Last fetch**: timestamp, feature count, output path (auto-updated)
- **Classification**: mapping rules discovered by `matcher discover-classes`

### Using Dataset Configs

```bash
# Fetch reference data for a configured dataset
matcher fetch reference us_boston_streets

# List available dataset configs
matcher fetch list

# Discover classification for a new dataset
matcher discover-classes data/raw/new_dataset.parquet
```

### Programmatic Access

```python
from matcher.datasets.schema import get_dataset_config, list_dataset_configs

# Get a specific config
config = get_dataset_config("us_boston_streets")
if config and config.fetch:
    print(f"Bbox: {config.fetch.bbox}")
    print(f"Class column: {config.fetch.class_column}")

# List all configs
for name in list_dataset_configs():
    print(name)
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
2. **Create YAML config** in `datasets/` with ArcGIS URL, bbox, class mappings
3. **Fetch all data**: `matcher fetch all <dataset_name>`
4. **Run initial matching**: `matcher match ... -m xgboost`
6. **Analyze class mappings**: `matcher discover-classes <dataset> --reference <overture> --bridge <bridge>`
7. **Label samples** to improve model: `streamlit run src/matcher/labeling/app.py`
8. **Retrain and iterate**: `matcher train && matcher eval-model`

See [docs/DATASET_INGESTION.md](docs/DATASET_INGESTION.md) for detailed instructions.

### Improving match quality
1. Identify weak spots: run `matcher eval-model` and check per-dataset metrics
2. Label more examples: `streamlit run src/matcher/labeling/app.py`
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

## Feature Computation Architecture

Understanding the feature computation paths is critical for preventing training/inference skew.

### Single Source of Truth

```
config.py::FEATURE_COLUMNS (44 features)
         │
         ├─► compute.py::compute_pair_features()  ◄── AUTHORITATIVE computation
         │           │
         │           ├─► ml.py::_compute_single_feature() (inference)
         │           │
         │           └─► labeling UI (training data generation)
         │
         └─► label_store.py::LABEL_COLUMNS (storage schema)
```

### Computation Paths

**Path 1: ML Inference (scoring candidates)**
```
ml.py::score_candidates()
    │
    ├─► Pre-compute endpoint features: compute_endpoint_features()
    ├─► Pre-compute topology features: compute_all_topology()
    ├─► Pre-compute graphlet features: precompute_graphlet_features()
    ├─► Pre-compute alignments: compute_alignment_batch()
    │
    └─► Parallel workers call _compute_single_feature()
            │
            └─► compute_pair_features(..., endpoint_features=pre_computed, ...)
```

**Path 2: Labeling UI (training data generation)**
```
labeling UI
    │
    └─► compute_pair_features() directly
            │
            └─► LabelStore.add(features=computed_features)
```

**Path 3: Training (loading labels)**
```
ml.py::train()
    │
    └─► LabelStore.load_all()
            │
            └─► Features already stored in CSV (from labeling)
```

### Why Pre-computation Matters

The ML scorer pre-computes certain features **before** parallelization for efficiency:

| Feature Type | Pre-computed? | Why |
|--------------|---------------|-----|
| Endpoint proximity | Yes | Requires spatial index over all segments |
| Topology (degrees) | Yes | Requires Union-Find over full network |
| Graphlet features | Yes | Requires building road graph |
| Alignments | Yes | Expensive geometry operations |
| Geometric/semantic | No | Computed per-pair in workers |

**Critical invariant**: Pre-computed features must produce the same values as direct computation. This is tested in `tests/unit/test_ml_pipeline_consistency.py`.

### Imputation Consistency

Missing feature values are imputed using medians computed from **training data only**:

```python
# During training (ml.py::train)
for feat in features:
    median = np.nanmedian(X_train[:, feat_idx])  # Training data only!
    self.feature_medians[feat] = median

# During inference (ml.py::_impute_missing)
fill_value = self.feature_medians.get(feat_name, 0.0)  # Uses stored median
```

**Risk**: If a new feature is added but not in `feature_medians`, inference falls back to 0.0, which may not be appropriate. This is tested in `tests/unit/test_ml_pipeline_consistency.py`.

### Test Coverage for Consistency

| Test File | What It Catches |
|-----------|----------------|
| `test_label_store.py` | Features computed but not saved to labels |
| `test_feature_consistency.py` | Error defaults, naming conventions |
| `test_ml_pipeline_consistency.py` | Pre-computation vs direct computation, imputation consistency |

## Change Tracking

### Before/After Comparison for PRs

When making changes to matching logic, feature computation, or optimization, run before/after comparisons on the Boston datasets to track impact:

```bash
# Before changes (on main branch)
git checkout main
matcher match data/raw/overture_segments.parquet data/raw/us_boston_streets.parquet \
    -m xgboost -o data/output/before_us_boston_streets_bridge.parquet

# After changes (on feature branch)
git checkout feature-branch
matcher match data/raw/overture_segments.parquet data/raw/us_boston_streets.parquet \
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
