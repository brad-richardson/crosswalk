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
├── config.py           # Pydantic settings
├── fetch/              # Data fetching (Overture, OSM, ArcGIS)
├── features/           # Feature computation (geometric, semantic)
├── blocking/           # Candidate generation via spatial indexing
├── matching/           # Matching algorithms (rules, ML)
│   ├── ml.py          # XGBoost-based ML matcher (parallelized)
│   ├── rules.py       # Rule-based matcher
│   └── optimizer.py   # Hungarian algorithm for 1:1/1:N optimization
├── pipeline/           # End-to-end pipeline runner
├── resolution/         # Bridge file generation
├── topology/           # Network topology reconstruction
├── labeling/           # Streamlit labeling UI
├── integration/        # Unmatched segment integration
└── integration_qa/     # QA app for integration review
```

## Key Commands

```bash
# Fetch data
matcher fetch --bbox -71.19,42.21,-70.92,42.40 -d overture -d osm

# Fetch Boston ArcGIS data
python scripts/fetch_boston.py

# Run matching with ML model
matcher match data/raw/overture_segments.parquet data/raw/boston_streets.parquet -m xgboost

# Train ML model on labels
matcher train

# Launch labeling UI
matcher label
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
- **Features**: 12 geometric/semantic features (hausdorff_distance, buffer_iou, name_levenshtein, etc.)
- **Parallelization**: Uses `ProcessPoolExecutor` with worker initialization for feature computation

**Note**: The trained model is not committed to git. After cloning, run `matcher train` before using `-m xgboost`.

### Key Thresholds
- `confidence >= 0.5` → MATCH (configurable via `settings.match_threshold`)
- `confidence >= 0.1` → REVIEW (configurable via `settings.review_threshold`)
- `confidence < 0.1` → NO_MATCH
- 1:N groups: `avg_confidence >= settings.review_threshold` → MATCH

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

### Adding a new dataset
1. Create fetch script in `scripts/` (see `fetch_boston.py`)
2. Run matching: `matcher match ... -m xgboost`
3. Label samples: `matcher label`
4. Retrain model: `matcher train`

### Improving match quality
1. Label more examples in the labeling UI
2. Retrain model: `matcher train`
3. Evaluate: `matcher eval-model`

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
