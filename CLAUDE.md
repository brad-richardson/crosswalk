# Matcher - Claude Development Guide

Road network conflation pipeline for linking local road datasets to Overture Maps GERS identifiers.

For project overview, installation, usage, and CLI reference, see [README.md](README.md).
For ML pipeline architecture and feature details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick Reference

```bash
# Install with all dependencies
pip install -e ".[dev,ml,label]"

# Train ML model (required after fresh clone)
matcher train

# Run tests
pytest tests/

# Format and lint
ruff format src/ tests/ && ruff check src/ tests/

# CLI help
matcher --help
```

## Adding a New Feature

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
- Features computed but not saved to labels -> ML can't use them for training
- Features in labels but not computed -> labels have stale/missing values
- The test catches these mismatches automatically

**Note:** `ml.py` imports `FEATURE_COLUMNS` from `config.py`, so no separate update needed there.

## Feature Computation Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full architecture including:
- 56 features across 14 categories (source of truth: `config.py::FEATURE_COLUMNS`)
- Three computation paths (ML inference, labeling UI, training)
- Pre-computation table (what's pre-computed and why)
- Imputation consistency (training medians)
- Decision thresholds (ML scoring vs optimizer settings)
- Test coverage matrix for consistency

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

- Wait for CI checks to complete
- Check for any failing tests or lint issues
- Address feedback with **new commits** (never amend unless explicitly requested)
- Push additional commits as needed

### Commit Rules

- **Never amend commits** unless explicitly requested by the user
- Always add new commits to address feedback or fix issues
- Use descriptive commit messages that explain the "why"
- Keep commits atomic and focused on single changes
