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
- `confidence >= 0.5` → MATCH
- `confidence >= 0.1` → REVIEW
- `confidence < 0.1` → NO_MATCH
- 1:N groups: `avg_confidence >= 0.5` → MATCH (in `optimizer.py`)

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

- **osmium-tool**: Required for OSM PBF extraction (`apt install osmium-tool`)
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
