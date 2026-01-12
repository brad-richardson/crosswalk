# Matcher

Road network conflation pipeline for linking local road datasets to Overture Maps GERS identifiers.

## Installation

```bash
pip install -e ".[dev]"
```

### Optional Dependencies

```bash
# For OSM data fetching via pyrosm
pip install -e ".[osm]"

# For machine learning matching (XGBoost, LightGBM)
pip install -e ".[ml]"

# For distributed processing with Apache Spark/Sedona
pip install -e ".[spark]"

# All optional dependencies
pip install -e ".[dev,osm,ml]"
```

## Usage

```bash
matcher match reference.parquet target.parquet -o output/
```

## Development

```bash
# Run tests
pytest tests/

# Format code
ruff format src/ tests/

# Lint
ruff check src/ tests/
```
