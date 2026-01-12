# Matcher

Road network conflation pipeline for linking local road datasets to Overture Maps GERS identifiers.

## Installation

```bash
pip install -e ".[dev]"
```

### System Dependencies

OSM data fetching requires `osmium-tool` for efficient PBF extraction:
```bash
# macOS
brew install osmium-tool

# Ubuntu/Debian
apt install osmium-tool
```

### Optional Dependencies

```bash
# For machine learning matching (XGBoost, LightGBM)
pip install -e ".[ml]"

# For distributed processing with Apache Spark/Sedona
pip install -e ".[spark]"

# All optional dependencies
pip install -e ".[dev,ml]"
```

## Usage

### Fetch data

```bash
# Fetch Overture data (default)
matcher fetch --bbox -122.7,45.5,-122.6,45.55

# Fetch OSM data (auto-downloads from Geofabrik)
matcher fetch --bbox -122.7,45.5,-122.6,45.55 -d osm

# Fetch both Overture and OSM
matcher fetch --bbox -122.7,45.5,-122.6,45.55 -d overture -d osm

# OSM with options
matcher fetch --bbox -71.08,42.35,-71.05,42.37 -d osm --no-cache --keep-pbf
```

### Run topology reconstruction

```bash
matcher topology data/raw/overture_segments.parquet
```

### Run matching

```bash
matcher match data/processed/edges.parquet data/raw/local_roads.parquet
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
