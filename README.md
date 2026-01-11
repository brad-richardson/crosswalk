# Matcher

Road network conflation pipeline for linking local road datasets to Overture Maps GERS identifiers.

## Installation

```bash
pip install -e ".[dev]"
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
