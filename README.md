# Matcher

Road network conflation pipeline for linking local road datasets to [Overture Maps](https://overturemaps.org/) GERS (Global Entity Reference System) identifiers.

## What is this matching?

This project determines which local road segments correspond to Overture GERS segments, producing a **bridge file** that links local IDs to GERS IDs with confidence scores.

- **Primary goal**: Link local road features to their Overture counterparts, enabling data interoperability and update tracking
- **Secondary goal**: Identify unmatched local segments as candidates for addition to the Overture transportation theme
- **Framing**: The tool is a funnel for surfacing meaningful new road features that don't yet exist in Overture

The bridge file enables:

- **Data interoperability** - Join local attributes with Overture's standardized schema
- **Update tracking** - Detect changes by comparing GERS matches over time
- **Network integration** - Merge local roads into the Overture network while preserving provenance

## What Is a Match?

A **match** means: this fraction of this GERS segment best represents this fraction of this local segment. They are the same physical road, even if the datasets differ in segmentation, naming, or classification.

A **no match** means: either a better option exists for this local segment, or no corresponding Overture feature exists at all (the local segment is genuinely new).

### Common Edge Cases

| Scenario | Result | Why |
|----------|--------|-----|
| Different segmentation points | Match | Same road, just split differently between datasets |
| Split carriageways vs single centerline | 1:N match | One Overture centerline corresponds to multiple local segments (e.g., divided highway) |
| Road vs parallel sidewalk | No match | Different physical features, even if close together |
| Same road, different names | Match | Names are a signal, not a requirement |
| Opposite carriageways of divided road | Match (each to its own GERS segment) | Each carriageway matches independently |

### 1:N Matching

A single Overture segment can correctly correspond to multiple local segments. This happens with split highways where the local dataset has separate segments for each direction but Overture has a single centerline. The optimizer handles this via the Hungarian algorithm.

## How It Works

The pipeline uses a machine learning approach to identify corresponding road segments between datasets, even when geometries don't align perfectly or naming conventions differ.

```mermaid
flowchart TB
    subgraph Data["1. Data Acquisition"]
        A[Identify Data Source] --> B[Create Dataset Config]
        B --> C[Fetch Local Dataset]
        C --> D[Fetch Overture Reference]
    end

    subgraph Match["2. Matching Pipeline"]
        D --> E[Generate Candidates<br/>Spatial indexing + filters]
        E --> F[Compute 67 Features<br/>Geometric, semantic, topological]
        F --> G[Score with XGBoost<br/>Binary classifier]
        G --> H[Optimize 1:N Matches<br/>Hungarian algorithm]
        H --> I{Quality<br/>Acceptable?}
    end

    subgraph Label["3. Labeling Loop"]
        I -->|No| J[Launch Labeling UI]
        J --> K[Label Match/No-Match]
        K --> L[Retrain Model]
        L --> F
    end

    subgraph Output["4. Integration"]
        I -->|Yes| M[Generate Bridge File<br/>local_id → GERS_id + confidence]
        M --> N[Integrate Unmatched Segments]
        N --> O[QA Review]
        O --> P[Final Network]
    end

    style Data fill:#e1f5fe
    style Match fill:#fff3e0
    style Label fill:#f3e5f5
    style Output fill:#e8f5e9
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Bridge File** | Links local segment IDs to Overture GERS IDs with confidence scores |
| **1:N Matching** | One Overture segment can match multiple local segments (different segmentation) |
| **Features** | 67 features across 15 categories: geometric, semantic, topological, alignment, and more |
| **Labeling** | Human-in-the-loop training data creation via Streamlit UI |

## Quick Start

After installation, here's the typical workflow for matching a new dataset:

```bash
# 1. Fetch all data (target + Overture reference) for a dataset
matcher data fetch all us_boston_streets

# 2. Train the ML model (required after fresh clone)
matcher train

# 3. Run matching
matcher match data/raw/us_boston_overture_segments.parquet data/raw/us_boston_streets.parquet \
    -m xgboost -o data/output/us_boston_streets_bridge.parquet

# 4. If match quality needs improvement, label more examples (auto-discovers datasets)
matcher ui

# 5. Retrain and re-match until satisfied
matcher train && matcher match ...
```

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

If not available, the system falls back to pyosmium (slower but no system deps).

### Optional Dependencies

```bash
# For machine learning matching (XGBoost, LightGBM)
pip install -e ".[ml]"

# For labeling UI (Streamlit)
pip install -e ".[label]"

# All optional dependencies
pip install -e ".[dev,ml,label]"
```

## Workflow Details

### Step 1: Data Acquisition

Fetch Overture reference data and your local dataset. Local data typically comes from:
- State/county GIS portals (ArcGIS FeatureServers)
- OpenStreetMap extracts
- Internal road databases

```bash
# Fetch all data (target + Overture reference) for a configured dataset
matcher data fetch all us_boston_streets

# Fetch target data only (from ArcGIS/WFS)
matcher data fetch target us_boston_streets
matcher data fetch target --prefix us_boston  # All datasets for a region

# Fetch reference data only (Overture by default)
matcher data fetch reference us_boston_streets
matcher data fetch reference us_boston_streets --source osm  # Use OSM instead

# List available datasets
matcher data fetch list
```

See [docs/DATASET_INGESTION.md](docs/DATASET_INGESTION.md) for detailed instructions on adding new datasets.

### Step 2: Feature Computation & Matching

The matcher computes 67 features for each candidate pair across 15 categories:

| Category | Count | Examples |
|----------|-------|----------|
| Geometric | 11 | Hausdorff distance (mean, p95), buffer IoU (5m/15m), heading delta, length ratio, angle histogram, edge distance RMSE |
| Name Similarity | 10 | Levenshtein, Jaro-Winkler, token sort, Soundex, Metaphone, presence flags, numeric match, route prefix |
| Class | 1 | Road class similarity |
| Endpoint/Connectivity | 3 | Min/max endpoint proximity, shared endpoint count |
| Lateral Offset | 3 | Median, IQR, 95th percentile |
| Topology | 12 | Degree features, dead-end/intersection flags and matches |
| Alignment Coverage | 4 | Ref/target/min coverage, coverage ratio |
| Graphlet | 2 | Network topology similarity, endpoint degree similarity |
| Clustering | 3 | Local clustering coefficient (ref/target), delta |
| Sinuosity | 3 | Ref/target sinuosity, delta |
| Heading Consistency | 3 | Ref/target consistency, delta |
| Vertex Density | 3 | Ref/target density, ratio |
| Length | 1 | Minimum length |
| Shape Complexity | 3 | Ref/target complexity (significant turns), delta |
| Parallel Sibling | 5 | Parallel sibling detection, fraction, offset ratios, representation mismatch |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the complete feature reference and computation architecture.

```bash
matcher match data/raw/us_boston_overture_segments.parquet data/raw/us_boston_streets.parquet \
    -m xgboost -o data/output/us_boston_streets_bridge.parquet
```

### Step 3: Labeling Loop

When match quality isn't sufficient, use the labeling UI to create training data:

```bash
# Launch labeling app (auto-discovers all datasets with data in data/raw/)
matcher ui
```

Label pairs as `match`, `no_match`, or `unsure`, then retrain:

```bash
matcher train
matcher ml eval  # Cross-validation evaluation (default: 5-fold)
```

### Step 4: Integration

Merge unmatched segments into the reference network:

```bash
matcher integrate run data/raw/overture_segments.parquet \
    -t boston_streets:data/output/bridge.parquet:data/output/unmatched.parquet:1 \
    -o data/integrated
```

Review integration results with the QA app:

```bash
matcher integrate qa -o data/integrated
```

### Dataset Classification

Discover class mappings for new datasets:

```bash
# Basic discovery - analyzes dataset structure
matcher class discover data/raw/new_dataset.parquet

# With match-based analysis (more accurate)
matcher class discover data/raw/new_dataset.parquet \
    --reference data/raw/overture_segments.parquet \
    --bridge data/output/new_dataset_bridge.parquet
```

## CLI Reference

### Top-Level Commands

| Command | Description |
|---------|-------------|
| `matcher match` | Run the matching pipeline |
| `matcher train` | Train ML model on labeled data |
| `matcher ui` | Launch combined UI (labeling, label review, integration QA) |
| `matcher match-eval` | Evaluate bridge file (matching output) quality |
| `matcher screen` | Screen segments for valid network additions |
| `matcher version` | Show version information |

### Command Groups

| Group | Command | Description |
|-------|---------|-------------|
| **data** | `data fetch target` | Fetch local road data |
| | `data fetch reference` | Fetch Overture/OSM reference |
| | `data fetch all` | Fetch both target and reference |
| | `data fetch list` | List available datasets |
| | `data topology` | Reconstruct network topology |
| | `data repair` | Repair topology issues |
| | `data quality` | Dataset quality fingerprint |
| | `data validate` | Validate data file versions |
| **ml** | `ml eval` | Cross-validation evaluation (or evaluate existing model with `--model`) |
| | `ml errors` | Analyze prediction errors and diagnose model issues |
| | `ml features` | Compute and cache features for dataset(s) |
| **labels** | `labels backfill` | Recompute features for labeled pairs |
| | `labels stats` | Show label statistics across datasets |
| **integrate** | `integrate run` | Integrate unmatched segments |
| | `integrate qa` | Launch integration QA app |
| **class** | `class discover` | Discover class mappings |
| | `class analyze` | Analyze class confusion |
| | `class detect-non-roads` | Detect non-road features |
| | `class train-predictor` | Train class predictor |
| | `class predict` | Apply class predictor |
| **agent** | `agent batch` | Generate candidates for agent labeling |
| | `agent run` | Run agent on batch |
| | `agent import` | Import agent labels from CSV |
| | `agent consensus` | Analyze agent consensus |
| **validate** | `validate matching` | Run validation experiments (Overture provenance) |

Run `matcher --help` or `matcher <command> --help` for detailed options.

## Project Structure

```
src/matcher/
├── cli.py              # CLI entry point (thin wrapper)
├── cli/                # CLI package with command groups
├── config.py           # Pydantic settings & feature definitions (source of truth)
├── fetch/              # Data fetching (Overture, OSM, ArcGIS)
├── features/           # Feature computation (geometric, semantic, topological)
├── blocking/           # Candidate generation via spatial indexing
├── matching/           # Matching algorithms (ML, optimizer)
├── pipeline/           # End-to-end pipeline orchestration
├── resolution/         # Bridge file generation
├── topology/           # Network topology reconstruction
├── labeling/           # Streamlit labeling UI & label/feature/data stores
├── datasets/           # Dataset configuration & discovery
├── classification/     # Road class prediction
├── integration/        # Unmatched segment integration
├── integration_qa/     # QA app for integration review
├── screen/             # Screening tests (water bodies, buildings, landcover)
├── quality/            # Quality metrics, fingerprinting, reports
├── post_integration/   # GPS drift detection, island detection, topology repair
├── validation/         # Ground-truth validation experiments
├── agent_labeling/     # AI agent labeling batch generation
├── external/           # External tool integration (Hootenanny)
└── utils/              # Shared utilities

datasets/               # YAML dataset configs (us_boston_streets.yaml, etc.)
labels/                 # Normalized training labels
│   ├── human/          #   Human labels (metadata CSV)
│   ├── agent/          #   Agent labels (metadata CSV)
│   ├── features/       #   Computed features (Parquet)
│   └── data/           #   Raw pair data & geometries (Parquet)
docs/                   # Architecture docs, dataset ingestion guide, benchmarks
research/               # Point-in-time research documents
```

## Development

```bash
# Run tests
pytest tests/

# Format and lint
ruff format src/ tests/ && ruff check src/ tests/
```

See [CLAUDE.md](CLAUDE.md) for development workflow, commit conventions, and feature addition checklist.

## License

See [LICENSE](LICENSE) for details.
