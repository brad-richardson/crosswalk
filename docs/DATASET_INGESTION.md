# Adding New Datasets

This guide covers the process for adding a new road dataset to the matcher pipeline, including fetching data, discovering class mappings, and validating results.

## Overview

The matcher pipeline links local road datasets to Overture Maps GERS identifiers. Each dataset needs:

1. A **fetch script** to download and normalize the data
2. A **YAML configuration** mapping source classifications to Overture classes
3. An entry in **datasets.csv** for tracking

## Step 1: Find and Fetch the Data

### Finding Public Road Data

Many states and counties publish road data through ArcGIS FeatureServers. Common sources:

- **State DOTs**: Utah SGID, Caltrans, etc.
- **County GIS portals**: Often have local street centerlines
- **Open data portals**: data.gov, state-specific portals

Look for datasets that include:
- Street names (enables name-based match verification)
- Classification codes (FHWA functional class, local codes, etc.)
- Geometry as LineStrings

### Creating a Fetch Script

Create a script in `scripts/fetch_<dataset>.py`. Example structure:

```python
#!/usr/bin/env python
"""Fetch <location> road data from <source>."""

from pathlib import Path
import geopandas as gpd
import requests
from loguru import logger

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# ArcGIS FeatureServer endpoint
ROADS_URL = "https://example.com/arcgis/rest/services/Roads/FeatureServer/0/query"

# Initial class mapping (refined later via match analysis)
CLASS_MAPPING = {
    1: "motorway",
    2: "trunk",
    3: "primary",
    # etc.
}

def fetch_roads(output_path: Path, batch_size: int = 2000) -> gpd.GeoDataFrame:
    """Fetch roads from FeatureServer.

    ArcGIS servers typically limit results to 2000 records per request,
    so we paginate through the full dataset.
    """
    # Get total count
    count_params = {
        "where": "1=1",  # Or filter by county/region
        "returnCountOnly": "true",
        "f": "json",
    }
    resp = requests.get(ROADS_URL, params=count_params)
    resp.raise_for_status()
    total_count = resp.json()["count"]
    logger.info(f"Total roads: {total_count}")

    # Fetch in batches
    all_features = []
    offset = 0
    while offset < total_count:
        params = {
            "where": "1=1",
            "outFields": "NAME,CLASS_COLUMN,OTHER_ATTRS",
            "outSR": "4326",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": batch_size,
        }
        resp = requests.get(ROADS_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        if "features" in data:
            all_features.extend(data["features"])
        offset += batch_size

    # Convert to GeoDataFrame with Overture-compatible schema
    gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")

    # Generate unique IDs
    gdf["id"] = [f"dataset_{i}" for i in range(len(gdf))]

    # Map to Overture class
    gdf["class"] = gdf["CLASS_COLUMN"].map(CLASS_MAPPING).fillna("unclassified")

    # Create names dict (Overture format)
    gdf["names"] = gdf["NAME"].apply(
        lambda n: {"primary": n} if n and n.strip() else None
    )

    # Store original attributes
    gdf["source_tags"] = gdf.apply(
        lambda row: {"CLASS_COLUMN": row["CLASS_COLUMN"]},
        axis=1,
    )

    # Save
    gdf[["id", "geometry", "names", "class", "source_tags"]].to_parquet(output_path)
    return gdf
```

### Common Issues

- **MultiLineString geometries**: Explode to LineStrings before matching:
  ```python
  gdf = gdf.explode(index_parts=False).reset_index(drop=True)
  ```

- **CRS issues**: Overture data sometimes has `None` CRS, set explicitly:
  ```python
  overture = overture.set_crs('EPSG:4326')
  ```

## Step 2: Fetch Reference Data

Fetch Overture data for your area of interest:

```bash
# Get bounding box from your local data
python -c "
import geopandas as gpd
gdf = gpd.read_parquet('data/raw/your_dataset.parquet')
print(f'{gdf.total_bounds[0]:.4f},{gdf.total_bounds[1]:.4f},{gdf.total_bounds[2]:.4f},{gdf.total_bounds[3]:.4f}')
"

# Fetch Overture segments
matcher fetch --bbox <xmin,ymin,xmax,ymax> -d overture
```

## Step 3: Run Initial Matching

Run the matcher to find correspondences:

```bash
matcher match data/raw/overture_segments.parquet data/raw/your_dataset.parquet \
    -m xgboost -o data/output/your_dataset_bridge.parquet
```

This produces a bridge file linking your dataset IDs to Overture GERS IDs.

## Step 4: Analyze Class Mappings

### Using the Discovery Command

```bash
matcher discover-classes data/raw/your_dataset.parquet \
    --reference data/raw/overture_segments.parquet \
    --bridge data/output/your_dataset_bridge.parquet
```

### Manual Analysis for Higher Accuracy

For datasets with street names, name-verified matching provides more accurate class mappings:

```python
import pandas as pd
import geopandas as gpd
from rapidfuzz import fuzz

# Load data
bridge = pd.read_parquet("data/output/your_dataset_bridge.parquet")
target = gpd.read_parquet("data/raw/your_dataset.parquet")
overture = gpd.read_parquet("data/raw/overture_segments.parquet")

# Merge in attributes
merged = bridge.merge(
    target[["id", "names", "source_tags"]],
    left_on="target_id",
    right_on="id",
    suffixes=("", "_target")
)
merged = merged.merge(
    overture[["id", "names", "class"]],
    left_on="gers_id",
    right_on="id",
    suffixes=("_target", "_ref")
)

# Extract names
def get_primary_name(names_dict):
    if isinstance(names_dict, dict):
        return names_dict.get("primary", "")
    return ""

merged["target_name"] = merged["names_target"].apply(get_primary_name)
merged["ref_name"] = merged["names_ref"].apply(get_primary_name)

# Filter to high-confidence matches with name verification
merged["name_similarity"] = merged.apply(
    lambda row: fuzz.token_sort_ratio(
        row["target_name"] or "",
        row["ref_name"] or ""
    ) / 100.0 if row["target_name"] and row["ref_name"] else 0,
    axis=1
)

# Use 70% name similarity threshold
name_verified = merged[
    (merged["confidence"] >= 0.5) &
    (merged["name_similarity"] >= 0.7)
]

print(f"Total matches: {len(bridge)}")
print(f"High-confidence: {len(merged[merged['confidence'] >= 0.5])}")
print(f"Name-verified: {len(name_verified)}")

# Extract source classification
name_verified["source_class"] = name_verified["source_tags"].apply(
    lambda x: x.get("CLASS_COLUMN") if isinstance(x, dict) else None
)

# Build confusion matrix
confusion = pd.crosstab(
    name_verified["source_class"],
    name_verified["class"],  # Overture class
    margins=True
)
print("\nConfusion matrix:")
print(confusion)

# Calculate accuracy per source class
for src_class in name_verified["source_class"].unique():
    subset = name_verified[name_verified["source_class"] == src_class]
    top_match = subset["class"].value_counts().index[0]
    accuracy = (subset["class"] == top_match).mean()
    print(f"{src_class} -> {top_match}: {accuracy:.1%} ({len(subset)} samples)")
```

### Interpreting Results

**High accuracy mappings (>80%)**:
- Local/residential roads typically map cleanly to `residential`
- Interstates/freeways map to `motorway`

**Ambiguous mappings (40-60%)**:
- Middle-tier roads (arterials, collectors) often split between `secondary` and `tertiary`
- Use the plurality class, but note the variance in the YAML config

**Overall accuracy targets**:
- 75%+ is good for geometric-only matches
- 80-85%+ is achievable with name-verified matches

## Step 5: Create YAML Configuration

Create `src/matcher/datasets/<dataset_name>.yaml`:

```yaml
name: your_dataset
description: |
  Description of the dataset source and coverage area.
  Note any quirks or data quality observations.

confidence: high  # or medium/low based on analysis
default_class: residential

source_classification:
  column: CLASS_COLUMN
  description: |
    Description of the classification system used.
  documentation_url: https://source-documentation-url
  values:
    1:
      description: Interstate
      count: 123
    2:
      description: Highway
      count: 456
    # etc.

class_mapping_rules:
  # Order by priority (highest first)
  - source_value: "1"
    target_class: motorway
    priority: 100

  - source_value: "2"
    target_class: trunk
    priority: 95

  # Add all mappings...

notes: |
  Analysis notes:
  - How many matches were analyzed
  - Key accuracy observations
  - Any special handling required
```

## Step 6: Register the Dataset

Add an entry to `datasets.csv`:

```csv
dataset_id,name,type,fetch_url,info_url,metadata
your_dataset,Display Name,road,https://feature-server-url,https://docs-url,{"classification_column": "CLASS_COLUMN"}
```

## Step 7: Validate

### List available configs

```bash
matcher list-datasets
```

### Run discovery on new data

```bash
matcher discover-classes data/raw/new_data.parquet
```

## Example: Utah Salt Lake County

### Key findings from Utah analysis:

1. **Data quality**: 99.4% of roads have street names (excellent)
2. **Classification**: CARTOCODE system (1-17 values)
3. **Match rate**: 71,603 match pairs from 63,005 source roads (some roads match multiple Overture segments)
4. **Name-verified matches**: 49,438 (84.4% of high-confidence)
5. **Accuracy**: 84.4% for name-verified matches

### Class mapping accuracy:

| Source | Target | Accuracy | Samples |
|--------|--------|----------|---------|
| 11 (Local Street) | residential | 88.7% | 46,070 |
| 5 (State Route) | primary | 73.7% | 1,413 |
| 10 (Secondary Street) | tertiary | 71.9% | 2,878 |
| 8 (Major Street) | secondary | 50.7% | 3,717 |

## Example: Fresno County

### Key findings from Fresno analysis:

1. **Data quality**: Limited street names (route IDs only)
2. **Classification**: FHWA Functional Class (F_System 1-7)
3. **Match rate**: 91% geometric match rate
4. **Accuracy**: 75.5% exact class match

### Key observation:
F_System 2 (Principal Arterial - Freeways) maps to Overture `motorway` rather than `trunk` in practice (95.1% agreement).

## Troubleshooting

### Low match rate (<50%)
- Check CRS alignment between datasets
- Verify geometry types (LineString vs MultiLineString)
- Check if datasets cover the same area

### Low class accuracy (<60%)
- Try name-verified analysis if names are available
- Source classification may not align with OSM/Overture conventions
- Consider multi-value mappings in the YAML config

### Missing data
- Check FeatureServer pagination (batch size limits)
- Verify query filters aren't too restrictive
- Check for NULL/empty geometries
