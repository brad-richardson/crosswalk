#!/usr/bin/env python3
"""Set up Japan MLIT and Tunisia ML Roads datasets.

Downloads are expected to already be in data/raw/staging/:
  - N06-24_GML.zip (Japan expressways)
  - N10-24_13_GML.zip (Japan emergency roads, Tokyo)
  - Northern_Africa.zip (MS Road Detections)

Usage:
    python scripts/setup_new_datasets.py japan
    python scripts/setup_new_datasets.py tunisia
    python scripts/setup_new_datasets.py both
"""

import json
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import h3
import pandas as pd
import shapely
from shapely.geometry import shape

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGING = PROJECT_ROOT / "data" / "raw" / "staging"
OUTPUT = PROJECT_ROOT / "data" / "raw"

H3_RESOLUTION = 8
H3_TRAILING_F_COUNT = 5


def compute_spatial_suffix(geom) -> str:
    midpoint = shapely.line_interpolate_point(geom, 0.5, normalized=True)
    h3_index = h3.latlng_to_cell(midpoint.y, midpoint.x, H3_RESOLUTION)
    return h3_index[:-H3_TRAILING_F_COUNT]


def make_trivial_lr(value):
    return [{"between": [0.0, 1.0], "value": value}]


def to_overture_schema(
    gdf: gpd.GeoDataFrame,
    *,
    id_prefix: str,
    id_col: str,
    source_name: str,
    name_col: str | None = None,
    class_col: str | None = None,
    class_mapping: dict | None = None,
    default_class: str = "unknown",
) -> gpd.GeoDataFrame:
    """Convert a GeoDataFrame to the matcher's Overture-compatible schema."""
    # Filter to LineString only
    is_line = gdf.geometry.geom_type == "LineString"
    if not is_line.all():
        multi = gdf.geometry.geom_type == "MultiLineString"
        single_part = multi & (gdf.geometry.apply(lambda g: len(g.geoms)) == 1)
        gdf.loc[single_part, "geometry"] = gdf.loc[single_part, "geometry"].apply(
            lambda g: g.geoms[0]
        )
        is_line = gdf.geometry.geom_type == "LineString"
        dropped = (~is_line).sum()
        if dropped:
            print(f"  Dropping {dropped} non-LineString geometries")
        gdf = gdf[is_line].copy()

    # Force 2D
    gdf["geometry"] = shapely.force_2d(gdf.geometry.values)

    # Store original columns as source_tags
    original_cols = [c for c in gdf.columns if c != "geometry"]
    source_tags = gdf[original_cols].to_dict(orient="records")

    # Compute spatial suffixes
    suffixes = [compute_spatial_suffix(g) for g in gdf.geometry]

    # Build IDs
    ids = [f"{id_prefix}_{uid}_{sfx}" for uid, sfx in zip(gdf[id_col].astype(str), suffixes)]

    # Names
    if name_col and name_col in gdf.columns:
        names = [{"primary": str(v)} if pd.notna(v) and v else None for v in gdf[name_col]]
    else:
        names = [None] * len(gdf)

    # Class
    if class_col and class_col in gdf.columns and class_mapping:
        normalized_mapping = {str(k): v for k, v in class_mapping.items()}
        classes = [
            normalized_mapping.get(str(v), default_class) if pd.notna(v) else default_class
            for v in gdf[class_col]
        ]
    else:
        classes = [default_class] * len(gdf)

    # Sources
    sources = [[{"dataset": source_name, "record_id": str(rid)}] for rid in gdf[id_col]]

    # Build result
    data = {
        "id": ids,
        "subtype": ["road"] * len(gdf),
        "sources": sources,
        "road_flags": [[] for _ in range(len(gdf))],
        "level_rules": [[] for _ in range(len(gdf))],
        "source_tags": source_tags,
        "names": names,
        "class": classes,
        "subclass": [None] * len(gdf),
        "oneway": [None] * len(gdf),
        "speed_limit_kph": [None] * len(gdf),
    }

    result = gpd.GeoDataFrame(data, geometry=gdf.geometry.values, crs="EPSG:4326")

    # Add linear-referenced columns
    result["names_lr"] = [make_trivial_lr(n["primary"] if n else None) for n in names]
    result["subclass_lr"] = [make_trivial_lr(None)] * len(result)
    result["level_lr"] = [make_trivial_lr(0)] * len(result)
    result["road_flags_lr"] = [make_trivial_lr([])] * len(result)
    result["oneway_lr"] = [make_trivial_lr(None)] * len(result)
    result["speed_limit_kph_lr"] = [make_trivial_lr(None)] * len(result)

    # Deduplicate by ID
    dupes = result["id"].duplicated(keep="first")
    if dupes.any():
        print(f"  Removing {dupes.sum()} duplicate IDs")
        result = result[~dupes].copy()

    return result


def setup_japan():
    """Combine N06 expressways + N10 emergency roads for Tokyo/Kanto area."""
    print("=== Setting up Japan MLIT dataset ===")

    kanto_bbox = (139.0, 35.2, 140.5, 36.2)

    # Load N06 expressways
    n06_path = STAGING / "UTF-8" / "N06-24_HighwaySection.geojson"
    if not n06_path.exists():
        print("Extracting N06 GeoJSON...")
        with zipfile.ZipFile(STAGING / "N06-24_GML.zip") as zf:
            zf.extract("UTF-8/N06-24_HighwaySection.geojson", STAGING)

    print("Loading N06 expressways...")
    n06 = gpd.read_file(n06_path)
    # Normalize CRS to WGS84 (N06 uses JGD2011 which is ~identical but technically different)
    if n06.crs and n06.crs.to_epsg() != 4326:
        n06 = n06.to_crs("EPSG:4326")
    print(f"  Total N06: {len(n06)}")

    # Filter to Kanto area (use representative point to avoid geographic CRS warning)
    centroids = n06.geometry.representative_point()
    in_kanto = (
        (centroids.x >= kanto_bbox[0])
        & (centroids.x <= kanto_bbox[2])
        & (centroids.y >= kanto_bbox[1])
        & (centroids.y <= kanto_bbox[3])
    )
    n06_kanto = n06[in_kanto].copy()
    print(f"  N06 in Kanto: {len(n06_kanto)}")

    # Normalize columns
    n06_kanto = n06_kanto.rename(
        columns={
            "N06_007": "road_name",
            "N06_003": "road_type_code",
        }
    )
    n06_kanto["source_dataset"] = "N06_expressways"
    n06_kanto["road_class"] = "motorway"  # All N06 are expressways
    n06_kanto["orig_id"] = [f"N06_{i}" for i in range(len(n06_kanto))]

    # Load N10 emergency roads (Tokyo)
    n10_path = STAGING / "N10-24_13_GML" / "N10-24_13.geojson"
    if not n10_path.exists():
        print("Extracting N10 GeoJSON...")
        with zipfile.ZipFile(STAGING / "N10-24_13_GML.zip") as zf:
            zf.extractall(STAGING)

    print("Loading N10 emergency roads (Tokyo)...")
    n10 = gpd.read_file(n10_path)
    print(f"  Total N10: {len(n10)}")

    # Map N10 road classification
    n10_class_mapping = {
        1: "motorway",
        2: "trunk",
        3: "secondary",
        9: "unknown",
    }
    n10 = n10.rename(
        columns={
            "N01_004": "road_name",
            "N01_002": "road_class_code",
        }
    )
    n10["source_dataset"] = "N10_emergency"
    n10["road_class"] = n10["road_class_code"].map(n10_class_mapping).fillna("unclassified")
    n10["orig_id"] = [f"N10_{i}" for i in range(len(n10))]

    # Combine
    keep_cols = ["road_name", "road_class", "source_dataset", "orig_id", "geometry"]
    combined = pd.concat(
        [n06_kanto[keep_cols], n10[keep_cols]],
        ignore_index=True,
    )
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
    print(f"  Combined: {len(combined)} segments")

    # Compute bbox
    bounds = combined.total_bounds
    print(f"  Bbox: [{bounds[0]:.4f}, {bounds[1]:.4f}, {bounds[2]:.4f}, {bounds[3]:.4f}]")

    # Transform to Overture schema
    result = to_overture_schema(
        combined,
        id_prefix="jp_tokyo",
        id_col="orig_id",
        source_name="jp_tokyo_roads",
        name_col="road_name",
        class_col="road_class",
        class_mapping=None,  # Already mapped
        default_class="unknown",
    )

    # The class column is already text, just pass through
    result["class"] = combined.loc[result.index, "road_class"].values

    out_path = OUTPUT / "jp_tokyo_roads_v1.0.parquet"
    result.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path} ({len(result)} features)")

    # Print sample
    print("\n  Sample names:")
    for _, row in result.head(5).iterrows():
        name = row["names"]["primary"] if row["names"] else "(unnamed)"
        print(f"    {row['id']}: {name} [{row['class']}]")

    return {
        "feature_count": len(result),
        "bbox": [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])],
    }


def setup_tunisia():
    """Extract Tunisia from MS Road Detections and clip to Tunis area."""
    print("=== Setting up Tunisia ML Roads dataset ===")

    # Tunis greater metro area
    tunis_bbox = (9.8, 36.5, 10.5, 37.1)

    zip_path = Path("Northern_Africa.zip")
    if not zip_path.exists():
        zip_path = STAGING / "Northern_Africa.zip"
    if not zip_path.exists():
        print("ERROR: Northern_Africa.zip not found")
        sys.exit(1)

    print(f"Reading Tunisia records from {zip_path}...")

    features = []
    total_tun = 0

    with zipfile.ZipFile(zip_path) as zf:
        # Find the TSV file inside
        tsv_files = [f for f in zf.namelist() if f.endswith(".tsv")]
        if not tsv_files:
            # Sometimes it's just a single file without .tsv extension
            tsv_files = [f for f in zf.namelist() if not f.endswith("/")]

        print(f"  Files in ZIP: {zf.namelist()[:10]}")

        for tsv_file in tsv_files:
            print(f"  Processing {tsv_file}...")
            with zf.open(tsv_file) as f:
                for line_bytes in f:
                    line = line_bytes.decode("utf-8").strip()
                    if not line:
                        continue
                    parts = line.split("\t", 1)
                    if len(parts) != 2:
                        continue
                    country_code, geojson_str = parts
                    if country_code != "TUN":
                        continue
                    total_tun += 1

                    try:
                        feature = json.loads(geojson_str)
                        geom = shape(feature["geometry"])
                        if geom.geom_type != "LineString":
                            continue

                        # Clip to Tunis bbox
                        centroid = geom.centroid
                        if not (
                            tunis_bbox[0] <= centroid.x <= tunis_bbox[2]
                            and tunis_bbox[1] <= centroid.y <= tunis_bbox[3]
                        ):
                            continue

                        props = feature.get("properties", {})
                        features.append(
                            {
                                "geometry": geom,
                                "width_m": props.get("WidthMeters"),
                            }
                        )
                    except (json.JSONDecodeError, Exception):
                        continue

    print(f"  Total TUN records: {total_tun}")
    print(f"  In Tunis bbox: {len(features)}")

    if not features:
        print("ERROR: No features found in Tunis area")
        sys.exit(1)

    # Build GeoDataFrame
    gdf = gpd.GeoDataFrame(
        features,
        geometry="geometry",
        crs="EPSG:4326",
    )
    gdf["orig_id"] = [f"TUN_{i}" for i in range(len(gdf))]

    bounds = gdf.total_bounds
    print(f"  Bbox: [{bounds[0]:.4f}, {bounds[1]:.4f}, {bounds[2]:.4f}, {bounds[3]:.4f}]")

    # Transform to Overture schema
    result = to_overture_schema(
        gdf,
        id_prefix="tn_tunis_ml",
        id_col="orig_id",
        source_name="tn_tunis_ml_roads",
        name_col=None,  # No names in ML-extracted data
        class_col=None,
        default_class="unknown",
    )

    out_path = OUTPUT / "tn_tunis_ml_roads_v1.0.parquet"
    result.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path} ({len(result)} features)")

    return {
        "feature_count": len(result),
        "bbox": [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])],
    }


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "both"

    if target in ("japan", "both"):
        japan_info = setup_japan()
        print(f"\nJapan: {japan_info['feature_count']} features, bbox={japan_info['bbox']}")

    if target in ("tunisia", "both"):
        tunisia_info = setup_tunisia()
        print(f"\nTunisia: {tunisia_info['feature_count']} features, bbox={tunisia_info['bbox']}")

    print("\nDone! Next steps:")
    print("  1. Create dataset YAML configs in datasets/")
    print("  2. Run: matcher data fetch reference <dataset_name>")
