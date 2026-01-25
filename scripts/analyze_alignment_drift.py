#!/usr/bin/env python3
"""Analyze alignment drift between stored labels and recomputed alignment.

This script compares the alignment fractions stored in label files with
freshly recomputed values using the current algorithm. It helps identify:
1. Cases where the divergence detection correctly truncates bad alignments
2. Potential regressions where good alignments are incorrectly truncated
3. Overall impact of algorithm changes on the labeled dataset

Usage:
    python scripts/analyze_alignment_drift.py [--threshold 0.05] [--output drift.csv]
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString
from shapely.ops import transform
from tqdm import tqdm

from matcher.features.alignment import linestring_alignment


def load_label_file(label_path: Path) -> pd.DataFrame | None:
    """Load a label CSV file."""
    try:
        return pd.read_csv(label_path)
    except Exception as e:
        print(f"Error loading {label_path}: {e}")
        return None


def load_geometries(
    data_dir: Path, dataset: str
) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]:
    """Load reference and target geometries for a dataset."""
    ref_path = data_dir / f"{dataset}_overture_segments_v1.0.parquet"
    target_path = data_dir / f"{dataset}_v1.0.parquet"

    if not ref_path.exists() or not target_path.exists():
        return None, None

    try:
        ref_gdf = gpd.read_parquet(ref_path)
        target_gdf = gpd.read_parquet(target_path)
        return ref_gdf, target_gdf
    except Exception as e:
        print(f"Error loading geometries for {dataset}: {e}")
        return None, None


def get_geometry(gdf: gpd.GeoDataFrame, feature_id: str) -> LineString | None:
    """Get a geometry by ID from a GeoDataFrame."""
    for id_col in ["id", "gers_id", "OBJECTID"]:
        if id_col in gdf.columns:
            mask = gdf[id_col].astype(str) == str(feature_id)
            if mask.any():
                geom = gdf.loc[mask, "geometry"].iloc[0]
                if isinstance(geom, LineString):
                    return geom
    return None


def project_geometry(geom: LineString, center_lon: float, center_lat: float) -> LineString:
    """Project a geometry to a local CRS centered on the given point."""
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +datum=WGS84 +units=m"
    )
    transformer = Transformer.from_crs(CRS.from_epsg(4326), local_crs, always_xy=True)
    return transform(transformer.transform, geom)


def analyze_alignment_drift(
    labels_dir: Path,
    data_dir: Path,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Compare stored vs recomputed alignment for all labels.

    Args:
        labels_dir: Path to labels directory (Hive-partitioned)
        data_dir: Path to raw data directory with parquet files
        threshold: Minimum drift to include in results (default 0.05 = 5%)

    Returns:
        DataFrame with alignment drift analysis
    """
    results = []

    # Find all dataset label files
    label_files = list(labels_dir.glob("dataset=*/data.csv"))
    print(f"Found {len(label_files)} label files")

    for label_file in label_files:
        dataset = label_file.parent.name.replace("dataset=", "")
        print(f"\nProcessing {dataset}...")

        # Load labels
        df = load_label_file(label_file)
        if df is None:
            continue

        # Check if alignment columns exist
        alignment_cols = ["ref_start_pct", "ref_end_pct", "target_start_pct", "target_end_pct"]
        has_alignment = all(col in df.columns for col in alignment_cols)

        if not has_alignment:
            # Try alternative column names
            alt_cols = ["ref_coverage", "target_coverage"]
            if not all(col in df.columns for col in alt_cols):
                print(f"  Skipping {dataset}: no alignment columns found")
                continue

        # Load geometries
        ref_gdf, target_gdf = load_geometries(data_dir, dataset)
        if ref_gdf is None or target_gdf is None:
            print(f"  Skipping {dataset}: geometry files not found")
            continue

        # Process each labeled pair
        processed = 0
        skipped = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {dataset}"):
            ref_id = row["gers_id"]
            target_id = row["target_id"]

            # Get stored alignment fractions
            if has_alignment:
                stored = {
                    "ref_start": row.get("ref_start_pct", 0),
                    "ref_end": row.get("ref_end_pct", 1),
                    "target_start": row.get("target_start_pct", 0),
                    "target_end": row.get("target_end_pct", 1),
                }
            else:
                # Infer from coverage (assume full alignment)
                stored = {
                    "ref_start": 0,
                    "ref_end": 1,
                    "target_start": 0,
                    "target_end": 1,
                }

            # Load geometries
            ref_geom = get_geometry(ref_gdf, ref_id)
            target_geom = get_geometry(target_gdf, target_id)

            if ref_geom is None or target_geom is None:
                skipped += 1
                continue

            # Project to local CRS
            centroid = ref_geom.centroid
            ref_proj = project_geometry(ref_geom, centroid.x, centroid.y)
            target_proj = project_geometry(target_geom, centroid.x, centroid.y)

            # Recompute alignment
            try:
                new_alignment = linestring_alignment(ref_proj, target_proj)
            except Exception:
                skipped += 1
                continue

            # Compute drifts
            drifts = {
                "ref_start_drift": abs(new_alignment.overture_start_frac - stored["ref_start"]),
                "ref_end_drift": abs(new_alignment.overture_end_frac - stored["ref_end"]),
                "target_start_drift": abs(
                    new_alignment.dataset_start_frac - stored["target_start"]
                ),
                "target_end_drift": abs(new_alignment.dataset_end_frac - stored["target_end"]),
            }

            max_drift = max(drifts.values())
            processed += 1

            # Only record if drift exceeds threshold
            if max_drift > threshold:
                # Compute coverage changes
                stored_ref_coverage = stored["ref_end"] - stored["ref_start"]
                stored_target_coverage = stored["target_end"] - stored["target_start"]
                new_ref_coverage = new_alignment.overture_coverage
                new_target_coverage = new_alignment.dataset_coverage

                results.append(
                    {
                        "dataset": dataset,
                        "ref_id": ref_id,
                        "target_id": target_id,
                        "label": row.get("label", "unknown"),
                        # Stored values
                        "stored_ref_start": stored["ref_start"],
                        "stored_ref_end": stored["ref_end"],
                        "stored_target_start": stored["target_start"],
                        "stored_target_end": stored["target_end"],
                        "stored_ref_coverage": stored_ref_coverage,
                        "stored_target_coverage": stored_target_coverage,
                        # New values
                        "new_ref_start": new_alignment.overture_start_frac,
                        "new_ref_end": new_alignment.overture_end_frac,
                        "new_target_start": new_alignment.dataset_start_frac,
                        "new_target_end": new_alignment.dataset_end_frac,
                        "new_ref_coverage": new_ref_coverage,
                        "new_target_coverage": new_target_coverage,
                        # Drifts
                        **drifts,
                        "max_drift": max_drift,
                        "coverage_change": new_ref_coverage - stored_ref_coverage,
                        # Quality metrics from labels
                        "hausdorff_m": row.get("hausdorff_distance_m"),
                        "mean_hausdorff_m": row.get("mean_hausdorff_distance_m"),
                        "buffer_iou_5m": row.get("buffer_iou_5m"),
                    }
                )

        print(f"  Processed {processed}, skipped {skipped}")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Analyze alignment drift in labels")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("labels"),
        help="Path to labels directory",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw data directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Minimum drift to include in results (default: 0.05)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("alignment_drift_analysis.csv"),
        help="Output CSV file path",
    )
    args = parser.parse_args()

    print(f"Analyzing alignment drift (threshold={args.threshold})")
    print(f"Labels: {args.labels_dir}")
    print(f"Data: {args.data_dir}")

    drift_df = analyze_alignment_drift(
        args.labels_dir,
        args.data_dir,
        threshold=args.threshold,
    )

    if len(drift_df) > 0:
        # Save results
        drift_df.to_csv(args.output, index=False)
        print(f"\nSaved results to {args.output}")

        # Print summary
        print(f"\nFound {len(drift_df)} examples with drift > {args.threshold}")

        print("\nBy dataset:")
        print(drift_df.groupby("dataset").size().to_string())

        print("\nBy label:")
        print(drift_df.groupby("label").size().to_string())

        print("\nCoverage change statistics:")
        print(f"  Mean: {drift_df['coverage_change'].mean():.3f}")
        print(f"  Std:  {drift_df['coverage_change'].std():.3f}")
        print(f"  Min:  {drift_df['coverage_change'].min():.3f}")
        print(f"  Max:  {drift_df['coverage_change'].max():.3f}")

        # Identify potential regressions (matches with significant coverage drop)
        regressions = drift_df[
            (drift_df["label"] == "match") & (drift_df["coverage_change"] < -0.1)
        ]
        if len(regressions) > 0:
            print(f"\nPotential regressions (matches with >10% coverage drop): {len(regressions)}")
            for _, row in regressions.head(5).iterrows():
                print(
                    f"  {row['dataset']}: {row['ref_id'][:8]}... "
                    f"coverage: {row['stored_ref_coverage']:.2f} -> {row['new_ref_coverage']:.2f}"
                )
    else:
        print(f"\nNo examples with drift > {args.threshold} found")


if __name__ == "__main__":
    main()
