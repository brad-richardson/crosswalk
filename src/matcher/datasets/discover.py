"""Dataset class discovery and mapping generation.

This module provides utilities to analyze a dataset and automatically
discover potential class mappings to Overture's road hierarchy.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger

from .config import (
    ClassMappingRule,
    DatasetConfig,
    PhysicalAttributes,
    SourceClassification,
    save_dataset_config,
)

# Common column name patterns for classification
CLASS_COLUMN_PATTERNS = [
    "class",
    "type",
    "category",
    "road_type",
    "roadtype",
    "highway",
    "fclass",
    "f_class",
    "func_class",
    "functional_class",
    "road_class",
    "classification",
]

# Common column name patterns for physical attributes
LANES_PATTERNS = ["lanes", "num_lanes", "lane_count", "nlanes"]
SPEED_PATTERNS = ["speed", "speed_limit", "maxspeed", "speed_lim", "speedlimit"]
WIDTH_PATTERNS = ["width", "road_width", "surface_width", "surface_wd", "roadwidth"]
TRAFFIC_PATTERNS = ["aadt", "adt", "traffic", "volume", "daily_traffic"]

# Overture road class hierarchy (ordered by importance)
OVERTURE_CLASSES = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "service",
    "unclassified",
    "living_street",
    "track",
]


@dataclass
class ColumnAnalysis:
    """Analysis of a single column."""

    name: str
    dtype: str
    non_null_count: int
    unique_values: list[Any]
    value_counts: dict[Any, int]
    is_classification: bool = False
    is_physical_attr: bool = False
    physical_attr_type: str | None = None  # lanes, speed, width, traffic


@dataclass
class DiscoveryReport:
    """Results of dataset analysis."""

    dataset_path: str
    total_rows: int
    columns: list[ColumnAnalysis]
    detected_class_column: str | None = None
    detected_physical_attrs: dict[str, str] = field(default_factory=dict)
    source_tags_analysis: dict | None = None
    match_analysis: dict | None = None
    suggested_config: DatasetConfig | None = None


def _find_column_by_patterns(columns: list[str], patterns: list[str]) -> str | None:
    """Find a column matching any of the patterns."""
    columns_lower = {c.lower(): c for c in columns}
    for pattern in patterns:
        if pattern in columns_lower:
            return columns_lower[pattern]
    return None


def _analyze_source_tags(gdf: gpd.GeoDataFrame, source_tags_col: str = "source_tags") -> dict:
    """Analyze the source_tags column if present."""
    if source_tags_col not in gdf.columns:
        return {}

    # Collect all keys and sample values
    all_keys: dict[str, dict] = {}

    for tags in gdf[source_tags_col].dropna().head(1000):
        if not isinstance(tags, dict):
            continue
        for key, value in tags.items():
            if key not in all_keys:
                all_keys[key] = {"count": 0, "sample_values": set(), "dtype": None}
            all_keys[key]["count"] += 1
            if len(all_keys[key]["sample_values"]) < 10:
                all_keys[key]["sample_values"].add(str(value)[:50])
            if all_keys[key]["dtype"] is None:
                all_keys[key]["dtype"] = type(value).__name__

    # Convert sample_values sets to lists
    for key in all_keys:
        all_keys[key]["sample_values"] = list(all_keys[key]["sample_values"])

    # Try to identify classification and physical attribute columns
    result = {
        "keys": all_keys,
        "detected_class_key": None,
        "detected_physical_attrs": {},
    }

    keys_lower = {k.lower(): k for k in all_keys}

    # Find classification column
    for pattern in CLASS_COLUMN_PATTERNS:
        if pattern in keys_lower:
            result["detected_class_key"] = keys_lower[pattern]
            break

    # Find physical attributes
    for pattern in LANES_PATTERNS:
        if pattern in keys_lower:
            result["detected_physical_attrs"]["lanes"] = keys_lower[pattern]
            break
    for pattern in SPEED_PATTERNS:
        if pattern in keys_lower:
            result["detected_physical_attrs"]["speed"] = keys_lower[pattern]
            break
    for pattern in WIDTH_PATTERNS:
        if pattern in keys_lower:
            result["detected_physical_attrs"]["width"] = keys_lower[pattern]
            break
    for pattern in TRAFFIC_PATTERNS:
        if pattern in keys_lower:
            result["detected_physical_attrs"]["traffic"] = keys_lower[pattern]
            break

    return result


def _analyze_with_reference(
    gdf: gpd.GeoDataFrame,
    reference: gpd.GeoDataFrame,
    bridge_path: Path | None,
    source_class_col: str,
) -> dict:
    """Analyze class mapping using matched pairs if available."""
    result = {
        "total_matches": 0,
        "name_verified_matches": 0,
        "confusion_matrix": {},
    }

    if bridge_path is None or not bridge_path.exists():
        return result

    try:
        from rapidfuzz import fuzz

        bridge = pd.read_parquet(bridge_path)

        # Filter to 1:1 high-confidence matches
        matches = bridge[(bridge["match_type"] == "1:1") & (bridge["confidence"] >= 0.8)]
        result["total_matches"] = len(matches)

        if len(matches) == 0:
            return result

        # Validate bridge has required columns
        required_bridge_cols = {"local_id", "gers_id", "match_type", "confidence"}
        missing_cols = required_bridge_cols - set(bridge.columns)
        if missing_cols:
            logger.warning(f"Bridge file missing required columns: {missing_cols}")
            return result

        # Validate gdf has required columns
        if "id" not in gdf.columns:
            logger.warning("Dataset missing 'id' column for merge")
            return result

        # Use the full composite ID for joining with the bridge file.
        # Bridge local_id stores the full target ID (e.g., "prefix_123_882a306603").
        gdf = gdf.copy()
        gdf["local_id"] = gdf["id"].astype(str)

        # Join with source data
        matches = matches.merge(
            gdf[["local_id", source_class_col, "names"]].rename(
                columns={
                    source_class_col: "source_class",
                    "names": "source_name",
                }
            ),
            on="local_id",
            how="left",
        )

        # Join with reference data
        matches = matches.merge(
            reference[["id", "class", "names"]].rename(
                columns={"id": "gers_id", "class": "ref_class", "names": "ref_name"}
            ),
            on="gers_id",
            how="left",
        )

        # Extract names for comparison
        def get_name(names):
            if names and isinstance(names, dict):
                return names.get("primary", "").lower()
            return str(names).lower() if names else ""

        matches["source_name_str"] = matches["source_name"].apply(get_name)
        matches["ref_name_str"] = matches["ref_name"].apply(get_name)

        # Compute name similarity
        def name_sim(row):
            if not row["source_name_str"] or not row["ref_name_str"]:
                return 0
            return fuzz.token_sort_ratio(row["source_name_str"], row["ref_name_str"]) / 100.0

        matches["name_sim"] = matches.apply(name_sim, axis=1)

        # Filter to name-verified matches
        name_verified = matches[matches["name_sim"] >= 0.7]
        result["name_verified_matches"] = len(name_verified)

        # Build confusion matrix from name-verified matches
        if len(name_verified) > 0:
            confusion = pd.crosstab(
                name_verified["source_class"].fillna("unknown"),
                name_verified["ref_class"].fillna("unknown"),
            )
            result["confusion_matrix"] = confusion.to_dict()

            # Calculate suggested mapping (most common target for each source)
            result["suggested_mapping"] = {}
            for source_val in name_verified["source_class"].unique():
                if pd.isna(source_val):
                    continue
                subset = name_verified[name_verified["source_class"] == source_val]
                if len(subset) > 0:
                    top_class = subset["ref_class"].mode()
                    if len(top_class) > 0:
                        pct = (subset["ref_class"] == top_class.iloc[0]).mean() * 100
                        result["suggested_mapping"][source_val] = {
                            "target": top_class.iloc[0],
                            "confidence": pct,
                            "sample_size": len(subset),
                        }

    except Exception as e:
        logger.warning(f"Error analyzing with reference: {e}")

    return result


def discover_dataset(
    dataset_path: Path,
    reference_path: Path | None = None,
    bridge_path: Path | None = None,
    source_tags_col: str = "source_tags",
) -> DiscoveryReport:
    """Analyze a dataset and discover class mapping.

    Args:
        dataset_path: Path to the target dataset (parquet/geojson)
        reference_path: Path to Overture reference data
        bridge_path: Path to existing bridge file for match analysis
        source_tags_col: Column containing source attributes dict

    Returns:
        DiscoveryReport with analysis results
    """
    logger.info(f"Analyzing dataset: {dataset_path}")

    # Load dataset
    gdf = gpd.read_parquet(dataset_path)

    report = DiscoveryReport(
        dataset_path=str(dataset_path),
        total_rows=len(gdf),
        columns=[],
    )

    # Analyze each column
    for col in gdf.columns:
        if col == "geometry":
            continue

        try:
            series = gdf[col]
            non_null = series.dropna()

            # Get unique values (limit to 50)
            if len(non_null) > 0:
                unique = non_null.unique()[:50].tolist()
                value_counts = non_null.value_counts().head(20).to_dict()
            else:
                unique = []
                value_counts = {}

            analysis = ColumnAnalysis(
                name=col,
                dtype=str(series.dtype),
                non_null_count=len(non_null),
                unique_values=unique,
                value_counts=value_counts,
            )

            # Check if this looks like a classification column
            col_lower = col.lower()
            if any(p in col_lower for p in CLASS_COLUMN_PATTERNS):
                analysis.is_classification = True

            # Check if this looks like a physical attribute
            if any(p in col_lower for p in LANES_PATTERNS):
                analysis.is_physical_attr = True
                analysis.physical_attr_type = "lanes"
            elif any(p in col_lower for p in SPEED_PATTERNS):
                analysis.is_physical_attr = True
                analysis.physical_attr_type = "speed"
            elif any(p in col_lower for p in WIDTH_PATTERNS):
                analysis.is_physical_attr = True
                analysis.physical_attr_type = "width"
            elif any(p in col_lower for p in TRAFFIC_PATTERNS):
                analysis.is_physical_attr = True
                analysis.physical_attr_type = "traffic"

            report.columns.append(analysis)

        except Exception as e:
            logger.warning(f"Error analyzing column {col}: {e}")

    # Find best classification column
    class_cols = [c for c in report.columns if c.is_classification]
    if class_cols:
        # Prefer 'class' if available, otherwise first match
        report.detected_class_column = next(
            (c.name for c in class_cols if c.name.lower() == "class"),
            class_cols[0].name,
        )

    # Find physical attribute columns
    for col in report.columns:
        if col.is_physical_attr and col.physical_attr_type:
            report.detected_physical_attrs[col.physical_attr_type] = col.name

    # Analyze source_tags if present
    if source_tags_col in gdf.columns:
        report.source_tags_analysis = _analyze_source_tags(gdf, source_tags_col)

        # Use source_tags classification if available and no direct column found
        if (
            report.source_tags_analysis.get("detected_class_key")
            and not report.detected_class_column
        ):
            report.detected_class_column = (
                f"{source_tags_col}.{report.source_tags_analysis['detected_class_key']}"
            )

        # Add physical attrs from source_tags
        for attr_type, key in report.source_tags_analysis.get(
            "detected_physical_attrs", {}
        ).items():
            if attr_type not in report.detected_physical_attrs:
                report.detected_physical_attrs[attr_type] = f"{source_tags_col}.{key}"

    # Analyze with reference if available
    if reference_path and reference_path.exists():
        reference = gpd.read_parquet(reference_path)

        # Validate reference has required columns
        if "id" not in reference.columns or "class" not in reference.columns:
            logger.warning("Reference file missing 'id' or 'class' column, skipping match analysis")
        elif gdf.empty:
            logger.warning("Dataset is empty, skipping match analysis")
        else:
            # Determine source class column for matching
            source_class_col: str | None = None

            if (
                report.source_tags_analysis
                and report.source_tags_analysis.get("detected_class_key")
                and source_tags_col in gdf.columns
            ):
                # Extract classification from source_tags for analysis
                class_key = report.source_tags_analysis["detected_class_key"]
                gdf[f"_source_{class_key}"] = gdf[source_tags_col].apply(
                    lambda t: t.get(class_key) if isinstance(t, dict) else None
                )
                source_class_col = f"_source_{class_key}"
            elif "class" in gdf.columns:
                # Fallback to main class column if present
                source_class_col = "class"

            if source_class_col and source_class_col in gdf.columns:
                report.match_analysis = _analyze_with_reference(
                    gdf, reference, bridge_path, source_class_col
                )
            else:
                logger.warning("Source class column not found in dataset, skipping match analysis")

    # Generate suggested configuration
    report.suggested_config = _generate_config(report, dataset_path)

    return report


def _generate_config(report: DiscoveryReport, dataset_path: Path) -> DatasetConfig:
    """Generate a suggested configuration from the discovery report."""
    name = dataset_path.stem

    # Determine source classification
    source_class = None
    source_col = None

    if report.source_tags_analysis and report.source_tags_analysis.get("detected_class_key"):
        source_col = report.source_tags_analysis["detected_class_key"]
        source_values = {}

        # Get value distribution from source_tags
        if source_col in report.source_tags_analysis.get("keys", {}):
            key_info = report.source_tags_analysis["keys"][source_col]
            for val in key_info.get("sample_values", []):
                source_values[val] = {"description": "auto-detected"}

        source_class = SourceClassification(
            column=source_col,
            description=f"Auto-detected from source_tags.{source_col}",
            values=source_values,
        )
    elif report.detected_class_column:
        source_col = report.detected_class_column
        col_analysis = next((c for c in report.columns if c.name == source_col), None)
        source_values = {}
        if col_analysis:
            for val, count in col_analysis.value_counts.items():
                source_values[val] = {"count": count}

        source_class = SourceClassification(
            column=source_col,
            description="Auto-detected classification column",
            values=source_values,
        )

    # Build physical attributes
    phys_attrs = None
    if report.detected_physical_attrs:
        phys_attrs = PhysicalAttributes(
            lanes_column=report.detected_physical_attrs.get("lanes"),
            speed_column=report.detected_physical_attrs.get("speed"),
            width_column=report.detected_physical_attrs.get("width"),
            traffic_column=report.detected_physical_attrs.get("traffic"),
        )

    # Build mapping rules from match analysis
    rules = []
    confidence = "low"

    if report.match_analysis and report.match_analysis.get("suggested_mapping"):
        mapping = report.match_analysis["suggested_mapping"]
        for source_val, info in mapping.items():
            if info["sample_size"] >= 10 and info["confidence"] >= 30:
                # Convert numpy types to native Python types for YAML serialization
                if hasattr(source_val, "item"):
                    source_val = source_val.item()
                elif isinstance(source_val, np.floating):
                    source_val = float(source_val)
                elif isinstance(source_val, np.integer):
                    source_val = int(source_val)
                rules.append(
                    ClassMappingRule(
                        source_value=source_val,
                        target_class=info["target"],
                        priority=0,
                    )
                )

        # Set confidence based on match quality
        if report.match_analysis.get("name_verified_matches", 0) >= 100:
            confidence = "medium"
        if report.match_analysis.get("name_verified_matches", 0) >= 500:
            confidence = "high"

    # Generate notes
    notes = f"Auto-generated from {report.total_rows} rows.\n"
    if report.match_analysis:
        notes += f"Based on {report.match_analysis.get('name_verified_matches', 0)} name-verified matches.\n"
    notes += "Review and adjust mapping rules as needed."

    return DatasetConfig(
        name=name,
        description=f"Configuration for {name}",
        source_classification=source_class,
        physical_attributes=phys_attrs,
        class_mapping_rules=rules,
        default_class="unclassified",
        confidence=confidence,
        notes=notes,
    )


def print_discovery_report(report: DiscoveryReport) -> None:
    """Print a human-readable discovery report."""
    print("=" * 70)
    print(f"DATASET DISCOVERY REPORT: {Path(report.dataset_path).name}")
    print("=" * 70)
    print(f"\nTotal rows: {report.total_rows}")

    print("\n" + "-" * 70)
    print("DETECTED COLUMNS")
    print("-" * 70)

    # Classification columns
    class_cols = [c for c in report.columns if c.is_classification]
    if class_cols:
        print("\nClassification columns:")
        for col in class_cols:
            print(f"  {col.name}: {list(col.value_counts.keys())[:5]}")

    # Physical attribute columns
    phys_cols = [c for c in report.columns if c.is_physical_attr]
    if phys_cols:
        print("\nPhysical attribute columns:")
        for col in phys_cols:
            print(f"  {col.name} ({col.physical_attr_type})")

    # Source tags analysis
    if report.source_tags_analysis:
        print("\n" + "-" * 70)
        print("SOURCE TAGS ANALYSIS")
        print("-" * 70)

        if report.source_tags_analysis.get("detected_class_key"):
            key = report.source_tags_analysis["detected_class_key"]
            print(f"\nDetected classification key: {key}")
            key_info = report.source_tags_analysis["keys"].get(key, {})
            print(f"  Sample values: {key_info.get('sample_values', [])[:5]}")

        if report.source_tags_analysis.get("detected_physical_attrs"):
            print("\nDetected physical attributes in source_tags:")
            for attr_type, key in report.source_tags_analysis["detected_physical_attrs"].items():
                print(f"  {attr_type}: {key}")

    # Match analysis
    if report.match_analysis:
        print("\n" + "-" * 70)
        print("MATCH-BASED ANALYSIS")
        print("-" * 70)
        print(f"\nTotal high-confidence matches: {report.match_analysis.get('total_matches', 0)}")
        print(f"Name-verified matches: {report.match_analysis.get('name_verified_matches', 0)}")

        if report.match_analysis.get("suggested_mapping"):
            print("\nSuggested class mapping:")
            for source_val, info in report.match_analysis["suggested_mapping"].items():
                print(
                    f"  {source_val} → {info['target']} ({info['confidence']:.1f}% of {info['sample_size']} matches)"
                )

    # Suggested configuration
    if report.suggested_config:
        print("\n" + "-" * 70)
        print("SUGGESTED CONFIGURATION")
        print("-" * 70)
        print(f"\nConfidence: {report.suggested_config.confidence}")
        print(f"Default class: {report.suggested_config.default_class}")

        if report.suggested_config.class_mapping_rules:
            print("\nMapping rules:")
            for rule in report.suggested_config.class_mapping_rules:
                cond_str = f" (when {rule.conditions})" if rule.conditions else ""
                print(f"  {rule.source_value} → {rule.target_class}{cond_str}")


def discover_and_save(
    dataset_path: Path,
    output_path: Path | None = None,
    reference_path: Path | None = None,
    bridge_path: Path | None = None,
) -> Path:
    """Discover dataset classes and save configuration.

    Args:
        dataset_path: Path to target dataset
        output_path: Path for output YAML (default: datasets/{name}.yaml)
        reference_path: Path to Overture reference data
        bridge_path: Path to bridge file

    Returns:
        Path to saved configuration
    """
    report = discover_dataset(dataset_path, reference_path, bridge_path)
    print_discovery_report(report)

    if report.suggested_config:
        saved_path = save_dataset_config(report.suggested_config, output_path)
        print(f"\nConfiguration saved to: {saved_path}")
        return saved_path

    raise ValueError("Could not generate configuration from discovery")
