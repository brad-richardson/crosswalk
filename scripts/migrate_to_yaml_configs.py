#!/usr/bin/env python3
"""Migrate dataset_configs.py and datasets.csv to unified YAML configs.

This script:
1. Reads dataset configurations from scripts/dataset_configs.py
2. Reads display info from datasets.csv
3. Creates unified YAML files in datasets/ directory
4. Preserves all existing metadata and mappings

Run from repo root:
    python scripts/migrate_to_yaml_configs.py
"""

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

# Import the legacy configs
from dataset_configs import ALL_DATASETS, CITY_BBOXES


def load_datasets_csv() -> dict[str, dict]:
    """Load display info from datasets.csv."""
    csv_path = Path(__file__).parent.parent / "datasets.csv"
    if not csv_path.exists():
        return {}

    result = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset_id = row["dataset_id"]
            metadata = {}
            if row.get("metadata"):
                try:
                    metadata = json.loads(row["metadata"])
                except json.JSONDecodeError:
                    pass
            result[dataset_id] = {
                "display_name": row.get("name", ""),
                "type": row.get("type", "road"),
                "fetch_url": row.get("fetch_url", ""),
                "info_url": row.get("info_url", ""),
                "metadata": metadata,
            }
    return result


def convert_class_mapping(class_mapping: dict | None) -> list[dict] | None:
    """Convert simple class_mapping dict to list of rule dicts."""
    if not class_mapping:
        return None

    rules = []
    for source_val, target_class in class_mapping.items():
        rules.append({
            "source_value": source_val,
            "target_class": target_class,
        })
    return rules


def migrate_dataset(ds: dict, city: str, csv_info: dict[str, dict]) -> dict:
    """Convert a single dataset from legacy format to new config dict."""
    name = ds["name"]

    # Get display info from CSV if available
    csv_data = csv_info.get(name, {})

    # Determine dataset type
    dataset_type = csv_data.get("type", "road")
    if not dataset_type:
        if "sidewalk" in name.lower():
            dataset_type = "sidewalk"
        elif "bike" in name.lower() or "trail" in name.lower():
            dataset_type = "bike"
        else:
            dataset_type = "road"

    # Build config dict
    config: dict[str, Any] = {
        "name": name,
        "display_name": csv_data.get("display_name") or ds.get("source_name") or name.replace("_", " ").title(),
        "type": dataset_type,
    }

    if ds.get("description"):
        config["description"] = ds["description"]

    # Source config
    source: dict[str, Any] = {
        "type": ds.get("fetch_type", "arcgis"),
    }
    if ds.get("url"):
        source["url"] = ds["url"]
    if ds.get("portal_url"):
        source["portal_url"] = ds["portal_url"]
    if ds.get("file_format"):
        source["file_format"] = ds["file_format"]
    if ds.get("where_clause"):
        source["where_clause"] = ds["where_clause"]
    if ds.get("api_key_env_var"):
        source["api_key_env_var"] = ds["api_key_env_var"]
    if ds.get("api_key_header"):
        source["api_key_header"] = ds["api_key_header"]

    config["source"] = source

    # Fetch config
    bbox = ds.get("bbox") or CITY_BBOXES.get(city)
    fetch: dict[str, Any] = {
        "crs": ds.get("crs", "EPSG:4326"),
    }
    if ds.get("id_prefix"):
        fetch["id_prefix"] = ds["id_prefix"]
    if ds.get("name_column"):
        fetch["name_column"] = ds["name_column"]
    if ds.get("class_column"):
        fetch["class_column"] = ds["class_column"]
    if ds.get("class_mapping"):
        # Store as simple dict, not rules
        fetch["class_mapping"] = ds["class_mapping"]
    if ds.get("subclass_column"):
        fetch["subclass_column"] = ds["subclass_column"]
    if ds.get("subclass_mapping"):
        fetch["subclass_mapping"] = ds["subclass_mapping"]
    if ds.get("level_column"):
        fetch["level_column"] = ds["level_column"]
    if bbox:
        fetch["bbox"] = list(bbox)

    config["fetch"] = fetch

    # Classification config if we have mapping rules
    if ds.get("class_column"):
        classification: dict[str, Any] = {
            "source_classification": {
                "column": ds["class_column"],
            },
        }
        rules = convert_class_mapping(ds.get("class_mapping"))
        if rules:
            classification["class_mapping_rules"] = rules
        config["classification"] = classification

    if ds.get("notes"):
        config["notes"] = ds["notes"]

    return config


def save_config(config: dict, path: Path) -> None:
    """Save config dict to YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def main():
    """Run the migration."""
    # Load existing CSV data
    csv_info = load_datasets_csv()
    print(f"Loaded {len(csv_info)} entries from datasets.csv")

    # Get output directory
    datasets_dir = Path(__file__).parent.parent / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    print(f"Writing YAML configs to: {datasets_dir}")

    # Track what we migrated
    migrated = []
    skipped = []

    # Migrate all datasets from dataset_configs.py
    for city, datasets in ALL_DATASETS.items():
        for ds in datasets:
            name = ds["name"]

            # Skip datasets without URLs (manual download only)
            if not ds.get("url") and ds.get("fetch_type") == "manual":
                print(f"  Skipping {name} (manual download, no URL)")
                skipped.append(name)
                continue

            config = migrate_dataset(ds, city, csv_info)

            # Save to YAML
            config_path = datasets_dir / f"{name}.yaml"
            save_config(config, config_path)
            print(f"  Created {config_path.name}")
            migrated.append(name)

    # Add any datasets from CSV that weren't in dataset_configs.py
    for dataset_id, info in csv_info.items():
        if dataset_id not in migrated and dataset_id not in skipped:
            # Create minimal config from CSV info only
            config: dict[str, Any] = {
                "name": dataset_id,
                "display_name": info.get("display_name") or dataset_id.replace("_", " ").title(),
                "type": info.get("type", "road"),
            }

            source: dict[str, Any] = {
                "type": "arcgis" if info.get("fetch_url") else "unknown",
            }
            if info.get("fetch_url"):
                source["url"] = info["fetch_url"]
            if info.get("info_url"):
                source["portal_url"] = info["info_url"]
            config["source"] = source

            fetch: dict[str, Any] = {
                "crs": info.get("metadata", {}).get("crs", "EPSG:4326"),
            }
            if info.get("metadata", {}).get("classification_column"):
                fetch["class_column"] = info["metadata"]["classification_column"]
            config["fetch"] = fetch

            config_path = datasets_dir / f"{dataset_id}.yaml"
            save_config(config, config_path)
            print(f"  Created {config_path.name} (from CSV only)")
            migrated.append(dataset_id)

    print(f"\nMigration complete!")
    print(f"  Migrated: {len(migrated)} datasets")
    print(f"  Skipped: {len(skipped)} datasets (manual download)")
    print(f"\nYAML configs saved to: {datasets_dir}")


if __name__ == "__main__":
    main()
