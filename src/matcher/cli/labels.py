"""Labels management CLI commands.

This module provides CLI commands for managing label data, including:
- migrate: Migrate from legacy embedded-feature format to normalized format
- backfill-agent-features: Compute features for agent labels
"""

import contextlib
import shutil
from pathlib import Path

import pandas as pd
import typer

from .utils import console

labels_app = typer.Typer(help="Label data management commands")


@labels_app.command()
def migrate(
    labels_dir: Path = typer.Option(
        Path("labels"),
        "--labels",
        "-l",
        help="Directory containing legacy labels",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be done without making changes",
    ),
    skip_archive: bool = typer.Option(
        False,
        "--skip-archive",
        help="Skip archiving old directories (just migrate in place)",
    ),
):
    """Migrate from legacy embedded-feature format to normalized format.

    This command migrates labels from the legacy format (labels/dataset=*/data.csv
    with embedded features) to the normalized format:

    - labels/human/dataset=*/data.csv - Human labels (metadata only)
    - labels/agent/dataset=*/data.csv - Agent labels
    - labels/features/dataset=*/data.parquet - Computed features
    - labels/data/dataset=*/data.parquet - Raw pair data (geometries)

    After migration, the old directories are archived to labels/_archive/.

    Examples:
        matcher labels migrate --dry-run  # Preview changes
        matcher labels migrate            # Run migration
    """
    from loguru import logger

    from ..config import FEATURE_COLUMNS
    from ..labeling.label_store import HUMAN_LABEL_COLUMNS

    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        console.print(f"[red]Labels directory not found: {labels_dir}[/red]")
        raise typer.Exit(1)

    # Define paths
    legacy_labels_dir = labels_dir  # labels/dataset=*/data.csv
    human_dir = labels_dir / "human"
    agent_dir = labels_dir / "agent"
    features_dir = labels_dir / "features"
    data_dir = labels_dir / "data"
    archive_dir = labels_dir / "_archive"

    # Check for legacy labels_agent directory (parallel to labels/)
    legacy_agent_dir = labels_dir.parent / "labels_agent"
    legacy_geometry_dir = labels_dir.parent / "label_geometries"

    console.print("[blue]Scanning for labels to migrate...[/blue]")

    # Find legacy label partitions
    legacy_partitions = list(legacy_labels_dir.glob("dataset=*/data.csv"))
    legacy_agent_partitions = (
        list(legacy_agent_dir.glob("dataset=*/data.csv")) if legacy_agent_dir.exists() else []
    )
    legacy_geometry_partitions = (
        list(legacy_geometry_dir.glob("dataset=*/data.csv")) if legacy_geometry_dir.exists() else []
    )

    if not legacy_partitions and not legacy_agent_partitions:
        console.print("[yellow]No legacy labels found to migrate.[/yellow]")
        raise typer.Exit(0)

    console.print(f"  Found {len(legacy_partitions)} legacy label partitions")
    console.print(f"  Found {len(legacy_agent_partitions)} legacy agent label partitions")
    console.print(f"  Found {len(legacy_geometry_partitions)} legacy geometry partitions")

    if dry_run:
        console.print("\n[yellow][DRY RUN] Would perform the following actions:[/yellow]\n")

    stats = {
        "human_labels": 0,
        "agent_labels": 0,
        "features": 0,
        "data": 0,
        "partitions": 0,
    }

    # Process legacy labels (with embedded features)
    for partition_path in legacy_partitions:
        dataset_id = partition_path.parent.name.split("=")[1]

        try:
            df = pd.read_csv(partition_path)
        except Exception as e:
            console.print(f"[red]Failed to read {partition_path}: {e}[/red]")
            continue

        if len(df) == 0:
            continue

        # Handle ref_id -> gers_id rename
        if "ref_id" in df.columns and "gers_id" not in df.columns:
            df = df.rename(columns={"ref_id": "gers_id"})

        # Extract human label metadata
        human_cols = [c for c in HUMAN_LABEL_COLUMNS if c in df.columns]
        human_df = df[human_cols].copy()

        # Extract features
        feature_cols = ["gers_id", "target_id", "feature_version"] + [
            c for c in FEATURE_COLUMNS if c in df.columns
        ]
        features_df = df[[c for c in feature_cols if c in df.columns]].copy()

        if dry_run:
            console.print(f"  {dataset_id}:")
            console.print(f"    - Extract {len(human_df)} human labels to human/{dataset_id}")
            console.print(f"    - Extract {len(features_df)} feature rows to features/{dataset_id}")
        else:
            # Write human labels
            human_partition = human_dir / f"dataset={dataset_id}"
            human_partition.mkdir(parents=True, exist_ok=True)
            human_df.to_csv(human_partition / "data.csv", index=False)

            # Write features to parquet
            features_partition = features_dir / f"dataset={dataset_id}"
            features_partition.mkdir(parents=True, exist_ok=True)
            features_df.to_parquet(features_partition / "data.parquet", index=False)

            logger.info(
                f"Migrated {dataset_id}: {len(human_df)} labels, {len(features_df)} features"
            )

        stats["human_labels"] += len(human_df)
        stats["features"] += len(features_df)
        stats["partitions"] += 1

    # Process legacy geometry store
    for partition_path in legacy_geometry_partitions:
        dataset_id = partition_path.parent.name.split("=")[1]

        try:
            # Use GeometryStore to load (handles WKT parsing)
            from ..labeling.geometry_store import GeometryStore

            geo_store = GeometryStore(dataset_id, geometries_dir=legacy_geometry_dir)
            geo_df = geo_store.df

            if len(geo_df) == 0:
                continue

            # Convert to DataStore format
            import geopandas as gpd
            from shapely import wkt

            # Parse WKT geometries
            ref_geoms = geo_df["ref_geometry_wkt"].apply(lambda x: wkt.loads(x) if x else None)
            target_geoms = geo_df["target_geometry_wkt"].apply(
                lambda x: wkt.loads(x) if x else None
            )

            # Create GeoDataFrame
            data_gdf = gpd.GeoDataFrame(
                {
                    "gers_id": geo_df["gers_id"],
                    "target_id": geo_df["target_id"],
                    "ref_geometry": ref_geoms,
                    "target_geometry": target_geoms,
                },
                geometry="ref_geometry",
                crs="EPSG:4326",
            )

            # Parse JSON attributes if present
            import json

            for idx, row in geo_df.iterrows():
                ref_attrs = {}
                target_attrs = {}
                if "ref_attributes" in geo_df.columns:
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        ref_attrs = json.loads(row.get("ref_attributes", "{}") or "{}")
                if "target_attributes" in geo_df.columns:
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        target_attrs = json.loads(row.get("target_attributes", "{}") or "{}")

                data_gdf.at[idx, "ref_name"] = ref_attrs.get("name")
                data_gdf.at[idx, "target_name"] = target_attrs.get("name")
                data_gdf.at[idx, "ref_class"] = ref_attrs.get("class")
                data_gdf.at[idx, "target_class"] = target_attrs.get("class")

            if dry_run:
                console.print(f"  {dataset_id} (geometry):")
                console.print(
                    f"    - Migrate {len(data_gdf)} geometry records to data/{dataset_id}"
                )
            else:
                from shapely import wkb

                data_partition = data_dir / f"dataset={dataset_id}"
                data_partition.mkdir(parents=True, exist_ok=True)

                # GeoParquet only supports one active geometry column
                # Store target_geometry as WKB bytes
                write_gdf = data_gdf.copy()
                write_gdf["target_geometry_wkb"] = write_gdf["target_geometry"].apply(
                    lambda g: wkb.dumps(g) if g is not None else None
                )
                write_gdf = write_gdf.drop(columns=["target_geometry"])
                write_gdf.to_parquet(data_partition / "data.parquet")
                logger.info(f"Migrated {dataset_id} geometry: {len(data_gdf)} records")

            stats["data"] += len(data_gdf)

        except Exception as e:
            console.print(f"[red]Failed to migrate geometry for {dataset_id}: {e}[/red]")

    # Process legacy agent labels
    for partition_path in legacy_agent_partitions:
        dataset_id = partition_path.parent.name.split("=")[1]

        try:
            agent_df = pd.read_csv(partition_path)
        except Exception as e:
            console.print(f"[red]Failed to read agent labels {partition_path}: {e}[/red]")
            continue

        if len(agent_df) == 0:
            continue

        # Handle ref_id -> gers_id rename
        if "ref_id" in agent_df.columns and "gers_id" not in agent_df.columns:
            agent_df = agent_df.rename(columns={"ref_id": "gers_id"})

        if dry_run:
            console.print(f"  {dataset_id} (agent):")
            console.print(f"    - Migrate {len(agent_df)} agent labels to agent/{dataset_id}")
        else:
            agent_partition = agent_dir / f"dataset={dataset_id}"
            agent_partition.mkdir(parents=True, exist_ok=True)
            agent_df.to_csv(agent_partition / "data.csv", index=False)
            logger.info(f"Migrated {dataset_id} agent labels: {len(agent_df)}")

        stats["agent_labels"] += len(agent_df)

    # Archive old directories
    if not dry_run and not skip_archive:
        console.print("\n[blue]Archiving old directories...[/blue]")
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Move legacy partitions to archive
        for partition_path in legacy_partitions:
            dataset_id = partition_path.parent.name.split("=")[1]
            archive_partition = archive_dir / "legacy_labels" / f"dataset={dataset_id}"
            archive_partition.mkdir(parents=True, exist_ok=True)
            shutil.copy2(partition_path, archive_partition / "data.csv")

        # Archive labels_agent
        if legacy_agent_dir.exists():
            archive_agent = archive_dir / "labels_agent"
            if archive_agent.exists():
                shutil.rmtree(archive_agent)
            shutil.copytree(legacy_agent_dir, archive_agent)
            console.print(f"  Archived labels_agent/ to {archive_agent}")

        # Archive label_geometries
        if legacy_geometry_dir.exists():
            archive_geometry = archive_dir / "label_geometries"
            if archive_geometry.exists():
                shutil.rmtree(archive_geometry)
            shutil.copytree(legacy_geometry_dir, archive_geometry)
            console.print(f"  Archived label_geometries/ to {archive_geometry}")

    # Print summary
    console.print("\n[green]Migration Summary:[/green]")
    console.print(f"  Human labels: {stats['human_labels']}")
    console.print(f"  Agent labels: {stats['agent_labels']}")
    console.print(f"  Feature rows: {stats['features']}")
    console.print(f"  Data records: {stats['data']}")
    console.print(f"  Partitions processed: {stats['partitions']}")

    if dry_run:
        console.print("\n[yellow]Run without --dry-run to apply changes.[/yellow]")


@labels_app.command("backfill")
def backfill_features(
    labels_dir: Path = typer.Option(
        Path("labels"),
        "--labels",
        "-l",
        help="Directory containing labels",
    ),
    data_dir: Path = typer.Option(
        Path("data/raw"),
        "--data-dir",
        "-d",
        help="Directory containing source data files",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be done without making changes",
    ),
    skip_missing: bool = typer.Option(
        True,
        "--skip-missing/--fail-missing",
        help="Skip pairs with missing source data",
    ),
    human_only: bool = typer.Option(
        False,
        "--human-only",
        help="Only backfill human labels (skip agent)",
    ),
    agent_only: bool = typer.Option(
        False,
        "--agent-only",
        help="Only backfill agent labels (skip human)",
    ),
):
    """Compute missing features for human and/or agent labels.

    This command finds labels (human and/or agent) that don't have corresponding
    features in labels/features/ and computes them from source data.

    Use this after:
    - Adding new labels via the UI
    - Migrating to enable weak supervision with agent labels
    - When features need to be recomputed (e.g., new feature version)

    Examples:
        matcher labels backfill --dry-run
        matcher labels backfill
        matcher labels backfill --agent-only
        matcher labels backfill --human-only
    """

    from ..labeling.feature_store import FeatureStore
    from ..labeling.label_store import LabelStore

    labels_dir = Path(labels_dir)
    human_dir = labels_dir / "human"
    agent_dir = labels_dir / "agent"
    features_dir = labels_dir / "features"

    # Collect all labels to process
    all_label_keys = set()
    label_sources = {}  # Track which source each key came from

    if not agent_only:
        if human_dir.exists():
            console.print("[blue]Loading human labels...[/blue]")
            human_labels = LabelStore.load_human_labels(human_dir)
            if len(human_labels) > 0:
                human_keys = set(
                    zip(
                        human_labels["gers_id"],
                        human_labels["target_id"],
                        human_labels["dataset"],
                    )
                )
                console.print(
                    f"  Found {len(human_labels)} human labels across {human_labels['dataset'].nunique()} datasets"
                )
                all_label_keys.update(human_keys)
                for k in human_keys:
                    label_sources[k] = "human"
            else:
                console.print("  [yellow]No human labels found[/yellow]")
        else:
            console.print(f"  [yellow]Human labels directory not found: {human_dir}[/yellow]")

    if not human_only:
        if agent_dir.exists():
            console.print("[blue]Loading agent labels...[/blue]")
            agent_labels = LabelStore.load_agent_labels(agent_dir)
            if len(agent_labels) > 0:
                agent_keys = set(
                    zip(
                        agent_labels["gers_id"],
                        agent_labels["target_id"],
                        agent_labels["dataset"],
                    )
                )
                console.print(
                    f"  Found {len(agent_labels)} agent labels across {agent_labels['dataset'].nunique()} datasets"
                )
                all_label_keys.update(agent_keys)
                for k in agent_keys:
                    if k not in label_sources:  # Don't overwrite human
                        label_sources[k] = "agent"
            else:
                console.print("  [yellow]No agent labels found[/yellow]")
        else:
            console.print(f"  [yellow]Agent labels directory not found: {agent_dir}[/yellow]")

    if len(all_label_keys) == 0:
        console.print("[yellow]No labels found to process.[/yellow]")
        raise typer.Exit(0)

    # Load existing features to find what's missing
    console.print("[blue]Loading existing features...[/blue]")
    existing_features = FeatureStore.load_all(features_dir)

    if len(existing_features) > 0:
        existing_keys = set(
            zip(
                existing_features["gers_id"],
                existing_features["target_id"],
                existing_features["dataset"],
            )
        )
        console.print(f"  Found {len(existing_keys)} existing feature records")
    else:
        existing_keys = set()
        console.print("  No existing features found")

    # Find labels without features
    missing_keys = all_label_keys - existing_keys

    # Count by source
    missing_human = sum(1 for k in missing_keys if label_sources.get(k) == "human")
    missing_agent = sum(1 for k in missing_keys if label_sources.get(k) == "agent")
    console.print(
        f"  {len(missing_keys)} labels need features ({missing_human} human, {missing_agent} agent)"
    )

    if len(missing_keys) == 0:
        console.print("[green]All labels already have features.[/green]")
        raise typer.Exit(0)

    if dry_run:
        console.print("\n[yellow][DRY RUN] Would compute features for:[/yellow]")
        # Group by dataset for summary
        missing_by_dataset = {}
        for _gers_id, _target_id, dataset in missing_keys:
            missing_by_dataset[dataset] = missing_by_dataset.get(dataset, 0) + 1

        for dataset, count in sorted(missing_by_dataset.items()):
            console.print(f"  {dataset}: {count} pairs")
        console.print("\n[yellow]Run without --dry-run to compute features.[/yellow]")
        raise typer.Exit(0)

    # Import heavy dependencies only when needed
    import geopandas as gpd

    from ..config import DEFAULT_SNAP_TOLERANCE_M
    from ..features.alignment import linestring_alignment
    from ..features.compute import (
        compute_graphlet_similarity,
        compute_pair_features,
        precompute_graphlet_features,
    )
    from ..features.relational import build_sibling_search_context
    from ..features.semantic import _extract_name_string
    from ..features.spatial_context import SpatialContextIndex, compute_aligned_endpoint_features
    from ..filenames import find_overture_segments, find_target_file
    from ..utils.geometry import filter_to_linestrings

    # Process by dataset - get unique datasets from missing keys
    datasets = sorted(set(d for _, _, d in missing_keys))
    total_computed = 0
    total_skipped = 0

    for dataset in datasets:
        dataset_missing = [(g, t) for g, t, d in missing_keys if d == dataset]
        if not dataset_missing:
            continue

        console.print(f"\n[blue]Processing {dataset} ({len(dataset_missing)} pairs)...[/blue]")

        # Load source data
        overture_path = find_overture_segments(data_dir, dataset)
        target_path = find_target_file(data_dir, dataset)

        if overture_path is None:
            if skip_missing:
                console.print("  [yellow]Skipping: Overture data not found[/yellow]")
                total_skipped += len(dataset_missing)
                continue
            else:
                console.print("  [red]Error: Overture data not found[/red]")
                raise typer.Exit(1)

        if target_path is None:
            if skip_missing:
                console.print("  [yellow]Skipping: Target data not found[/yellow]")
                total_skipped += len(dataset_missing)
                continue
            else:
                console.print("  [red]Error: Target data not found[/red]")
                raise typer.Exit(1)

        # Load and prepare data
        console.print(f"  Loading Overture from {overture_path.name}...")
        ref_gdf = gpd.read_parquet(overture_path)
        ref_gdf = filter_to_linestrings(ref_gdf, source_name="reference")
        ref_gdf["id"] = ref_gdf["id"].astype(str)
        ref_lookup = ref_gdf.set_index("id")

        console.print(f"  Loading target from {target_path.name}...")
        target_gdf = gpd.read_parquet(target_path)
        target_gdf = filter_to_linestrings(target_gdf, source_name="target")
        target_gdf["id"] = target_gdf["id"].astype(str)
        target_lookup = target_gdf.set_index("id")

        # Project to UTM
        if ref_gdf.crs is not None and ref_gdf.crs.is_geographic:
            utm_crs = ref_gdf.estimate_utm_crs()
            ref_gdf_proj = ref_gdf.to_crs(utm_crs)
            target_gdf_proj = target_gdf.to_crs(utm_crs)
        else:
            ref_gdf_proj = ref_gdf
            target_gdf_proj = target_gdf

        # Build spatial index
        target_context = SpatialContextIndex()
        target_context.build_from_gdf(target_gdf_proj, id_column="id")

        # Build graphlet data
        ref_has_connectors = "connectors" in ref_gdf.columns
        ref_graphlet_data = precompute_graphlet_features(
            ref_gdf_proj,
            id_column="id",
            tolerance_m=DEFAULT_SNAP_TOLERANCE_M,
            connectors_column="connectors" if ref_has_connectors else None,
        )
        target_graphlet_data = precompute_graphlet_features(
            target_gdf_proj,
            id_column="id",
            tolerance_m=DEFAULT_SNAP_TOLERANCE_M,
        )

        _, target_seg_to_connectors, _, _ = (
            target_graphlet_data if target_graphlet_data else (None, None, None, None)
        )

        # Build sibling search contexts for per-pair parallel sibling detection
        # These hold the spatial index and segment metadata needed to search for
        # parallel siblings on aligned sublines (not precomputed on full geometries)
        console.print("  Building sibling search contexts...")
        ref_sibling_context = build_sibling_search_context(
            geometries=list(ref_gdf_proj.geometry),
            segment_ids=[str(sid) for sid in ref_gdf_proj["id"]],
            names=list(ref_gdf_proj.get("names", [None] * len(ref_gdf_proj))),
            classes=list(ref_gdf_proj.get("class", [None] * len(ref_gdf_proj))),
        )
        target_sibling_context = build_sibling_search_context(
            geometries=list(target_gdf_proj.geometry),
            segment_ids=[str(sid) for sid in target_gdf_proj["id"]],
            names=list(target_gdf_proj.get("names", [None] * len(target_gdf_proj))),
            classes=list(target_gdf_proj.get("class", [None] * len(target_gdf_proj))),
        )

        # Initialize feature store for this dataset
        feature_store = FeatureStore(dataset, features_dir=features_dir)

        computed = 0
        skipped = 0

        for gers_id, target_id in dataset_missing:
            # Check if IDs exist in data
            if gers_id not in ref_lookup.index or target_id not in target_lookup.index:
                skipped += 1
                continue

            # Get geometries
            ref_idx = ref_gdf[ref_gdf["id"] == gers_id].index[0]
            target_idx = target_gdf[target_gdf["id"] == target_id].index[0]

            ref_geom = ref_gdf_proj.geometry.loc[ref_idx]
            target_geom = target_gdf_proj.geometry.loc[target_idx]

            if ref_geom is None or ref_geom.is_empty or target_geom is None or target_geom.is_empty:
                skipped += 1
                continue

            # Compute alignment
            alignment = linestring_alignment(ref_geom, target_geom)

            # Compute endpoint features
            target_filtered_idx = (
                target_gdf_proj[target_gdf_proj["id"] == target_id].index[0]
                if target_id in target_gdf_proj["id"].values
                else None
            )
            endpoint_features = compute_aligned_endpoint_features(
                target_geom,
                target_context,
                start_frac=alignment.dataset_start_frac,
                end_frac=alignment.dataset_end_frac,
                exclude_segment_idx=target_filtered_idx,
                seg_id=target_id,
                seg_to_connectors=target_seg_to_connectors,
            )

            # Compute graphlet similarity
            graphlet_features = compute_graphlet_similarity(
                gers_id,
                target_id,
                ref_graphlet_data,
                target_graphlet_data,
                alignment,
            )

            # Get names and classes
            ref_row = ref_lookup.loc[gers_id]
            target_row = target_lookup.loc[target_id]
            ref_name = _extract_name_string(ref_row["names"]) if "names" in ref_row.index else None
            target_name = (
                _extract_name_string(target_row["names"]) if "names" in target_row.index else None
            )
            ref_class = ref_row["class"] if "class" in ref_row.index else None
            target_class = target_row["class"] if "class" in target_row.index else None
            ref_subclass = ref_row["subclass"] if "subclass" in ref_row.index else None
            target_subclass = target_row["subclass"] if "subclass" in target_row.index else None

            # Compute all features
            features = compute_pair_features(
                ref_geom,
                target_geom,
                ref_name,
                target_name,
                ref_class,
                target_class,
                ref_subclass,
                target_subclass,
                endpoint_features=endpoint_features,
                alignment=alignment,
                graphlet_features=graphlet_features,
                ref_graphlet_data=ref_graphlet_data,
                target_graphlet_data=target_graphlet_data,
                ref_seg_id=gers_id,
                target_seg_id=target_id,
                ref_sibling_context=ref_sibling_context,
                target_sibling_context=target_sibling_context,
            )

            # Add to feature store
            feature_store.add(gers_id=gers_id, target_id=target_id, features=features)
            computed += 1

        # Save feature store
        if computed > 0:
            feature_store.save()
            console.print(f"  [green]Computed {computed} features, skipped {skipped}[/green]")
        else:
            console.print(f"  [yellow]Skipped all {skipped} pairs (IDs not in data)[/yellow]")

        total_computed += computed
        total_skipped += skipped

    console.print(
        f"\n[green]Backfill complete: {total_computed} features computed, {total_skipped} skipped[/green]"
    )


@labels_app.command("stats")
def label_stats(
    labels_dir: Path = typer.Option(
        Path("labels"),
        "--labels",
        "-l",
        help="Directory containing labels",
    ),
):
    """Show statistics about label data.

    Displays counts and distribution of human and agent labels across datasets.
    """
    from ..labeling.feature_store import FeatureStore
    from ..labeling.label_store import LabelStore

    labels_dir = Path(labels_dir)

    console.print("[blue]Loading labels...[/blue]\n")

    # Try legacy format first
    legacy_labels = LabelStore.load_all(labels_dir)
    human_labels = LabelStore.load_human_labels(labels_dir / "human")
    agent_labels = LabelStore.load_agent_labels(labels_dir / "agent")
    features = FeatureStore.load_all(labels_dir / "features")

    console.print("[bold]Label Statistics[/bold]\n")

    # Legacy labels
    if len(legacy_labels) > 0:
        console.print(f"Legacy labels (embedded features): {len(legacy_labels)}")
        if "dataset" in legacy_labels.columns:
            for dataset in sorted(legacy_labels["dataset"].unique()):
                count = (legacy_labels["dataset"] == dataset).sum()
                console.print(f"  {dataset}: {count}")
        console.print()

    # Human labels (normalized)
    if len(human_labels) > 0:
        console.print(f"Human labels (normalized): {len(human_labels)}")
        if "dataset" in human_labels.columns:
            for dataset in sorted(human_labels["dataset"].unique()):
                count = (human_labels["dataset"] == dataset).sum()
                console.print(f"  {dataset}: {count}")
        console.print()

    # Agent labels
    if len(agent_labels) > 0:
        console.print(f"Agent labels: {len(agent_labels)}")
        if "dataset" in agent_labels.columns:
            for dataset in sorted(agent_labels["dataset"].unique()):
                count = (agent_labels["dataset"] == dataset).sum()
                console.print(f"  {dataset}: {count}")
        console.print()

        # Label distribution
        if "label" in agent_labels.columns:
            console.print("Agent label distribution:")
            for label in sorted(agent_labels["label"].unique()):
                count = (agent_labels["label"] == label).sum()
                console.print(f"  {label}: {count}")
        console.print()

    # Features
    if len(features) > 0:
        console.print(f"Feature records: {len(features)}")
        if "dataset" in features.columns:
            for dataset in sorted(features["dataset"].unique()):
                count = (features["dataset"] == dataset).sum()
                console.print(f"  {dataset}: {count}")
        console.print()

    # Summary
    total_human = len(legacy_labels) + len(human_labels)
    total_agent = len(agent_labels)
    console.print(f"[bold]Total: {total_human} human, {total_agent} agent labels[/bold]")
