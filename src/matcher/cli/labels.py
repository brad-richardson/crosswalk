"""Labels management CLI commands.

This module provides CLI commands for managing label data, including:
- backfill: Recompute features for labels
- stats: Show label statistics
"""

from pathlib import Path

import typer

from .utils import console

labels_app = typer.Typer(help="Label data management commands")


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
    include_agent: bool = typer.Option(
        False,
        "--include-agent",
        help="Also backfill agent labels (by default only human labels are processed)",
    ),
    agent_only: bool = typer.Option(
        False,
        "--agent-only",
        help="Only backfill agent labels (skip human)",
    ),
    missing_only: bool = typer.Option(
        False,
        "--missing-only",
        help="Only compute features for labels without any features (skip existing)",
    ),
):
    """Recompute features for human labels using current feature computation code.

    By default, this recomputes ALL features for all human labels to ensure
    consistency when feature computation logic changes or new features are added.

    Use --include-agent to also process agent labels.
    Use --missing-only to only compute features for labels that don't have any.

    Use this after:
    - Adding new features to the ML pipeline
    - Changing feature computation logic
    - Adding new labels via the UI (with --missing-only)

    Examples:
        matcher labels backfill --dry-run       # Preview what would be recomputed
        matcher labels backfill                 # Recompute all human label features
        matcher labels backfill --include-agent # Also include agent labels
        matcher labels backfill --missing-only  # Only compute for labels without features
    """

    from ..labeling.feature_store import FeatureStore
    from ..labeling.label_store import LabelStore

    labels_dir = Path(labels_dir)
    human_dir = labels_dir / "human"
    agent_dir = labels_dir / "agent"
    features_dir = labels_dir / "features"

    # Determine what to process based on flags
    process_human = not agent_only
    process_agent = include_agent or agent_only

    # Collect all labels to process
    all_label_keys = set()
    label_sources = {}  # Track which source each key came from

    if process_human:
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

    if process_agent:
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

    # Determine which labels to process
    if missing_only:
        # Only compute for labels without any features
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

        keys_to_process = all_label_keys - existing_keys
        action_verb = "missing"
    else:
        # Recompute all features (default)
        keys_to_process = all_label_keys
        action_verb = "total"

    # Count by source
    human_count = sum(1 for k in keys_to_process if label_sources.get(k) == "human")
    agent_count = sum(1 for k in keys_to_process if label_sources.get(k) == "agent")
    console.print(
        f"  {len(keys_to_process)} {action_verb} labels to process ({human_count} human, {agent_count} agent)"
    )

    if len(keys_to_process) == 0:
        if missing_only:
            console.print("[green]All labels already have features.[/green]")
        else:
            console.print("[yellow]No labels to process.[/yellow]")
        raise typer.Exit(0)

    if dry_run:
        console.print("\n[yellow][DRY RUN] Would compute features for:[/yellow]")
        # Group by dataset for summary
        by_dataset = {}
        for _gers_id, _target_id, dataset in keys_to_process:
            by_dataset[dataset] = by_dataset.get(dataset, 0) + 1

        for dataset, count in sorted(by_dataset.items()):
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

    # Process by dataset - get unique datasets from keys to process
    datasets = sorted(set(d for _, _, d in keys_to_process))
    total_computed = 0
    total_skipped = 0

    for dataset in datasets:
        dataset_keys = [(g, t) for g, t, d in keys_to_process if d == dataset]
        if not dataset_keys:
            continue

        console.print(f"\n[blue]Processing {dataset} ({len(dataset_keys)} pairs)...[/blue]")

        # Load source data
        overture_path = find_overture_segments(data_dir, dataset)
        target_path = find_target_file(data_dir, dataset)

        if overture_path is None:
            if skip_missing:
                console.print("  [yellow]Skipping: Overture data not found[/yellow]")
                total_skipped += len(dataset_keys)
                continue
            else:
                console.print("  [red]Error: Overture data not found[/red]")
                raise typer.Exit(1)

        if target_path is None:
            if skip_missing:
                console.print("  [yellow]Skipping: Target data not found[/yellow]")
                total_skipped += len(dataset_keys)
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

        # Initialize feature store and data store for this dataset
        feature_store = FeatureStore(dataset, features_dir=features_dir)

        # Load stored pair data (geometries captured at labeling time)
        # This is critical because target IDs are not stable across data refreshes
        from ..labeling.data_store import DataStore

        data_store = DataStore(dataset, data_dir=labels_dir / "data")
        has_stored_data = len(data_store.gdf) > 0
        if has_stored_data:
            console.print(
                f"  Using stored geometries from labels/data ({len(data_store.gdf)} pairs)"
            )
        else:
            console.print("  [yellow]No stored geometries - using raw data lookup[/yellow]")

        computed = 0
        skipped = 0
        used_stored = 0
        used_lookup = 0

        for gers_id, target_id in dataset_keys:
            ref_geom = None
            target_geom = None
            ref_name = None
            target_name = None
            ref_class = None
            target_class = None
            ref_subclass = None
            target_subclass = None

            # First, try to get geometries from stored data (preferred - stable)
            if has_stored_data:
                pair_data = data_store.get_pair(gers_id, target_id)
                if pair_data is not None:
                    # Get geometries from stored data (in WGS84)
                    stored_ref = pair_data.get("ref_geometry")
                    stored_target = pair_data.get("target_geometry")
                    if stored_ref is not None and stored_target is not None:
                        # Project to UTM
                        ref_geom = (
                            gpd.GeoSeries([stored_ref], crs="EPSG:4326").to_crs(utm_crs).iloc[0]
                        )
                        target_geom = (
                            gpd.GeoSeries([stored_target], crs="EPSG:4326").to_crs(utm_crs).iloc[0]
                        )
                        # Get attributes from stored data
                        ref_name = pair_data.get("ref_name")
                        target_name = pair_data.get("target_name")
                        ref_class = pair_data.get("ref_class")
                        target_class = pair_data.get("target_class")
                        ref_subclass = pair_data.get("ref_subclass")
                        target_subclass = pair_data.get("target_subclass")
                        used_stored += 1

            # Fall back to raw data lookup if stored data not available
            if ref_geom is None or target_geom is None:
                if gers_id not in ref_lookup.index or target_id not in target_lookup.index:
                    skipped += 1
                    continue

                # Get geometries from raw data files
                ref_idx = ref_gdf[ref_gdf["id"] == gers_id].index[0]
                target_idx = target_gdf[target_gdf["id"] == target_id].index[0]

                ref_geom = ref_gdf_proj.geometry.loc[ref_idx]
                target_geom = target_gdf_proj.geometry.loc[target_idx]

                # Get attributes from raw data
                ref_row = ref_lookup.loc[gers_id]
                target_row = target_lookup.loc[target_id]
                ref_name = (
                    _extract_name_string(ref_row["names"]) if "names" in ref_row.index else None
                )
                target_name = (
                    _extract_name_string(target_row["names"])
                    if "names" in target_row.index
                    else None
                )
                ref_class = ref_row["class"] if "class" in ref_row.index else None
                target_class = target_row["class"] if "class" in target_row.index else None
                ref_subclass = ref_row["subclass"] if "subclass" in ref_row.index else None
                target_subclass = target_row["subclass"] if "subclass" in target_row.index else None
                used_lookup += 1

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

            # Note: oneway_lr and speed_limit_kph_lr columns are fetched but not used as features
            # See docs/RESEARCH_GRAVEYARD.md - ablation showed these hurt model performance

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
            source_info = f"(stored={used_stored}, lookup={used_lookup})" if used_stored > 0 else ""
            console.print(
                f"  [green]Computed {computed} features {source_info}, skipped {skipped}[/green]"
            )
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
