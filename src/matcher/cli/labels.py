"""Labels management CLI commands.

This module provides CLI commands for managing label data, including:
- backfill: Recompute features for labels
- stats: Show label statistics
"""

from pathlib import Path

import typer

from .utils import console

labels_app = typer.Typer(help="Label data management commands")


def _auto_fetch_overture(dataset: str, data_dir: Path) -> Path | None:
    """Auto-fetch Overture segments when missing, using dataset config bbox.

    Delegates to the existing fetch reference implementation to avoid
    duplicating the fetch logic.

    Returns the path to the fetched file, or None if no config found.
    """
    from ..datasets.schema import get_dataset_config
    from ..filenames import find_overture_segments
    from .data import _fetch_reference_impl
    from .utils import console

    config = get_dataset_config(dataset)
    if config is None or config.fetch is None or config.fetch.bbox is None:
        return None

    console.print(f"  [blue]Auto-fetching Overture data for {dataset}...[/blue]")
    _fetch_reference_impl(
        dataset_name=dataset,
        output_dir=data_dir,
        sources={"overture"},
    )

    return find_overture_segments(data_dir, dataset)


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
    require_stored_data: bool = typer.Option(
        False,
        "--require-stored-data",
        help="Reject pairs without stored geometries (no fallback to raw data)",
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
    import pandas as pd

    from ..blocking.spatial_index import CandidatePair
    from ..features.pipeline import prepare_worker_data
    from ..filenames import find_overture_segments, find_target_file
    from ..matching.ml import _compute_feature_chunk, _init_worker
    from ..utils.geometry import filter_to_linestrings

    # Process by dataset - get unique datasets from keys to process
    datasets = sorted(set(d for _, _, d in keys_to_process))
    total_computed = 0
    total_skipped = 0
    total_errored = 0

    for dataset in datasets:
        dataset_keys = [(g, t) for g, t, d in keys_to_process if d == dataset]
        if not dataset_keys:
            continue

        console.print(f"\n[blue]Processing {dataset} ({len(dataset_keys)} pairs)...[/blue]")

        # Load source data
        overture_path = find_overture_segments(data_dir, dataset)
        target_path = find_target_file(data_dir, dataset)

        if overture_path is None:
            # Auto-fetch Overture data using dataset config bbox
            overture_path = _auto_fetch_overture(dataset, data_dir)
            if overture_path is None:
                console.print(
                    f"  [red]Error: Overture data not found and auto-fetch failed "
                    f"(no dataset config for '{dataset}')[/red]"
                )
                raise typer.Exit(1)

        if target_path is None and not skip_missing:
            console.print("  [red]Error: Target data not found[/red]")
            raise typer.Exit(1)

        # Load and prepare data
        console.print(f"  Loading Overture from {overture_path.name}...")
        ref_gdf = gpd.read_parquet(overture_path)
        ref_gdf = filter_to_linestrings(ref_gdf, source_name="reference")
        ref_gdf["id"] = ref_gdf["id"].astype(str)
        ref_lookup = ref_gdf.set_index("id")

        # Project reference to UTM
        if ref_gdf.crs is not None and ref_gdf.crs.is_geographic:
            utm_crs = ref_gdf.estimate_utm_crs()
            ref_gdf_proj = ref_gdf.to_crs(utm_crs)
        else:
            utm_crs = ref_gdf.crs
            ref_gdf_proj = ref_gdf

        # Load target raw data if available (not required when stored geometries exist)
        if target_path is not None:
            console.print(f"  Loading target from {target_path.name}...")
            target_gdf = gpd.read_parquet(target_path)
            target_gdf = filter_to_linestrings(target_gdf, source_name="target")
            target_gdf["id"] = target_gdf["id"].astype(str)
            target_lookup = target_gdf.set_index("id")
            target_gdf_proj = (
                target_gdf.to_crs(utm_crs) if utm_crs != target_gdf.crs else target_gdf
            )
        else:
            console.print("  [yellow]Target data not found - using stored geometries only[/yellow]")
            target_gdf = None
            target_gdf_proj = None
            target_lookup = None

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
        errored = 0
        used_stored = 0
        used_lookup = 0
        no_stored_rejected = 0

        # --- Phase 1: Resolve geometries for all labeled pairs ---
        # Collect stored target geometries for building the augmented target GDF
        resolved_pairs = []  # list of (gers_id, target_id, pair_data)
        stored_target_overrides = {}  # target_id -> WGS84 geometry
        stored_target_attrs = {}  # target_id -> attribute dict (for segments not in raw data)

        for gers_id, target_id in dataset_keys:
            pair_data = None
            used_stored_for_pair = False

            if has_stored_data:
                pair_data = data_store.get_pair(gers_id, target_id)
                if pair_data is not None:
                    stored_ref = pair_data.get("ref_geometry")
                    stored_target = pair_data.get("target_geometry")
                    if stored_ref is not None and stored_target is not None:
                        used_stored_for_pair = True

            if not used_stored_for_pair:
                if require_stored_data:
                    no_stored_rejected += 1
                    skipped += 1
                    continue
                if (
                    gers_id not in ref_lookup.index
                    or target_lookup is None
                    or target_id not in target_lookup.index
                ):
                    skipped += 1
                    continue
                used_lookup += 1
            else:
                used_stored += 1
                stored_target_overrides[target_id] = pair_data["target_geometry"]
                if target_lookup is None or target_id not in target_lookup.index:
                    stored_target_attrs[target_id] = {
                        "names": pair_data.get("target_name"),
                        "names_lr": pair_data.get("target_names_lr"),
                        "class": pair_data.get("target_class"),
                        "subclass": pair_data.get("target_subclass"),
                    }

            resolved_pairs.append((gers_id, target_id, pair_data))

        if not resolved_pairs:
            reason = "IDs not in data"
            if no_stored_rejected > 0:
                reason += f", {no_stored_rejected} rejected (no stored data)"
            console.print(f"  [yellow]Skipped all {skipped} pairs ({reason})[/yellow]")
            total_skipped += skipped
            continue

        # --- Phase 2: Build augmented target GeoDataFrame ---
        # Start with full raw target (for sibling contexts, spatial index, topology),
        # then override/append stored geometries for labeled pairs
        if target_gdf_proj is not None:
            augmented_target = target_gdf_proj.copy()
        else:
            augmented_target = gpd.GeoDataFrame(
                {"id": pd.Series(dtype=str)},
                geometry=gpd.GeoSeries([], crs=utm_crs),
            )

        target_id_set = (
            set(augmented_target["id"].astype(str)) if len(augmented_target) > 0 else set()
        )

        # Override geometry for stored-data targets already in raw data
        override_ids = [tid for tid in stored_target_overrides if tid in target_id_set]
        if override_ids:
            override_geoms = gpd.GeoSeries(
                [stored_target_overrides[tid] for tid in override_ids],
                crs="EPSG:4326",
            ).to_crs(utm_crs)
            for tid, geom in zip(override_ids, override_geoms):
                mask = augmented_target["id"].astype(str) == tid
                augmented_target.loc[mask, "geometry"] = geom

        # Append new rows for stored-data targets not in raw data
        append_ids = [tid for tid in stored_target_attrs if tid not in target_id_set]
        if append_ids:
            append_geoms = gpd.GeoSeries(
                [stored_target_overrides[tid] for tid in append_ids],
                crs="EPSG:4326",
            ).to_crs(utm_crs)
            append_rows = []
            for tid, geom in zip(append_ids, append_geoms):
                row = {"id": tid, "geometry": geom}
                row.update(stored_target_attrs[tid])
                append_rows.append(row)
            new_gdf = gpd.GeoDataFrame(append_rows, geometry="geometry", crs=utm_crs)
            augmented_target = pd.concat([augmented_target, new_gdf], ignore_index=True)

        # --- Phase 3: Create CandidatePair objects ---
        ref_id_to_idx = {str(rid): idx for idx, rid in enumerate(ref_gdf_proj["id"])}
        target_id_to_idx = {str(tid): idx for idx, tid in enumerate(augmented_target["id"])}

        candidates = []
        candidate_metadata = []  # parallel list tracking (gers_id, target_id, pair_data)

        for gers_id, target_id, pair_data in resolved_pairs:
            ref_idx = ref_id_to_idx.get(str(gers_id))
            target_idx = target_id_to_idx.get(str(target_id))

            if ref_idx is None or target_idx is None:
                skipped += 1
                continue

            ref_geom = ref_gdf_proj.geometry.iloc[ref_idx]
            target_geom = augmented_target.geometry.iloc[target_idx]
            if ref_geom is None or ref_geom.is_empty or target_geom is None or target_geom.is_empty:
                skipped += 1
                continue

            candidates.append(
                CandidatePair(
                    ref_id=gers_id,
                    ref_idx=ref_idx,
                    target_id=target_id,
                    target_idx=target_idx,
                    distance_estimate=0.0,
                    heading_diff=0.0,
                    length_ratio=1.0,
                )
            )
            candidate_metadata.append((gers_id, target_id, pair_data))

        if not candidates:
            reason = "no valid geometries"
            if no_stored_rejected > 0:
                reason += f", {no_stored_rejected} rejected (no stored data)"
            console.print(f"  [yellow]Skipped all pairs ({reason})[/yellow]")
            total_skipped += skipped
            continue

        # --- Phase 4: Prepare worker data through shared pipeline ---
        # One call replaces manual spatial index, graphlet, sibling context,
        # topology, alignment, and endpoint feature computation
        console.print("  Running shared feature pipeline...")
        pipeline_result = prepare_worker_data(
            candidates=candidates,
            reference=ref_gdf_proj,
            target=augmented_target,
            n_jobs=1,  # small dataset, no need for parallel alignment
        )
        worker_data = pipeline_result.worker_data

        # --- Phase 5: Override topology with stored values ---
        # 3-tier fallback: stored topology > computed by pipeline > NaN defaults
        for cand, (_gid, _tid, pair_data) in zip(candidates, candidate_metadata):
            if pair_data is not None:
                stored_ref_topo = pair_data.get("ref_topology")
                stored_target_topo = pair_data.get("target_topology")
                if stored_ref_topo:
                    worker_data["ref_topology"][cand.ref_idx] = stored_ref_topo
                if stored_target_topo:
                    worker_data["target_topology"][cand.target_idx] = stored_target_topo

        # --- Phase 6: Compute features through shared code path ---
        # Uses the exact same _compute_feature_chunk() that inference uses,
        # including batch geometric computation and LR attribute extraction
        console.print("  Computing features...")
        _init_worker(worker_data)
        work_items = [(cand.ref_idx, cand.target_idx) for cand in candidates]
        results, _errors = _compute_feature_chunk(work_items)

        # --- Phase 7: Process results and persist ---
        for i, (result, (gers_id, target_id, pair_data)) in enumerate(
            zip(results, candidate_metadata)
        ):
            if result is None:
                skipped += 1
                continue

            if result.get("_error"):
                errored += 1

            # Backfill topology into data store if not already stored
            if has_stored_data and pair_data is not None and pair_data.get("ref_topology") is None:
                ref_topo = worker_data["ref_topology"].get(candidates[i].ref_idx)
                target_topo = worker_data["target_topology"].get(candidates[i].target_idx)
                if ref_topo is not None and target_topo is not None:
                    data_store.update_topology(
                        gers_id,
                        target_id,
                        ref_topology=ref_topo,
                        target_topology=target_topo,
                    )

            feature_store.add(gers_id=gers_id, target_id=target_id, features=result)
            computed += 1

        # Save feature store and data store (topology backfill)
        if computed > 0:
            feature_store.save()
            if has_stored_data:
                data_store.save()
            parts = [f"Computed {computed} features"]
            if used_stored > 0 or used_lookup > 0:
                parts[0] += f" (stored={used_stored}, lookup={used_lookup})"
            if no_stored_rejected > 0:
                parts.append(f"rejected={no_stored_rejected} (no stored data)")
            if errored > 0:
                error_rate = errored / computed
                parts.append(f"[red]errored={errored} ({error_rate:.0%})[/red]")
            parts.append(f"skipped={skipped}")
            console.print("  " + ", ".join(parts))
        else:
            reason = "IDs not in data"
            if no_stored_rejected > 0:
                reason += f", {no_stored_rejected} rejected (no stored data)"
            console.print(f"  [yellow]Skipped all {skipped} pairs ({reason})[/yellow]")

        total_computed += computed
        total_skipped += skipped
        total_errored += errored

    # Report results
    console.print(
        f"\nBackfill complete: {total_computed} features computed, {total_skipped} skipped"
    )
    if total_errored > 0:
        error_rate = total_errored / total_computed if total_computed > 0 else 0
        console.print(
            f"[red]WARNING: {total_errored} pairs ({error_rate:.1%}) fell back to error features. "
            f"This likely indicates a bug in feature computation or bad input data.[/red]"
        )
        if error_rate > 0.05:
            console.print(
                "[red]ERROR: Error rate exceeds 5% threshold. "
                "Features were saved but should NOT be committed until the issue is resolved.[/red]"
            )
            raise typer.Exit(1)


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
