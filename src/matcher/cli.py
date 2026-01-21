"""CLI entry point for the road network conflation pipeline."""

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="matcher",
    help="Road network conflation pipeline - link local roads to Overture GERS",
    no_args_is_help=True,
)
console = Console()


@app.command()
def fetch(
    bbox: str | None = typer.Option(
        None,
        "--bbox",
        "-b",
        help="Bounding box: xmin,ymin,xmax,ymax (EPSG:4326). Required unless --for-dataset is used.",
    ),
    for_dataset: str | None = typer.Option(
        None,
        "--for-dataset",
        "-f",
        help="Fetch reference data for a target dataset (uses bbox from dataset config, names outputs accordingly)",
    ),
    output_dir: Path = typer.Option(
        Path("data/raw"),
        "--output",
        "-o",
        help="Output directory for fetched data",
    ),
    dataset: list[str] = typer.Option(
        ["overture"],
        "--dataset",
        "-d",
        help="Dataset(s) to fetch: 'overture' or 'osm' (can specify multiple)",
    ),
    cache_dir: Path | None = typer.Option(
        None,
        "--cache-dir",
        help="Cache directory for PBF files (default: ~/.cache/matcher/pbf/)",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Force fresh download, ignore cache",
    ),
    keep_pbf: bool = typer.Option(
        False,
        "--keep-pbf",
        help="Keep extracted PBF file for debugging",
    ),
    bbox_buffer: float | None = typer.Option(
        None,
        "--bbox-buffer",
        help="Expand bbox by this distance (meters). Defaults to 1km for complete network coverage.",
    ),
):
    """Fetch road data for an area of interest.

    Examples:
        # Fetch by explicit bbox
        matcher fetch --bbox -122.7,45.5,-122.6,45.55                    # Overture (default)
        matcher fetch --bbox -122.7,45.5,-122.6,45.55 -d osm             # OSM only
        matcher fetch --bbox -122.7,45.5,-122.6,45.55 -d overture -d osm # Both

        # Fetch for a configured dataset (auto-uses bbox, auto-names outputs)
        matcher fetch --for-dataset boston_streets -d osm              # boston_streets_osm_segments.parquet
        matcher fetch -f boston_streets -d overture -d osm             # Both, named for boston_streets

    Note: OSM fetching requires osmium-tool to be installed:
        brew install osmium-tool (macOS) or apt install osmium-tool (Ubuntu)
    """
    from .fetch import osm as osm_module
    from .fetch import overture as ov_module

    # Validate datasets
    valid_datasets = {"overture", "osm"}
    datasets = {d.lower() for d in dataset}
    invalid = datasets - valid_datasets
    if invalid:
        console.print(
            f"[red]Error: Invalid dataset(s): {invalid}. Must be 'overture' or 'osm'[/red]"
        )
        raise typer.Exit(1)

    # Determine bbox and dataset name
    dataset_name: str | None = None

    if for_dataset:
        # Look up bbox from dataset YAML configs
        from .datasets.schema import get_dataset_config, list_dataset_configs

        config = get_dataset_config(for_dataset)
        if config is None:
            console.print(f"[red]Error: Could not find dataset config for '{for_dataset}'[/red]")
            available = list_dataset_configs()
            if available:
                console.print("[yellow]Available datasets: " + ", ".join(sorted(available)[:10]))
                if len(available) > 10:
                    console.print(f"  ... and {len(available) - 10} more[/yellow]")
            raise typer.Exit(1)

        if config.fetch is None or config.fetch.bbox is None:
            console.print(f"[red]Error: Dataset '{for_dataset}' has no bbox configured[/red]")
            raise typer.Exit(1)

        xmin, ymin, xmax, ymax = config.fetch.bbox
        dataset_name = for_dataset
        console.print(f"[blue]Using bbox from dataset config: {for_dataset}[/blue]")
        console.print(f"[blue]  bbox: {xmin},{ymin},{xmax},{ymax}[/blue]")

    elif bbox:
        coords = [float(x.strip()) for x in bbox.split(",")]
        if len(coords) != 4:
            console.print("[red]Error: bbox must have 4 values: xmin,ymin,xmax,ymax[/red]")
            raise typer.Exit(1)
        xmin, ymin, xmax, ymax = coords

    else:
        console.print("[red]Error: Either --bbox or --for-dataset is required[/red]")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    original_bbox = ov_module.BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    # For Overture fetches, use a default buffer to avoid fringe effects
    # User can override with --bbox-buffer
    overture_buffer = bbox_buffer
    if overture_buffer is None and "overture" in datasets:
        overture_buffer = ov_module.DEFAULT_OVERTURE_BUFFER_M
        console.print(
            f"[blue]Using default {overture_buffer}m buffer for Overture data "
            f"(override with --bbox-buffer)[/blue]"
        )

    # Create bbox for Overture (potentially buffered)
    # Use explicit None check to allow --bbox-buffer=0 to disable buffering
    if overture_buffer is not None and overture_buffer > 0:
        overture_bbox = original_bbox.expand(overture_buffer)
        console.print(
            f"[blue]  Buffered bbox: {overture_bbox.xmin:.6f},{overture_bbox.ymin:.6f},"
            f"{overture_bbox.xmax:.6f},{overture_bbox.ymax:.6f}[/blue]"
        )
    else:
        overture_bbox = original_bbox
        if overture_buffer == 0 and "overture" in datasets:
            console.print("[blue]Buffer explicitly disabled (--bbox-buffer=0)[/blue]")
        overture_buffer = None

    # For OSM, also use a default buffer to avoid fringe effects
    osm_buffer = bbox_buffer
    if osm_buffer is None and "osm" in datasets:
        osm_buffer = osm_module.DEFAULT_OSM_BUFFER_M
        console.print(
            f"[blue]Using default {osm_buffer}m buffer for OSM data "
            f"(override with --bbox-buffer)[/blue]"
        )

    # Create bbox for OSM (potentially buffered)
    # Use explicit None check to allow --bbox-buffer=0 to disable buffering
    if osm_buffer is not None and osm_buffer > 0:
        osm_bbox = original_bbox.expand(osm_buffer)
        if "osm" in datasets:
            console.print(
                f"[blue]  Buffered bbox: {osm_bbox.xmin:.6f},{osm_bbox.ymin:.6f},"
                f"{osm_bbox.xmax:.6f},{osm_bbox.ymax:.6f}[/blue]"
            )
    else:
        osm_bbox = original_bbox
        if osm_buffer == 0 and "osm" in datasets:
            console.print("[blue]Buffer explicitly disabled (--bbox-buffer=0)[/blue]")
        osm_buffer = None

    if "overture" in datasets:
        # Name outputs based on dataset if provided
        overture_prefix = f"{dataset_name}_overture" if dataset_name else "overture"
        console.print("[blue]Fetching Overture segments...[/blue]")
        segments_path = ov_module.fetch_overture_segments(
            bbox=overture_bbox,
            output_path=output_dir / f"{overture_prefix}_segments.parquet",
            original_bbox=original_bbox,
            buffer_m=overture_buffer,
        )
        console.print(f"[green]Saved Overture segments to {segments_path}[/green]")

        console.print("[blue]Fetching Overture connectors...[/blue]")
        connectors_path = ov_module.fetch_overture_connectors(
            bbox=overture_bbox,
            output_path=output_dir / f"{overture_prefix}_connectors.parquet",
            original_bbox=original_bbox,
            buffer_m=overture_buffer,
        )
        console.print(f"[green]Saved Overture connectors to {connectors_path}[/green]")

    if "osm" in datasets:
        # Name outputs based on dataset if provided
        osm_name = f"{dataset_name}_osm" if dataset_name else "osm"

        # When using --for-dataset, OSM uses unbuffered bbox with fully-inside filter
        # This ensures OSM coverage matches the target dataset exactly for validation
        use_validation_mode = for_dataset is not None

        if use_validation_mode:
            fetch_bbox = original_bbox
            actual_buffer = None
            console.print(
                "[blue]OSM: using unbuffered bbox, filtering to fully-inside features "
                "(--for-dataset mode)[/blue]"
            )
        else:
            fetch_bbox = osm_bbox
            actual_buffer = osm_buffer

        console.print("[blue]Fetching OSM data...[/blue]")
        segments_path, connectors_path = osm_module.fetch_osm_data(
            bbox=fetch_bbox,
            output_dir=output_dir,
            cache_dir=cache_dir,
            force_download=no_cache,
            keep_pbf=keep_pbf,
            original_bbox=original_bbox,
            buffer_m=actual_buffer,
            name=osm_name,
            filter_fully_inside=use_validation_mode,
        )
        console.print(f"[green]Saved OSM segments (ways) to {segments_path}[/green]")
        console.print(f"[green]Saved OSM connectors (nodes) to {connectors_path}[/green]")

    # Update last_fetch in dataset config if using --for-dataset
    if dataset_name:
        from datetime import UTC, datetime

        from .datasets.schema import (
            LastFetch,
            get_dataset_config,
            get_datasets_dir,
            save_dataset_config,
        )

        config = get_dataset_config(dataset_name)
        if config:
            # Determine which buffer was used
            buffer_m = overture_buffer if "overture" in datasets else osm_buffer

            # Get feature count from metadata if available
            feature_count = 0
            geometry_types: list[str] = []

            # Try to read metadata from fetched file
            if "overture" in datasets:
                meta_path = output_dir / f"{overture_prefix}_segments.parquet.meta.yaml"
                if meta_path.exists():
                    from .fetch.metadata import load_metadata

                    meta = load_metadata(output_dir / f"{overture_prefix}_segments.parquet")
                    if meta:
                        feature_count = meta.feature_count
                        geometry_types = meta.geometry_types
            elif "osm" in datasets:
                meta_path = output_dir / f"{osm_name}_segments.parquet.meta.yaml"
                if meta_path.exists():
                    from .fetch.metadata import load_metadata

                    meta = load_metadata(output_dir / f"{osm_name}_segments.parquet")
                    if meta:
                        feature_count = meta.feature_count
                        geometry_types = meta.geometry_types

            config.last_fetch = LastFetch(
                fetched_at=datetime.now(UTC),
                bbox=original_bbox.to_tuple(),
                bbox_buffered=(overture_bbox if "overture" in datasets else osm_bbox).to_tuple()
                if buffer_m
                else None,
                bbox_buffer_m=buffer_m,
                feature_count=feature_count,
                geometry_types=geometry_types,
                output_path=str(output_dir),
            )

            config_path = get_datasets_dir() / f"{dataset_name}.yaml"
            save_dataset_config(config, config_path)
            console.print(f"[blue]Updated last_fetch in {config_path.name}[/blue]")


@app.command()
def topology(
    input_file: Path = typer.Argument(..., help="Input GeoParquet or GeoJSON file"),
    output_dir: Path = typer.Option(
        Path("data/processed"),
        "--output",
        "-o",
        help="Output directory",
    ),
    snap_tolerance_m: float = typer.Option(
        2.0,
        "--snap-tolerance-m",
        "-s",
        help="Snap tolerance for undershoots/overshoots in meters",
    ),
    respect_z_levels: bool = typer.Option(
        True,
        "--respect-z-levels/--ignore-z-levels",
        help="Respect bridge/tunnel z-levels when detecting intersections",
    ),
):
    """Reconstruct topology from spaghetti road data."""
    import geopandas as gpd

    from .topology import planarize

    console.print(f"[blue]Loading {input_file}...[/blue]")
    if input_file.suffix == ".parquet":
        gdf = gpd.read_parquet(input_file)
    else:
        gdf = gpd.read_file(input_file)

    console.print(f"[blue]Loaded {len(gdf)} features[/blue]")
    console.print(
        f"[blue]Planarizing with snap_tolerance_m={snap_tolerance_m}m, "
        f"respect_z_levels={respect_z_levels}...[/blue]"
    )

    network = planarize(
        gdf,
        snap_tolerance_m=snap_tolerance_m,
        respect_z_levels=respect_z_levels,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    network.nodes.to_parquet(output_dir / "nodes.parquet")
    network.edges.to_parquet(output_dir / "edges.parquet")

    console.print(f"[green]Created {len(network.nodes)} nodes, {len(network.edges)} edges[/green]")
    console.print(f"[green]Saved to {output_dir}[/green]")


@app.command()
def match(
    reference: Path = typer.Argument(..., help="Reference edges (Overture)"),
    target: Path = typer.Argument(..., help="Target edges (local data)"),
    output: Path = typer.Option(
        Path("data/output/bridge.parquet"),
        "--output",
        "-o",
        help="Output bridge file path",
    ),
    method: str = typer.Option(
        "rule",
        "--method",
        "-m",
        help="Matching method: rule, xgboost",
    ),
    buffer_distance_m: float = typer.Option(
        50.0,
        "--buffer-m",
        "-b",
        help="Candidate search radius in meters",
    ),
):
    """Run the full matching pipeline."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from .pipeline import run_pipeline

    console.print("[blue]Running matching pipeline...[/blue]")
    console.print(f"  Reference: {reference}")
    console.print(f"  Target: {target}")
    console.print(f"  Method: {method}")
    console.print(f"  Buffer: {buffer_distance_m}m")

    output.parent.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Matching...", total=None)

        result = run_pipeline(
            reference_path=reference,
            target_path=target,
            output_path=output,
            method=method,
            buffer_distance_m=buffer_distance_m,
        )

        progress.update(task, completed=True)

    console.print(f"[green]Matched {result.n_matched} / {result.n_target} features[/green]")
    console.print(f"[green]Bridge file: {output}[/green]")


@app.command()
def evaluate(
    bridge_file: Path = typer.Argument(..., help="Bridge file to evaluate"),
    ground_truth: Path | None = typer.Option(
        None,
        "--ground-truth",
        "-g",
        help="Ground truth labels CSV (columns: gers_id, target_id, label)",
    ),
):
    """Evaluate match quality.

    If ground truth is provided, computes precision/recall/F1 metrics.

    Examples:
        # Basic bridge file stats
        matcher evaluate data/output/boston_streets_bridge.parquet

        # With ground truth evaluation
        matcher evaluate data/output/boston_streets_bridge.parquet \\
            --ground-truth labels/dataset=boston_streets/data.csv
    """
    import pandas as pd

    console.print(f"[blue]Evaluating {bridge_file}...[/blue]")

    bridge = pd.read_parquet(bridge_file)
    console.print(f"Total matches: {len(bridge)}")
    console.print(f"Mean confidence: {bridge['confidence'].mean():.3f}")
    console.print("Confidence distribution:")
    console.print(f"  >= 0.9: {(bridge['confidence'] >= 0.9).sum()}")
    console.print(
        f"  0.75-0.9: {((bridge['confidence'] >= 0.75) & (bridge['confidence'] < 0.9)).sum()}"
    )
    console.print(
        f"  0.5-0.75: {((bridge['confidence'] >= 0.5) & (bridge['confidence'] < 0.75)).sum()}"
    )
    console.print(f"  < 0.5: {(bridge['confidence'] < 0.5).sum()}")

    if ground_truth:
        if not ground_truth.exists():
            console.print(f"[red]Error: Ground truth file not found: {ground_truth}[/red]")
            raise typer.Exit(1)

        # Load ground truth - support both CSV and parquet
        if ground_truth.suffix == ".csv":
            gt_df = pd.read_csv(ground_truth)
        else:
            gt_df = pd.read_parquet(ground_truth)

        console.print()
        console.print("[blue]Ground Truth Evaluation[/blue]")
        console.print(f"  Ground truth file: {ground_truth}")
        console.print(f"  Total labeled pairs: {len(gt_df)}")

        # Build lookup: (gers_id, target_id) -> label
        gt_lookup: dict[tuple[str, str], str] = {}
        for _, row in gt_df.iterrows():
            gers_id = str(row["gers_id"])
            target_id = str(row["target_id"])
            label = row["label"]
            gt_lookup[(gers_id, target_id)] = label

        # Count ground truth labels
        gt_match_count = sum(1 for label in gt_lookup.values() if label == "match")
        gt_no_match_count = sum(1 for label in gt_lookup.values() if label == "no_match")
        console.print(f"  Ground truth matches: {gt_match_count}")
        console.print(f"  Ground truth no_match: {gt_no_match_count}")

        # Build set of predicted pairs from bridge file
        predicted_pairs: set[tuple[str, str]] = set()
        for _, row in bridge.iterrows():
            gers_id = str(row["gers_id"])
            local_id = str(row["local_id"])
            predicted_pairs.add((gers_id, local_id))

        # Warn about predictions not in ground truth
        predictions_not_in_gt = len(predicted_pairs - set(gt_lookup.keys()))
        if predictions_not_in_gt > 0:
            console.print(
                f"  [yellow]Warning: {predictions_not_in_gt} predictions not in ground truth "
                "(excluded from metrics)[/yellow]"
            )

        # Compute metrics (only over ground truth pairs)
        # True Positives: predicted as match AND ground truth is match
        true_positives = sum(
            1
            for (gers_id, target_id), label in gt_lookup.items()
            if label == "match" and (gers_id, target_id) in predicted_pairs
        )

        # False Positives: predicted as match BUT ground truth is no_match
        false_positives = sum(
            1
            for (gers_id, target_id), label in gt_lookup.items()
            if label == "no_match" and (gers_id, target_id) in predicted_pairs
        )

        # False Negatives: ground truth is match BUT not in predictions
        false_negatives = sum(
            1
            for (gers_id, target_id), label in gt_lookup.items()
            if label == "match" and (gers_id, target_id) not in predicted_pairs
        )

        # Calculate precision, recall, F1
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        console.print()
        console.print("[green]Metrics:[/green]")
        console.print(f"  True Positives: {true_positives}")
        console.print(f"  False Positives: {false_positives}")
        console.print(f"  False Negatives: {false_negatives}")
        console.print()
        console.print(f"  [bold]Precision: {precision:.3f}[/bold]")
        console.print(f"  [bold]Recall: {recall:.3f}[/bold]")
        console.print(f"  [bold]F1 Score: {f1:.3f}[/bold]")


@app.command()
def label(
    reference: Path = typer.Argument(
        ...,
        help="Reference edges (Overture segments parquet)",
    ),
    target: Path = typer.Argument(
        ...,
        help="Target edges (local data parquet)",
    ),
    labels_path: Path = typer.Option(
        Path("data/labels/labels.parquet"),
        "--labels",
        "-l",
        help="Path to labels file (created if not exists)",
    ),
    port: int = typer.Option(
        8501,
        "--port",
        "-p",
        help="Streamlit server port",
    ),
):
    """Launch the labeling UI for creating training data.

    Example:
        matcher label data/raw/overture_segments.parquet data/raw/boston_streets.parquet
    """
    import os
    import subprocess
    import sys

    # Set environment variables for the Streamlit app
    env = {
        **os.environ,
        "MATCHER_REFERENCE_PATH": str(reference.absolute()),
        "MATCHER_TARGET_PATH": str(target.absolute()),
        "MATCHER_LABELS_PATH": str(labels_path.absolute()),
    }

    # Find the app.py path
    app_path = Path(__file__).parent / "labeling" / "app.py"

    if not app_path.exists():
        console.print(f"[red]Error: Labeling app not found at {app_path}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Starting labeling UI on port {port}...[/blue]")
    console.print(f"  Reference: {reference}")
    console.print(f"  Target: {target}")
    console.print(f"  Labels: {labels_path}")
    console.print()
    console.print(f"[green]Open http://localhost:{port} in your browser[/green]")

    # Launch Streamlit
    result = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)],
        env=env,
    )
    if result.returncode != 0:
        console.print(f"[red]Error: Streamlit exited with code {result.returncode}[/red]")
        raise typer.Exit(result.returncode)


@app.command()
def train(
    labels_dir: Path = typer.Option(
        Path("labels"),
        "--labels",
        "-l",
        help="Labels directory (Hive-partitioned CSV format)",
    ),
    output: Path = typer.Option(
        Path("data/models/matcher_model_combined.joblib"),
        "--output",
        "-o",
        help="Output path for trained model",
    ),
    exclude_semantic: bool = typer.Option(
        False,
        "--exclude-semantic",
        help="Exclude semantic features (name_*, class_similarity) for geometry-only model",
    ),
):
    """Train an ML model on labeled data.

    Loads labels from Hive-partitioned CSV format (labels/dataset=*/data.csv).

    Examples:
        matcher train
        matcher train --labels labels -o data/models/my_model.joblib

        # Train geometry-only model (no name/class features)
        matcher train --exclude-semantic -o data/models/matcher_model_geom_only.joblib
    """
    from .labeling.label_store import LabelStore
    from .matching.ml import MLMatcher

    if not labels_dir.exists():
        console.print(f"[red]Labels directory not found: {labels_dir}[/red]")
        raise typer.Exit(1)

    # Check for dataset partitions
    partitions = list(labels_dir.glob("dataset=*/data.csv"))
    if not partitions:
        console.print(f"[red]No label partitions found in {labels_dir}[/red]")
        console.print("[yellow]Expected format: labels/dataset=*/data.csv[/yellow]")
        raise typer.Exit(1)

    console.print(f"[blue]Loading labels from {labels_dir}...[/blue]")
    df = LabelStore.load_all(labels_dir)
    console.print(f"  Found {len(df)} labels from {df['dataset'].nunique()} datasets")

    # Train model
    model_type = "geometry-only" if exclude_semantic else "full"
    console.print(f"[blue]Training {model_type} model...[/blue]")
    matcher = MLMatcher()
    metrics = matcher.train(
        labels_dir=labels_dir, test_size=0.2, binary=True, exclude_semantic=exclude_semantic
    )

    # Save model
    output.parent.mkdir(parents=True, exist_ok=True)
    matcher.save_model(str(output))

    console.print(f"\n[green]Model saved to {output}[/green]")
    console.print(f"[green]Holdout accuracy: {metrics['test_accuracy']:.1%}[/green]")


@app.command("eval-model")
def eval_model(
    model: Path = typer.Argument(..., help="Path to trained model"),
    labels_dir: Path = typer.Option(
        Path("labels"),
        "--labels",
        "-l",
        help="Labels directory (Hive-partitioned CSV format)",
    ),
    by_dataset: bool = typer.Option(
        True,
        "--by-dataset/--overall",
        help="Show metrics broken down by dataset",
    ),
    holdout: bool = typer.Option(
        True,
        "--holdout/--no-holdout",
        help="Use 20%% holdout set for evaluation (default: True for unbiased metrics)",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for holdout split (use same seed for comparable results)",
    ),
):
    """Evaluate ML model performance on labeled data.

    By default, evaluates on a 20%% holdout set for unbiased metrics.
    Use --no-holdout to evaluate on ALL data (may include training data).

    Examples:
        matcher eval-model data/models/matcher_model.joblib
        matcher eval-model data/models/combined.joblib --no-holdout
        matcher eval-model data/models/combined.joblib --seed 123
    """
    from .matching.ml import evaluate_by_dataset

    if not model.exists():
        console.print(f"[red]Model not found: {model}[/red]")
        raise typer.Exit(1)

    if holdout:
        console.print(f"[blue]Evaluating {model.name} on 20% holdout (seed={seed})...[/blue]")
    else:
        console.print(
            f"[yellow]Evaluating {model.name} on all data (may include training data)...[/yellow]"
        )

    evaluate_by_dataset(
        str(model), str(labels_dir), show_by_dataset=by_dataset, holdout=holdout, seed=seed
    )

    console.print("[green]Evaluation complete[/green]")


@app.command()
def integrate(
    reference: Path = typer.Argument(
        ...,
        help="Reference edges (Overture segments parquet)",
    ),
    target: list[str] = typer.Option(
        ...,
        "--target",
        "-t",
        help="Target dataset: name:bridge_path:unmatched_path:priority (can specify multiple)",
    ),
    output_dir: Path = typer.Option(
        Path("data/integrated"),
        "--output",
        "-o",
        help="Output directory",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="YAML config file (alternative to --target options)",
    ),
    overlap_threshold: float = typer.Option(
        0.8,
        "--overlap-threshold",
        help="IoU threshold for overlap detection",
    ),
    min_length_m: float = typer.Option(
        3.0,
        "--min-length-m",
        help="Minimum segment length to include (meters)",
    ),
    connection_tolerance_m: float = typer.Option(
        3.0,
        "--connection-tolerance-m",
        help="Distance (meters) to consider segment connected to reference network",
    ),
    min_merge_length_m: float = typer.Option(
        20.0,
        "--min-merge-length-m",
        help="Minimum net-new length (meters) to merge a connected segment",
    ),
    net_new_buffer_m: float = typer.Option(
        5.0,
        "--net-new-buffer-m",
        help="Buffer around reference (meters) for net-new calculation",
    ),
    max_hops: int = typer.Option(
        2,
        "--max-hops",
        help="Maximum transitive connectivity hops from reference network",
    ),
    fringe_buffer_m: float = typer.Option(
        50.0,
        "--fringe-buffer-m",
        help="Buffer around reference coverage (meters) for fringe detection",
    ),
    no_fringe_filter: bool = typer.Option(
        False,
        "--no-fringe-filter",
        help="Disable fringe detection (include all segments regardless of coverage)",
    ),
    transitive_tolerance_m: float = typer.Option(
        None,
        "--transitive-tolerance-m",
        help="Tolerance (meters) for transitive connections between targets. Defaults to 2x connection-tolerance-m.",
    ),
    debug_connectivity: bool = typer.Option(
        False,
        "--debug-connectivity",
        help="Enable debug logging for transitive connectivity analysis",
    ),
):
    """Integrate unmatched segments into reference network.

    Takes the output of the matching pipeline and creates a unified
    planarized network, flagging disconnected orphan components for QA.

    Examples:
        # Single dataset
        matcher integrate data/raw/overture.parquet \\
            -t boston_streets:data/output/bridge.parquet:data/output/unmatched.parquet:1 \\
            -o data/integrated

        # Multiple datasets with priority
        matcher integrate data/raw/overture.parquet \\
            -t boston_streets:data/boston_streets/bridge.parquet:data/boston_streets/unmatched.parquet:1 \\
            -t boston_bikes:data/boston_bikes/bridge.parquet:data/boston_bikes/unmatched.parquet:2 \\
            -o data/integrated

        # From config file
        matcher integrate data/raw/overture.parquet -c integration_config.yaml -o data/integrated
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from .integration import TargetConfig, run_integration_from_config, run_integration_pipeline

    if config:
        # Use config file
        console.print(f"[blue]Running integration from config: {config}[/blue]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Integrating...", total=None)
            result = run_integration_from_config(config, output_dir)
            progress.update(task, completed=True)
    else:
        # Parse target options
        target_configs = []
        for t in target:
            parts = t.split(":")
            if len(parts) != 4:
                console.print(
                    "[red]Error: Target must be name:bridge_path:unmatched_path:priority[/red]"
                )
                raise typer.Exit(1)

            name, bridge_path, unmatched_path, priority = parts
            target_configs.append(
                TargetConfig(
                    name=name,
                    bridge_path=Path(bridge_path),
                    unmatched_path=Path(unmatched_path),
                    priority=int(priority),
                )
            )

        console.print("[blue]Running integration pipeline...[/blue]")
        console.print(f"  Reference: {reference}")
        console.print(f"  Targets: {len(target_configs)}")
        for tc in target_configs:
            console.print(f"    - {tc.name} (priority {tc.priority})")
        console.print(f"  Output: {output_dir}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Integrating...", total=None)
            result = run_integration_pipeline(
                reference_path=reference,
                target_configs=target_configs,
                output_dir=output_dir,
                overlap_iou_threshold=overlap_threshold,
                min_segment_length_m=min_length_m,
                connection_tolerance_m=connection_tolerance_m,
                min_merge_length_m=min_merge_length_m,
                net_new_buffer_m=net_new_buffer_m,
                max_hops=max_hops,
                fringe_buffer_m=fringe_buffer_m,
                enable_fringe_detection=not no_fringe_filter,
                transitive_tolerance_m=transitive_tolerance_m,
                debug_connectivity=debug_connectivity,
            )
            progress.update(task, completed=True)

    # Print summary
    stats = result.statistics
    console.print()
    console.print("[green]Integration complete![/green]")
    console.print(f"  Reference edges: {stats.reference_edges}")
    console.print(f"  Target edges (matched): {stats.target_edges_matched}")
    console.print(f"  Target edges (unmatched): {stats.target_edges_unmatched}")
    console.print(f"  Dropped overlaps: {stats.dropped_overlaps}")
    console.print(f"  Total nodes: {stats.total_nodes}")
    console.print(f"  Total edges: {stats.total_edges}")
    console.print(f"  Main component edges: {stats.main_component_edges}")
    console.print(f"  Orphan edges: {stats.orphan_edges}")
    console.print(f"  Orphan components: {stats.orphan_components}")
    console.print()
    console.print(f"[green]Outputs saved to {output_dir}[/green]")


@app.command("qa-integration")
def qa_integration(
    output_dir: Path = typer.Option(
        Path("data/integrated"),
        "--output",
        "-o",
        help="Integration output directory",
    ),
    port: int = typer.Option(
        8502,
        "--port",
        "-p",
        help="Streamlit server port",
    ),
    host: str = typer.Option(
        "localhost",
        "--host",
        "-H",
        help="Server host (use 0.0.0.0 to expose on all interfaces)",
    ),
):
    """Launch the integration QA app.

    Review orphan components and merged edges from the integration pipeline.

    Example:
        matcher qa-integration -o data/integrated
        matcher qa-integration -o data/integrated --host 0.0.0.0
    """
    import os
    import subprocess
    import sys

    # Set environment variables
    env = {
        **os.environ,
        "INTEGRATION_DIR": str(output_dir.absolute()),
    }

    # Find the app.py path
    app_path = Path(__file__).parent / "integration_qa" / "app.py"

    if not app_path.exists():
        console.print(f"[red]Error: QA app not found at {app_path}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Starting integration QA on port {port}...[/blue]")
    console.print(f"  Integration output: {output_dir}")
    console.print()
    display_host = "localhost" if host == "0.0.0.0" else host
    console.print(f"[green]Open http://{display_host}:{port} in your browser[/green]")

    # Launch Streamlit
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            str(port),
            "--server.address",
            host,
        ],
        env=env,
    )
    if result.returncode != 0:
        console.print(f"[red]Error: Streamlit exited with code {result.returncode}[/red]")
        raise typer.Exit(result.returncode)


@app.command()
def validate(
    overture: Path = typer.Argument(
        ...,
        help="Path to Overture segments parquet file",
    ),
    bbox: str = typer.Option(
        ...,
        "--bbox",
        "-b",
        help="Bounding box for fresh OSM fetch: xmin,ymin,xmax,ymax",
    ),
    output: Path = typer.Option(
        Path("validation/experiment"),
        "--output",
        "-o",
        help="Output directory for experiment results",
    ),
    strategy: str = typer.Option(
        "random",
        "--strategy",
        "-s",
        help="Drop strategy: random, bbox, source, class",
    ),
    method: str = typer.Option(
        "rule",
        "--method",
        "-m",
        help="Matching method: rule, xgboost",
    ),
    fraction: float = typer.Option(
        0.1,
        "--fraction",
        "-f",
        help="Fraction to drop for 'random' strategy (0.0-1.0)",
    ),
    source_dataset: str = typer.Option(
        "TomTom",
        "--source-dataset",
        help="Dataset to drop for 'source' strategy",
    ),
    road_class: str = typer.Option(
        "residential",
        "--road-class",
        help="Road class to drop for 'class' strategy",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
    fast: bool = typer.Option(
        False,
        "--fast",
        help="Fast mode: only match segments that should match (dropped record_ids)",
    ),
):
    """Run a validation experiment using ground-truth from Overture provenance.

    This command:
    1. Drops segments from Overture based on the chosen strategy
    2. Fetches fresh OSM data for the bounding box
    3. Runs the matcher to see if dropped segments get matched back
    4. Evaluates results and computes recall metrics

    Examples:
        # Drop 10% of OSM segments randomly
        matcher validate data/raw/overture.parquet \\
            --bbox "-71.19,42.21,-70.92,42.40" \\
            --strategy random --fraction 0.1 \\
            --output validation/random_10pct/

        # Drop all TomTom segments
        matcher validate data/raw/overture.parquet \\
            --bbox "-71.19,42.21,-70.92,42.40" \\
            --strategy source --source-dataset TomTom \\
            --output validation/tomtom_holdout/

        # Drop residential roads
        matcher validate data/raw/overture.parquet \\
            --bbox "-71.19,42.21,-70.92,42.40" \\
            --strategy class --road-class residential \\
            --output validation/residential_holdout/
    """
    from .validation import run_validation_experiment

    # Parse bbox
    try:
        coords = [float(x.strip()) for x in bbox.split(",")]
        if len(coords) != 4:
            raise ValueError()
        bbox_tuple = tuple(coords)
    except ValueError:
        console.print(
            "[red]Error: bbox must be 4 comma-separated values: xmin,ymin,xmax,ymax[/red]"
        )
        raise typer.Exit(1) from None

    # Validate strategy
    valid_strategies = {"random", "bbox", "source", "class"}
    if strategy not in valid_strategies:
        console.print(f"[red]Error: strategy must be one of {valid_strategies}[/red]")
        raise typer.Exit(1)

    # Validate fraction for random strategy
    if strategy == "random" and not 0.0 <= fraction <= 1.0:
        console.print("[red]Error: fraction must be between 0.0 and 1.0[/red]")
        raise typer.Exit(1)

    # Validate input file
    if not overture.exists():
        console.print(f"[red]Error: Overture file not found: {overture}[/red]")
        raise typer.Exit(1)

    console.print("[blue]Starting validation experiment...[/blue]")
    console.print(f"  Overture: {overture}")
    console.print(f"  Strategy: {strategy}")
    console.print(f"  Matcher: {method}")
    console.print(f"  Output: {output}")

    try:
        result = run_validation_experiment(
            overture_path=overture,
            output_dir=output,
            bbox=bbox_tuple,
            strategy=strategy,
            matcher_method=method,
            fraction=fraction,
            source_dataset=source_dataset,
            road_class=road_class,
            seed=seed,
            fast_mode=fast,
        )

        console.print()
        console.print("[green]Experiment complete![/green]")
        console.print(f"  Overture segments: {result.n_overture}")
        console.print(f"  Dropped segments: {result.n_dropped}")
        console.print(f"  Fresh OSM segments: {result.n_fresh_osm}")
        console.print(f"  Matched: {result.n_matched}")
        console.print(f"  Unmatched: {result.n_unmatched}")
        console.print()
        console.print(f"  [bold]Recall: {result.metrics['recall']:.3f}[/bold]")
        if result.metrics.get("mean_confidence"):
            console.print(f"  Mean confidence: {result.metrics['mean_confidence']:.3f}")
        console.print()
        console.print(f"[green]Results saved to {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("discover-classes")
def discover_classes(
    dataset: Path = typer.Argument(..., help="Target dataset parquet file"),
    reference: Path | None = typer.Option(
        None,
        "--reference",
        "-r",
        help="Reference data (Overture segments) for match-based analysis",
    ),
    bridge: Path | None = typer.Option(
        None,
        "--bridge",
        "-b",
        help="Existing bridge file for match-based analysis",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output YAML path (default: datasets/{name}.yaml). Merges with existing config if present.",
    ),
    print_only: bool = typer.Option(
        False,
        "--print-only",
        "-p",
        help="Only print report, don't save config",
    ),
):
    """Discover class mapping for a new dataset.

    Analyzes the dataset structure, detects classification columns,
    and generates a mapping configuration to Overture road classes.

    Examples:
        # Basic discovery
        matcher discover-classes data/raw/boston_streets.parquet

        # With match-based analysis (more accurate)
        matcher discover-classes data/raw/boston_streets.parquet \\
            --reference data/raw/overture_segments.parquet \\
            --bridge data/output/boston_streets_bridge.parquet

        # Print report only (don't save config)
        matcher discover-classes data/raw/new_dataset.parquet --print-only
    """
    from .datasets.discover import discover_dataset, print_discovery_report, save_dataset_config

    if not dataset.exists():
        console.print(f"[red]Error: Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    if reference and not reference.exists():
        console.print(f"[red]Error: Reference not found: {reference}[/red]")
        raise typer.Exit(1)

    if bridge and not bridge.exists():
        console.print(f"[red]Error: Bridge file not found: {bridge}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Analyzing dataset: {dataset.name}[/blue]")
    if reference:
        console.print(f"  Reference: {reference}")
    if bridge:
        console.print(f"  Bridge: {bridge}")
    console.print()

    report = discover_dataset(
        dataset_path=dataset,
        reference_path=reference,
        bridge_path=bridge,
    )

    print_discovery_report(report)

    if not print_only and report.suggested_config:
        from .datasets.schema import (
            ClassificationConfig,
            get_dataset_config,
            get_datasets_dir,
        )
        from .datasets.schema import (
            ClassMappingRule as NewClassMappingRule,
        )
        from .datasets.schema import (
            SourceClassification as NewSourceClassification,
        )
        from .datasets.schema import (
            save_dataset_config as save_new_config,
        )

        dataset_name = dataset.stem
        if output:
            saved_path = output
        else:
            saved_path = get_datasets_dir() / f"{dataset_name}.yaml"

        # Check if config already exists and merge
        existing_config = get_dataset_config(dataset_name)
        if existing_config:
            console.print(f"[blue]Merging with existing config: {dataset_name}.yaml[/blue]")

            # Build new classification from discovered rules
            old_config = report.suggested_config
            new_rules = []
            for rule in old_config.class_mapping_rules:
                new_rules.append(
                    NewClassMappingRule(
                        source_value=rule.source_value,
                        target_class=rule.target_class,
                        conditions=rule.conditions,
                        priority=rule.priority,
                    )
                )

            new_source_class = None
            if old_config.source_classification:
                new_source_class = NewSourceClassification(
                    column=old_config.source_classification.column,
                    description=old_config.source_classification.description,
                    values=old_config.source_classification.values,
                    documentation_url=old_config.source_classification.documentation_url,
                )

            existing_config.classification = ClassificationConfig(
                source_classification=new_source_class,
                class_mapping_rules=new_rules,
                default_class=old_config.default_class,
                confidence=old_config.confidence,
            )

            if old_config.notes:
                existing_config.notes = old_config.notes

            save_new_config(existing_config, saved_path)
        else:
            # No existing config - save using old format for now
            # TODO: Create new format config from scratch
            saved_path = save_dataset_config(report.suggested_config, output)

        console.print()
        console.print(f"[green]Configuration saved to: {saved_path}[/green]")
        console.print("[yellow]Review and adjust the mapping rules as needed.[/yellow]")


@app.command("list-datasets")
def list_datasets():
    """List available dataset configurations."""
    from .datasets.schema import get_datasets_dir, list_dataset_configs

    configs = list_dataset_configs()
    console.print(f"[blue]Dataset configs directory: {get_datasets_dir()}[/blue]")
    if not configs:
        console.print("[yellow]No dataset configurations found.[/yellow]")
        console.print("Use 'matcher discover-classes' to create one.")
        return

    console.print("[blue]Available dataset configurations:[/blue]")
    for name in sorted(configs):
        console.print(f"  - {name}")


@app.command("generate-agent-batch")
def generate_agent_batch(
    dataset: str = typer.Argument(..., help="Target dataset name (e.g., 'boston_streets')"),
    n_candidates: int = typer.Option(
        100,
        "--n-candidates",
        "-n",
        help="Number of candidates to sample",
    ),
    output_dir: Path = typer.Option(
        Path("agent_labels"),
        "--output",
        "-o",
        help="Output directory for agent labeling batches",
    ),
    reference: Path = typer.Option(
        Path("data/raw/overture_segments.parquet"),
        "--reference",
        "-r",
        help="Reference segments (Overture)",
    ),
    target: Path | None = typer.Option(
        None,
        "--target",
        "-t",
        help="Target segments (inferred from dataset name if not provided)",
    ),
    model: Path | None = typer.Option(
        None,
        "--model",
        "-m",
        help="ML model for confidence scoring (uses rules if not provided)",
    ),
    no_satellite: bool = typer.Option(
        False,
        "--no-satellite",
        help="Skip satellite imagery (faster, geometry-only images)",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
):
    """Generate a batch of candidates for AI agent labeling.

    Samples diverse candidates across confidence ranges and creates
    packages with metadata YAML and images for each candidate.

    Examples:
        # Generate 100 candidates for boston_streets
        matcher generate-agent-batch boston_streets

        # Generate 50 candidates with custom paths
        matcher generate-agent-batch boston_streets -n 50 \\
            -r data/raw/overture_segments.parquet \\
            -t data/raw/boston_streets.parquet \\
            -o agent_labels

        # Use ML model for confidence scoring
        matcher generate-agent-batch boston_streets \\
            -m data/models/matcher_model_combined.joblib
    """
    from .agent_labeling import SamplingConfig, sample_candidates
    from .agent_labeling.context_generator import generate_batch

    # Infer target path if not provided
    if target is None:
        target = Path(f"data/raw/{dataset}.parquet")

    # Validate paths
    if not reference.exists():
        console.print(f"[red]Error: Reference file not found: {reference}[/red]")
        raise typer.Exit(1)

    if not target.exists():
        console.print(f"[red]Error: Target file not found: {target}[/red]")
        console.print("[yellow]Hint: Specify target path with --target[/yellow]")
        raise typer.Exit(1)

    if model and not model.exists():
        console.print(
            f"[yellow]Warning: Model not found: {model}, using rule-based scoring[/yellow]"
        )
        model = None

    console.print("[blue]Generating agent labeling batch...[/blue]")
    console.print(f"  Dataset: {dataset}")
    console.print(f"  Reference: {reference}")
    console.print(f"  Target: {target}")
    console.print(f"  Candidates: {n_candidates}")
    console.print(f"  Output: {output_dir}")

    # Sample candidates
    config = SamplingConfig(
        n_candidates=n_candidates,
        seed=seed,
    )

    candidates = sample_candidates(
        reference_path=reference,
        target_path=target,
        config=config,
        dataset_name=dataset,
        model_path=model,
    )

    if not candidates:
        console.print("[red]Error: No candidates generated[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Sampled {len(candidates)} candidates[/green]")

    # Generate batch
    batch_dir = generate_batch(
        candidates=candidates,
        output_dir=output_dir,
        dataset_name=dataset,
        fetch_satellite=not no_satellite,
        config_info={
            "n_candidates": n_candidates,
            "seed": seed,
            "reference": str(reference),
            "target": str(target),
            "model": str(model) if model else None,
        },
    )

    console.print()
    console.print(f"[green]Batch generated at {batch_dir}[/green]")
    console.print()
    console.print("Next steps:")
    console.print(f"  1. Review candidates in {batch_dir / 'candidates'}")
    console.print("  2. Have agents label candidates")
    console.print(f"  3. Import labels: matcher import-agent-labels {batch_dir} --agent-id <id>")


@app.command("import-agent-labels")
def import_agent_labels(
    batch_dir: Path = typer.Argument(..., help="Batch directory"),
    agent_id: str = typer.Option(
        ...,
        "--agent-id",
        "-a",
        help="Agent identifier (e.g., 'claude', 'gpt4', 'human')",
    ),
    labels_file: Path = typer.Option(
        ...,
        "--labels",
        "-l",
        help="Path to labels CSV file",
    ),
):
    """Import agent labels from a CSV file.

    The CSV must have columns: ref_id, target_id, label
    Optional columns: confidence, reasoning

    Examples:
        # Import Claude's labels
        matcher import-agent-labels agent_labels/batches/batch_2026-01-18_001 \\
            --agent-id claude --labels claude_labels.csv

        # Import with confidence and reasoning
        matcher import-agent-labels agent_labels/batches/batch_* \\
            -a gpt4 -l gpt4_labels.csv
    """
    from .agent_labeling.agent_store import import_labels_csv

    # Validate paths
    if not batch_dir.exists():
        console.print(f"[red]Error: Batch directory not found: {batch_dir}[/red]")
        raise typer.Exit(1)

    if not labels_file.exists():
        console.print(f"[red]Error: Labels file not found: {labels_file}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Importing labels for agent '{agent_id}'...[/blue]")
    console.print(f"  Batch: {batch_dir}")
    console.print(f"  Labels: {labels_file}")

    count = import_labels_csv(batch_dir, agent_id, labels_file)

    console.print(f"[green]Imported {count} labels[/green]")


@app.command("agent-consensus")
def agent_consensus(
    batch_dir: Path = typer.Argument(..., help="Batch directory"),
    show_disagreements: bool = typer.Option(
        False,
        "--disagreements",
        "-d",
        help="Show only disagreements between agents",
    ),
    min_agents: int = typer.Option(
        2,
        "--min-agents",
        help="Minimum agents required for consensus",
    ),
):
    """Analyze agent consensus and disagreements.

    Shows agreement statistics across multiple agents and identifies
    candidates where agents disagree for human review.

    Examples:
        # Show consensus summary
        matcher agent-consensus agent_labels/batches/batch_2026-01-18_001

        # Show disagreements only
        matcher agent-consensus agent_labels/batches/batch_* --disagreements
    """
    from .agent_labeling.agent_store import AgentLabelStore

    if not batch_dir.exists():
        console.print(f"[red]Error: Batch directory not found: {batch_dir}[/red]")
        raise typer.Exit(1)

    # List agents
    agents = AgentLabelStore.list_agents(batch_dir)
    if not agents:
        console.print("[yellow]No agent labels found in batch[/yellow]")
        return

    console.print(f"[blue]Agents who have labeled: {', '.join(agents)}[/blue]")
    console.print()

    # Show per-agent stats
    for agent_id in agents:
        store = AgentLabelStore(batch_dir, agent_id)
        stats = store.get_stats()
        console.print(f"  {agent_id}: {stats['total']} labels")
        console.print(
            f"    match: {stats['match']}, no_match: {stats['no_match']}, unsure: {stats['unsure']}"
        )

    console.print()

    if show_disagreements:
        # Show disagreements
        disagreements = AgentLabelStore.find_disagreements(batch_dir)
        if len(disagreements) == 0:
            console.print("[green]No disagreements found![/green]")
            return

        console.print(f"[yellow]Found {len(disagreements)} disagreements:[/yellow]")
        for _, row in disagreements.iterrows():
            console.print(f"  {row['ref_id']} <-> {row['target_id']}")
            console.print(f"    Labels: {row['labels']}")
            console.print(f"    Agreement: {row['agreement_ratio']:.0%}")
    else:
        # Show consensus
        consensus = AgentLabelStore.compute_consensus(batch_dir, min_agents)
        if len(consensus) == 0:
            console.print(f"[yellow]No candidates have >= {min_agents} agent labels[/yellow]")
            return

        console.print(f"[green]Consensus on {len(consensus)} candidates:[/green]")

        # Summary by consensus label
        label_counts = consensus["consensus_label"].value_counts()
        for label, count in label_counts.items():
            console.print(f"  {label}: {count}")

        # Agreement distribution
        mean_agreement = consensus["agreement_ratio"].mean()
        console.print(f"\n  Mean agreement: {mean_agreement:.0%}")

        # Count perfect agreement
        perfect = (consensus["agreement_ratio"] == 1.0).sum()
        console.print(f"  Perfect agreement: {perfect}/{len(consensus)}")


@app.command("generate-agent-test-batch")
def generate_agent_test_batch(
    n_samples: int = typer.Option(
        100,
        "--n-samples",
        "-n",
        help="Number of labeled pairs to sample for testing",
    ),
    output_dir: Path = typer.Option(
        Path("agent_labels"),
        "--output",
        "-o",
        help="Output directory for agent labeling batches",
    ),
    labels_dir: Path = typer.Option(
        Path("labels"),
        "--labels",
        "-l",
        help="Directory containing human labels (Hive-partitioned)",
    ),
    reference: Path = typer.Option(
        Path("data/raw/overture_segments.parquet"),
        "--reference",
        "-r",
        help="Reference segments (Overture)",
    ),
    datasets: list[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Datasets to include (can specify multiple). If not specified, uses all.",
    ),
    labeler: str | None = typer.Option(
        None,
        "--labeler",
        help="Filter by labeler name",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
    no_satellite: bool = typer.Option(
        False,
        "--no-satellite",
        help="Skip satellite imagery (faster, geometry-only images)",
    ),
):
    """Generate a batch from existing human labels for agent agreement testing.

    Unlike generate-agent-batch which samples NEW candidates, this command
    uses existing human-labeled pairs so you can measure agent agreement
    with human ground truth.

    Examples:
        # Generate 200 samples across all datasets
        matcher generate-agent-test-batch -n 200

        # Specific datasets only
        matcher generate-agent-test-batch -n 100 -d boston_streets -d boston_bikes

        # Filter by labeler
        matcher generate-agent-test-batch -n 50 --labeler brad
    """
    from datetime import UTC, datetime

    import geopandas as gpd
    import pandas as pd

    from .agent_labeling.context_generator import write_candidate_package
    from .agent_labeling.sampler import SampledCandidate

    # Load human labels
    if not labels_dir.exists():
        console.print(f"[red]Error: Labels directory not found: {labels_dir}[/red]")
        raise typer.Exit(1)

    # Find all label files
    label_files = list(labels_dir.glob("dataset=*/data.csv"))
    if not label_files:
        console.print(f"[red]Error: No label files found in {labels_dir}[/red]")
        raise typer.Exit(1)

    # Load and combine labels
    all_labels = []
    for f in label_files:
        dataset = f.parent.name.replace("dataset=", "")
        if datasets and dataset not in datasets:
            continue
        df = pd.read_csv(f)
        df["dataset"] = dataset
        all_labels.append(df)

    if not all_labels:
        console.print("[red]Error: No labels found for specified datasets[/red]")
        raise typer.Exit(1)

    labels_df = pd.concat(all_labels, ignore_index=True)

    # Filter by labeler if specified
    if labeler and "labeler" in labels_df.columns:
        labels_df = labels_df[labels_df["labeler"].str.lower() == labeler.lower()]
        if len(labels_df) == 0:
            console.print(f"[red]Error: No labels found for labeler '{labeler}'[/red]")
            raise typer.Exit(1)

    # Filter to match/no_match only (exclude unsure for cleaner testing)
    if "label" in labels_df.columns:
        labels_df = labels_df[labels_df["label"].isin(["match", "no_match"])]
        if len(labels_df) == 0:
            console.print("[red]Error: No match/no_match labels found after filtering[/red]")
            raise typer.Exit(1)

    console.print(f"[blue]Found {len(labels_df)} labeled pairs[/blue]")

    # Stratified sample across datasets
    import numpy as np

    rng = np.random.default_rng(seed)

    sampled = []
    for dataset in labels_df["dataset"].unique():
        dataset_df = labels_df[labels_df["dataset"] == dataset]
        n_dataset = max(1, int(n_samples * len(dataset_df) / len(labels_df)))
        n_dataset = min(n_dataset, len(dataset_df))
        indices = rng.choice(len(dataset_df), size=n_dataset, replace=False)
        sampled.append(dataset_df.iloc[indices])

    sampled_df = pd.concat(sampled, ignore_index=True)
    console.print(f"[blue]Sampled {len(sampled_df)} pairs for testing[/blue]")

    # Load reference data
    if not reference.exists():
        console.print(f"[red]Error: Reference file not found: {reference}[/red]")
        raise typer.Exit(1)

    ref_gdf = gpd.read_parquet(reference)
    ref_lookup = ref_gdf.set_index("id")

    # Load target datasets
    target_gdfs = {}
    dataset_paths = {
        "boston_streets": Path("data/raw/boston_streets.parquet"),
        "boston_sidewalks": Path("data/raw/boston_sidewalks.parquet"),
        "boston_bikes": Path("data/raw/boston_bike_network.parquet"),
        "osm": Path("data/raw/osm_segments.parquet"),
    }

    for dataset in sampled_df["dataset"].unique():
        path = dataset_paths.get(dataset)
        if path and path.exists():
            target_gdfs[dataset] = gpd.read_parquet(path).set_index("id")
        else:
            console.print(f"[yellow]Warning: No data file for {dataset}[/yellow]")

    # Generate batch
    batch_id = f"test_batch_{datetime.now(UTC).strftime('%Y-%m-%d_%H%M%S')}"
    batch_dir = output_dir / "batches" / batch_id
    candidates_dir = batch_dir / "candidates"
    labels_out_dir = batch_dir / "labels"

    batch_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[blue]Generating batch: {batch_id}[/blue]")

    # Build SampledCandidate objects and write packages
    candidates = []
    for _, row in sampled_df.iterrows():
        ref_id = row["gers_id"]
        target_id = row["target_id"]
        dataset = row["dataset"]

        if dataset not in target_gdfs:
            continue

        try:
            ref_row = ref_lookup.loc[ref_id]
            target_row = target_gdfs[dataset].loc[target_id]
        except KeyError:
            continue

        # Extract features from row (if available)
        feature_cols = [
            "hausdorff_distance",
            "buffer_iou",
            "heading_delta",
            "length_ratio",
            "name_levenshtein",
            "name_jaro_winkler",
            "class_similarity",
            "centroid_distance",
            "overlap_ratio",
            "mean_hausdorff_distance",
            "degree_match_score",
            "dead_end_match",
            "intersection_match",
        ]
        features = {col: row.get(col, 0.0) for col in feature_cols if col in row.index}

        candidate = SampledCandidate(
            ref_id=str(ref_id),
            target_id=str(target_id),
            ref_geometry=ref_row.geometry,
            target_geometry=target_row.geometry,
            ref_name=ref_row.get("names") if hasattr(ref_row, "get") else None,
            target_name=target_row.get("names") if hasattr(target_row, "get") else None,
            ref_class=ref_row.get("class") if hasattr(ref_row, "get") else None,
            target_class=target_row.get("class") if hasattr(target_row, "get") else None,
            ml_confidence=row.get("original_confidence", 0.5),
            ml_decision=row.get("original_decision", "review"),
            features=features,
            dataset=dataset,
            confidence_bucket="ground_truth",
        )
        candidates.append(candidate)

        # Write candidate package with images
        write_candidate_package(
            output_dir=candidates_dir,
            candidate=candidate,
            batch_id=batch_id,
            fetch_satellite=not no_satellite,
        )

        if (len(candidates)) % 20 == 0:
            console.print(f"  Progress: {len(candidates)}/{len(sampled_df)}")

    # Write ground truth labels
    ground_truth_path = labels_out_dir / "ground_truth" / "data.csv"
    ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df[["gers_id", "target_id", "label", "dataset"]].rename(
        columns={"gers_id": "ref_id"}
    ).to_csv(ground_truth_path, index=False)

    # Write manifest with ground truth info
    import yaml

    manifest = {
        "batch_id": batch_id,
        "batch_type": "agent_test",
        "created_at": datetime.now(UTC).isoformat(),
        "total_candidates": len(candidates),
        "datasets": list(sampled_df["dataset"].unique()),
        "labeler_filter": labeler,
        "ground_truth": {
            "file": "labels/ground_truth/data.csv",
            "total": len(sampled_df),
            "by_label": sampled_df["label"].value_counts().to_dict(),
            "by_dataset": sampled_df["dataset"].value_counts().to_dict(),
        },
        "candidates": [
            {"ref_id": c.ref_id, "target_id": c.target_id, "dataset": c.dataset} for c in candidates
        ],
    }
    (batch_dir / "manifest.yaml").write_text(
        yaml.dump(manifest, default_flow_style=False, sort_keys=False)
    )

    console.print()
    console.print(f"[green]Batch generated at {batch_dir}[/green]")
    console.print(f"  Candidates: {len(candidates)}")
    console.print(f"  Ground truth: {ground_truth_path}")
    console.print()
    console.print("Next steps:")
    console.print("  1. Have agents label candidates in candidates/")
    console.print(
        f"  2. Import labels: matcher import-agent-labels {batch_dir} -a <agent-id> -l <labels.csv>"
    )
    console.print(f"  3. Compare: matcher agent-consensus {batch_dir}")


@app.command()
def version():
    """Show version information."""
    from . import __version__

    console.print(f"matcher version {__version__}")


if __name__ == "__main__":
    app()
