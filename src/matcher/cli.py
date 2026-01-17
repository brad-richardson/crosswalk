"""CLI entry point for the road network conflation pipeline."""

from pathlib import Path
from typing import Optional

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
    bbox: str = typer.Option(
        ...,
        "--bbox",
        "-b",
        help="Bounding box: xmin,ymin,xmax,ymax (EPSG:4326)",
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
    cache_dir: Optional[Path] = typer.Option(
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
):
    """Fetch road data for an area of interest.

    Examples:
        matcher fetch --bbox -122.7,45.5,-122.6,45.55                    # Overture (default)
        matcher fetch --bbox -122.7,45.5,-122.6,45.55 -d osm             # OSM only
        matcher fetch --bbox -122.7,45.5,-122.6,45.55 -d overture -d osm # Both

    Note: OSM fetching requires osmium-tool to be installed:
        brew install osmium-tool (macOS) or apt install osmium-tool (Ubuntu)
    """
    from .fetch import overture as ov_module, osm as osm_module

    # Validate datasets
    valid_datasets = {"overture", "osm"}
    datasets = {d.lower() for d in dataset}
    invalid = datasets - valid_datasets
    if invalid:
        console.print(f"[red]Error: Invalid dataset(s): {invalid}. Must be 'overture' or 'osm'[/red]")
        raise typer.Exit(1)

    coords = [float(x.strip()) for x in bbox.split(",")]
    if len(coords) != 4:
        console.print("[red]Error: bbox must have 4 values: xmin,ymin,xmax,ymax[/red]")
        raise typer.Exit(1)

    xmin, ymin, xmax, ymax = coords
    output_dir.mkdir(parents=True, exist_ok=True)
    bbox_obj = ov_module.BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    if "overture" in datasets:
        console.print(f"[blue]Fetching Overture segments for bbox {bbox}...[/blue]")
        segments_path = ov_module.fetch_overture_segments(
            bbox=bbox_obj,
            output_path=output_dir / "overture_segments.parquet",
        )
        console.print(f"[green]Saved Overture segments to {segments_path}[/green]")

        console.print("[blue]Fetching Overture connectors...[/blue]")
        connectors_path = ov_module.fetch_overture_connectors(
            bbox=bbox_obj,
            output_path=output_dir / "overture_connectors.parquet",
        )
        console.print(f"[green]Saved Overture connectors to {connectors_path}[/green]")

    if "osm" in datasets:
        console.print(f"[blue]Fetching OSM data for bbox {bbox}...[/blue]")
        segments_path, connectors_path = osm_module.fetch_osm_data(
            bbox=bbox_obj,
            output_dir=output_dir,
            cache_dir=cache_dir,
            force_download=no_cache,
            keep_pbf=keep_pbf,
        )
        console.print(f"[green]Saved OSM segments to {segments_path}[/green]")
        console.print(f"[green]Saved OSM connectors to {connectors_path}[/green]")


@app.command()
def topology(
    input_file: Path = typer.Argument(..., help="Input GeoParquet or GeoJSON file"),
    output_dir: Path = typer.Option(
        Path("data/processed"),
        "--output",
        "-o",
        help="Output directory",
    ),
    snap_tolerance: float = typer.Option(
        2.0,
        "--snap-tolerance",
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
        f"[blue]Planarizing with snap_tolerance={snap_tolerance}m, "
        f"respect_z_levels={respect_z_levels}...[/blue]"
    )

    network = planarize(
        gdf,
        snap_tolerance=snap_tolerance,
        respect_z_levels=respect_z_levels,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    network.nodes.to_parquet(output_dir / "nodes.parquet")
    network.edges.to_parquet(output_dir / "edges.parquet")

    console.print(
        f"[green]Created {len(network.nodes)} nodes, {len(network.edges)} edges[/green]"
    )
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
    buffer_distance: float = typer.Option(
        50.0,
        "--buffer",
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
    console.print(f"  Buffer: {buffer_distance}m")

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
            buffer_distance=buffer_distance,
        )

        progress.update(task, completed=True)

    console.print(f"[green]Matched {result.n_matched} / {result.n_target} features[/green]")
    console.print(f"[green]Bridge file: {output}[/green]")


@app.command()
def evaluate(
    bridge_file: Path = typer.Argument(..., help="Bridge file to evaluate"),
    ground_truth: Optional[Path] = typer.Option(
        None,
        "--ground-truth",
        "-g",
        help="Ground truth matches for precision/recall",
    ),
):
    """Evaluate match quality."""
    import pandas as pd

    console.print(f"[blue]Evaluating {bridge_file}...[/blue]")

    bridge = pd.read_parquet(bridge_file)
    console.print(f"Total matches: {len(bridge)}")
    console.print(f"Mean confidence: {bridge['confidence'].mean():.3f}")
    console.print(f"Confidence distribution:")
    console.print(f"  >= 0.9: {(bridge['confidence'] >= 0.9).sum()}")
    console.print(f"  0.75-0.9: {((bridge['confidence'] >= 0.75) & (bridge['confidence'] < 0.9)).sum()}")
    console.print(f"  0.5-0.75: {((bridge['confidence'] >= 0.5) & (bridge['confidence'] < 0.75)).sum()}")
    console.print(f"  < 0.5: {(bridge['confidence'] < 0.5).sum()}")

    if ground_truth:
        gt = pd.read_parquet(ground_truth)
        # TODO: Compute precision/recall against ground truth
        console.print("[yellow]Ground truth evaluation not yet implemented[/yellow]")


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
    labels: Path = typer.Option(
        Path("data/labels"),
        "--labels",
        "-l",
        help="Labels directory or parquet file",
    ),
    output: Path = typer.Option(
        Path("data/models/matcher_model_combined.joblib"),
        "--output",
        "-o",
        help="Output path for trained model",
    ),
    combined: bool = typer.Option(
        False,
        "--combined",
        "-c",
        help="Combine all label files in directory for training",
    ),
):
    """Train an ML model on labeled data.

    Examples:
        matcher train --labels data/labels/labels_boston_streets.parquet
        matcher train --labels data/labels --combined -o data/models/combined.joblib
    """
    from .matching.ml import MLMatcher, train_model

    import pandas as pd

    labels_path = Path(labels)

    if labels_path.is_dir() and combined:
        # Combine all label files
        console.print(f"[blue]Loading labels from {labels_path}...[/blue]")
        label_files = list(labels_path.glob("labels_*.parquet"))
        if not label_files:
            console.print("[red]No label files found[/red]")
            raise typer.Exit(1)

        dfs = []
        for f in label_files:
            df = pd.read_parquet(f)
            df["_source"] = f.stem
            dfs.append(df)
            console.print(f"  {f.name}: {len(df)} labels")

        combined_df = pd.concat(dfs, ignore_index=True)
        console.print(f"[green]Combined: {len(combined_df)} total labels[/green]")

        # Save combined temporarily
        temp_path = labels_path / "_combined_temp.parquet"
        combined_df.to_parquet(temp_path)

        try:
            train_model(str(temp_path), str(output))
        finally:
            temp_path.unlink()  # Clean up temp file
    else:
        if labels_path.is_dir():
            console.print("[red]Specify a parquet file or use --combined[/red]")
            raise typer.Exit(1)

        console.print(f"[blue]Training on {labels_path}...[/blue]")
        train_model(str(labels_path), str(output))

    console.print(f"\n[green]Model saved to {output}[/green]")


@app.command("eval-model")
def eval_model(
    model: Path = typer.Argument(..., help="Path to trained model"),
    labels_dir: Path = typer.Option(
        Path("data/labels"),
        "--labels",
        "-l",
        help="Labels directory for evaluation",
    ),
    by_dataset: bool = typer.Option(
        True,
        "--by-dataset/--overall",
        help="Show metrics broken down by dataset",
    ),
):
    """Evaluate ML model performance on labeled data.

    Examples:
        matcher eval-model data/models/matcher_model.joblib
        matcher eval-model data/models/combined.joblib --labels data/labels
    """
    from .matching.ml import evaluate_by_dataset

    if not model.exists():
        console.print(f"[red]Model not found: {model}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Evaluating {model.name}...[/blue]")
    evaluate_by_dataset(str(model), str(labels_dir), show_by_dataset=by_dataset)

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
    config: Optional[Path] = typer.Option(
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
    min_length: float = typer.Option(
        3.0,
        "--min-length",
        help="Minimum segment length to include (meters)",
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
                    f"[red]Error: Target must be name:bridge_path:unmatched_path:priority[/red]"
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
                min_segment_length=min_length,
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
):
    """Launch the integration QA app.

    Review orphan components and merged edges from the integration pipeline.

    Example:
        matcher qa-integration -o data/integrated
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
        console.print("[red]Error: bbox must be 4 comma-separated values: xmin,ymin,xmax,ymax[/red]")
        raise typer.Exit(1)

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
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    from . import __version__

    console.print(f"matcher version {__version__}")


if __name__ == "__main__":
    app()
