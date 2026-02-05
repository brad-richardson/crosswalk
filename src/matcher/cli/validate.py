"""Validation experiment commands."""

from pathlib import Path

import typer

from .utils import console

# Create validate group
validate_app = typer.Typer(
    name="validate",
    help="Validation experiment commands",
    no_args_is_help=True,
)


@validate_app.command("matching")
def validate_matching(
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
        "xgboost",
        "--method",
        "-m",
        help="Matching method: xgboost",
    ),
    fraction: float = typer.Option(
        0.1,
        "--fraction",
        "-f",
        help="Fraction to drop for 'random' strategy (0.0-1.0)",
    ),
    source_dataset: str = typer.Option(
        "OpenStreetMap",
        "--source-dataset",
        help="Dataset to drop for 'source' strategy (e.g., OpenStreetMap, Meta)",
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

    This command tests matching quality by:
    1. Dropping segments from Overture based on the chosen strategy
    2. Fetching fresh OSM data for the bounding box
    3. Running the matcher to see if dropped segments get matched back
    4. Evaluating results and computing recall metrics

    Note: To validate data file versions, use 'matcher data validate' instead.

    Examples:
        # Drop 10% of OSM segments randomly
        matcher validate matching data/raw/overture.parquet \\
            --bbox "-71.19,42.21,-70.92,42.40" \\
            --strategy random --fraction 0.1 \\
            --output validation/random_10pct/

        # Drop all segments from a specific source
        matcher validate matching data/raw/overture.parquet \\
            --bbox "-71.19,42.21,-70.92,42.40" \\
            --strategy source --source-dataset Meta \\
            --output validation/meta_holdout/

        # Drop residential roads
        matcher validate matching data/raw/overture.parquet \\
            --bbox "-71.19,42.21,-70.92,42.40" \\
            --strategy class --road-class residential \\
            --output validation/residential_holdout/
    """
    from ..validation import run_validation_experiment

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
        raise typer.Exit(1) from None
