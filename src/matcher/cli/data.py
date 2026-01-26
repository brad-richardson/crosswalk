"""Data commands: validate-matching, validate-data, discover-classes, version."""

from pathlib import Path

import typer

from ._app import app, console


@app.command("validate-matching")
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

    This command tests matching quality by:
    1. Dropping segments from Overture based on the chosen strategy
    2. Fetching fresh OSM data for the bounding box
    3. Running the matcher to see if dropped segments get matched back
    4. Evaluating results and computing recall metrics

    Note: To validate data file versions, use 'validate-data' instead.

    Examples:
        # Drop 10% of OSM segments randomly
        matcher validate-matching data/raw/overture.parquet \\
            --bbox "-71.19,42.21,-70.92,42.40" \\
            --strategy random --fraction 0.1 \\
            --output validation/random_10pct/

        # Drop all TomTom segments
        matcher validate-matching data/raw/overture.parquet \\
            --bbox "-71.19,42.21,-70.92,42.40" \\
            --strategy source --source-dataset TomTom \\
            --output validation/tomtom_holdout/

        # Drop residential roads
        matcher validate-matching data/raw/overture.parquet \\
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
        matcher discover-classes data/raw/us_boston_streets.parquet

        # With match-based analysis (more accurate)
        matcher discover-classes data/raw/us_boston_streets.parquet \\
            --reference data/raw/us_boston_overture_segments.parquet \\
            --bridge data/output/us_boston_streets_bridge.parquet

        # Print report only (don't save config)
        matcher discover-classes data/raw/new_dataset.parquet --print-only
    """
    from ..datasets.discover import discover_dataset, print_discovery_report, save_dataset_config

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
        from ..datasets.schema import (
            ClassificationConfig,
            get_dataset_config,
            get_datasets_dir,
        )
        from ..datasets.schema import (
            ClassMappingRule as NewClassMappingRule,
        )
        from ..datasets.schema import (
            SourceClassification as NewSourceClassification,
        )
        from ..datasets.schema import (
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


@app.command("validate-data")
def validate_data(
    data_dir: Path = typer.Argument(
        Path("data/raw"),
        help="Directory containing data files",
    ),
):
    """Validate data files for version compatibility.

    For each parquet file in the data directory, checks that any version
    suffix present matches the current code version. Files without a version
    suffix are treated as legacy data and reported with a warning but do not
    cause validation to fail. This helps catch stale versioned data that
    needs to be re-fetched after code updates.

    Examples:
        matcher validate-data
        matcher validate-data data/raw
    """
    from ..config import DATA_VERSION
    from ..filenames import extract_version_from_filename

    console.print(f"[blue]Current data version: {DATA_VERSION}[/blue]\n")

    if not data_dir.exists():
        console.print(f"[red]Directory not found: {data_dir}[/red]")
        raise typer.Exit(1)

    parquet_files = list(data_dir.glob("*.parquet"))

    if not parquet_files:
        console.print(f"[yellow]No parquet files found in {data_dir}[/yellow]")
        return

    has_errors = False
    expected = DATA_VERSION.lstrip("v")

    for path in sorted(parquet_files):
        version = extract_version_from_filename(path)

        if version is None:
            console.print(f"[yellow]{path.name}: No version suffix[/yellow]")
            # Don't count as error - could be legacy file
        elif version == expected:
            console.print(f"[green]{path.name}: OK ({DATA_VERSION})[/green]")
        else:
            console.print(f"[red]{path.name}: v{version} != {DATA_VERSION}[/red]")
            has_errors = True

    if has_errors:
        console.print("\n[red]Validation failed. Re-fetch data to fix.[/red]")
        raise typer.Exit(1)
    else:
        console.print("\n[green]All versioned files valid.[/green]")


@app.command()
def version():
    """Show version information."""
    from .. import __version__

    console.print(f"matcher version {__version__}")
