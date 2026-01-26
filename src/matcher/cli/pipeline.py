"""Core pipeline commands: topology, match, eval-bridge."""

from pathlib import Path

import typer

from ._app import app, console


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

    from ..topology import planarize

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
        "xgboost",
        "--method",
        "-m",
        help="Matching method: xgboost",
    ),
    buffer_distance_m: float = typer.Option(
        50.0,
        "--buffer-m",
        "-b",
        help="Candidate search radius in meters",
    ),
    workers: int = typer.Option(
        -1,
        "--workers",
        "-w",
        help="Number of parallel workers (-1 for auto). Reduce for large datasets to save memory.",
    ),
):
    """Run the full matching pipeline."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from ..pipeline import run_pipeline

    console.print("[blue]Running matching pipeline...[/blue]")
    console.print(f"  Reference: {reference}")
    console.print(f"  Target: {target}")
    console.print(f"  Method: {method}")
    console.print(f"  Buffer: {buffer_distance_m}m")
    if workers != -1:
        console.print(f"  [yellow]Workers: {workers}[/yellow]")

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
            n_jobs=workers,
        )

        progress.update(task, completed=True)

    console.print(f"[green]Matched {result.n_matched} / {result.n_target} features[/green]")
    console.print(f"[green]Bridge file: {output}[/green]")


@app.command("eval-bridge")
def eval_bridge(
    bridge_file: Path = typer.Argument(..., help="Bridge file to evaluate"),
    ground_truth: Path | None = typer.Option(
        None,
        "--ground-truth",
        "-g",
        help="Ground truth labels CSV (columns: gers_id, target_id, label)",
    ),
):
    """Evaluate bridge file (matching output) quality.

    Shows confidence distribution and, if ground truth is provided,
    computes precision/recall/F1 metrics.

    Note: To evaluate ML model quality on training data, use 'eval-model' instead.

    Examples:
        # Basic bridge file stats
        matcher eval-bridge data/output/us_boston_streets_bridge.parquet

        # With ground truth evaluation
        matcher eval-bridge data/output/us_boston_streets_bridge.parquet \\
            --ground-truth labels/dataset=us_boston_streets/data.csv
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
