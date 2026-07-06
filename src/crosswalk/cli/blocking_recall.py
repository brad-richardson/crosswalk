"""CLI command for measuring blocking-stage recall against labeled matches."""

import math
from pathlib import Path

import typer

from .utils import console


def _format_recall(recall: float) -> str:
    """Render a recall value, showing n/a when undefined (no resolvable match labels)."""
    if math.isnan(recall):
        return "[dim]n/a[/dim]"
    style = "green" if recall >= 0.99 else "yellow" if recall >= 0.95 else "red"
    return f"[{style}]{recall:.2%}[/{style}]"


def register_blocking_recall_commands(app: typer.Typer) -> None:
    """Register the blocking-recall command on the main app."""

    @app.command("blocking-recall")
    def blocking_recall(
        reference_file: Path = typer.Argument(..., help="Reference (Overture segments) GeoParquet"),
        target_file: Path = typer.Argument(..., help="Target (local data) GeoParquet"),
        dataset: str = typer.Option(
            ...,
            "--dataset",
            "-d",
            help="Dataset ID whose human labels to evaluate (labels/human/dataset=<ID>/)",
        ),
        buffer_distance: float | None = typer.Option(
            None,
            "--buffer-distance",
            "-b",
            help="Blocking search radius in meters (default: settings.buffer_distance_m)",
        ),
        labels_dir: Path = typer.Option(
            Path("labels"),
            "--labels-dir",
            help="Base labels directory (contains human/ subdir)",
        ),
        max_missed: int = typer.Option(
            50,
            "--max-missed",
            help="Maximum number of missed pairs to print",
        ),
    ):
        """Measure blocking-stage recall: do labeled true matches survive candidate generation?

        Runs the SAME generate_candidates blocking used at inference over the
        given datasets and reports what fraction of human-labeled match pairs
        appear in the candidate set. Any true match lost at blocking never
        reaches the ML scorer and is invisible to downstream metrics.

        Example:
            crosswalk blocking-recall \\
                data/raw/us_boston_streets_overture_segments_v1.0.parquet \\
                data/raw/us_boston_streets_v1.0.parquet \\
                -d us_boston_streets
        """
        import geopandas as gpd
        from rich.table import Table

        from ..labeling.label_store import LabelStore
        from ..quality.blocking_recall import compute_blocking_recall

        for path in (reference_file, target_file):
            if not path.exists():
                console.print(f"[red]File not found: {path}[/red]")
                raise typer.Exit(1)

        # Load human labels for the dataset
        human_labels = LabelStore.load_human_labels(labels_dir / "human")
        labels = human_labels[human_labels["dataset"] == dataset]
        if len(labels) == 0:
            console.print(
                f"[red]No human labels found for dataset '{dataset}' in {labels_dir}[/red]"
            )
            raise typer.Exit(1)

        n_match_labels = (labels["label"] == "match").sum()
        console.print(
            f"[blue]Loaded {len(labels)} human labels for {dataset} "
            f"({n_match_labels} match labels)[/blue]"
        )

        # Load datasets as the pipeline does (raw; filtering happens inside
        # compute_blocking_recall so MultiLineString drops can be reported)
        reference = gpd.read_parquet(reference_file)
        target = gpd.read_parquet(target_file)
        if reference.crs is None:
            reference = reference.set_crs("EPSG:4326")
        if target.crs is None:
            target = target.set_crs("EPSG:4326")

        result = compute_blocking_recall(
            reference_gdf=reference,
            target_gdf=target,
            labels_df=labels,
            buffer_distance_m=buffer_distance,
        )

        # Summary table
        table = Table(title=f"Blocking recall: {dataset} (buffer={result.buffer_distance_m:g}m)")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Total match labels", str(result.total_match_labels))
        table.add_row("Blocked (found)", str(result.blocked))
        table.add_row("Missed", str(len(result.missed)))
        table.add_row("Unresolvable", str(len(result.unresolvable)))
        table.add_row("MultiLineString-dropped", str(len(result.multilinestring_dropped)))
        table.add_row("Recall", _format_recall(result.recall))
        console.print(table)

        # Recall at alternative buffers
        buffer_table = Table(title="Recall by buffer distance")
        buffer_table.add_column("Buffer (m)", justify="right")
        buffer_table.add_column("Recall", justify="right")
        for buf, recall in sorted(result.recall_at_buffer.items()):
            marker = " (current)" if buf == result.buffer_distance_m else ""
            buffer_table.add_row(f"{buf:g}{marker}", _format_recall(recall))
        console.print(buffer_table)

        # Missed pairs with distances
        if result.missed:
            missed_table = Table(title=f"Missed pairs (up to {max_missed})")
            missed_table.add_column("gers_id")
            missed_table.add_column("target_id")
            missed_table.add_column("Min distance (m)", justify="right")
            for pair in result.missed[:max_missed]:
                missed_table.add_row(pair.gers_id, pair.target_id, f"{pair.distance_m:.1f}")
            console.print(missed_table)
            if len(result.missed) > max_missed:
                console.print(f"  ... and {len(result.missed) - max_missed} more missed pairs")

        if result.unresolvable:
            console.print(
                f"[yellow]{len(result.unresolvable)} labeled pairs have IDs not present in the "
                f"loaded datasets (stale labels or wrong data version) - excluded from recall[/yellow]"
            )
        if result.multilinestring_dropped:
            console.print(
                f"[yellow]{len(result.multilinestring_dropped)} labeled pairs have MultiLineString "
                f"targets - dropped at ingest by the pipeline, can never match[/yellow]"
            )
