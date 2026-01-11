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
        help="Bounding box: minx,miny,maxx,maxy (EPSG:4326)",
    ),
    output_dir: Path = typer.Option(
        Path("data/raw"),
        "--output",
        "-o",
        help="Output directory for fetched data",
    ),
    overture: bool = typer.Option(True, "--overture/--no-overture", help="Fetch Overture data"),
    osm: bool = typer.Option(False, "--osm/--no-osm", help="Fetch OSM data"),
    osm_pbf: Optional[Path] = typer.Option(
        None,
        "--osm-pbf",
        help="Local OSM PBF file (optional, will download if not provided)",
    ),
):
    """Fetch road data for an area of interest."""
    from .fetch import overture as ov_module, osm as osm_module

    coords = [float(x.strip()) for x in bbox.split(",")]
    if len(coords) != 4:
        console.print("[red]Error: bbox must have 4 values: minx,miny,maxx,maxy[/red]")
        raise typer.Exit(1)

    minx, miny, maxx, maxy = coords
    output_dir.mkdir(parents=True, exist_ok=True)

    if overture:
        console.print(f"[blue]Fetching Overture segments for bbox {bbox}...[/blue]")
        bbox_obj = ov_module.BoundingBox(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
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

    if osm:
        console.print(f"[blue]Fetching OSM roads for bbox {bbox}...[/blue]")
        osm_module.fetch_osm_roads(
            bbox=(minx, miny, maxx, maxy),
            pbf_path=osm_pbf,
            output_path=output_dir / "osm_roads.parquet",
        )
        console.print(f"[green]Saved OSM roads to {output_dir / 'osm_roads.parquet'}[/green]")


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
def version():
    """Show version information."""
    from . import __version__

    console.print(f"matcher version {__version__}")


if __name__ == "__main__":
    app()
