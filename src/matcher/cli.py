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
def version():
    """Show version information."""
    from . import __version__

    console.print(f"matcher version {__version__}")


if __name__ == "__main__":
    app()
