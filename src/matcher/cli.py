"""CLI entry point for the road network conflation pipeline."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="matcher",
    help="Road network conflation pipeline - link local roads to Overture GERS",
    no_args_is_help=True,
)
console = Console()

# Create fetch subcommand group
fetch_app = typer.Typer(
    name="fetch",
    help="Fetch road data from various sources",
    no_args_is_help=True,
)
app.add_typer(fetch_app, name="fetch")


@fetch_app.command("target")
def fetch_target(
    dataset_name: str = typer.Argument(
        None,
        help="Dataset name to fetch (e.g., us_boston_streets)",
    ),
    prefix: str | None = typer.Option(
        None,
        "--prefix",
        "-p",
        help="Fetch all datasets matching this prefix (e.g., us_boston)",
    ),
    output_dir: Path = typer.Option(
        Path("data/raw"),
        "--output",
        "-o",
        help="Output directory for fetched data",
    ),
    page_size: int | None = typer.Option(
        None,
        "--page-size",
        help="Override page size for ArcGIS fetches (default: 5000)",
    ),
    fetch_all: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Fetch all available datasets",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-fetch even if file already exists",
    ),
):
    """Fetch target/local road data from municipal GIS portals.

    Downloads data from ArcGIS, WFS, OGC API Features, or direct download
    based on the dataset's YAML configuration. By default, skips files that
    already exist (use --force to re-fetch).

    Examples:
        matcher fetch target us_boston_streets      # Fetch specific dataset
        matcher fetch target --prefix us_boston     # Fetch all Boston datasets
        matcher fetch target --all                  # Fetch all datasets
        matcher fetch target us_boston_streets --force  # Re-fetch existing
    """
    from .fetch import target as target_module

    output_dir.mkdir(parents=True, exist_ok=True)

    if fetch_all:
        console.print("[blue]Fetching all datasets...[/blue]")
        results = target_module.fetch_all_datasets(output_dir, page_size, force)
        success = sum(1 for p in results.values() if p is not None)
        console.print(f"[green]Fetched {success}/{len(results)} datasets[/green]")

    elif prefix:
        console.print(f"[blue]Fetching datasets with prefix '{prefix}'...[/blue]")
        results = target_module.fetch_datasets_by_prefix(prefix, output_dir, page_size, force)
        if not results:
            console.print(f"[red]No datasets found matching prefix: {prefix}[/red]")
            raise typer.Exit(1)
        success = sum(1 for p in results.values() if p is not None)
        console.print(f"[green]Fetched {success}/{len(results)} datasets[/green]")

    elif dataset_name:
        console.print(f"[blue]Fetching dataset: {dataset_name}[/blue]")
        result = target_module.fetch_dataset(dataset_name, output_dir, page_size, force)
        if result:
            console.print(f"[green]Saved to {result}[/green]")
        else:
            console.print("[red]Fetch failed or requires manual download[/red]")
            raise typer.Exit(1)

    else:
        console.print("[red]Error: Provide a dataset name, --prefix, or --all[/red]")
        raise typer.Exit(1)


@fetch_app.command("reference")
def fetch_reference(
    dataset_name: str = typer.Argument(
        ...,
        help="Dataset name to fetch reference data for (uses bbox from config)",
    ),
    output_dir: Path = typer.Option(
        Path("data/raw"),
        "--output",
        "-o",
        help="Output directory for fetched data",
    ),
    source: list[str] = typer.Option(
        ["overture"],
        "--source",
        "-s",
        help="Reference source(s): 'overture' or 'osm' (can specify multiple)",
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
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-fetch even if files already exist",
    ),
):
    """Fetch reference data (Overture/OSM) for a dataset.

    Uses the bounding box from the dataset's YAML configuration.
    By default, skips files that already exist (use --force to re-fetch).

    Examples:
        matcher fetch reference us_boston_streets           # Fetch Overture (default)
        matcher fetch reference us_boston_streets -s osm    # Fetch OSM
        matcher fetch reference us_boston_streets -s overture -s osm  # Both
    """
    _fetch_reference_impl(
        dataset_name=dataset_name,
        output_dir=output_dir,
        sources=set(s.lower() for s in source),
        cache_dir=cache_dir,
        no_cache=no_cache,
        keep_pbf=keep_pbf,
        bbox_buffer=bbox_buffer,
        force=force,
    )


def _fetch_reference_impl(
    dataset_name: str,
    output_dir: Path,
    sources: set[str],
    cache_dir: Path | None = None,
    no_cache: bool = False,
    keep_pbf: bool = False,
    bbox_buffer: float | None = None,
    force: bool = False,
) -> None:
    """Implementation of reference data fetching."""
    from .datasets.schema import get_dataset_config, list_dataset_configs
    from .fetch import osm as osm_module
    from .fetch import overture as ov_module
    from .filenames import (
        osm_segments_filename,
        overture_connectors_filename,
        overture_segments_filename,
    )

    # Validate sources
    valid_sources = {"overture", "osm"}
    invalid = sources - valid_sources
    if invalid:
        console.print(
            f"[red]Error: Invalid source(s): {invalid}. Must be 'overture' or 'osm'[/red]"
        )
        raise typer.Exit(1)

    # Look up bbox from dataset config
    config = get_dataset_config(dataset_name)
    if config is None:
        console.print(f"[red]Error: Could not find dataset config for '{dataset_name}'[/red]")
        available = list_dataset_configs()
        if available:
            console.print("[yellow]Available datasets: " + ", ".join(sorted(available)[:10]))
            if len(available) > 10:
                console.print(f"  ... and {len(available) - 10} more[/yellow]")
        raise typer.Exit(1)

    if config.fetch is None or config.fetch.bbox is None:
        console.print(f"[red]Error: Dataset '{dataset_name}' has no bbox configured[/red]")
        raise typer.Exit(1)

    xmin, ymin, xmax, ymax = config.fetch.bbox
    console.print(f"[blue]Using bbox from dataset config: {dataset_name}[/blue]")
    console.print(f"[blue]  bbox: {xmin},{ymin},{xmax},{ymax}[/blue]")

    output_dir.mkdir(parents=True, exist_ok=True)
    original_bbox = ov_module.BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    # Get buffered bboxes for each source
    overture_bbox, overture_buffer = ov_module.get_buffered_bbox(
        original_bbox, bbox_buffer, ov_module.DEFAULT_OVERTURE_BUFFER_M
    )
    osm_bbox, osm_buffer = ov_module.get_buffered_bbox(
        original_bbox, bbox_buffer, osm_module.DEFAULT_OSM_BUFFER_M
    )

    # Log buffer info
    if "overture" in sources and overture_buffer:
        console.print(
            f"[blue]Using {overture_buffer}m buffer for Overture data "
            f"(override with --bbox-buffer)[/blue]"
        )
        console.print(
            f"[blue]  Buffered bbox: {overture_bbox.xmin:.6f},{overture_bbox.ymin:.6f},"
            f"{overture_bbox.xmax:.6f},{overture_bbox.ymax:.6f}[/blue]"
        )
    elif "overture" in sources and bbox_buffer == 0:
        console.print("[blue]Buffer explicitly disabled (--bbox-buffer=0)[/blue]")

    if "osm" in sources and osm_buffer:
        console.print(
            f"[blue]Using {osm_buffer}m buffer for OSM data (override with --bbox-buffer)[/blue]"
        )
        console.print(
            f"[blue]  Buffered bbox: {osm_bbox.xmin:.6f},{osm_bbox.ymin:.6f},"
            f"{osm_bbox.xmax:.6f},{osm_bbox.ymax:.6f}[/blue]"
        )
    elif "osm" in sources and bbox_buffer == 0:
        console.print("[blue]Buffer explicitly disabled (--bbox-buffer=0)[/blue]")

    if "overture" in sources:
        overture_seg_path = output_dir / overture_segments_filename(dataset_name)
        overture_conn_path = output_dir / overture_connectors_filename(dataset_name)

        if not force and overture_seg_path.exists() and overture_conn_path.exists():
            console.print(
                f"[blue]Skipping Overture: {overture_seg_path.name} already exists "
                f"(use --force to re-fetch)[/blue]"
            )
        else:
            console.print("[blue]Fetching Overture segments...[/blue]")
            segments_path = ov_module.fetch_overture_segments(
                bbox=overture_bbox,
                output_path=overture_seg_path,
                original_bbox=original_bbox,
                buffer_m=overture_buffer,
            )
            console.print(f"[green]Saved Overture segments to {segments_path}[/green]")

            console.print("[blue]Fetching Overture connectors...[/blue]")
            connectors_path = ov_module.fetch_overture_connectors(
                bbox=overture_bbox,
                output_path=overture_conn_path,
                original_bbox=original_bbox,
                buffer_m=overture_buffer,
            )
            console.print(f"[green]Saved Overture connectors to {connectors_path}[/green]")

    if "osm" in sources:
        osm_seg_path = output_dir / osm_segments_filename(dataset_name)

        if not force and osm_seg_path.exists():
            console.print(
                f"[blue]Skipping OSM: {osm_seg_path.name} already exists "
                f"(use --force to re-fetch)[/blue]"
            )
        else:
            # OSM uses unbuffered bbox with fully-inside filter for dataset mode
            console.print(
                "[blue]OSM: using unbuffered bbox, filtering to fully-inside features[/blue]"
            )
            console.print("[blue]Fetching OSM data...[/blue]")
            segments_path, connectors_path = osm_module.fetch_osm_data(
                bbox=original_bbox,
                output_dir=output_dir,
                cache_dir=cache_dir,
                force_download=no_cache,
                keep_pbf=keep_pbf,
                original_bbox=original_bbox,
                buffer_m=None,
                name=dataset_name,
                filter_fully_inside=True,
            )
            console.print(f"[green]Saved OSM segments (ways) to {segments_path}[/green]")
            console.print(f"[green]Saved OSM connectors (nodes) to {connectors_path}[/green]")

    # Update last_fetch in dataset config
    from datetime import UTC, datetime

    from .datasets.schema import LastFetch, get_datasets_dir, save_dataset_config
    from .fetch.metadata import load_metadata

    config = get_dataset_config(dataset_name)
    if config:
        buffer_m = overture_buffer if "overture" in sources else osm_buffer
        feature_count = 0
        geometry_types: list[str] = []

        if "overture" in sources:
            overture_seg_file = overture_segments_filename(dataset_name)
            meta = load_metadata(output_dir / overture_seg_file)
            if meta:
                feature_count = meta.feature_count
                geometry_types = meta.geometry_types
        elif "osm" in sources:
            osm_seg_file = osm_segments_filename(dataset_name)
            meta = load_metadata(output_dir / osm_seg_file)
            if meta:
                feature_count = meta.feature_count
                geometry_types = meta.geometry_types

        config.last_fetch = LastFetch(
            fetched_at=datetime.now(UTC),
            bbox=original_bbox.to_tuple(),
            bbox_buffered=(overture_bbox if "overture" in sources else osm_bbox).to_tuple()
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


@fetch_app.command("all")
def fetch_all(
    dataset_name: str = typer.Argument(
        ...,
        help="Dataset name to fetch all data for",
    ),
    output_dir: Path = typer.Option(
        Path("data/raw"),
        "--output",
        "-o",
        help="Output directory for fetched data",
    ),
    page_size: int | None = typer.Option(
        None,
        "--page-size",
        help="Override page size for ArcGIS fetches (default: 5000)",
    ),
    bbox_buffer: float | None = typer.Option(
        None,
        "--bbox-buffer",
        help="Expand bbox by this distance (meters) for reference data",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-fetch even if files already exist",
    ),
):
    """Fetch both target and reference data for a dataset.

    Fetches target (local/ArcGIS) data and Overture reference data in parallel.
    This is the command to use when setting up a new dataset for labeling.
    By default, skips files that already exist (use --force to re-fetch).

    Examples:
        matcher fetch all us_boston_streets    # Fetch both target + Overture
    """
    from .fetch import target as target_module

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[blue]Fetching all data for {dataset_name}...[/blue]")

    errors = []

    def fetch_target_data():
        try:
            result = target_module.fetch_dataset(dataset_name, output_dir, page_size, force)
            return ("target", result)
        except Exception as e:
            return ("target", e)

    def fetch_reference_data():
        try:
            _fetch_reference_impl(
                dataset_name=dataset_name,
                output_dir=output_dir,
                sources={"overture"},
                bbox_buffer=bbox_buffer,
                force=force,
            )
            return ("reference", True)
        except Exception as e:
            return ("reference", e)

    # Run both fetches in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(fetch_target_data),
            executor.submit(fetch_reference_data),
        ]

        for future in as_completed(futures):
            name, result = future.result()
            if isinstance(result, Exception):
                errors.append((name, result))
                console.print(f"[red]Error fetching {name}: {result}[/red]")
            elif name == "target":
                if result:
                    console.print(f"[green]Target data saved to {result}[/green]")
                else:
                    errors.append((name, "Fetch failed or requires manual download"))
            else:
                console.print("[green]Reference data fetched successfully[/green]")

    if errors:
        console.print(f"[yellow]Completed with {len(errors)} error(s)[/yellow]")
        raise typer.Exit(1)
    else:
        console.print("[green]All data fetched successfully![/green]")


@fetch_app.command("list")
def fetch_list(
    prefix: str | None = typer.Option(
        None,
        "--prefix",
        "-p",
        help="Filter datasets by prefix (e.g., us_boston)",
    ),
):
    """List available datasets.

    Examples:
        matcher fetch list                  # List all datasets
        matcher fetch list --prefix us_     # List US datasets
    """
    from .fetch import target as target_module

    target_module.print_datasets(prefix)


@fetch_app.command("verify")
def fetch_verify(
    dataset_name: str = typer.Argument(
        None,
        help="Dataset name to verify (e.g., us_boston_streets)",
    ),
    prefix: str | None = typer.Option(
        None,
        "--prefix",
        "-p",
        help="Verify all datasets matching prefix",
    ),
    all_datasets: bool = typer.Option(
        False,
        "--all",
        help="Verify all datasets",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Only check config validity, don't test URLs",
    ),
):
    """Verify dataset configurations are valid and accessible.

    Checks:
    - YAML config is valid and parseable
    - Required fields are present (source.url or source.product_id)
    - API keys are available if required
    - Source URL/endpoint is accessible (unless --dry-run)
    - Bounding box area is reasonable

    Examples:
        matcher fetch verify us_boston_streets      # Verify single dataset
        matcher fetch verify --prefix ae_           # Verify by prefix
        matcher fetch verify --all --dry-run        # Check all configs
    """
    import os

    import requests

    from .datasets.schema import get_dataset_config, list_dataset_configs

    def verify_dataset(name: str, dry_run: bool) -> tuple[str, bool, str]:
        """Verify a single dataset config. Returns (name, success, message)."""
        config = get_dataset_config(name)
        if config is None:
            return (name, False, "Config file not found or invalid YAML")

        # Check source type
        source_type = config.source.type if config.source else "unknown"
        if source_type == "manual":
            return (name, True, f"[yellow]Manual download[/yellow] - skipped")

        # Check required fields based on source type
        if source_type == "os_downloads":
            if not config.source or not config.source.product_id:
                return (name, False, "Missing source.product_id for os_downloads type")
        else:
            if not config.source or not config.source.url:
                return (name, False, "Missing source.url")

        # Check API key if required
        if config.source and config.source.api_key_env_var:
            env_var = config.source.api_key_env_var
            if not os.environ.get(env_var):
                return (name, True, f"[yellow]API key {env_var} not set[/yellow]")

        # Check bbox area
        if config.fetch and config.fetch.bbox:
            xmin, ymin, xmax, ymax = config.fetch.bbox
            # Approximate area in km²
            width_km = (xmax - xmin) * 111 * 0.7  # Rough lon to km at mid-lat
            height_km = (ymax - ymin) * 111
            area_km2 = width_km * height_km
            if area_km2 > 50000:
                return (
                    name,
                    True,
                    f"[yellow]Large bbox: ~{area_km2:.0f} km²[/yellow]",
                )

        if dry_run:
            return (name, True, "[green]Config valid[/green]")

        # Test URL accessibility
        url = config.source.url if config.source else None
        if url and source_type in ("arcgis", "ogc_features", "wfs"):
            try:
                # For ArcGIS, test the metadata endpoint
                test_url = url if source_type != "arcgis" else f"{url}?f=json"
                resp = requests.get(test_url, timeout=10)
                if resp.status_code == 200:
                    return (name, True, "[green]URL accessible[/green]")
                else:
                    return (name, False, f"URL returned status {resp.status_code}")
            except requests.exceptions.RequestException as e:
                return (name, False, f"URL error: {e}")
        elif url and source_type == "download":
            try:
                resp = requests.head(url, timeout=10, allow_redirects=True)
                if resp.status_code == 200:
                    return (name, True, "[green]URL accessible[/green]")
                else:
                    return (name, False, f"URL returned status {resp.status_code}")
            except requests.exceptions.RequestException as e:
                return (name, False, f"URL error: {e}")

        return (name, True, "[green]Config valid[/green]")

    # Determine which datasets to verify
    if all_datasets:
        datasets = list_dataset_configs()
    elif prefix:
        all_configs = list_dataset_configs()
        datasets = [d for d in all_configs if d.startswith(prefix)]
    elif dataset_name:
        datasets = [dataset_name]
    else:
        console.print("[red]Error: Provide dataset name, --prefix, or --all[/red]")
        raise typer.Exit(1)

    if not datasets:
        console.print("[yellow]No datasets found to verify[/yellow]")
        raise typer.Exit(0)

    console.print(f"[blue]Verifying {len(datasets)} dataset(s)...[/blue]")
    console.print()

    success_count = 0
    warning_count = 0
    error_count = 0

    for name in sorted(datasets):
        name, success, message = verify_dataset(name, dry_run)
        if success:
            if "yellow" in message:
                warning_count += 1
            else:
                success_count += 1
        else:
            error_count += 1
        status_icon = "[green]✓[/green]" if success else "[red]✗[/red]"
        console.print(f"  {status_icon} {name}: {message}")

    console.print()
    console.print(
        f"[green]{success_count} passed[/green], "
        f"[yellow]{warning_count} warnings[/yellow], "
        f"[red]{error_count} errors[/red]"
    )

    if error_count > 0:
        raise typer.Exit(1)


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
    workers: int = typer.Option(
        -1,
        "--workers",
        "-w",
        help="Number of parallel workers (-1 for auto). Reduce for large datasets to save memory.",
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
        matcher label data/raw/us_boston_overture_segments.parquet data/raw/us_boston_streets.parquet
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
    exclude_dataset: list[str] = typer.Option(
        [],
        "--exclude-dataset",
        "-x",
        help="Dataset(s) to exclude from training (for leave-one-out evaluation). Can be repeated.",
    ),
):
    """Train an ML model on labeled data.

    Loads labels from Hive-partitioned CSV format (labels/dataset=*/data.csv).

    Examples:
        matcher train
        matcher train --labels labels -o data/models/my_model.joblib

        # Train geometry-only model (no name/class features)
        matcher train --exclude-semantic -o data/models/matcher_model_geom_only.joblib

        # Leave-one-out: train without Frisco labels to test generalization
        matcher train -x us_frisco_trails -o data/models/no_frisco.joblib
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

    if exclude_dataset:
        console.print(f"[yellow]Excluding datasets: {', '.join(exclude_dataset)}[/yellow]")

    # Train model
    model_type = "geometry-only" if exclude_semantic else "full"
    console.print(f"[blue]Training {model_type} model...[/blue]")
    matcher = MLMatcher()
    metrics = matcher.train(
        labels_dir=labels_dir,
        test_size=0.2,
        binary=True,
        exclude_semantic=exclude_semantic,
        exclude_datasets=list(exclude_dataset) if exclude_dataset else None,
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
    dataset: list[str] = typer.Option(
        [],
        "--dataset",
        "-d",
        help="Only evaluate on specific dataset(s). Can be repeated.",
    ),
    holdout: bool = typer.Option(
        True,
        "--holdout/--no-holdout",
        help="Use holdout set for evaluation (default: True for unbiased metrics)",
    ),
    holdout_pct: float = typer.Option(
        0.2,
        "--holdout-pct",
        help="Fraction of data to hold out for testing (default: 0.2 = 20%%)",
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
        matcher eval-model data/models/combined.joblib --holdout-pct 0.3

        # Evaluate only on specific dataset (for leave-one-out testing)
        matcher eval-model data/models/no_frisco.joblib -d us_frisco_trails --no-holdout
    """
    from .matching.ml import evaluate_by_dataset

    if not model.exists():
        console.print(f"[red]Model not found: {model}[/red]")
        raise typer.Exit(1)

    if dataset:
        console.print(f"[blue]Filtering to datasets: {', '.join(dataset)}[/blue]")

    if holdout:
        console.print(
            f"[blue]Evaluating {model.name} on {holdout_pct * 100:.0f}% holdout (seed={seed})...[/blue]"
        )
    else:
        console.print(
            f"[yellow]Evaluating {model.name} on all data (may include training data)...[/yellow]"
        )

    evaluate_by_dataset(
        str(model),
        str(labels_dir),
        show_by_dataset=by_dataset,
        holdout=holdout,
        holdout_pct=holdout_pct,
        seed=seed,
        filter_datasets=list(dataset) if dataset else None,
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
        matcher integrate data/raw/us_boston_overture_segments.parquet \\
            -t us_boston_streets:data/output/bridge.parquet:data/output/unmatched.parquet:1 \\
            -o data/integrated

        # Multiple datasets with priority
        matcher integrate data/raw/us_boston_overture_segments.parquet \\
            -t us_boston_streets:data/us_boston_streets/bridge.parquet:data/us_boston_streets/unmatched.parquet:1 \\
            -t us_boston_bike_network:data/us_boston_bike_network/bridge.parquet:data/us_boston_bike_network/unmatched.parquet:2 \\
            -o data/integrated

        # From config file
        matcher integrate data/raw/us_boston_overture_segments.parquet -c integration_config.yaml -o data/integrated
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
        matcher discover-classes data/raw/us_boston_streets.parquet

        # With match-based analysis (more accurate)
        matcher discover-classes data/raw/us_boston_streets.parquet \\
            --reference data/raw/us_boston_overture_segments.parquet \\
            --bridge data/output/us_boston_streets_bridge.parquet

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


@app.command("generate-agent-batch")
def generate_agent_batch(
    dataset: str = typer.Argument(..., help="Target dataset name (e.g., 'us_boston_streets')"),
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
    """Generate NEW candidates for AI agent labeling.

    Samples diverse candidates across confidence ranges from unlabeled pairs
    and creates packages with metadata YAML and images for each candidate.
    Use this to expand training data with agent-labeled examples.

    Note: To test agent accuracy against existing human labels, use
    'generate-agent-test-batch' instead.

    Examples:
        # Generate 100 candidates for us_boston_streets
        matcher generate-agent-batch us_boston_streets

        # Generate 50 candidates with custom paths
        matcher generate-agent-batch us_boston_streets -n 50 \\
            -r data/raw/us_boston_overture_segments.parquet \\
            -t data/raw/us_boston_streets.parquet \\
            -o agent_labels

        # Use ML model for confidence scoring
        matcher generate-agent-batch us_boston_streets \\
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
    """Generate test batch from EXISTING human labels for agent accuracy testing.

    Samples from existing human-labeled pairs so you can measure agent agreement
    with human ground truth. Includes the ground truth labels in the output.

    Note: To generate NEW unlabeled candidates for agent labeling, use
    'generate-agent-batch' instead.

    Examples:
        # Generate 200 samples across all datasets
        matcher generate-agent-test-batch -n 200

        # Specific datasets only
        matcher generate-agent-test-batch -n 100 -d us_boston_streets -d us_boston_bike_network

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

    # Load target datasets - auto-discover paths based on dataset name
    target_gdfs = {}
    data_dir = Path("data/raw")

    for dataset in sampled_df["dataset"].unique():
        # Try different naming patterns
        candidates = [
            data_dir / f"{dataset}.parquet",  # e.g., us_boston_streets.parquet
            data_dir
            / f"{dataset}_segments.parquet",  # e.g., us_boston_streets_osm_segments.parquet
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path:
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


@app.command("backfill-labels")
def backfill_labels(
    labels_dir: Path = typer.Option(
        Path("labels"),
        "--labels",
        "-l",
        help="Labels directory (Hive-partitioned CSV format)",
    ),
    overture: Path = typer.Option(
        None,
        "--overture",
        "-r",
        help="Path to Overture segments parquet. If not specified, looks for "
        "{dataset}_overture_segments.parquet in the data directory.",
    ),
    data_dir: Path = typer.Option(
        Path("data/raw"),
        "--data-dir",
        "-d",
        help="Directory containing target dataset parquet files",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Compute features but don't write to disk",
    ),
    skip_missing: bool = typer.Option(
        False,
        "--skip-missing",
        help="Skip datasets with missing data files instead of failing",
    ),
    report: bool = typer.Option(
        False,
        "--report",
        help="Report what can/cannot be backfilled without modifying any files",
    ),
    drop_orphaned: bool = typer.Option(
        False,
        "--drop-orphaned",
        help="Remove labels where IDs are not found in current data (orphaned labels)",
    ),
):
    """Recompute features for all existing labels.

    This command is needed after changes to feature computation logic to ensure
    existing training labels have consistent features. It recomputes:
    - Alignment fractions (where segments overlap)
    - All topology features (using alignment-aware computation)
    - Endpoint proximity features
    - Graphlet similarity features
    - All other geometric/semantic features
    - Data version tracking columns (ref_data_version, target_data_version, feature_version)

    The command preserves the label (match/no_match) but updates all feature
    columns, alignment fractions, and version tracking.

    Labels are considered "orphaned" when their IDs are not found in the current
    data files (e.g., if data was re-fetched with different IDs).

    By default, the command will FAIL if any dataset is missing required data
    files. Use --skip-missing to skip those datasets instead.

    Examples:
        # Backfill all labels (fails if any data is missing)
        matcher backfill-labels

        # Skip datasets with missing data
        matcher backfill-labels --skip-missing

        # Report what can/cannot be backfilled (no changes)
        matcher backfill-labels --report

        # Dry run (compute but don't write)
        matcher backfill-labels --dry-run

        # Drop labels that can't be backfilled (orphaned IDs)
        matcher backfill-labels --drop-orphaned
    """
    from .labeling.label_store import backfill_features

    if not labels_dir.exists():
        console.print(f"[red]Labels directory not found: {labels_dir}[/red]")
        raise typer.Exit(1)

    if overture is not None and not overture.exists():
        console.print(f"[red]Overture file not found: {overture}[/red]")
        raise typer.Exit(1)

    if dry_run:
        console.print("[yellow]Dry run mode - no files will be modified[/yellow]")
    if report:
        console.print("[yellow]Report mode - no files will be modified[/yellow]")
    if drop_orphaned:
        console.print(
            "[yellow]Drop orphaned mode - labels with missing IDs will be removed[/yellow]"
        )

    console.print("[blue]Starting feature backfill...[/blue]")
    console.print(f"  Labels: {labels_dir}")
    console.print(f"  Overture: {overture or '(auto-discover per dataset)'}")
    console.print(f"  Data dir: {data_dir}")
    if skip_missing:
        console.print(
            "[yellow]  Skip missing: enabled (will skip datasets with missing data)[/yellow]"
        )

    try:
        results = backfill_features(
            labels_dir=labels_dir,
            overture_path=overture,
            data_dir=data_dir,
            dry_run=dry_run,
            skip_missing=skip_missing,
            report_only=report,
            drop_orphaned=drop_orphaned,
        )

        console.print()
        if report:
            console.print("[green]Backfill report complete![/green]")
        else:
            console.print("[green]Backfill complete![/green]")

        console.print("Results by dataset:")
        total_updated = 0
        total_orphaned = 0
        for dataset, stats in sorted(results.items()):
            updated = stats.get("updated", 0)
            orphaned = stats.get("orphaned", 0)
            total = stats.get("total", 0)
            skipped = stats.get("skipped")
            dropped = stats.get("dropped", 0)

            total_updated += updated
            total_orphaned += orphaned

            if skipped:
                console.print(f"  {dataset}: [yellow]skipped ({skipped})[/yellow]")
            elif orphaned > 0:
                orphan_status = f"[red]{orphaned} orphaned[/red]"
                if dropped > 0:
                    orphan_status += f" [yellow]({dropped} dropped)[/yellow]"
                console.print(f"  {dataset}: {updated}/{total} updated, {orphan_status}")
            else:
                console.print(f"  {dataset}: {updated}/{total} updated")

        console.print(f"\n  Total: {total_updated} updated, {total_orphaned} orphaned")

        if dry_run:
            console.print(
                "\n[yellow]Dry run - no files were modified. "
                "Remove --dry-run to apply changes.[/yellow]"
            )
        if report:
            console.print(
                "\n[yellow]Report mode - no files were modified. "
                "Remove --report to apply changes.[/yellow]"
            )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


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
    from .config import DATA_VERSION
    from .filenames import extract_version_from_filename

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
def benchmark(
    labels_dir: Path = typer.Option(
        Path("labels"),
        "--labels",
        "-l",
        help="Labels directory (Hive-partitioned CSV format)",
    ),
    output_dir: Path = typer.Option(
        Path("benchmarks"),
        "--output",
        "-o",
        help="Output directory for benchmark results",
    ),
    train_size: float = typer.Option(
        0.7,
        "--train-size",
        "-t",
        help="Fraction of data for training (default: 0.7 = 70/30 split)",
    ),
    seed: int = typer.Option(
        999,
        "--seed",
        "-s",
        help="Random seed for train/test split",
    ),
    skip_save: bool = typer.Option(
        False,
        "--skip-save",
        help="Skip saving results to CSV (just print)",
    ),
):
    """Run a benchmark: train on subset, evaluate on holdout.

    Uses segment-aware splitting to prevent data leakage - no segment
    appears in both train and test sets. Results are saved to
    benchmarks/model_performance.csv for tracking over time.

    Examples:
        matcher benchmark
        matcher benchmark --train-size 0.8
        matcher benchmark --seed 123 --skip-save
    """
    import csv
    from datetime import UTC, datetime

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    from .labeling.label_store import LabelStore
    from .matching.ml import MLMatcher, segment_aware_split

    if not labels_dir.exists():
        console.print(f"[red]Labels directory not found: {labels_dir}[/red]")
        raise typer.Exit(1)

    # Validate train size
    if not 0.1 <= train_size <= 0.9:
        console.print("[red]--train-size must be between 0.1 and 0.9[/red]")
        raise typer.Exit(1)

    run_date = datetime.now(UTC)
    test_pct = int((1 - train_size) * 100)
    train_pct = int(train_size * 100)

    console.print("[blue]Loading labels...[/blue]")
    all_labels = LabelStore.load_all(labels_dir)
    console.print(f"  Total labels: {len(all_labels)}")

    # Filter to valid labels only
    valid_labels = {"match", "no_match"}
    all_labels = all_labels[all_labels["label"].isin(valid_labels)].copy()
    console.print(f"  Valid labels (match/no_match): {len(all_labels)}")

    # Segment-aware split to prevent leakage
    console.print(f"\n[blue]Splitting {train_pct}/{test_pct} with segment-aware split...[/blue]")
    train_idx, test_idx = segment_aware_split(
        all_labels, test_size=1 - train_size, random_state=seed
    )

    train_df = all_labels.iloc[train_idx].copy()
    test_df = all_labels.iloc[test_idx].copy()

    console.print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    console.print(f"  Train labels: {train_df['label'].value_counts().to_dict()}")
    console.print(f"  Test labels: {test_df['label'].value_counts().to_dict()}")

    # Save train labels to temp directory for training
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save each dataset's train portion
        for dataset in train_df["dataset"].unique():
            ds_train = train_df[train_df["dataset"] == dataset]
            ds_dir = tmpdir / f"dataset={dataset}"
            ds_dir.mkdir(parents=True, exist_ok=True)
            ds_train.to_csv(ds_dir / "data.csv", index=False)

        # Train model on train set only (no internal split since we already split)
        console.print(f"\n[blue]Training model on {len(train_df)} samples...[/blue]")
        matcher = MLMatcher()
        matcher.train(labels_dir=str(tmpdir), binary=True, test_size=0.0)

        # Save the model
        model_dir = Path("data/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "matcher_model_combined.joblib"
        matcher.save_model(str(model_path))
        console.print(f"  Model saved to {model_path}")

        # Evaluate on test set (completely unseen during training)
        console.print(f"\n[blue]Evaluating on {len(test_df)} holdout samples...[/blue]")

        X_test, y_test = matcher._extract_features_and_labels(test_df, binary=True)
        X_test = matcher._impute_missing(X_test)
        y_pred = matcher.model.predict(X_test)

        # Overall metrics
        overall_acc = accuracy_score(y_test, y_pred)
        overall_f1 = f1_score(y_test, y_pred, average="weighted")
        overall_precision = precision_score(y_test, y_pred, average="weighted")
        overall_recall = recall_score(y_test, y_pred, average="weighted")

        console.print(f"\n{'=' * 60}")
        console.print(f"[bold]EVALUATION ON {test_pct}% HOLDOUT ({len(test_df)} samples)[/bold]")
        console.print("=" * 60)
        console.print("\nOverall:")
        console.print(f"  Accuracy:  {overall_acc:.3f}")
        console.print(f"  F1:        {overall_f1:.3f}")
        console.print(f"  Precision: {overall_precision:.3f}")
        console.print(f"  Recall:    {overall_recall:.3f}")

        # Extract top 5 feature importances
        feature_importances = dict(zip(matcher.feature_names, matcher.model.feature_importances_))
        top_5_features = sorted(feature_importances.items(), key=lambda x: -x[1])[:5]

        console.print("\nTop 5 features by importance:")
        for feat, imp in top_5_features:
            console.print(f"  {feat}: {imp:.3f}")

        # Per-dataset metrics
        results = {}
        console.print("\nPer-dataset results:")
        for dataset in sorted(test_df["dataset"].unique()):
            ds_test = test_df[test_df["dataset"] == dataset]
            X_ds, y_ds = matcher._extract_features_and_labels(ds_test, binary=True)
            X_ds = matcher._impute_missing(X_ds)
            y_ds_pred = matcher.model.predict(X_ds)

            ds_acc = accuracy_score(y_ds, y_ds_pred)
            ds_f1 = f1_score(y_ds, y_ds_pred, average="weighted")
            ds_precision = precision_score(y_ds, y_ds_pred, average="weighted", zero_division=0)
            ds_recall = recall_score(y_ds, y_ds_pred, average="weighted", zero_division=0)
            n_match = int((y_ds == 1).sum())
            n_no_match = int((y_ds == 0).sum())

            console.print(
                f"  {dataset}: acc={ds_acc:.3f}, f1={ds_f1:.3f} "
                f"(n={len(ds_test)}, match={n_match}, no_match={n_no_match})"
            )

            results[dataset] = {
                "n_samples": len(ds_test),
                "n_match": n_match,
                "n_no_match": n_no_match,
                "accuracy": ds_acc,
                "f1": ds_f1,
                "precision": ds_precision,
                "recall": ds_recall,
            }

        # Save results to CSV
        if not skip_save:
            output_dir.mkdir(parents=True, exist_ok=True)
            results_file = output_dir / "model_performance.csv"

            fieldnames = [
                "run_date",
                "data_pull_date",
                "dataset",
                "n_train",
                "n_test",
                "train_size",
                "n_samples",
                "n_match",
                "n_no_match",
                "accuracy",
                "f1",
                "precision",
                "recall",
                "split_seed",
                "model_name",
                "top1_feature",
                "top1_importance",
                "top2_feature",
                "top2_importance",
                "top3_feature",
                "top3_importance",
                "top4_feature",
                "top4_importance",
                "top5_feature",
                "top5_importance",
            ]

            write_header = not results_file.exists()

            with open(results_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                if write_header:
                    writer.writeheader()

                for dataset_name, metrics in results.items():
                    row = {
                        "run_date": run_date.isoformat(),
                        "data_pull_date": run_date.isoformat(),
                        "dataset": dataset_name,
                        "n_train": len(train_df),
                        "n_test": len(test_df),
                        "train_size": train_size,
                        "n_samples": metrics.get("n_samples", 0),
                        "n_match": metrics.get("n_match", 0),
                        "n_no_match": metrics.get("n_no_match", 0),
                        "accuracy": f"{metrics.get('accuracy', 0):.4f}",
                        "f1": f"{metrics.get('f1', 0):.4f}",
                        "precision": f"{metrics.get('precision', 0):.4f}",
                        "recall": f"{metrics.get('recall', 0):.4f}",
                        "split_seed": seed,
                        "model_name": model_path.name,
                        "top1_feature": top_5_features[0][0] if len(top_5_features) > 0 else "",
                        "top1_importance": f"{top_5_features[0][1]:.4f}"
                        if len(top_5_features) > 0
                        else "",
                        "top2_feature": top_5_features[1][0] if len(top_5_features) > 1 else "",
                        "top2_importance": f"{top_5_features[1][1]:.4f}"
                        if len(top_5_features) > 1
                        else "",
                        "top3_feature": top_5_features[2][0] if len(top_5_features) > 2 else "",
                        "top3_importance": f"{top_5_features[2][1]:.4f}"
                        if len(top_5_features) > 2
                        else "",
                        "top4_feature": top_5_features[3][0] if len(top_5_features) > 3 else "",
                        "top4_importance": f"{top_5_features[3][1]:.4f}"
                        if len(top_5_features) > 3
                        else "",
                        "top5_feature": top_5_features[4][0] if len(top_5_features) > 4 else "",
                        "top5_importance": f"{top_5_features[4][1]:.4f}"
                        if len(top_5_features) > 4
                        else "",
                    }
                    writer.writerow(row)

            console.print(f"\n[green]Results saved to {results_file}[/green]")

    console.print("\n[green]Benchmark complete![/green]")


@app.command()
def version():
    """Show version information."""
    from . import __version__

    console.print(f"matcher version {__version__}")


if __name__ == "__main__":
    app()
