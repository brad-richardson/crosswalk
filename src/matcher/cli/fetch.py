"""Fetch subcommands for downloading road data."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer

from ._app import console, fetch_app


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
    workers: int = typer.Option(
        4,
        "--workers",
        "-w",
        help="Number of parallel download workers (default: 4)",
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
        matcher fetch target --all --workers 8      # Fetch all with 8 workers
    """
    from ..fetch import target as target_module

    output_dir.mkdir(parents=True, exist_ok=True)

    if fetch_all:
        console.print(f"[blue]Fetching all datasets ({workers} workers)...[/blue]")
        results = target_module.fetch_all_datasets(output_dir, page_size, force, workers)
        success = sum(1 for p in results.values() if p is not None)
        console.print(f"[green]Fetched {success}/{len(results)} datasets[/green]")

    elif prefix:
        console.print(
            f"[blue]Fetching datasets with prefix '{prefix}' ({workers} workers)...[/blue]"
        )
        results = target_module.fetch_datasets_by_prefix(
            prefix, output_dir, page_size, force, workers
        )
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
    from ..datasets.schema import get_dataset_config, list_dataset_configs
    from ..fetch import osm as osm_module
    from ..fetch import overture as ov_module
    from ..filenames import (
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

    from ..datasets.schema import LastFetch, get_datasets_dir, save_dataset_config
    from ..fetch.metadata import load_metadata

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


@fetch_app.command("overture")
def fetch_overture(
    dataset_name: str = typer.Argument(
        None,
        help="Dataset name to fetch Overture data for (uses bbox from config)",
    ),
    prefix: str | None = typer.Option(
        None,
        "--prefix",
        "-p",
        help="Fetch Overture data for all datasets matching prefix",
    ),
    all_datasets: bool = typer.Option(
        False,
        "--all",
        help="Fetch Overture data for all datasets",
    ),
    output_dir: Path = typer.Option(
        Path("data/raw"),
        "--output",
        "-o",
        help="Output directory for fetched data",
    ),
    bbox_buffer: float | None = typer.Option(
        None,
        "--bbox-buffer",
        help="Expand bbox by this distance (meters). Defaults to 1km.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-fetch even if files already exist",
    ),
):
    """Fetch Overture reference data for dataset(s).

    Examples:
        matcher fetch overture us_boston_streets      # Single dataset
        matcher fetch overture --prefix us_           # All US datasets
        matcher fetch overture --all                  # All datasets
    """
    from ..datasets.schema import list_dataset_configs

    # Determine which datasets to fetch
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
        console.print("[yellow]No datasets found[/yellow]")
        raise typer.Exit(0)

    console.print(f"[blue]Fetching Overture data for {len(datasets)} dataset(s)...[/blue]")

    errors = []
    for name in sorted(datasets):
        try:
            console.print(f"\n[blue]{'=' * 60}[/blue]")
            console.print(f"[blue]Fetching Overture for: {name}[/blue]")
            _fetch_reference_impl(
                dataset_name=name,
                output_dir=output_dir,
                sources={"overture"},
                bbox_buffer=bbox_buffer,
                force=force,
            )
        except Exception as e:
            console.print(f"[red]Error fetching {name}: {e}[/red]")
            errors.append(name)

    if errors:
        console.print(f"\n[red]Failed: {', '.join(errors)}[/red]")
        raise typer.Exit(1)
    console.print(
        f"\n[green]Successfully fetched Overture data for {len(datasets)} dataset(s)[/green]"
    )


@fetch_app.command("osm")
def fetch_osm(
    dataset_name: str = typer.Argument(
        None,
        help="Dataset name to fetch OSM data for (uses bbox from config)",
    ),
    prefix: str | None = typer.Option(
        None,
        "--prefix",
        "-p",
        help="Fetch OSM data for all datasets matching prefix",
    ),
    all_datasets: bool = typer.Option(
        False,
        "--all",
        help="Fetch OSM data for all datasets",
    ),
    output_dir: Path = typer.Option(
        Path("data/raw"),
        "--output",
        "-o",
        help="Output directory for fetched data",
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
        help="Expand bbox by this distance (meters). Defaults to 5km for OSM.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-fetch even if files already exist",
    ),
):
    """Fetch OSM reference data for dataset(s).

    Examples:
        matcher fetch osm us_boston_streets      # Single dataset
        matcher fetch osm --prefix us_           # All US datasets
        matcher fetch osm --all                  # All datasets
    """
    from ..datasets.schema import list_dataset_configs

    # Determine which datasets to fetch
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
        console.print("[yellow]No datasets found[/yellow]")
        raise typer.Exit(0)

    console.print(f"[blue]Fetching OSM data for {len(datasets)} dataset(s)...[/blue]")

    errors = []
    for name in sorted(datasets):
        try:
            console.print(f"\n[blue]{'=' * 60}[/blue]")
            console.print(f"[blue]Fetching OSM for: {name}[/blue]")
            _fetch_reference_impl(
                dataset_name=name,
                output_dir=output_dir,
                sources={"osm"},
                cache_dir=cache_dir,
                no_cache=no_cache,
                keep_pbf=keep_pbf,
                bbox_buffer=bbox_buffer,
                force=force,
            )
        except Exception as e:
            console.print(f"[red]Error fetching {name}: {e}[/red]")
            errors.append(name)

    if errors:
        console.print(f"\n[red]Failed: {', '.join(errors)}[/red]")
        raise typer.Exit(1)
    console.print(f"\n[green]Successfully fetched OSM data for {len(datasets)} dataset(s)[/green]")


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
    from ..fetch import target as target_module

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
    from ..fetch import target as target_module

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

    from ..datasets.schema import get_dataset_config, list_dataset_configs

    def verify_dataset(name: str, dry_run: bool) -> tuple[str, bool, str]:
        """Verify a single dataset config. Returns (name, success, message)."""
        config = get_dataset_config(name)
        if config is None:
            return (name, False, "Config file not found or invalid YAML")

        # Check source type
        source_type = config.source.type if config.source else "unknown"
        if source_type == "manual":
            return (name, True, "[yellow]Manual download[/yellow] - skipped")

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

    for dataset in sorted(datasets):
        _name, success, message = verify_dataset(dataset, dry_run)
        if success:
            if "yellow" in message:
                warning_count += 1
            else:
                success_count += 1
        else:
            error_count += 1
        status_icon = "[green]✓[/green]" if success else "[red]✗[/red]"
        console.print(f"  {status_icon} {dataset}: {message}")

    console.print()
    console.print(
        f"[green]{success_count} passed[/green], "
        f"[yellow]{warning_count} warnings[/yellow], "
        f"[red]{error_count} errors[/red]"
    )

    if error_count > 0:
        raise typer.Exit(1)
