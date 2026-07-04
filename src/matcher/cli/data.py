"""Data acquisition and management commands."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from rich.markup import escape

from .utils import console

# Create data group
data_app = typer.Typer(
    name="data",
    help="Data acquisition and management commands",
    no_args_is_help=True,
)

# Create fetch subgroup
fetch_app = typer.Typer(
    name="fetch",
    help="Fetch road data from various sources",
    no_args_is_help=True,
)
data_app.add_typer(fetch_app, name="fetch")


def _print_fetch_results_summary(results: dict[str, Path | None]) -> None:
    """Print a summary of fetch results, listing any failures."""
    failed = sorted(name for name, path in results.items() if path is None)
    success = len(results) - len(failed)
    if failed:
        console.print(f"\n[green]Fetched {success}/{len(results)} datasets[/green]")
        console.print(f"[yellow]Failed ({len(failed)}):[/yellow]")
        for name in failed:
            console.print(f"  [red]✗[/red] {name}")
    else:
        console.print(f"\n[green]All {len(results)} datasets fetched successfully![/green]")


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
    skip_quality_check: bool = typer.Option(
        False,
        "--skip-quality-check",
        help="Skip quality regression check against saved fingerprint",
    ),
):
    """Fetch target/local road data from municipal GIS portals.

    Downloads data from ArcGIS, WFS, OGC API Features, or direct download
    based on the dataset's YAML configuration. By default, skips files that
    already exist (use --force to re-fetch).

    Examples:
        matcher data fetch target us_boston_streets      # Fetch specific dataset
        matcher data fetch target --prefix us_boston     # Fetch all Boston datasets
        matcher data fetch target --all                  # Fetch all datasets
        matcher data fetch target --all --workers 8      # Fetch all with 8 workers
    """
    from ..fetch import target as target_module

    output_dir.mkdir(parents=True, exist_ok=True)

    if fetch_all:
        console.print(f"[blue]Fetching all datasets ({workers} workers)...[/blue]")
        results = target_module.fetch_all_datasets(
            output_dir, page_size, force, workers, skip_quality_check
        )
        _print_fetch_results_summary(results)

    elif prefix:
        console.print(
            f"[blue]Fetching datasets with prefix '{prefix}' ({workers} workers)...[/blue]"
        )
        results = target_module.fetch_datasets_by_prefix(
            prefix, output_dir, page_size, force, workers, skip_quality_check
        )
        if not results:
            console.print(f"[red]No datasets found matching prefix: {prefix}[/red]")
            raise typer.Exit(1)
        _print_fetch_results_summary(results)

    elif dataset_name:
        console.print(f"[blue]Fetching dataset: {dataset_name}[/blue]")
        result = target_module.fetch_dataset(
            dataset_name, output_dir, page_size, force, skip_quality_check
        )
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
        matcher data fetch reference us_boston_streets           # Fetch Overture (default)
        matcher data fetch reference us_boston_streets -s osm    # Fetch OSM
        matcher data fetch reference us_boston_streets -s overture -s osm  # Both
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
            dataset_list = ", ".join(sorted(available)[:10])
            if len(available) > 10:
                dataset_list += f"\n  ... and {len(available) - 10} more"
            console.print(f"[yellow]Available datasets: {dataset_list}[/yellow]")
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

    overture_fetched = False
    osm_fetched = False

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
            overture_fetched = True

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
            osm_fetched = True

    # Update last_fetch in dataset config (per source type)
    from ..datasets.schema import update_last_fetch
    from ..fetch.metadata import load_metadata

    if overture_fetched:
        overture_seg_file = overture_segments_filename(dataset_name)
        meta = load_metadata(output_dir / overture_seg_file)
        update_last_fetch(
            dataset_name,
            fetch_type="reference",
            bbox=original_bbox.to_tuple(),
            bbox_buffered=overture_bbox.to_tuple() if overture_buffer else None,
            bbox_buffer_m=overture_buffer,
            feature_count=meta.feature_count if meta else 0,
            geometry_types=meta.geometry_types if meta else [],
            output_path=str(output_dir),
        )
        console.print(f"[blue]Updated last_fetch.reference for {dataset_name}[/blue]")

    if osm_fetched:
        osm_seg_file = osm_segments_filename(dataset_name)
        meta = load_metadata(output_dir / osm_seg_file)
        update_last_fetch(
            dataset_name,
            fetch_type="osm",
            bbox=original_bbox.to_tuple(),
            bbox_buffered=osm_bbox.to_tuple() if osm_buffer else None,
            bbox_buffer_m=osm_buffer,
            feature_count=meta.feature_count if meta else 0,
            geometry_types=meta.geometry_types if meta else [],
            output_path=str(output_dir),
        )
        console.print(f"[blue]Updated last_fetch.osm for {dataset_name}[/blue]")


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
    workers: int = typer.Option(
        2,
        "--workers",
        "-w",
        help="Number of parallel workers for batch fetch (default: 2, Overture uses DuckDB parallelism internally)",
    ),
):
    """Fetch Overture reference data for dataset(s).

    Examples:
        matcher data fetch overture us_boston_streets      # Single dataset
        matcher data fetch overture --prefix us_           # All US datasets
        matcher data fetch overture --all                  # All datasets
        matcher data fetch overture --all --workers 3      # Parallel batch
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

    errors = _parallel_reference_fetch(
        datasets=datasets,
        output_dir=output_dir,
        sources={"overture"},
        bbox_buffer=bbox_buffer,
        force=force,
        workers=workers,
    )

    if errors:
        console.print(f"\n[red]Failed: {', '.join(errors)}[/red]")
        raise typer.Exit(1)
    console.print(
        f"\n[green]Successfully fetched Overture data for {len(datasets)} dataset(s)[/green]"
    )


def _parallel_reference_fetch(
    datasets: list[str],
    output_dir: Path,
    sources: set[str],
    bbox_buffer: float | None = None,
    force: bool = False,
    workers: int = 2,
    cache_dir: Path | None = None,
    no_cache: bool = False,
    keep_pbf: bool = False,
) -> list[str]:
    """Run reference fetch across multiple datasets in parallel.

    Args:
        datasets: List of dataset names to fetch
        output_dir: Output directory
        sources: Set of sources ("overture", "osm")
        bbox_buffer: Bbox buffer in meters
        force: Re-fetch even if files exist
        workers: Number of parallel workers
        cache_dir: Cache directory for PBF files
        no_cache: Force fresh download
        keep_pbf: Keep PBF files

    Returns:
        List of dataset names that failed
    """
    errors: list[str] = []

    # Single dataset — no threading needed
    if len(datasets) == 1:
        name = datasets[0]
        try:
            _fetch_reference_impl(
                dataset_name=name,
                output_dir=output_dir,
                sources=sources,
                bbox_buffer=bbox_buffer,
                force=force,
                cache_dir=cache_dir,
                no_cache=no_cache,
                keep_pbf=keep_pbf,
            )
        except Exception as e:
            console.print(f"[red]Error fetching {name}: {escape(str(e))}[/red]")
            errors.append(name)
        return errors

    # Multiple datasets — parallelize
    console.print(f"[blue]Using {workers} parallel worker(s)[/blue]")

    def _fetch_one(name: str) -> tuple[str, Exception | None]:
        try:
            _fetch_reference_impl(
                dataset_name=name,
                output_dir=output_dir,
                sources=sources,
                bbox_buffer=bbox_buffer,
                force=force,
                cache_dir=cache_dir,
                no_cache=no_cache,
                keep_pbf=keep_pbf,
            )
            return (name, None)
        except Exception as e:
            return (name, e)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, name): name for name in sorted(datasets)}

        for future in as_completed(futures):
            name, error = future.result()
            if error is not None:
                console.print(f"[red]Error fetching {name}: {escape(str(error))}[/red]")
                errors.append(name)
            else:
                console.print(f"[green]Fetched reference data for {name}[/green]")

    return errors


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
    workers: int = typer.Option(
        4,
        "--workers",
        "-w",
        help="Number of parallel workers for batch fetch (default: 4)",
    ),
):
    """Fetch OSM reference data for dataset(s).

    Examples:
        matcher data fetch osm us_boston_streets      # Single dataset
        matcher data fetch osm --prefix us_           # All US datasets
        matcher data fetch osm --all                  # All datasets
        matcher data fetch osm --all --workers 4      # Parallel batch
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

    errors = _parallel_reference_fetch(
        datasets=datasets,
        output_dir=output_dir,
        sources={"osm"},
        bbox_buffer=bbox_buffer,
        force=force,
        workers=workers,
        cache_dir=cache_dir,
        no_cache=no_cache,
        keep_pbf=keep_pbf,
    )

    if errors:
        console.print(f"\n[red]Failed: {', '.join(errors)}[/red]")
        raise typer.Exit(1)
    console.print(f"\n[green]Successfully fetched OSM data for {len(datasets)} dataset(s)[/green]")


@fetch_app.command("all")
def fetch_all(
    dataset_name: str = typer.Argument(
        None,
        help="Dataset name to fetch all data for",
    ),
    prefix: str | None = typer.Option(
        None,
        "--prefix",
        "-p",
        help="Fetch all data for datasets matching this prefix",
    ),
    all_datasets: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Fetch all data for all datasets",
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
    workers: int = typer.Option(
        4,
        "--workers",
        "-w",
        help="Number of parallel workers for batch mode (default: 4)",
    ),
    skip_quality_check: bool = typer.Option(
        False,
        "--skip-quality-check",
        help="Skip quality regression checks when re-fetching",
    ),
):
    """Fetch both target and reference data for dataset(s).

    Fetches target (local/ArcGIS) data and Overture reference data in parallel.
    This is the command to use when setting up a new dataset for labeling.
    By default, skips files that already exist (use --force to re-fetch).

    Examples:
        matcher data fetch all us_boston_streets       # Single dataset
        matcher data fetch all --prefix us_boston      # All Boston datasets
        matcher data fetch all --all --workers 4      # All datasets, parallel
    """
    from ..datasets.schema import list_dataset_configs
    from ..fetch import target as target_module

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which datasets to process
    if all_datasets:
        datasets = list_dataset_configs()
    elif prefix:
        all_configs = list_dataset_configs()
        datasets = [d for d in all_configs if d.startswith(prefix)]
    elif dataset_name:
        datasets = [dataset_name]
    else:
        console.print("[red]Error: Provide a dataset name, --prefix, or --all[/red]")
        raise typer.Exit(1)

    if not datasets:
        console.print("[yellow]No datasets found[/yellow]")
        raise typer.Exit(0)

    def _fetch_all_for_one(ds_name: str) -> list[tuple[str, str | Exception]]:
        """Fetch target + reference for a single dataset. Returns list of errors."""
        ds_errors: list[tuple[str, str | Exception]] = []

        def fetch_target_data():
            try:
                result = target_module.fetch_dataset(
                    ds_name, output_dir, page_size, force, skip_quality_check
                )
                return ("target", result)
            except Exception as e:
                return ("target", e)

        def fetch_reference_data():
            try:
                _fetch_reference_impl(
                    dataset_name=ds_name,
                    output_dir=output_dir,
                    sources={"overture"},
                    bbox_buffer=bbox_buffer,
                    force=force,
                )
                return ("reference", True)
            except Exception as e:
                return ("reference", e)

        # Run target + reference in parallel (always 2 tasks per dataset)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(fetch_target_data),
                executor.submit(fetch_reference_data),
            ]

            for future in as_completed(futures):
                fetch_name, result = future.result()
                if isinstance(result, Exception):
                    ds_errors.append((f"{ds_name}/{fetch_name}", result))
                    console.print(
                        f"[red]Error fetching {ds_name}/{fetch_name}: {escape(str(result))}[/red]"
                    )
                elif fetch_name == "target":
                    if result:
                        console.print(f"[green]{ds_name}: target saved to {result}[/green]")
                    else:
                        ds_errors.append(
                            (f"{ds_name}/target", "Fetch failed or requires manual download")
                        )
                else:
                    console.print(f"[green]{ds_name}: reference fetched[/green]")

        return ds_errors

    # Single dataset — run directly
    if len(datasets) == 1:
        console.print(f"[blue]Fetching all data for {datasets[0]}...[/blue]")
        errors = _fetch_all_for_one(datasets[0])
    else:
        # Multiple datasets — parallelize across datasets
        console.print(
            f"[blue]Fetching all data for {len(datasets)} dataset(s) ({workers} workers)...[/blue]"
        )
        errors = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fetch_all_for_one, ds): ds for ds in sorted(datasets)}
            for future in as_completed(futures):
                ds_errors = future.result()
                errors.extend(ds_errors)

    if errors:
        console.print(f"\n[yellow]Completed with {len(errors)} error(s):[/yellow]")
        for name, err in sorted(errors, key=lambda x: x[0]):
            err_str = str(err)
            # Truncate long error messages but keep the useful part
            if len(err_str) > 120:
                err_str = err_str[:120] + "..."
            console.print(f"  [red]✗[/red] {name}: {escape(err_str)}")
        raise typer.Exit(1)
    else:
        console.print("\n[green]All data fetched successfully![/green]")


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
        matcher data fetch list                  # List all datasets
        matcher data fetch list --prefix us_     # List US datasets
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
        matcher data fetch verify us_boston_streets      # Verify single dataset
        matcher data fetch verify --prefix ae_           # Verify by prefix
        matcher data fetch verify --all --dry-run        # Check all configs
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


@data_app.command("topology")
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


@data_app.command("repair")
def repair(
    data_path: Path = typer.Argument(
        ...,
        help="Path to GeoParquet file with road edges",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output path for repaired parquet file",
    ),
    snap_tolerance_m: float = typer.Option(
        5.0,
        "--snap-tolerance",
        help="Tolerance in meters for endpoint snapping",
    ),
    remove_islands: bool = typer.Option(
        True,
        "--remove-islands/--keep-islands",
        help="Remove critical-severity islands (single isolated segments)",
    ),
    id_column: str | None = typer.Option(
        None,
        "--id-column",
        help="Column containing edge IDs (auto-detected if not specified)",
    ),
):
    """Repair topology issues in a road network.

    Performs two types of repairs:
    - Snap near-miss endpoints within tolerance
    - Remove isolated segments (critical-severity islands)

    Examples:
        matcher data repair integrated.parquet -o repaired.parquet
        matcher data repair streets.parquet -o fixed.parquet --snap-tolerance 10.0
        matcher data repair data.parquet -o data.parquet --keep-islands
    """
    import geopandas as gpd

    from ..post_integration import repair_topology as _repair_topology

    console.print(f"[blue]Repairing topology in {data_path}[/blue]")

    try:
        gdf = gpd.read_parquet(data_path)
        console.print(f"  Loaded {len(gdf):,} edges")

        repaired_gdf, result = _repair_topology(
            gdf,
            snap_tolerance_m=snap_tolerance_m,
            remove_critical_islands=remove_islands,
            id_column=id_column,
        )

        # Print summary
        console.print("\n[bold]Repair Results:[/bold]")
        console.print(f"  Edges snapped: {result.edges_snapped}")
        console.print(f"  Edges removed: {result.edges_removed}")
        console.print(f"  Output edges: {len(repaired_gdf):,}")

        # Save result
        output.parent.mkdir(parents=True, exist_ok=True)
        repaired_gdf.to_parquet(output)
        console.print(f"\n[green]Repaired data saved to {output}[/green]")

    except Exception as e:
        console.print(f"[red]Topology repair failed: {escape(str(e))}[/red]")
        raise typer.Exit(1) from None


@data_app.command("quality")
def data_quality(
    dataset: str = typer.Argument(
        ...,
        help="Dataset name from datasets/*.yaml",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for JSON quality report",
    ),
    name_column: str | None = typer.Option(
        None,
        "--name-column",
        help="Column containing road names (auto-detected if not specified)",
    ),
    class_column: str | None = typer.Option(
        None,
        "--class-column",
        help="Column containing road class (auto-detected if not specified)",
    ),
    save_yaml: bool = typer.Option(
        False,
        "--save-yaml",
        "-s",
        help="Save fingerprint to the dataset's YAML config file",
    ),
    skip_drift: bool = typer.Option(
        False,
        "--skip-drift",
        help="Skip GPS drift detection (faster)",
    ),
    skip_duplicates: bool = typer.Option(
        False,
        "--skip-duplicates",
        help="Skip near-duplicate detection (faster)",
    ),
):
    """Generate a quality fingerprint for a dataset.

    Computes comprehensive quality metrics including:
    - Basic statistics (segment count, total length)
    - Length distribution (min, max, median, percentiles)
    - Geometry quality (vertex density, sharp turns, sinuosity)
    - GPS drift detection (zigzag, spike, loop patterns)
    - Near-duplicate detection
    - Topology metrics (islands, dead ends, connectivity)
    - Attribute metrics (name/class coverage, distribution)

    Examples:
        matcher data quality us_boston_streets
        matcher data quality us_boston_streets --save-yaml
        matcher data quality us_boston_streets -o quality.json -s
    """
    from ..config import CLASS_COLUMN, NAMES_COLUMN
    from ..datasets.loader import DatasetLoader
    from ..datasets.schema import (
        fingerprint_from_quality,
        get_dataset_config,
        update_quality_fingerprint,
    )
    from ..quality import save_quality_report
    from ..quality.metrics import compute_quality_metrics

    # Load from dataset config
    config = get_dataset_config(dataset)
    if config is None:
        console.print(f"[red]Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    # Load data
    try:
        console.print(f"[blue]Loading data for dataset: {dataset}[/blue]")
        loader = DatasetLoader()
        gdf = loader.load_target(dataset)

        # Use standardized column names from config module
        if name_column is None:
            name_column = NAMES_COLUMN
        if class_column is None:
            class_column = CLASS_COLUMN
    except FileNotFoundError as e:
        console.print(f"[red]Data file not found: {escape(str(e))}[/red]")
        raise typer.Exit(1) from None

    # Compute metrics
    try:
        console.print("[blue]Computing quality metrics...[/blue]")
        fingerprint = compute_quality_metrics(
            edges_gdf=gdf,
            dataset_name=dataset,
            name_column=name_column,
            class_column=class_column,
            detect_drift=not skip_drift,
            detect_duplicates=not skip_duplicates,
        )

        # Print summary
        console.print("\n[bold]Quality Fingerprint:[/bold]")
        console.print(f"  Dataset: {fingerprint.dataset_name}")
        console.print(f"  Segments: {fingerprint.total_segments:,}")
        console.print(f"  Total length: {fingerprint.total_length_m / 1000:.1f} km")

        console.print("\n[bold]Length Distribution:[/bold]")
        console.print(f"  Min: {fingerprint.length_min_m:.1f}m")
        console.print(f"  Median: {fingerprint.length_median_m:.1f}m")
        console.print(f"  Max: {fingerprint.length_max_m:.1f}m")
        console.print(f"  P5-P95: {fingerprint.length_p5_m:.1f}m - {fingerprint.length_p95_m:.1f}m")

        console.print("\n[bold]Geometry Quality:[/bold]")
        console.print(
            f"  Vertex density: {fingerprint.vertex_density_mean:.4f} "
            f"(±{fingerprint.vertex_density_std:.4f}) vertices/m"
        )
        console.print(f"  Invalid geometries: {fingerprint.invalid_geometry_count}")
        console.print(
            f"  Sharp turns (>150°): {fingerprint.sharp_angle_count} "
            f"({fingerprint.sharp_angle_ratio:.1%})"
        )
        console.print(f"  Mean sinuosity: {fingerprint.mean_segment_sinuosity:.3f}")
        console.print(
            f"  High sinuosity (>1.5): {fingerprint.high_sinuosity_count} "
            f"({fingerprint.high_sinuosity_ratio:.1%})"
        )

        if not skip_drift:
            console.print("\n[bold]GPS Artifacts:[/bold]")
            console.print(f"  Zigzag patterns: {fingerprint.zigzag_segment_count}")
            console.print(f"  Spikes: {fingerprint.spike_segment_count}")
            console.print(f"  Small loops: {fingerprint.loop_segment_count}")
            console.print(f"  Drift affected: {fingerprint.drift_affected_ratio:.1%}")

        if not skip_duplicates:
            console.print("\n[bold]Duplicates:[/bold]")
            console.print(
                f"  Near-duplicates: {fingerprint.near_duplicate_count} "
                f"({fingerprint.near_duplicate_ratio:.1%})"
            )

        console.print("\n[bold]Topology:[/bold]")
        console.print(f"  Connected components: {fingerprint.connected_components}")
        console.print(f"  Largest component: {fingerprint.largest_component_ratio:.1%}")
        console.print(f"  Islands: {fingerprint.island_count}")
        console.print(
            f"  Dead ends: {fingerprint.dead_end_count} ({fingerprint.dead_end_ratio:.1%})"
        )

        console.print("\n[bold]Attributes:[/bold]")
        console.print(f"  Name coverage: {fingerprint.name_coverage_ratio:.1%}")
        console.print(f"  Class coverage: {fingerprint.class_coverage_ratio:.1%}")

        if fingerprint.class_distribution:
            console.print("\n[bold]Class distribution:[/bold]")
            for cls, count in list(fingerprint.class_distribution.items())[:10]:
                console.print(f"  {cls}: {count:,}")

        # Save JSON report if requested
        if output:
            save_quality_report(fingerprint, output)
            console.print(f"\n[green]Quality report saved to {output}[/green]")

        # Save to dataset YAML if requested
        if save_yaml:
            yaml_fingerprint = fingerprint_from_quality(fingerprint)
            result = update_quality_fingerprint(dataset, yaml_fingerprint)
            if result:
                console.print(
                    f"\n[green]Quality fingerprint saved to datasets/{dataset}.yaml[/green]"
                )
            else:
                console.print(f"[red]Failed to save to dataset YAML: {dataset}[/red]")

    except Exception as e:
        console.print(f"[red]Quality analysis failed: {escape(str(e))}[/red]")
        raise typer.Exit(1) from None


@data_app.command("validate")
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
        matcher data validate
        matcher data validate data/raw
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


@data_app.command("cache")
def compute_features(
    dataset: str = typer.Argument(
        None,
        help="Dataset ID to compute features for (e.g., us_boston_streets)",
    ),
    all_datasets: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Compute features for all datasets with fetched data",
    ),
    prefix: str = typer.Option(
        None,
        "--prefix",
        "-p",
        help="Compute features for datasets matching this prefix",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Recompute even if cached (ignores existing feature cache)",
    ),
    workers: int = typer.Option(
        -1,
        "--workers",
        "-w",
        help="Number of parallel workers for feature computation (-1 for auto)",
    ),
    generate_candidates: bool = typer.Option(
        False,
        "--generate-candidates",
        "-c",
        help="Also generate scored candidates cache for labeling UI (runs ML scoring)",
    ),
    needs_labels: bool = typer.Option(
        False,
        "--needs-labels",
        help="Only cache datasets with fewer than MIN_LABELS_PER_DATASET labels",
    ),
):
    """Compute and cache features for dataset(s) without ML scoring.

    This pre-computes the expensive feature computation step (~90% of processing time)
    and caches the results. When the labeling UI loads, it can use the cached features
    and only run ML prediction (fast).

    Features are versioned - when feature computation logic changes, bump FEATURE_VERSION
    in config.py and old caches will be ignored.

    Use --generate-candidates to also run ML scoring and cache the final candidates,
    making the labeling UI load instantly.

    Examples:
        matcher data cache us_boston_streets           # Single dataset
        matcher data cache --prefix us_                # All US datasets
        matcher data cache --all                       # All datasets with data
        matcher data cache --all --force               # Recompute all
        matcher data cache us_boston_streets -w 4      # Limit workers
        matcher data cache --all --generate-candidates # Full precache for UI
    """
    from ..datasets.loader import DatasetLoader
    from ..labeling.data_loader import (
        build_views_from_feature_df,
        compute_features_only,
        get_feature_cache_info,
        load_feature_cache,
        save_candidates_to_cache,
        save_feature_cache,
    )

    loader = DatasetLoader()

    def compute_for_dataset(dataset_id: str) -> tuple[bool, str]:
        """Compute features for a single dataset. Returns (success, error_msg)."""
        ref_path = loader.find_reference_path(dataset_id)
        target_path = loader.find_target_path(dataset_id)
        if ref_path is None or target_path is None:
            console.print(f"[yellow]Skipping {dataset_id}: missing data files[/yellow]")
            return False, "missing data files"

        # Check cache
        cache_info = get_feature_cache_info(dataset_id, ref_path, target_path)
        feature_cache_exists = cache_info["exists"]

        if feature_cache_exists and not force and not generate_candidates:
            console.print(
                f"[blue]Skipping {dataset_id}: feature cache exists "
                f"({cache_info['candidate_count']:,} candidates, "
                f"version {cache_info['version']})[/blue]"
            )
            return True, ""

        # Use standardized Overture-format column names for parquet files
        # (the fetch step transforms source columns to these during data ingestion)
        from ..config import CLASS_COLUMN, NAMES_COLUMN

        ref_name_column = NAMES_COLUMN
        target_name_column = NAMES_COLUMN
        ref_class_column = CLASS_COLUMN
        target_class_column = CLASS_COLUMN

        try:
            feature_df = None
            reference = None
            target = None

            # Load or compute features
            if feature_cache_exists and not force:
                console.print(f"[blue]Loading cached features for {dataset_id}...[/blue]")
                feature_df = load_feature_cache(dataset_id)
            else:
                console.print(f"[blue]Computing features for {dataset_id}...[/blue]")

                # Load data
                reference = loader._load_gdf(ref_path)
                target = loader._load_gdf(target_path)

                console.print(f"  Reference: {len(reference):,} segments")
                console.print(f"  Target: {len(target):,} segments")

                # Compute features
                feature_df = compute_features_only(
                    reference=reference,
                    target=target,
                    ref_name_column=ref_name_column,
                    target_name_column=target_name_column,
                    ref_class_column=ref_class_column,
                    target_class_column=target_class_column,
                    n_jobs=workers,
                )

                if len(feature_df) == 0:
                    console.print(f"[yellow]  No candidates generated for {dataset_id}[/yellow]")
                    return False, "no candidates generated"

                # Save feature cache
                cache_path = save_feature_cache(dataset_id, feature_df)
                console.print(
                    f"[green]  Saved {len(feature_df):,} features to {cache_path.name}[/green]"
                )

            # Generate candidates cache if requested
            if generate_candidates and feature_df is not None:
                console.print(f"[blue]  Generating scored candidates for {dataset_id}...[/blue]")

                # Load geodataframes if not already loaded (when using cached features)
                if reference is None:
                    reference = loader._load_gdf(ref_path)
                if target is None:
                    target = loader._load_gdf(target_path)

                # Build views (runs ML scoring, filter to review band for labeling UI)
                views = build_views_from_feature_df(
                    feature_df=feature_df,
                    reference=reference,
                    target=target,
                    ref_id_column="id",
                    target_id_column="id",
                    ref_name_column=ref_name_column,
                    target_name_column=target_name_column,
                    ref_class_column=ref_class_column,
                    target_class_column=target_class_column,
                    filter_to_review_band=True,
                )

                if views:
                    candidates_path = save_candidates_to_cache(dataset_id, views)
                    console.print(
                        f"[green]  Saved {len(views):,} candidates to "
                        f"{candidates_path.name}[/green]"
                    )
                else:
                    console.print(f"[yellow]  No candidates to cache for {dataset_id}[/yellow]")

            return True, ""

        except Exception as e:
            console.print(f"[red]  Error computing features for {dataset_id}: {e}[/red]")
            return False, str(e)

    # Determine which datasets to process
    datasets_to_process: list[str] = []

    if all_datasets:
        # Find all datasets with both target and reference data
        datasets_to_process = loader.list_available()

        if not datasets_to_process:
            console.print("[yellow]No datasets found with fetched data[/yellow]")
            raise typer.Exit(1)

        console.print(f"[blue]Found {len(datasets_to_process)} datasets with data[/blue]")

    elif prefix:
        # Find datasets matching prefix
        for dataset_id in loader.list_available():
            if dataset_id.startswith(prefix):
                datasets_to_process.append(dataset_id)

        if not datasets_to_process:
            console.print(f"[yellow]No datasets found with prefix '{prefix}'[/yellow]")
            raise typer.Exit(1)

        console.print(
            f"[blue]Found {len(datasets_to_process)} datasets matching prefix '{prefix}'[/blue]"
        )

    elif dataset:
        datasets_to_process = [dataset]

    else:
        console.print("[red]Error: Provide a dataset name, --prefix, or --all[/red]")
        raise typer.Exit(1)

    if needs_labels:
        from ..config import MIN_LABELS_PER_DATASET
        from ..labeling.label_store import LabelStore

        filtered = []
        for ds in datasets_to_process:
            store = LabelStore(ds)
            if len(store.df) < MIN_LABELS_PER_DATASET:
                filtered.append(ds)

        skipped = len(datasets_to_process) - len(filtered)
        datasets_to_process = filtered
        console.print(
            f"[blue]Filtered to {len(datasets_to_process)} datasets needing labels "
            f"({skipped} already have >= {MIN_LABELS_PER_DATASET})[/blue]"
        )

        if not datasets_to_process:
            console.print(
                f"[yellow]All datasets already have >= {MIN_LABELS_PER_DATASET} labels; nothing to do.[/yellow]"
            )
            raise typer.Exit(0)

    # Process datasets sequentially
    success_count = 0
    skip_count = 0
    errors: list[tuple[str, str]] = []

    for dataset_id in datasets_to_process:
        # Check cache BEFORE computation to distinguish skip vs compute
        cache_info_before = get_feature_cache_info(dataset_id)
        had_cache = cache_info_before.get("exists", False)

        success, err_msg = compute_for_dataset(dataset_id)

        if not success:
            errors.append((dataset_id, err_msg))
        elif had_cache and not force:
            skip_count += 1
        else:
            success_count += 1

    # Summary
    console.print()
    console.print("[bold]Summary:[/bold]")
    console.print(f"  Computed: {success_count}")
    console.print(f"  Skipped (cached): {skip_count}")
    if errors:
        console.print(f"  [red]Failed: {len(errors)}[/red]")
        for name, err in sorted(errors):
            err_str = err if len(err) <= 120 else err[:120] + "..."
            console.print(f"    [red]✗[/red] {name}: {escape(err_str)}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Reclassify command — fix incorrect class mappings in stored data
# ---------------------------------------------------------------------------

# Per-dataset corrected mapping logic.  Each function takes a GeoDataFrame
# (from the raw parquet) and returns a Series of corrected class values.


def _reclassify_co_bogota(gdf):
    """Bogotá: MVITCLA 4=Peatonal→pedestrian, 6=Sin definir→unclassified (were swapped)."""
    cls = gdf["class"].copy()
    tags = gdf.get("source_tags")
    if tags is not None:
        for i, tag in enumerate(tags):
            if not isinstance(tag, dict):
                continue
            raw = str(tag.get("MVITCLA", ""))
            if raw == "4":
                cls.iat[i] = "pedestrian"
            elif raw == "6":
                cls.iat[i] = "unclassified"
    return cls


def _reclassify_nl_amsterdam(gdf):
    """Amsterdam: switch from wegbehsrt (administrator) to bst_code (road type)."""
    import pandas as pd

    bst_map = {
        "FP": "cycleway",
        "VP": "footway",
        "BUS": "service",
        "ERF": "living_street",
        "PP": "service",
        "VZ": "service",
        "PST": "service",
        "AFR": "trunk",
        "OPR": "trunk",
        "PAR": "service",
        "NRB": "residential",
        "HR": "primary",
    }
    frc_map = {
        0: "motorway",
        1: "trunk",
        2: "primary",
        3: "secondary",
        4: "tertiary",
        5: "tertiary",
        6: "residential",
        7: "residential",
    }

    tags = gdf.get("source_tags")
    new_cls = pd.Series("unclassified", index=gdf.index)
    if tags is None:
        return new_cls

    for i, tag in enumerate(tags):
        if not isinstance(tag, dict):
            continue
        bst = str(tag.get("bst_code", "")).strip().upper()
        if bst in bst_map:
            new_cls.iat[i] = bst_map[bst]
        elif bst == "RB":
            # RB = carriageway — fall back to FRC hierarchy
            frc = tag.get("frc")
            if frc is not None:
                try:
                    new_cls.iat[i] = frc_map.get(int(frc), "residential")
                except (ValueError, TypeError):
                    new_cls.iat[i] = "residential"
            else:
                new_cls.iat[i] = "residential"
        else:
            # Unknown bst_code — use frc fallback
            frc = tag.get("frc")
            if frc is not None:
                try:
                    new_cls.iat[i] = frc_map.get(int(frc), "unclassified")
                except (ValueError, TypeError):
                    new_cls.iat[i] = "unclassified"
    return new_cls


def _reclassify_de_berlin(gdf):
    """Berlin: override V-class segments using strassenklasse2."""
    v_overrides = {
        "FUWE": "footway",
        "PLAT": "pedestrian",
        "FUBR": "footway",
        "PSTR": "path",
        "KGA": "service",
        "PARK": "service",
    }

    tags = gdf.get("source_tags")
    cls = gdf["class"].copy()
    if tags is None:
        return cls

    for i, tag in enumerate(tags):
        if not isinstance(tag, dict):
            continue
        sk1 = str(tag.get("strassenklasse1", "")).strip()
        if sk1 == "V":
            sk2 = str(tag.get("strassenklasse2", "")).strip().upper()
            if sk2 in v_overrides:
                cls.iat[i] = v_overrides[sk2]
            # else: keep "service" (the V default)
    return cls


def _reclassify_ke_kisumu(gdf):
    """Kisumu: derive class from RID_8 prefix letter."""
    import pandas as pd

    prefix_map = {
        "A": "trunk",
        "B": "trunk",
        "C": "primary",
        "D": "secondary",
        "E": "tertiary",
    }

    tags = gdf.get("source_tags")
    new_cls = pd.Series("unclassified", index=gdf.index)
    if tags is None:
        return new_cls

    for i, tag in enumerate(tags):
        if not isinstance(tag, dict):
            continue
        rid8 = str(tag.get("RID_8", "")).strip()
        if rid8:
            prefix = rid8[0].upper()
            new_cls.iat[i] = prefix_map.get(prefix, "unclassified")
    return new_cls


def _reclassify_sg_singapore_footpaths(gdf):
    """Singapore footpaths: fix unknown→footway (type=sidewalk)."""
    cls = gdf["class"].copy()
    cls = cls.replace({"unknown": "footway"})
    return cls


_RECLASSIFY_HANDLERS: dict = {
    "co_bogota_roads": _reclassify_co_bogota,
    "nl_amsterdam_roads": _reclassify_nl_amsterdam,
    "de_berlin_roads": _reclassify_de_berlin,
    "ke_kisumu_roads": _reclassify_ke_kisumu,
    "sg_singapore_footpaths": _reclassify_sg_singapore_footpaths,
}


@data_app.command("reclassify")
def reclassify(
    dataset_name: str = typer.Argument(
        None,
        help="Dataset to reclassify (e.g., co_bogota_roads). Omit for --all.",
    ),
    all_datasets: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Reclassify all datasets with known mapping errors",
    ),
    data_dir: Path = typer.Option(
        Path("data/raw"),
        "--data-dir",
        "-d",
        help="Directory containing raw parquet files",
    ),
    labels_dir: Path = typer.Option(
        Path("labels"),
        "--labels",
        "-l",
        help="Directory containing labels (for data store updates)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would change without modifying files",
    ),
):
    """Fix incorrect road class mappings in raw parquet and data store.

    Updates the `class` column in raw parquet files and corresponding
    data store records for datasets with known classification errors.

    After reclassifying, run `matcher backfill` to recompute class_similarity
    features for all affected labeled pairs.

    Supported datasets: co_bogota_roads, nl_amsterdam_roads, de_berlin_roads,
    ke_kisumu_roads, sg_singapore_footpaths.

    Examples:
        matcher data reclassify co_bogota_roads          # Single dataset
        matcher data reclassify --all                     # All known datasets
        matcher data reclassify co_bogota_roads --dry-run # Preview changes
    """
    import geopandas as gpd

    from ..filenames import find_target_file
    from ..labeling.data_store import DataStore

    datasets: list[str] = []
    if all_datasets:
        datasets = list(_RECLASSIFY_HANDLERS.keys())
    elif dataset_name:
        if dataset_name not in _RECLASSIFY_HANDLERS:
            console.print(f"[red]No reclassify handler for '{dataset_name}'[/red]")
            console.print(f"[yellow]Available: {', '.join(sorted(_RECLASSIFY_HANDLERS))}[/yellow]")
            raise typer.Exit(1)
        datasets = [dataset_name]
    else:
        console.print("[red]Error: Provide a dataset name or --all[/red]")
        raise typer.Exit(1)

    total_raw_changed = 0
    total_store_updated = 0

    for ds_name in sorted(datasets):
        console.print(f"\n[bold blue]Reclassifying {ds_name}...[/bold blue]")
        handler = _RECLASSIFY_HANDLERS[ds_name]

        # 1. Load raw parquet
        target_path = find_target_file(data_dir, ds_name)
        if target_path is None:
            console.print(f"  [yellow]Raw parquet not found in {data_dir}, skipping[/yellow]")
            continue

        gdf = gpd.read_parquet(target_path)
        old_cls = gdf["class"].copy()

        # 2. Compute corrected classes
        new_cls = handler(gdf)

        # 3. Summarize changes to raw parquet
        changed_mask = old_cls != new_cls
        n_changed = int(changed_mask.sum())
        total_raw_changed += n_changed

        console.print(f"  Raw parquet: {n_changed:,} / {len(gdf):,} records changed")

        if n_changed > 0:
            # Show before/after distribution for changed records
            old_dist = old_cls[changed_mask].value_counts()
            new_dist = new_cls[changed_mask].value_counts()
            console.print("  [dim]Changed records — old class distribution:[/dim]")
            for cls_val, count in old_dist.items():
                console.print(f"    {cls_val}: {count:,}")
            console.print("  [dim]Changed records — new class distribution:[/dim]")
            for cls_val, count in new_dist.items():
                console.print(f"    {cls_val}: {count:,}")

        # 4. Update data store records
        data_store_dir = Path(labels_dir) / "data"
        store = DataStore(ds_name, data_dir=data_store_dir)
        store_gdf = store.gdf

        store_updated = 0
        if len(store_gdf) > 0:
            # Build lookup: target_id → corrected class for ALL records
            # (covers both raw-parquet changes and stale data store values)
            id_to_class: dict[str, str] = {}
            for idx in gdf.index:
                tid = str(gdf.at[idx, "id"])
                id_to_class[tid] = str(
                    new_cls.iat[idx] if hasattr(new_cls, "iat") else new_cls.iloc[idx]
                )

            # Update data store records where target_class differs
            for _, row in store_gdf.iterrows():
                target_id = str(row["target_id"])
                if target_id in id_to_class:
                    new_val = id_to_class[target_id]
                    old_val = str(row.get("target_class", ""))
                    if old_val != new_val and store.update_class(
                        gers_id=str(row["gers_id"]),
                        target_id=target_id,
                        target_class=new_val,
                    ):
                        store_updated += 1

        total_store_updated += store_updated
        console.print(f"  Data store: {store_updated:,} / {len(store_gdf):,} records updated")

        # 5. Save changes
        if not dry_run and (n_changed > 0 or store_updated > 0):
            if n_changed > 0:
                gdf["class"] = new_cls
                gdf.to_parquet(target_path)
                console.print(f"  [green]Saved raw parquet: {target_path}[/green]")
            if store_updated > 0:
                store.save()
                console.print(f"  [green]Saved data store for {ds_name}[/green]")
        elif dry_run:
            console.print("  [yellow]Dry run — no files modified[/yellow]")

    # Summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Raw parquet records changed: {total_raw_changed:,}")
    console.print(f"  Data store records updated: {total_store_updated:,}")
    if dry_run:
        console.print("  [yellow]Dry run — no files were modified[/yellow]")
    else:
        console.print(
            "\n[blue]Next step: run 'matcher backfill' to recompute "
            "class_similarity features[/blue]"
        )


def _fill_spatial_context(groups: list[dict], dataset_name: str) -> None:
    """Fill in spatial context segments for each group.

    For every group, computes an envelope (capped at ~500m x 500m) from all
    group geometries, spatial-queries the raw parquet files, and adds
    nearby-but-not-in-group segments as context.
    """
    import math

    import geopandas as gpd
    from shapely.geometry import box, mapping, shape
    from shapely.ops import unary_union

    from ..config import CLASS_COLUMN, NAMES_COLUMN
    from ..filenames import PROJECT_ROOT, find_overture_segments, find_target_file
    from ..pipeline.runner import _extract_name_string, _is_nan

    data_dir = PROJECT_ROOT / "data" / "raw"

    ref_path = find_overture_segments(data_dir, dataset_name)
    target_path = find_target_file(data_dir, dataset_name)

    if not ref_path or not target_path:
        console.print("  [yellow]Cannot fill spatial context: missing data files[/yellow]")
        return

    ref_gdf = gpd.read_parquet(ref_path, columns=["id", "geometry", NAMES_COLUMN, CLASS_COLUMN])
    target_gdf = gpd.read_parquet(
        target_path, columns=["id", "geometry", NAMES_COLUMN, CLASS_COLUMN]
    )

    # Ensure spatial index exists
    _ = ref_gdf.sindex
    _ = target_gdf.sindex

    for group in groups:
        ref_geoms = group.get("ref_geometries", {})
        target_geoms = group.get("target_geometries", {})

        if not ref_geoms and not target_geoms:
            continue

        # Collect all geometries and compute envelope
        all_shapes = []
        for geom_dict in list(ref_geoms.values()) + list(target_geoms.values()):
            try:
                all_shapes.append(shape(geom_dict))
            except Exception:
                continue

        if not all_shapes:
            continue

        envelope = unary_union(all_shapes).envelope
        minx, miny, maxx, maxy = envelope.bounds

        # Cap at ~500m x 500m using degree approximation
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * math.cos(math.radians(cy))
        max_half_deg_lat = 0.25 / km_per_deg_lat  # 250m in degrees
        max_half_deg_lon = 0.25 / km_per_deg_lon if km_per_deg_lon > 0 else 0.00225

        if (maxy - miny) > 2 * max_half_deg_lat or (maxx - minx) > 2 * max_half_deg_lon:
            minx = cx - max_half_deg_lon
            maxx = cx + max_half_deg_lon
            miny = cy - max_half_deg_lat
            maxy = cy + max_half_deg_lat

        # Snap envelope bounds outward to coord precision so clipped+rounded geoms stay inside
        scale = 10**6
        minx = math.floor(minx * scale) / scale
        miny = math.floor(miny * scale) / scale
        maxx = math.ceil(maxx * scale) / scale
        maxy = math.ceil(maxy * scale) / scale
        envelope = box(minx, miny, maxx, maxy)

        # Spatial query for ref segments within envelope
        ref_ids_in_group = set(group.get("ref_ids", list(ref_geoms.keys())))
        ref_hits = ref_gdf.sindex.query(envelope, predicate="intersects")
        context_ref = {}
        context_ref_names = {}
        context_ref_classes = {}
        for idx in ref_hits:
            row = ref_gdf.iloc[idx]
            rid = str(row.get("id", ""))
            if rid in ref_ids_in_group or not rid:
                continue
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            clipped = geom.intersection(envelope)
            if clipped.is_empty:
                continue
            context_ref[rid] = _round_geojson_coords(mapping(clipped))
            name_val = row.get(NAMES_COLUMN)
            context_ref_names[rid] = _extract_name_string(name_val) if not _is_nan(name_val) else ""
            cls_val = row.get(CLASS_COLUMN)
            context_ref_classes[rid] = str(cls_val) if not _is_nan(cls_val) else ""

        # Spatial query for target segments within envelope
        target_ids_in_group = set(group.get("target_ids", list(target_geoms.keys())))
        target_hits = target_gdf.sindex.query(envelope, predicate="intersects")
        context_target = {}
        context_target_names = {}
        context_target_classes = {}
        for idx in target_hits:
            row = target_gdf.iloc[idx]
            tid = str(row.get("id", ""))
            if tid in target_ids_in_group or not tid:
                continue
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            clipped = geom.intersection(envelope)
            if clipped.is_empty:
                continue
            context_target[tid] = _round_geojson_coords(mapping(clipped))
            name_val = row.get(NAMES_COLUMN)
            context_target_names[tid] = (
                _extract_name_string(name_val) if not _is_nan(name_val) else ""
            )
            cls_val = row.get(CLASS_COLUMN)
            context_target_classes[tid] = str(cls_val) if not _is_nan(cls_val) else ""

        # Clip group's own geometries to envelope (remove if entirely outside)
        for rid in list(ref_geoms.keys()):
            try:
                g_shape = shape(ref_geoms[rid])
                clipped = g_shape.intersection(envelope)
                if clipped.is_empty:
                    del ref_geoms[rid]
                else:
                    ref_geoms[rid] = _round_geojson_coords(mapping(clipped))
            except Exception:
                pass
        for tid in list(target_geoms.keys()):
            try:
                g_shape = shape(target_geoms[tid])
                clipped = g_shape.intersection(envelope)
                if clipped.is_empty:
                    del target_geoms[tid]
                else:
                    target_geoms[tid] = _round_geojson_coords(mapping(clipped))
            except Exception:
                pass

        # Sync ID lists, names, classes, and edges with surviving geometries
        surviving_refs = set(ref_geoms.keys())
        surviving_targets = set(target_geoms.keys())
        group["ref_ids"] = [rid for rid in group.get("ref_ids", []) if rid in surviving_refs]
        group["target_ids"] = [
            tid for tid in group.get("target_ids", []) if tid in surviving_targets
        ]
        if "ref_names" in group:
            group["ref_names"] = {
                k: v for k, v in group["ref_names"].items() if k in surviving_refs
            }
        if "target_names" in group:
            group["target_names"] = {
                k: v for k, v in group["target_names"].items() if k in surviving_targets
            }
        if "ref_classes" in group:
            group["ref_classes"] = {
                k: v for k, v in group["ref_classes"].items() if k in surviving_refs
            }
        if "target_classes" in group:
            group["target_classes"] = {
                k: v for k, v in group["target_classes"].items() if k in surviving_targets
            }
        edges_before = len(group.get("edges", []))
        if "edges" in group:
            group["edges"] = [
                e
                for e in group["edges"]
                if e["ref_id"] in surviving_refs and e["target_id"] in surviving_targets
            ]

        # Envelope clipping can drop most of a large group's edges. Any
        # pre-computed alternatives / optimizer_assignment were derived from the
        # FULL (pre-clip) edge set, so they may reference edges that no longer
        # exist in the group. Re-sync them to the surviving edges so the batch
        # never ships options that are not subsets of the group's own edges.
        if len(group.get("edges", [])) != edges_before:
            from ..matching.alternatives import prune_group_options_to_edges

            prune_group_options_to_edges(group, ref_geoms=ref_geoms, target_geoms=target_geoms)

        # Store on group dict
        group["envelope"] = mapping(envelope)
        group["context_ref_ids"] = list(context_ref.keys())
        group["context_target_ids"] = list(context_target.keys())
        group["context_ref_geometries"] = context_ref
        group["context_target_geometries"] = context_target
        group["context_ref_names"] = context_ref_names
        group["context_target_names"] = context_target_names
        group["context_ref_classes"] = context_ref_classes
        group["context_target_classes"] = context_target_classes


def _round_geojson_coords(geojson: dict, precision: int = 6) -> dict:
    """Round coordinates in a GeoJSON geometry dict."""

    def _round_coords(coords):
        if isinstance(coords[0], (int, float)):
            return [round(v, precision) for v in coords]
        return [_round_coords(c) for c in coords]

    return {**geojson, "coordinates": _round_coords(geojson["coordinates"])}


@data_app.command("stitch-batch")
def stitch_batch(
    dataset: str = typer.Argument(
        None,
        help="Dataset ID to generate stitch batch for (e.g., us_boston_streets)",
    ),
    all_datasets: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Generate batches for all datasets with groups sidecars",
    ),
    batch_size: int = typer.Option(
        15,
        "--batch-size",
        "-n",
        help="Number of groups per batch (default: 15)",
    ),
    k_alternatives: int = typer.Option(
        5,
        "--alternatives",
        "-k",
        help="Number of top-K alternatives per group (default: 5)",
    ),
):
    """Generate a curated batch of M:N groups for stitching review.

    Reads the groups sidecar JSON from pipeline output, scores groups
    by review value (label overlap + borderline confidence), and writes
    a batch file with pre-computed alternatives.

    Examples:
        matcher data stitch-batch us_boston_streets        # Single dataset
        matcher data stitch-batch --all                    # All datasets with groups
        matcher data stitch-batch us_boston_streets -n 30  # Custom batch size
    """
    import json
    from datetime import UTC, datetime

    from ..filenames import (
        PROJECT_ROOT,
        groups_sidecar_path,
        stitch_batch_path,
    )
    from ..labeling.stitching_store import StitchingLabelStore
    from ..matching.alternatives import generate_top_k_alternatives
    from ..matching.batch_selection import select_stitching_batch

    if batch_size <= 0:
        console.print("[red]Error: --batch-size must be positive[/red]")
        raise typer.Exit(1)
    if k_alternatives <= 0:
        console.print("[red]Error: --alternatives must be positive[/red]")
        raise typer.Exit(1)

    output_dir = PROJECT_ROOT / "data" / "output"

    # Determine which datasets to process
    if all_datasets:
        # Find all groups sidecar files
        datasets_to_process = []
        if output_dir.exists():
            for sidecar_file in sorted(output_dir.glob("*_groups.json")):
                # Extract dataset name from filename: us_boston_streets_groups.json
                ds_name = sidecar_file.stem.replace("_groups", "")
                datasets_to_process.append(ds_name)
        if not datasets_to_process:
            console.print("[yellow]No groups sidecar files found in data/output/[/yellow]")
            raise typer.Exit(0)
        console.print(
            f"[blue]Found {len(datasets_to_process)} datasets with groups sidecars[/blue]"
        )
    elif dataset:
        datasets_to_process = [dataset]
    else:
        console.print("[red]Error: Provide a dataset name or --all[/red]")
        raise typer.Exit(1)

    for ds_name in datasets_to_process:
        console.print(f"\n[bold blue]Processing {ds_name}...[/bold blue]")

        # Find groups sidecar
        from ..filenames import bridge_filename

        bridge_path = output_dir / bridge_filename(ds_name)
        sidecar_path = groups_sidecar_path(bridge_path)

        if not sidecar_path.exists():
            console.print(f"  [yellow]No groups sidecar found at {sidecar_path}[/yellow]")
            continue

        # Load groups sidecar
        try:
            sidecar = json.loads(sidecar_path.read_text())
            groups = sidecar.get("groups", [])
        except (json.JSONDecodeError, OSError) as e:
            console.print(f"  [red]Failed to load groups sidecar: {e}[/red]")
            continue

        console.print(f"  Loaded {len(groups)} groups from sidecar")

        if not groups:
            console.print("  [yellow]No groups to process[/yellow]")
            continue

        # Load existing stitching labels to skip already-reviewed
        stitch_store = StitchingLabelStore(ds_name)
        reviewed_ids = stitch_store.get_reviewed_group_ids(ds_name)
        console.print(f"  Already reviewed: {len(reviewed_ids)} groups")

        # Pre-compute alternatives per group. These drive both the batch
        # selection scoring AND the review UI's one-click option picker, so
        # they are intentionally retained in the batch file. Alternatives are
        # just ID pairs + confidence (no geometries), so the size cost is small.
        console.print(f"  Computing top-{k_alternatives} alternatives per group...")
        for group in groups:
            alternatives = generate_top_k_alternatives(
                component_edges=group.get("edges", []),
                ref_geoms=group.get("ref_geometries", {}),
                target_geoms=group.get("target_geometries", {}),
                k=k_alternatives,
            )
            group["alternatives"] = alternatives

        # Select batch
        selected = select_stitching_batch(
            groups=groups,
            reviewed_group_ids=reviewed_ids,
            k=batch_size,
        )

        if not selected:
            console.print("  [yellow]No groups selected for batch[/yellow]")
            continue

        # NOTE: alternatives and optimizer_assignment are deliberately kept on
        # each selected group so the review UI can pre-seed the optimizer's
        # proposed assignment and offer the top-K alternatives as one-click
        # options ("verify, don't construct").

        # Fill in spatial context for each group
        console.print("  Filling spatial context...")
        _fill_spatial_context(selected, ds_name)
        ctx_counts = [
            len(g.get("context_ref_ids", [])) + len(g.get("context_target_ids", []))
            for g in selected
        ]
        console.print(
            f"  Added context: {sum(ctx_counts)} segments "
            f"(avg {sum(ctx_counts) / len(ctx_counts):.0f}/group)"
        )

        # Write batch file
        batch = {
            "dataset_id": ds_name,
            "generated_at": datetime.now(UTC).isoformat(),
            "batch_size": len(selected),
            "groups": selected,
        }

        batch_path = stitch_batch_path(ds_name)
        batch_path.parent.mkdir(parents=True, exist_ok=True)
        batch_path.write_text(json.dumps(batch, indent=2))

        # Count tiers
        tier_counts = {}
        for g in selected:
            tier = g.get("review_tier", "unknown")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        console.print(f"  [green]Wrote batch of {len(selected)} groups to {batch_path}[/green]")
        tier_str = ", ".join(f"{t}={c}" for t, c in sorted(tier_counts.items()))
        console.print(f"  Tiers: {tier_str}")
