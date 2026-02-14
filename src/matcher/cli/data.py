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
