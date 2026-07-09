"""Data acquisition and management commands."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.markup import escape

from .utils import console

if TYPE_CHECKING:
    from shapely.geometry import Polygon

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
        crosswalk data fetch target us_boston_streets      # Fetch specific dataset
        crosswalk data fetch target --prefix us_boston     # Fetch all Boston datasets
        crosswalk data fetch target --all                  # Fetch all datasets
        crosswalk data fetch target --all --workers 8      # Fetch all with 8 workers
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
        crosswalk data fetch reference us_boston_streets           # Fetch Overture (default)
        crosswalk data fetch reference us_boston_streets -s osm    # Fetch OSM
        crosswalk data fetch reference us_boston_streets -s overture -s osm  # Both
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
        crosswalk data fetch overture us_boston_streets      # Single dataset
        crosswalk data fetch overture --prefix us_           # All US datasets
        crosswalk data fetch overture --all                  # All datasets
        crosswalk data fetch overture --all --workers 3      # Parallel batch
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
        crosswalk data fetch osm us_boston_streets      # Single dataset
        crosswalk data fetch osm --prefix us_           # All US datasets
        crosswalk data fetch osm --all                  # All datasets
        crosswalk data fetch osm --all --workers 4      # Parallel batch
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
        crosswalk data fetch all us_boston_streets       # Single dataset
        crosswalk data fetch all --prefix us_boston      # All Boston datasets
        crosswalk data fetch all --all --workers 4      # All datasets, parallel
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
        crosswalk data fetch list                  # List all datasets
        crosswalk data fetch list --prefix us_     # List US datasets
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
        crosswalk data fetch verify us_boston_streets      # Verify single dataset
        crosswalk data fetch verify --prefix ae_           # Verify by prefix
        crosswalk data fetch verify --all --dry-run        # Check all configs
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
        crosswalk data repair integrated.parquet -o repaired.parquet
        crosswalk data repair streets.parquet -o fixed.parquet --snap-tolerance 10.0
        crosswalk data repair data.parquet -o data.parquet --keep-islands
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
        crosswalk data quality us_boston_streets
        crosswalk data quality us_boston_streets --save-yaml
        crosswalk data quality us_boston_streets -o quality.json -s
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
        crosswalk data validate
        crosswalk data validate data/raw
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
        crosswalk data cache us_boston_streets           # Single dataset
        crosswalk data cache --prefix us_                # All US datasets
        crosswalk data cache --all                       # All datasets with data
        crosswalk data cache --all --force               # Recompute all
        crosswalk data cache us_boston_streets -w 4      # Limit workers
        crosswalk data cache --all --generate-candidates # Full precache for UI
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

    After reclassifying, run `crosswalk backfill` to recompute class_similarity
    features for all affected labeled pairs.

    Supported datasets: co_bogota_roads, nl_amsterdam_roads, de_berlin_roads,
    ke_kisumu_roads, sg_singapore_footpaths.

    Examples:
        crosswalk data reclassify co_bogota_roads          # Single dataset
        crosswalk data reclassify --all                     # All known datasets
        crosswalk data reclassify co_bogota_roads --dry-run # Preview changes
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
            "\n[blue]Next step: run 'crosswalk backfill' to recompute "
            "class_similarity features[/blue]"
        )


# Minimum half-extent (in meters) for the context envelope. Small groups are
# padded out to at least a 2*CONTEXT_MIN_HALF_M box so they still pull in
# surrounding roads for orientation. Large groups are NOT capped to this — the
# envelope always fully contains the group's own geometry (see
# _compute_context_envelope). Renamed intent from the old destructive 500m cap:
# the box now bounds only the CONTEXT layer, never the group itself.
CONTEXT_MIN_HALF_M = 250.0

# Fraction of the group's own extent added as a context margin around it, so a
# large elongated group still gets a proportional ring of context roads rather
# than a fixed hairline border.
CONTEXT_MARGIN_RATIO = 0.15


def _compute_context_envelope(all_shapes: list) -> "Polygon":
    """Compute the context envelope box for a group's geometries.

    The returned box is guaranteed to fully contain every group geometry (plus a
    proportional margin) and to be at least ``2 * CONTEXT_MIN_HALF_M`` across so
    small groups still gather surrounding context. Unlike the previous
    implementation this NEVER shrinks below the group's own bounds — group
    segments are never clipped or deleted.
    """
    import math

    from shapely.geometry import box
    from shapely.ops import unary_union

    envelope = unary_union(all_shapes).envelope
    minx, miny, maxx, maxy = envelope.bounds

    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(cy))
    min_half_deg_lat = (CONTEXT_MIN_HALF_M / 1000.0) / km_per_deg_lat
    min_half_deg_lon = (
        (CONTEXT_MIN_HALF_M / 1000.0) / km_per_deg_lon if km_per_deg_lon > 0 else 0.00225
    )

    # Half-extent = the group's own half-extent plus a proportional margin, but
    # never below the minimum context radius. max() ensures the box always
    # contains the full group, however large.
    half_deg_lat = max(min_half_deg_lat, (maxy - miny) / 2 * (1 + 2 * CONTEXT_MARGIN_RATIO))
    half_deg_lon = max(min_half_deg_lon, (maxx - minx) / 2 * (1 + 2 * CONTEXT_MARGIN_RATIO))

    minx, maxx = cx - half_deg_lon, cx + half_deg_lon
    miny, maxy = cy - half_deg_lat, cy + half_deg_lat

    # Snap envelope bounds outward to coord precision so rounded geoms stay inside
    scale = 10**6
    minx = math.floor(minx * scale) / scale
    miny = math.floor(miny * scale) / scale
    maxx = math.ceil(maxx * scale) / scale
    maxy = math.ceil(maxy * scale) / scale
    return box(minx, miny, maxx, maxy)


def _add_spatial_context_to_group(
    group: dict,
    ref_gdf,
    target_gdf,
    *,
    names_column: str,
    class_column: str,
) -> None:
    """Add nearby-but-not-in-group context segments to a single group.

    Pure with respect to file IO (callers pass already-loaded GeoDataFrames) so
    the preservation invariant can be unit-tested. The group's OWN segments,
    edges, optimizer_assignment and alternatives are left completely intact — the
    context envelope bounds only the added context layer.
    """
    from shapely.geometry import mapping, shape

    from ..pipeline.runner import _extract_name_string, _is_nan

    ref_geoms = group.get("ref_geometries", {})
    target_geoms = group.get("target_geometries", {})

    if not ref_geoms and not target_geoms:
        return

    all_shapes = []
    for geom_dict in list(ref_geoms.values()) + list(target_geoms.values()):
        try:
            all_shapes.append(shape(geom_dict))
        except Exception:
            continue

    if not all_shapes:
        return

    envelope = _compute_context_envelope(all_shapes)

    # Audit metric: the group's full edge set is never reduced by context
    # filling. Record it before/after so a regression that reintroduces clipping
    # is loud (see the assertion below and the stored n_edges_full/rendered).
    n_edges_full = len(group.get("edges", []))

    # Spatial query for ref segments within envelope (context only — never group)
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
        name_val = row.get(names_column)
        context_ref_names[rid] = _extract_name_string(name_val) if not _is_nan(name_val) else ""
        cls_val = row.get(class_column)
        context_ref_classes[rid] = str(cls_val) if not _is_nan(cls_val) else ""

    # Spatial query for target segments within envelope (context only)
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
        name_val = row.get(names_column)
        context_target_names[tid] = _extract_name_string(name_val) if not _is_nan(name_val) else ""
        cls_val = row.get(class_column)
        context_target_classes[tid] = str(cls_val) if not _is_nan(cls_val) else ""

    # NOTE: The group's own geometries, ref_ids, target_ids, edges,
    # optimizer_assignment and alternatives are intentionally left UNTOUCHED.
    # A previous version clipped these to the envelope and deleted out-of-box
    # segments/edges, silently truncating large chains before the pack/UI ever
    # saw them (root cause of the optimizer "under-selection" investigation).

    # Store context + audit metrics on the group dict
    group["envelope"] = mapping(envelope)
    group["context_ref_ids"] = list(context_ref.keys())
    group["context_target_ids"] = list(context_target.keys())
    group["context_ref_geometries"] = context_ref
    group["context_target_geometries"] = context_target
    group["context_ref_names"] = context_ref_names
    group["context_target_names"] = context_target_names
    group["context_ref_classes"] = context_ref_classes
    group["context_target_classes"] = context_target_classes

    n_edges_rendered = len(group.get("edges", []))
    group["n_edges_full"] = n_edges_full
    group["n_edges_rendered"] = n_edges_rendered
    group["context_clipped"] = n_edges_rendered != n_edges_full
    # Post-fix invariant: context filling must never drop a group edge.
    assert n_edges_rendered == n_edges_full, (
        f"context fill dropped {n_edges_full - n_edges_rendered} edges from group "
        f"{group.get('group_id', '?')} — group data must never be clipped"
    )


def _fill_spatial_context(groups: list[dict], dataset_name: str) -> None:
    """Fill in spatial context segments for each group.

    For every group, computes a context envelope (at least ~500m x 500m, and
    always large enough to fully contain the group), spatial-queries the raw
    parquet files, and adds nearby-but-not-in-group segments as context. The
    group's own segments/edges are never clipped or removed — the envelope
    bounds only the context layer.
    """
    import geopandas as gpd

    from ..config import CLASS_COLUMN, NAMES_COLUMN
    from ..filenames import PROJECT_ROOT, find_overture_segments, find_target_file

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

    n_clipped = 0
    for group in groups:
        _add_spatial_context_to_group(
            group,
            ref_gdf,
            target_gdf,
            names_column=NAMES_COLUMN,
            class_column=CLASS_COLUMN,
        )
        if group.get("context_clipped"):
            n_clipped += 1

    if n_clipped:
        console.print(
            f"  [red]WARNING: {n_clipped} group(s) lost edges to context filling — "
            f"group data must never be clipped[/red]"
        )


def _round_geojson_coords(geojson: dict, precision: int = 6) -> dict:
    """Round coordinates in a GeoJSON geometry dict."""

    def _round_coords(coords):
        if isinstance(coords[0], (int, float)):
            return [round(v, precision) for v in coords]
        return [_round_coords(c) for c in coords]

    # GeometryCollection holds sub-geometries under "geometries", not
    # "coordinates" — recurse into each member.
    if geojson.get("type") == "GeometryCollection" and isinstance(geojson.get("geometries"), list):
        return {
            **geojson,
            "geometries": [_round_geojson_coords(g, precision) for g in geojson["geometries"]],
        }

    if "coordinates" not in geojson:
        return geojson

    return {**geojson, "coordinates": _round_coords(geojson["coordinates"])}


def _generate_stitch_batch_for_dataset(
    ds_name: str,
    *,
    output_dir,
    batch_size: int,
    k_alternatives: int,
    include_unvoted: bool,
) -> bool:
    """Generate one dataset's stitching review queue (``{ds}_batch.json``).

    Shared per-dataset body behind ``crosswalk data stitch-batch`` (single and
    ``--all``) and the refresh phase of ``stitch-batch-all``, so all three write
    identical queues. Returns True iff a non-empty batch was written.
    """
    import json
    from datetime import UTC, datetime

    from ..agent_labeling.panel_routing import (
        attach_panel_route_reasons,
        panel_failed_group_ids,
    )
    from ..filenames import bridge_filename, groups_sidecar_path, stitch_batch_path
    from ..labeling.stitching_store import StitchingLabelStore
    from ..matching.alternatives import generate_top_k_alternatives
    from ..matching.batch_selection import select_stitching_batch

    bridge_path = output_dir / bridge_filename(ds_name)
    sidecar_path = groups_sidecar_path(bridge_path)

    if not sidecar_path.exists():
        console.print(f"  [yellow]No groups sidecar found at {sidecar_path}[/yellow]")
        return False

    # Load groups sidecar
    try:
        sidecar = json.loads(sidecar_path.read_text())
        groups = sidecar.get("groups", [])
    except (json.JSONDecodeError, OSError) as e:
        console.print(f"  [red]Failed to load groups sidecar: {e}[/red]")
        return False

    console.print(f"  Loaded {len(groups)} groups from sidecar")

    if not groups:
        console.print("  [yellow]No groups to process[/yellow]")
        return False

    # Load existing stitching labels to skip already-reviewed
    stitch_store = StitchingLabelStore(ds_name)
    reviewed_ids = stitch_store.get_reviewed_group_ids(ds_name)
    console.print(f"  Already reviewed: {len(reviewed_ids)} groups")

    # Gate the human queue to groups the agent panel routed to human_review
    # (unless --include-unvoted). Restrict to the current sidecar's group ids
    # so stale votes on ids that no longer exist (re-segmentation) are
    # dropped. Passing this allow-list to select_stitching_batch confines the
    # tier sampling to panel failures.
    candidate_ids: set[str] | None = None
    if not include_unvoted:
        sidecar_ids = {g.get("group_id") for g in groups}
        failed_ids = panel_failed_group_ids(ds_name)
        candidate_ids = failed_ids & sidecar_ids
        eligible = candidate_ids - reviewed_ids
        console.print(
            f"  Panel-failure gate: {len(failed_ids)} routed to human_review, "
            f"{len(candidate_ids)} in current sidecar, {len(eligible)} unreviewed "
            f"(use --include-unvoted to sample all groups)"
        )
        if not eligible:
            console.print(
                "  [yellow]No unreviewed panel failures — writing an empty queue. "
                "Run the agent panel (crosswalk agent stitch-batch/stitch-run) to "
                "produce more, or pass --include-unvoted.[/yellow]"
            )

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
        candidate_group_ids=candidate_ids,
    )

    if not selected:
        console.print("  [yellow]No groups selected for batch[/yellow]")
        if not include_unvoted:
            # Gating is active and nothing qualified. Write an EMPTY queue so
            # a previously-generated (ungated) batch is not left serving stale
            # never-voted groups to the reviewer.
            empty_batch = {
                "dataset_id": ds_name,
                "generated_at": datetime.now(UTC).isoformat(),
                "batch_size": 0,
                "groups": [],
            }
            empty_path = stitch_batch_path(ds_name)
            empty_path.parent.mkdir(parents=True, exist_ok=True)
            empty_path.write_text(json.dumps(empty_batch, indent=2))
            console.print(f"  [green]Wrote empty queue to {empty_path}[/green]")
        return False

    # NOTE: alternatives and optimizer_assignment are deliberately kept on
    # each selected group so the review UI can pre-seed the optimizer's
    # proposed assignment and offer the top-K alternatives as one-click
    # options ("verify, don't construct").

    # Annotate each queued group with WHY the panel routed it to a human
    # (panel_route_reason + a short display variant). Pure annotation from
    # the latest consensus row — never changes which groups are selected.
    n_reasons = attach_panel_route_reasons(selected, ds_name)
    if n_reasons:
        console.print(f"  Panel route reasons attached: {n_reasons}/{len(selected)}")

    # Fill in spatial context for each group
    console.print("  Filling spatial context...")
    _fill_spatial_context(selected, ds_name)
    ctx_counts = [
        len(g.get("context_ref_ids", [])) + len(g.get("context_target_ids", [])) for g in selected
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

    # Maintenance invariant: a freshly generated queue must be in parity with
    # the sidecar it was built from (each entry's optimizer_assignment == the
    # sidecar group's selected edges). This is trivially true here because the
    # entries ARE the sidecar groups, but running the check makes the
    # stale-proposal invariant a guarded property of the rebuild path itself.
    from ..matching.stitch_queue_refresh import check_queue_optimizer_parity

    sidecar_by_id = {g.get("group_id"): g for g in groups}
    drift = check_queue_optimizer_parity(selected, sidecar_by_id)
    if drift:
        console.print(
            f"  [red]WARNING: {len(drift)} generated entries drifted from the "
            f"sidecar selected set (stale proposals): "
            f"{', '.join(d['group_id'] for d in drift[:5])}[/red]"
        )

    return True


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
        8,
        "--alternatives",
        "-k",
        help="Number of top-K organic alternatives per group (default: 8; "
        "two whole-group seed options are appended on top)",
    ),
    include_unvoted: bool = typer.Option(
        False,
        "--include-unvoted",
        help="Escape hatch: restore the legacy behavior of sampling ANY "
        "unreviewed group (including groups the agent panel never voted on). "
        "By default the human queue is gated to groups the panel routed to "
        "human_review, so it only surfaces panel failures.",
    ),
):
    """Generate a curated batch of M:N groups for HUMAN stitching review.

    Reads the groups sidecar JSON from pipeline output, scores groups by review
    value, and writes ``data/cache/stitch/{dataset}_batch.json`` for the web
    ``/stitching-review`` queue.

    By default the queue is GATED to groups the agent panel could not
    auto-accept (routed to ``human_review`` in ``data/agents/stitching/batches``,
    most-recent vote per group). This keeps the human's time on genuine panel
    failures rather than never-voted calibration samples. The tier sampling still
    runs, but only over that gated pool. Pass ``--include-unvoted`` to restore the
    legacy behavior of sampling any unreviewed group. (The agent PANEL feed —
    ``crosswalk agent stitch-batch`` — is unaffected and keeps sampling fresh
    groups.)

    Examples:
        crosswalk data stitch-batch us_boston_streets        # Panel-failure queue
        crosswalk data stitch-batch --all                    # All datasets with groups
        crosswalk data stitch-batch us_boston_streets -n 30  # Custom batch size
        crosswalk data stitch-batch us_boston_streets --include-unvoted  # Legacy
    """
    from ..filenames import PROJECT_ROOT

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
        _generate_stitch_batch_for_dataset(
            ds_name,
            output_dir=output_dir,
            batch_size=batch_size,
            k_alternatives=k_alternatives,
            include_unvoted=include_unvoted,
        )


@data_app.command("stitch-batch-all")
def stitch_batch_all(
    refresh: bool = typer.Option(
        True,
        "--refresh/--no-refresh",
        help="Regenerate every dataset's per-dataset queue from its groups "
        "sidecar before combining (default). Pass --no-refresh to combine the "
        "existing data/cache/stitch/*_batch.json files as-is.",
    ),
    batch_size: int = typer.Option(
        15,
        "--batch-size",
        "-n",
        help="Per-dataset groups sampled during --refresh (default: 15)",
    ),
    k_alternatives: int = typer.Option(
        8,
        "--alternatives",
        "-k",
        help="Number of top-K organic alternatives per group during --refresh",
    ),
    include_unvoted: bool = typer.Option(
        False,
        "--include-unvoted",
        help="During --refresh, sample ANY unreviewed group rather than gating "
        "to agent-panel failures (see stitch-batch).",
    ),
):
    """Combine every per-dataset queue into ONE cross-dataset review queue.

    Writes ``data/cache/stitch/__all___batch.json`` — a single queue holding
    every dataset's unreviewed M:N groups so the web ``/stitching-review`` mode
    can serve them all without switching datasets. Each group is stamped with its
    owning ``dataset_id`` so labels are recorded back to the correct
    ``labels/stitching/dataset=*/`` partition. Groups are ordered by dataset
    (alphabetical), preserving each dataset's value-sorted order within.

    By default this first regenerates each dataset's per-dataset queue (same
    machinery as ``stitch-batch --all``) so the combined queue is fresh; pass
    ``--no-refresh`` to combine the existing ``*_batch.json`` files verbatim.
    The combined file is a snapshot; the UI still filters out already-reviewed
    groups live (per-dataset labels), so re-running is cheap and safe.

    Examples:
        crosswalk data stitch-batch-all                 # Refresh all + combine
        crosswalk data stitch-batch-all --no-refresh    # Combine existing as-is
    """
    import json
    from datetime import UTC, datetime

    from ..datasets.loader import DatasetLoader
    from ..filenames import PROJECT_ROOT, STITCH_ALL_QUEUE, stitch_batch_path

    output_dir = PROJECT_ROOT / "data" / "output"

    # Gate everything on real dataset membership. The sidecar glob and the batch
    # cache both contain before_/after_/baseline_ comparison artifacts (from the
    # CLAUDE.md change-tracking workflow); those are NOT datasets. Admitting one
    # into the queue would route its review labels to a junk
    # labels/stitching/dataset=before_*/ partition no consumer reads (silent
    # loss), so they are excluded from BOTH the refresh and the combine.
    real_datasets = set(DatasetLoader().list_available())

    if refresh:
        # Regenerate every real dataset that has a groups sidecar, so the
        # combined queue reflects the current optimizer/panel state.
        datasets_to_refresh = []
        skipped = []
        if output_dir.exists():
            for sidecar_file in sorted(output_dir.glob("*_groups.json")):
                ds_name = sidecar_file.stem.replace("_groups", "")
                if ds_name in real_datasets:
                    datasets_to_refresh.append(ds_name)
                else:
                    skipped.append(ds_name)
        if skipped:
            console.print(
                f"[dim]Skipping {len(skipped)} non-dataset sidecars "
                f"(comparison artifacts): {', '.join(skipped)}[/dim]"
            )
        if not datasets_to_refresh:
            console.print("[yellow]No real datasets with groups sidecars found[/yellow]")
            raise typer.Exit(0)
        console.print(
            f"[blue]Refreshing {len(datasets_to_refresh)} per-dataset queues "
            f"before combining...[/blue]"
        )
        for ds_name in datasets_to_refresh:
            console.print(f"\n[bold blue]Processing {ds_name}...[/bold blue]")
            # One dataset's failure (e.g. a malformed panel CSV) must not abort
            # the whole combine — its existing cached batch is reused as-is.
            try:
                _generate_stitch_batch_for_dataset(
                    ds_name,
                    output_dir=output_dir,
                    batch_size=batch_size,
                    k_alternatives=k_alternatives,
                    include_unvoted=include_unvoted,
                )
            except Exception as e:
                console.print(
                    f"  [red]Refresh failed for {ds_name} ({e}); "
                    f"keeping its existing cached queue.[/red]"
                )

    # Combine every per-dataset batch file (skip the aggregate itself) into one
    # queue. Each group is tagged with its owning dataset_id so downstream
    # rendering + label recording route to the correct partition.
    cache_dir = stitch_batch_path(STITCH_ALL_QUEUE).parent
    combined_groups: list[dict] = []
    source_summary: list[tuple[str, int]] = []
    combine_skipped: list[str] = []
    if cache_dir.exists():
        for batch_file in sorted(cache_dir.glob("*_batch.json")):
            ds_name = batch_file.stem.replace("_batch", "")
            if ds_name == STITCH_ALL_QUEUE:
                continue
            try:
                batch = json.loads(batch_file.read_text())
            except (json.JSONDecodeError, OSError) as e:
                console.print(f"  [red]Skipping {batch_file.name}: {e}[/red]")
                continue
            groups = batch.get("groups", [])
            owner = batch.get("dataset_id", ds_name)
            # Never fold a non-dataset cache (comparison artifact, or an id that
            # is no longer a known dataset) into the queue — its labels would
            # route to a junk partition. Apply the gate to the resolved OWNER,
            # not just the filename, so a mislabeled batch can't sneak in.
            if owner not in real_datasets:
                if groups:
                    combine_skipped.append(owner)
                continue
            for g in groups:
                # Stamp the owning dataset so the UI can route labels back and
                # resolve per-group spatial-context membership.
                g["dataset_id"] = owner
            combined_groups.extend(groups)
            if groups:
                source_summary.append((owner, len(groups)))

    combined = {
        "dataset_id": STITCH_ALL_QUEUE,
        "generated_at": datetime.now(UTC).isoformat(),
        "batch_size": len(combined_groups),
        "groups": combined_groups,
    }

    all_path = stitch_batch_path(STITCH_ALL_QUEUE)
    all_path.parent.mkdir(parents=True, exist_ok=True)
    all_path.write_text(json.dumps(combined, indent=2))

    console.print(
        f"\n[green]Wrote combined queue of {len(combined_groups)} groups "
        f"from {len(source_summary)} datasets to {all_path}[/green]"
    )
    for owner, n in source_summary:
        console.print(f"  {owner}: {n}")
    if combine_skipped:
        console.print(
            f"[dim]Excluded {len(combine_skipped)} non-dataset batch caches "
            f"(comparison artifacts): {', '.join(sorted(set(combine_skipped)))}[/dim]"
        )
    if not combined_groups:
        console.print(
            "[yellow]Combined queue is empty — no per-dataset batches with groups. "
            "Generate some with 'crosswalk data stitch-batch --all' first.[/yellow]"
        )


@data_app.command("stitch-refresh-queue")
def stitch_refresh_queue(
    dataset: str = typer.Argument(
        None,
        help="Dataset ID to refresh the review queue for (e.g., us_boston_streets)",
    ),
    all_datasets: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Refresh queues for all datasets with an existing batch cache",
    ),
    k_alternatives: int = typer.Option(
        8,
        "--alternatives",
        "-k",
        help="Number of top-K organic alternatives per group (default: 8; "
        "two whole-group seed options are appended on top)",
    ),
    backup_suffix: str = typer.Option(
        "",
        "--backup-suffix",
        help="If set, copy the existing cache to '{path}{suffix}' before writing",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report what would change without writing the cache",
    ),
):
    """Refresh a stitching review queue against the CURRENT groups sidecar.

    Fixes the "stale proposal" bug class: a queue rebuilt by preserving
    unreviewed entries verbatim keeps a pre-retrain / pre-prune vintage of each
    group's ``optimizer_assignment``, per-edge ``confidence`` and enumerated
    ``alternatives`` while the sidecar has moved on, so the review UI shows and
    reviewers ratify stale proposals.

    This rebuilds every queue entry whose group still exists in the sidecar from
    the authoritative sidecar group (same machinery as ``stitch-batch``:
    alternatives + review tier/score + spatial context), IN PLACE — same group
    ids, same queue order (the queue is never reshaped; the reviewer may be
    mid-review). Entries whose group no longer exists (old grouping) cannot be
    refreshed; they are flagged ``stale_grouping`` so the UI shows a visible
    notice.

    Examples:
        crosswalk data stitch-refresh-queue us_boston_streets
        crosswalk data stitch-refresh-queue --all --backup-suffix .prestalefix.bak
        crosswalk data stitch-refresh-queue us_seattle_sidewalks --dry-run
    """
    import copy
    import json
    from datetime import UTC, datetime

    from ..agent_labeling.panel_routing import attach_panel_route_reasons
    from ..filenames import (
        PROJECT_ROOT,
        bridge_filename,
        groups_sidecar_path,
        stitch_batch_path,
    )
    from ..matching.alternatives import generate_top_k_alternatives
    from ..matching.batch_selection import select_stitching_batch
    from ..matching.stitch_queue_refresh import (
        STALE_GROUPING_KEY,
        check_queue_optimizer_parity,
        plan_queue_refresh,
    )

    if k_alternatives <= 0:
        console.print("[red]Error: --alternatives must be positive[/red]")
        raise typer.Exit(1)

    cache_dir = stitch_batch_path("x").parent
    if all_datasets:
        datasets_to_process = []
        if cache_dir.exists():
            for cache_file in sorted(cache_dir.glob("*_batch.json")):
                datasets_to_process.append(cache_file.stem.replace("_batch", ""))
        if not datasets_to_process:
            console.print("[yellow]No batch caches found in data/cache/stitch/[/yellow]")
            raise typer.Exit(0)
    elif dataset:
        datasets_to_process = [dataset]
    else:
        console.print("[red]Error: Provide a dataset name or --all[/red]")
        raise typer.Exit(1)

    output_dir = PROJECT_ROOT / "data" / "output"

    for ds_name in datasets_to_process:
        console.print(f"\n[bold blue]Refreshing queue for {ds_name}...[/bold blue]")

        batch_path = stitch_batch_path(ds_name)
        if not batch_path.exists():
            console.print(f"  [yellow]No batch cache at {batch_path}[/yellow]")
            continue

        sidecar_path = groups_sidecar_path(output_dir / bridge_filename(ds_name))
        if not sidecar_path.exists():
            console.print(f"  [yellow]No groups sidecar at {sidecar_path}[/yellow]")
            continue

        try:
            cache = json.loads(batch_path.read_text())
            sidecar = json.loads(sidecar_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            console.print(f"  [red]Failed to load cache/sidecar: {e}[/red]")
            continue

        queue = cache.get("groups", [])
        sidecar_by_id = {g.get("group_id"): g for g in sidecar.get("groups", [])}

        refreshable_ids, stale_ids = plan_queue_refresh(queue, sidecar_by_id)
        pre_drift = check_queue_optimizer_parity(queue, sidecar_by_id)
        console.print(
            f"  Queue: {len(queue)} entries — {len(refreshable_ids)} refreshable, "
            f"{len(stale_ids)} stale-grouping (unrefreshable). "
            f"{len(pre_drift)} currently drifted from sidecar."
        )

        # Rebuild refreshable entries from the authoritative sidecar group,
        # preserving queue order. Stale-grouping entries are kept verbatim but
        # flagged so the UI can surface a notice.
        new_queue: list[dict] = []
        to_fill_context: list[dict] = []
        for entry in queue:
            gid = entry.get("group_id")
            sidecar_group = sidecar_by_id.get(gid)
            if sidecar_group is None:
                stale_entry = dict(entry)
                stale_entry[STALE_GROUPING_KEY] = True
                new_queue.append(stale_entry)
                continue
            g = copy.deepcopy(sidecar_group)
            g["alternatives"] = generate_top_k_alternatives(
                component_edges=g.get("edges", []),
                ref_geoms=g.get("ref_geometries", {}),
                target_geoms=g.get("target_geometries", {}),
                k=k_alternatives,
            )
            # Attach review_tier / review_score. select_stitching_batch annotates
            # and returns COPIES (it does not mutate its input), so use the
            # returned entry; k=1 so the single group is always selected. Order
            # is ignored here — the queue order is preserved by new_queue.
            scored = select_stitching_batch(groups=[g], reviewed_group_ids=set(), k=1)
            g = scored[0] if scored else g
            g[STALE_GROUPING_KEY] = False
            new_queue.append(g)
            to_fill_context.append(g)

        # Spatial context (envelope + context_* + n_edges_full/rendered/clipped)
        # for the rebuilt entries. Reuses the exact stitch-batch machinery.
        if to_fill_context:
            console.print(f"  Filling spatial context for {len(to_fill_context)} entries...")
            _fill_spatial_context(to_fill_context, ds_name)

        # Re-attach panel route reasons (rebuilt entries come from the sidecar,
        # which carries no panel fields). Annotation only.
        attach_panel_route_reasons(to_fill_context, ds_name)

        # Post-refresh parity: refreshable entries MUST now match the sidecar.
        post_drift = check_queue_optimizer_parity(new_queue, sidecar_by_id)
        if post_drift:
            console.print(
                f"  [red]ERROR: {len(post_drift)} entries still drifted after refresh: "
                f"{', '.join(d['group_id'] for d in post_drift[:5])}[/red]"
            )
            raise typer.Exit(1)
        console.print(
            f"  [green]Parity OK[/green] — {len(refreshable_ids)} refreshed, "
            f"{len(stale_ids)} flagged stale_grouping"
        )

        if dry_run:
            console.print("  [yellow]--dry-run: not writing[/yellow]")
            continue

        if backup_suffix:
            backup_path = batch_path.with_name(batch_path.name + backup_suffix)
            backup_path.write_text(batch_path.read_text())
            console.print(f"  Backed up existing cache to {backup_path}")

        cache["groups"] = new_queue
        cache["batch_size"] = len(new_queue)
        cache["stale_refreshed_at"] = datetime.now(UTC).isoformat()
        prior_source = cache.get("source", "")
        cache["source"] = (
            "stale-proposal refresh: rebuilt refreshable entries from current "
            "sidecar (alternatives + review + context), flagged old-grouping "
            "entries stale_grouping; queue order + ids preserved"
            + (f" | prior: {prior_source}" if prior_source else "")
        )
        batch_path.write_text(json.dumps(cache, indent=2))
        console.print(f"  [green]Wrote refreshed queue to {batch_path}[/green]")


@data_app.command("stitch-reinterpret-sets")
def stitch_reinterpret_sets(
    dataset: str = typer.Argument(
        None,
        help="Dataset ID to reinterpret (e.g., us_boston_streets). Omit with --all.",
    ),
    all_datasets: bool = typer.Option(
        False, "--all", "-a", help="Reinterpret every dataset with a stitching label CSV"
    ),
    labeler: str = typer.Option(
        "", "--labeler", help="Only convert rows from this labeler (default: all non-panel)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would change without writing the CSV"
    ),
):
    """Reinterpret historical cross-product manual labels as SET-semantics labels.

    Manual / de-anchored stitching submits used to record the full ref×target
    cross-product of the active pills as individual pair assertions, even though
    the reviewer only asserted group MEMBERSHIP. This converts those artifacts to
    ``label_semantics=set`` (membership in ref_ids/target_ids, empty
    selected_edges) using the SAME cross-product signature the review renderer
    flags (``crosswalk.agent_labeling.xprod.is_crossproduct_artifact``): a
    non-panel row whose stored pairs are EXACTLY the ref×target grid within the
    candidate universe (>=2 refs and >=2 targets) AND add pairs beyond the
    optimizer.

    Safe by construction:
      * panel_* rows and rows already ``label_semantics=set`` are never touched
        (idempotent — re-running converts nothing new);
      * explicit option-ratification rows (not a cross-product signature) are
        left as pair labels;
      * a ``.csv.bak`` backup is written before any change.

    Examples:
        crosswalk data stitch-reinterpret-sets us_boston_streets --dry-run
        crosswalk data stitch-reinterpret-sets --all
    """
    import json

    from ..agent_labeling.xprod import parse_selected_edges, reinterpret_row_to_set
    from ..filenames import PROJECT_ROOT, STITCH_CACHE_DIR
    from ..labeling.stitching_store import (
        DEFAULT_STITCHING_DIR,
        LABEL_SEMANTICS_SET,
        StitchingLabelStore,
    )

    def _gid_key(gid: str) -> str:
        return str(gid)[:8]

    def _index_groups(groups: list[dict]) -> dict[str, dict]:
        idx: dict[str, dict] = {}
        for g in groups:
            idx.setdefault(_gid_key(g.get("group_id", "")), g)
        return idx

    def _load_groups(path: Path) -> dict[str, dict]:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            return {}
        groups = data["groups"] if isinstance(data, dict) else data
        return _index_groups(groups or [])

    stitch_root = DEFAULT_STITCHING_DIR
    if all_datasets:
        datasets = sorted(
            p.name.replace("dataset=", "")
            for p in stitch_root.glob("dataset=*")
            if (p / "data.csv").exists()
        )
    elif dataset:
        datasets = [dataset]
    else:
        console.print("[red]Provide a dataset argument or --all[/red]")
        raise typer.Exit(1)

    grand_total = 0
    for ds in datasets:
        store = StitchingLabelStore(ds)
        df = store.df
        if df.empty:
            console.print(f"[yellow]{ds}: no stitching labels[/yellow]")
            continue

        cache_groups = _load_groups(STITCH_CACHE_DIR / f"{ds}_batch.json")
        sidecar_groups = _load_groups(PROJECT_ROOT / "data" / "output" / f"{ds}_groups.json")
        if not cache_groups and not sidecar_groups:
            console.print(
                f"[yellow]{ds}: neither a stitch cache ({STITCH_CACHE_DIR}/{ds}_batch.json) "
                f"nor a groups sidecar (data/output/{ds}_groups.json); "
                "cannot determine candidate universe — skipping[/yellow]"
            )
            continue

        converted: list[tuple[str, int, int, int]] = []
        skipped_no_universe = 0
        for idx, row in df.iterrows():
            lab = str(row.get("labeler") or "")
            if labeler and lab != labeler:
                continue
            key = _gid_key(row["group_id"])
            cache_group = cache_groups.get(key)
            sidecar_group = sidecar_groups.get(key)
            # EITHER source can supply the candidate universe (the sidecar holds
            # the full current grouping; the cache only the review queue).
            # Count a no-universe skip only for rows that were otherwise
            # eligible (non-panel, still pair) so the report is meaningful.
            if (
                cache_group is None
                and sidecar_group is None
                and not lab.startswith("panel")
                and str(row.get("label_semantics") or "pair") != LABEL_SEMANTICS_SET
            ):
                skipped_no_universe += 1
                continue
            decision = reinterpret_row_to_set(row, cache_group, sidecar_group)
            if decision is None:
                continue
            refs, tgts = decision
            npairs = len(parse_selected_edges(row.get("selected_edges")))
            converted.append((str(row["group_id"]), npairs, len(refs), len(tgts)))
            if not dry_run:
                df.at[idx, "label_semantics"] = LABEL_SEMANTICS_SET
                df.at[idx, "selected_edges"] = "[]"
                df.at[idx, "ref_ids"] = json.dumps(refs)
                df.at[idx, "target_ids"] = json.dumps(tgts)
                df.at[idx, "num_refs"] = len(refs)
                df.at[idx, "num_targets"] = len(tgts)

        verb = "Would convert" if dry_run else "Converted"
        console.print(
            f"[bold]{ds}[/bold]: {verb} {len(converted)} cross-product label(s) to set "
            f"({skipped_no_universe} skipped: group in neither cache nor sidecar)"
        )
        for gid, npairs, nrefs, ntgts in converted:
            console.print(f"  {gid[:8]}  {npairs} pairs -> set(refs={nrefs}, targets={ntgts})")
        grand_total += len(converted)

        if converted and not dry_run:
            store._df = df
            store.save()  # writes .csv.bak backup then atomic replace
            console.print(f"  [green]Wrote {ds} (backup at {store.csv_path}.bak)[/green]")

    if dry_run:
        console.print(f"\n[cyan]Dry run: {grand_total} label(s) would be reinterpreted.[/cyan]")
    else:
        console.print(f"\n[green]Reinterpreted {grand_total} label(s) total.[/green]")
