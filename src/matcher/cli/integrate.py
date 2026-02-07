"""Integration workflow commands."""

from pathlib import Path

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from .utils import console

# Create integrate group
integrate_app = typer.Typer(
    name="integrate",
    help="Integration workflow commands",
    no_args_is_help=True,
)


@integrate_app.command("run")
def integrate_run(
    reference: Path = typer.Argument(
        ...,
        help="Reference edges (Overture segments parquet)",
    ),
    target: list[str] = typer.Option(
        ...,
        "--target",
        "-t",
        help="Target dataset: name:bridge_path:unmatched_path:priority[:target_path] (can specify multiple). Optional target_path enables partial-match remnant extraction.",
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
        help="Buffer around reference (meters) for net-new calculation (unmatched segments)",
    ),
    matched_net_new_buffer_m: float = typer.Option(
        15.0,
        "--matched-net-new-buffer-m",
        help="Buffer around reference (meters) for net-new calculation (matched segments)",
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
        matcher integrate run data/raw/us_boston_overture_segments.parquet \\
            -t us_boston_streets:data/output/bridge.parquet:data/output/unmatched.parquet:1 \\
            -o data/integrated

        # Multiple datasets with priority
        matcher integrate run data/raw/us_boston_overture_segments.parquet \\
            -t us_boston_streets:data/us_boston_streets/bridge.parquet:data/us_boston_streets/unmatched.parquet:1 \\
            -t us_boston_bike_network:data/us_boston_bike_network/bridge.parquet:data/us_boston_bike_network/unmatched.parquet:2 \\
            -o data/integrated

        # From config file
        matcher integrate run data/raw/us_boston_overture_segments.parquet -c integration_config.yaml -o data/integrated
    """
    from ..integration import TargetConfig, run_integration_from_config, run_integration_pipeline

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
            if len(parts) == 5:
                name, bridge_path, unmatched_path, priority, target_path = parts
            elif len(parts) == 4:
                name, bridge_path, unmatched_path, priority = parts
                target_path = None
            else:
                console.print(
                    "[red]Error: Target must be name:bridge_path:unmatched_path:priority[:target_path][/red]"
                )
                raise typer.Exit(1)

            target_configs.append(
                TargetConfig(
                    name=name,
                    bridge_path=Path(bridge_path),
                    unmatched_path=Path(unmatched_path),
                    priority=int(priority),
                    target_path=Path(target_path) if target_path else None,
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
                matched_net_new_buffer_m=matched_net_new_buffer_m,
                max_hops=max_hops,
                fringe_buffer_m=fringe_buffer_m,
                enable_fringe_screening=not no_fringe_filter,
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
    console.print(f"  Disconnected edges: {stats.disconnected_edges}")
    console.print(f"  Filtered edges: {stats.filtered_edges}")
    console.print()
    console.print(f"[green]Outputs saved to {output_dir}[/green]")
