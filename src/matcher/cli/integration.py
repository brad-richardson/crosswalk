"""Integration commands: integrate, qa-integration."""

from pathlib import Path

import typer

from ._app import app, console


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
    app_path = Path(__file__).parent.parent / "integration_qa" / "app.py"

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
