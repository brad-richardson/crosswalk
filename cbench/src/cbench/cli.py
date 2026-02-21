"""cbench CLI - conflation benchmarking harness."""

from __future__ import annotations

import tomllib
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from cbench.util import setup_logging

app = typer.Typer(
    name="cbench",
    help="Conflation benchmarking harness - compare road matching tools against ground truth.",
    no_args_is_help=True,
)
console = Console()


def _parse_opts(opt: list[str]) -> dict[str, str]:
    """Parse --opt key=value pairs into a dict.

    Raises typer.Exit(1) on invalid format.
    """
    kwargs: dict[str, str] = {}
    for o in opt:
        if "=" not in o:
            console.print(f"[red]Invalid option format: {o} (expected key=value)[/red]")
            raise typer.Exit(1)
        k, v = o.split("=", 1)
        kwargs[k] = v
    return kwargs


def _coerce_path_opts(kwargs: dict) -> dict:
    """Coerce known path-like option values to Path objects."""
    for key in ("hoot_dir", "connectors"):
        if key in kwargs:
            kwargs[key] = Path(kwargs[key])
    return kwargs


def _get_adapter(tool: str):
    """Look up a tool adapter by name, or exit with error."""
    from cbench.adapters import REGISTRY

    if tool not in REGISTRY:
        console.print(f"[red]Unknown tool: {tool}[/red]")
        console.print(f"Available: {', '.join(REGISTRY.keys())}")
        raise typer.Exit(1)
    return REGISTRY[tool]()


def _print_eval_result(tool: str, dataset: str, result) -> None:
    """Print evaluation results for a single run."""
    er = result.eval_result
    console.print(f"[bold]Results: {tool} on {dataset}[/bold]")
    console.print(f"  Precision: {er.precision:.4f}")
    console.print(f"  Recall:    {er.recall:.4f}")
    console.print(f"  F1:        [bold]{er.f1:.4f}[/bold]")
    console.print(
        f"  TP={er.true_positives}  FP={er.false_positives}  "
        f"FN={er.false_negatives}  Unlabeled={er.unlabeled_predictions}"
    )
    if result.resource_stats:
        rs = result.resource_stats
        console.print(
            f"  Time: {rs.wall_time_s:.1f}s  "
            f"CPU: {rs.cpu_user_s:.1f}u+{rs.cpu_system_s:.1f}s  "
            f"Peak RSS: {rs.peak_rss_mb:.0f} MB"
        )
    if result.stitch_result is not None:
        sr = result.stitch_result
        console.print(f"  [bold]Stitch-level ({sr.groups_evaluated} groups):[/bold]")
        console.print(f"    Precision: {sr.precision:.4f}")
        console.print(f"    Recall:    {sr.recall:.4f}")
        console.print(f"    F1:        [bold]{sr.f1:.4f}[/bold]")
        console.print(
            f"    Curated edges: {sr.total_curated_edges}  "
            f"Extra: {sr.total_extra_edges}"
        )


def load_datasets_config(config_path: Path) -> dict:
    """Load and validate datasets TOML config.

    Returns:
        Dict with 'defaults' and 'datasets' keys.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "rb") as f:
        try:
            config = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"Invalid TOML syntax in {config_path}: {exc}") from exc
    if "datasets" not in config:
        raise ValueError(f"Config file missing [datasets] section: {config_path}")
    for name, ds in config["datasets"].items():
        for key in ("reference", "target"):
            if key not in ds:
                raise ValueError(f"Dataset '{name}' missing required key '{key}' in {config_path}")
    return config


@app.command()
def run(
    tool: str = typer.Argument(help="Tool adapter name (e.g., 'matcher', 'hootenanny')"),
    dataset: str = typer.Argument(help="Dataset name (must have labels)"),
    labels: Path = typer.Option(..., "--labels", "-l", help="Path to labels directory"),
    reference: Path = typer.Option(..., "--reference", "-r", help="Reference parquet file"),
    target: Path = typer.Option(..., "--target", "-t", help="Target parquet file"),
    output_dir: Path = typer.Option(
        Path("cbench_output"), "--output-dir", "-o", help="Output directory"
    ),
    results_file: Path = typer.Option(
        Path("cbench_results.jsonl"), "--results", help="JSONL results file"
    ),
    stitch_labels: Path | None = typer.Option(
        None, "--stitch-labels", help="Path to stitching labels directory"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    opt: list[str] = typer.Option([], "--opt", help="Tool option as key=value"),
) -> None:
    """Run a tool on a dataset and evaluate against ground truth."""
    setup_logging(verbose)

    from cbench.runner import run_single

    adapter = _get_adapter(tool)
    kwargs = _coerce_path_opts(_parse_opts(opt))

    try:
        result = run_single(
            adapter=adapter,
            dataset=dataset,
            reference=reference,
            target=target,
            labels_dir=labels,
            output_dir=output_dir,
            results_file=results_file,
            stitch_labels_dir=stitch_labels,
            **kwargs,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print()
    _print_eval_result(tool, dataset, result)
    console.print(f"\nResult saved to {results_file}")


@app.command("run-batch")
def run_batch(
    tool: str = typer.Argument(help="Tool adapter name (e.g., 'matcher', 'hootenanny')"),
    config: Path = typer.Option(
        Path("datasets.toml"), "--config", "-c", help="Datasets TOML config file"
    ),
    data_dir: Path | None = typer.Option(
        None, "--data-dir", "-d", help="Override data directory from config"
    ),
    labels_dir: Path | None = typer.Option(
        None, "--labels-dir", "-l", help="Override labels directory from config"
    ),
    dataset: list[str] | None = typer.Option(
        None, "--dataset", help="Run only these datasets (repeatable)"
    ),
    output_dir: Path = typer.Option(
        Path("cbench_output"), "--output-dir", "-o", help="Output directory"
    ),
    results_file: Path = typer.Option(
        Path("cbench_results.jsonl"), "--results", help="JSONL results file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    opt: list[str] = typer.Option([], "--opt", help="Tool option as key=value"),
) -> None:
    """Run a tool on multiple datasets from a TOML config."""
    setup_logging(verbose)

    from cbench.runner import run_single

    adapter = _get_adapter(tool)
    extra_kwargs = _coerce_path_opts(_parse_opts(opt))

    try:
        cfg = load_datasets_config(config)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    defaults = cfg.get("defaults", {})
    resolved_data_dir = data_dir or Path(defaults.get("data_dir", "."))
    resolved_labels_dir = labels_dir or Path(defaults.get("labels_dir", "."))
    resolved_stitch_dir = (
        Path(defaults["stitch_labels_dir"]) if "stitch_labels_dir" in defaults else None
    )

    datasets = cfg["datasets"]
    if dataset:
        unknown = set(dataset) - set(datasets)
        if unknown:
            console.print(f"[red]Unknown datasets: {', '.join(sorted(unknown))}[/red]")
            console.print(f"Available: {', '.join(datasets.keys())}")
            raise typer.Exit(1)
        datasets = {k: v for k, v in datasets.items() if k in dataset}

    from cbench.runner import RunResult

    batch_results: list[tuple[str, RunResult | None, str | None]] = []

    for ds_name, ds_cfg in datasets.items():
        console.rule(f"[bold]{ds_name}[/bold]")

        reference = resolved_data_dir / ds_cfg["reference"]
        target = resolved_data_dir / ds_cfg["target"]

        # Auto-wire connectors from config into tool kwargs
        run_kwargs = dict(extra_kwargs)
        if "connectors" in ds_cfg and "connectors" not in run_kwargs:
            run_kwargs["connectors"] = resolved_data_dir / ds_cfg["connectors"]

        try:
            result = run_single(
                adapter=adapter,
                dataset=ds_name,
                reference=reference,
                target=target,
                labels_dir=resolved_labels_dir,
                output_dir=output_dir,
                results_file=results_file,
                stitch_labels_dir=resolved_stitch_dir,
                **run_kwargs,
            )
            f1 = result.eval_result.f1
            rs = result.resource_stats
            time_str = f"  ({rs.wall_time_s:.1f}s)" if rs else ""
            console.print(f"  F1: [bold]{f1:.4f}[/bold]{time_str}")
            batch_results.append((ds_name, result, None))
        except Exception as exc:
            console.print(f"  [red]FAILED: {exc}[/red]")
            batch_results.append((ds_name, None, str(exc)))

    # Summary table
    console.print()
    table = Table(title=f"Batch Results: {tool}")
    table.add_column("Dataset", style="cyan")
    table.add_column("F1", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Peak RSS", justify="right")
    table.add_column("Status", justify="center")

    passed = 0
    for ds_name, res, _error in batch_results:
        if res is not None:
            rs = res.resource_stats
            time_str = f"{rs.wall_time_s:.1f}s" if rs else "-"
            rss_str = f"{rs.peak_rss_mb:.0f} MB" if rs else "-"
            table.add_row(
                ds_name, f"{res.eval_result.f1:.4f}", time_str, rss_str, "[green]OK[/green]"
            )
            passed += 1
        else:
            table.add_row(ds_name, "-", "-", "-", "[red]FAIL[/red]")

    console.print(table)
    console.print(f"\n{passed}/{len(batch_results)} datasets completed successfully.")


@app.command("list-datasets")
def list_datasets(
    config: Path = typer.Option(
        Path("datasets.toml"), "--config", "-c", help="Datasets TOML config file"
    ),
    labels_dir: Path | None = typer.Option(
        None, "--labels-dir", "-l", help="Labels directory to check availability"
    ),
) -> None:
    """List datasets from a TOML config and check label availability."""
    try:
        cfg = load_datasets_config(config)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    defaults = cfg.get("defaults", {})
    resolved_labels_dir = labels_dir or Path(defaults.get("labels_dir", "."))

    table = Table(title="Configured Datasets")
    table.add_column("Dataset", style="cyan")
    table.add_column("Reference")
    table.add_column("Target")
    table.add_column("Connectors")
    table.add_column("Labels", justify="center")

    for name, ds_cfg in cfg["datasets"].items():
        labels_path = resolved_labels_dir / f"dataset={name}" / "data.csv"
        has_labels = "[green]yes[/green]" if labels_path.exists() else "[red]no[/red]"
        table.add_row(
            name,
            ds_cfg.get("reference", "-"),
            ds_cfg.get("target", "-"),
            ds_cfg.get("connectors", "-"),
            has_labels,
        )

    console.print(table)


@app.command()
def compare(
    results_file: Path = typer.Argument(..., help="JSONL results file to compare"),
    dataset: str | None = typer.Option(None, "--dataset", "-d", help="Filter by dataset"),
    tool: str | None = typer.Option(None, "--tool", "-t", help="Filter by tool"),
) -> None:
    """Compare benchmark results from a JSONL results file."""
    from cbench.results.store import compare_results, load_results

    if not results_file.exists():
        console.print(f"[red]Results file not found: {results_file}[/red]")
        raise typer.Exit(1)

    results = load_results(results_file)
    if dataset:
        results = [r for r in results if r.dataset == dataset]
    if tool:
        results = [r for r in results if r.tool == tool]

    if not results:
        console.print("[yellow]No matching results found.[/yellow]")
        raise typer.Exit(0)

    compare_results(results, console=console)


@app.command("list-tools")
def list_tools() -> None:
    """List available tool adapters."""
    from cbench.adapters import REGISTRY

    console.print("[bold]Available tool adapters:[/bold]")
    for name, cls in REGISTRY.items():
        adapter = cls()
        console.print(f"  [cyan]{name}[/cyan] - eval_mode={adapter.eval_mode.value}")
