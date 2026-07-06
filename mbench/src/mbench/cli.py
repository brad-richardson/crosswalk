"""mbench CLI - conflation benchmarking harness."""

from __future__ import annotations

import tomllib
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from mbench.util import setup_logging

app = typer.Typer(
    name="mbench",
    help="Conflation benchmarking harness - compare road matching tools against ground truth.",
    no_args_is_help=True,
)
console = Console()


def cbench_deprecated() -> None:
    """Deprecated ``cbench`` console-script alias: warn, then forward to ``mbench``.

    The harness was renamed ``cbench`` -> ``mbench`` (2026-07-05). This shim keeps
    the old entry point working while emitting a deprecation warning to stderr;
    it will be removed in a future release.
    """
    import sys

    print(
        "warning: 'cbench' has been renamed to 'mbench'. The 'cbench' alias is "
        "deprecated and will be removed in a future release; please use 'mbench'.",
        file=sys.stderr,
    )
    app()


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


def _validate_match_level(match_level: str) -> str:
    """Validate --match-level value, or exit with error."""
    from mbench.eval.metrics import MATCH_LEVELS

    if match_level not in MATCH_LEVELS:
        console.print(
            f"[red]Invalid --match-level: {match_level} "
            f"(expected one of: {', '.join(MATCH_LEVELS)})[/red]"
        )
        raise typer.Exit(1)
    return match_level


def _get_adapter(tool: str):
    """Look up a tool adapter by name, or exit with error.

    Deprecated engine-id aliases (e.g. ``matcher`` -> ``crosswalk``) are
    accepted with a stderr warning and forwarded to the canonical adapter.
    """
    import sys

    from mbench.adapters import DEPRECATED_ALIASES, REGISTRY

    if tool in DEPRECATED_ALIASES:
        canonical = DEPRECATED_ALIASES[tool]
        print(
            f"warning: engine '{tool}' has been renamed to '{canonical}'. The "
            f"'{tool}' alias is deprecated and will be removed in a future "
            f"release; please use '{canonical}'.",
            file=sys.stderr,
        )
        tool = canonical

    if tool not in REGISTRY:
        console.print(f"[red]Unknown tool: {tool}[/red]")
        console.print(f"Available: {', '.join(REGISTRY.keys())}")
        raise typer.Exit(1)
    return REGISTRY[tool]()


def _print_eval_result(tool: str, dataset: str, result) -> None:
    """Print evaluation results for a single run."""
    er = result.eval_result
    console.print(f"[bold]Results: {tool} on {dataset}[/bold] (match_level={er.match_level})")
    console.print(f"  Precision: {er.precision:.4f}")
    console.print(f"  Recall:    {er.recall:.4f}")
    console.print(f"  F1:        [bold]{er.f1:.4f}[/bold]")
    console.print(
        f"  TP={er.true_positives}  FP={er.false_positives}  "
        f"FN={er.false_negatives}  Unlabeled={er.unlabeled_predictions}  "
        f"Unsure skipped={er.skipped_unsure}"
    )
    console.print(
        f"  Labeled coverage: {er.labeled_coverage:.4f} "
        f"(predictions on unlabeled pairs are excluded from precision)"
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
        console.print(
            f"    Raw:      P={sr.precision:.4f} R={sr.recall:.4f} "
            f"F1=[bold]{sr.f1:.4f}[/bold] exact={sr.exact_match_rate:.4f}"
        )
        console.print(
            f"    Filtered: P={sr.precision_filtered:.4f} R={sr.recall_filtered:.4f} "
            f"F1=[bold]{sr.f1_filtered:.4f}[/bold] exact={sr.exact_match_rate_filtered:.4f} "
            f"({sr.groups_sliver_affected} sliver-affected)"
        )
        console.print(f"    Curated edges: {sr.total_curated_edges}  Extra: {sr.total_extra_edges}")
        if sr.label_counts_by_labeler:
            counts = "  ".join(f"{k}={v}" for k, v in sr.label_counts_by_labeler.items())
            console.print(f"    Labels by labeler: {counts}")
        for cls, m in sr.metrics_by_labeler.items():
            console.print(
                f"      [{cls}] n={m['n']} F1={m['f1']:.4f} exact={m['exact_match_rate']:.4f}"
            )


def _apply_stitch_gate(
    config: Path,
    outcomes_input: list[tuple[str, object]],
) -> bool:
    """Evaluate the stitch-level quality gate for the given runs.

    ``outcomes_input`` is a list of ``(dataset, stitch_result_or_None)``. Loads
    per-dataset floors from ``[gate.*]`` in the config, prints a status line per
    dataset, and returns True iff any ARMED dataset failed its floor (blocking).

    Never raises on config problems: a missing/malformed config just means no
    floors, so every dataset reports ``no_config`` (non-blocking).
    """
    from mbench.eval.gate import evaluate_gate, load_gate_config

    try:
        cfg = load_datasets_config(config)
        floors = load_gate_config(cfg)
    except (FileNotFoundError, ValueError):
        floors = {}

    console.print("\n[bold]Stitch-level quality gate[/bold]")
    if not floors:
        console.print("  [yellow]No [gate.*] floors configured; nothing to enforce.[/yellow]")
        return False

    colors = {"pass": "green", "fail": "red", "skip_unarmed": "yellow", "no_config": "dim"}
    any_blocking = False
    for dataset, stitch_result in outcomes_input:
        outcome = evaluate_gate(dataset, stitch_result, floors.get(dataset))
        color = colors.get(outcome.status, "white")
        console.print(
            f"  [{color}]{outcome.status.upper():13}[/{color}] {dataset}: {outcome.message}"
        )
        if outcome.blocking:
            any_blocking = True

    if any_blocking:
        console.print("  [bold red]GATE FAILED[/bold red] — stitch quality regressed below floor.")
    else:
        console.print("  [green]Gate OK[/green]")
    return any_blocking


def _resolve_config_default(config_path: Path, value: str) -> Path:
    """Resolve a config *default* path relative to the config file's directory.

    Default paths in ``datasets.toml`` (e.g. ``../labels/human``, ``../data/raw``)
    are written relative to the config file's location, not the process CWD.
    Resolving them against CWD is the root cause of "labels not found" when
    ``mbench`` is invoked from anywhere other than the config's directory (e.g.
    the repo root). Absolute values are returned unchanged.
    """
    p = Path(value)
    if p.is_absolute():
        return p
    return (config_path.parent / p).resolve()


def _resolve_single_run_paths(
    config: Path,
    dataset: str,
    reference: Path | None,
    target: Path | None,
    labels: Path | None,
    stitch_labels: Path | None,
) -> tuple[Path, Path, Path, Path | None, Path | None]:
    """Fill unspecified single-run paths from ``datasets.toml``.

    Explicit CLI paths are honored as-is (CWD-relative). Any path left as ``None``
    is resolved from the config: reference/target/connectors from the dataset
    entry (relative to the config's ``data_dir``), and labels/stitch from the
    config defaults. All config-derived paths anchor to the config file's
    directory so resolution is independent of the current working directory.

    Returns ``(reference, target, labels, stitch_labels, connectors)``.

    Raises:
        FileNotFoundError: config needed but missing.
        ValueError: dataset not present in config.
    """
    connectors: Path | None = None
    # Stitch labels are optional (runner falls back to the labels-dir sibling
    # `stitching/`), so a missing stitch path must NOT force loading the config:
    # `mbench run ... -r ... -t ... -l ...` should work with no datasets.toml.
    need_config = reference is None or target is None or labels is None
    if not need_config:
        return reference, target, labels, stitch_labels, connectors

    cfg = load_datasets_config(config)  # raises FileNotFoundError/ValueError
    defaults = cfg.get("defaults", {})
    datasets = cfg["datasets"]
    if dataset not in datasets:
        raise ValueError(
            f"Dataset '{dataset}' not found in {config}. Available: {', '.join(datasets.keys())}"
        )
    ds_cfg = datasets[dataset]

    data_dir = _resolve_config_default(config, defaults.get("data_dir", "."))
    if reference is None:
        reference = data_dir / ds_cfg["reference"]
    if target is None:
        target = data_dir / ds_cfg["target"]
    if labels is None:
        labels = _resolve_config_default(config, defaults.get("labels_dir", "."))
    if stitch_labels is None and "stitch_labels_dir" in defaults:
        stitch_labels = _resolve_config_default(config, defaults["stitch_labels_dir"])
    if "connectors" in ds_cfg:
        connectors = data_dir / ds_cfg["connectors"]

    return reference, target, labels, stitch_labels, connectors


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
    tool: str = typer.Argument(help="Tool adapter name (e.g., 'crosswalk', 'hootenanny')"),
    dataset: str = typer.Argument(help="Dataset name (must have labels)"),
    labels: Path | None = typer.Option(
        None, "--labels", "-l", help="Path to labels directory (default: from config)"
    ),
    reference: Path | None = typer.Option(
        None, "--reference", "-r", help="Reference parquet file (default: from config)"
    ),
    target: Path | None = typer.Option(
        None, "--target", "-t", help="Target parquet file (default: from config)"
    ),
    config: Path = typer.Option(
        Path("datasets.toml"),
        "--config",
        "-c",
        help="Datasets TOML config used to resolve unspecified paths",
    ),
    output_dir: Path = typer.Option(
        Path("mbench_output"), "--output-dir", "-o", help="Output directory"
    ),
    results_file: Path = typer.Option(
        Path("mbench_results.jsonl"), "--results", help="JSONL results file"
    ),
    stitch_labels: Path | None = typer.Option(
        None, "--stitch-labels", help="Path to stitching labels directory (default: from config)"
    ),
    match_level: str = typer.Option(
        "target", "--match-level", help="Evaluation level: 'target' (default) or 'pair'"
    ),
    gate: bool = typer.Option(
        False,
        "--gate",
        help="Enforce the stitch-level quality gate: exit nonzero if an armed "
        "dataset's sliver-filtered edge-F1/exact-match falls below its "
        "[gate.*] floor in the config.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    opt: list[str] = typer.Option([], "--opt", help="Tool option as key=value"),
) -> None:
    """Run a tool on a dataset and evaluate against ground truth.

    Reference/target/labels default to the entries in ``datasets.toml`` (resolved
    relative to the config file's location), so ``mbench run crosswalk <dataset>``
    works out of the box from a repo checkout. Explicit ``-r/-t/-l`` override.
    """
    setup_logging(verbose)

    from mbench.runner import run_single

    adapter = _get_adapter(tool)
    match_level = _validate_match_level(match_level)
    kwargs = _coerce_path_opts(_parse_opts(opt))

    try:
        reference, target, labels, stitch_labels, connectors = _resolve_single_run_paths(
            config=config,
            dataset=dataset,
            reference=reference,
            target=target,
            labels=labels,
            stitch_labels=stitch_labels,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if connectors is not None and "connectors" not in kwargs:
        kwargs["connectors"] = connectors

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
            match_level=match_level,
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

    if gate:
        blocking = _apply_stitch_gate(config, [(dataset, result.stitch_result)])
        if blocking:
            raise typer.Exit(1)


@app.command("run-batch")
def run_batch(
    tool: str = typer.Argument(help="Tool adapter name (e.g., 'crosswalk', 'hootenanny')"),
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
        Path("mbench_output"), "--output-dir", "-o", help="Output directory"
    ),
    results_file: Path = typer.Option(
        Path("mbench_results.jsonl"), "--results", help="JSONL results file"
    ),
    match_level: str = typer.Option(
        "target", "--match-level", help="Evaluation level: 'target' (default) or 'pair'"
    ),
    gate: bool = typer.Option(
        False,
        "--gate",
        help="Enforce the stitch-level quality gate: exit nonzero if any armed "
        "dataset's sliver-filtered edge-F1/exact-match falls below its "
        "[gate.*] floor in the config.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    opt: list[str] = typer.Option([], "--opt", help="Tool option as key=value"),
) -> None:
    """Run a tool on multiple datasets from a TOML config."""
    setup_logging(verbose)

    from mbench.runner import run_single

    adapter = _get_adapter(tool)
    match_level = _validate_match_level(match_level)
    extra_kwargs = _coerce_path_opts(_parse_opts(opt))

    try:
        cfg = load_datasets_config(config)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    defaults = cfg.get("defaults", {})
    # Explicit CLI overrides are honored as-is (CWD-relative, user intent);
    # config defaults resolve relative to the config file's directory so that
    # `mbench run-batch` works from any CWD (e.g. the repo root).
    resolved_data_dir = data_dir or _resolve_config_default(config, defaults.get("data_dir", "."))
    resolved_labels_dir = labels_dir or _resolve_config_default(
        config, defaults.get("labels_dir", ".")
    )
    resolved_stitch_dir = (
        _resolve_config_default(config, defaults["stitch_labels_dir"])
        if "stitch_labels_dir" in defaults
        else None
    )

    datasets = cfg["datasets"]
    if dataset:
        unknown = set(dataset) - set(datasets)
        if unknown:
            console.print(f"[red]Unknown datasets: {', '.join(sorted(unknown))}[/red]")
            console.print(f"Available: {', '.join(datasets.keys())}")
            raise typer.Exit(1)
        datasets = {k: v for k, v in datasets.items() if k in dataset}

    from mbench.runner import RunResult

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
                match_level=match_level,
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

    if gate:
        gate_inputs = [
            (ds_name, res.stitch_result if res is not None else None)
            for ds_name, res, _err in batch_results
        ]
        blocking = _apply_stitch_gate(config, gate_inputs)
        if blocking:
            raise typer.Exit(1)


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
    resolved_labels_dir = labels_dir or _resolve_config_default(
        config, defaults.get("labels_dir", ".")
    )

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
    from mbench.results.store import compare_results, load_results

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
    from mbench.adapters import REGISTRY

    console.print("[bold]Available tool adapters:[/bold]")
    for name, cls in REGISTRY.items():
        adapter = cls()
        console.print(f"  [cyan]{name}[/cyan] - eval_mode={adapter.eval_mode.value}")
