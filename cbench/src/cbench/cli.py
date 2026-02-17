"""cbench CLI - conflation benchmarking harness."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from cbench.util import setup_logging

app = typer.Typer(
    name="cbench",
    help="Conflation benchmarking harness - compare road matching tools against ground truth.",
    no_args_is_help=True,
)
console = Console()


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
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    # Tool-specific options passed as key=value
    opt: list[str] = typer.Option([], "--opt", help="Tool option as key=value"),
) -> None:
    """Run a tool on a dataset and evaluate against ground truth."""
    setup_logging(verbose)

    from cbench.adapters import REGISTRY
    from cbench.eval.labels import load_labels
    from cbench.eval.metrics import evaluate
    from cbench.results.store import create_result, save_result

    # Validate input files exist
    for path, desc in [
        (reference, "Reference file"),
        (target, "Target file"),
        (labels, "Labels dir"),
    ]:
        if not path.exists():
            console.print(f"[red]{desc} not found: {path}[/red]")
            raise typer.Exit(1)

    if tool not in REGISTRY:
        console.print(f"[red]Unknown tool: {tool}[/red]")
        console.print(f"Available: {', '.join(REGISTRY.keys())}")
        raise typer.Exit(1)

    # Parse tool options
    kwargs = {}
    for o in opt:
        if "=" not in o:
            console.print(f"[red]Invalid option format: {o} (expected key=value)[/red]")
            raise typer.Exit(1)
        k, v = o.split("=", 1)
        kwargs[k] = v

    # Coerce path-like options
    for key in ("hoot_dir", "connectors"):
        if key in kwargs:
            kwargs[key] = Path(kwargs[key])

    adapter = REGISTRY[tool]()
    tool_output_dir = output_dir / tool / dataset

    # Run tool
    console.print(f"[bold]Running {tool}[/bold] on {dataset}...")
    output_path = adapter.run(reference, target, tool_output_dir, **kwargs)

    # Parse output
    console.print("Parsing output...")
    tool_output = adapter.parse_output(output_path)
    console.print(f"Found {len(tool_output.matches)} match predictions")

    # Load labels and evaluate
    console.print("Evaluating against ground truth...")
    ground_truth = load_labels(labels, dataset)
    result = evaluate(tool_output.matches, ground_truth)

    # Display results
    console.print()
    console.print(f"[bold]Results: {tool} on {dataset}[/bold]")
    console.print(f"  Precision: {result.precision:.4f}")
    console.print(f"  Recall:    {result.recall:.4f}")
    console.print(f"  F1:        [bold]{result.f1:.4f}[/bold]")
    console.print(
        f"  TP={result.true_positives}  FP={result.false_positives}  "
        f"FN={result.false_negatives}  Unlabeled={result.unlabeled_predictions}"
    )

    # Save result
    bench_result = create_result(
        tool=tool,
        dataset=dataset,
        metrics=result.to_dict(),
        metadata=tool_output.metadata,
    )
    save_result(bench_result, results_file)
    console.print(f"\nResult saved to {results_file}")


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
