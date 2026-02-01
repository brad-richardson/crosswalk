"""Top-level CLI commands."""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from .utils import console


def register_commands(app: typer.Typer) -> None:
    """Register top-level commands on the given app."""

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
            "xgboost",
            "--method",
            "-m",
            help="Matching method: xgboost",
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
        profile: bool = typer.Option(
            False,
            "--profile",
            help="Enable per-feature timing breakdown (sets MATCHER_PROFILE=1)",
        ),
    ):
        """Run the full matching pipeline."""
        from ..pipeline import run_pipeline

        if profile:
            os.environ["MATCHER_PROFILE"] = "1"

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
        exclude_features: list[str] = typer.Option(
            [],
            "--exclude-features",
            "-e",
            help="Feature(s) to exclude from training (for feature importance analysis). Can be repeated.",
        ),
        agent_weight: float = typer.Option(
            0.0,
            "--agent-weight",
            help="Weight for agent labels (0.0=ignore, 1.0=equal to human). Enables weak supervision.",
        ),
        min_agent_confidence: float = typer.Option(
            0.0,
            "--min-agent-confidence",
            help="Minimum confidence for including agent labels (0.0-1.0).",
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

            # Train with weak supervision from agent labels
            matcher train --agent-weight 0.5 --min-agent-confidence 0.7
        """
        from ..labeling.label_store import LabelStore
        from ..matching.ml import MLMatcher

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

        if exclude_features:
            console.print(f"[yellow]Excluding features: {', '.join(exclude_features)}[/yellow]")

        if agent_weight > 0:
            console.print(
                f"[yellow]Including agent labels with weight={agent_weight}, "
                f"min_confidence={min_agent_confidence}[/yellow]"
            )

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
            exclude_features=list(exclude_features) if exclude_features else None,
            agent_weight=agent_weight,
            min_agent_confidence=min_agent_confidence,
        )

        # Save model
        output.parent.mkdir(parents=True, exist_ok=True)
        matcher.save_model(str(output))

        console.print(f"\n[green]Model saved to {output}[/green]")
        console.print(f"[green]Holdout accuracy: {metrics['test_accuracy']:.1%}[/green]")

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
        # Set environment variables for the Streamlit app
        env = {
            **os.environ,
            "MATCHER_REFERENCE_PATH": str(reference.absolute()),
            "MATCHER_TARGET_PATH": str(target.absolute()),
            "MATCHER_LABELS_PATH": str(labels_path.absolute()),
        }

        # Find the app.py path
        app_path = Path(__file__).parent.parent / "labeling" / "app.py"

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

    @app.command("match-eval")
    def match_eval(
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

        Note: To evaluate ML model quality on training data, use 'matcher ml eval' instead.

        Examples:
            # Basic bridge file stats
            matcher match-eval data/output/us_boston_streets_bridge.parquet

            # With ground truth evaluation
            matcher match-eval data/output/us_boston_streets_bridge.parquet \\
                --ground-truth labels/dataset=us_boston_streets/data.csv
        """
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
    def screen(
        target_path: Path = typer.Argument(
            ...,
            help="Path to target parquet file to screen",
        ),
        bridge_path: Path | None = typer.Option(
            None,
            "--bridge",
            "-b",
            help="Path to bridge parquet file (screens only unmatched targets if provided)",
        ),
        output: Path | None = typer.Option(
            None,
            "--output",
            "-o",
            help="Output path for valid candidates (default: adds _screened suffix)",
        ),
        tests: list[str] | None = typer.Option(
            None,
            "--test",
            "-t",
            help="Specific test(s) to run (default: all). Options: water_body, building, landcover",
        ),
        report_only: bool = typer.Option(
            False,
            "--report-only",
            help="Generate report without outputting candidates",
        ),
        report_output: Path | None = typer.Option(
            None,
            "--report",
            "-r",
            help="Output path for JSON report",
        ),
    ):
        """Screen target segments for valid network additions.

        Validates unmatched target segments using external context (water bodies,
        buildings, etc.) to identify which segments are valid candidates for
        addition to the network.

        If a bridge file is provided, only screens unmatched targets (those not
        in the bridge file). Otherwise screens all targets.

        Examples:
            matcher screen target.parquet
            matcher screen target.parquet -b bridge.parquet
            matcher screen target.parquet -t water_body --report-only
        """
        import json

        from ..screen import run_screen

        # Determine output path
        if output is None and not report_only:
            output = target_path.parent / f"{target_path.stem}_screened.parquet"

        console.print(f"[blue]Running screen tests on {target_path}[/blue]")
        if bridge_path:
            console.print(f"[blue]Filtering to unmatched targets using {bridge_path}[/blue]")

        try:
            valid_gdf, report = run_screen(
                target_path=target_path,
                bridge_path=bridge_path,
                test_names=tests,
                output_path=output,
                report_only=report_only,
            )

            # Print summary
            console.print("\n[bold]Screen Results:[/bold]")
            console.print(f"  Total candidates: {report.total_candidates}")
            console.print(f"  Passed: [green]{report.passed}[/green] ({report.pass_rate:.2%})")
            console.print(f"  Failed: [red]{report.failed}[/red] ({report.fail_rate:.2%})")
            console.print(f"  Warned: [yellow]{report.warned}[/yellow] ({report.warn_rate:.2%})")

            # Per-test breakdown
            if report.test_results:
                console.print("\n[bold]Per-test breakdown:[/bold]")
                for test_name, counts in report.test_results.items():
                    console.print(
                        f"  {test_name}: pass={counts['pass']}, fail={counts['fail']}, "
                        f"warn={counts['warn']}, skip={counts['skip']}"
                    )

            # Save report if requested
            if report_output:
                report_output.parent.mkdir(parents=True, exist_ok=True)
                with open(report_output, "w") as f:
                    json.dump(report.to_dict(), f, indent=2)
                console.print(f"\n[green]Report saved to {report_output}[/green]")

            if output and not report_only:
                console.print(f"\n[green]Valid candidates saved to {output}[/green]")

        except Exception as e:
            console.print(f"[red]Screen tests failed: {e}[/red]")
            raise typer.Exit(1) from None

    @app.command()
    def version():
        """Show version information."""
        from .. import __version__

        console.print(f"matcher version {__version__}")
