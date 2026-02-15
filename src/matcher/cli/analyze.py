"""Analysis and diagnostic commands."""

import json
from pathlib import Path

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from .utils import console

# Create analyze group
analyze_app = typer.Typer(
    name="analyze",
    help="Analysis and diagnostic commands",
    no_args_is_help=True,
)


@analyze_app.command("bridge")
def analyze_bridge(
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

    Note: To evaluate ML model quality on training data, use 'matcher eval' instead.

    Examples:
        # Basic bridge file stats
        matcher analyze bridge data/output/us_boston_streets_bridge.parquet

        # With ground truth evaluation
        matcher analyze bridge data/output/us_boston_streets_bridge.parquet \\
            --ground-truth labels/dataset=us_boston_streets/data.csv
    """
    import pandas as pd

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


@analyze_app.command("screen")
def analyze_screen(
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
        matcher analyze screen target.parquet
        matcher analyze screen target.parquet -b bridge.parquet
        matcher analyze screen target.parquet -t water_body --report-only
    """
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


@analyze_app.command("errors")
def analyze_errors(
    model: Path = typer.Option(
        Path("data/models/matcher_model_combined.joblib"),
        "--model",
        "-m",
        help="Path to trained model",
    ),
    labels_dir: Path = typer.Option(
        Path("labels"),
        "--labels",
        "-l",
        help="Labels directory (Hive-partitioned CSV format)",
    ),
    dataset: list[str] = typer.Option(
        [],
        "--dataset",
        "-d",
        help="Focus on specific dataset(s). Can be repeated. Default: all datasets.",
    ),
    top_n: int = typer.Option(
        20,
        "--top",
        "-n",
        help="Number of worst errors to show per dataset",
    ),
    output_dir: Path = typer.Option(
        Path("analysis"),
        "--output",
        "-o",
        help="Output directory for error analysis reports",
    ),
    min_confidence: float = typer.Option(
        0.7,
        "--min-confidence",
        "-c",
        help="Minimum model confidence to consider 'high confidence' errors",
    ),
):
    """Analyze prediction errors to diagnose model performance issues.

    This command identifies error patterns in model predictions on labeled data:

    1. **Confusion Matrix**: False positives vs false negatives breakdown
    2. **High-Confidence Errors**: Wrong predictions with model confidence >= threshold
    3. **Feature Analysis**: Which features correlate with errors
    4. **Error Export**: CSV of worst errors for review in labeling UI

    Use this to diagnose why certain datasets underperform and whether issues
    are model failures or potential label quality problems.

    Examples:
        # Analyze errors on all labeled datasets
        matcher analyze errors

        # Focus on underperforming datasets
        matcher analyze errors -d br_sao_paulo_roads -d us_fort_collins_streets

        # Show top 50 worst errors
        matcher analyze errors --top 50

        # Use custom confidence threshold for "high confidence" errors
        matcher analyze errors --min-confidence 0.8
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    from ..config import METRIC_AVERAGE
    from ..labeling.label_store import LabelStore
    from ..matching.ml import MLMatcher

    if not model.exists():
        console.print(f"[red]Model not found: {model}[/red]")
        raise typer.Exit(1)

    if not labels_dir.exists():
        console.print(f"[red]Labels directory not found: {labels_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Loading model: {model.name}[/blue]")
    matcher = MLMatcher()
    matcher.load_model(str(model))

    console.print("[blue]Loading labels...[/blue]")
    all_labels = LabelStore.load_all(labels_dir)
    console.print(f"  Total labels: {len(all_labels)}")

    # Filter to valid labels only
    valid_labels = {"match", "no_match"}
    all_labels = all_labels[all_labels["label"].isin(valid_labels)].copy()
    console.print(f"  Valid labels (match/no_match): {len(all_labels)}")

    # Filter to specific datasets if requested
    if dataset:
        all_labels = all_labels[all_labels["dataset"].isin(dataset)].copy()
        console.print(f"  Filtered to datasets {dataset}: {len(all_labels)}")

    if len(all_labels) == 0:
        console.print("[yellow]No labels to analyze[/yellow]")
        raise typer.Exit(0)

    # Extract features and get predictions
    console.print("[blue]Computing predictions...[/blue]")
    X, y_true = matcher._extract_features_and_labels(all_labels, binary=True)
    X = matcher._cap_infinities(X)

    y_pred = matcher.model.predict(X)
    y_prob = matcher.model.predict_proba(X)

    # Get confidence scores (probability of predicted class)
    # For binary classification: column 0 = no_match, column 1 = match
    match_probs = y_prob[:, 1]  # Probability of being a match
    pred_confidence = np.where(y_pred == 1, match_probs, 1 - match_probs)

    # Add predictions to dataframe for analysis
    all_labels = all_labels.copy()
    all_labels["y_true"] = y_true
    all_labels["y_pred"] = y_pred
    all_labels["match_prob"] = match_probs
    all_labels["pred_confidence"] = pred_confidence
    all_labels["is_error"] = y_true != y_pred
    all_labels["is_fp"] = (y_pred == 1) & (y_true == 0)  # Predicted match, actually no_match
    all_labels["is_fn"] = (y_pred == 0) & (y_true == 1)  # Predicted no_match, actually match

    # Overall metrics
    overall_acc = accuracy_score(y_true, y_pred)
    overall_f1 = f1_score(y_true, y_pred, average=METRIC_AVERAGE, zero_division=0)
    overall_cm = confusion_matrix(y_true, y_pred)

    console.print(f"\n{'=' * 70}")
    console.print("[bold]ERROR ANALYSIS SUMMARY[/bold]")
    console.print("=" * 70)

    console.print("\n[bold]Overall Performance[/bold]")
    console.print(f"  Accuracy: {overall_acc:.3f}")
    console.print(f"  F1 Score: {overall_f1:.3f}")
    console.print(f"  Total errors: {all_labels['is_error'].sum()} / {len(all_labels)}")

    console.print("\n[bold]Confusion Matrix[/bold]")
    console.print("                  Predicted")
    console.print("                  no_match  match")
    console.print(
        f"  Actual no_match   {overall_cm[0, 0]:5d}    {overall_cm[0, 1]:5d}  (FP rate: {overall_cm[0, 1] / overall_cm[0].sum():.1%})"
    )
    console.print(
        f"  Actual match      {overall_cm[1, 0]:5d}    {overall_cm[1, 1]:5d}  (FN rate: {overall_cm[1, 0] / overall_cm[1].sum():.1%})"
    )

    n_fp = all_labels["is_fp"].sum()
    n_fn = all_labels["is_fn"].sum()
    console.print(f"\n  False Positives (predicted match, was no_match): {n_fp}")
    console.print(f"  False Negatives (predicted no_match, was match): {n_fn}")

    # High-confidence errors (these are the most diagnostic)
    high_conf_errors = all_labels[
        (all_labels["is_error"]) & (all_labels["pred_confidence"] >= min_confidence)
    ]
    console.print(f"\n[bold]High-Confidence Errors (confidence >= {min_confidence})[/bold]")
    console.print(f"  Total: {len(high_conf_errors)} / {all_labels['is_error'].sum()} errors")

    if len(high_conf_errors) > 0:
        hc_fp = high_conf_errors["is_fp"].sum()
        hc_fn = high_conf_errors["is_fn"].sum()
        console.print(
            f"  High-conf FP: {hc_fp} (model confident it's a match, but labeled no_match)"
        )
        console.print(f"  High-conf FN: {hc_fn} (model confident it's no_match, but labeled match)")
        console.print("\n  [yellow]High-confidence errors suggest either:[/yellow]")
        console.print("    - Systematic model failure on certain patterns")
        console.print("    - Potential label quality issues (mislabeled examples)")

    # Per-dataset breakdown
    console.print("\n[bold]Per-Dataset Error Analysis[/bold]")
    console.print("-" * 70)

    dataset_errors = []
    for ds in sorted(all_labels["dataset"].unique()):
        ds_df = all_labels[all_labels["dataset"] == ds]
        ds_acc = accuracy_score(ds_df["y_true"], ds_df["y_pred"])
        ds_f1 = f1_score(ds_df["y_true"], ds_df["y_pred"], average=METRIC_AVERAGE, zero_division=0)
        ds_n_errors = ds_df["is_error"].sum()
        ds_n_fp = ds_df["is_fp"].sum()
        ds_n_fn = ds_df["is_fn"].sum()
        ds_hc_errors = len(
            ds_df[(ds_df["is_error"]) & (ds_df["pred_confidence"] >= min_confidence)]
        )

        dataset_errors.append(
            {
                "dataset": ds,
                "n": len(ds_df),
                "acc": ds_acc,
                "f1": ds_f1,
                "errors": ds_n_errors,
                "fp": ds_n_fp,
                "fn": ds_n_fn,
                "hc_errors": ds_hc_errors,
            }
        )

        console.print(
            f"  {ds}: F1={ds_f1:.3f}, Acc={ds_acc:.3f}, "
            f"Errors={ds_n_errors} (FP={ds_n_fp}, FN={ds_n_fn}), "
            f"HiConf={ds_hc_errors}"
        )

    # Feature correlation with errors
    console.print("\n[bold]Feature Analysis: Errors vs Correct[/bold]")
    console.print("-" * 70)

    # Compute mean feature values for errors vs correct predictions
    errors_df = all_labels[all_labels["is_error"]]
    correct_df = all_labels[~all_labels["is_error"]]

    feature_diffs = []
    for feat_name in matcher.feature_names:
        if feat_name in all_labels.columns:
            err_mean = errors_df[feat_name].mean()
            cor_mean = correct_df[feat_name].mean()
            # Handle inf/nan values and division by zero
            if np.isfinite(err_mean) and np.isfinite(cor_mean) and cor_mean != 0:
                pct_diff = (err_mean - cor_mean) / abs(cor_mean) * 100
            elif np.isfinite(err_mean) and np.isfinite(cor_mean):
                pct_diff = 0 if err_mean == cor_mean else float("inf")
            else:
                pct_diff = np.nan  # Skip features with inf/nan means
            feature_diffs.append(
                {
                    "feature": feat_name,
                    "error_mean": err_mean,
                    "correct_mean": cor_mean,
                    "pct_diff": pct_diff,
                }
            )

    # Sort by absolute percentage difference (filter out nan values for sorting)
    feature_diffs = [fd for fd in feature_diffs if np.isfinite(fd["pct_diff"])]
    feature_diffs.sort(key=lambda x: abs(x["pct_diff"]), reverse=True)

    console.print("  Features with largest difference between errors and correct predictions:")
    for fd in feature_diffs[:10]:
        direction = "↑" if fd["pct_diff"] > 0 else "↓"
        console.print(
            f"    {fd['feature']}: errors={fd['error_mean']:.3f}, "
            f"correct={fd['correct_mean']:.3f} ({direction}{abs(fd['pct_diff']):.1f}%)"
        )

    # Export worst errors for review
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export all errors with details
    error_export = all_labels[all_labels["is_error"]].copy()
    error_export = error_export.sort_values("pred_confidence", ascending=False)

    # Select relevant columns for export
    export_cols = [
        "dataset",
        "gers_id",
        "target_id",
        "label",
        "y_pred",
        "match_prob",
        "pred_confidence",
        "is_fp",
        "is_fn",
    ]
    # Add key features
    key_features = [
        "name_jaro_winkler",
        "buffer_iou_5m",
        "hausdorff_distance_m",
        "class_similarity",
        "lateral_offset_m",
        "min_coverage",
    ]
    for feat in key_features:
        if feat in error_export.columns:
            export_cols.append(feat)

    error_export = error_export[[c for c in export_cols if c in error_export.columns]]

    errors_path = output_dir / "prediction_errors.csv"
    error_export.to_csv(errors_path, index=False)
    console.print(f"\n[green]Exported {len(error_export)} errors to {errors_path}[/green]")

    # Export high-confidence errors separately (priority for review)
    if len(high_conf_errors) > 0:
        hc_export = high_conf_errors[[c for c in export_cols if c in high_conf_errors.columns]]
        hc_export = hc_export.sort_values("pred_confidence", ascending=False)
        hc_path = output_dir / "high_confidence_errors.csv"
        hc_export.to_csv(hc_path, index=False)
        console.print(
            f"[green]Exported {len(hc_export)} high-confidence errors to {hc_path}[/green]"
        )

    # Print top errors per dataset
    console.print(f"\n[bold]Top {top_n} Worst Errors Per Dataset[/bold]")
    console.print("(Sorted by model confidence - these are systematic failures or label issues)")
    console.print("-" * 70)

    for ds in sorted(all_labels["dataset"].unique()):
        ds_errors = all_labels[
            (all_labels["dataset"] == ds) & (all_labels["is_error"])
        ].sort_values("pred_confidence", ascending=False)

        if len(ds_errors) == 0:
            continue

        console.print(f"\n[bold]{ds}[/bold] ({len(ds_errors)} errors)")

        for i, (_, row) in enumerate(ds_errors.head(top_n).iterrows()):
            error_type = "FP" if row["is_fp"] else "FN"
            pred_label = "match" if row["y_pred"] == 1 else "no_match"
            true_label = row["label"]

            # Get key feature values
            name_sim = row.get("name_jaro_winkler", 0)
            buf_iou = row.get("buffer_iou_5m", 0)
            haus = row.get("hausdorff_distance_m", 0)

            console.print(
                f"  {i + 1}. [{error_type}] conf={row['pred_confidence']:.2f} "
                f"pred={pred_label} actual={true_label}"
            )
            console.print(f"      gers={row['gers_id'][:20]}... target={row['target_id'][:20]}...")
            console.print(f"      name_jw={name_sim:.2f} buf_iou={buf_iou:.2f} haus={haus:.1f}m")

    # Summary recommendations
    console.print(f"\n{'=' * 70}")
    console.print("[bold]RECOMMENDATIONS[/bold]")
    console.print("=" * 70)

    # Identify datasets that need attention
    worst_datasets = sorted(dataset_errors, key=lambda x: x["f1"])[:3]
    if worst_datasets:
        console.print("\n[bold]Priority Datasets for Review:[/bold]")
        for ds in worst_datasets:
            console.print(f"  - {ds['dataset']}: F1={ds['f1']:.3f}, {ds['errors']} errors")

    # Recommend actions based on error patterns
    total_hc = sum(d["hc_errors"] for d in dataset_errors)
    if total_hc > 10:
        console.print(f"\n[yellow]Found {total_hc} high-confidence errors.[/yellow]")
        console.print("  → Review these in the labeling UI to check if labels are correct")
        console.print(f"  → See: {output_dir / 'high_confidence_errors.csv'}")

    if n_fp > n_fn * 2:
        console.print("\n[yellow]Model has 2x more False Positives than False Negatives.[/yellow]")
        console.print("  → Model is too aggressive in predicting matches")
        console.print("  → Consider raising the match threshold or adding more no_match examples")
    elif n_fn > n_fp * 2:
        console.print("\n[yellow]Model has 2x more False Negatives than False Positives.[/yellow]")
        console.print("  → Model is too conservative in predicting matches")
        console.print("  → Consider lowering the match threshold or adding more match examples")

    console.print("\n[green]Error analysis complete![/green]")


@analyze_app.command("labels")
def analyze_labels(
    labels_dir: Path = typer.Option(
        Path("labels"),
        "--labels",
        "-l",
        help="Directory containing labels",
    ),
):
    """Show statistics about label data.

    Displays counts and distribution of human and agent labels across datasets.

    Examples:
        matcher analyze labels
        matcher analyze labels --labels /path/to/labels
    """
    from ..labeling.feature_store import FeatureStore
    from ..labeling.label_store import LabelStore

    labels_dir = Path(labels_dir)

    console.print("[blue]Loading labels...[/blue]\n")

    # Try legacy format first
    legacy_labels = LabelStore.load_all(labels_dir)
    human_labels = LabelStore.load_human_labels(labels_dir / "human")
    agent_labels = LabelStore.load_agent_labels(labels_dir / "agent")
    features = FeatureStore.load_all(labels_dir / "features")

    console.print("[bold]Label Statistics[/bold]\n")

    # Legacy labels
    if len(legacy_labels) > 0:
        console.print(f"Legacy labels (embedded features): {len(legacy_labels)}")
        if "dataset" in legacy_labels.columns:
            for dataset in sorted(legacy_labels["dataset"].unique()):
                count = (legacy_labels["dataset"] == dataset).sum()
                console.print(f"  {dataset}: {count}")
        console.print()

    # Human labels (normalized)
    if len(human_labels) > 0:
        console.print(f"Human labels (normalized): {len(human_labels)}")
        if "dataset" in human_labels.columns:
            for dataset in sorted(human_labels["dataset"].unique()):
                count = (human_labels["dataset"] == dataset).sum()
                console.print(f"  {dataset}: {count}")
        console.print()

    # Agent labels
    if len(agent_labels) > 0:
        console.print(f"Agent labels: {len(agent_labels)}")
        if "dataset" in agent_labels.columns:
            for dataset in sorted(agent_labels["dataset"].unique()):
                count = (agent_labels["dataset"] == dataset).sum()
                console.print(f"  {dataset}: {count}")
        console.print()

        # Label distribution
        if "label" in agent_labels.columns:
            console.print("Agent label distribution:")
            for label in sorted(agent_labels["label"].unique()):
                count = (agent_labels["label"] == label).sum()
                console.print(f"  {label}: {count}")
        console.print()

    # Features
    if len(features) > 0:
        console.print(f"Feature records: {len(features)}")
        if "dataset" in features.columns:
            for dataset in sorted(features["dataset"].unique()):
                count = (features["dataset"] == dataset).sum()
                console.print(f"  {dataset}: {count}")
        console.print()

    # Summary
    total_human = len(legacy_labels) + len(human_labels)
    total_agent = len(agent_labels)
    console.print(f"[bold]Total: {total_human} human, {total_agent} agent labels[/bold]")


@analyze_app.command("integrate")
def analyze_integrate(
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
    no_connectivity_gating: bool = typer.Option(
        False,
        "--no-connectivity-gating",
        help="Disable connectivity gating (don't promote bridge segments)",
    ),
    min_bridge_overlap_m: float = typer.Option(
        10.0,
        "--min-bridge-overlap-m",
        help="Minimum overlap (meters) at each end with reference to qualify as bridge",
    ),
):
    """Integrate unmatched segments into reference network.

    Takes the output of the matching pipeline and creates a unified
    planarized network, flagging disconnected orphan components for QA.

    Examples:
        # Single dataset
        matcher analyze integrate data/raw/us_boston_overture_segments.parquet \\
            -t us_boston_streets:data/output/bridge.parquet:data/output/unmatched.parquet:1 \\
            -o data/integrated

        # Multiple datasets with priority
        matcher analyze integrate data/raw/us_boston_overture_segments.parquet \\
            -t us_boston_streets:data/us_boston_streets/bridge.parquet:data/us_boston_streets/unmatched.parquet:1 \\
            -t us_boston_bike_network:data/us_boston_bike_network/bridge.parquet:data/us_boston_bike_network/unmatched.parquet:2 \\
            -o data/integrated

        # From config file
        matcher analyze integrate data/raw/us_boston_overture_segments.parquet -c integration_config.yaml -o data/integrated
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
                enable_connectivity_gating=not no_connectivity_gating,
                min_bridge_overlap_m=min_bridge_overlap_m,
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


@analyze_app.command("validate")
def analyze_validate(
    overture: Path = typer.Argument(
        ...,
        help="Path to Overture segments parquet file",
    ),
    bbox: str = typer.Option(
        ...,
        "--bbox",
        "-b",
        help="Bounding box for fresh OSM fetch: xmin,ymin,xmax,ymax",
    ),
    output: Path = typer.Option(
        Path("validation/experiment"),
        "--output",
        "-o",
        help="Output directory for experiment results",
    ),
    strategy: str = typer.Option(
        "random",
        "--strategy",
        "-s",
        help="Drop strategy: random, bbox, source, class",
    ),
    method: str = typer.Option(
        "xgboost",
        "--method",
        "-m",
        help="Matching method: xgboost",
    ),
    fraction: float = typer.Option(
        0.1,
        "--fraction",
        "-f",
        help="Fraction to drop for 'random' strategy (0.0-1.0)",
    ),
    source_dataset: str = typer.Option(
        "OpenStreetMap",
        "--source-dataset",
        help="Dataset to drop for 'source' strategy (e.g., OpenStreetMap, Meta)",
    ),
    road_class: str = typer.Option(
        "residential",
        "--road-class",
        help="Road class to drop for 'class' strategy",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
    fast: bool = typer.Option(
        False,
        "--fast",
        help="Fast mode: only match segments that should match (dropped record_ids)",
    ),
):
    """Run a validation experiment using ground-truth from Overture provenance.

    This command tests matching quality by:
    1. Dropping segments from Overture based on the chosen strategy
    2. Fetching fresh OSM data for the bounding box
    3. Running the matcher to see if dropped segments get matched back
    4. Evaluating results and computing recall metrics

    Note: To validate data file versions, use 'matcher data validate' instead.

    Examples:
        # Drop 10% of OSM segments randomly
        matcher analyze validate data/raw/overture.parquet \\
            --bbox "-71.19,42.21,-70.92,42.40" \\
            --strategy random --fraction 0.1 \\
            --output validation/random_10pct/

        # Drop all segments from a specific source
        matcher analyze validate data/raw/overture.parquet \\
            --bbox "-71.19,42.21,-70.92,42.40" \\
            --strategy source --source-dataset Meta \\
            --output validation/meta_holdout/

        # Drop residential roads
        matcher analyze validate data/raw/overture.parquet \\
            --bbox "-71.19,42.21,-70.92,42.40" \\
            --strategy class --road-class residential \\
            --output validation/residential_holdout/
    """
    from ..validation import run_validation_experiment

    # Parse bbox
    try:
        coords = [float(x.strip()) for x in bbox.split(",")]
        if len(coords) != 4:
            raise ValueError()
        bbox_tuple = tuple(coords)
    except ValueError:
        console.print(
            "[red]Error: bbox must be 4 comma-separated values: xmin,ymin,xmax,ymax[/red]"
        )
        raise typer.Exit(1) from None

    # Validate strategy
    valid_strategies = {"random", "bbox", "source", "class"}
    if strategy not in valid_strategies:
        console.print(f"[red]Error: strategy must be one of {valid_strategies}[/red]")
        raise typer.Exit(1)

    # Validate fraction for random strategy
    if strategy == "random" and not 0.0 <= fraction <= 1.0:
        console.print("[red]Error: fraction must be between 0.0 and 1.0[/red]")
        raise typer.Exit(1)

    # Validate input file
    if not overture.exists():
        console.print(f"[red]Error: Overture file not found: {overture}[/red]")
        raise typer.Exit(1)

    console.print("[blue]Starting validation experiment...[/blue]")
    console.print(f"  Overture: {overture}")
    console.print(f"  Strategy: {strategy}")
    console.print(f"  Matcher: {method}")
    console.print(f"  Output: {output}")

    try:
        result = run_validation_experiment(
            overture_path=overture,
            output_dir=output,
            bbox=bbox_tuple,
            strategy=strategy,
            matcher_method=method,
            fraction=fraction,
            source_dataset=source_dataset,
            road_class=road_class,
            seed=seed,
            fast_mode=fast,
        )

        console.print()
        console.print("[green]Experiment complete![/green]")
        console.print(f"  Overture segments: {result.n_overture}")
        console.print(f"  Dropped segments: {result.n_dropped}")
        console.print(f"  Fresh OSM segments: {result.n_fresh_osm}")
        console.print(f"  Matched: {result.n_matched}")
        console.print(f"  Unmatched: {result.n_unmatched}")
        console.print()
        console.print(f"  [bold]Recall: {result.metrics['recall']:.3f}[/bold]")
        if result.metrics.get("mean_confidence"):
            console.print(f"  Mean confidence: {result.metrics['mean_confidence']:.3f}")
        console.print()
        console.print(f"[green]Results saved to {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None
