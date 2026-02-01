"""Classification management commands."""

import json
from pathlib import Path

import pandas as pd
import typer

from .utils import console

# Create class group
class_app = typer.Typer(
    name="class",
    help="Classification management commands",
    no_args_is_help=True,
)


@class_app.command("discover")
def discover_classes(
    dataset: Path = typer.Argument(..., help="Target dataset parquet file"),
    reference: Path | None = typer.Option(
        None,
        "--reference",
        "-r",
        help="Reference data (Overture segments) for match-based analysis",
    ),
    bridge: Path | None = typer.Option(
        None,
        "--bridge",
        "-b",
        help="Existing bridge file for match-based analysis",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output YAML path (default: datasets/{name}.yaml). Merges with existing config if present.",
    ),
    print_only: bool = typer.Option(
        False,
        "--print-only",
        "-p",
        help="Only print report, don't save config",
    ),
):
    """Discover class mapping for a new dataset.

    Analyzes the dataset structure, detects classification columns,
    and generates a mapping configuration to Overture road classes.

    Examples:
        # Basic discovery
        matcher class discover data/raw/us_boston_streets.parquet

        # With match-based analysis (more accurate)
        matcher class discover data/raw/us_boston_streets.parquet \\
            --reference data/raw/us_boston_overture_segments.parquet \\
            --bridge data/output/us_boston_streets_bridge.parquet

        # Print report only (don't save config)
        matcher class discover data/raw/new_dataset.parquet --print-only
    """
    from ..datasets.discover import discover_dataset, print_discovery_report, save_dataset_config

    if not dataset.exists():
        console.print(f"[red]Error: Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    if reference and not reference.exists():
        console.print(f"[red]Error: Reference not found: {reference}[/red]")
        raise typer.Exit(1)

    if bridge and not bridge.exists():
        console.print(f"[red]Error: Bridge file not found: {bridge}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Analyzing dataset: {dataset.name}[/blue]")
    if reference:
        console.print(f"  Reference: {reference}")
    if bridge:
        console.print(f"  Bridge: {bridge}")
    console.print()

    report = discover_dataset(
        dataset_path=dataset,
        reference_path=reference,
        bridge_path=bridge,
    )

    print_discovery_report(report)

    if not print_only and report.suggested_config:
        from ..datasets.schema import (
            ClassificationConfig,
            get_dataset_config,
            get_datasets_dir,
        )
        from ..datasets.schema import (
            ClassMappingRule as NewClassMappingRule,
        )
        from ..datasets.schema import (
            SourceClassification as NewSourceClassification,
        )
        from ..datasets.schema import (
            save_dataset_config as save_new_config,
        )

        dataset_name = dataset.stem
        if output:
            saved_path = output
        else:
            saved_path = get_datasets_dir() / f"{dataset_name}.yaml"

        # Check if config already exists and merge
        existing_config = get_dataset_config(dataset_name)
        if existing_config:
            console.print(f"[blue]Merging with existing config: {dataset_name}.yaml[/blue]")

            # Build new classification from discovered rules
            old_config = report.suggested_config
            new_rules = []
            for rule in old_config.class_mapping_rules:
                new_rules.append(
                    NewClassMappingRule(
                        source_value=rule.source_value,
                        target_class=rule.target_class,
                        conditions=rule.conditions,
                        priority=rule.priority,
                    )
                )

            new_source_class = None
            if old_config.source_classification:
                new_source_class = NewSourceClassification(
                    column=old_config.source_classification.column,
                    description=old_config.source_classification.description,
                    values=old_config.source_classification.values,
                    documentation_url=old_config.source_classification.documentation_url,
                )

            existing_config.classification = ClassificationConfig(
                source_classification=new_source_class,
                class_mapping_rules=new_rules,
                default_class=old_config.default_class,
                confidence=old_config.confidence,
            )

            if old_config.notes:
                existing_config.notes = old_config.notes

            save_new_config(existing_config, saved_path)
        else:
            # No existing config - save using old format for now
            # TODO: Create new format config from scratch
            saved_path = save_dataset_config(report.suggested_config, output)

        console.print()
        console.print(f"[green]Configuration saved to: {saved_path}[/green]")
        console.print("[yellow]Review and adjust the mapping rules as needed.[/yellow]")


@class_app.command("analyze")
def analyze_classes(
    bridge_file: Path = typer.Argument(
        None,
        help="Bridge file to analyze (optional if using --from-labels)",
    ),
    from_labels: bool = typer.Option(
        False,
        "--from-labels",
        help="Analyze from labeled data instead of bridge file",
    ),
    reference: Path = typer.Option(
        None,
        "--reference",
        "-r",
        help="Reference parquet file (required with bridge file)",
    ),
    target: Path = typer.Option(
        None,
        "--target",
        "-t",
        help="Target parquet file (required with bridge file)",
    ),
    labels_dir: Path = typer.Option(
        Path("labels"),
        "--labels-dir",
        help="Directory containing Hive-partitioned label CSVs",
    ),
    geometries_dir: Path = typer.Option(
        Path("label_geometries"),
        "--geometries-dir",
        help="Directory containing geometry companion files",
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON file for detailed report",
    ),
    low_threshold: float = typer.Option(
        0.3,
        "--low-threshold",
        help="Threshold for low similarity matches",
    ),
    high_threshold: float = typer.Option(
        0.8,
        "--high-threshold",
        help="Threshold for high similarity no-matches",
    ),
):
    """Analyze class confusion to identify mapping issues.

    This command analyzes class mappings by examining:
    - Tier violations: vehicle<->pedestrian pairs labeled as match (likely mapping errors)
    - Low similarity matches: class differs but humans labeled match
    - High similarity no-matches: class matches but humans labeled no_match

    Examples:
        # Analyze from labeled data (recommended)
        matcher class analyze --from-labels

        # Analyze from bridge file
        matcher class analyze bridge.parquet \\
            --reference ref.parquet --target target.parquet

        # Save detailed report
        matcher class analyze --from-labels --output analysis_report.json
    """
    from ..quality.class_analysis import (
        analyze_class_confusion_from_bridge,
        analyze_class_confusion_from_labels,
        format_analysis_report,
    )

    if from_labels:
        console.print("[blue]Analyzing class confusion from labels...[/blue]")
        report = analyze_class_confusion_from_labels(
            labels_dir=labels_dir,
            geometries_dir=geometries_dir,
            low_similarity_threshold=low_threshold,
            high_similarity_threshold=high_threshold,
        )
    else:
        if bridge_file is None:
            console.print("[red]Error: Provide a bridge file or use --from-labels[/red]")
            raise typer.Exit(1)

        if reference is None or target is None:
            console.print(
                "[red]Error: --reference and --target are required when analyzing a bridge file[/red]"
            )
            raise typer.Exit(1)

        console.print(f"[blue]Analyzing class confusion from {bridge_file}...[/blue]")
        report = analyze_class_confusion_from_bridge(
            bridge_path=bridge_file,
            reference_path=reference,
            target_path=target,
            low_similarity_threshold=low_threshold,
            high_similarity_threshold=high_threshold,
        )

    # Print formatted report
    console.print()
    console.print(format_analysis_report(report))

    # Save JSON if requested
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        console.print(f"\n[green]Detailed report saved to {output}[/green]")


@class_app.command("detect-non-roads")
def detect_non_roads(
    data_file: Path = typer.Argument(
        ...,
        help="Data file to analyze (parquet or geojson)",
    ),
    type_code_column: str = typer.Option(
        None,
        "--type-code-column",
        "-c",
        help="Column containing type codes (e.g., 'source_tags.cd_tipo_logradouro')",
    ),
    non_road_codes: str = typer.Option(
        None,
        "--non-road-codes",
        help="Comma-separated list of non-road type codes (e.g., 'PC,PQ,ES')",
    ),
    check_closed_loops: bool = typer.Option(
        True,
        "--check-closed-loops/--no-check-closed-loops",
        help="Enable geometry-based closed loop detection",
    ),
    compactness_threshold: float = typer.Option(
        0.3,
        "--compactness-threshold",
        help="Compactness ratio above which to flag as non-road (0-1)",
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON file for detailed report",
    ),
    output_filtered: Path = typer.Option(
        None,
        "--output-filtered",
        help="Output parquet file with non-roads filtered out",
    ),
):
    """Detect non-road features like plazas, barriers, and stairs.

    Some datasets include road-like features that aren't actual roads.
    This command identifies them using:
    - Closed loop detection (geometry-based)
    - Compactness ratio (distinguishes plaza outlines from roads)
    - Dataset-specific type codes

    Examples:
        # Basic detection with closed loops
        matcher class detect-non-roads data/raw/br_sao_paulo_roads.parquet

        # With type code filtering for São Paulo
        matcher class detect-non-roads data/raw/br_sao_paulo_roads.parquet \\
            --type-code-column "source_tags.cd_tipo_logradouro" \\
            --non-road-codes PC,PQ,ES,ESC,LG

        # Save filtered output
        matcher class detect-non-roads data/raw/br_sao_paulo_roads.parquet \\
            --output-filtered data/raw/br_sao_paulo_roads_filtered.parquet
    """
    import geopandas as gpd

    from ..quality.non_road_detection import (
        analyze_non_road_features,
        detect_non_road_features,
        format_non_road_report,
    )

    console.print(f"[blue]Loading {data_file}...[/blue]")

    if data_file.suffix == ".parquet":
        gdf = gpd.read_parquet(data_file)
    else:
        gdf = gpd.read_file(data_file)

    console.print(f"  Loaded {len(gdf):,} features")

    # Parse non-road codes
    non_road_type_codes = None
    if non_road_codes:
        non_road_type_codes = {code.strip() for code in non_road_codes.split(",")}
        console.print(f"  Non-road type codes: {non_road_type_codes}")

    # Analyze
    console.print("[blue]Detecting non-road features...[/blue]")
    report = analyze_non_road_features(
        gdf,
        type_code_column=type_code_column,
        non_road_type_codes=non_road_type_codes,
        check_closed_loops=check_closed_loops,
        compactness_threshold=compactness_threshold,
    )

    # Print formatted report
    console.print()
    console.print(format_non_road_report(report))

    # Save JSON if requested
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        console.print(f"\n[green]Detailed report saved to {output}[/green]")

    # Save filtered output if requested
    if output_filtered:
        console.print("\n[blue]Filtering non-road features...[/blue]")

        flags = detect_non_road_features(
            gdf,
            type_code_column=type_code_column,
            non_road_type_codes=non_road_type_codes,
            check_closed_loops=check_closed_loops,
            compactness_threshold=compactness_threshold,
        )

        filtered_gdf = gdf[~flags]
        console.print(f"  Keeping {len(filtered_gdf):,} road features (removed {flags.sum():,})")

        output_filtered.parent.mkdir(parents=True, exist_ok=True)
        filtered_gdf.to_parquet(output_filtered)
        console.print(f"[green]Filtered data saved to {output_filtered}[/green]")


@class_app.command("train-predictor")
def train_class_predictor(
    bridge_file: Path = typer.Argument(
        ...,
        help="Bridge file with confident matches",
    ),
    reference: Path = typer.Option(
        ...,
        "--reference",
        "-r",
        help="Reference parquet file (Overture)",
    ),
    target: Path = typer.Option(
        ...,
        "--target",
        "-t",
        help="Target parquet file",
    ),
    output: Path = typer.Option(
        Path("data/models/class_predictor.joblib"),
        "--output",
        "-o",
        help="Output path for trained model",
    ),
    confidence_threshold: float = typer.Option(
        0.9,
        "--confidence-threshold",
        help="Minimum confidence to include in training",
    ),
    name_column: str = typer.Option(
        "names",
        "--name-column",
        help="Column containing road names in target",
    ),
):
    """Train a lightweight class predictor from confident matches.

    The predictor learns to predict Overture road classes from target
    features like names and physical attributes. This is useful for
    suggesting class mappings for datasets with poor or missing mappings.

    Examples:
        matcher class train-predictor bridge.parquet \\
            --reference overture.parquet --target local.parquet

        matcher class train-predictor bridge.parquet \\
            --reference overture.parquet --target local.parquet \\
            --confidence-threshold 0.85 --output model.joblib
    """
    import geopandas as gpd

    from ..classification import LightweightClassPredictor

    console.print("[blue]Loading data...[/blue]")

    # Load bridge file
    bridge = pd.read_parquet(bridge_file)
    console.print(f"  Bridge file: {len(bridge):,} entries")

    # Filter to confident matches
    confident = bridge[bridge["confidence"] >= confidence_threshold]
    console.print(f"  Confident matches (>= {confidence_threshold}): {len(confident):,}")

    if len(confident) < 50:
        console.print("[red]Error: Not enough confident matches for training[/red]")
        raise typer.Exit(1)

    # Load reference to get class labels
    ref_gdf = gpd.read_parquet(reference)
    ref_gdf["id"] = ref_gdf["id"].astype(str)
    ref_lookup = ref_gdf.set_index("id")
    console.print(f"  Reference: {len(ref_gdf):,} segments")

    # Load target
    target_gdf = gpd.read_parquet(target)
    target_gdf["id"] = target_gdf["id"].astype(str)
    console.print(f"  Target: {len(target_gdf):,} segments")

    # Build training set
    console.print("[blue]Building training set...[/blue]")
    train_indices = []
    train_labels = []

    for _, row in confident.iterrows():
        gers_id = str(row.get("gers_id", row.get("ref_id", "")))
        target_id = str(row.get("local_id", row.get("target_id", "")))

        if gers_id in ref_lookup.index:
            ref_class = ref_lookup.loc[gers_id].get("class")
            if ref_class and ref_class != "unknown":
                # Find target row
                target_mask = target_gdf["id"] == target_id
                if target_mask.any():
                    target_idx = target_gdf.index[target_mask][0]
                    train_indices.append(target_idx)
                    train_labels.append(ref_class)

    console.print(f"  Training samples: {len(train_indices):,}")

    if len(train_indices) < 50:
        console.print("[red]Error: Not enough training samples with valid classes[/red]")
        raise typer.Exit(1)

    # Create training GeoDataFrame
    train_gdf = target_gdf.loc[train_indices]
    train_labels_series = pd.Series(train_labels, index=train_indices)

    # Train predictor
    console.print("[blue]Training predictor...[/blue]")
    predictor = LightweightClassPredictor()
    stats = predictor.train(train_gdf, train_labels_series, name_column=name_column)

    console.print("\n[green]Training complete![/green]")
    console.print(f"  Samples: {stats['n_samples']:,}")
    console.print(f"  Classes: {stats['n_classes']}")
    console.print(f"  Accuracy: {stats['accuracy']:.3f}")

    # Show feature importance
    importance = predictor.feature_importance()
    console.print("\n[blue]Top features:[/blue]")
    for feat, imp in importance.head(10).items():
        console.print(f"  {feat}: {imp:.3f}")

    # Save model
    predictor.save(output)
    console.print(f"\n[green]Model saved to {output}[/green]")


@class_app.command("predict")
def predict_classes(
    target: Path = typer.Argument(
        ...,
        help="Target parquet file to predict classes for",
    ),
    predictor_path: Path = typer.Option(
        Path("data/models/class_predictor.joblib"),
        "--predictor",
        "-p",
        help="Path to trained predictor model",
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output parquet file with predictions (default: adds _predicted suffix)",
    ),
    name_column: str = typer.Option(
        "names",
        "--name-column",
        help="Column containing road names",
    ),
):
    """Apply trained class predictor to a target dataset.

    Adds 'predicted_class' and 'prediction_confidence' columns to the output.

    Examples:
        matcher class predict target.parquet

        matcher class predict target.parquet \\
            --predictor model.joblib --output target_with_predictions.parquet
    """
    import geopandas as gpd

    from ..classification import LightweightClassPredictor

    console.print(f"[blue]Loading predictor from {predictor_path}...[/blue]")
    predictor = LightweightClassPredictor.load(predictor_path)

    console.print(f"[blue]Loading target data from {target}...[/blue]")
    gdf = gpd.read_parquet(target)
    console.print(f"  Loaded {len(gdf):,} segments")

    console.print("[blue]Predicting classes...[/blue]")
    predictions = predictor.predict(gdf, name_column=name_column)
    probabilities = predictor.predict_proba(gdf, name_column=name_column)

    # Add predictions to GeoDataFrame
    gdf["predicted_class"] = predictions
    gdf["prediction_confidence"] = probabilities.max(axis=1)

    # Summary
    console.print("\n[blue]Prediction summary:[/blue]")
    for cls in sorted(predictions.unique()):
        count = (predictions == cls).sum()
        pct = count / len(predictions) * 100
        console.print(f"  {cls}: {count:,} ({pct:.1f}%)")

    # Confidence stats
    console.print("\n[blue]Confidence distribution:[/blue]")
    console.print(f"  Mean: {gdf['prediction_confidence'].mean():.3f}")
    console.print(f"  Median: {gdf['prediction_confidence'].median():.3f}")
    console.print(f"  >= 0.8: {(gdf['prediction_confidence'] >= 0.8).sum():,}")
    console.print(f"  >= 0.5: {(gdf['prediction_confidence'] >= 0.5).sum():,}")

    # Save output
    if output is None:
        output = target.with_stem(target.stem + "_predicted")

    output.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output)
    console.print(f"\n[green]Predictions saved to {output}[/green]")


@class_app.command("analyze-predictor")
def analyze_class_predictor(
    bridge_file: Path = typer.Argument(
        ...,
        help="Bridge file with confident matches",
    ),
    reference: Path = typer.Option(
        ...,
        "--reference",
        "-r",
        help="Reference parquet file (Overture)",
    ),
    target: Path = typer.Option(
        ...,
        "--target",
        "-t",
        help="Target parquet file",
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON file for detailed analysis",
    ),
    confidence_threshold: float = typer.Option(
        0.9,
        "--confidence-threshold",
        help="Minimum confidence to include in analysis",
    ),
    name_column: str = typer.Option(
        "names",
        "--name-column",
        help="Column containing road names in target",
    ),
    class_column: str = typer.Option(
        "class",
        "--class-column",
        help="Column containing source class in target",
    ),
):
    """Analyze class predictor performance on known matches.

    This command:
    1. Trains a class predictor on confident matches
    2. Evaluates prediction quality (exact, tier, hierarchy accuracy)
    3. Analyzes how source classes map to Overture classes
    4. Identifies tier violations and mapping errors

    Examples:
        matcher class analyze-predictor bridge.parquet \\
            --reference overture.parquet --target local.parquet

        matcher class analyze-predictor bridge.parquet \\
            --reference overture.parquet --target local.parquet \\
            --output analysis.json
    """
    import geopandas as gpd

    from ..classification import (
        LightweightClassPredictor,
        analyze_predictions,
        analyze_source_class_mapping,
        format_prediction_analysis,
    )

    console.print("[blue]Loading data...[/blue]")

    # Load bridge file
    bridge = pd.read_parquet(bridge_file)
    console.print(f"  Bridge file: {len(bridge):,} entries")

    # Filter to confident matches
    confident = bridge[bridge["confidence"] >= confidence_threshold]
    console.print(f"  Confident matches (>= {confidence_threshold}): {len(confident):,}")

    if len(confident) < 50:
        console.print("[red]Error: Not enough confident matches for analysis[/red]")
        raise typer.Exit(1)

    # Load reference
    ref_gdf = gpd.read_parquet(reference)
    ref_gdf["id"] = ref_gdf["id"].astype(str)
    ref_lookup = ref_gdf.set_index("id")
    console.print(f"  Reference: {len(ref_gdf):,} segments")

    # Load target
    target_gdf = gpd.read_parquet(target)
    target_gdf["id"] = target_gdf["id"].astype(str)
    console.print(f"  Target: {len(target_gdf):,} segments")

    # Build training/evaluation set
    console.print("[blue]Building evaluation set...[/blue]")
    eval_indices = []
    true_classes = []
    source_classes = []

    for _, row in confident.iterrows():
        gers_id = str(row.get("gers_id", row.get("ref_id", "")))
        target_id = str(row.get("local_id", row.get("target_id", "")))

        if gers_id in ref_lookup.index:
            ref_class = ref_lookup.loc[gers_id].get("class")
            if ref_class and ref_class != "unknown":
                target_mask = target_gdf["id"] == target_id
                if target_mask.any():
                    target_idx = target_gdf.index[target_mask][0]
                    eval_indices.append(target_idx)
                    true_classes.append(ref_class)

                    src_class = target_gdf.loc[target_idx].get(class_column)
                    source_classes.append(src_class if src_class else "unknown")

    console.print(f"  Evaluation samples: {len(eval_indices):,}")

    if len(eval_indices) < 50:
        console.print("[red]Error: Not enough samples with valid classes[/red]")
        raise typer.Exit(1)

    # Create evaluation dataframes
    eval_gdf = target_gdf.loc[eval_indices]
    true_classes_series = pd.Series(true_classes, index=eval_indices)
    source_classes_series = pd.Series(source_classes, index=eval_indices)

    # Get names for error reporting
    if name_column in eval_gdf.columns:
        names_series = eval_gdf[name_column]
    else:
        names_series = None

    # Train predictor
    console.print("[blue]Training class predictor...[/blue]")
    predictor = LightweightClassPredictor()
    train_stats = predictor.train(
        eval_gdf,
        true_classes_series,
        name_column=name_column,
        class_column=class_column,
    )

    console.print(f"  Training accuracy: {train_stats['accuracy']:.1%}")
    console.print(f"  Training tier accuracy: {train_stats['tier_accuracy']:.1%}")

    # Predict on evaluation set
    console.print("[blue]Evaluating predictions...[/blue]")
    predictions = predictor.predict(eval_gdf)

    # Analyze predictions
    pred_analysis = analyze_predictions(
        true_classes_series,
        predictions,
        segment_ids=eval_gdf["id"] if "id" in eval_gdf.columns else None,
        names=names_series,
    )

    # Print prediction analysis
    console.print()
    console.print(format_prediction_analysis(pred_analysis))

    # Analyze source class mapping
    console.print()
    console.print("[blue]Analyzing source class mappings...[/blue]")
    mapping_analysis = analyze_source_class_mapping(
        source_classes_series,
        true_classes_series,
    )

    console.print("\nSource Class -> Overture Class Mappings:")
    console.print(f"  (Based on {mapping_analysis['n_samples']:,} matched pairs)")
    console.print()

    # Show suggested mappings sorted by confidence
    sorted_mappings = sorted(
        mapping_analysis["suggested_mapping"].items(),
        key=lambda x: mapping_analysis["mapping_confidence"].get(x[0], 0),
        reverse=True,
    )

    for src_class, suggested in sorted_mappings[:20]:
        conf = mapping_analysis["mapping_confidence"].get(src_class, 0)
        dist = mapping_analysis["mapping_distribution"].get(src_class, {})
        total = sum(dist.values())

        # Show top 3 mappings
        top_3 = sorted(dist.items(), key=lambda x: -x[1])[:3]
        dist_str = ", ".join(f"{c}:{n}" for c, n in top_3)

        console.print(f"  {src_class:25} -> {suggested:15} ({conf:.0%} of {total}) [{dist_str}]")

    # Feature importance
    console.print()
    console.print("[blue]Top Predictive Features:[/blue]")
    importance = predictor.feature_importance()
    for feat, imp in importance.head(15).items():
        console.print(f"  {feat:35} {imp:.3f}")

    # Save detailed output
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "training_stats": train_stats,
            "prediction_analysis": pred_analysis.to_dict(),
            "source_mapping_analysis": mapping_analysis,
            "feature_importance": importance.to_dict(),
        }
        with open(output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        console.print(f"\n[green]Detailed analysis saved to {output}[/green]")


@class_app.command("update-mappings")
def update_class_mappings(
    datasets: list[str] = typer.Argument(
        None,
        help="Specific datasets to update (default: all with labels)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Don't save changes, just show what would change",
    ),
    min_confidence: float = typer.Option(
        0.5,
        "--min-confidence",
        help="Minimum confidence threshold for mappings",
    ),
    tier_change_min_confidence: float = typer.Option(
        0.7,
        "--tier-change-min-confidence",
        help="Higher confidence required for tier-changing mappings (vehicle<->pedestrian)",
    ),
    min_samples: int = typer.Option(
        3,
        "--min-samples",
        help="Minimum samples per class",
    ),
    labels_dir: Path = typer.Option(
        Path("labels"),
        "--labels-dir",
        help="Labels directory",
    ),
    geometries_dir: Path = typer.Option(
        Path("label_geometries"),
        "--geometries-dir",
        help="Label geometries directory",
    ),
):
    """Update dataset YAML configs with class mappings derived from labeled matches.

    Analyzes human-labeled matches to determine the best mapping from source
    classification to Overture classes. Applies safety checks for tier-changing
    mappings (vehicle <-> pedestrian).

    IMPORTANT: This command only works correctly for datasets WITHOUT existing
    class_mapping. The geometry store captures post-mapping class values, so
    datasets with existing mappings will produce incorrect results (mapping
    already-mapped values like "residential" instead of raw source values).

    Examples:
        # Dry run on all datasets with labels
        matcher class update-mappings --dry-run

        # Update specific datasets (only those without existing class_mapping)
        matcher class update-mappings us_montana_helena us_usfs_lolo

        # Apply changes (no dry run)
        matcher class update-mappings us_montana_helena
    """
    from glob import glob

    import geopandas as gpd

    from ..classification.predictor import OVERTURE_TIERS, analyze_source_class_mapping
    from ..datasets.schema import (
        FetchConfig,
        get_dataset_config,
        get_datasets_dir,
        save_dataset_config,
    )
    from ..labeling.data_store import DataStore

    labels_dir = labels_dir or Path("labels")
    data_dir = Path("labels/data")

    def get_tier(class_name: str) -> str:
        return OVERTURE_TIERS.get(class_name.lower(), "unknown")

    def is_tier_change(src_class: str, tgt_class: str) -> bool:
        src_tier = get_tier(src_class)
        tgt_tier = get_tier(tgt_class)
        risky_tiers = {"vehicle", "pedestrian"}
        if src_tier in risky_tiers and tgt_tier in risky_tiers:
            return src_tier != tgt_tier
        return False

    # Find datasets with labels
    if datasets:
        dataset_list = datasets
    else:
        dataset_list = [
            p.name.replace("dataset=", "")
            for p in labels_dir.iterdir()
            if p.is_dir() and p.name.startswith("dataset=")
        ]

    console.print(f"Analyzing {len(dataset_list)} datasets...")
    console.print(f"  Min confidence: {min_confidence}")
    console.print(f"  Tier-change min confidence: {tier_change_min_confidence}")
    console.print(f"  Min samples: {min_samples}")
    console.print(f"  Dry run: {dry_run}")

    results = {"updated": [], "no_data": [], "no_changes": []}
    all_warnings = []

    for dataset_id in sorted(dataset_list):
        console.print(f"\n[bold]Dataset: {dataset_id}[/bold]")

        # Load dataset config
        config = get_dataset_config(dataset_id)
        if not config:
            console.print("  [red]No dataset config found[/red]")
            results["no_data"].append(dataset_id)
            continue

        # Get the raw class column name from config
        class_column = config.fetch.class_column if config.fetch else None
        if not class_column:
            console.print("  [yellow]No class_column configured, using geometry store[/yellow]")

        labels_path = labels_dir / f"dataset={dataset_id}" / "data.csv"
        if not labels_path.exists():
            console.print("  Not enough labeled data")
            results["no_data"].append(dataset_id)
            continue

        labels_df = pd.read_csv(labels_path)
        match_labels = labels_df[labels_df["label"] == "match"]

        if len(match_labels) < 10:
            console.print("  Not enough match labels")
            results["no_data"].append(dataset_id)
            continue

        # Load data store for ref class (Overture class is always correct)
        data_store = DataStore(dataset_id=dataset_id, data_dir=data_dir)

        # Try to load raw target data for source class
        target_files = glob(f"data/raw/{dataset_id}*.parquet")
        # Filter out overture and osm files, prefer exact dataset match
        target_files = [f for f in target_files if "overture" not in f.lower()]
        # Prefer exact match (dataset_id_v*.parquet) over variants (dataset_id_osm_*.parquet)
        exact_matches = [
            f for f in target_files if f"/{dataset_id}_v" in f or f"/{dataset_id}.parquet" in f
        ]
        if exact_matches:
            target_files = exact_matches

        raw_class_lookup = {}
        if target_files and class_column:
            try:
                target_gdf = gpd.read_parquet(target_files[0])
                # Build lookup from id -> raw class value from source_tags
                for _, row in target_gdf.iterrows():
                    target_id = row.get("id")
                    source_tags = row.get("source_tags", {})
                    if isinstance(source_tags, dict) and class_column in source_tags:
                        raw_class_lookup[target_id] = source_tags[class_column]
                console.print(
                    f"  Loaded {len(raw_class_lookup)} raw class values from source_tags.{class_column}"
                )
            except Exception as e:
                console.print(f"  [yellow]Could not load raw data: {e}[/yellow]")

        source_classes = []
        overture_classes = []

        for _, row in match_labels.iterrows():
            pair = data_store.get_pair(row["gers_id"], row["target_id"])
            if pair is None:
                continue

            # Get Overture class (ref class is always correct in geometry store)
            ref_class = pair.get("ref_class")

            # Get raw source class - prefer raw lookup, fallback to geometry store
            target_id = row["target_id"]
            if target_id in raw_class_lookup:
                src_class = raw_class_lookup[target_id]
            else:
                src_class = pair.get("target_class")

            if src_class and ref_class:
                source_classes.append(str(src_class).lower().strip())
                overture_classes.append(str(ref_class).lower().strip())

        if len(source_classes) < 10:
            console.print("  Not enough class data")
            results["no_data"].append(dataset_id)
            continue

        # Analyze mappings
        analysis = analyze_source_class_mapping(
            pd.Series(source_classes),
            pd.Series(overture_classes),
        )

        # Build mapping with safety checks
        mapping = {}
        warnings = []

        for src_class, suggested in analysis.get("suggested_mapping", {}).items():
            conf = analysis.get("mapping_confidence", {}).get(src_class, 0)
            dist = analysis.get("mapping_distribution", {}).get(src_class, {})
            total = sum(dist.values()) if dist else 0

            if total < min_samples:
                continue

            tier_change = is_tier_change(src_class, suggested)
            required_conf = tier_change_min_confidence if tier_change else min_confidence

            if conf >= required_conf:
                mapping[src_class] = suggested
                if tier_change:
                    warnings.append(
                        {
                            "type": "tier_change",
                            "source": src_class,
                            "target": suggested,
                            "confidence": conf,
                        }
                    )
            elif tier_change:
                warnings.append(
                    {
                        "type": "tier_change_skipped",
                        "source": src_class,
                        "target": suggested,
                        "confidence": conf,
                        "required": required_conf,
                    }
                )

        # Show warnings
        for w in warnings:
            if w["type"] == "tier_change":
                console.print(
                    f"  [yellow]⚠️ TIER CHANGE: {w['source']} -> {w['target']} (conf={w['confidence']:.0%})[/yellow]"
                )
            else:
                console.print(
                    f"  [red]⛔ SKIPPED: {w['source']} -> {w['target']} (conf={w['confidence']:.0%} < {w['required']:.0%})[/red]"
                )
            all_warnings.append({**w, "dataset": dataset_id})

        if not mapping:
            console.print("  No mappings passed confidence thresholds")
            results["no_data"].append(dataset_id)
            continue

        console.print(f"  Derived mapping ({len(mapping)} classes):")
        for src, tgt in sorted(mapping.items()):
            console.print(f"    {src:25} -> {tgt:15} ({get_tier(tgt)})")

        # Update YAML
        config = get_dataset_config(dataset_id)
        if config is None:
            console.print(f"  [red]No config found for {dataset_id}[/red]")
            results["no_changes"].append(dataset_id)
            continue

        if config.fetch is None:
            config.fetch = FetchConfig()

        # Normalize all keys to lowercase strings for consistency
        def normalize_key(k):
            return str(k).lower().strip()

        existing = config.fetch.class_mapping or {}
        # Normalize existing keys
        existing_normalized = {normalize_key(k): v for k, v in existing.items()}

        # Check for changes against normalized existing
        new_keys = set(mapping.keys()) - set(existing_normalized.keys())
        changed_keys = {
            k for k in mapping if k in existing_normalized and existing_normalized[k] != mapping[k]
        }

        if not new_keys and not changed_keys:
            # Still save if existing keys need normalization:
            # - Non-string keys (ints from YAML)
            # - Duplicate keys after normalization (both 4 and '4' exist)
            # - Case/whitespace differences
            needs_normalization = (
                len(existing) != len(existing_normalized)  # duplicates collapsed
                or any(not isinstance(k, str) or k != normalize_key(k) for k in existing)
            )
            if not needs_normalization:
                console.print("  No changes needed")
                results["no_changes"].append(dataset_id)
                continue
            console.print("  Normalizing existing keys to lowercase strings")

        if new_keys or changed_keys:
            console.print("  Changes:")
            for k in sorted(new_keys):
                console.print(f"    [green]+ {k} -> {mapping[k]}[/green]")
            for k in sorted(changed_keys):
                console.print(
                    f"    [yellow]~ {k}: {existing_normalized[k]} -> {mapping[k]}[/yellow]"
                )

        if dry_run:
            console.print("  [dim](dry run - not saving)[/dim]")
        else:
            # Merge and ensure all keys are normalized lowercase strings
            merged = {**existing_normalized, **mapping}
            config.fetch.class_mapping = merged
            config_path = get_datasets_dir() / f"{dataset_id}.yaml"
            save_dataset_config(config, config_path)
            console.print(f"  [green]Saved to {config_path}[/green]")

        results["updated"].append(dataset_id)

    # Summary
    console.print("\n[bold]Summary[/bold]")
    console.print(f"  Updated: {len(results['updated'])} datasets")
    console.print(f"  No data: {len(results['no_data'])} datasets")
    console.print(f"  No changes: {len(results['no_changes'])} datasets")

    skipped = [w for w in all_warnings if w["type"] == "tier_change_skipped"]
    if skipped:
        console.print(f"\n[red]⛔ {len(skipped)} tier-changing mappings SKIPPED[/red]")

    if dry_run and results["updated"]:
        console.print("\n[dim]Run without --dry-run to apply changes.[/dim]")


@class_app.command("suggest-mapping")
def suggest_class_mapping(
    target_file: Path = typer.Argument(
        ...,
        help="Target parquet file to analyze",
    ),
    class_column: str = typer.Option(
        "class",
        "--class-column",
        help="Column containing source classification",
    ),
    name_column: str = typer.Option(
        "names",
        "--name-column",
        help="Column containing road names",
    ),
    output: Path = typer.Option(
        None,
        "-o",
        "--output",
        help="Output JSON file for mapping suggestions",
    ),
):
    """Suggest initial class mappings for a new dataset using heuristics.

    Uses keyword matching and name patterns to suggest Overture class mappings
    for datasets without labels. These are initial suggestions that should be
    refined after labeling.

    Examples:
        # Analyze a new dataset
        matcher class suggest-mapping data/raw/new_city_roads.parquet

        # Specify different column names
        matcher class suggest-mapping data.parquet --class-column road_type

        # Save to JSON
        matcher class suggest-mapping data.parquet -o suggestions.json
    """
    import geopandas as gpd

    from ..classification.predictor import OVERTURE_TIERS, SOURCE_CLASS_KEYWORDS

    console.print(f"Analyzing {target_file}...")

    # Load target file
    gdf = gpd.read_parquet(target_file)
    console.print(f"  Loaded {len(gdf)} features")

    if class_column not in gdf.columns:
        console.print(f"[red]Error: Column '{class_column}' not found in {target_file}[/red]")
        console.print(f"Available columns: {list(gdf.columns)}")
        raise typer.Exit(1)

    # Get unique source classes
    source_classes = gdf[class_column].dropna().astype(str).str.lower().str.strip()
    unique_classes = source_classes.value_counts()

    console.print("\n[bold]Source class distribution:[/bold]")
    for cls, count in unique_classes.head(20).items():
        console.print(f"  {cls:30} {count:6} ({count / len(source_classes) * 100:.1f}%)")

    # Suggest mappings using keyword matching
    suggestions = {}
    confidence = {}

    for src_class in unique_classes.index:
        src_lower = src_class.lower()
        best_match = None
        best_score = 0

        for overture_class, keywords in SOURCE_CLASS_KEYWORDS.items():
            # Check for exact match first
            if src_lower == overture_class:
                best_match = overture_class
                best_score = 1.0
                break

            # Check for keyword matches
            for keyword in keywords:
                if keyword in src_lower or src_lower in keyword:
                    # Partial match - score based on length overlap
                    score = len(keyword) / max(len(src_lower), len(keyword))
                    if score > best_score:
                        best_match = overture_class
                        best_score = score

        if best_match and best_score >= 0.3:
            suggestions[src_class] = best_match
            confidence[src_class] = best_score

    console.print(f"\n[bold]Suggested mappings ({len(suggestions)} classes):[/bold]")

    # Sort by confidence
    sorted_suggestions = sorted(
        suggestions.items(), key=lambda x: confidence.get(x[0], 0), reverse=True
    )

    for src, tgt in sorted_suggestions:
        conf = confidence.get(src, 0)
        tier = OVERTURE_TIERS.get(tgt, "?")
        count = unique_classes.get(src, 0)
        conf_color = "green" if conf >= 0.8 else "yellow" if conf >= 0.5 else "red"
        console.print(
            f"  {src:25} -> {tgt:15} ({tier:10}) [{conf_color}]conf={conf:.0%}[/{conf_color}] (n={count})"
        )

    # Show unmapped classes
    unmapped = set(unique_classes.index) - set(suggestions.keys())
    if unmapped:
        console.print(f"\n[yellow]Unmapped classes ({len(unmapped)}):[/yellow]")
        for cls in sorted(unmapped)[:10]:
            count = unique_classes.get(cls, 0)
            console.print(f"  {cls:25} (n={count})")
        if len(unmapped) > 10:
            console.print(f"  ... and {len(unmapped) - 10} more")

    # Output to JSON if requested
    if output:
        result = {
            "source_file": str(target_file),
            "class_column": class_column,
            "total_features": len(gdf),
            "unique_classes": len(unique_classes),
            "suggestions": suggestions,
            "confidence": confidence,
            "unmapped": list(unmapped),
            "class_distribution": unique_classes.to_dict(),
        }
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        console.print(f"\n[green]Saved suggestions to {output}[/green]")

    # Show YAML snippet
    console.print("\n[bold]YAML snippet for dataset config:[/bold]")
    console.print("fetch:")
    console.print(f"  class_column: {class_column}")
    console.print("  class_mapping:")
    for src, tgt in sorted_suggestions[:10]:
        console.print(f"    {src}: {tgt}")
    if len(sorted_suggestions) > 10:
        console.print(f"    # ... and {len(sorted_suggestions) - 10} more")
