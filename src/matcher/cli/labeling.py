"""Labeling commands: label UI, agent batch generation, backfill."""

from pathlib import Path

import typer

from ._app import app, console


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
    import os
    import subprocess
    import sys

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


@app.command("generate-agent-batch")
def generate_agent_batch(
    dataset: str = typer.Argument(..., help="Target dataset name (e.g., 'us_boston_streets')"),
    n_candidates: int = typer.Option(
        100,
        "--n-candidates",
        "-n",
        help="Number of candidates to sample",
    ),
    output_dir: Path = typer.Option(
        Path("agent_labels"),
        "--output",
        "-o",
        help="Output directory for agent labeling batches",
    ),
    reference: Path = typer.Option(
        Path("data/raw/overture_segments.parquet"),
        "--reference",
        "-r",
        help="Reference segments (Overture)",
    ),
    target: Path | None = typer.Option(
        None,
        "--target",
        "-t",
        help="Target segments (inferred from dataset name if not provided)",
    ),
    model: Path | None = typer.Option(
        None,
        "--model",
        "-m",
        help="ML model for confidence scoring (uses rules if not provided)",
    ),
    no_satellite: bool = typer.Option(
        False,
        "--no-satellite",
        help="Skip satellite imagery (faster, geometry-only images)",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
):
    """Generate NEW candidates for AI agent labeling.

    Samples diverse candidates across confidence ranges from unlabeled pairs
    and creates packages with metadata YAML and images for each candidate.
    Use this to expand training data with agent-labeled examples.

    Note: To test agent accuracy against existing human labels, use
    'generate-agent-test-batch' instead.

    Examples:
        # Generate 100 candidates for us_boston_streets
        matcher generate-agent-batch us_boston_streets

        # Generate 50 candidates with custom paths
        matcher generate-agent-batch us_boston_streets -n 50 \\
            -r data/raw/us_boston_overture_segments.parquet \\
            -t data/raw/us_boston_streets.parquet \\
            -o agent_labels

        # Use ML model for confidence scoring
        matcher generate-agent-batch us_boston_streets \\
            -m data/models/matcher_model_combined.joblib
    """
    from ..agent_labeling import SamplingConfig, sample_candidates
    from ..agent_labeling.context_generator import generate_batch

    # Infer target path if not provided
    if target is None:
        target = Path(f"data/raw/{dataset}.parquet")

    # Validate paths
    if not reference.exists():
        console.print(f"[red]Error: Reference file not found: {reference}[/red]")
        raise typer.Exit(1)

    if not target.exists():
        console.print(f"[red]Error: Target file not found: {target}[/red]")
        console.print("[yellow]Hint: Specify target path with --target[/yellow]")
        raise typer.Exit(1)

    if model and not model.exists():
        console.print(
            f"[yellow]Warning: Model not found: {model}, using rule-based scoring[/yellow]"
        )
        model = None

    console.print("[blue]Generating agent labeling batch...[/blue]")
    console.print(f"  Dataset: {dataset}")
    console.print(f"  Reference: {reference}")
    console.print(f"  Target: {target}")
    console.print(f"  Candidates: {n_candidates}")
    console.print(f"  Output: {output_dir}")

    # Sample candidates
    config = SamplingConfig(
        n_candidates=n_candidates,
        seed=seed,
    )

    candidates = sample_candidates(
        reference_path=reference,
        target_path=target,
        config=config,
        dataset_name=dataset,
        model_path=model,
    )

    if not candidates:
        console.print("[red]Error: No candidates generated[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Sampled {len(candidates)} candidates[/green]")

    # Generate batch
    batch_dir = generate_batch(
        candidates=candidates,
        output_dir=output_dir,
        dataset_name=dataset,
        fetch_satellite=not no_satellite,
        config_info={
            "n_candidates": n_candidates,
            "seed": seed,
            "reference": str(reference),
            "target": str(target),
            "model": str(model) if model else None,
        },
    )

    console.print()
    console.print(f"[green]Batch generated at {batch_dir}[/green]")
    console.print()
    console.print("Next steps:")
    console.print(f"  1. Review candidates in {batch_dir / 'candidates'}")
    console.print("  2. Have agents label candidates")
    console.print(f"  3. Import labels: matcher import-agent-labels {batch_dir} --agent-id <id>")


@app.command("import-agent-labels")
def import_agent_labels(
    batch_dir: Path = typer.Argument(..., help="Batch directory"),
    agent_id: str = typer.Option(
        ...,
        "--agent-id",
        "-a",
        help="Agent identifier (e.g., 'claude', 'gpt4', 'human')",
    ),
    labels_file: Path = typer.Option(
        ...,
        "--labels",
        "-l",
        help="Path to labels CSV file",
    ),
):
    """Import agent labels from a CSV file.

    The CSV must have columns: ref_id, target_id, label
    Optional columns: confidence, reasoning

    Examples:
        # Import Claude's labels
        matcher import-agent-labels agent_labels/batches/batch_2026-01-18_001 \\
            --agent-id claude --labels claude_labels.csv

        # Import with confidence and reasoning
        matcher import-agent-labels agent_labels/batches/batch_* \\
            -a gpt4 -l gpt4_labels.csv
    """
    from ..agent_labeling.agent_store import import_labels_csv

    # Validate paths
    if not batch_dir.exists():
        console.print(f"[red]Error: Batch directory not found: {batch_dir}[/red]")
        raise typer.Exit(1)

    if not labels_file.exists():
        console.print(f"[red]Error: Labels file not found: {labels_file}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Importing labels for agent '{agent_id}'...[/blue]")
    console.print(f"  Batch: {batch_dir}")
    console.print(f"  Labels: {labels_file}")

    count = import_labels_csv(batch_dir, agent_id, labels_file)

    console.print(f"[green]Imported {count} labels[/green]")


@app.command("agent-consensus")
def agent_consensus(
    batch_dir: Path = typer.Argument(..., help="Batch directory"),
    show_disagreements: bool = typer.Option(
        False,
        "--disagreements",
        "-d",
        help="Show only disagreements between agents",
    ),
    min_agents: int = typer.Option(
        2,
        "--min-agents",
        help="Minimum agents required for consensus",
    ),
):
    """Analyze agent consensus and disagreements.

    Shows agreement statistics across multiple agents and identifies
    candidates where agents disagree for human review.

    Examples:
        # Show consensus summary
        matcher agent-consensus agent_labels/batches/batch_2026-01-18_001

        # Show disagreements only
        matcher agent-consensus agent_labels/batches/batch_* --disagreements
    """
    from ..agent_labeling.agent_store import AgentLabelStore

    if not batch_dir.exists():
        console.print(f"[red]Error: Batch directory not found: {batch_dir}[/red]")
        raise typer.Exit(1)

    # List agents
    agents = AgentLabelStore.list_agents(batch_dir)
    if not agents:
        console.print("[yellow]No agent labels found in batch[/yellow]")
        return

    console.print(f"[blue]Agents who have labeled: {', '.join(agents)}[/blue]")
    console.print()

    # Show per-agent stats
    for agent_id in agents:
        store = AgentLabelStore(batch_dir, agent_id)
        stats = store.get_stats()
        console.print(f"  {agent_id}: {stats['total']} labels")
        console.print(
            f"    match: {stats['match']}, no_match: {stats['no_match']}, unsure: {stats['unsure']}"
        )

    console.print()

    if show_disagreements:
        # Show disagreements
        disagreements = AgentLabelStore.find_disagreements(batch_dir)
        if len(disagreements) == 0:
            console.print("[green]No disagreements found![/green]")
            return

        console.print(f"[yellow]Found {len(disagreements)} disagreements:[/yellow]")
        for _, row in disagreements.iterrows():
            console.print(f"  {row['ref_id']} <-> {row['target_id']}")
            console.print(f"    Labels: {row['labels']}")
            console.print(f"    Agreement: {row['agreement_ratio']:.0%}")
    else:
        # Show consensus
        consensus = AgentLabelStore.compute_consensus(batch_dir, min_agents)
        if len(consensus) == 0:
            console.print(f"[yellow]No candidates have >= {min_agents} agent labels[/yellow]")
            return

        console.print(f"[green]Consensus on {len(consensus)} candidates:[/green]")

        # Summary by consensus label
        label_counts = consensus["consensus_label"].value_counts()
        for label_val, count in label_counts.items():
            console.print(f"  {label_val}: {count}")

        # Agreement distribution
        mean_agreement = consensus["agreement_ratio"].mean()
        console.print(f"\n  Mean agreement: {mean_agreement:.0%}")

        # Count perfect agreement
        perfect = (consensus["agreement_ratio"] == 1.0).sum()
        console.print(f"  Perfect agreement: {perfect}/{len(consensus)}")


@app.command("generate-agent-test-batch")
def generate_agent_test_batch(
    n_samples: int = typer.Option(
        100,
        "--n-samples",
        "-n",
        help="Number of labeled pairs to sample for testing",
    ),
    output_dir: Path = typer.Option(
        Path("agent_labels"),
        "--output",
        "-o",
        help="Output directory for agent labeling batches",
    ),
    labels_dir: Path = typer.Option(
        Path("labels"),
        "--labels",
        "-l",
        help="Directory containing human labels (Hive-partitioned)",
    ),
    reference: Path = typer.Option(
        Path("data/raw/overture_segments.parquet"),
        "--reference",
        "-r",
        help="Reference segments (Overture)",
    ),
    datasets: list[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Datasets to include (can specify multiple). If not specified, uses all.",
    ),
    labeler: str | None = typer.Option(
        None,
        "--labeler",
        help="Filter by labeler name",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
    no_satellite: bool = typer.Option(
        False,
        "--no-satellite",
        help="Skip satellite imagery (faster, geometry-only images)",
    ),
):
    """Generate test batch from EXISTING human labels for agent accuracy testing.

    Samples from existing human-labeled pairs so you can measure agent agreement
    with human ground truth. Includes the ground truth labels in the output.

    Note: To generate NEW unlabeled candidates for agent labeling, use
    'generate-agent-batch' instead.

    Examples:
        # Generate 200 samples across all datasets
        matcher generate-agent-test-batch -n 200

        # Specific datasets only
        matcher generate-agent-test-batch -n 100 -d us_boston_streets -d us_boston_bike_network

        # Filter by labeler
        matcher generate-agent-test-batch -n 50 --labeler brad
    """
    from datetime import UTC, datetime

    import geopandas as gpd
    import pandas as pd

    from ..agent_labeling.context_generator import write_candidate_package
    from ..agent_labeling.sampler import SampledCandidate

    # Load human labels
    if not labels_dir.exists():
        console.print(f"[red]Error: Labels directory not found: {labels_dir}[/red]")
        raise typer.Exit(1)

    # Find all label files
    label_files = list(labels_dir.glob("dataset=*/data.csv"))
    if not label_files:
        console.print(f"[red]Error: No label files found in {labels_dir}[/red]")
        raise typer.Exit(1)

    # Load and combine labels
    all_labels = []
    for f in label_files:
        dataset = f.parent.name.replace("dataset=", "")
        if datasets and dataset not in datasets:
            continue
        df = pd.read_csv(f)
        df["dataset"] = dataset
        all_labels.append(df)

    if not all_labels:
        console.print("[red]Error: No labels found for specified datasets[/red]")
        raise typer.Exit(1)

    labels_df = pd.concat(all_labels, ignore_index=True)

    # Filter by labeler if specified
    if labeler and "labeler" in labels_df.columns:
        labels_df = labels_df[labels_df["labeler"].str.lower() == labeler.lower()]
        if len(labels_df) == 0:
            console.print(f"[red]Error: No labels found for labeler '{labeler}'[/red]")
            raise typer.Exit(1)

    # Filter to match/no_match only (exclude unsure for cleaner testing)
    if "label" in labels_df.columns:
        labels_df = labels_df[labels_df["label"].isin(["match", "no_match"])]
        if len(labels_df) == 0:
            console.print("[red]Error: No match/no_match labels found after filtering[/red]")
            raise typer.Exit(1)

    console.print(f"[blue]Found {len(labels_df)} labeled pairs[/blue]")

    # Stratified sample across datasets
    import numpy as np

    rng = np.random.default_rng(seed)

    sampled = []
    for dataset in labels_df["dataset"].unique():
        dataset_df = labels_df[labels_df["dataset"] == dataset]
        n_dataset = max(1, int(n_samples * len(dataset_df) / len(labels_df)))
        n_dataset = min(n_dataset, len(dataset_df))
        indices = rng.choice(len(dataset_df), size=n_dataset, replace=False)
        sampled.append(dataset_df.iloc[indices])

    sampled_df = pd.concat(sampled, ignore_index=True)
    console.print(f"[blue]Sampled {len(sampled_df)} pairs for testing[/blue]")

    # Load reference data
    if not reference.exists():
        console.print(f"[red]Error: Reference file not found: {reference}[/red]")
        raise typer.Exit(1)

    ref_gdf = gpd.read_parquet(reference)
    ref_lookup = ref_gdf.set_index("id")

    # Load target datasets - auto-discover paths based on dataset name
    target_gdfs = {}
    data_dir = Path("data/raw")

    for dataset in sampled_df["dataset"].unique():
        # Try different naming patterns
        candidates = [
            data_dir / f"{dataset}.parquet",  # e.g., us_boston_streets.parquet
            data_dir
            / f"{dataset}_segments.parquet",  # e.g., us_boston_streets_osm_segments.parquet
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path:
            target_gdfs[dataset] = gpd.read_parquet(path).set_index("id")
        else:
            console.print(f"[yellow]Warning: No data file for {dataset}[/yellow]")

    # Generate batch
    batch_id = f"test_batch_{datetime.now(UTC).strftime('%Y-%m-%d_%H%M%S')}"
    batch_dir = output_dir / "batches" / batch_id
    candidates_dir = batch_dir / "candidates"
    labels_out_dir = batch_dir / "labels"

    batch_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[blue]Generating batch: {batch_id}[/blue]")

    # Build SampledCandidate objects and write packages
    candidates_list = []
    for _, row in sampled_df.iterrows():
        ref_id = row["gers_id"]
        target_id = row["target_id"]
        dataset = row["dataset"]

        if dataset not in target_gdfs:
            continue

        try:
            ref_row = ref_lookup.loc[ref_id]
            target_row = target_gdfs[dataset].loc[target_id]
        except KeyError:
            continue

        # Extract features from row (if available)
        feature_cols = [
            "hausdorff_distance",
            "buffer_iou",
            "heading_delta",
            "length_ratio",
            "name_levenshtein",
            "name_jaro_winkler",
            "class_similarity",
            "centroid_distance",
            "overlap_ratio",
            "mean_hausdorff_distance",
            "degree_match_score",
            "dead_end_match",
            "intersection_match",
        ]
        features = {col: row.get(col, 0.0) for col in feature_cols if col in row.index}

        candidate = SampledCandidate(
            ref_id=str(ref_id),
            target_id=str(target_id),
            ref_geometry=ref_row.geometry,
            target_geometry=target_row.geometry,
            ref_name=ref_row.get("names") if hasattr(ref_row, "get") else None,
            target_name=target_row.get("names") if hasattr(target_row, "get") else None,
            ref_class=ref_row.get("class") if hasattr(ref_row, "get") else None,
            target_class=target_row.get("class") if hasattr(target_row, "get") else None,
            ml_confidence=row.get("original_confidence", 0.5),
            ml_decision=row.get("original_decision", "review"),
            features=features,
            dataset=dataset,
            confidence_bucket="ground_truth",
        )
        candidates_list.append(candidate)

        # Write candidate package with images
        write_candidate_package(
            output_dir=candidates_dir,
            candidate=candidate,
            batch_id=batch_id,
            fetch_satellite=not no_satellite,
        )

        if (len(candidates_list)) % 20 == 0:
            console.print(f"  Progress: {len(candidates_list)}/{len(sampled_df)}")

    # Write ground truth labels
    ground_truth_path = labels_out_dir / "ground_truth" / "data.csv"
    ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df[["gers_id", "target_id", "label", "dataset"]].rename(
        columns={"gers_id": "ref_id"}
    ).to_csv(ground_truth_path, index=False)

    # Write manifest with ground truth info
    import yaml

    manifest = {
        "batch_id": batch_id,
        "batch_type": "agent_test",
        "created_at": datetime.now(UTC).isoformat(),
        "total_candidates": len(candidates_list),
        "datasets": list(sampled_df["dataset"].unique()),
        "labeler_filter": labeler,
        "ground_truth": {
            "file": "labels/ground_truth/data.csv",
            "total": len(sampled_df),
            "by_label": sampled_df["label"].value_counts().to_dict(),
            "by_dataset": sampled_df["dataset"].value_counts().to_dict(),
        },
        "candidates": [
            {"ref_id": c.ref_id, "target_id": c.target_id, "dataset": c.dataset}
            for c in candidates_list
        ],
    }
    (batch_dir / "manifest.yaml").write_text(
        yaml.dump(manifest, default_flow_style=False, sort_keys=False)
    )

    console.print()
    console.print(f"[green]Batch generated at {batch_dir}[/green]")
    console.print(f"  Candidates: {len(candidates_list)}")
    console.print(f"  Ground truth: {ground_truth_path}")
    console.print()
    console.print("Next steps:")
    console.print("  1. Have agents label candidates in candidates/")
    console.print(
        f"  2. Import labels: matcher import-agent-labels {batch_dir} -a <agent-id> -l <labels.csv>"
    )
    console.print(f"  3. Compare: matcher agent-consensus {batch_dir}")


@app.command("backfill-labels")
def backfill_labels(
    labels_dir: Path = typer.Option(
        Path("labels"),
        "--labels",
        "-l",
        help="Labels directory (Hive-partitioned CSV format)",
    ),
    overture: Path = typer.Option(
        None,
        "--overture",
        "-r",
        help="Path to Overture segments parquet. If not specified, looks for "
        "{dataset}_overture_segments.parquet in the data directory.",
    ),
    data_dir: Path = typer.Option(
        Path("data/raw"),
        "--data-dir",
        "-d",
        help="Directory containing target dataset parquet files",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Compute features but don't write to disk",
    ),
    skip_missing: bool = typer.Option(
        False,
        "--skip-missing",
        help="Skip datasets with missing data files instead of failing",
    ),
    report: bool = typer.Option(
        False,
        "--report",
        help="Report what can/cannot be backfilled without modifying any files",
    ),
    drop_orphaned: bool = typer.Option(
        False,
        "--drop-orphaned",
        help="Remove labels where IDs are not found in current data (orphaned labels)",
    ),
):
    """Recompute features for all existing labels.

    This command is needed after changes to feature computation logic to ensure
    existing training labels have consistent features. It recomputes:
    - Alignment fractions (where segments overlap)
    - All topology features (using alignment-aware computation)
    - Endpoint proximity features
    - Graphlet similarity features
    - All other geometric/semantic features
    - Data version tracking columns (ref_data_version, target_data_version, feature_version)

    The command preserves the label (match/no_match) but updates all feature
    columns, alignment fractions, and version tracking.

    Labels are considered "orphaned" when their IDs are not found in the current
    data files (e.g., if data was re-fetched with different IDs).

    By default, the command will FAIL if any dataset is missing required data
    files. Use --skip-missing to skip those datasets instead.

    Examples:
        # Backfill all labels (fails if any data is missing)
        matcher backfill-labels

        # Skip datasets with missing data
        matcher backfill-labels --skip-missing

        # Report what can/cannot be backfilled (no changes)
        matcher backfill-labels --report

        # Dry run (compute but don't write)
        matcher backfill-labels --dry-run

        # Drop labels that can't be backfilled (orphaned IDs)
        matcher backfill-labels --drop-orphaned
    """
    from ..labeling.label_store import backfill_features

    if not labels_dir.exists():
        console.print(f"[red]Labels directory not found: {labels_dir}[/red]")
        raise typer.Exit(1)

    if overture is not None and not overture.exists():
        console.print(f"[red]Overture file not found: {overture}[/red]")
        raise typer.Exit(1)

    if dry_run:
        console.print("[yellow]Dry run mode - no files will be modified[/yellow]")
    if report:
        console.print("[yellow]Report mode - no files will be modified[/yellow]")
    if drop_orphaned:
        console.print(
            "[yellow]Drop orphaned mode - labels with missing IDs will be removed[/yellow]"
        )

    console.print("[blue]Starting feature backfill...[/blue]")
    console.print(f"  Labels: {labels_dir}")
    console.print(f"  Overture: {overture or '(auto-discover per dataset)'}")
    console.print(f"  Data dir: {data_dir}")
    if skip_missing:
        console.print(
            "[yellow]  Skip missing: enabled (will skip datasets with missing data)[/yellow]"
        )

    try:
        results = backfill_features(
            labels_dir=labels_dir,
            overture_path=overture,
            data_dir=data_dir,
            dry_run=dry_run,
            skip_missing=skip_missing,
            report_only=report,
            drop_orphaned=drop_orphaned,
        )

        console.print()
        if report:
            console.print("[green]Backfill report complete![/green]")
        else:
            console.print("[green]Backfill complete![/green]")

        console.print("Results by dataset:")
        total_updated = 0
        total_orphaned = 0
        for dataset, stats in sorted(results.items()):
            updated = stats.get("updated", 0)
            orphaned = stats.get("orphaned", 0)
            total = stats.get("total", 0)
            skipped = stats.get("skipped")
            dropped = stats.get("dropped", 0)

            total_updated += updated
            total_orphaned += orphaned

            if skipped:
                console.print(f"  {dataset}: [yellow]skipped ({skipped})[/yellow]")
            elif orphaned > 0:
                orphan_status = f"[red]{orphaned} orphaned[/red]"
                if dropped > 0:
                    orphan_status += f" [yellow]({dropped} dropped)[/yellow]"
                console.print(f"  {dataset}: {updated}/{total} updated, {orphan_status}")
            else:
                console.print(f"  {dataset}: {updated}/{total} updated")

        console.print(f"\n  Total: {total_updated} updated, {total_orphaned} orphaned")

        if dry_run:
            console.print(
                "\n[yellow]Dry run - no files were modified. "
                "Remove --dry-run to apply changes.[/yellow]"
            )
        if report:
            console.print(
                "\n[yellow]Report mode - no files were modified. "
                "Remove --report to apply changes.[/yellow]"
            )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e
