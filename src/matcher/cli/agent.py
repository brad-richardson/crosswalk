"""AI agent labeling commands."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import typer

from .utils import console

# Create agent group
agent_app = typer.Typer(
    name="agent",
    help="AI agent labeling commands",
    no_args_is_help=True,
)


@agent_app.command("batch")
def generate_agent_batch(
    dataset: str = typer.Argument(..., help="Target dataset name (e.g., 'us_boston_streets')"),
    n_candidates: int = typer.Option(
        100,
        "--n-candidates",
        "-n",
        help="Number of candidates to sample",
    ),
    output_dir: Path = typer.Option(
        Path("data/agents"),
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
        True,
        "--no-satellite/--satellite",
        help="Skip satellite imagery (default: geometry-only, which performs equally well and is faster)",
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
    'matcher agent test-batch' instead.

    Examples:
        # Generate 100 candidates for us_boston_streets
        matcher agent batch us_boston_streets

        # Generate 50 candidates with custom paths
        matcher agent batch us_boston_streets -n 50 \\
            -r data/raw/us_boston_overture_segments.parquet \\
            -t data/raw/us_boston_streets.parquet \\
            -o data/agents

        # Use ML model for confidence scoring
        matcher agent batch us_boston_streets \\
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
    console.print(f"  3. Import labels: matcher agent import {batch_dir} --agent-id <id>")


@agent_app.command("test-batch")
def generate_agent_test_batch(
    n_samples: int = typer.Option(
        100,
        "--n-samples",
        "-n",
        help="Number of labeled pairs to sample for testing",
    ),
    output_dir: Path = typer.Option(
        Path("data/agents"),
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
        True,
        "--no-satellite/--satellite",
        help="Skip satellite imagery (default: geometry-only, which performs equally well and is faster)",
    ),
):
    """Generate test batch from EXISTING human labels for agent accuracy testing.

    Samples from existing human-labeled pairs so you can measure agent agreement
    with human ground truth. Includes the ground truth labels in the output.

    Note: To generate NEW unlabeled candidates for agent labeling, use
    'matcher agent batch' instead.

    Examples:
        # Generate 200 samples across all datasets
        matcher agent test-batch -n 200

        # Specific datasets only
        matcher agent test-batch -n 100 -d us_boston_streets -d us_boston_bike_network

        # Filter by labeler
        matcher agent test-batch -n 50 --labeler brad
    """
    import geopandas as gpd
    import numpy as np
    import yaml

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
        f"  2. Import labels: matcher agent import {batch_dir} -a <agent-id> -l <labels.csv>"
    )
    console.print(f"  3. Compare: matcher agent consensus {batch_dir}")


@agent_app.command("sweep")
def generate_basemap_sweep(
    datasets: list[str] = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Datasets to include (can specify multiple)",
    ),
    n_per_dataset: int = typer.Option(
        4,
        "--n",
        "-n",
        help="Number of candidates per dataset (half match, half no_match)",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
    output_dir: Path = typer.Option(
        Path("data/agents"),
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
    geom_dir: Path = typer.Option(
        Path("labels/data"),
        "--geometries",
        "-g",
        help="Directory containing geometry data (GeoParquet)",
    ),
    data_dir: Path = typer.Option(
        Path("data/raw"),
        "--data-dir",
        help="Directory containing data files (for Overture segments)",
    ),
):
    """Generate basemap sweep batch for comparing AI agent accuracy across visualization variants.

    Samples stratified match/no_match candidates from existing human labels,
    then generates 6 image variants per candidate:
      - geometry_only.png: white background with geometry lines
      - carto_positron.png: CartoDB light map basemap
      - road_context.png: nearby Overture roads as gray context lines
      - road_context.svg: same as above in SVG format
      - subline_geometry_only.png: faded dashed full segments + solid bright aligned sublines
      - subline_road_context.png: same + gray dashed context roads

    Examples:
        matcher agent sweep \\
            -d us_boston_streets -d us_boston_sidewalks \\
            -d us_boston_bike_network -d us_frisco_trails \\
            -n 25 --seed 42
    """
    import geopandas as gpd
    import numpy as np
    import yaml
    from shapely import wkt
    from shapely.geometry import box as shapely_box

    from ..agent_labeling.context_generator import write_candidate_sweep_package
    from ..agent_labeling.sampler import SampledCandidate
    from ..filenames import find_overture_segments

    # Validate directories
    if not labels_dir.exists():
        console.print(f"[red]Error: Labels directory not found: {labels_dir}[/red]")
        raise typer.Exit(1)

    if not geom_dir.exists():
        console.print(f"[red]Error: Geometries directory not found: {geom_dir}[/red]")
        raise typer.Exit(1)

    rng = np.random.default_rng(seed)

    # Sweep variants
    variants = [
        {"basemap": "geometry_only", "format": "png"},
        {"basemap": "carto_positron", "format": "png"},
        {"basemap": "road_context", "format": "png"},
        {"basemap": "road_context", "format": "svg"},
        {"basemap": "subline_geometry_only", "format": "png"},
        {"basemap": "subline_road_context", "format": "png"},
    ]

    # Load labels and geometries per dataset, sample candidates
    all_candidates = []
    all_ground_truth = []

    for dataset in datasets:
        label_file = labels_dir / f"dataset={dataset}" / "data.csv"
        geom_file = geom_dir / f"dataset={dataset}" / "data.csv"

        if not label_file.exists():
            console.print(f"[yellow]Warning: No labels for {dataset}, skipping[/yellow]")
            continue
        if not geom_file.exists():
            console.print(f"[yellow]Warning: No geometries for {dataset}, skipping[/yellow]")
            continue

        # Load labels
        labels_df = pd.read_csv(label_file)
        labels_df = labels_df[labels_df["label"].isin(["match", "no_match"])]

        if len(labels_df) == 0:
            console.print(f"[yellow]Warning: No match/no_match labels for {dataset}[/yellow]")
            continue

        # Deduplicate by (gers_id, target_id) - take first occurrence
        labels_df = labels_df.drop_duplicates(subset=["gers_id", "target_id"], keep="first")

        # Stratified sample: n/2 match + n/2 no_match
        n_match = n_per_dataset // 2
        n_no_match = n_per_dataset - n_match

        match_df = labels_df[labels_df["label"] == "match"]
        no_match_df = labels_df[labels_df["label"] == "no_match"]

        n_match = min(n_match, len(match_df))
        n_no_match = min(n_no_match, len(no_match_df))

        if n_match == 0 and n_no_match == 0:
            console.print(f"[yellow]Warning: Insufficient labels for {dataset}[/yellow]")
            continue

        sampled_match = match_df.iloc[rng.choice(len(match_df), size=n_match, replace=False)]
        sampled_no_match = no_match_df.iloc[
            rng.choice(len(no_match_df), size=n_no_match, replace=False)
        ]
        sampled = pd.concat([sampled_match, sampled_no_match], ignore_index=True)

        # Load geometries
        geom_df = pd.read_csv(geom_file)
        geom_lookup = {}
        for _, row in geom_df.iterrows():
            key = (str(row["gers_id"]), str(row["target_id"]))
            geom_lookup[key] = row

        # Load Overture segments for road context
        overture_path = find_overture_segments(data_dir, dataset)
        ref_gdf = None
        if overture_path and overture_path.exists():
            try:
                ref_gdf = gpd.read_parquet(overture_path)
                if "id" in ref_gdf.columns:
                    ref_gdf = ref_gdf.set_index("id")
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Could not load Overture segments for {dataset}: {e}[/yellow]"
                )

        console.print(f"[blue]{dataset}: sampled {len(sampled)} candidates[/blue]")

        for _, row in sampled.iterrows():
            ref_id = str(row["gers_id"])
            target_id = str(row["target_id"])
            key = (ref_id, target_id)

            if key not in geom_lookup:
                console.print(f"  [yellow]Skipping {ref_id}: geometry not found[/yellow]")
                continue

            geom_row = geom_lookup[key]
            try:
                ref_geom = wkt.loads(geom_row["ref_geometry_wkt"])
                target_geom = wkt.loads(geom_row["target_geometry_wkt"])
            except Exception:
                console.print(f"  [yellow]Skipping {ref_id}: invalid WKT geometry[/yellow]")
                continue

            # Find nearby roads for road_context variant
            context_roads = []
            if ref_gdf is not None:
                try:
                    combined = ref_geom.union(target_geom)
                    minx, miny, maxx, maxy = combined.bounds
                    # Expand bbox by ~50m in degrees (rough)
                    expand = 0.0005
                    expanded_bbox = (minx - expand, miny - expand, maxx + expand, maxy + expand)
                    bbox_poly = shapely_box(*expanded_bbox)

                    if hasattr(ref_gdf, "sindex"):
                        possible_idx = list(ref_gdf.sindex.intersection(expanded_bbox))
                        nearby = ref_gdf.iloc[possible_idx]
                    else:
                        nearby = ref_gdf[ref_gdf.geometry.intersects(bbox_poly)]

                    # Exclude the reference geometry itself
                    if ref_id in nearby.index:
                        nearby = nearby.drop(ref_id, errors="ignore")

                    context_roads = nearby.geometry.tolist()
                except Exception:
                    pass  # Non-critical, proceed without context roads

            # Extract features from label row
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

            # Parse names/classes from geometry attributes if available
            import json

            ref_name = None
            target_name = None
            ref_class = None
            target_class = None
            if "ref_attributes" in geom_row.index:
                try:
                    attrs = json.loads(geom_row["ref_attributes"])
                    ref_name = attrs.get("name")
                    ref_class = attrs.get("class")
                except Exception:
                    pass  # Malformed JSON attributes - continue with None values
            if "target_attributes" in geom_row.index:
                try:
                    attrs = json.loads(geom_row["target_attributes"])
                    target_name = attrs.get("name")
                    target_class = attrs.get("class")
                except Exception:
                    pass  # Malformed JSON attributes - continue with None values

            # Read alignment fractions from labels CSV (NaN → defaults)
            def _safe_frac(val, default):
                try:
                    f = float(val)
                    return default if pd.isna(f) else f
                except (TypeError, ValueError):
                    return default

            ref_start_frac = _safe_frac(row.get("ref_start_pct", None), 0.0)
            ref_end_frac = _safe_frac(row.get("ref_end_pct", None), 1.0)
            target_start_frac = _safe_frac(row.get("target_start_pct", None), 0.0)
            target_end_frac = _safe_frac(row.get("target_end_pct", None), 1.0)

            candidate = SampledCandidate(
                ref_id=ref_id,
                target_id=target_id,
                ref_geometry=ref_geom,
                target_geometry=target_geom,
                ref_name=ref_name,
                target_name=target_name,
                ref_class=ref_class,
                target_class=target_class,
                ml_confidence=row.get("original_confidence", 0.5),
                ml_decision=row.get("original_decision", "review"),
                features=features,
                dataset=dataset,
                confidence_bucket="ground_truth",
                ref_start_frac=ref_start_frac,
                ref_end_frac=ref_end_frac,
                target_start_frac=target_start_frac,
                target_end_frac=target_end_frac,
            )
            all_candidates.append((candidate, context_roads))
            all_ground_truth.append(
                {
                    "ref_id": ref_id,
                    "target_id": target_id,
                    "label": row["label"],
                    "dataset": dataset,
                }
            )

    if not all_candidates:
        console.print("[red]Error: No candidates generated[/red]")
        raise typer.Exit(1)

    # Create batch

    batch_id = f"sweep_{datetime.now(UTC).strftime('%Y-%m-%d_%H%M%S')}"
    batch_dir = output_dir / "batches" / batch_id
    candidates_dir = batch_dir / "candidates"
    labels_out_dir = batch_dir / "labels"

    batch_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[blue]Generating sweep batch: {batch_id}[/blue]")
    console.print(f"  Total candidates: {len(all_candidates)}")
    console.print(f"  Variants per candidate: {len(variants)}")

    for i, (candidate, ctx_roads) in enumerate(all_candidates):
        write_candidate_sweep_package(
            output_dir=candidates_dir,
            candidate=candidate,
            batch_id=batch_id,
            variants=variants,
            context_roads=ctx_roads,
        )
        if (i + 1) % 4 == 0:
            console.print(f"  Progress: {i + 1}/{len(all_candidates)}")

    # Write ground truth
    ground_truth_path = labels_out_dir / "ground_truth" / "data.csv"
    ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
    gt_df = pd.DataFrame(all_ground_truth)
    gt_df.to_csv(ground_truth_path, index=False)

    # Write manifest
    manifest = {
        "batch_id": batch_id,
        "batch_type": "basemap_sweep",
        "created_at": datetime.now(UTC).isoformat(),
        "total_candidates": len(all_candidates),
        "datasets": datasets,
        "n_per_dataset": n_per_dataset,
        "seed": seed,
        "variants": [f"{v['basemap']}.{v['format']}" for v in variants],
        "ground_truth": {
            "file": "labels/ground_truth/data.csv",
            "total": len(gt_df),
            "by_label": gt_df["label"].value_counts().to_dict(),
            "by_dataset": gt_df["dataset"].value_counts().to_dict(),
        },
        "candidates": [
            {"ref_id": c.ref_id, "target_id": c.target_id, "dataset": c.dataset}
            for c, _ in all_candidates
        ],
    }
    (batch_dir / "manifest.yaml").write_text(
        yaml.dump(manifest, default_flow_style=False, sort_keys=False)
    )

    console.print()
    console.print(f"[green]Sweep batch generated at {batch_dir}[/green]")
    console.print(f"  Candidates: {len(all_candidates)}")
    console.print(f"  Ground truth: {ground_truth_path}")
    console.print(f"  Variants: {', '.join(v['basemap'] + '.' + v['format'] for v in variants)}")
    console.print()
    console.print("Next steps:")
    console.print(f"  matcher agent run claude --batch {batch_dir} --variant geometry_only")
    console.print(f"  matcher agent eval-sweep {batch_dir}")


@agent_app.command("run")
def run_agent(
    batch_dir: Path = typer.Option(..., "--batch", "-b", help="Batch directory"),
    model: str = typer.Option("opus", "--model", "-m", help="Model variant (e.g. opus, sonnet)"),
    variant: str = typer.Option(..., "--variant", "-v", help="Image variant name"),
    limit: int = typer.Option(0, "--limit", "-l", help="Max candidates to process (0=no limit)"),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Start fresh, discard existing labels"
    ),
    few_shot: int = typer.Option(
        4, "--few-shot", "-f", help="Number of few-shot examples (0=none)"
    ),
    few_shot_source: Path | None = typer.Option(
        None, "--few-shot-source", help="Explicit batch to source few-shot examples from"
    ),
    timeout: int = typer.Option(300, "--timeout", help="Timeout per chunk in seconds"),
    chunk_size: int = typer.Option(25, "--chunk-size", help="Candidates per CLI invocation"),
):
    """Run Claude agent in batch mode on labeling candidates.

    Builds a prompt with few-shot examples from other batches, then invokes
    Claude Code CLI to process candidates in chunks. The agent reads images
    and metadata itself and writes results to a CSV file.

    Resumes by default - existing labels are skipped. Use --overwrite to start fresh.

    Examples:
        matcher agent run --batch data/agents/batches/sweep_2026-02-01_001051 \\
            --model opus --variant geometry_only --limit 5

        matcher agent run --batch data/agents/batches/sweep_2026-02-01_001051 \\
            --model sonnet --variant road_context --overwrite

        matcher agent run --batch data/agents/batches/sweep_2026-02-01_001051 \\
            --model opus --variant subline_road_context --few-shot 8 --chunk-size 20
    """
    from ..agent_labeling.runner import run_agent_batch

    run_agent_batch(
        model=model,
        variant=variant,
        batch_dir=batch_dir,
        limit=limit,
        overwrite=overwrite,
        n_few_shot=few_shot,
        few_shot_source=few_shot_source,
        timeout=timeout,
        chunk_size=chunk_size,
    )


@agent_app.command("import")
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
        matcher agent import data/agents/batches/batch_2026-01-18_001 \\
            --agent-id claude --labels claude_labels.csv

        # Import with confidence and reasoning
        matcher agent import data/agents/batches/batch_* \\
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


@agent_app.command("consensus")
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
        matcher agent consensus data/agents/batches/batch_2026-01-18_001

        # Show disagreements only
        matcher agent consensus data/agents/batches/batch_* --disagreements
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
        for label, count in label_counts.items():
            console.print(f"  {label}: {count}")

        # Agreement distribution
        mean_agreement = consensus["agreement_ratio"].mean()
        console.print(f"\n  Mean agreement: {mean_agreement:.0%}")

        # Count perfect agreement
        perfect = (consensus["agreement_ratio"] == 1.0).sum()
        console.print(f"  Perfect agreement: {perfect}/{len(consensus)}")


@agent_app.command("eval-sweep")
def eval_agent_sweep(
    batch_dir: Path = typer.Argument(..., help="Sweep batch directory"),
    show_reasoning: bool = typer.Option(
        False,
        "--reasoning",
        help="Show per-candidate reasoning for qualitative review",
    ),
):
    """Evaluate agent sweep results against ground truth.

    Discovers all agent label directories in the batch, joins with ground
    truth, and computes accuracy/precision/recall/F1 per variant and dataset.

    Examples:
        matcher agent eval-sweep data/agents/batches/sweep_2026-01-28_120000

        # Include reasoning text
        matcher agent eval-sweep data/agents/batches/sweep_2026-01-28_120000 --reasoning
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    if not batch_dir.exists():
        console.print(f"[red]Error: Batch directory not found: {batch_dir}[/red]")
        raise typer.Exit(1)

    # Load ground truth
    gt_path = batch_dir / "labels" / "ground_truth" / "data.csv"
    if not gt_path.exists():
        console.print(f"[red]Error: Ground truth not found: {gt_path}[/red]")
        raise typer.Exit(1)

    gt_df = pd.read_csv(gt_path, dtype=str)
    gt_df["key"] = gt_df["ref_id"] + "__" + gt_df["target_id"]
    console.print(f"[blue]Ground truth: {len(gt_df)} candidates[/blue]")
    console.print(f"  Labels: {gt_df['label'].value_counts().to_dict()}")
    console.print()

    # Discover agent label directories
    labels_base = batch_dir / "labels"
    agent_dirs = [
        d
        for d in labels_base.iterdir()
        if d.is_dir() and d.name != "ground_truth" and (d / "data.csv").exists()
    ]

    if not agent_dirs:
        console.print("[yellow]No agent labels found in batch[/yellow]")
        return

    # Evaluate each agent/variant
    results = []
    for agent_dir in sorted(agent_dirs):
        agent_name = agent_dir.name
        agent_df = pd.read_csv(agent_dir / "data.csv", dtype={"ref_id": str, "target_id": str})

        if "ref_id" not in agent_df.columns or "label" not in agent_df.columns:
            console.print(f"  [yellow]{agent_name}: invalid CSV format, skipping[/yellow]")
            continue

        agent_df["key"] = agent_df["ref_id"].astype(str) + "__" + agent_df["target_id"].astype(str)

        # Join with ground truth
        merged = gt_df.merge(agent_df[["key", "label"]], on="key", suffixes=("_gt", "_pred"))
        merged = merged[merged["label_pred"].isin(["match", "no_match"])]

        if len(merged) == 0:
            console.print(f"  [yellow]{agent_name}: no overlapping labels[/yellow]")
            continue

        y_true = (merged["label_gt"] == "match").astype(int)
        y_pred = (merged["label_pred"] == "match").astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append(
            {
                "agent_variant": agent_name,
                "n_evaluated": len(merged),
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        )

        console.print(f"[bold]{agent_name}[/bold]  (n={len(merged)})")
        console.print(
            f"  Accuracy: {acc:.3f}  Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}"
        )

        # Per-dataset breakdown
        for dataset in sorted(merged["dataset"].unique()):
            ds = merged[merged["dataset"] == dataset]
            ds_true = (ds["label_gt"] == "match").astype(int)
            ds_pred = (ds["label_pred"] == "match").astype(int)
            ds_acc = accuracy_score(ds_true, ds_pred)
            console.print(f"    {dataset}: {ds_acc:.3f} ({len(ds)} candidates)")

        console.print()

    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return

    # Summary comparison table
    console.print("[bold]Summary Comparison[/bold]")
    console.print(f"{'Variant':<40} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>4}")
    console.print("-" * 70)
    for r in results:
        console.print(
            f"{r['agent_variant']:<40} {r['accuracy']:>6.3f} {r['precision']:>6.3f} "
            f"{r['recall']:>6.3f} {r['f1']:>6.3f} {r['n_evaluated']:>4}"
        )

    # Show per-candidate reasoning if requested
    if show_reasoning:
        console.print()
        console.print("[bold]Per-Candidate Reasoning[/bold]")
        for agent_dir in sorted(agent_dirs):
            agent_name = agent_dir.name
            agent_df = pd.read_csv(agent_dir / "data.csv", dtype={"ref_id": str, "target_id": str})
            if "reasoning" not in agent_df.columns:
                continue

            console.print(f"\n[bold]{agent_name}[/bold]")
            agent_df["key"] = (
                agent_df["ref_id"].astype(str) + "__" + agent_df["target_id"].astype(str)
            )
            merged = gt_df.merge(
                agent_df[["key", "label", "reasoning"]], on="key", suffixes=("_gt", "_pred")
            )

            for _, row in merged.iterrows():
                correct = row["label_gt"] == row["label_pred"]
                status = "[green]CORRECT[/green]" if correct else "[red]WRONG[/red]"
                console.print(
                    f"  {row['ref_id']}: gt={row['label_gt']}, pred={row['label_pred']} {status}"
                )
                if pd.notna(row.get("reasoning")):
                    console.print(f"    Reasoning: {row['reasoning']}")


@agent_app.command("export")
def export_agent_labels(
    batches_dir: Path = typer.Option(
        Path("data/agents/batches"),
        "--batches",
        "-b",
        help="Directory containing agent batches",
    ),
    output_dir: Path = typer.Option(
        Path("labels/agent"),
        "--output",
        "-o",
        help="Output directory for consolidated labels (Hive-partitioned)",
    ),
    agent_filter: str | None = typer.Option(
        None,
        "--agent",
        "-a",
        help="Filter to specific agent (e.g., 'gemini-flash')",
    ),
    append: bool = typer.Option(
        True,
        "--append/--replace",
        help="Append to existing labels (default) or replace them",
    ),
):
    """Export agent labels from batches to tracked Hive-partitioned directory.

    Consolidates labels from data/agents/batches/*/labels/*/data.csv into
    labels/agent/dataset=X/data.csv with labeler field set to agent name.

    Examples:
        # Export all agent labels
        matcher agent export

        # Export only gemini-flash labels
        matcher agent export --agent gemini-flash

        # Append to existing labels
        matcher agent export --append
    """
    if not batches_dir.exists():
        console.print(f"[red]Error: Batches directory not found: {batches_dir}[/red]")
        raise typer.Exit(1)

    # Find all agent label files
    label_files = list(batches_dir.glob("*/labels/*/data.csv"))
    # Exclude ground_truth
    label_files = [f for f in label_files if "ground_truth" not in str(f)]

    if not label_files:
        console.print("[yellow]No agent labels found in batches[/yellow]")
        return

    console.print(f"[blue]Found {len(label_files)} agent label files[/blue]")

    # Collect all labels with metadata
    all_labels = []
    for label_file in label_files:
        agent_name = label_file.parent.name  # e.g., "gemini-flash"

        if agent_filter and agent_name != agent_filter:
            continue

        try:
            df = pd.read_csv(label_file, dtype={"ref_id": str, "target_id": str})
        except Exception as e:
            console.print(f"  [yellow]Skipping {label_file}: {e}[/yellow]")
            continue

        if "label" not in df.columns:
            console.print(f"  [yellow]Skipping {label_file}: no label column[/yellow]")
            continue

        # Add/override labeler column
        df["labeler"] = agent_name

        # Try to get dataset from the batch manifest
        batch_dir = label_file.parent.parent.parent
        manifest_path = batch_dir / "manifest.yaml"
        if manifest_path.exists():
            import yaml

            try:
                manifest = yaml.safe_load(manifest_path.read_text())
                # Get dataset from manifest root (batch-level) or fall back to unknown
                batch_dataset = manifest.get("dataset", "unknown")
                df["dataset"] = batch_dataset
            except (yaml.YAMLError, KeyError, TypeError) as e:
                console.print(
                    f"  [yellow]Warning: Could not parse manifest {batch_dir.name}: {e}[/yellow]"
                )

        # If no dataset column, try to infer from batch name or mark as unknown
        if "dataset" not in df.columns:
            df["dataset"] = "unknown"

        all_labels.append(df)
        console.print(f"  {agent_name}: {len(df)} labels from {batch_dir.name}")

    if not all_labels:
        console.print("[yellow]No labels to export after filtering[/yellow]")
        return

    # Combine all labels
    combined = pd.concat(all_labels, ignore_index=True)

    # Deduplicate: keep latest label per (ref_id, target_id, labeler)
    combined = combined.drop_duplicates(subset=["ref_id", "target_id", "labeler"], keep="last")

    console.print(f"\n[blue]Exporting {len(combined)} labels[/blue]")

    # Group by dataset and write Hive-partitioned output
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets_written = []

    for dataset, group in combined.groupby("dataset"):
        partition_dir = output_dir / f"dataset={dataset}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        csv_path = partition_dir / "data.csv"

        if append and csv_path.exists():
            existing = pd.read_csv(csv_path, dtype={"ref_id": str, "target_id": str})
            # Combine and dedupe
            merged = pd.concat([existing, group], ignore_index=True)
            merged = merged.drop_duplicates(subset=["ref_id", "target_id", "labeler"], keep="last")
            merged.to_csv(csv_path, index=False)
            console.print(f"  {dataset}: appended {len(group)} → {len(merged)} total labels")
        else:
            group.to_csv(csv_path, index=False)
            console.print(f"  {dataset}: wrote {len(group)} labels")

        datasets_written.append(dataset)

    console.print()
    console.print(f"[green]Exported to {output_dir}[/green]")
    console.print(f"  Datasets: {', '.join(datasets_written)}")
    console.print()
    console.print("Labels are now tracked in git. Run:")
    console.print(f"  git add {output_dir}")
    console.print("  git commit -m 'Add agent labels'")
