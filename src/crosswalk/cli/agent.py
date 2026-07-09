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
    'crosswalk agent test-batch' instead.

    Examples:
        # Generate 100 candidates for us_boston_streets
        crosswalk agent batch us_boston_streets

        # Generate 50 candidates with custom paths
        crosswalk agent batch us_boston_streets -n 50 \\
            -r data/raw/us_boston_overture_segments.parquet \\
            -t data/raw/us_boston_streets.parquet \\
            -o data/agents

        # Use ML model for confidence scoring
        crosswalk agent batch us_boston_streets \\
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
    console.print(f"  3. Import labels: crosswalk agent import {batch_dir} --agent-id <id>")


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
        Path("labels/human"),
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
    'crosswalk agent batch' instead.

    Examples:
        # Generate 200 samples across all datasets
        crosswalk agent test-batch -n 200

        # Specific datasets only
        crosswalk agent test-batch -n 100 -d us_boston_streets -d us_boston_bike_network

        # Filter by labeler
        crosswalk agent test-batch -n 50 --labeler brad
    """
    import geopandas as gpd
    import numpy as np
    import yaml

    from ..agent_labeling.context_generator import write_candidate_package
    from ..agent_labeling.sampler import SampledCandidate
    from ..features.semantic import resolve_best_name_variant
    from ..utils.geometry import filter_to_linestrings

    def _resolve_names(ref_row, target_row) -> tuple[str | None, str | None]:
        """Resolve best name pair using bilateral variant resolution."""
        ref_names = ref_row.get("names") if hasattr(ref_row, "get") else None
        target_names = target_row.get("names") if hasattr(target_row, "get") else None
        return resolve_best_name_variant(
            ref_names if isinstance(ref_names, dict) else None,
            target_names if isinstance(target_names, dict) else None,
        )

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
    ref_gdf = filter_to_linestrings(ref_gdf, source_name="reference")
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
            target_gdf = gpd.read_parquet(path)
            target_gdf = filter_to_linestrings(target_gdf, source_name=path.name)
            target_gdfs[dataset] = target_gdf.set_index("id")
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
            "name_levenshtein",
            "name_jaro_winkler",
            "class_similarity",
            "overlap_ratio",
            "mean_hausdorff_distance",
            "degree_match_score",
            "dead_end_match",
            "intersection_match",
        ]
        features = {col: row.get(col, 0.0) for col in feature_cols if col in row.index}

        resolved_ref_name, resolved_target_name = _resolve_names(ref_row, target_row)
        candidate = SampledCandidate(
            ref_id=str(ref_id),
            target_id=str(target_id),
            ref_geometry=ref_row.geometry,
            target_geometry=target_row.geometry,
            ref_name=resolved_ref_name,
            target_name=resolved_target_name,
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
        f"  2. Import labels: crosswalk agent import {batch_dir} -a <agent-id> -l <labels.csv>"
    )
    console.print(f"  3. Compare: crosswalk agent consensus {batch_dir}")


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
        Path("labels/human"),
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
    then generates 3 subline image variants per candidate:
      - subline_geometry_only.png: white background with faded full segments + bright aligned sublines
      - subline_road_context.png: same + gray dashed context roads
      - subline_carto_positron.png: same on CartoDB light map tiles

    Examples:
        crosswalk agent sweep \\
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
    from ..config import FEATURE_COLUMNS
    from ..features.semantic import resolve_best_name_variant
    from ..filenames import find_overture_segments
    from ..labeling.feature_store import FeatureStore

    # Validate directories
    if not labels_dir.exists():
        console.print(f"[red]Error: Labels directory not found: {labels_dir}[/red]")
        raise typer.Exit(1)

    if not geom_dir.exists():
        console.print(f"[red]Error: Geometries directory not found: {geom_dir}[/red]")
        raise typer.Exit(1)

    rng = np.random.default_rng(seed)

    # Sweep variants (subline only — highlights aligned portions)
    variants = [
        {"basemap": "subline_geometry_only", "format": "png"},
        {"basemap": "subline_road_context", "format": "png"},
        {"basemap": "subline_carto_positron", "format": "png"},
    ]

    # Load labels and geometries per dataset, sample candidates
    all_candidates = []
    all_ground_truth = []

    for dataset in datasets:
        label_file = labels_dir / f"dataset={dataset}" / "data.csv"
        geom_csv = geom_dir / f"dataset={dataset}" / "data.csv"
        geom_parquet = geom_dir / f"dataset={dataset}" / "data.parquet"

        if not label_file.exists():
            console.print(f"[yellow]Warning: No labels for {dataset}, skipping[/yellow]")
            continue
        if geom_parquet.exists():
            geom_file = geom_parquet
        elif geom_csv.exists():
            geom_file = geom_csv
        else:
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
        if geom_file.suffix == ".parquet":
            geom_df = pd.read_parquet(geom_file)
        else:
            geom_df = pd.read_csv(geom_file)
        geom_lookup = {}
        for _, row in geom_df.iterrows():
            key = (str(row["gers_id"]), str(row["target_id"]))
            geom_lookup[key] = row

        # Load features from feature store and merge into sampled labels
        feat_store = FeatureStore(dataset)
        if len(feat_store.df) > 0:
            sampled = sampled.merge(
                feat_store.df,
                on=["gers_id", "target_id"],
                how="left",
            )

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
                # Handle both WKT (CSV) and WKB (parquet) geometry formats
                if "ref_geometry_wkt" in geom_row.index:
                    ref_geom = wkt.loads(geom_row["ref_geometry_wkt"])
                elif "ref_geometry" in geom_row.index:
                    from shapely import wkb

                    ref_geom = wkb.loads(geom_row["ref_geometry"])
                else:
                    raise KeyError("No ref geometry column found")

                if "target_geometry_wkt" in geom_row.index:
                    target_geom = wkt.loads(geom_row["target_geometry_wkt"])
                elif "target_geometry" in geom_row.index:
                    from shapely import wkb

                    target_geom = wkb.loads(geom_row["target_geometry"])
                else:
                    raise KeyError("No target geometry column found")
            except Exception:
                console.print(f"  [yellow]Skipping {ref_id}: invalid geometry[/yellow]")
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

            # Extract features from label row (merged from FeatureStore)
            features = {}
            for col in FEATURE_COLUMNS:
                if col in row.index:
                    val = row[col]
                    features[col] = None if pd.isna(val) else val

            # Parse names/classes from stored data
            import json

            ref_names_raw = None
            target_names_raw = None
            ref_name = None
            target_name = None
            ref_class = None
            target_class = None

            # Prefer full names structs for bilateral resolution
            if "ref_names" in geom_row.index and pd.notna(geom_row.get("ref_names")):
                val = geom_row["ref_names"]
                ref_names_raw = json.loads(val) if isinstance(val, str) else val
            if "target_names" in geom_row.index and pd.notna(geom_row.get("target_names")):
                val = geom_row["target_names"]
                target_names_raw = json.loads(val) if isinstance(val, str) else val

            # Resolve best name variant pair
            ref_name, target_name = resolve_best_name_variant(ref_names_raw, target_names_raw)

            # Class from direct columns
            if "ref_class" in geom_row.index:
                ref_class = geom_row["ref_class"]
            if "target_class" in geom_row.index:
                target_class = geom_row["target_class"]

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
    console.print(f"  crosswalk agent run --batch {batch_dir} --variant subline_road_context")
    console.print(f"  crosswalk agent eval-sweep {batch_dir}")


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
    timeout: int = typer.Option(600, "--timeout", help="Timeout per chunk in seconds"),
    chunk_size: int = typer.Option(25, "--chunk-size", help="Candidates per CLI invocation"),
):
    """Run Claude agent in batch mode on labeling candidates.

    Builds a prompt with few-shot examples from other batches, then invokes
    Claude Code CLI to process candidates in chunks. The agent reads images
    and metadata itself and writes results to a CSV file.

    Resumes by default - existing labels are skipped. Use --overwrite to start fresh.

    Examples:
        crosswalk agent run --batch data/agents/batches/sweep_2026-02-08_131609 \\
            --model opus --variant subline_road_context --limit 5

        crosswalk agent run --batch data/agents/batches/sweep_2026-02-08_131609 \\
            --model sonnet --variant subline_carto_positron --overwrite

        crosswalk agent run --batch data/agents/batches/sweep_2026-02-08_131609 \\
            --model opus --variant subline_geometry_only --few-shot 8 --chunk-size 20
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


@agent_app.command("stitch-batch")
def generate_stitch_batch(
    dataset: str = typer.Argument(..., help="Dataset name (e.g. us_boston_streets)"),
    group_ids: str = typer.Option(
        None,
        "--group-ids",
        help="Comma-separated group_ids to generate (default: use tier-selected batch)",
    ),
    group_ids_file: Path = typer.Option(
        None,
        "--group-ids-file",
        help="File with group_ids (JSON list or newline-separated)",
    ),
    recover_labeled: bool = typer.Option(
        False,
        "--recover-labeled",
        help="Auto-select sidecar groups that best-correspond to existing human labels",
    ),
    recover_empty: bool = typer.Option(
        False,
        "--recover-empty",
        help="Also include reject-all (empty-edge) human labels whose group_id "
        "still exists verbatim in the sidecar (combine with --recover-labeled)",
    ),
    output_dir: Path = typer.Option(
        Path("data/agents/stitching/batches"),
        "--output",
        "-o",
        help="Root output dir for stitching evidence batches",
    ),
    batch_name: str = typer.Option(None, "--name", help="Batch name (default: dataset name)"),
    batch_size: int = typer.Option(
        15, "--batch-size", "-n", help="Tier-selected batch size (when no explicit group_ids)"
    ),
    k_alternatives: int = typer.Option(
        8,
        "--alternatives",
        "-k",
        help="Top-K organic alternatives per group (default: 8; two whole-group "
        "seed options are appended on top)",
    ),
):
    """Generate evidence packs for agent stitching-group labeling.

    Reads the groups sidecar, computes top-K alternatives + spatial context per
    group, and writes one evidence pack per group (overview + per-option images,
    metadata.yaml, prompt.txt) under {output}/{name}/{group_id}/.

    Examples:
        # Tier-selected batch (like the web review batch)
        crosswalk agent stitch-batch us_boston_streets

        # Specific groups (e.g. for eval against human labels)
        crosswalk agent stitch-batch us_boston_streets --group-ids abc123,def456

        # Auto-recover the groups matching existing human labels
        crosswalk agent stitch-batch us_boston_streets --recover-labeled
    """
    import json

    from ..agent_labeling.stitch_evidence import generate_stitch_evidence
    from ..filenames import (
        PROJECT_ROOT,
        bridge_filename,
        groups_sidecar_path,
    )
    from ..labeling.stitching_store import StitchingLabelStore
    from ..matching.alternatives import generate_top_k_alternatives
    from ..matching.batch_selection import select_stitching_batch

    out_root = PROJECT_ROOT / "data" / "output"
    bridge_path = out_root / bridge_filename(dataset)
    sidecar_path = groups_sidecar_path(bridge_path)
    if not sidecar_path.exists():
        console.print(f"[red]No groups sidecar at {sidecar_path}[/red]")
        console.print("[yellow]Run `crosswalk stitch` first.[/yellow]")
        raise typer.Exit(1)

    sidecar = json.loads(sidecar_path.read_text())
    groups = sidecar.get("groups", [])
    console.print(f"[blue]Loaded {len(groups)} groups from sidecar[/blue]")

    # Determine requested group_ids.
    requested: list[str] | None = None
    if group_ids:
        requested = [g.strip() for g in group_ids.split(",") if g.strip()]
    elif group_ids_file:
        text = group_ids_file.read_text().strip()
        try:
            # Normalize items to stripped strings: JSON numbers (unquoted hex
            # ids like 21b67ef2 are rare but possible) would otherwise never
            # match the string group_ids in the sidecar.
            requested = [str(g).strip() for g in json.loads(text)]
        except json.JSONDecodeError:
            requested = [ln.strip() for ln in text.splitlines() if ln.strip()]
    elif recover_labeled or recover_empty:
        from ..agent_labeling.stitch_eval import (
            recover_empty_reject_all,
            recover_labeled_groups,
        )

        stitch_store = StitchingLabelStore(dataset)
        human_df = stitch_store.load(dataset)
        requested = []
        if recover_labeled:
            rec = recover_labeled_groups(groups, human_df)
            requested = list(rec["target_group_ids"])
            console.print(
                f"[blue]Label recovery:[/blue] {len(rec['clean'])} clean, "
                f"{len(rec['split'])} split, {len(rec['empty'])} empty(NONE), "
                f"{len(rec['lost'])} lost, {len(rec.get('set', []))} set, "
                f"{len(rec.get('set_lost', []))} set-lost "
                f"-> {len(rec['target_group_ids'])} target groups"
            )
        if recover_empty:
            emp = recover_empty_reject_all(groups, human_df)
            new_empty = [g for g in emp["recovered"] if g not in requested]
            requested += new_empty
            console.print(
                f"[blue]Reject-all recovery:[/blue] {len(emp['recovered'])} recoverable "
                f"(+{len(new_empty)} new), {len(emp['unrecoverable'])} unrecoverable"
            )

    # Select groups to render.
    if requested is not None:
        gmap = {g["group_id"]: g for g in groups}
        selected = [gmap[gid] for gid in requested if gid in gmap]
        missing = [gid for gid in requested if gid not in gmap]
        if missing:
            console.print(f"[yellow]{len(missing)} requested group_ids not in sidecar[/yellow]")
    else:
        for g in groups:
            # Pass ref geometries so multi-ref contiguous chains can be offered.
            g["alternatives"] = generate_top_k_alternatives(
                g.get("edges", []),
                ref_geoms=g.get("ref_geometries", {}),
                target_geoms=g.get("target_geometries", {}),
                k=k_alternatives,
            )
        reviewed = StitchingLabelStore(dataset).get_reviewed_group_ids(dataset)
        selected = select_stitching_batch(groups, reviewed, k=batch_size)

    if not selected:
        console.print("[red]No groups selected[/red]")
        raise typer.Exit(1)

    # Ensure alternatives present, then fill spatial context.
    for g in selected:
        if "alternatives" not in g:
            g["alternatives"] = generate_top_k_alternatives(
                g.get("edges", []),
                ref_geoms=g.get("ref_geometries", {}),
                target_geoms=g.get("target_geometries", {}),
                k=k_alternatives,
            )
    console.print(f"[blue]Filling spatial context for {len(selected)} groups...[/blue]")
    from .data import _fill_spatial_context

    _fill_spatial_context(selected, dataset)

    name = batch_name or dataset
    batch_dir = output_dir / name
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch = {"dataset_id": dataset, "groups": selected}
    (batch_dir / "batch.json").write_text(json.dumps(batch))

    console.print(f"[blue]Generating evidence packs -> {batch_dir}[/blue]")
    generated = generate_stitch_evidence(batch, batch_dir)
    console.print(f"[green]Generated {len(generated)} evidence packs[/green]")
    console.print(f"  Batch dir: {batch_dir}")
    console.print("Next: crosswalk agent stitch-run --batch " + str(batch_dir))


@agent_app.command("stitch-run")
def run_stitch_panel(
    batch_dir: Path = typer.Option(..., "--batch", "-b", help="Batch dir with evidence packs"),
    group_ids: str = typer.Option(None, "--group-ids", help="Comma-separated subset to run"),
    timeout: int = typer.Option(240, "--timeout", help="Per-provider timeout (s)"),
    invocation_budget: float = typer.Option(
        300.0,
        "--invocation-budget",
        help="Seconds to back off + retry a down provider (quota/rate-limit/network) "
        "before hard-failing the run. Worst-case wall time is this + one --timeout.",
    ),
    limit: int = typer.Option(0, "--limit", "-l", help="Max groups (0=all)"),
    claude_model: str = typer.Option("claude-opus-4-8", "--claude-model"),
    claude_effort: str = typer.Option("medium", "--claude-effort"),
    codex_model: str = typer.Option("gpt-5.5", "--codex-model"),
    codex_effort: str = typer.Option("low", "--codex-effort"),
    agy_model: str = typer.Option("Gemini 3.5 Flash (Medium)", "--agy-model"),
    panel_name: str = typer.Option(
        "default",
        "--panel",
        help="Named panel config: 'default'/'v2' (3 voters: claude+codex+agy), "
        "'v3-candidate' (adds a 4th opencode/Qwen3-VL voter — OFF by default, "
        "opt-in only; does not affect production waves), or 'no-agy' "
        "(claude+codex+opencode quota-outage fallback; its labels are refused by "
        "stitch-export without --allow-nonstandard-panel).",
    ),
    opencode_model: str = typer.Option(
        "openrouter/qwen/qwen3-vl-235b-a22b-instruct",
        "--opencode-model",
        help="Model string for the opencode voter (used by --panel v3-candidate and no-agy).",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from votes.partial.csv/consensus.partial.csv, skipping "
        "already-completed groups (per-group persistence makes runs interruptible).",
    ),
    pack_feedback: bool = typer.Option(
        False,
        "--pack-feedback",
        help="DIAGNOSTIC (default off): ask each panelist to append a structured "
        "pack_feedback self-report (missing_info/ambiguities/confidence_basis) to its "
        "answer, recorded in votes.csv. Augments the prompt for this run only; the "
        "default production prompt is untouched.",
    ),
):
    """Run the consensus panel (claude + codex + agy; opt-in 4th voter) on a batch.

    Writes votes.csv (every raw vote — audit data) and consensus.csv (per-group
    routing) into the batch dir. Writes NOTHING into labels/.

    Examples:
        crosswalk agent stitch-run --batch data/agents/stitching/batches/us_boston_streets
        crosswalk agent stitch-run --batch <dir> --panel v3-candidate  # 4-voter candidate
    """
    from ..agent_labeling.stitch_runner import (
        ProviderInvocationError,
        ProviderSpec,
        get_panel,
        run_batch,
    )

    if not batch_dir.exists():
        console.print(f"[red]Batch dir not found: {batch_dir}[/red]")
        raise typer.Exit(1)

    # Build the panel from the named config, applying the per-provider model/effort
    # overrides so the incumbent flags keep working. DEFAULT_PANEL is unchanged:
    # the 4th voter is only present when --panel v3-candidate is selected.
    overrides = {
        "claude": {"model": claude_model, "effort": claude_effort},
        "codex": {"model": codex_model, "effort": codex_effort},
        "agy": {"model": agy_model},
        "opencode": {"model": opencode_model},
    }
    panel = [
        ProviderSpec(
            name=p.name,
            model=overrides.get(p.name, {}).get("model", p.model),
            effort=overrides.get(p.name, {}).get("effort", p.effort),
        )
        for p in get_panel(panel_name)
    ]
    gids = [g.strip() for g in group_ids.split(",") if g.strip()] if group_ids else None

    console.print(f"[blue]Running panel on batch {batch_dir}[/blue]")
    console.print(f"  Panel ({panel_name}): {', '.join(f'{p.name}={p.model}' for p in panel)}")
    if pack_feedback:
        console.print(
            "  [yellow]pack-feedback ON: appending diagnostic self-report request[/yellow]"
        )
    try:
        votes_df, consensus_df = run_batch(
            batch_dir,
            panel=panel,
            group_ids=gids,
            timeout=timeout,
            limit=limit,
            collect_feedback=pack_feedback,
            resume=resume,
            invocation_budget_s=invocation_budget,
        )
    except ProviderInvocationError as e:
        console.print(f"[red]Panel halted — provider down:[/red] {e}")
        console.print(
            "[yellow]Completed groups were flushed; re-run with --resume once the "
            "provider is healthy to continue.[/yellow]"
        )
        raise typer.Exit(1) from e

    console.print(f"[green]{len(consensus_df)} groups, {len(votes_df)} votes[/green]")
    tier_counts = consensus_df["consensus"].value_counts().to_dict()
    route_counts = consensus_df["routing"].value_counts().to_dict()
    console.print(f"  Consensus: {tier_counts}")
    console.print(f"  Routing:   {route_counts}")
    console.print(f"  Wrote {batch_dir / 'votes.csv'} and {batch_dir / 'consensus.csv'}")


@agent_app.command("stitch-eval")
def eval_stitch_panel(
    batch_dir: Path = typer.Option(..., "--batch", "-b", help="Batch dir with consensus.csv"),
    dataset: str = typer.Option("us_boston_streets", "--dataset", "-d"),
    labels_dir: Path = typer.Option(Path("labels/stitching"), "--labels", "-l"),
    output: Path = typer.Option(
        None, "--output", "-o", help="Write a markdown report to this path"
    ),
):
    """Validate panel results against human stitching labels.

    Disagreement is NOT treated as agent failure: the report frames panel-vs-
    human contradictions as label-quality review candidates and reports
    option-coverage gaps.

    Examples:
        crosswalk agent stitch-eval --batch data/agents/stitching/batches/us_boston_streets
    """
    import pandas as pd

    from ..agent_labeling.stitch_eval import (
        disagreement_report,
        evaluate_batch,
        evaluate_set_labels,
        summarize,
        summarize_set,
    )

    human_path = labels_dir / f"dataset={dataset}" / "data.csv"
    if not human_path.exists():
        console.print(f"[red]No human labels at {human_path}[/red]")
        raise typer.Exit(1)
    human_df = pd.read_csv(human_path, dtype={"group_id": str})

    results = evaluate_batch(batch_dir, human_df)
    set_results = evaluate_set_labels(batch_dir, human_df)
    if not results and not set_results:
        console.print("[yellow]No panel groups mapped to human labels[/yellow]")
        raise typer.Exit(0)

    summary = summarize(results)
    disagreements = disagreement_report(results)

    if results:
        console.print(f"[bold]Panel eval: {summary['n_groups']} mapped groups (pair labels)[/bold]")
        console.print(
            f"  Panel exact edge-set match: {summary['panel_exact_rate']:.0%}  "
            f"mean F1: {summary['panel_mean_f1']:.3f}"
        )
        console.print("  Per provider:")
        for prov, s in summary["by_provider"].items():
            console.print(
                f"    {prov:<8} exact={s['exact_rate']:.0%} f1={s['mean_f1']:.3f} (n={s['n']})"
            )
        console.print("  Per consensus tier:")
        for tier, s in summary["by_consensus"].items():
            console.print(
                f"    {tier:<10} exact={s['exact_rate']:.0%} f1={s['mean_f1']:.3f} (n={s['n']})"
            )
        oc = summary["option_coverage"]
        console.print(
            f"  Option-coverage gap: {oc['gap']}/{summary['n_groups']} ({oc['gap_rate']:.0%})"
        )
        console.print(f"  Disagreements (label-quality review candidates): {len(disagreements)}")
    else:
        console.print("[bold]Panel eval: 0 pair-label groups mapped[/bold]")

    if set_results:
        ss = summarize_set(set_results)
        console.print(f"[bold]Set-label eval: {ss['n_set_groups']} mapped set labels[/bold]")
        console.print(
            f"  Membership exact: {ss['membership_exact_rate']:.0%}  "
            f"boundary precision: {ss['boundary_precision']:.3f}  "
            f"coverage: {ss['coverage']:.3f}"
        )

    if output and results:
        _write_stitch_eval_report(output, summary, results, disagreements, dataset, batch_dir)
        console.print(f"[green]Wrote report to {output}[/green]")
    elif output:
        console.print("[yellow]No pair-label report written (set labels only)[/yellow]")


def _write_stitch_eval_report(output, summary, results, disagreements, dataset, batch_dir) -> None:
    """Write a markdown eval report."""

    def _edges_str(es) -> str:
        if not es:
            return "(none)"
        return ", ".join(f"{r[:8]}..->..{t[-14:]}" for r, t in sorted(es))

    lines = []
    lines.append("# Agent Stitching Panel Eval\n")
    lines.append(f"Dataset: `{dataset}`  |  Batch: `{batch_dir}`\n")
    lines.append(f"Mapped groups: {summary['n_groups']}\n")
    lines.append("\n## Agreement with human labels\n")
    lines.append(
        f"- Panel exact edge-set match: **{summary['panel_exact_rate']:.0%}**, "
        f"mean edge F1: **{summary['panel_mean_f1']:.3f}**\n"
    )
    lines.append("\n### Per provider\n")
    lines.append("| Provider | Exact | Mean F1 | N |\n|---|---|---|---|\n")
    for prov, s in summary["by_provider"].items():
        lines.append(f"| {prov} | {s['exact_rate']:.0%} | {s['mean_f1']:.3f} | {s['n']} |\n")
    lines.append("\n### Per consensus tier\n")
    lines.append("| Consensus | Exact | Mean F1 | N |\n|---|---|---|---|\n")
    for tier, s in summary["by_consensus"].items():
        lines.append(f"| {tier} | {s['exact_rate']:.0%} | {s['mean_f1']:.3f} | {s['n']} |\n")
    oc = summary["option_coverage"]
    lines.append(
        f"\n### Option-coverage gap\n\n{oc['gap']}/{summary['n_groups']} "
        f"({oc['gap_rate']:.0%}) human edge sets match NO current option — "
        f"signal for whether top-K is large enough.\n"
    )
    lines.append("\n## Disagreement report (label-quality review candidates)\n")
    lines.append(
        "Groups where the panel contradicts the human label. Framed as candidates "
        "for human re-review, not agent errors.\n\n"
    )
    for r in disagreements:
        lines.append(
            f"### Group `{r.group_id}` (human `{r.human_group_id}`, {r.match_type}) "
            f"— {r.consensus}\n"
        )
        lines.append(f"- Human edge set: {_edges_str(r.human_edge_set)}\n")
        lines.append(f"- Panel choice `{r.panel_choice}`: {_edges_str(r.panel_edge_set)}\n")
        lines.append(f"- Option-covered: {r.option_covered}  |  F1: {r.f1:.2f}\n")
        for prov, (choice, es) in r.provider_votes.items():
            lines.append(f"  - {prov}: choice={choice} edges={_edges_str(es)}\n")
        lines.append("\n")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("".join(lines))


@agent_app.command("stitch-expressibility")
def stitch_expressibility(
    dataset: str = typer.Argument(..., help="Dataset name (e.g. us_boston_streets)"),
    sidecar: Path = typer.Option(
        None,
        "--sidecar",
        help="Groups sidecar JSON (default: data/output/{dataset}_groups.json)",
    ),
    k_alternatives: int = typer.Option(
        8, "--alternatives", "-k", help="Top-K organic alternatives per group"
    ),
    show_misses: int = typer.Option(
        20, "--show-misses", help="Max inexpressible labels to list (0 = none)"
    ),
):
    """Measure option-menu EXPRESSIBILITY of the stitch generator vs settled labels.

    Reports the fraction of settled (pair-semantics, non-reject-all) stitching
    labels whose exact edge set is expressible by the current option generator
    for the sidecar group they correspond to. Reads the sidecar and labels READ
    ONLY; runs no provider. Useful as a before/after gate on generator changes.

    Examples:
        crosswalk agent stitch-expressibility us_boston_streets
        crosswalk agent stitch-expressibility us_seattle_sidewalks -k 8
    """
    import json

    from ..agent_labeling.stitch_expressibility import measure_expressibility
    from ..filenames import PROJECT_ROOT

    sidecar_path = sidecar or (PROJECT_ROOT / "data" / "output" / f"{dataset}_groups.json")
    if not sidecar_path.exists():
        console.print(f"[red]No groups sidecar at {sidecar_path}[/red]")
        raise typer.Exit(1)
    label_path = PROJECT_ROOT / "labels" / "stitching" / f"dataset={dataset}" / "data.csv"
    if not label_path.exists():
        console.print(f"[red]No stitching labels at {label_path}[/red]")
        raise typer.Exit(1)

    groups = json.loads(sidecar_path.read_text()).get("groups", [])
    labels_df = pd.read_csv(label_path, dtype={"group_id": str})

    report = measure_expressibility(dataset, groups, labels_df, k=k_alternatives)
    s = report.summary()
    console.print(f"[bold]Expressibility: {dataset} (k={report.k})[/bold]")
    console.print(
        f"  settled={s['n_settled']}  clean-recoverable={s['n_recoverable']}  "
        f"covered={s['n_covered']}"
    )
    rate = s["expressibility"]
    console.print(f"  EXPRESSIBILITY = [green]{rate:.1%}[/green]" if rate is not None else "  n/a")
    console.print(f"  inexpressible (recoverable but no option matches): {s['n_misses']}")
    if show_misses:
        for m in sorted(report.misses, key=lambda x: -x.n_label_edges)[:show_misses]:
            console.print(
                f"    - label {m.label_group_id} -> group {m.sidecar_group_id} "
                f"({m.match_type}): {m.n_label_edges} edges / {m.n_group_edges} in group, "
                f"{m.n_options} options"
            )


@agent_app.command("stitch-export")
def export_stitch_panel(
    batches: list[str] = typer.Option(
        ...,
        "--batch",
        "-b",
        help=(
            "Batch dir(s) with consensus.csv + batch.json. Repeatable and/or "
            "comma-separated; later batches supersede earlier ones per group_id."
        ),
    ),
    dataset: str = typer.Option("us_boston_streets", "--dataset", "-d"),
    labels_dir: Path = typer.Option(Path("labels/stitching"), "--labels", "-l"),
    max_edges: int = typer.Option(20, "--max-edges", help="Skip groups with > this many edges"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report only; write nothing"),
    empty_set: bool = typer.Option(
        True,
        "--empty-set/--no-empty-set",
        help=(
            "Also export unanimous-NONE verdicts (panel rejected every option) as "
            "empty-set reject-all labels tagged panel_unanimous_none_v3. Default on "
            "(this is required label production for the learned resolver); pass "
            "--no-empty-set to plan/write the accept path only."
        ),
    ),
    allow_nonstandard_panel: bool = typer.Option(
        False,
        "--allow-nonstandard-panel",
        help=(
            "Export even when a batch's votes.csv provider set differs from the "
            "default claude+codex+agy panel (e.g. --panel no-agy). Labels are "
            "still stamped with the panel_unanimous_* labelers, so only use this "
            "after an explicit provenance decision."
        ),
    ),
):
    """Export unanimous panel consensus into human-equivalent stitching labels.

    Two verdict classes are promoted. Unanimous ``auto_accept`` groups export
    their chosen edge set (labeler ``panel_unanimous_v3``); with ``--empty-set``
    (default) unanimous-NONE groups export a reject-all EMPTY-SET label (labeler
    ``panel_unanimous_none_v3``, ``selected_edges == []``). Gates are applied in
    order and reported per group: (a) routing, (b) size, (c) class-consistency,
    (d) sliver canonicalization, (e) human precedence (the class/sliver gates are
    vacuous on an empty set and are skipped there). Rows upsert by group_id
    (idempotent). Batches voted by a nonstandard panel composition are refused
    unless ``--allow-nonstandard-panel`` is passed (composition is provenance).

    Examples:
        crosswalk agent stitch-export \\
            -b data/agents/stitching/batches/us_boston_streets_phase2 \\
            -b data/agents/stitching/batches/us_boston_streets_phase3
    """
    from ..agent_labeling.stitch_export import (
        REASON_HUMAN_PRECEDENCE,
        nonstandard_panel_batches,
        plan_exports,
        write_exports,
        write_vote_provenance,
    )

    # Support both repeatable --batch and comma-separated values.
    batch_dirs: list[Path] = []
    for raw in batches:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                batch_dirs.append(Path(part))

    for bd in batch_dirs:
        if not (bd / "consensus.csv").exists():
            console.print(f"[red]No consensus.csv in {bd}[/red]")
            raise typer.Exit(1)
        if not (bd / "batch.json").exists():
            console.print(
                f"[yellow]Warning: no batch.json in {bd} — sliver canonicalization "
                "and edge-overlap precedence degrade for its groups[/yellow]"
            )

    offending = nonstandard_panel_batches(batch_dirs)
    if offending and not allow_nonstandard_panel:
        for name, providers in sorted(offending.items()):
            console.print(
                f"[red]Batch {name} was voted by a nonstandard panel "
                f"({', '.join(sorted(providers))}) — refusing to stamp its labels "
                f"with the panel_unanimous_* labelers. Re-run with "
                f"--allow-nonstandard-panel only after an explicit provenance "
                f"decision.[/red]"
            )
        raise typer.Exit(1)

    report = plan_exports(
        batch_dirs, dataset, labels_dir, max_edges=max_edges, export_empty_set=empty_set
    )

    none_note = f", {report.n_unanimous_none} unanimous-NONE candidates" if empty_set else ""
    console.print(
        f"[bold]Panel export: {report.n_total_groups} merged groups, "
        f"{report.n_auto_accept} auto_accept candidates{none_note}[/bold]"
    )
    console.print(f"  Batches (in precedence order): {', '.join(b.name for b in batch_dirs)}")

    # Per-group report.
    for g in report.groups:
        if g.exported and g.is_empty_set:
            console.print(
                f"  [green]EXPORT-EMPTY[/green] {g.group_id} [{g.source_batch}] "
                f"{g.match_type} reject-all (0 edges) conf={g.mean_confidence:.3f}"
            )
        elif g.exported:
            slivers = f" (-{g.n_slivers_dropped} sliver)" if g.n_slivers_dropped else ""
            console.print(
                f"  [green]EXPORT[/green] {g.group_id} [{g.source_batch}] "
                f"{g.match_type} {g.n_edges_final} edges{slivers} "
                f"conf={g.mean_confidence:.3f}"
            )
        else:
            extra = ""
            if g.reason == REASON_HUMAN_PRECEDENCE and g.human_group_id:
                extra = f" (human {g.human_group_id})"
            elif g.reason == "over_max_edges":
                extra = f" ({g.n_edges_raw} > {max_edges})"
            elif g.reason == "emptied_by_sliver":
                extra = f" (-{g.n_slivers_dropped} sliver)"
            console.print(
                f"  [yellow]SKIP[/yellow]   {g.group_id} [{g.source_batch}] -> {g.reason}{extra}"
            )

    empty_note = f" ({len(report.exported_empty)} empty-set)" if empty_set else ""
    console.print(
        f"\n[bold]Summary:[/bold] {len(report.exported)} exported{empty_note}, "
        f"{len(report.skipped)} skipped, "
        f"{report.total_slivers_dropped()} sliver edges dropped"
    )
    by_reason = report.skipped_by_reason()
    if by_reason:
        console.print("  Skips: " + ", ".join(f"{r}={n}" for r, n in sorted(by_reason.items())))

    if dry_run:
        console.print("[cyan]Dry run — no labels written.[/cyan]")
        return

    written = write_exports(report, dataset, labels_dir)
    n_empty = len(report.exported_empty)
    empty_written = f" ({n_empty} reject-all empty-set)" if n_empty else ""
    console.print(
        f"[green]Wrote {written} panel labels{empty_written} to "
        f"{labels_dir}/dataset={dataset}[/green]"
    )

    # Best-effort: labels are already persisted above, so a malformed batch CSV
    # must not crash the command and leave an inconsistent "failed" export.
    try:
        n_votes, n_consensus = write_vote_provenance(batch_dirs, dataset)
        console.print(
            f"[green]Archived vote provenance: {n_votes} ballots, {n_consensus} consensus "
            f"rows to labels/votes/dataset={dataset}[/green]"
        )
    except Exception as e:  # noqa: BLE001 - provenance is best-effort, never fail the export
        console.print(
            f"[yellow]Warning: vote-provenance archival skipped ({e}); "
            f"labels were still written.[/yellow]"
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
        crosswalk agent import data/agents/batches/batch_2026-01-18_001 \\
            --agent-id claude --labels claude_labels.csv

        # Import with confidence and reasoning
        crosswalk agent import data/agents/batches/batch_* \\
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
        crosswalk agent consensus data/agents/batches/batch_2026-01-18_001

        # Show disagreements only
        crosswalk agent consensus data/agents/batches/batch_* --disagreements
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
        crosswalk agent eval-sweep data/agents/batches/sweep_2026-01-28_120000

        # Include reasoning text
        crosswalk agent eval-sweep data/agents/batches/sweep_2026-01-28_120000 --reasoning
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
        crosswalk agent export

        # Export only gemini-flash labels
        crosswalk agent export --agent gemini-flash

        # Append to existing labels
        crosswalk agent export --append
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
