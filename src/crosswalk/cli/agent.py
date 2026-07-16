"""AI agent labeling commands."""

import shutil
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
    include_oversize: bool = typer.Option(
        False,
        "--include-oversize",
        help="Calibration escape hatch: keep groups over the export backstop "
        "(stitch_export_backstop_max_edges candidate edges) eligible for tier "
        "selection. Default excludes them — no panel verdict on an over-backstop "
        "group can ever export, so production waves must not burn quota on them "
        "(their verdicts route to human review via the size gate).",
    ),
    calibration: bool = typer.Option(
        False,
        "--calibration",
        help="Calibration wave: keep HUMAN-LABELED groups votable so voter "
        "accuracy can be scored against settled ground truth. By default a "
        "production wave excludes exact-id-reviewed groups AND drift-mapped "
        "FULLY-covered groups (labeling/stitch_coverage.py) — re-adjudicating "
        "them burns quota on settled answers. Partially-covered groups stay "
        "votable either way (genuinely open questions). Tier-sampled batches "
        "only; explicit --group-ids/--recover-* selections are unaffected.",
    ),
    decompose: bool = typer.Option(
        False,
        "--decompose",
        help="Split over-backstop groups into panel-sized sub-problems (biconnected "
        "decomposition of the candidate-edge graph; #367 Mode B). Each sub-problem "
        "gets its own evidence pack and panel vote; a whole-group label is minted "
        "at export only when EVERY sub-problem resolves unanimously. Over-backstop "
        "groups stay ELIGIBLE for tier selection under this flag (decomposition "
        "makes their verdicts exportable); an irreducible monster is still dropped "
        "from the wave unless --include-oversize. Default off — the flow is "
        "byte-identical without this flag.",
    ),
    decompose_max_edges: int = typer.Option(
        0,
        "--decompose-max-edges",
        help="Sub-problem edge budget for --decompose "
        "(0 = settings.stitch_export_backstop_max_edges, the panel's export envelope).",
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

        # Split monster groups into panel-sized sub-problems
        crosswalk agent stitch-batch co_bogota_roads --group-ids 3c3e6853 --decompose
    """
    import json

    from ..agent_labeling.stitch_evidence import (
        generate_stitch_evidence,
        missing_evidence_packs,
    )
    from ..agent_labeling.stitch_provenance import artifact_descriptor
    from ..config import settings
    from ..filenames import (
        PROJECT_ROOT,
        bridge_filename,
        candidates_sidecar_path,
        groups_sidecar_path,
    )
    from ..labeling.stitching_store import StitchingLabelStore
    from ..matching.alternatives import generate_top_k_alternatives
    from ..matching.batch_selection import (
        group_candidate_edge_count,
        select_stitching_batch,
    )
    from ..provenance import source_commit_provenance

    # Validate --decompose-max-edges against the export backstop (#367 Mode B).
    # A budget ABOVE the backstop is a silent mini-void: sub-problems sized
    # between the backstop and the budget pass the panel (they get packed and
    # voted), but the consensus-time size gate (#386) then demotes their verdicts
    # to human_review, so they can never auto-accept — and one un-exportable
    # sub-verdict conservatively blocks the parent's whole-group label forever.
    # Fail loud rather than burn panel quota on votes that structurally cannot
    # contribute a label (repo style: no silent partial-void).
    if decompose_max_edges:
        backstop = settings.stitch_export_backstop_max_edges
        if decompose_max_edges < 1:
            console.print(
                f"[red]--decompose-max-edges must be >= 1 (got {decompose_max_edges}).[/red]"
            )
            raise typer.Exit(1)
        if decompose_max_edges > backstop:
            console.print(
                f"[red]--decompose-max-edges ({decompose_max_edges}) exceeds the export "
                f"backstop (stitch_export_backstop_max_edges={backstop}). Sub-problems sized "
                f"{backstop + 1}..{decompose_max_edges} would pass the panel but be size-gated "
                f"at export, silently blocking their parent's whole-group label. Choose a "
                f"budget <= {backstop} (or raise the backstop).[/red]"
            )
            raise typer.Exit(1)
        if not decompose:
            console.print(
                "[yellow]--decompose-max-edges has no effect without --decompose "
                "(ignored).[/yellow]"
            )

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
    candidates_path = candidates_sidecar_path(bridge_path)
    source_artifacts = {
        "groups_sidecar": artifact_descriptor(sidecar_path, root=PROJECT_ROOT),
        "candidates_parquet": artifact_descriptor(candidates_path, root=PROJECT_ROOT),
        "bridge_parquet": artifact_descriptor(bridge_path, root=PROJECT_ROOT),
    }
    batch_generation_source = {"source_commit": source_commit_provenance(PROJECT_ROOT)}

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
        # Pre-vote reviewed exclusion. Production waves must not spend votes
        # re-adjudicating settled ground truth: exact-id-reviewed groups AND
        # drift-mapped FULLY-covered groups (a regenerated sidecar re-mints
        # group_ids for already-reviewed geometry — see stitch_coverage) are
        # dropped before sampling. Partially-covered groups stay votable (new
        # membership makes them genuinely open). --calibration inverts this:
        # a calibration wave deliberately votes on labeled groups to score
        # voter accuracy, so NO reviewed exclusion is applied.
        store = StitchingLabelStore(dataset)
        if calibration:
            reviewed = set()
            console.print(
                "[yellow]--calibration: human-labeled groups stay votable "
                "(no reviewed exclusion)[/yellow]"
            )
        else:
            from ..labeling.stitch_coverage import (
                compute_prior_coverage,
                fully_covered_group_ids,
            )

            reviewed = store.get_reviewed_group_ids(dataset)
            coverage = compute_prior_coverage(groups, store.load(dataset))
            drift_covered = fully_covered_group_ids(coverage) - reviewed
            if drift_covered:
                console.print(
                    f"[blue]Drift-aware exclusion: {len(drift_covered)} group(s) "
                    f"fully covered by drift-mapped prior labels dropped from the "
                    f"wave (pass --calibration to vote labeled groups)[/blue]"
                )
            reviewed |= drift_covered
        # Panel waves exclude over-backstop groups by default: the export
        # backstop blocks any verdict on them from minting a label, so
        # selecting them (the old Tier-1 rule ALWAYS took the single largest
        # group) burned quota on 50-570 KB prompts for unexportable verdicts.
        # --include-oversize restores them for deliberate calibration waves.
        # --decompose ALSO lifts the exclusion: decomposition converts an
        # over-backstop group into panel-sized sub-problems whose recomposed
        # verdict CAN export, so the quota-void rationale no longer applies
        # (an irreducible monster is re-dropped in the decompose block below
        # unless --include-oversize keeps it as a calibration wave).
        max_candidate = (
            None if (include_oversize or decompose) else settings.stitch_export_backstop_max_edges
        )
        if max_candidate is not None:
            n_over = sum(
                1
                for g in groups
                if g.get("group_id") not in reviewed
                and group_candidate_edge_count(g) > max_candidate
            )
            if n_over:
                console.print(
                    f"[yellow]Excluding {n_over} over-backstop group(s) "
                    f"(> {max_candidate} candidate edges) from wave selection — "
                    f"their verdicts cannot export; pass --include-oversize for a "
                    f"deliberate calibration wave[/yellow]"
                )
        selected = select_stitching_batch(
            groups, reviewed, k=batch_size, max_candidate_edges=max_candidate
        )

    if not selected:
        console.print("[red]No groups selected[/red]")
        raise typer.Exit(1)

    # Opt-in decomposition (#367 Mode B): replace each over-backstop group with
    # panel-sized sub-problems (biconnected decomposition of its candidate-edge
    # graph). Sub-problems flow through the existing machinery (alternatives,
    # context, packs, votes) as ordinary groups keyed by their content-hash id;
    # the parent group is kept in batch.json (pack-less, marked
    # ``decomposed_parent``) as the export-time recomposition roster + gate
    # data. Default off: without --decompose this block is a no-op.
    decomposition_manifest: dict[str, dict] = {}
    decomposed_parents: list[dict] = []
    if decompose:
        # Route-reason vocabulary shared with the consensus-time size gate
        # (#386): an oversized irreducible sub-problem is size_gated to human
        # review, exactly like an over-backstop group.
        from ..agent_labeling.panel_routing import REASON_SIZE_GATED
        from ..matching.group_decomposition import (
            build_subproblem_group,
            decompose_group,
        )

        budget = decompose_max_edges or settings.stitch_export_backstop_max_edges
        expanded: list[dict] = []
        for g in selected:
            d = decompose_group(g, budget)
            if not d.is_decomposed:
                if d.n_edges > budget:
                    # Irreducible monster (single biconnected blob): splitting
                    # achieved nothing, so #386's quota-void rule reapplies —
                    # a whole-group verdict on it can never export. Drop it
                    # from the wave unless --include-oversize deliberately
                    # keeps it (calibration; its verdict size-gates to human).
                    if include_oversize:
                        console.print(
                            f"[yellow]Group {g['group_id']}: {d.n_edges} edges but "
                            f"irreducible (single biconnected blob) — kept whole "
                            f"(--include-oversize)[/yellow]"
                        )
                        expanded.append(g)
                    else:
                        console.print(
                            f"[yellow]Group {g['group_id']}: {d.n_edges} edges but "
                            f"irreducible (single biconnected blob) — dropped from "
                            f"the wave (its verdict cannot export; pass "
                            f"--include-oversize to vote it anyway)[/yellow]"
                        )
                    continue
                expanded.append(g)
                continue
            subs = [build_subproblem_group(g, s, len(d.subproblems)) for s in d.subproblems]
            n_over = sum(1 for s in d.subproblems if s.oversized)
            console.print(
                f"[blue]Group {g['group_id']}: {d.n_edges} edges -> "
                f"{len(subs)} sub-problems"
                + (f" ({n_over} still over budget, human-routed)" if n_over else "")
                + "[/blue]"
            )
            expanded.extend(subs)
            parent_entry = {k: v for k, v in g.items() if k != "alternatives"}
            parent_entry["decomposed_parent"] = True
            parent_entry["subproblem_ids"] = [s.subproblem_id for s in d.subproblems]
            parent_entry["decompose_max_edges"] = budget
            decomposed_parents.append(parent_entry)
            decomposition_manifest[str(g["group_id"])] = {
                "n_edges": d.n_edges,
                "max_edges": budget,
                "subproblems": [
                    {
                        "id": s.subproblem_id,
                        "n_edges": s.n_edges,
                        "n_refs": len(s.ref_ids),
                        "n_targets": len(s.target_ids),
                        "n_blocks": s.n_blocks,
                        "oversized": s.oversized,
                        **(
                            {"route": "human_review", "route_reason": REASON_SIZE_GATED}
                            if s.oversized
                            else {}
                        ),
                    }
                    for s in d.subproblems
                ],
            }
        selected = expanded
        if not selected:
            console.print(
                "[red]No groups left after decomposition (all selected groups were "
                "irreducible monsters)[/red]"
            )
            raise typer.Exit(1)

    # Oversized irreducible sub-problems get no pack (they are size-gated to
    # human review, like monster groups today); everything else is packed.
    packable = [g for g in selected if not g.get("subproblem_oversized")]

    # Ensure alternatives present, then fill spatial context (packed groups only).
    for g in packable:
        if "alternatives" not in g:
            g["alternatives"] = generate_top_k_alternatives(
                g.get("edges", []),
                ref_geoms=g.get("ref_geometries", {}),
                target_geoms=g.get("target_geometries", {}),
                k=k_alternatives,
            )
    console.print(f"[blue]Filling spatial context for {len(packable)} groups...[/blue]")
    from .data import _fill_spatial_context

    _fill_spatial_context(packable, dataset)

    name = batch_name or dataset
    batch_dir = output_dir / name
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch = {
        "schema_version": 2,
        "dataset_id": dataset,
        "source_artifacts": source_artifacts,
        "batch_generation_source": batch_generation_source,
        "groups": selected + decomposed_parents,
    }

    # A batch name may be intentionally reused. Remove only obsolete generated
    # evidence directories (identified by their managed pack files) so the new
    # schema-v2 roster cannot accidentally inherit and vote a prior wave.
    current_group_ids = {str(group["group_id"]) for group in batch["groups"]}
    for child in batch_dir.iterdir():
        if (
            child.is_dir()
            and child.name not in current_group_ids
            and any(
                (child / name).exists() for name in ("prompt.txt", "metadata.yaml", "evidence.json")
            )
        ):
            shutil.rmtree(child)
    (batch_dir / "batch.json").write_text(json.dumps(batch))
    if decomposition_manifest:
        (batch_dir / "decomposition.json").write_text(json.dumps(decomposition_manifest, indent=2))
        console.print(f"  Decomposition manifest: {batch_dir / 'decomposition.json'}")
    elif (batch_dir / "decomposition.json").exists():
        (batch_dir / "decomposition.json").unlink()

    console.print(f"[blue]Generating evidence packs -> {batch_dir}[/blue]")
    generated = generate_stitch_evidence(
        batch, batch_dir, group_ids=[str(g["group_id"]) for g in packable]
    )
    console.print(f"[green]Generated {len(generated)} evidence packs[/green]")

    # Confirm the pack count: a packable group that produced no pack (no options)
    # is silently unvoted. Benign for a normal group, but a missing decomposed
    # sub-problem pack (#367 Mode B) permanently blocks its parent's whole-group
    # recomposition with no downstream signal — so refuse rather than ship a
    # decomposition that can never fully recompose.
    missing_subs, missing_other = missing_evidence_packs(packable, generated)
    if missing_other:
        console.print(
            f"[yellow]WARNING: {len(missing_other)} requested group(s) produced no "
            f"evidence pack (no options) and will not be voted: "
            f"{', '.join(missing_other[:5])}[/yellow]"
        )
    if missing_subs:
        detail = ", ".join(f"{sid} (parent {pgid})" for sid, pgid in missing_subs[:5])
        console.print(
            f"[red]ERROR: {len(missing_subs)} decomposed sub-problem(s) produced no "
            f"evidence pack, so their parent group(s) can never fully recompose to a "
            f"whole-group label: {detail}[/red]"
        )
        raise typer.Exit(1)

    console.print(f"  Batch dir: {batch_dir}")
    console.print("Next: crosswalk agent stitch-run --batch " + str(batch_dir))


@agent_app.command("stitch-run")
def run_stitch_panel(
    batch_dir: Path = typer.Option(..., "--batch", "-b", help="Batch dir with evidence packs"),
    group_ids: str = typer.Option(None, "--group-ids", help="Comma-separated subset to run"),
    timeout: int = typer.Option(
        None,
        "--timeout",
        help="Per-provider timeout (s). Default: each provider spec's own timeout "
        "(480 for the kimi/Kimi K2.6 and muse/Muse Spark voters, whose thinking "
        "runs long on large packs), else 240. An explicit value overrides both.",
    ),
    invocation_budget: float = typer.Option(
        300.0,
        "--invocation-budget",
        help="Seconds to back off + retry a down provider (quota/rate-limit/network) "
        "before hard-failing the run. Worst-case wall time is this + one --timeout.",
    ),
    limit: int = typer.Option(0, "--limit", "-l", help="Max groups (0=all)"),
    # Per-provider model/effort flags are TRUE overrides (default None -> the
    # named panel's spec). An unconditional typer default would silently clobber
    # the spec for every composition — the #397 opencode fix, now required for
    # every provider because specs differ across panel eras (codex is
    # gpt-5.6-sol/high on v7-candidate, gpt-5.6-terra/medium on the
    # default/v5/v4/v6 panels, and gpt-5.5/low on the v3 panels).
    claude_model: str = typer.Option(
        None,
        "--claude-model",
        help="Override the claude voter's model (default: the named panel's spec — "
        "claude-opus-4-8 on all current panels).",
    ),
    claude_effort: str = typer.Option(
        None,
        "--claude-effort",
        help="Override the claude voter's reasoning effort (default: the named "
        "panel's spec — high on v7-candidate and medium on the other current "
        "panels).",
    ),
    codex_model: str = typer.Option(
        None,
        "--codex-model",
        help="Override the codex voter's model (default: the named panel's spec — "
        "gpt-5.6-sol on v7-candidate, gpt-5.6-terra on default/v5/v4/v6, "
        "and gpt-5.5 on the v3-era panels).",
    ),
    codex_effort: str = typer.Option(
        None,
        "--codex-effort",
        help="Override the codex voter's reasoning effort (default: the named "
        "panel's spec — high on v7-candidate, medium on default/v5/v4/v6, "
        "and low on the v3-era panels).",
    ),
    agy_model: str = typer.Option(
        None,
        "--agy-model",
        help="Override the agy voter's model string (default: the named panel's "
        "spec — 'Gemini 3.5 Flash (Medium)'; agy only sits on the v3-era panels).",
    ),
    panel_name: str = typer.Option(
        "default",
        "--panel",
        help="Named panel config: 'default'/'v5' (the blessed v5 QUAD: claude + "
        "codex/gpt-5.6-terra + kimi/Kimi K2.6 + muse/Muse Spark 1.1, paired "
        "with the quorum consensus rule), 'v4' (the former 3-seat default: "
        "claude + codex/gpt-5.6-terra + kimi/Kimi), 'v3'/'v2' (the "
        "claude+codex+agy default before that; its exports stamp the v3 "
        "labelers), 'v3-candidate' (v3 + a 4th opencode/Qwen3-VL voter), "
        "'no-agy' (v3 with agy swapped for opencode/Qwen — quota-outage "
        "fallback), 'v4-candidate' (the #397 validation composition: v3 with "
        "agy swapped for Kimi, codex still gpt-5.5), 'meta-candidate' (the v4 "
        "trio with kimi swapped for muse — superseded by v5), or "
        "'quad-candidate' (alias of the v5 default: the calibration "
        "composition that became v5). Non-blessed compositions are refused by "
        "stitch-export without --allow-nonstandard-panel. 'v6-candidate' is the "
        "lean Claude + Codex + Muse trio; it remains nonstandard until its "
        "calibration gate passes. 'v6-agy-calibration' and "
        "'v6-flex-calibration' isolate the two Gemini routes on the same logical "
        "fourth seat for experimental parity testing. 'v7-candidate' is the "
        "canonical-rubric high-effort Claude + Codex/gpt-5.6-sol + Muse replay "
        "panel; it remains nonstandard pending manual review.",
    ),
    opencode_model: str = typer.Option(
        None,
        "--opencode-model",
        help="Override the 'opencode'-named voter's model string. Post-rename this "
        "seat is ONLY the residual v3-era Qwen3-VL voter (in v3-candidate and "
        "no-agy; default: the named panel's spec — "
        "openrouter/qwen/qwen3-vl-235b-a22b-instruct). NO-OP on "
        "default/v5/v4/v4-candidate/quad-candidate/meta-candidate, which have no "
        "'opencode' seat — the Kimi seat is now named 'kimi' (use --kimi-model) "
        "and Muse is 'muse' (use --muse-model).",
    ),
    kimi_model: str = typer.Option(
        None,
        "--kimi-model",
        help="Override the 'kimi' voter's model string (default: the named panel's "
        "spec — openrouter/moonshotai/kimi-k2.6 on "
        "default/v5/v4/v4-candidate/quad-candidate). The Kimi seat carries its own "
        "'kimi' provider name (distinct from the opencode-transport 'opencode'/Qwen "
        "seat and from 'muse'), so this targets ONLY Kimi; a no-op on panels "
        "without a Kimi seat (v3/v2/v3-candidate/no-agy/meta-candidate).",
    ),
    muse_model: str = typer.Option(
        None,
        "--muse-model",
        help="Override the 'muse' voter's model string (default: the named panel's "
        "spec — meta/muse-spark-1.1 on the default/v5 quad, meta-candidate, and "
        "the v6/v7 candidates). Distinct "
        "from --kimi-model: Muse and Kimi both ride the opencode transport but "
        "carry separate provider names ('muse'/'kimi') so both can be seated "
        "(the v5 quad) and pinned independently.",
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
    """Run the consensus panel (default v5 quad: claude + codex + kimi + muse) on a batch.

    Writes votes.csv (every raw vote — audit data) and consensus.csv (per-group
    routing) into the batch dir. Writes NOTHING into labels/. Routing uses the
    quorum rule: all valid votes agreeing with >=3 valid auto-accepts (a 3-of-4
    accept over an abstention is stamped "quorum", distinct from "unanimous").

    Examples:
        crosswalk agent stitch-run --batch data/agents/stitching/batches/us_boston_streets
        crosswalk agent stitch-run --batch <dir> --panel v4  # former 3-seat default
        crosswalk agent stitch-run --batch <dir> --panel v3  # claude+codex+agy era
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

    # Build the panel from the named config, applying only the per-provider
    # model/effort overrides that were actually passed (None = keep the spec).
    # The spec's own timeout is carried through untouched — the CLI --timeout
    # (when passed) overrides it downstream via resolve_timeout.
    overrides: dict[str, dict[str, str | None]] = {
        "claude": {"model": claude_model, "effort": claude_effort},
        "codex": {"model": codex_model, "effort": codex_effort},
        "agy": {"model": agy_model},
        # Kimi and Muse ride the same opencode transport but carry DISTINCT
        # provider names ("kimi"/"muse"), so --kimi-model and --muse-model pin
        # them independently (both seated on quad-candidate). "opencode" is now
        # only the residual v3-era Qwen seat (v3-candidate/no-agy), so
        # --opencode-model is a no-op on the v4/quad/meta panels.
        "opencode": {"model": opencode_model},
        "kimi": {"model": kimi_model},
        "muse": {"model": muse_model},
    }

    def _with_overrides(p: ProviderSpec) -> ProviderSpec:
        ov = {k: v for k, v in overrides.get(p.name, {}).items() if v is not None}
        return ProviderSpec(
            name=p.name,
            model=ov.get("model", p.model),
            effort=ov.get("effort", p.effort),
            timeout=p.timeout,
            # Carry the opencode agent through the rebuild — dropping it would
            # strip the tool-less ``vote`` agent that both the Kimi and Muse seats
            # run under (e.g. on the v4/meta-candidate CLI path), reverting them to
            # the agentic ``build`` default and its answer-stalling tool loop.
            opencode_agent=p.opencode_agent,
            # Route order is ballot-changing provenance for the v6 Gemini
            # seat; preserve it through every CLI model/effort override.
            routes=p.routes,
        )

    try:
        named_panel = get_panel(panel_name)
    except ValueError as e:
        # Unknown panel names hard-error (era-load-bearing choice, see
        # get_panel) — surface the valid names instead of a traceback.
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    panel = [_with_overrides(p) for p in named_panel]
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
        console.print(f"  Option-coverage gap: {oc['gap']}/{oc['n_opt']} ({oc['gap_rate']:.0%})")
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
        f"\n### Option-coverage gap\n\n{oc['gap']}/{oc['n_opt']} "
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


@agent_app.command("panel-stats")
def panel_stats(
    dataset: str = typer.Option(
        None, "--dataset", "-d", help="Restrict to one dataset (default: all committed votes)"
    ),
    data_root: Path = typer.Option(
        Path("."), "--data-root", help="Repo root holding labels/votes/"
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit nonzero if ANY voter trips an alarm (for CI/cron). Off by default.",
    ),
):
    """Per-voter bias monitor for the stitch panel.

    Reads committed vote provenance (labels/votes/dataset=*/{votes,consensus}.csv)
    and prints a per-voter table plus any tripped alarms. Makes voter defects LOUD
    instead of found by accident — e.g. voter `agy` voting the first-listed option
    "A" in 11/12 valid ballots at a flat 0.95 confidence (POSITION_ANCHOR +
    CONSTANT_CONFIDENCE). This is a MONITOR by design: we do NOT shuffle option
    letters (that would hide the anchor), we surface it.

    Examples:
        crosswalk agent panel-stats
        crosswalk agent panel-stats --dataset us_boston_streets
        crosswalk agent panel-stats --strict   # nonzero exit on any alarm (CI)
    """
    from rich.table import Table

    from ..agent_labeling.panel_monitor import (
        CONSTANT_CONFIDENCE,
        POSITION_ANCHOR,
        compute_voter_stats,
        load_vote_provenance,
    )
    from ..config import settings

    votes_df, consensus_df = load_vote_provenance(data_root, dataset=dataset)
    if len(votes_df) == 0:
        console.print(f"[yellow]No committed votes found under {data_root}/labels/votes[/yellow]")
        raise typer.Exit(0)

    stats = compute_voter_stats(votes_df, consensus_df)
    scope = dataset or "all datasets"
    n_datasets = votes_df["dataset"].nunique() if "dataset" in votes_df.columns else 1

    def _f(x: float, pct: bool = False) -> str:
        if x != x:  # NaN
            return "-"
        return f"{x:.0%}" if pct else f"{x:.3f}"

    table = Table(
        title=f"Panel voter stats — {scope} ({len(votes_df)} votes across {n_datasets} dataset(s))"
    )
    table.add_column("voter", style="bold")
    table.add_column("model")
    table.add_column("valid", justify="right")
    table.add_column("modal pos", justify="right")
    table.add_column("share", justify="right")
    table.add_column("dissent", justify="right")
    table.add_column("none", justify="right")
    table.add_column("abstain", justify="right")
    table.add_column("conf μ", justify="right")
    table.add_column("conf σ", justify="right")
    table.add_column("conf min/max", justify="right")
    table.add_column("calib gap", justify="right")
    table.add_column("alarms", style="red")

    for s in sorted(stats, key=lambda s: s.provider):
        alarm_txt = ("[red]" + " ".join(s.alarms) + "[/red]") if s.alarms else "[green]ok[/green]"
        share_txt = _f(s.modal_position_share, pct=True)
        if POSITION_ANCHOR in s.alarms:
            share_txt = f"[red]{share_txt}[/red]"
        std_txt = _f(s.conf_std)
        if CONSTANT_CONFIDENCE in s.alarms:
            std_txt = f"[red]{std_txt}[/red]"
        table.add_row(
            s.provider,
            s.model or "-",
            str(s.n_valid),
            f"{s.modal_letter}",
            share_txt,
            _f(s.dissent_rate, pct=True),
            str(s.n_none),
            _f(s.abstain_rate, pct=True),
            _f(s.conf_mean),
            std_txt,
            f"{_f(s.conf_min)}/{_f(s.conf_max)}",
            _f(s.calibration_gap),
            alarm_txt,
        )

    console.print(table)

    tripped = [(s.provider, a) for s in stats for a in s.alarms]
    if tripped:
        console.print("\n[bold red]Tripped alarms[/bold red]")
        for s in stats:
            for a in s.alarms:
                if a == POSITION_ANCHOR:
                    console.print(
                        f"  [red]{a}[/red] {s.provider}: "
                        f"{s.modal_position_share:.0%} of {s.n_valid} valid ballots on "
                        f"position {s.modal_letter} (threshold "
                        f"{settings.panel_monitor_position_anchor_share:.0%}) — picking by slot"
                    )
                elif a == CONSTANT_CONFIDENCE:
                    console.print(
                        f"  [red]{a}[/red] {s.provider}: confidence σ={s.conf_std:.4f} over "
                        f"{s.n_valid} valid ballots (threshold "
                        f"{settings.panel_monitor_constant_confidence_std}) — a rubber stamp"
                    )
        if strict:
            raise typer.Exit(1)
    else:
        console.print("\n[green]No alarms tripped.[/green]")


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
    for the sidecar group they correspond to. SET-semantics labels (group
    membership assertions, no specific edges) are scored separately and
    reported alongside as best-option membership/boundary-precision/coverage —
    they used to be silently dropped from this metric, which hid large-group
    failures where every generated option uses the full ref/target set. Reads
    the sidecar and labels READ ONLY; runs no provider. Useful as a
    before/after gate on generator changes.

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

    # SET-semantics labels (group MEMBERSHIP assertions, no specific edges):
    # scored on membership/boundary/coverage against the best generated
    # option, reported alongside (not folded into) the pair numbers above.
    console.print(
        f"\n[bold]Set-label expressibility: {dataset}[/bold]  "
        f"(dropped={s['n_dropped']} — pair reject-all / empty-membership rows)"
    )
    console.print(
        f"  set-settled={s['n_set_settled']}  set-clean-recoverable={s['n_set_recoverable']}  "
        f"set-covered={s['n_set_covered']}"
    )
    set_rate = s["set_expressibility"]
    console.print(
        f"  SET EXPRESSIBILITY = [green]{set_rate:.1%}[/green]" if set_rate is not None else "  n/a"
    )
    bp = s["set_mean_best_boundary_precision"]
    cov = s["set_mean_best_coverage"]
    console.print(
        f"  mean best-option boundary precision: {bp:.3f}  |  mean best-option coverage: {cov:.3f}"
        if bp is not None
        else "  mean best-option boundary precision: n/a"
    )
    console.print(
        f"  inexpressible (recoverable but no option nails membership): {s['n_set_misses']}"
    )
    if show_misses:
        for m in sorted(report.set_misses, key=lambda x: x.best_boundary_precision)[:show_misses]:
            console.print(
                f"    - label {m.label_group_id} -> group {m.sidecar_group_id} "
                f"({m.match_type}): {m.n_ref_members} refs / {m.n_target_members} targets, "
                f"best boundary precision {m.best_boundary_precision:.3f}, "
                f"coverage {m.best_coverage:.3f}, {m.n_options} options"
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
    allow_nonstandard_panel: bool = typer.Option(
        False,
        "--allow-nonstandard-panel",
        help=(
            "Export even when a batch's votes.csv (provider, model) voter set "
            "matches no blessed panel composition (v5: claude+codex/gpt-5.6-terra"
            "+kimi/Kimi+muse/Muse Spark; v4: that trio without muse; v3: "
            "claude+codex/gpt-5.5+agy). The known v6 and v7 candidates also "
            "require this flag until calibration promotes them. Labels are "
            "still stamped with the panel_* labelers, so only use this "
            "after an explicit provenance decision. A composition with no known "
            "era additionally needs --stamp-era to say WHICH labeler generation "
            "to mint."
        ),
    ),
    stamp_era: str = typer.Option(
        None,
        "--stamp-era",
        help=(
            "Declare the labeler era ('v3', 'v4', 'v5', 'v6', or 'v7') for batches whose "
            "composition resolves to NO era. FILL-IN only: batches that "
            "resolve to a blessed or known-historical era always keep their "
            "own era — this flag never re-stamps them. Required when any "
            "batch matches no known era — export refuses to guess which "
            "panel_* generation unknown provenance belongs to."
        ),
    ),
):
    """Export accepted panel consensus into human-equivalent stitching labels.

    ``auto_accept`` groups export their chosen edge set — labeler
    ``panel_unanimous_v5`` for a fully unanimous
    accept, or the DISTINCT ``panel_quorum_v5`` for a quorum accept (all valid
    votes agree over an abstention, v5 rule). Panel ``NONE`` is never exported
    directly because it can mean reject-all, no exact offered option, or
    insufficient evidence; it stays in human review for explicit confirmation.
    Gates are applied in
    order and reported per group: (a) routing, (b) size, (c) class-consistency,
    (d) exactness-preserving sliver review, (e) human precedence. Rows upsert by group_id
    (idempotent). Provenance is gated on (provider, model) voter pairs: batches
    matching an older blessed era exactly still export, stamped with that era's
    labelers (as do the known-historical v3 transport-swap batches); anything
    matching no blessed composition is refused unless
    ``--allow-nonstandard-panel`` is passed (composition is provenance), and a
    composition with no known era additionally requires ``--stamp-era``.

    Examples:
        crosswalk agent stitch-export \\
            -b data/agents/stitching/batches/us_boston_streets_phase2 \\
            -b data/agents/stitching/batches/us_boston_streets_phase3
    """
    from ..agent_labeling.stitch_export import (
        LABELERS_BY_ERA,
        REASON_HUMAN_PRECEDENCE,
        batch_panel_era,
        filter_exportable_batch_dirs,
        nonstandard_panel_batches,
        plan_exports,
        write_exports,
        write_vote_provenance,
    )

    if stamp_era is not None and stamp_era not in LABELERS_BY_ERA:
        console.print(
            f"[red]--stamp-era must be one of {sorted(LABELERS_BY_ERA)}, got {stamp_era!r}[/red]"
        )
        raise typer.Exit(1)

    # Support both repeatable --batch and comma-separated values.
    batch_dirs: list[Path] = []
    for raw in batches:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                batch_dirs.append(Path(part))

    # Calibration batches carrying a .no-export marker never mint labels — only
    # the human review queue reads them (panel_routing honors no marker). Drop
    # them up front, BEFORE the era/composition pre-checks, so a calibration
    # batch cannot block a real export and vote-provenance archival skips it too.
    exportable = filter_exportable_batch_dirs(batch_dirs)
    for bd in batch_dirs:
        if bd not in exportable:
            console.print(f"[yellow]Skipping {bd.name}: .no-export marker present[/yellow]")
    batch_dirs = exportable
    if not batch_dirs:
        console.print("[yellow]No exportable batch dirs (all carry a .no-export marker)[/yellow]")
        raise typer.Exit(0)

    # Era stamping and vote-provenance archival key batches by BASENAME —
    # duplicates would mis-attribute one dir's era/ballots to another. Refuse
    # up front, before any planning.
    names = [bd.name for bd in batch_dirs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        console.print(
            f"[red]Batch dirs have duplicate basenames {dupes} — era stamping "
            f"and vote archival key on the basename and would collapse them. "
            f"Pass uniquely-named batch dirs.[/red]"
        )
        raise typer.Exit(1)

    for bd in batch_dirs:
        if not (bd / "consensus.csv").exists():
            console.print(f"[red]No consensus.csv in {bd}[/red]")
            raise typer.Exit(1)
        if not (bd / "batch.json").exists():
            console.print(
                f"[yellow]Warning: no batch.json in {bd} — sliver detection "
                "and edge-overlap precedence degrade for its groups[/yellow]"
            )

    offending = nonstandard_panel_batches(batch_dirs)
    if offending and not allow_nonstandard_panel:
        for name, voters in sorted(offending.items()):
            voters_str = ", ".join(f"{p}:{m or '<no model>'}" for p, m in sorted(voters))
            console.print(
                f"[red]Batch {name} was voted by a nonstandard panel "
                f"({voters_str}) — refusing to stamp its labels "
                f"with the panel_* labelers. Re-run with "
                f"--allow-nonstandard-panel only after an explicit provenance "
                f"decision.[/red]"
            )
        raise typer.Exit(1)

    # Per-batch labeler era, resolved from votes.csv. --stamp-era is a FILL-IN
    # for batches that resolve to NO era (resolution first, matching
    # plan_exports): a genuinely-resolved batch always keeps its own era, so
    # the flag can never re-stamp a blessed-v4 batch as v3 in a mixed run. A
    # batch with no known era and no --stamp-era is refused — export never
    # guesses which panel_* generation to mint.
    resolved = {bd: batch_panel_era(bd) for bd in batch_dirs}
    eras = {bd: (resolved[bd] or stamp_era) for bd in batch_dirs}
    era_less = sorted(bd.name for bd, era in eras.items() if not era)
    if era_less:
        console.print(
            f"[red]Batches {era_less} match no blessed or known-historical panel "
            f"era — refusing to guess which panel_* labeler generation to mint. "
            f"Re-run with an explicit --stamp-era "
            f"{{{', '.join(sorted(LABELERS_BY_ERA))}}} after a provenance "
            f"decision.[/red]"
        )
        raise typer.Exit(1)

    report = plan_exports(
        batch_dirs,
        dataset,
        labels_dir,
        max_edges=max_edges,
        stamp_era=stamp_era,
    )

    decomp_note = (
        f", {report.n_decomposed_parents} decomposed parents "
        f"({report.n_subproblem_rows} sub-problem rows)"
        if report.n_decomposed_parents
        else ""
    )
    console.print(
        f"[bold]Panel export: {report.n_total_groups} merged groups, "
        f"{report.n_auto_accept} auto_accept candidates{decomp_note}[/bold]"
    )
    console.print(f"  Batches (in precedence order): {', '.join(b.name for b in batch_dirs)}")
    # Per-batch resolved era + the labeler tags a (re-)export will mint, so an
    # operator re-exporting an old batch SEES what it will be stamped as.
    for bd in batch_dirs:
        era = eras[bd]
        tags = LABELERS_BY_ERA[era]
        minted = [tags.accept, tags.decomposed]
        # Quorum labeler variants exist from v5 on (quorum consensus rule).
        minted += [t for t in (tags.accept_quorum, tags.decomposed_quorum) if t]
        # Mark only the batches that actually took the fill-in (unresolved
        # composition + --stamp-era), not every line whenever the flag is set.
        override = " (--stamp-era fill-in)" if stamp_era and not resolved[bd] else ""
        # Parentheses, not square brackets: rich would swallow [tags] as markup.
        console.print(f"  Stamp era: {bd.name} -> {era}{override} (mints {' / '.join(minted)})")

    # Per-group report.
    for g in report.groups:
        recomposed = (
            f" (recomposed {g.n_subproblems_resolved}/{g.n_subproblems} sub-problems)"
            if g.from_decomposition
            else ""
        )
        if g.exported:
            console.print(
                f"  [green]EXPORT[/green] {g.group_id} [{g.source_batch}] "
                f"{g.match_type} {g.n_edges_final} edges "
                f"conf={g.mean_confidence:.3f}{recomposed}"
            )
        else:
            extra = ""
            if g.reason == REASON_HUMAN_PRECEDENCE and g.human_group_id:
                extra = f" (human {g.human_group_id})"
            elif g.reason == "over_max_edges":
                extra = f" ({g.n_edges_raw} > {max_edges})"
            elif g.reason == "contains_sliver":
                extra = " (exact selection contains a SLIVER-tagged edge; human review required)"
            console.print(
                f"  [yellow]SKIP[/yellow]   {g.group_id} [{g.source_batch}] -> "
                f"{g.reason}{extra}{recomposed}"
            )

    console.print(
        f"\n[bold]Summary:[/bold] {len(report.exported)} exported, {len(report.skipped)} skipped"
    )
    by_reason = report.skipped_by_reason()
    if by_reason:
        console.print("  Skips: " + ", ".join(f"{r}={n}" for r, n in sorted(by_reason.items())))

    if dry_run:
        console.print("[cyan]Dry run — no labels written.[/cyan]")
        return

    # Provenance is part of the label contract, not a best-effort afterthought:
    # archive the ballots, consensus and exact displayed menu before minting a
    # panel label. If this fails, no durable label is written without its audit
    # trail. An archive can safely precede a later label-store failure; it is
    # evidence of a completed panel judgment, not itself a label.
    try:
        n_votes, n_consensus = write_vote_provenance(
            batch_dirs,
            dataset,
            require_evidence=True,
        )
    except Exception as e:  # noqa: BLE001 - surfaced as an operator-facing CLI failure
        console.print(
            f"[red]Evidence-provenance archival failed ({e}); refusing to write "
            f"panel labels without their exact displayed menu.[/red]"
        )
        raise typer.Exit(1) from e
    console.print(
        f"[green]Archived vote provenance: {n_votes} ballots, {n_consensus} consensus "
        f"rows plus exact evidence menus to labels/votes/dataset={dataset}[/green]"
    )

    written = write_exports(report, dataset, labels_dir)
    console.print(f"[green]Wrote {written} panel labels to {labels_dir}/dataset={dataset}[/green]")


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
