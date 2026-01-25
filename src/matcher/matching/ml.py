"""Machine learning-based matcher using gradient boosted trees.

This module provides XGBoost-based matching trained on labeled data.
The model learns to classify road segment pairs as match/no_match
based on geometric and semantic features.

Training Data Format:
--------------------
Uses labels from Hive-partitioned CSVs in labels/ which contains:
- gers_id: Overture reference segment ID (GERS ID)
- target_id: Target segment identifier
- label: Human label (match, no_match, unsure; legacy: associated)
- Feature columns: hausdorff_distance, buffer_iou, etc.

Model Architecture:
------------------
- XGBoost classifier with binary (match vs no_match) or multiclass output
- Features: Normalized geometric + semantic scores (same as rule-based)
- Handles class imbalance via scale_pos_weight or class_weight
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, cross_val_score

from ..config import (
    DEFAULT_TOPOLOGY_FEATURES,
    FEATURE_COLUMNS,
    MAX_DISTANCE_METERS,
    SEMANTIC_FEATURES,
)
from .rules import MatchDecision, MatchResult

# Module-level globals for multiprocessing worker data
_worker_data = None


def _init_worker(data):
    """Initialize worker process with shared data."""
    global _worker_data
    _worker_data = data


def _compute_single_feature(args):
    """Compute features for a single candidate pair (worker function).

    This function delegates to compute_pair_features() to ensure consistency
    between the ML scoring pipeline and the backfill pipeline (training data).

    Returns a dict of features, or a dict with error defaults if computation fails.
    """
    from ..features.compute import (
        _get_error_features,
        compute_graphlet_similarity,
        compute_pair_features,
    )

    ref_idx, target_idx = args

    try:
        # Extract data from worker globals
        ref_geom = _worker_data["ref_geoms"][ref_idx]
        target_geom = _worker_data["target_geoms"][target_idx]

        # Get pre-computed alignment if available
        alignment = _worker_data.get("alignments", {}).get((ref_idx, target_idx))

        # Get pre-computed endpoint features for target segment
        endpoint_features = _worker_data.get("endpoint_features", {}).get(target_idx)

        # Get pre-computed topology features
        ref_topology = _worker_data.get("ref_topology", {}).get(ref_idx)
        target_topology = _worker_data.get("target_topology", {}).get(target_idx)

        # Compute graphlet similarity using precomputed graph data
        # Pass alignment for alignment-aware connector lookup
        ref_graphlet_data = _worker_data.get("ref_graphlet_data")
        target_graphlet_data = _worker_data.get("target_graphlet_data")
        ref_seg_id = str(_worker_data["ref_ids"][ref_idx])
        target_seg_id = str(_worker_data["target_ids"][target_idx])
        graphlet_features = compute_graphlet_similarity(
            ref_seg_id, target_seg_id, ref_graphlet_data, target_graphlet_data, alignment
        )

        # Delegate to shared compute_pair_features function
        # This ensures consistency with backfill pipeline (training data generation)
        # Pass graphlet_data for alignment-aware topology computation (partial overlaps)
        features = compute_pair_features(
            ref_geom,
            target_geom,
            _worker_data["ref_names"][ref_idx],
            _worker_data["target_names"][target_idx],
            _worker_data["ref_classes"][ref_idx],
            _worker_data["target_classes"][target_idx],
            _worker_data["ref_subclasses"][ref_idx],
            _worker_data["target_subclasses"][target_idx],
            endpoint_features=endpoint_features,
            ref_topology=ref_topology,
            target_topology=target_topology,
            alignment=alignment,
            graphlet_features=graphlet_features,
            ref_graphlet_data=ref_graphlet_data,
            target_graphlet_data=target_graphlet_data,
            ref_seg_id=ref_seg_id,
            target_seg_id=target_seg_id,
        )
        features["_error"] = None
        return features

    except Exception as e:
        # Return error marker with default values (will result in low confidence)
        error_features = _get_error_features()
        error_features["_error"] = str(e)
        return error_features


def select_model_for_dataset(
    target_gdf,
    full_model_path: str | None = None,
    geom_only_model_path: str | None = None,
    name_column: str = "names",
    min_name_coverage: float = 0.5,
) -> str:
    """Select model based on dataset attributes.

    Automatically chooses between full model (with semantic features) and
    geometry-only model based on the target dataset's name coverage.

    Args:
        target_gdf: Target GeoDataFrame
        full_model_path: Path to full model with semantic features
        geom_only_model_path: Path to geometry-only model
        name_column: Column name for segment names
        min_name_coverage: Minimum fraction of rows with non-null names to use full model

    Returns:
        Path to selected model
    """
    from ..config import settings

    # Use configured paths if not explicitly provided
    if full_model_path is None:
        full_model_path = str(settings.model_path)
    if geom_only_model_path is None:
        geom_only_model_path = str(settings.model_geom_only_path)

    # Check for name column variations
    has_names = name_column in target_gdf.columns or "name" in target_gdf.columns
    name_col = name_column if name_column in target_gdf.columns else "name"

    if has_names:
        # Calculate effective name coverage (non-null AND non-empty strings)
        non_empty_mask = target_gdf[name_col].notna() & (
            target_gdf[name_col].astype(str).str.strip() != ""
        )
        effective_coverage = non_empty_mask.mean()
    else:
        effective_coverage = 0.0

    logger.info(f"Target dataset name coverage: {effective_coverage:.1%}")

    # Check if geometry-only model exists
    geom_only_exists = Path(geom_only_model_path).exists()

    if effective_coverage >= min_name_coverage:
        logger.info(
            f"Using full model (name coverage {effective_coverage:.1%} >= {min_name_coverage:.0%})"
        )
        return full_model_path
    elif geom_only_exists:
        logger.info(
            f"Using geometry-only model (name coverage {effective_coverage:.1%} < {min_name_coverage:.0%})"
        )
        return geom_only_model_path
    else:
        logger.warning(
            f"Geometry-only model not found at {geom_only_model_path}, falling back to full model"
        )
        return full_model_path


def create_segment_groups(df: pd.DataFrame) -> pd.Series:
    """Create group IDs for segment-aware train/test splitting.

    Uses Union-Find to ensure pairs sharing any segment are in the same group.
    This prevents data leakage where the model sees a segment during training
    and then evaluates on the same segment.

    Args:
        df: DataFrame with 'gers_id' and 'target_id' columns

    Returns:
        Series of group IDs (one per row in df)

    Raises:
        ValueError: If gers_id or target_id columns contain null values
    """
    from collections import defaultdict

    # Validate no null values in segment ID columns
    if df["gers_id"].isna().any() or df["target_id"].isna().any():
        raise ValueError("gers_id and target_id columns must not contain null values")

    # Union-Find implementation
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Map segments to pairs (using DataFrame index)
    segment_to_pairs = defaultdict(list)
    for idx in df.index:
        row = df.loc[idx]
        segment_to_pairs[row["gers_id"]].append(idx)
        segment_to_pairs[row["target_id"]].append(idx)

    # Union pairs that share a segment
    for _segment, pair_idxs in segment_to_pairs.items():
        for i in range(1, len(pair_idxs)):
            union(pair_idxs[0], pair_idxs[i])

    # Return group for each row
    return pd.Series([find(idx) for idx in df.index], index=df.index)


def segment_aware_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split data ensuring no segment appears in both train and test sets.

    Uses Union-Find to group pairs that share segments, then splits by group.
    This prevents data leakage where the model trains on a segment and then
    evaluates on the same segment in a different pair.

    Args:
        df: DataFrame with 'gers_id' and 'target_id' columns
        test_size: Fraction of data to use for testing (0.0 to 1.0)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, test_indices) as numpy arrays

    Raises:
        ValueError: If test_size is not in range [0.0, 1.0]
    """
    # Validate test_size
    if not 0.0 <= test_size <= 1.0:
        raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")

    # Handle empty DataFrame
    if len(df) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Handle test_size=0.0 (no split, all training)
    if test_size == 0.0:
        return np.arange(len(df)), np.array([], dtype=int)

    groups = create_segment_groups(df)
    n_groups = groups.nunique()

    # Need at least 2 groups to split
    if n_groups < 2:
        logger.warning(
            f"Only {n_groups} segment group(s) found - cannot split. "
            "All pairs are transitively connected. Placing all in training set."
        )
        return np.arange(len(df)), np.array([], dtype=int)

    # Use GroupShuffleSplit to split by group
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=groups))

    logger.info(
        f"Segment-aware split: {len(train_idx)} train, {len(test_idx)} test "
        f"across {n_groups} groups"
    )

    return train_idx, test_idx


class MLMatcher:
    """Machine learning-based matcher using gradient boosted trees."""

    def __init__(self, model_path: str | None = None, auto_select: bool = False):
        """Initialize the ML matcher.

        Args:
            model_path: Path to trained model (optional)
            auto_select: If True, defer model loading until score_candidates is called
                        so that model can be selected based on target dataset
        """
        self.model = None
        self.model_path = model_path
        self.feature_names = FEATURE_COLUMNS.copy()
        self.feature_medians = {}  # For imputing missing values during inference
        self.label_encoder = {"match": 1, "no_match": 0, "associated": 2}
        self.label_decoder = {1: "match", 0: "no_match", 2: "associated"}
        self.is_binary = True  # Track if model is binary or multiclass
        self._auto_select = auto_select

        if model_path and not auto_select:
            self.load_model(model_path)

    def load_model(self, path: str) -> None:
        """Load a trained model from disk.

        Args:
            path: Path to model file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data.get("feature_names", FEATURE_COLUMNS.copy())
        self.feature_medians = data.get("feature_medians", {})
        self.label_encoder = data.get("label_encoder", self.label_encoder)
        self.label_decoder = data.get("label_decoder", self.label_decoder)
        self.is_binary = data.get("is_binary", True)
        logger.info(f"Loaded model from {path}")

    def save_model(self, path: str) -> None:
        """Save the trained model to disk.

        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "feature_medians": self.feature_medians,
            "label_encoder": self.label_encoder,
            "label_decoder": self.label_decoder,
            "is_binary": self.is_binary,
        }
        joblib.dump(data, path)
        logger.info(f"Saved model to {path}")

    def train(
        self,
        labels_dir: str = "labels",
        binary: bool = True,
        test_size: float = 0.2,
        exclude_semantic: bool = False,
        exclude_datasets: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Train the model on labeled data.

        Args:
            labels_dir: Path to Hive-partitioned labels directory
            binary: If True (default), train binary classifier (match vs non-match)
                   If False, train multiclass (legacy, includes associated)
            test_size: Fraction of data to hold out for testing
            exclude_semantic: If True, exclude semantic features (name_*, class_similarity)
                             for training a geometry-only model
            exclude_datasets: List of dataset names to exclude from training
                             (for leave-one-out cross-validation)
            **kwargs: Additional XGBoost parameters

        Returns:
            Dictionary of training metrics
        """
        try:
            import xgboost as xgb
        except ImportError as err:
            raise ImportError(
                "XGBoost is required for ML training. "
                "Install it with: pip install 'matcher[ml]' or pip install xgboost"
            ) from err

        from ..labeling.label_store import LabelStore

        self.is_binary = binary

        # Set feature columns based on exclude_semantic
        if exclude_semantic:
            self.feature_names = [f for f in FEATURE_COLUMNS if f not in SEMANTIC_FEATURES]
            logger.info(
                f"Training geometry-only model with {len(self.feature_names)} features (excluding semantic)"
            )
        else:
            self.feature_names = FEATURE_COLUMNS.copy()

        # Load all partitions using LabelStore
        df = LabelStore.load_all(Path(labels_dir))
        logger.info(
            f"Loaded {len(df)} labeled pairs from {df['dataset'].nunique() if 'dataset' in df.columns else 1} datasets"
        )

        # Exclude specified datasets (for leave-one-out evaluation)
        if exclude_datasets:
            before_count = len(df)
            df = df[~df["dataset"].isin(exclude_datasets)].copy()
            excluded_count = before_count - len(df)
            logger.info(f"Excluded {excluded_count} labels from datasets: {exclude_datasets}")
            logger.info(f"Training on {len(df)} labels from {df['dataset'].nunique()} datasets")

        # Filter to only valid labels (exclude unsure, skip, and any unexpected values)
        valid_labels = {"match", "no_match", "associated"}
        invalid_mask = ~df["label"].isin(valid_labels)
        if invalid_mask.any():
            invalid_labels = df.loc[invalid_mask, "label"].value_counts().to_dict()
            logger.warning(f"Filtering out invalid labels: {invalid_labels}")
        df = df[df["label"].isin(valid_labels)].copy()
        logger.info(f"After filtering to valid labels: {len(df)} pairs")

        # Extract features (without imputation - we'll do that after split)
        X, y = self._extract_features_and_labels(df, binary=binary)

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")

        # Segment-aware train/test split to prevent data leakage
        train_idx, test_idx = segment_aware_split(df, test_size=test_size, random_state=42)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Verify labels have all expected features before computing medians
        # This catches bugs where new features are added to FEATURE_COLUMNS but
        # the labels (created with older code) don't have them
        expected_features = (
            [f for f in FEATURE_COLUMNS if f not in SEMANTIC_FEATURES]
            if exclude_semantic
            else FEATURE_COLUMNS
        )
        missing_in_labels = set(expected_features) - set(self.feature_names)
        if missing_in_labels:
            raise ValueError(
                f"Labels are missing {len(missing_in_labels)} expected features: {sorted(missing_in_labels)}. "
                f"This usually means labels were created with an older version. "
                f"Run backfill to add missing features, or retrain with updated labels."
            )

        # Compute imputation values from TRAINING data only
        self.feature_medians = {}
        for i, feat_name in enumerate(self.feature_names):
            col_vals = X_train[:, i]
            median_val = np.nanmedian(col_vals)
            self.feature_medians[feat_name] = median_val if not np.isnan(median_val) else 0.0

        # Apply imputation to both train and test using training medians
        X_train = self._impute_missing(X_train)
        X_test = self._impute_missing(X_test)

        # Handle class imbalance
        if binary:
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
            logger.info(
                f"Class balance: {n_pos} positive, {n_neg} negative, scale={scale_pos_weight:.2f}"
            )
        else:
            scale_pos_weight = None

        # Default XGBoost parameters
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "binary:logistic" if binary else "multi:softprob",
            "eval_metric": "logloss" if binary else "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        if scale_pos_weight and binary:
            default_params["scale_pos_weight"] = scale_pos_weight
        if not binary:
            default_params["num_class"] = len(self.label_encoder)

        # Override with user params
        params = {**default_params, **kwargs}

        # Train
        logger.info(f"Training XGBoost with params: {params}")
        self.model = xgb.XGBClassifier(**params)

        # Only use eval_set if we have test data
        if len(X_test) > 0:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)

        # Results dict
        target_names = ["no_match", "match"] if binary else ["no_match", "match", "associated"]
        results = {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "feature_importance": dict(zip(self.feature_names, self.model.feature_importances_)),
        }

        # Evaluate on test set if we have one
        if len(X_test) > 0:
            y_pred = self.model.predict(X_test)

            # Cross-validation score (need to impute full X for this)
            X_imputed = self._impute_missing(X.copy())
            cv_scores = cross_val_score(self.model, X_imputed, y, cv=5, scoring="f1_weighted")

            results.update(
                {
                    "test_accuracy": (y_pred == y_test).mean(),
                    "cv_f1_mean": cv_scores.mean(),
                    "cv_f1_std": cv_scores.std(),
                    "classification_report": classification_report(
                        y_test,
                        y_pred,
                        target_names=target_names,
                        output_dict=True,
                    ),
                    "confusion_matrix": sklearn_confusion_matrix(y_test, y_pred).tolist(),
                }
            )

            # Print summary with test metrics
            print("\n" + "=" * 50)
            print("TRAINING RESULTS")
            print("=" * 50)
            print(f"Training samples: {results['n_train']}")
            print(f"Test samples: {results['n_test']}")
            print(f"Test accuracy: {results['test_accuracy']:.3f}")
            print(f"CV F1 (5-fold): {results['cv_f1_mean']:.3f} ± {results['cv_f1_std']:.3f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=target_names))
        else:
            # No test set - just print training info
            print("\n" + "=" * 50)
            print("TRAINING RESULTS (no test set)")
            print("=" * 50)
            print(f"Training samples: {results['n_train']}")

        print("\nFeature Importance (top 5):")
        importance = sorted(results["feature_importance"].items(), key=lambda x: -x[1])
        for feat, imp in importance[:5]:
            print(f"  {feat}: {imp:.3f}")

        return results

    def _extract_features_and_labels(
        self, df: pd.DataFrame, binary: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and labels from dataframe.

        Does NOT perform imputation - that should be done after train/test split.

        Args:
            df: Labels dataframe
            binary: If True, convert to binary labels (match=1, else=0)

        Returns:
            Tuple of (X features, y labels)
        """
        if len(df) == 0:
            raise ValueError("Cannot extract features from empty dataframe")

        # Check if features are in a nested dict or individual columns
        # Handle null/empty first row gracefully
        has_features_col = "features" in df.columns
        first_features = df["features"].iloc[0] if has_features_col else None
        if has_features_col and first_features and isinstance(first_features, dict):
            return self._extract_from_dict(df, binary)
        else:
            return self._extract_from_columns(df, binary)

    def _extract_from_columns(
        self, df: pd.DataFrame, binary: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract features from individual columns."""
        # Use pre-set feature_names if already configured (e.g., by exclude_semantic)
        # Otherwise, build list from all available FEATURE_COLUMNS
        base_features = (
            self.feature_names
            if hasattr(self, "feature_names") and self.feature_names
            else FEATURE_COLUMNS
        )

        # Build list of actual feature columns present in the data
        actual_features = [feat for feat in base_features if feat in df.columns]
        self.feature_names = actual_features
        logger.info(f"Using features: {self.feature_names}")

        # Extract feature matrix (without imputation)
        X = df[self.feature_names].values.astype(np.float32)

        # Extract labels
        if binary:
            y = (df["label"] == "match").astype(int).values
        else:
            y = df["label"].map(self.label_encoder).values

        return X, y

    def _extract_from_dict(
        self, df: pd.DataFrame, binary: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract features from nested dict column."""
        feature_rows = []

        for _, row in df.iterrows():
            features = row.get("features", {})
            if features and isinstance(features, dict):
                feat_row = [features.get(col, np.nan) for col in self.feature_names]
                feature_rows.append(feat_row)
            else:
                # Fill with NaNs for rows without features
                feature_rows.append([np.nan] * len(self.feature_names))

        X = np.array(feature_rows, dtype=np.float32)

        if binary:
            y = (df["label"] == "match").astype(int).values
        else:
            y = df["label"].map(self.label_encoder).values

        return X, y

    def _impute_missing(self, X: np.ndarray) -> np.ndarray:
        """Impute missing and infinite values using stored medians.

        Args:
            X: Feature matrix with potential NaNs or infinite values

        Returns:
            Feature matrix with NaNs replaced by medians and infinities capped
        """
        X = X.copy()
        for i, feat_name in enumerate(self.feature_names):
            # Handle NaN values
            nan_mask = np.isnan(X[:, i])
            if nan_mask.any():
                fill_value = self.feature_medians.get(feat_name, 0.0)
                X[nan_mask, i] = fill_value
            # Handle infinite values (cap at MAX_DISTANCE_METERS)
            inf_mask = np.isinf(X[:, i])
            if inf_mask.any():
                X[inf_mask, i] = MAX_DISTANCE_METERS
        return X

    def predict(self, features: list[dict[str, float]]) -> list[float]:
        """Predict match probabilities.

        Args:
            features: List of feature dictionaries

        Returns:
            List of match probabilities (0-1)
        """
        if self.model is None:
            raise ValueError("No model loaded - call train() or load_model() first")

        X = self._features_to_array(features)
        probs = self.model.predict_proba(X)

        # Find the index for 'match' class dynamically
        match_class = self.label_encoder.get("match", 1)
        class_indices = list(self.model.classes_)
        if match_class in class_indices:
            match_idx = class_indices.index(match_class)
        else:
            # Fallback for binary where classes are [0, 1]
            match_idx = 1

        return probs[:, match_idx].tolist()

    def predict_class(self, features: list[dict[str, float]]) -> list[str]:
        """Predict class labels.

        Args:
            features: List of feature dictionaries

        Returns:
            List of predicted labels
        """
        if self.model is None:
            raise ValueError("No model loaded")

        X = self._features_to_array(features)
        y_pred = self.model.predict(X)
        return [self.label_decoder.get(int(y), "unknown") for y in y_pred]

    def _features_to_array(self, features: list[dict[str, float]]) -> np.ndarray:
        """Convert feature dicts to numpy array, using stored medians for missing values.

        Also handles infinite values by replacing them with MAX_DISTANCE_METERS,
        which prevents XGBoost from producing NaN predictions.
        """
        rows = []
        for feat_dict in features:
            row = []
            for col in self.feature_names:
                val = feat_dict.get(col, np.nan)
                # Use stored median if value is missing or NaN
                if pd.isna(val):
                    val = self.feature_medians.get(col, 0.0)
                # Replace infinite values with MAX_DISTANCE_METERS to avoid XGBoost issues
                elif np.isinf(val):
                    val = MAX_DISTANCE_METERS
                row.append(val)
            rows.append(row)
        return np.array(rows, dtype=np.float32)

    def score_candidates(
        self,
        candidates: list,
        reference,
        target,
        ref_name_column: str = "names",
        target_name_column: str = "names",
        ref_class_column: str = "class",
        target_class_column: str = "class",
        ref_subclass_column: str = "subclass",
        target_subclass_column: str = "subclass",
        n_jobs: int = -1,
    ) -> list[MatchResult]:
        """Score candidates using the ML model.

        Args:
            candidates: List of CandidatePair objects
            reference: Reference GeoDataFrame
            target: Target GeoDataFrame
            ref_name_column: Column name for reference names
            target_name_column: Column name for target names
            ref_class_column: Column name for reference class
            target_class_column: Column name for target class
            ref_subclass_column: Column name for reference subclass
            target_subclass_column: Column name for target subclass
            n_jobs: Number of parallel jobs (-1 for all cores)

        Returns:
            List of MatchResult objects
        """
        # Handle auto model selection
        if self.model is None and self._auto_select:
            from ..config import settings

            if settings.auto_select_model:
                # Auto-select model based on target dataset
                # select_model_for_dataset uses settings defaults when paths are None
                selected_model = select_model_for_dataset(
                    target,
                    full_model_path=self.model_path,
                    name_column=target_name_column,
                )
                self.load_model(selected_model)
            elif self.model_path:
                self.load_model(self.model_path)

        if self.model is None:
            logger.warning("No ML model loaded, falling back to rules")
            from .rules import score_candidates

            return score_candidates(candidates, reference, target)

        # Handle empty candidates list
        if not candidates:
            return []

        # Project to meter-based CRS for accurate distance computations
        # All distance features will be in meters after this projection
        working_ref = reference
        working_target = target
        if reference.crs is not None and reference.crs.is_geographic:
            utm_crs = reference.estimate_utm_crs()
            logger.debug(f"Projecting to {utm_crs} for meter-based feature computation")
            working_ref = reference.to_crs(utm_crs)
            working_target = target.to_crs(utm_crs)

        # Pre-extract data into NumPy arrays for memory efficiency
        ref_geoms = working_ref.geometry.to_numpy()
        target_geoms = working_target.geometry.to_numpy()
        ref_names = (
            reference[ref_name_column].to_numpy()
            if ref_name_column in reference.columns
            else np.full(len(reference), None, dtype=object)
        )
        target_names = (
            target[target_name_column].to_numpy()
            if target_name_column in target.columns
            else np.full(len(target), None, dtype=object)
        )
        ref_classes = (
            reference[ref_class_column].to_numpy()
            if ref_class_column in reference.columns
            else np.full(len(reference), None, dtype=object)
        )
        target_classes = (
            target[target_class_column].to_numpy()
            if target_class_column in target.columns
            else np.full(len(target), None, dtype=object)
        )
        ref_subclasses = (
            reference[ref_subclass_column].to_numpy()
            if ref_subclass_column in reference.columns
            else np.full(len(reference), None, dtype=object)
        )
        target_subclasses = (
            target[target_subclass_column].to_numpy()
            if target_subclass_column in target.columns
            else np.full(len(target), None, dtype=object)
        )

        # Pre-compute endpoint, topology, and graphlet features for both reference and target
        # These capture network connectivity without requiring explicit topology
        from ..config import settings
        from ..features.alignment import compute_alignment_batch
        from ..features.compute import precompute_graphlet_features
        from ..features.spatial_context import (
            SpatialContextIndex,
            compute_all_topology,
            compute_endpoint_features,
        )

        # Get unique indices from candidates to avoid recomputation
        unique_target_indices = set(cand.target_idx for cand in candidates)
        unique_ref_indices = set(cand.ref_idx for cand in candidates)

        # Filter target to only segments that appear in candidates
        # This dramatically speeds up spatial index building (e.g., 11k -> ~1k segments)
        # IMPORTANT: Use working_target (projected CRS) not target (WGS84) to match the
        # CRS of target_geoms which are used as query points in compute_endpoint_features
        sorted_target_indices = sorted(unique_target_indices)
        target_candidates_only = working_target.iloc[sorted_target_indices].reset_index(drop=True)
        logger.info(
            f"Filtered target to {len(target_candidates_only)} candidate segments "
            f"(from {len(target)} total)"
        )

        # Create mapping from original index to filtered index
        # SpatialContextIndex uses 0-indexed positions within the GDF passed to build_from_gdf()
        original_to_filtered = {orig: filt for filt, orig in enumerate(sorted_target_indices)}

        # Build spatial index for endpoint proximity features (filtered target only)
        logger.info("Building spatial index for endpoint features...")
        target_index = SpatialContextIndex()
        target_index.build_from_gdf(target_candidates_only, id_column="id")

        # Pre-compute endpoint features for target segments
        target_endpoint_features = {}
        logger.info(
            f"Pre-computing endpoint features for {len(unique_target_indices)} target segments..."
        )
        for target_idx in unique_target_indices:
            target_geom = target_geoms[target_idx]
            if target_geom is not None and not target_geom.is_empty:
                # Map original index to filtered index for spatial context lookup
                filtered_idx = original_to_filtered[target_idx]
                ep_feats = compute_endpoint_features(
                    target_geom, target_index, exclude_segment_idx=filtered_idx
                )
                target_endpoint_features[target_idx] = ep_feats
            else:
                target_endpoint_features[target_idx] = {
                    "min_endpoint_proximity_m": MAX_DISTANCE_METERS,
                    "max_endpoint_proximity_m": MAX_DISTANCE_METERS,
                    "shared_endpoint_count": 0,
                }

        # Pre-compute topology features using efficient Union-Find batch computation
        # Get unique segment IDs for only the candidates we need
        target_ids = target["id"].to_numpy()
        ref_ids = reference["id"].to_numpy()
        unique_target_ids = {str(target_ids[idx]) for idx in unique_target_indices}
        unique_ref_ids = {str(ref_ids[idx]) for idx in unique_ref_indices}

        logger.info(
            f"Computing topology features for {len(unique_target_ids)} target "
            f"and {len(unique_ref_ids)} reference segments (batch)..."
        )

        # Check for explicit connector data (Overture/OSM style)
        # When available, use explicit topology instead of geometry inference
        ref_has_connectors = "connectors" in reference.columns
        target_has_connectors = "connectors" in target.columns

        # Compute topology for target and reference
        # Uses explicit connectors when available, falls back to geometry inference
        target_topology_by_id = compute_all_topology(
            target,
            id_column="id",
            tolerance_m=5.0,
            ids_to_compute=unique_target_ids,
            connectors_column="connectors" if target_has_connectors else None,
        )
        ref_topology_by_id = compute_all_topology(
            reference,
            id_column="id",
            tolerance_m=5.0,
            ids_to_compute=unique_ref_ids,
            connectors_column="connectors" if ref_has_connectors else None,
        )

        # Map topology from segment IDs to DataFrame indices
        target_topology_features = {}
        for target_idx in unique_target_indices:
            seg_id = str(target_ids[target_idx])
            target_topology_features[target_idx] = target_topology_by_id.get(
                seg_id, DEFAULT_TOPOLOGY_FEATURES.copy()
            )

        ref_topology_features = {}
        for ref_idx in unique_ref_indices:
            seg_id = str(ref_ids[ref_idx])
            ref_topology_features[ref_idx] = ref_topology_by_id.get(
                seg_id, DEFAULT_TOPOLOGY_FEATURES.copy()
            )

        # Pre-compute graphlet features for network topology similarity
        # For Overture reference data, use explicit connector positions for alignment-aware comparison
        # Filter to only candidate segments to reduce graph building overhead
        sorted_ref_indices = sorted(unique_ref_indices)
        ref_candidates_only = working_ref.iloc[sorted_ref_indices].reset_index(drop=True)
        target_candidates_only_proj = working_target.iloc[sorted_target_indices].reset_index(
            drop=True
        )

        logger.info(
            f"Computing graphlet features for {len(ref_candidates_only)} reference "
            f"and {len(target_candidates_only_proj)} target segments..."
        )
        # ref_has_connectors already defined earlier for topology computation
        ref_graphlet_data = precompute_graphlet_features(
            ref_candidates_only,
            id_column="id",
            tolerance_m=5.0,
            connectors_column="connectors" if ref_has_connectors else None,
        )
        # Target data typically doesn't have explicit connectors, use endpoint-based inference
        target_graphlet_data = precompute_graphlet_features(
            target_candidates_only_proj, id_column="id", tolerance_m=5.0
        )

        # Pre-compute linestring alignments if enabled
        # Alignments are used to compute similarity features on aligned sublines
        alignments = {}
        use_aligned_features = settings.alignment_enabled
        if use_aligned_features:
            logger.info("Computing linestring alignments...")
            alignments = compute_alignment_batch(candidates, ref_geoms, target_geoms, n_jobs=n_jobs)
        else:
            logger.info("Alignment disabled, computing features on full geometries")

        # Determine number of workers (leave 2 cores for system)
        if n_jobs == -1:
            n_workers = max(1, mp.cpu_count() - 2)
        else:
            n_workers = n_jobs

        n_candidates = len(candidates)
        logger.info(
            f"Computing features for {n_candidates} candidates using {n_workers} processes..."
        )

        # Prepare worker data (passed via initializer to avoid repeated pickling)
        worker_data = {
            "ref_geoms": ref_geoms,
            "target_geoms": target_geoms,
            "ref_names": ref_names,
            "target_names": target_names,
            "ref_classes": ref_classes,
            "target_classes": target_classes,
            "ref_subclasses": ref_subclasses,
            "target_subclasses": target_subclasses,
            "ref_ids": ref_ids,
            "target_ids": target_ids,
            "endpoint_features": target_endpoint_features,
            "ref_topology": ref_topology_features,
            "target_topology": target_topology_features,
            "ref_graphlet_data": ref_graphlet_data,
            "target_graphlet_data": target_graphlet_data,
            "alignments": alignments,
            "use_aligned_features": use_aligned_features,
        }

        # Prepare work items as simple tuples
        work_items = [(cand.ref_idx, cand.target_idx) for cand in candidates]

        # Process with map for ordered results
        # Use smaller chunks for more frequent progress updates
        chunk_size = max(100, min(1000, n_candidates // (n_workers * 10)))
        features_list = []

        logger.info(f"Starting parallel feature computation (chunk_size={chunk_size})...")

        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker, initargs=(worker_data,)
        ) as executor:
            # Process in chunks to show progress
            for i in range(0, len(work_items), chunk_size * n_workers):
                batch = work_items[i : i + chunk_size * n_workers]
                batch_results = list(
                    executor.map(_compute_single_feature, batch, chunksize=chunk_size)
                )
                features_list.extend(batch_results)
                processed = min(i + len(batch), len(work_items))
                pct = processed / len(work_items) * 100
                logger.info(f"Feature computation: {processed:,}/{len(work_items):,} ({pct:.0f}%)")

        # Log any errors encountered during feature computation
        errors = [f for f in features_list if f.get("_error")]
        if errors:
            logger.warning(f"{len(errors)} candidates had feature computation errors")

        # Batch prediction - use probability (confidence), not predicted class
        # This allows the downstream optimizer to use confidence threshold
        logger.info(f"Running XGBoost prediction on {len(features_list):,} candidates...")
        probs = self.predict(features_list)
        logger.info("XGBoost prediction complete")

        # Build results - use confidence-based decision, not class-based
        # This ensures high-confidence matches aren't filtered just because
        # the model's decision boundary puts them in "no_match" class
        logger.info(f"Building {len(candidates):,} MatchResult objects...")
        results = []
        for i, cand in enumerate(candidates):
            prob = probs[i]

            # Use confidence thresholds instead of class prediction
            # This makes the ML model behave more like a confidence scorer
            if prob >= 0.5:
                decision = MatchDecision.MATCH
            elif prob >= 0.1:
                decision = MatchDecision.REVIEW  # Low confidence but possible
            else:
                decision = MatchDecision.NO_MATCH

            # Get alignment for linear reference fields
            alignment = alignments.get((cand.ref_idx, cand.target_idx))

            results.append(
                MatchResult(
                    ref_id=cand.ref_id,
                    target_id=cand.target_id,
                    decision=decision,
                    confidence=prob,
                    score_breakdown={},  # ML doesn't have component scores
                    features=features_list[i],
                    gers_start_frac=alignment.overture_start_frac if alignment else None,
                    gers_end_frac=alignment.overture_end_frac if alignment else None,
                    local_start_frac=alignment.dataset_start_frac if alignment else None,
                    local_end_frac=alignment.dataset_end_frac if alignment else None,
                )
            )

            # Progress logging every 100k
            if (i + 1) % 100000 == 0:
                logger.info(f"Built {i + 1:,}/{len(candidates):,} MatchResult objects...")

        logger.info(f"Built {len(results):,} MatchResult objects")
        return results


def train_model(
    labels_dir: str = "labels",
    output_path: str = "data/models/matcher_model.joblib",
    binary: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Convenience function to train and save a model.

    Args:
        labels_dir: Path to Hive-partitioned labels directory
        output_path: Path to save trained model
        binary: Train binary (match/no_match) or multiclass
        **kwargs: XGBoost parameters

    Returns:
        Training results dict
    """
    matcher = MLMatcher()
    results = matcher.train(labels_dir=labels_dir, binary=binary, **kwargs)
    matcher.save_model(output_path)
    return results


def evaluate_by_dataset(
    model_path: str,
    labels_dir: str = "labels",
    binary: bool = True,
    show_by_dataset: bool = True,
    holdout: bool = True,
    holdout_pct: float = 0.2,
    seed: int = 42,
    filter_datasets: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate model performance broken down by dataset.

    Loads all label partitions and evaluates the model on each dataset separately.

    Args:
        model_path: Path to trained model
        labels_dir: Directory containing Hive-partitioned label CSVs
        binary: Evaluate as binary (match vs no_match)
        show_by_dataset: If True, show per-dataset metrics; if False, only show overall
        holdout: If True (default), use holdout set for unbiased evaluation
        holdout_pct: Fraction of data to hold out for testing (default 0.2 = 20%)
        seed: Random seed for holdout split (for reproducibility)
        filter_datasets: If provided, only evaluate on these datasets

    Returns:
        Dictionary mapping dataset name to metrics dict
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    from ..labeling.label_store import LabelStore

    # Load model
    matcher = MLMatcher(model_path)

    # Load all labels using LabelStore
    all_labels = LabelStore.load_all(Path(labels_dir))

    if len(all_labels) == 0:
        logger.warning(f"No labels found in {labels_dir}")
        return {}

    # Get unique datasets
    if "dataset" not in all_labels.columns:
        logger.warning("No 'dataset' column found - cannot evaluate by dataset")
        return {}

    # Filter to specific datasets if requested
    if filter_datasets:
        all_labels = all_labels[all_labels["dataset"].isin(filter_datasets)].copy()
        if len(all_labels) == 0:
            logger.warning(f"No labels found for datasets: {filter_datasets}")
            return {}
        logger.info(f"Filtered to {len(all_labels)} labels from: {filter_datasets}")

    # If holdout requested, split the data first using segment-aware splitting
    if holdout:
        valid_labels = {"match", "no_match", "associated"}
        eval_df = all_labels[all_labels["label"].isin(valid_labels)].copy()
        _, test_idx = segment_aware_split(eval_df, test_size=holdout_pct, random_state=seed)
        all_labels = eval_df.iloc[test_idx]
        print(
            f"\n[Holdout mode: evaluating on {len(all_labels)} samples ({holdout_pct * 100:.0f}% of data, seed={seed})]"
        )

    datasets = all_labels["dataset"].unique()

    results = {}
    all_y_true = []
    all_y_pred = []

    if show_by_dataset:
        print("\n" + "=" * 60)
        print("EVALUATION BY DATASET")
        print("=" * 60)

    for dataset_name in sorted(datasets):
        # Filter to this dataset
        df = all_labels[all_labels["dataset"] == dataset_name].copy()

        # Filter to valid labels
        valid_labels = {"match", "no_match", "associated"}
        df = df[df["label"].isin(valid_labels)].copy()

        if len(df) == 0:
            logger.warning(f"No valid labels for dataset {dataset_name}")
            continue

        # Extract features
        X, y = matcher._extract_features_and_labels(df, binary=binary)

        # Impute missing values
        X = matcher._impute_missing(X)

        # Predict
        y_pred = matcher.model.predict(X)

        # Compute metrics
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="weighted")
        precision = precision_score(y, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y, y_pred, average="weighted", zero_division=0)

        # Count labels
        n_match = (y == 1).sum() if binary else (df["label"] == "match").sum()
        n_no_match = (y == 0).sum() if binary else (df["label"] == "no_match").sum()

        results[dataset_name] = {
            "n_samples": len(df),
            "n_match": int(n_match),
            "n_no_match": int(n_no_match),
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        }

        # Accumulate for overall
        all_y_true.extend(y)
        all_y_pred.extend(y_pred)

        # Print summary (only if showing by dataset)
        if show_by_dataset:
            print(f"\n{dataset_name}:")
            print(f"  Samples: {len(df)} ({n_match} match, {n_no_match} no_match)")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  F1: {f1:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")

    # Overall metrics
    if all_y_true:
        overall_accuracy = accuracy_score(all_y_true, all_y_pred)
        overall_f1 = f1_score(all_y_true, all_y_pred, average="weighted")

        if show_by_dataset:
            print("\n" + "-" * 60)
        else:
            print("\n" + "=" * 60)
        print("OVERALL:")
        print(f"  Total samples: {len(all_y_true)}")
        print(f"  Accuracy: {overall_accuracy:.3f}")
        print(f"  F1: {overall_f1:.3f}")

        results["_overall"] = {
            "n_samples": len(all_y_true),
            "accuracy": overall_accuracy,
            "f1": overall_f1,
        }

    return results
