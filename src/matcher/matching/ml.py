"""Machine learning-based matcher using gradient boosted trees.

This module provides XGBoost-based matching trained on labeled data.
The model learns to classify road segment pairs as match/no_match
based on geometric and semantic features.

Training Data Format:
--------------------
Uses labels from data/labels/labels.parquet which contains:
- ref_id, target_id: Segment identifiers
- label: Human label (match, no_match, unsure; legacy: associated)
- features: Dict of precomputed features (hausdorff_distance, buffer_iou, etc.)

Model Architecture:
------------------
- XGBoost classifier with binary (match vs no_match) or multiclass output
- Features: Normalized geometric + semantic scores (same as rule-based)
- Handles class imbalance via scale_pos_weight or class_weight
"""

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

from .rules import MatchDecision, MatchResult


# Features used for ML model (must match what's stored in labels)
# Note: projection_distance is excluded because it's now identical to mean_hausdorff_distance
# (both compute bidirectional mean of min distances). Including both would double-weight.
FEATURE_COLUMNS = [
    "hausdorff_distance",
    "mean_hausdorff_distance",
    "buffer_iou",
    "overlap_ratio",
    "heading_delta",
    "length_ratio",
    "centroid_distance",
    "name_levenshtein",
    "name_jaro_winkler",
    "name_token_sort",
    "class_similarity",
]


class MLMatcher:
    """Machine learning-based matcher using gradient boosted trees."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the ML matcher.

        Args:
            model_path: Path to trained model (optional)
        """
        self.model = None
        self.model_path = model_path
        self.feature_names = FEATURE_COLUMNS.copy()
        self.feature_medians = {}  # For imputing missing values during inference
        self.label_encoder = {"match": 1, "no_match": 0, "associated": 2}
        self.label_decoder = {1: "match", 0: "no_match", 2: "associated"}
        self.is_binary = True  # Track if model is binary or multiclass

        if model_path:
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
        labels_path: str = "data/labels/labels.parquet",
        binary: bool = True,
        test_size: float = 0.2,
        **kwargs,
    ) -> dict[str, Any]:
        """Train the model on labeled data.

        Args:
            labels_path: Path to labels parquet file
            binary: If True (default), train binary classifier (match vs non-match)
                   If False, train multiclass (legacy, includes associated)
            test_size: Fraction of data to hold out for testing
            **kwargs: Additional XGBoost parameters

        Returns:
            Dictionary of training metrics
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "XGBoost is required for ML training. "
                "Install it with: pip install 'matcher[ml]' or pip install xgboost"
            )

        self.is_binary = binary

        # Load and prepare data
        df = pd.read_parquet(labels_path)
        logger.info(f"Loaded {len(df)} labeled pairs")

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

        # Train/test split BEFORE imputation to avoid data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
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
            logger.info(f"Class balance: {n_pos} positive, {n_neg} negative, scale={scale_pos_weight:.2f}")
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
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred = self.model.predict(X_test)

        # Cross-validation score (need to impute full X for this)
        X_imputed = self._impute_missing(X.copy())
        cv_scores = cross_val_score(self.model, X_imputed, y, cv=5, scoring="f1_weighted")

        # Results
        target_names = ["no_match", "match"] if binary else ["no_match", "match", "associated"]
        results = {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "test_accuracy": (y_pred == y_test).mean(),
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=target_names,
                output_dict=True,
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "feature_importance": dict(zip(self.feature_names, self.model.feature_importances_)),
        }

        # Print summary
        print("\n" + "=" * 50)
        print("TRAINING RESULTS")
        print("=" * 50)
        print(f"Training samples: {results['n_train']}")
        print(f"Test samples: {results['n_test']}")
        print(f"Test accuracy: {results['test_accuracy']:.3f}")
        print(f"CV F1 (5-fold): {results['cv_f1_mean']:.3f} ± {results['cv_f1_std']:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
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
        # Build list of actual feature columns to use (no duplicates)
        actual_features = []
        for feat in FEATURE_COLUMNS:
            if feat in df.columns:
                actual_features.append(feat)
            elif feat == "mean_hausdorff_distance" and "hausdorff_distance" in df.columns:
                # Skip - we'll use hausdorff_distance instead, but don't duplicate
                if "hausdorff_distance" not in actual_features:
                    actual_features.append("hausdorff_distance")
            elif feat == "overlap_ratio" and "buffer_iou" in df.columns:
                # Skip - we'll use buffer_iou instead, but don't duplicate
                if "buffer_iou" not in actual_features:
                    actual_features.append("buffer_iou")

        # Remove duplicates while preserving order
        seen = set()
        unique_features = []
        for f in actual_features:
            if f not in seen:
                seen.add(f)
                unique_features.append(f)

        self.feature_names = unique_features
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
                feat_row = []
                for col in self.feature_names:
                    val = features.get(col, np.nan)
                    if col == "mean_hausdorff_distance" and pd.isna(val):
                        val = features.get("hausdorff_distance", np.nan)
                    if col == "overlap_ratio" and pd.isna(val):
                        val = features.get("buffer_iou", np.nan)
                    feat_row.append(val)
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
        """Impute missing values using stored medians.

        Args:
            X: Feature matrix with potential NaNs

        Returns:
            Feature matrix with NaNs replaced by medians
        """
        X = X.copy()
        for i, feat_name in enumerate(self.feature_names):
            nan_mask = np.isnan(X[:, i])
            if nan_mask.any():
                fill_value = self.feature_medians.get(feat_name, 0.0)
                X[nan_mask, i] = fill_value
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
        """Convert feature dicts to numpy array, using stored medians for missing values."""
        rows = []
        for feat_dict in features:
            row = []
            for col in self.feature_names:
                val = feat_dict.get(col, np.nan)
                # Use stored median if value is missing
                if pd.isna(val):
                    val = self.feature_medians.get(col, 0.0)
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
    ) -> list[MatchResult]:
        """Score candidates using the ML model.

        Args:
            candidates: List of CandidatePair objects
            reference: Reference GeoDataFrame
            target: Target GeoDataFrame

        Returns:
            List of MatchResult objects
        """
        from ..features.geometric import compute_geometric_features
        from ..features.semantic import compute_class_similarity, compute_name_similarity

        if self.model is None:
            logger.warning("No ML model loaded, falling back to rules")
            from .rules import score_candidates
            return score_candidates(candidates, reference, target)

        # Handle empty candidates list
        if not candidates:
            return []

        features_list = []

        for cand in candidates:
            ref_row = reference.iloc[cand.ref_idx]
            target_row = target.iloc[cand.target_idx]

            # Compute features
            geom_features = compute_geometric_features(
                ref_row.geometry, target_row.geometry
            )
            name_sim = compute_name_similarity(
                ref_row.get(ref_name_column),
                target_row.get(target_name_column),
            )
            class_sim = compute_class_similarity(
                ref_row.get(ref_class_column),
                target_row.get(target_class_column),
            )

            features = {
                "hausdorff_distance": geom_features.hausdorff_distance,
                "mean_hausdorff_distance": geom_features.mean_hausdorff_distance,
                "buffer_iou": geom_features.buffer_iou,
                "overlap_ratio": geom_features.overlap_ratio,
                "heading_delta": geom_features.heading_delta,
                "length_ratio": geom_features.length_ratio,
                "projection_distance": geom_features.projection_distance,
                "centroid_distance": geom_features.centroid_distance,
                "name_levenshtein": name_sim["levenshtein_ratio"],
                "name_jaro_winkler": name_sim["jaro_winkler"],
                "name_token_sort": name_sim["token_sort_ratio"],
                "class_similarity": class_sim,
            }
            features_list.append(features)

        # Batch prediction
        probs = self.predict(features_list)
        labels = self.predict_class(features_list)

        # Build results
        results = []
        for i, cand in enumerate(candidates):
            prob = probs[i]
            label = labels[i]

            if label == "match":
                decision = MatchDecision.MATCH
            elif label == "associated":
                decision = MatchDecision.REVIEW  # Treat associated as review
            else:
                decision = MatchDecision.NO_MATCH

            results.append(MatchResult(
                ref_id=cand.ref_id,
                target_id=cand.target_id,
                decision=decision,
                confidence=prob,
                score_breakdown={},  # ML doesn't have component scores
                features=features_list[i],
            ))

        return results


def train_model(
    labels_path: str = "data/labels/labels.parquet",
    output_path: str = "data/models/matcher_model.joblib",
    binary: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Convenience function to train and save a model.

    Args:
        labels_path: Path to labels parquet
        output_path: Path to save trained model
        binary: Train binary (match/no_match) or multiclass
        **kwargs: XGBoost parameters

    Returns:
        Training results dict
    """
    matcher = MLMatcher()
    results = matcher.train(labels_path=labels_path, binary=binary, **kwargs)
    matcher.save_model(output_path)
    return results
