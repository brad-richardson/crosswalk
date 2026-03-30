"""Lightweight class predictor using name patterns, physical attributes, and source classification.

This module provides a simple ML model that predicts Overture road classes
from source features, useful for improving class mappings.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from loguru import logger

# Name patterns for class prediction
# Each pattern maps to an Overture class hint
NAME_PATTERNS = {
    # Highway/Major road patterns
    "motorway": r"\b(interstate|i-\d+|freeway|expressway|autobahn)\b",
    "trunk": r"\b(highway|hwy|us-?\d+|state\s*route|sr-?\d+)\b",
    "primary": r"\b(boulevard|blvd|avenue|ave)\b",
    # Local road patterns
    "residential": r"\b(street|st|road|rd|drive|dr|lane|ln|way|court|ct|circle|cir|place|pl)\b",
    "service": r"\b(service|alley|driveway|parking)\b",
    # Pedestrian/bike patterns
    "path_trail": r"\b(trail|path|walkway|greenway)\b",
    "footway": r"\b(sidewalk|footpath|footway|pavement)\b",
    "cycleway": r"\b(bike\s*lane|bicycle|cycleway|bikeway|bike\s*path)\b",
}

# Compiled patterns for rule-based prediction
_COMPILED_PATTERNS = {
    name: re.compile(pattern, re.IGNORECASE) for name, pattern in NAME_PATTERNS.items()
}

# Source classification keywords that hint at Overture classes
# Used for soft matching when source classes don't map 1:1
SOURCE_CLASS_KEYWORDS = {
    # Vehicular - major
    "motorway": ["motorway", "freeway", "interstate", "autobahn", "expressway"],
    "trunk": ["trunk", "highway", "national", "federal", "arterial_major"],
    "primary": ["primary", "arterial", "major", "principal"],
    "secondary": ["secondary", "collector", "distributor"],
    "tertiary": ["tertiary", "minor_arterial", "local_collector"],
    # Vehicular - local
    "residential": ["residential", "local", "neighborhood", "street", "urban"],
    "service": ["service", "access", "driveway", "parking", "alley"],
    "unclassified": ["unclassified", "unknown", "other", "rural"],
    "track": ["track", "unpaved", "dirt", "gravel", "farm", "forest"],
    # Non-vehicular
    "footway": ["footway", "sidewalk", "pedestrian", "foot", "walking"],
    "path": ["path", "trail", "hiking", "footpath", "walkway"],
    "cycleway": ["cycleway", "bicycle", "bike", "cycling"],
    "steps": ["steps", "stairs", "stairway"],
}

# Traffic tier mapping for evaluation
OVERTURE_TIERS = {
    "motorway": "vehicle",
    "motorway_link": "vehicle",
    "trunk": "vehicle",
    "trunk_link": "vehicle",
    "primary": "vehicle",
    "primary_link": "vehicle",
    "secondary": "vehicle",
    "secondary_link": "vehicle",
    "tertiary": "vehicle",
    "tertiary_link": "vehicle",
    "residential": "vehicle",
    "living_street": "vehicle",
    "service": "vehicle",
    "unclassified": "vehicle",
    "track": "vehicle",
    "cycleway": "bicycle",
    "footway": "pedestrian",
    "sidewalk": "pedestrian",
    "path": "pedestrian",
    "pedestrian": "pedestrian",
    "steps": "pedestrian",
    "bridleway": "other",
}

# Hierarchy levels for evaluating "close enough" predictions
OVERTURE_HIERARCHY = {
    "motorway": 1,
    "motorway_link": 1,
    "trunk": 2,
    "trunk_link": 2,
    "primary": 3,
    "primary_link": 3,
    "secondary": 4,
    "secondary_link": 4,
    "tertiary": 5,
    "tertiary_link": 5,
    "residential": 6,
    "living_street": 6,
    "service": 7,
    "unclassified": 8,
    "track": 9,
    "footway": 10,
    "sidewalk": 10,
    "cycleway": 10,
    "path": 10,
    "pedestrian": 10,
    "steps": 10,
}


@dataclass
class LightweightClassPredictor:
    """Predict Overture class from source features.

    Features used:
    - Name patterns (regex matches for "Trail", "Highway", "Lane", "Path", etc.)
    - Physical attributes if available (lanes, width, speed_limit)
    - Length (very short segments often pedestrian)

    This predictor is intentionally simple and interpretable, suitable for
    suggesting class mappings rather than production classification.
    """

    model: Any = None
    feature_names: list[str] = field(default_factory=list)
    class_labels: list[str] = field(default_factory=list)
    is_trained: bool = False

    def extract_features(
        self,
        gdf: gpd.GeoDataFrame,
        name_column: str = "names",
        class_column: str | None = "class",
        class_mapping: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Extract prediction features from geodataframe.

        Args:
            gdf: GeoDataFrame with road segments
            name_column: Column containing road names
            class_column: Column containing source classification (if any)
            class_mapping: Optional dict mapping source class -> Overture class
                          Used to compute mapping confidence features

        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame(index=gdf.index)

        # Extract name string from dict if needed
        if name_column in gdf.columns:
            names = gdf[name_column].apply(self._extract_name_string)
        else:
            names = pd.Series("", index=gdf.index)

        # Name pattern features (use raw patterns for pandas str.contains)
        for pattern_name, pattern_str in NAME_PATTERNS.items():
            features[f"name_{pattern_name}"] = (
                names.fillna("")
                .str.contains(pattern_str, case=False, na=False, regex=True)
                .astype(float)
            )

        # === SOURCE CLASSIFICATION FEATURES ===
        if class_column and class_column in gdf.columns:
            source_classes = gdf[class_column].fillna("unknown").astype(str).str.lower().str.strip()

            # Approach 1: One-hot encoding of known source classes
            unique_classes = source_classes.unique()
            for src_class in unique_classes:
                if src_class and src_class != "unknown":
                    # Sanitize column name
                    safe_name = re.sub(r"[^a-z0-9_]", "_", src_class)[:30]
                    features[f"src_class_{safe_name}"] = (source_classes == src_class).astype(float)

            # Approach 2: Keyword-based soft matching to Overture classes
            for overture_class, keywords in SOURCE_CLASS_KEYWORDS.items():
                pattern = "|".join(re.escape(kw) for kw in keywords)
                features[f"src_hints_{overture_class}"] = source_classes.str.contains(
                    pattern, case=False, na=False, regex=True
                ).astype(float)

            # Approach 3: Class mapping confidence (if mapping provided)
            if class_mapping:
                # Compute whether source class has a known mapping
                has_mapping = source_classes.apply(
                    lambda x: x in class_mapping or str(x) in class_mapping
                )
                features["has_class_mapping"] = has_mapping.astype(float)

                # Get the mapped class tier
                def get_mapped_tier(src):
                    mapped = class_mapping.get(src) or class_mapping.get(str(src))
                    if mapped:
                        return OVERTURE_TIERS.get(mapped.lower(), "unknown")
                    return "unknown"

                mapped_tiers = source_classes.apply(get_mapped_tier)
                features["mapped_tier_vehicle"] = (mapped_tiers == "vehicle").astype(float)
                features["mapped_tier_pedestrian"] = (mapped_tiers == "pedestrian").astype(float)
                features["mapped_tier_bicycle"] = (mapped_tiers == "bicycle").astype(float)

        # Physical attributes (if available)
        for col in ["lanes", "width", "speed_limit", "surface"]:
            if col in gdf.columns:
                if col == "surface":
                    # Binary: paved vs unpaved
                    paved_values = {"paved", "asphalt", "concrete", "cobblestone"}
                    features[f"{col}_paved"] = (
                        gdf[col].fillna("").astype(str).str.lower().isin(paved_values).astype(float)
                    )
                else:
                    features[col] = pd.to_numeric(gdf[col], errors="coerce").fillna(-1)

        # Geometry features
        features["length_m"] = gdf.geometry.length

        # Short segment flag (often pedestrian infrastructure)
        features["is_short"] = (features["length_m"] < 50).astype(float)

        # Very long segment flag (often major roads)
        features["is_long"] = (features["length_m"] > 500).astype(float)

        # Sinuosity (curvy roads)
        features["sinuosity"] = gdf.geometry.apply(self._compute_sinuosity)

        # Vertex density (points per meter)
        features["vertex_density"] = gdf.geometry.apply(
            lambda g: len(g.coords) / max(g.length, 1) if g else 0
        )

        # Fill NaN
        features = features.fillna(0)

        return features

    def _compute_sinuosity(self, geom) -> float:
        """Compute sinuosity (actual length / straight-line distance)."""
        if geom is None or geom.is_empty or len(geom.coords) < 2:
            return 1.0

        coords = list(geom.coords)
        start = coords[0]
        end = coords[-1]

        straight_dist = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
        if straight_dist < 0.001:  # Nearly circular
            return float("nan")

        return geom.length / straight_dist

    def _extract_name_string(self, name) -> str:
        """Extract string from name, handling dict format."""
        if name is None:
            return ""
        if isinstance(name, str):
            return name
        if isinstance(name, dict):
            for key in ["primary", "common", "name", "value"]:
                if key in name and name[key]:
                    val = name[key]
                    if isinstance(val, str):
                        return val
            for v in name.values():
                if isinstance(v, str) and v:
                    return v
        return ""

    def train(
        self,
        gdf: gpd.GeoDataFrame,
        labels: pd.Series,
        name_column: str = "names",
        class_column: str | None = "class",
        class_mapping: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Train on labeled data (e.g., from confident matches).

        Args:
            gdf: GeoDataFrame with road segments
            labels: Series of Overture class labels
            name_column: Column containing road names
            class_column: Column containing source classification
            class_mapping: Optional existing mapping for confidence features

        Returns:
            Dict with training stats
        """
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.error("XGBoost not available. Install with: pip install xgboost")
            raise

        # Store config for later use
        self.name_column = name_column
        self.class_column = class_column
        self.class_mapping = class_mapping

        X = self.extract_features(gdf, name_column, class_column, class_mapping)
        y = labels

        # Remove samples with missing labels
        valid_mask = y.notna() & (y != "")
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 10:
            raise ValueError(f"Not enough training samples: {len(X)}")

        # Encode labels
        self.class_labels = sorted(y.unique().tolist())
        label_to_idx = {label: idx for idx, label in enumerate(self.class_labels)}
        y_encoded = y.map(label_to_idx)

        # Train model
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        self.model.fit(X, y_encoded)

        self.feature_names = X.columns.tolist()
        self.is_trained = True

        # Compute training accuracy
        y_pred = self.model.predict(X)
        accuracy = (y_pred == y_encoded).mean()

        # Compute tier accuracy (mode of travel)
        y_true_tiers = y.apply(
            lambda c: OVERTURE_TIERS.get(c.lower(), "unknown") if c else "unknown"
        )
        y_pred_labels = pd.Series([self.class_labels[i] for i in y_pred], index=y.index)
        y_pred_tiers = y_pred_labels.apply(
            lambda c: OVERTURE_TIERS.get(c.lower(), "unknown") if c else "unknown"
        )
        tier_accuracy = (y_true_tiers == y_pred_tiers).mean()

        return {
            "n_samples": len(X),
            "n_classes": len(self.class_labels),
            "classes": self.class_labels,
            "accuracy": accuracy,
            "tier_accuracy": tier_accuracy,
            "feature_names": self.feature_names,
        }

    def predict(
        self,
        gdf: gpd.GeoDataFrame,
        name_column: str | None = None,
        class_column: str | None = None,
        class_mapping: dict[str, str] | None = None,
    ) -> pd.Series:
        """Predict Overture class for each segment.

        Args:
            gdf: GeoDataFrame with road segments
            name_column: Column containing road names (defaults to training config)
            class_column: Column containing source class (defaults to training config)
            class_mapping: Optional mapping (defaults to training config)

        Returns:
            Series of predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Use training config as defaults
        name_column = name_column or getattr(self, "name_column", "names")
        class_column = (
            class_column if class_column is not None else getattr(self, "class_column", "class")
        )
        class_mapping = class_mapping or getattr(self, "class_mapping", None)

        X = self.extract_features(gdf, name_column, class_column, class_mapping)

        # Ensure columns match training
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        y_pred = self.model.predict(X)
        return pd.Series([self.class_labels[i] for i in y_pred], index=gdf.index)

    def predict_proba(
        self,
        gdf: gpd.GeoDataFrame,
        name_column: str | None = None,
        class_column: str | None = None,
        class_mapping: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Predict class probabilities for each segment.

        Args:
            gdf: GeoDataFrame with road segments
            name_column: Column containing road names (defaults to training config)
            class_column: Column containing source class (defaults to training config)
            class_mapping: Optional mapping (defaults to training config)

        Returns:
            DataFrame with probability for each class
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Use training config as defaults
        name_column = name_column or getattr(self, "name_column", "names")
        class_column = (
            class_column if class_column is not None else getattr(self, "class_column", "class")
        )
        class_mapping = class_mapping or getattr(self, "class_mapping", None)

        X = self.extract_features(gdf, name_column, class_column, class_mapping)

        # Ensure columns match training
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        proba = self.model.predict_proba(X)
        return pd.DataFrame(proba, columns=self.class_labels, index=gdf.index)

    def feature_importance(self) -> pd.Series:
        """Get feature importance scores.

        Returns:
            Series of feature importance values
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        """Save trained model to file.

        Args:
            path: Output path (joblib format)
        """
        import joblib

        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")

        data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "class_labels": self.class_labels,
            "name_column": getattr(self, "name_column", "names"),
            "class_column": getattr(self, "class_column", "class"),
            "class_mapping": getattr(self, "class_mapping", None),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, path)
        logger.info(f"Saved predictor to {path}")

    @classmethod
    def load(cls, path: Path) -> "LightweightClassPredictor":
        """Load trained model from file.

        Args:
            path: Path to saved model (joblib format)

        Returns:
            Loaded predictor
        """
        import joblib

        data = joblib.load(path)
        predictor = cls()
        predictor.model = data["model"]
        predictor.feature_names = data["feature_names"]
        predictor.class_labels = data["class_labels"]
        predictor.name_column = data.get("name_column", "names")
        predictor.class_column = data.get("class_column", "class")
        predictor.class_mapping = data.get("class_mapping")
        predictor.is_trained = True
        logger.info(f"Loaded predictor from {path}")
        return predictor


def predict_class_from_name(name: str | None) -> str | None:
    """Simple rule-based class prediction from name alone.

    This function uses regex patterns to guess the road class based on
    common naming conventions. Useful for quick analysis without training.

    Args:
        name: Road name string

    Returns:
        Predicted class or None if no pattern matched
    """
    if not name:
        return None

    name_lower = name.lower()

    # Check patterns in order of specificity
    if _COMPILED_PATTERNS["motorway"].search(name_lower):
        return "motorway"

    if _COMPILED_PATTERNS["cycleway"].search(name_lower):
        return "cycleway"

    if _COMPILED_PATTERNS["footway"].search(name_lower):
        return "footway"

    if _COMPILED_PATTERNS["path_trail"].search(name_lower):
        return "path"

    if _COMPILED_PATTERNS["service"].search(name_lower):
        return "service"

    if _COMPILED_PATTERNS["trunk"].search(name_lower):
        return "trunk"

    if _COMPILED_PATTERNS["primary"].search(name_lower):
        return "primary"

    if _COMPILED_PATTERNS["residential"].search(name_lower):
        return "residential"

    return None


@dataclass
class ClassPredictionAnalysis:
    """Results of class prediction analysis on known matches."""

    # Accuracy metrics
    exact_accuracy: float = 0.0  # Exact class match
    tier_accuracy: float = 0.0  # Same traffic tier (vehicle/pedestrian/bicycle)
    hierarchy_accuracy_1: float = 0.0  # Within 1 hierarchy level
    hierarchy_accuracy_2: float = 0.0  # Within 2 hierarchy levels

    # Confusion breakdown
    confusion_matrix: dict = field(default_factory=dict)  # true_class -> pred_class -> count
    tier_confusion: dict = field(default_factory=dict)  # true_tier -> pred_tier -> count

    # Problem cases
    tier_violations: list = field(default_factory=list)  # Wrong tier predictions
    major_hierarchy_errors: list = field(default_factory=list)  # >2 levels off

    # Per-class stats
    per_class_accuracy: dict = field(default_factory=dict)
    per_class_tier_accuracy: dict = field(default_factory=dict)

    # Sample counts
    n_samples: int = 0
    n_with_predictions: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "accuracy": {
                "exact": self.exact_accuracy,
                "tier": self.tier_accuracy,
                "within_1_level": self.hierarchy_accuracy_1,
                "within_2_levels": self.hierarchy_accuracy_2,
            },
            "confusion_matrix": self.confusion_matrix,
            "tier_confusion": self.tier_confusion,
            "tier_violations": self.tier_violations[:50],  # Limit examples
            "major_hierarchy_errors": self.major_hierarchy_errors[:50],
            "per_class_accuracy": self.per_class_accuracy,
            "per_class_tier_accuracy": self.per_class_tier_accuracy,
            "n_samples": self.n_samples,
            "n_with_predictions": self.n_with_predictions,
        }


def analyze_predictions(
    true_classes: pd.Series,
    predicted_classes: pd.Series,
    segment_ids: pd.Series | None = None,
    names: pd.Series | None = None,
    max_examples: int = 100,
) -> ClassPredictionAnalysis:
    """Analyze prediction quality against known true classes.

    Args:
        true_classes: Series of true Overture class labels
        predicted_classes: Series of predicted class labels
        segment_ids: Optional segment IDs for error reporting
        names: Optional segment names for error reporting
        max_examples: Maximum number of error examples to collect

    Returns:
        ClassPredictionAnalysis with detailed metrics
    """
    analysis = ClassPredictionAnalysis()

    # Align series
    common_idx = true_classes.index.intersection(predicted_classes.index)
    true = true_classes.loc[common_idx].fillna("unknown").astype(str).str.lower()
    pred = predicted_classes.loc[common_idx].fillna("unknown").astype(str).str.lower()

    analysis.n_samples = len(true)
    analysis.n_with_predictions = (pred != "unknown").sum()

    if analysis.n_samples == 0:
        return analysis

    # Compute tiers
    true_tiers = true.apply(lambda c: OVERTURE_TIERS.get(c, "unknown"))
    pred_tiers = pred.apply(lambda c: OVERTURE_TIERS.get(c, "unknown"))

    # Compute hierarchy levels
    true_levels = true.apply(lambda c: OVERTURE_HIERARCHY.get(c, 6))  # Default to residential
    pred_levels = pred.apply(lambda c: OVERTURE_HIERARCHY.get(c, 6))

    # === Accuracy metrics ===
    analysis.exact_accuracy = (true == pred).mean()
    analysis.tier_accuracy = (true_tiers == pred_tiers).mean()
    analysis.hierarchy_accuracy_1 = (abs(true_levels - pred_levels) <= 1).mean()
    analysis.hierarchy_accuracy_2 = (abs(true_levels - pred_levels) <= 2).mean()

    # === Confusion matrices ===
    for t, p in zip(true, pred):
        if t not in analysis.confusion_matrix:
            analysis.confusion_matrix[t] = {}
        analysis.confusion_matrix[t][p] = analysis.confusion_matrix[t].get(p, 0) + 1

    for t, p in zip(true_tiers, pred_tiers):
        if t not in analysis.tier_confusion:
            analysis.tier_confusion[t] = {}
        analysis.tier_confusion[t][p] = analysis.tier_confusion[t].get(p, 0) + 1

    # === Per-class accuracy ===
    for cls in true.unique():
        mask = true == cls
        if mask.sum() > 0:
            analysis.per_class_accuracy[cls] = (pred[mask] == cls).mean()
            true_tier = OVERTURE_TIERS.get(cls, "unknown")
            analysis.per_class_tier_accuracy[cls] = (pred_tiers[mask] == true_tier).mean()

    # === Collect error examples ===
    # Tier violations (most important)
    tier_mismatch = true_tiers != pred_tiers
    for idx in true.index[tier_mismatch][:max_examples]:
        example = {
            "true_class": true.loc[idx],
            "predicted_class": pred.loc[idx],
            "true_tier": true_tiers.loc[idx],
            "predicted_tier": pred_tiers.loc[idx],
        }
        if segment_ids is not None and idx in segment_ids.index:
            example["segment_id"] = str(segment_ids.loc[idx])
        if names is not None and idx in names.index:
            name = names.loc[idx]
            if isinstance(name, dict):
                name = name.get("primary", str(name))
            example["name"] = str(name)[:50] if name else None
        analysis.tier_violations.append(example)

    # Major hierarchy errors (>2 levels)
    major_errors = abs(true_levels - pred_levels) > 2
    for idx in true.index[major_errors][:max_examples]:
        example = {
            "true_class": true.loc[idx],
            "predicted_class": pred.loc[idx],
            "true_level": int(true_levels.loc[idx]),
            "predicted_level": int(pred_levels.loc[idx]),
            "level_diff": int(abs(true_levels.loc[idx] - pred_levels.loc[idx])),
        }
        if segment_ids is not None and idx in segment_ids.index:
            example["segment_id"] = str(segment_ids.loc[idx])
        if names is not None and idx in names.index:
            name = names.loc[idx]
            if isinstance(name, dict):
                name = name.get("primary", str(name))
            example["name"] = str(name)[:50] if name else None
        analysis.major_hierarchy_errors.append(example)

    return analysis


def format_prediction_analysis(analysis: ClassPredictionAnalysis) -> str:
    """Format prediction analysis for console output."""
    lines = []
    lines.append(f"=== Class Prediction Analysis ({analysis.n_samples:,} samples) ===")
    lines.append("")

    lines.append("Accuracy Metrics:")
    lines.append(f"  Exact class match: {analysis.exact_accuracy:.1%}")
    lines.append(f"  Correct traffic tier: {analysis.tier_accuracy:.1%}")
    lines.append(f"  Within 1 hierarchy level: {analysis.hierarchy_accuracy_1:.1%}")
    lines.append(f"  Within 2 hierarchy levels: {analysis.hierarchy_accuracy_2:.1%}")
    lines.append("")

    lines.append("Tier Confusion Matrix:")
    tier_order = ["vehicle", "pedestrian", "bicycle", "other", "unknown"]
    existing_tiers = [t for t in tier_order if t in analysis.tier_confusion]

    # Header
    header = "  True\\Pred  " + "  ".join(f"{t[:6]:>6}" for t in existing_tiers)
    lines.append(header)

    for true_tier in existing_tiers:
        row_counts = analysis.tier_confusion.get(true_tier, {})
        cells = []
        for pred_tier in existing_tiers:
            count = row_counts.get(pred_tier, 0)
            cells.append(f"{count:>6}")
        lines.append(f"  {true_tier:>10}  " + "  ".join(cells))
    lines.append("")

    # Per-class accuracy (sorted by tier accuracy)
    lines.append("Per-Class Tier Accuracy:")
    sorted_classes = sorted(
        analysis.per_class_tier_accuracy.items(), key=lambda x: x[1], reverse=True
    )
    for cls, acc in sorted_classes[:15]:
        exact_acc = analysis.per_class_accuracy.get(cls, 0)
        tier = OVERTURE_TIERS.get(cls, "?")
        lines.append(f"  {cls:20} tier={tier:10} tier_acc={acc:.1%}  exact_acc={exact_acc:.1%}")
    lines.append("")

    # Tier violations
    if analysis.tier_violations:
        lines.append(f"Tier Violations ({len(analysis.tier_violations)} examples):")
        for ex in analysis.tier_violations[:10]:
            name = ex.get("name", "unnamed")
            lines.append(
                f"  {ex['true_tier']:>10} -> {ex['predicted_tier']:<10} "
                f"({ex['true_class']} -> {ex['predicted_class']}) "
                f"name='{name}'"
            )
        if len(analysis.tier_violations) > 10:
            lines.append(f"  ... and {len(analysis.tier_violations) - 10} more")
    lines.append("")

    # Major hierarchy errors
    if analysis.major_hierarchy_errors:
        lines.append(
            f"Major Hierarchy Errors (>2 levels, {len(analysis.major_hierarchy_errors)} examples):"
        )
        for ex in analysis.major_hierarchy_errors[:10]:
            name = ex.get("name", "unnamed")
            lines.append(
                f"  {ex['true_class']:15} (L{ex['true_level']}) -> "
                f"{ex['predicted_class']:15} (L{ex['predicted_level']}) "
                f"diff={ex['level_diff']} name='{name}'"
            )

    return "\n".join(lines)


def analyze_source_class_mapping(
    source_classes: pd.Series,
    true_overture_classes: pd.Series,
    current_mapping: dict[str, str] | None = None,
) -> dict:
    """Analyze how source classes map to Overture classes in known matches.

    This helps identify mapping errors and suggest improvements.

    Args:
        source_classes: Series of source classification values
        true_overture_classes: Series of true Overture classes (from matches)
        current_mapping: Optional current mapping to compare against

    Returns:
        Dict with mapping analysis:
        - suggested_mapping: Most common Overture class per source class
        - confidence: How dominant the top mapping is
        - current_errors: Errors in current mapping if provided
    """
    # Align and clean
    common_idx = source_classes.index.intersection(true_overture_classes.index)
    src = source_classes.loc[common_idx].fillna("unknown").astype(str).str.lower()
    true = true_overture_classes.loc[common_idx].fillna("unknown").astype(str).str.lower()

    result = {
        "suggested_mapping": {},
        "mapping_confidence": {},
        "mapping_distribution": {},
        "current_mapping_errors": [],
        "n_samples": len(common_idx),
    }

    # Analyze each source class
    for src_class in src.unique():
        mask = src == src_class
        true_for_src = true[mask]
        counts = true_for_src.value_counts()

        if len(counts) == 0:
            continue

        # Most common mapping
        top_class = counts.index[0]
        top_count = counts.iloc[0]
        total = counts.sum()
        confidence = top_count / total

        result["suggested_mapping"][src_class] = top_class
        result["mapping_confidence"][src_class] = confidence
        result["mapping_distribution"][src_class] = counts.to_dict()

        # Check current mapping
        if current_mapping:
            current_target = current_mapping.get(src_class)
            if current_target:
                current_target = current_target.lower()
                if current_target != top_class:
                    # Potential mapping error
                    current_tier = OVERTURE_TIERS.get(current_target, "unknown")
                    suggested_tier = OVERTURE_TIERS.get(top_class, "unknown")
                    current_count = counts.get(current_target, 0)

                    result["current_mapping_errors"].append(
                        {
                            "source_class": src_class,
                            "current_mapping": current_target,
                            "current_tier": current_tier,
                            "current_count": current_count,
                            "suggested_mapping": top_class,
                            "suggested_tier": suggested_tier,
                            "suggested_count": top_count,
                            "total_samples": total,
                            "tier_mismatch": current_tier != suggested_tier,
                        }
                    )

    # Sort errors by severity (tier mismatches first, then by sample count)
    result["current_mapping_errors"].sort(
        key=lambda x: (not x["tier_mismatch"], -x["total_samples"])
    )

    return result
