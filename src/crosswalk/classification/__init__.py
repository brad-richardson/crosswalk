"""Classification module for predicting road classes.

This module provides tools for predicting Overture road classes from
source features like names, physical attributes, and geometry.
"""

from .predictor import (
    NAME_PATTERNS,
    OVERTURE_HIERARCHY,
    OVERTURE_TIERS,
    SOURCE_CLASS_KEYWORDS,
    ClassPredictionAnalysis,
    LightweightClassPredictor,
    analyze_predictions,
    analyze_source_class_mapping,
    format_prediction_analysis,
    predict_class_from_name,
)

__all__ = [
    "LightweightClassPredictor",
    "NAME_PATTERNS",
    "SOURCE_CLASS_KEYWORDS",
    "OVERTURE_TIERS",
    "OVERTURE_HIERARCHY",
    "ClassPredictionAnalysis",
    "analyze_predictions",
    "format_prediction_analysis",
    "analyze_source_class_mapping",
    "predict_class_from_name",
]
