"""Quality fingerprint module for dataset analysis.

Generates comprehensive quality metrics for road network datasets,
useful for comparing datasets and tracking quality over time.
"""

from .class_analysis import (
    ClassAnalysisReport,
    analyze_class_confusion_from_bridge,
    analyze_class_confusion_from_labels,
    format_analysis_report,
)
from .fingerprint import QualityFingerprint
from .metrics import compute_quality_metrics
from .non_road_detection import (
    KNOWN_NON_ROAD_TYPE_CODES,
    NonRoadDetectionReport,
    analyze_non_road_features,
    compute_compactness_ratio,
    detect_non_road_features,
    format_non_road_report,
    is_closed_loop,
)
from .report import (
    compare_fingerprints,
    generate_quality_report,
    load_quality_report,
    save_quality_report,
)

__all__ = [
    # Fingerprint
    "QualityFingerprint",
    "compute_quality_metrics",
    "compare_fingerprints",
    "generate_quality_report",
    "load_quality_report",
    "save_quality_report",
    # Class analysis
    "ClassAnalysisReport",
    "analyze_class_confusion_from_labels",
    "analyze_class_confusion_from_bridge",
    "format_analysis_report",
    # Non-road detection
    "NonRoadDetectionReport",
    "detect_non_road_features",
    "analyze_non_road_features",
    "format_non_road_report",
    "is_closed_loop",
    "compute_compactness_ratio",
    "KNOWN_NON_ROAD_TYPE_CODES",
]
