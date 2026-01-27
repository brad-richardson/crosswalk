"""Dataset configuration management.

This module provides utilities for loading and applying dataset-specific
configurations, including class mappings from source classification systems
to Overture's road hierarchy.

Example usage:
    from matcher.datasets import load_dataset_config, apply_class_mapping

    config = load_dataset_config("boston_streets")
    gdf = apply_class_mapping(gdf, config)

    from matcher.datasets import DatasetLoader

    loader = DatasetLoader()
    pair = loader.load_pair("us_boston_streets")
"""

from .config import (
    ClassMappingRule,
    DatasetConfig,
    apply_class_mapping,
    list_dataset_configs,
    load_dataset_config,
)
from .loader import DatasetLoader, LoadedPair

__all__ = [
    "DatasetConfig",
    "ClassMappingRule",
    "DatasetLoader",
    "LoadedPair",
    "load_dataset_config",
    "list_dataset_configs",
    "apply_class_mapping",
]
