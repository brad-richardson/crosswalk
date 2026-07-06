"""Dependency injection for the crosswalk web UI."""

from functools import lru_cache


@lru_cache(maxsize=1)
def get_matcher():
    """Load and cache the ML model for matching predictions.

    Returns the trained matcher model, loading it once and caching for reuse.
    """
    from ..ml import load_model

    return load_model()


def get_dataset_loader():
    """Get the dataset loader for accessing available datasets.

    Returns a DatasetLoader instance for discovering and loading datasets.
    """
    from ..datasets.loader import DatasetLoader

    return DatasetLoader()
