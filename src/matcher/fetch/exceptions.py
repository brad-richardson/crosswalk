"""Exceptions for the fetch pipeline."""


class QualityRegressionError(ValueError):
    """Raised when a re-fetched dataset shows catastrophic quality regression.

    This error is raised when key quality metrics (name coverage, class coverage,
    segment count) deviate significantly from the saved quality fingerprint,
    suggesting a data quality issue that should be investigated.
    """
