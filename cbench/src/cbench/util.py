"""Logging setup for cbench."""

import sys

from loguru import logger


def setup_logging(verbose: bool = False) -> None:
    """Configure loguru for cbench CLI output."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, format="<level>{level:<8}</level> | {message}")
