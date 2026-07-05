"""Logging setup for mbench."""

import sys

from loguru import logger


def setup_logging(verbose: bool = False) -> None:
    """Configure loguru for mbench CLI output."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, format="<level>{level:<8}</level> | {message}")
