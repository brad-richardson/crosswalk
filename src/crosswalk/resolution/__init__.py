"""Match resolution and bridge file generation."""

from .bridge import BRIDGE_SCHEMA, generate_bridge_file, generate_unmatched_report

__all__ = [
    "generate_bridge_file",
    "generate_unmatched_report",
    "BRIDGE_SCHEMA",
]
