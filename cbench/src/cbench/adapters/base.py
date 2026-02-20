"""Base protocol and types for tool adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


class EvalMode(Enum):
    """What kind of output a tool produces for evaluation."""

    PAIR_MATCH = "pair_match"  # Pair-level match identification
    # Future:
    # STITCH = "stitch"              # Graph-level consistency evaluation
    # MERGE = "merge"                # Integration quality evaluation


@dataclass
class ToolOutput:
    """Parsed output from a tool run.

    Attributes:
        matches: DataFrame with columns [ref_id, target_id, confidence].
        metadata: Tool-specific metadata (version, params, etc.).
    """

    matches: pd.DataFrame
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        required = {"ref_id", "target_id", "confidence"}
        missing = required - set(self.matches.columns)
        if missing:
            raise ValueError(f"matches DataFrame missing required columns: {missing}")


@runtime_checkable
class ToolAdapter(Protocol):
    """Protocol that every tool adapter must satisfy."""

    name: str
    eval_mode: EvalMode

    def run(self, reference: Path, target: Path, output_dir: Path, **kwargs) -> Path:
        """Run the tool and return path to its raw output.

        Args:
            reference: Path to reference parquet file.
            target: Path to target parquet file.
            output_dir: Directory for tool output files.
            **kwargs: Tool-specific options.

        Returns:
            Path to the tool's primary output file.
        """
        ...

    def parse_output(self, output_path: Path) -> ToolOutput:
        """Parse the tool's output into a standardized ToolOutput.

        Args:
            output_path: Path returned by run().

        Returns:
            Parsed matches and metadata.
        """
        ...
