"""Matcher tool adapter - shells out to `matcher stitch` CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
from loguru import logger

from cbench.adapters.base import EvalMode, ToolOutput

# Matcher project root (cbench/src/cbench/adapters/matcher.py → 4 levels up)
_MATCHER_ROOT = Path(__file__).parents[4]


class MatcherAdapter:
    """Adapter for the matcher road conflation tool.

    Runs `matcher stitch` via subprocess and parses the bridge parquet output.
    """

    name: str = "matcher"
    eval_mode: EvalMode = EvalMode.STITCH

    def run(self, reference: Path, target: Path, output_dir: Path, **kwargs) -> Path:
        """Run matcher stitch and return path to bridge parquet.

        Args:
            reference: Path to reference (Overture) parquet.
            target: Path to target parquet.
            output_dir: Directory for output files.
            **kwargs: Extra options passed as CLI flags.
                model: Model type (default: "xgboost").

        Returns:
            Path to the bridge parquet output.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        bridge_path = output_dir / "bridge.parquet"

        # Resolve to absolute paths so they work regardless of cwd
        reference = reference.resolve()
        target = target.resolve()
        bridge_path = bridge_path.resolve()

        model = kwargs.get("model", "xgboost")

        cmd = [
            "uv",
            "run",
            "matcher",
            "stitch",
            "-r",
            str(reference),
            "-t",
            str(target),
            "-m",
            model,
            "-o",
            str(bridge_path),
        ]

        timeout = int(kwargs.get("timeout", 3600))

        logger.info(f"Running: {' '.join(cmd)}")
        try:
            # Run from matcher project root so relative model path resolves
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=_MATCHER_ROOT
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"matcher stitch timed out after {timeout}s") from exc

        if result.returncode != 0:
            logger.error(f"matcher stitch failed:\n{result.stderr}")
            raise RuntimeError(f"matcher stitch exited with code {result.returncode}")

        if not bridge_path.exists():
            raise FileNotFoundError(f"Expected output not found: {bridge_path}")

        logger.info(f"Matcher output: {bridge_path}")
        return bridge_path

    def parse_output(self, output_path: Path) -> ToolOutput:
        """Parse matcher bridge parquet into standardized ToolOutput.

        Bridge parquet has columns: gers_id, local_id, confidence, match_type, ...
        Maps gers_id -> ref_id, local_id -> target_id.
        """
        bridge = pd.read_parquet(output_path)
        logger.info(f"Loaded bridge with {len(bridge)} matches")

        if "confidence" in bridge.columns:
            confidence = bridge["confidence"].astype(float)
        else:
            confidence = pd.Series(1.0, index=bridge.index)

        matches = pd.DataFrame(
            {
                "ref_id": bridge["gers_id"].astype(str),
                "target_id": bridge["local_id"].astype(str),
                "confidence": confidence,
            }
        )

        if "match_type" in bridge.columns:
            match_type_counts = bridge["match_type"].value_counts().to_dict()
        else:
            match_type_counts = {}

        metadata = {
            "match_type_counts": match_type_counts,
            "total_rows": len(bridge),
        }

        return ToolOutput(matches=matches, metadata=metadata)
