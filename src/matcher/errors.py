"""Error tracking infrastructure for feature computation.

This module provides structured error tracking to surface silent failures
in worker processes and enable better debugging of feature computation issues.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorPhase(Enum):
    """Phase of feature computation where an error occurred."""

    ALIGNMENT = "alignment"
    BATCH_GEOMETRIC = "batch_geometric"
    PERPENDICULAR_OFFSET = "perpendicular_offset"
    PAIR_FEATURES = "pair_features"
    GRAPHLET = "graphlet"  # Expected failures (missing topology data)


class ErrorSeverity(Enum):
    """Severity classification for feature computation errors."""

    EXPECTED = "expected"  # Known/acceptable (e.g., missing graphlet data)
    WARNING = "warning"  # Unexpected but recoverable
    CRITICAL = "critical"  # Should not happen


@dataclass
class FeatureError:
    """A single feature computation error."""

    phase: ErrorPhase
    severity: ErrorSeverity
    error_type: str  # Exception class name
    message: str
    ref_idx: int | None = None
    target_idx: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a serializable dictionary."""
        return {
            "phase": self.phase.value,
            "severity": self.severity.value,
            "error_type": self.error_type,
            "message": self.message,
            "ref_idx": self.ref_idx,
            "target_idx": self.target_idx,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureError":
        """Create from a serializable dictionary."""
        return cls(
            phase=ErrorPhase(data["phase"]),
            severity=ErrorSeverity(data["severity"]),
            error_type=data["error_type"],
            message=data["message"],
            ref_idx=data.get("ref_idx"),
            target_idx=data.get("target_idx"),
        )


@dataclass
class ErrorAggregator:
    """Aggregates errors across feature computation phases.

    Designed to work across process boundaries by supporting serialization
    for multiprocessing workers. Tracks counts and keeps one sample error
    per phase:type combination for debugging.
    """

    counts_by_phase: dict[str, int] = field(default_factory=dict)
    counts_by_type: dict[str, int] = field(default_factory=dict)
    sample_errors: dict[str, FeatureError] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """Total number of errors tracked."""
        return sum(self.counts_by_phase.values())

    def add(self, error: FeatureError) -> None:
        """Add an error to the aggregator."""
        phase_key = error.phase.value
        type_key = error.error_type

        # Update counts
        self.counts_by_phase[phase_key] = self.counts_by_phase.get(phase_key, 0) + 1
        self.counts_by_type[type_key] = self.counts_by_type.get(type_key, 0) + 1

        # Keep one sample error per phase:type combination
        sample_key = f"{phase_key}:{type_key}"
        if sample_key not in self.sample_errors:
            self.sample_errors[sample_key] = error

    def add_simple(
        self,
        phase: ErrorPhase,
        exception: Exception,
        severity: ErrorSeverity = ErrorSeverity.WARNING,
        ref_idx: int | None = None,
        target_idx: int | None = None,
    ) -> None:
        """Convenience method to add an error from an exception."""
        self.add(
            FeatureError(
                phase=phase,
                severity=severity,
                error_type=type(exception).__name__,
                message=str(exception),
                ref_idx=ref_idx,
                target_idx=target_idx,
            )
        )

    def merge(self, other: "ErrorAggregator") -> None:
        """Merge another aggregator's errors into this one."""
        # Merge counts
        for phase, count in other.counts_by_phase.items():
            self.counts_by_phase[phase] = self.counts_by_phase.get(phase, 0) + count
        for error_type, count in other.counts_by_type.items():
            self.counts_by_type[error_type] = self.counts_by_type.get(error_type, 0) + count

        # Merge sample errors (keep first sample for each type)
        for key, error in other.sample_errors.items():
            if key not in self.sample_errors:
                self.sample_errors[key] = error

    def to_serializable(self) -> dict[str, Any]:
        """Convert to a serializable dict for pickling across processes."""
        return {
            "counts_by_phase": dict(self.counts_by_phase),
            "counts_by_type": dict(self.counts_by_type),
            "sample_errors": {k: v.to_dict() for k, v in self.sample_errors.items()},
        }

    def merge_serialized(self, data: dict[str, Any]) -> None:
        """Merge from a serialized dict (from worker process)."""
        # Merge counts
        for phase, count in data.get("counts_by_phase", {}).items():
            self.counts_by_phase[phase] = self.counts_by_phase.get(phase, 0) + count
        for error_type, count in data.get("counts_by_type", {}).items():
            self.counts_by_type[error_type] = self.counts_by_type.get(error_type, 0) + count

        # Merge sample errors (keep first sample for each type)
        for key, error_dict in data.get("sample_errors", {}).items():
            if key not in self.sample_errors:
                self.sample_errors[key] = FeatureError.from_dict(error_dict)

    def summary(self) -> dict[str, Any]:
        """Generate a summary dict for logging."""
        return {
            "total": self.total,
            "by_phase": dict(self.counts_by_phase),
            "by_type": dict(self.counts_by_type),
            "samples": {
                k: {"message": v.message, "phase": v.phase.value}
                for k, v in self.sample_errors.items()
            },
        }

    def has_errors(self) -> bool:
        """Check if any errors have been recorded."""
        return self.total > 0
