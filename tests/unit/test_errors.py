"""Tests for error tracking infrastructure."""

from matcher.errors import (
    ErrorAggregator,
    ErrorPhase,
    ErrorSeverity,
    FeatureError,
)


class TestFeatureError:
    """Tests for FeatureError dataclass."""

    def test_create_feature_error(self):
        """Test basic FeatureError creation."""
        error = FeatureError(
            phase=ErrorPhase.BATCH_GEOMETRIC,
            severity=ErrorSeverity.WARNING,
            error_type="ValueError",
            message="test error message",
            ref_idx=10,
            target_idx=20,
        )

        assert error.phase == ErrorPhase.BATCH_GEOMETRIC
        assert error.severity == ErrorSeverity.WARNING
        assert error.error_type == "ValueError"
        assert error.message == "test error message"
        assert error.ref_idx == 10
        assert error.target_idx == 20

    def test_to_dict(self):
        """Test serialization to dict."""
        error = FeatureError(
            phase=ErrorPhase.ALIGNMENT,
            severity=ErrorSeverity.CRITICAL,
            error_type="RuntimeError",
            message="alignment failed",
            ref_idx=5,
            target_idx=None,
        )

        d = error.to_dict()

        assert d["phase"] == "alignment"
        assert d["severity"] == "critical"
        assert d["error_type"] == "RuntimeError"
        assert d["message"] == "alignment failed"
        assert d["ref_idx"] == 5
        assert d["target_idx"] is None

    def test_from_dict_roundtrip(self):
        """Test deserialization from dict."""
        original = FeatureError(
            phase=ErrorPhase.PERPENDICULAR_OFFSET,
            severity=ErrorSeverity.EXPECTED,
            error_type="KeyError",
            message="missing key",
            ref_idx=1,
            target_idx=2,
        )

        restored = FeatureError.from_dict(original.to_dict())

        assert restored.phase == original.phase
        assert restored.severity == original.severity
        assert restored.error_type == original.error_type
        assert restored.message == original.message
        assert restored.ref_idx == original.ref_idx
        assert restored.target_idx == original.target_idx


class TestErrorAggregator:
    """Tests for ErrorAggregator."""

    def test_empty_aggregator(self):
        """Test empty aggregator state."""
        agg = ErrorAggregator()

        assert agg.total == 0
        assert not agg.has_errors()
        assert agg.counts_by_phase == {}
        assert agg.counts_by_type == {}
        assert agg.sample_errors == {}

    def test_add_error(self):
        """Test adding a single error."""
        agg = ErrorAggregator()
        error = FeatureError(
            phase=ErrorPhase.BATCH_GEOMETRIC,
            severity=ErrorSeverity.WARNING,
            error_type="ValueError",
            message="test",
        )

        agg.add(error)

        assert agg.total == 1
        assert agg.has_errors()
        assert agg.counts_by_phase == {"batch_geometric": 1}
        assert agg.counts_by_type == {"ValueError": 1}
        assert "batch_geometric:ValueError" in agg.sample_errors

    def test_add_simple(self):
        """Test add_simple convenience method."""
        agg = ErrorAggregator()

        agg.add_simple(
            ErrorPhase.PAIR_FEATURES,
            ValueError("simple error"),
            ErrorSeverity.WARNING,
            ref_idx=10,
            target_idx=20,
        )

        assert agg.total == 1
        assert agg.counts_by_phase == {"pair_features": 1}
        assert agg.counts_by_type == {"ValueError": 1}
        sample = agg.sample_errors["pair_features:ValueError"]
        assert sample.ref_idx == 10
        assert sample.target_idx == 20

    def test_add_multiple_same_type(self):
        """Test adding multiple errors of the same type."""
        agg = ErrorAggregator()

        for i in range(5):
            agg.add_simple(
                ErrorPhase.BATCH_GEOMETRIC,
                ValueError(f"error {i}"),
                ErrorSeverity.WARNING,
            )

        assert agg.total == 5
        assert agg.counts_by_phase == {"batch_geometric": 5}
        assert agg.counts_by_type == {"ValueError": 5}
        # Only first sample is kept
        sample = agg.sample_errors["batch_geometric:ValueError"]
        assert sample.message == "error 0"

    def test_add_different_types(self):
        """Test adding errors of different types and phases."""
        agg = ErrorAggregator()

        agg.add_simple(ErrorPhase.BATCH_GEOMETRIC, ValueError("v1"))
        agg.add_simple(ErrorPhase.BATCH_GEOMETRIC, RuntimeError("r1"))
        agg.add_simple(ErrorPhase.PERPENDICULAR_OFFSET, ValueError("v2"))
        agg.add_simple(ErrorPhase.PAIR_FEATURES, KeyError("k1"))

        assert agg.total == 4
        assert agg.counts_by_phase == {
            "batch_geometric": 2,
            "perpendicular_offset": 1,
            "pair_features": 1,
        }
        assert agg.counts_by_type == {
            "ValueError": 2,
            "RuntimeError": 1,
            "KeyError": 1,
        }
        assert len(agg.sample_errors) == 4

    def test_merge(self):
        """Test merging two aggregators."""
        agg1 = ErrorAggregator()
        agg1.add_simple(ErrorPhase.BATCH_GEOMETRIC, ValueError("v1"))
        agg1.add_simple(ErrorPhase.BATCH_GEOMETRIC, ValueError("v2"))

        agg2 = ErrorAggregator()
        agg2.add_simple(ErrorPhase.BATCH_GEOMETRIC, ValueError("v3"))
        agg2.add_simple(ErrorPhase.PERPENDICULAR_OFFSET, RuntimeError("r1"))

        agg1.merge(agg2)

        assert agg1.total == 4
        assert agg1.counts_by_phase == {"batch_geometric": 3, "perpendicular_offset": 1}
        assert agg1.counts_by_type == {"ValueError": 3, "RuntimeError": 1}
        # Original sample is kept for batch_geometric:ValueError
        assert agg1.sample_errors["batch_geometric:ValueError"].message == "v1"
        # New sample is added for perpendicular_offset:RuntimeError
        assert agg1.sample_errors["perpendicular_offset:RuntimeError"].message == "r1"

    def test_to_serializable(self):
        """Test serialization for multiprocessing."""
        agg = ErrorAggregator()
        agg.add_simple(
            ErrorPhase.BATCH_GEOMETRIC,
            ValueError("test"),
            ErrorSeverity.WARNING,
            ref_idx=1,
            target_idx=2,
        )

        serialized = agg.to_serializable()

        assert serialized["counts_by_phase"] == {"batch_geometric": 1}
        assert serialized["counts_by_type"] == {"ValueError": 1}
        assert "batch_geometric:ValueError" in serialized["sample_errors"]
        sample = serialized["sample_errors"]["batch_geometric:ValueError"]
        assert sample["phase"] == "batch_geometric"
        assert sample["severity"] == "warning"
        assert sample["error_type"] == "ValueError"
        assert sample["message"] == "test"
        assert sample["ref_idx"] == 1
        assert sample["target_idx"] == 2

    def test_merge_serialized(self):
        """Test merging from serialized dict."""
        agg1 = ErrorAggregator()
        agg1.add_simple(ErrorPhase.BATCH_GEOMETRIC, ValueError("original"))

        # Simulate worker results
        worker_data = {
            "counts_by_phase": {"batch_geometric": 2, "pair_features": 1},
            "counts_by_type": {"ValueError": 2, "KeyError": 1},
            "sample_errors": {
                "batch_geometric:ValueError": {
                    "phase": "batch_geometric",
                    "severity": "warning",
                    "error_type": "ValueError",
                    "message": "worker error",
                    "ref_idx": None,
                    "target_idx": None,
                },
                "pair_features:KeyError": {
                    "phase": "pair_features",
                    "severity": "warning",
                    "error_type": "KeyError",
                    "message": "missing",
                    "ref_idx": 10,
                    "target_idx": 20,
                },
            },
        }

        agg1.merge_serialized(worker_data)

        assert agg1.total == 4  # 1 original + 3 from worker
        assert agg1.counts_by_phase == {"batch_geometric": 3, "pair_features": 1}
        assert agg1.counts_by_type == {"ValueError": 3, "KeyError": 1}
        # Original sample is kept
        assert agg1.sample_errors["batch_geometric:ValueError"].message == "original"
        # New sample is added
        assert agg1.sample_errors["pair_features:KeyError"].message == "missing"

    def test_summary(self):
        """Test summary generation."""
        agg = ErrorAggregator()
        agg.add_simple(ErrorPhase.BATCH_GEOMETRIC, ValueError("v1"))
        agg.add_simple(ErrorPhase.BATCH_GEOMETRIC, RuntimeError("r1"))
        agg.add_simple(ErrorPhase.PAIR_FEATURES, ValueError("v2"))

        summary = agg.summary()

        assert summary["total"] == 3
        assert summary["by_phase"] == {"batch_geometric": 2, "pair_features": 1}
        assert summary["by_type"] == {"ValueError": 2, "RuntimeError": 1}
        assert len(summary["samples"]) == 3
        for _key, sample_info in summary["samples"].items():
            assert "message" in sample_info
            assert "phase" in sample_info

    def test_roundtrip_serialization(self):
        """Test full serialization roundtrip (simulating multiprocessing)."""
        # Worker aggregator
        worker_agg = ErrorAggregator()
        worker_agg.add_simple(
            ErrorPhase.BATCH_GEOMETRIC,
            ValueError("batch failed"),
            ErrorSeverity.CRITICAL,
            ref_idx=100,
            target_idx=200,
        )
        worker_agg.add_simple(
            ErrorPhase.PAIR_FEATURES,
            KeyError("missing key"),
            ErrorSeverity.WARNING,
        )

        # Serialize (as if sending across process boundary)
        serialized = worker_agg.to_serializable()

        # Parent process aggregator
        parent_agg = ErrorAggregator()
        parent_agg.merge_serialized(serialized)

        assert parent_agg.total == 2
        assert parent_agg.counts_by_phase == {"batch_geometric": 1, "pair_features": 1}
        assert parent_agg.counts_by_type == {"ValueError": 1, "KeyError": 1}

        # Verify sample errors are properly restored
        batch_sample = parent_agg.sample_errors["batch_geometric:ValueError"]
        assert batch_sample.phase == ErrorPhase.BATCH_GEOMETRIC
        assert batch_sample.severity == ErrorSeverity.CRITICAL
        assert batch_sample.ref_idx == 100
        assert batch_sample.target_idx == 200


class TestErrorPhase:
    """Tests for ErrorPhase enum."""

    def test_all_phases(self):
        """Ensure all expected phases exist."""
        expected = {
            "alignment",
            "batch_geometric",
            "perpendicular_offset",
            "pair_features",
            "graphlet",
        }
        actual = {phase.value for phase in ErrorPhase}
        assert actual == expected


class TestErrorSeverity:
    """Tests for ErrorSeverity enum."""

    def test_all_severities(self):
        """Ensure all expected severities exist."""
        expected = {"expected", "warning", "critical"}
        actual = {severity.value for severity in ErrorSeverity}
        assert actual == expected
