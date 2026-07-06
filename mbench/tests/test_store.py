"""Tests for results storage."""

from mbench.results.store import (
    BenchmarkResult,
    compare_results,
    create_result,
    load_results,
    save_result,
)


def test_create_result():
    r = create_result("crosswalk", "boston", {"f1": 0.85}, {"version": "1.0"})
    assert r.tool == "crosswalk"
    assert r.dataset == "boston"
    assert r.metrics["f1"] == 0.85
    assert r.timestamp


def test_save_and_load(tmp_path):
    path = tmp_path / "results.jsonl"

    r1 = create_result("crosswalk", "boston", {"f1": 0.85})
    r2 = create_result("hootenanny", "boston", {"f1": 0.70})

    save_result(r1, path)
    save_result(r2, path)

    loaded = load_results(path)
    assert len(loaded) == 2
    assert loaded[0].tool == "crosswalk"
    assert loaded[1].tool == "hootenanny"


def test_load_nonexistent(tmp_path):
    results = load_results(tmp_path / "nope.jsonl")
    assert results == []


def test_benchmark_result_roundtrip():
    r = BenchmarkResult(
        tool="crosswalk",
        dataset="boston",
        timestamp="2026-02-17T00:00:00Z",
        metrics={"f1": 0.85, "precision": 0.9},
        metadata={"model": "xgboost"},
    )
    json_str = r.to_json()
    loaded = BenchmarkResult.from_json(json_str)
    assert loaded.tool == r.tool
    assert loaded.metrics["f1"] == 0.85
    assert loaded.metadata["model"] == "xgboost"


def test_compare_results_returns_table():
    results = [
        create_result(
            "crosswalk",
            "boston",
            {
                "precision": 0.9,
                "recall": 0.8,
                "f1": 0.85,
                "true_positives": 10,
                "false_positives": 1,
                "false_negatives": 2,
            },
        ),
        create_result(
            "hootenanny",
            "boston",
            {
                "precision": 0.7,
                "recall": 0.6,
                "f1": 0.65,
                "true_positives": 8,
                "false_positives": 3,
                "false_negatives": 4,
            },
        ),
    ]
    table = compare_results(results)
    assert table.title == "Benchmark Comparison"
    assert table.row_count == 2
