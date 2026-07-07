"""Toy end-to-end test of the Spark + Sedona integration path.

This is a deliberately *minimal* smoke test whose only job is to prove the
Spark/Sedona stack actually loads and runs a geospatial job end-to-end, plus
that the Spark-portable booster this package ships (``crosswalk.spark``) loads
and scores inside a real ``SparkSession``. It is not comprehensive coverage.

It exercises three things the production Spark consumer (the tf-data-platform
sister project) relies on:

1. A local ``SparkSession`` with Sedona registered (``SedonaContext``) starts and
   the Sedona SQL functions (``ST_GeomFromWKT``, ``ST_DWithin``, ...) resolve —
   i.e. the Sedona jars pulled via ``spark.jars.packages`` load correctly.
2. A Sedona spatial join over toy WKT linestrings selects the overlapping
   ref/target pair and rejects the far-apart one (a real distributed geospatial
   job, however tiny).
3. The bundled Spark-portable XGBoost booster (``crosswalk.spark.spark_model_json``)
   loads inside a Spark ``applyInPandas`` UDF, is fed features in
   ``manifest["features"]`` order, and scores an obvious-match feature row higher
   than an obvious-non-match row.

Requires the ``[spark]`` extra (pyspark + apache-sedona) *and* a JDK on the host.
Without either, every test here SKIPS (never errors), so a plain
``uv run pytest`` that did not install the extra stays green. Select these tests
explicitly with ``-m spark`` (see the ``spark`` CI job).

First run downloads the Sedona jars from Maven Central (network required) — this
can take a minute or two before the session is ready.
"""

from __future__ import annotations

import os
import shutil

import pytest

# --- Skip guards -----------------------------------------------------------
# Guard the heavy imports so the suite skips (never errors) when the [spark]
# extra or a JDK is absent.
pyspark = pytest.importorskip("pyspark", reason="pyspark not installed ([spark] extra)")
pytest.importorskip("sedona", reason="apache-sedona not installed ([spark] extra)")


def _java_available() -> bool:
    """True if a JDK/JRE is reachable (JAVA_HOME/bin/java or java on PATH)."""
    java_home = os.environ.get("JAVA_HOME")
    if java_home and os.path.exists(os.path.join(java_home, "bin", "java")):
        return True
    return shutil.which("java") is not None


pytestmark = [
    pytest.mark.spark,
    pytest.mark.skipif(not _java_available(), reason="no JDK found (need Java for Spark)"),
]

# pyspark 3.5.x <-> apache-sedona 1.6.x. The shaded jar bundles Sedona + its
# GeoTools deps; the geotools-wrapper supplies the remaining GeoTools classes.
_SEDONA_PACKAGES = (
    "org.apache.sedona:sedona-spark-shaded-3.5_2.12:1.6.1,"
    "org.datasyslab:geotools-wrapper:1.6.1-28.2"
)


@pytest.fixture(scope="module")
def sedona(tmp_path_factory):
    """A local single-worker SparkSession with Sedona registered."""
    import sys

    from sedona.spark import SedonaContext

    # Pin the Python worker to this very interpreter (the one with pyspark +
    # xgboost + pandas installed). Without this, Spark launches workers with a
    # bare `python3` from PATH and fails with "check PYSPARK_PYTHON ...".
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    warehouse = tmp_path_factory.mktemp("spark-warehouse")
    builder = (
        SedonaContext.builder()
        .master("local[1]")
        .appName("crosswalk-sedona-e2e")
        .config("spark.jars.packages", _SEDONA_PACKAGES)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config(
            "spark.kryo.registrator",
            "org.apache.sedona.core.serde.SedonaKryoRegistrator",
        )
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.warehouse.dir", str(warehouse))
    )
    spark = builder.getOrCreate()
    session = SedonaContext.create(spark)
    try:
        yield session
    finally:
        spark.stop()


def test_sedona_spatial_join_selects_overlapping_pair(sedona):
    """Sedona ST_DWithin join keeps the near pair, drops the far pair.

    Two ref linestrings and two targets: ``tgt_near`` is ~1e-5 deg off ``ref_a``
    (effectively overlapping); ``tgt_far`` is ~100 deg away. The distance join
    must produce exactly the (ref_a, tgt_near) candidate.
    """
    ref_rows = [
        ("ref_a", "LINESTRING (0 0, 0 10)"),
        ("ref_b", "LINESTRING (0 0, 10 0)"),
    ]
    tgt_rows = [
        ("tgt_near", "LINESTRING (0.00001 0, 0.00001 10)"),  # hugs ref_a
        ("tgt_far", "LINESTRING (100 100, 100 110)"),  # nowhere near anything
    ]

    ref_df = sedona.createDataFrame(ref_rows, ["rid", "wkt"]).selectExpr(
        "rid", "ST_GeomFromWKT(wkt) AS geom"
    )
    tgt_df = sedona.createDataFrame(tgt_rows, ["tid", "wkt"]).selectExpr(
        "tid", "ST_GeomFromWKT(wkt) AS geom"
    )
    ref_df.createOrReplaceTempView("ref")
    tgt_df.createOrReplaceTempView("tgt")

    pairs = sedona.sql(
        """
        SELECT r.rid AS rid, t.tid AS tid
        FROM ref r JOIN tgt t
        ON ST_DWithin(r.geom, t.geom, 0.001)
        """
    ).collect()

    got = {(row["rid"], row["tid"]) for row in pairs}
    assert ("ref_a", "tgt_near") in got, f"expected near pair in join result, got {got}"
    assert not any(tid == "tgt_far" for _, tid in got), (
        f"far target should not join with anything, got {got}"
    )


# --- Toy feature vectors for the shipped booster ---------------------------
# Directional values: distance/divergence features LOW and overlap/coverage/name
# features HIGH describe an obvious match; the inverse describes an obvious
# non-match. We only assert match > non-match (and match is confident), so the
# exact magnitudes need not be calibrated.
_MATCH_FEATURES = {
    "hausdorff_distance_m": 1.0,
    "mean_hausdorff_distance_m": 0.5,
    "hausdorff_p95_m": 1.5,
    "buffer_iou_5m": 0.95,
    "buffer_iou_15m": 0.98,
    "heading_delta": 1.0,
    "collinear_gap_ratio": 0.0,
    "edge_distance_rmse_m": 0.5,
    "name_levenshtein": 1.0,
    "name_token_sort": 1.0,
    "name_numeric_match": 1.0,
    "class_similarity": 1.0,
    "lateral_offset_m": 0.5,
    "lateral_offset_iqr_m": 0.2,
    "lateral_offset_p95_m": 1.0,
    "ref_coverage": 0.98,
    "target_coverage": 0.98,
    "min_coverage": 0.98,
    "coverage_ratio": 1.0,
    "sinuosity_ref": 1.0,
    "sinuosity_target": 1.0,
    "heading_consistency_target": 1.0,
    "min_length_m": 50.0,
    "aligned_length_m": 50.0,
    "shape_complexity_target": 0.1,
    "offset_over_expected_halfwidth": 0.1,
    "post_node_continuation_m": 10.0,
    "endpoint_heading_divergence": 2.0,
}
_NON_MATCH_FEATURES = {
    "hausdorff_distance_m": 500.0,
    "mean_hausdorff_distance_m": 400.0,
    "hausdorff_p95_m": 600.0,
    "buffer_iou_5m": 0.0,
    "buffer_iou_15m": 0.0,
    "heading_delta": 90.0,
    "collinear_gap_ratio": 5.0,
    "edge_distance_rmse_m": 300.0,
    "name_levenshtein": 0.0,
    "name_token_sort": 0.0,
    "name_numeric_match": 0.0,
    "class_similarity": 0.0,
    "lateral_offset_m": 300.0,
    "lateral_offset_iqr_m": 200.0,
    "lateral_offset_p95_m": 500.0,
    "ref_coverage": 0.01,
    "target_coverage": 0.02,
    "min_coverage": 0.01,
    "coverage_ratio": 0.02,
    "sinuosity_ref": 1.0,
    "sinuosity_target": 3.0,
    "heading_consistency_target": 0.1,
    "min_length_m": 5.0,
    "aligned_length_m": 0.5,
    "shape_complexity_target": 5.0,
    "offset_over_expected_halfwidth": 10.0,
    "post_node_continuation_m": 0.0,
    "endpoint_heading_divergence": 90.0,
}


def test_shipped_booster_scores_in_spark(sedona):
    """The bundled Spark-portable booster loads + scores inside a Spark UDF.

    Reconstructs the documented Spark scoring path (SPARK_MODEL_CARD.md /
    crosswalk.spark): broadcast the ordered ``manifest["features"]``, load the
    booster from ``spark_model_json()``, and predict on features in that order.
    Asserts the obvious-match row scores higher than the obvious-non-match row
    and both scores are finite probabilities.
    """
    import pandas as pd

    from crosswalk.spark import spark_manifest

    features = spark_manifest()["features"]

    match_row = {"row_kind": "match", **_MATCH_FEATURES}
    non_match_row = {"row_kind": "non_match", **_NON_MATCH_FEATURES}
    pdf = pd.DataFrame([match_row, non_match_row])
    # Column order fed to Spark does not matter — the UDF reindexes to
    # `features` order before building the DMatrix (mirrors the real consumer).
    sdf = sedona.createDataFrame(pdf)

    result_schema = "row_kind string, score double"

    def _score(group: pd.DataFrame) -> pd.DataFrame:
        import xgboost as xgb

        from crosswalk.spark import spark_model_json

        booster = xgb.Booster()
        booster.load_model(bytearray(spark_model_json().encode()))
        x = group[features].astype("float32")
        dmatrix = xgb.DMatrix(x.values, feature_names=features)
        preds = booster.predict(dmatrix)
        return pd.DataFrame({"row_kind": group["row_kind"].to_numpy(), "score": preds})

    scored = sedona.createDataFrame(sdf.rdd, sdf.schema)  # ensure a plain DataFrame
    rows = scored.groupby("row_kind").applyInPandas(_score, schema=result_schema).collect()

    scores = {row["row_kind"]: row["score"] for row in rows}
    assert set(scores) == {"match", "non_match"}, scores
    assert all(0.0 <= s <= 1.0 for s in scores.values()), scores
    assert scores["match"] > scores["non_match"], (
        f"obvious-match row should outscore non-match row: {scores}"
    )
    assert scores["match"] > 0.5, f"obvious-match row should score confidently: {scores}"
