"""Tests for relational feature extraction."""

import math

import geopandas as gpd
import numpy as np
import pytest
from shapely import LineString

from matcher.features.relational import (
    compute_endpoint_proximity,
    compute_parallel_alignment,
    compute_perpendicular_offset,
    compute_perpendicular_offset_batch,
)
from matcher.features.spatial_context import (
    AnchorRoadMatcher,
    SpatialContextIndex,
    compute_endpoint_features,
)


class TestRelationalFeatures:
    """Tests for core relational feature functions."""

    @pytest.mark.parametrize(
        "line_b,expected_alignment",
        [
            (LineString([(0, 10), (100, 10)]), 1.0),  # Parallel
            (LineString([(100, 10), (0, 10)]), 1.0),  # Parallel, reversed
            (LineString([(50, 0), (50, 100)]), 0.0),  # Perpendicular
            (LineString([(0, 0), (100, 100)]), 0.5),  # 45 degrees
        ],
    )
    def test_parallel_alignment(self, line_b, expected_alignment):
        line_a = LineString([(0, 0), (100, 0)])
        assert compute_parallel_alignment(line_a, line_b) == pytest.approx(
            expected_alignment, abs=0.1
        )

    def test_perpendicular_offset(self):
        road = LineString([(0, 0), (100, 0)])
        sidewalk = LineString([(0, 3), (100, 3)])
        offset, iqr, p95 = compute_perpendicular_offset(sidewalk, road)
        assert offset == pytest.approx(3.0, abs=0.1)
        assert iqr < 0.5  # Consistent parallel offset (low IQR)
        assert p95 < 4.0  # P95 should be close to mean for consistent offset

    def test_endpoint_proximity(self):
        target = LineString([(0, 0), (100, 0)])
        other_endpoints = np.array([[0, 1], [200, 0]])  # One near start, one far
        start_prox, end_prox, shared = compute_endpoint_proximity(
            target, other_endpoints, tolerance_m=5.0
        )
        assert start_prox == pytest.approx(1.0, abs=0.1)
        assert end_prox > 50
        assert shared >= 1

    def test_perpendicular_offset_partial_overlap(self):
        """Lateral offset should be computed on overlapping portion only.

        Regression test for bug where target extending beyond reference
        inflated the offset (e.g., 73m instead of 1m).
        """
        # Reference: 100m segment
        reference = LineString([(0, 0), (100, 0)])
        # Target: 300m segment, first 100m overlaps with reference at 3m offset,
        # then extends 200m beyond
        target = LineString([(0, 3), (100, 3), (300, 3)])

        # Full geometry offset would include points far from reference
        offset_full, _, _ = compute_perpendicular_offset(target, reference)
        # Points at 200m and 300m along target are >100m from reference
        assert offset_full > 50, "Full geometry should have large offset"

        # With aligned sublines (only overlapping portion), offset should be ~3m
        from matcher.features.alignment import create_subline, linestring_alignment

        alignment = linestring_alignment(reference, target)
        target_subline = create_subline(
            target, alignment.dataset_start_frac, alignment.dataset_end_frac
        )
        ref_subline = create_subline(
            reference, alignment.overture_start_frac, alignment.overture_end_frac
        )

        offset_aligned, iqr, _ = compute_perpendicular_offset(target_subline, ref_subline)
        assert offset_aligned == pytest.approx(3.0, abs=0.5), "Aligned offset should be ~3m"
        assert iqr < 1.0, "IQR should be low for consistent offset"


class TestSpatialContextIndex:
    """Tests for spatial context indexing - the primary utility for endpoint features."""

    @pytest.fixture
    def segments(self):
        return gpd.GeoDataFrame(
            {
                "id": ["s1", "s2", "s3"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(100, 0), (200, 0)]),  # Connected to s1 at (100,0)
                    LineString([(0, 50), (100, 50)]),  # Not connected
                ],
            },
            crs="EPSG:32618",
        )

    def test_infer_connectivity(self, segments):
        ctx = SpatialContextIndex()
        ctx.build_from_gdf(segments, id_column="id")
        connected = ctx.infer_connectivity(0, tolerance_m=5.0)
        assert 1 in connected  # s2 shares endpoint
        assert 2 not in connected  # s3 is far

    def test_endpoint_features(self, segments):
        ctx = SpatialContextIndex()
        ctx.build_from_gdf(segments, id_column="id")
        features = compute_endpoint_features(
            segments.iloc[0].geometry, ctx, exclude_segment_idx=0, tolerance_m=5.0
        )
        # Now uses direction-invariant min/max instead of start/end
        assert features["min_endpoint_proximity_m"] < 5.0
        assert features["shared_endpoint_count"] >= 1


class TestAnchorRoadMatcher:
    """Tests for anchor road matching."""

    def test_find_anchor_road(self):
        roads = gpd.GeoDataFrame(
            {
                "id": ["road1", "road2"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(0, 50), (100, 50)]),
                ],
            },
            crs="EPSG:32618",
        )
        matcher = AnchorRoadMatcher(roads, max_offset=10.0, id_column="id")

        match = matcher.find_anchor_road(LineString([(0, 3), (100, 3)]))
        assert match is not None
        assert match.anchor_id == "road1"
        assert match.perpendicular_offset == pytest.approx(3.0, abs=0.5)

    def test_no_anchor_when_too_far(self):
        roads = gpd.GeoDataFrame(
            {"id": ["road1"], "geometry": [LineString([(0, 0), (100, 0)])]},
            crs="EPSG:32618",
        )
        matcher = AnchorRoadMatcher(roads, max_offset=5.0, id_column="id")
        assert matcher.find_anchor_road(LineString([(0, 20), (100, 20)])) is None


class TestInferEndpointDegree:
    """Tests for topology inference from endpoint proximity."""

    @pytest.fixture
    def intersection_network(self):
        """Create a simple network with a 4-way intersection at (100, 100)."""
        return gpd.GeoDataFrame(
            {
                "id": ["n", "s", "e", "w"],
                "geometry": [
                    LineString([(100, 100), (100, 200)]),  # North
                    LineString([(100, 0), (100, 100)]),  # South
                    LineString([(100, 100), (200, 100)]),  # East
                    LineString([(0, 100), (100, 100)]),  # West
                ],
            },
            crs="EPSG:32618",
        )

    @pytest.fixture
    def dead_end_network(self):
        """Create a network with dead ends connecting at endpoints."""
        return gpd.GeoDataFrame(
            {
                "id": ["main", "dead1", "dead2"],
                "geometry": [
                    LineString([(0, 0), (100, 0)]),  # Main road
                    LineString([(100, 0), (100, 50)]),  # Dead end from main's end
                    LineString([(200, 0), (250, 0)]),  # Isolated (both ends dead)
                ],
            },
            crs="EPSG:32618",
        )

    def test_infer_degree_at_intersection(self, intersection_network):
        """Segments meeting at intersection should have high degree at that endpoint."""
        from matcher.features.spatial_context import (
            SpatialContextIndex,
            infer_endpoint_degree,
        )

        ctx = SpatialContextIndex()
        ctx.build_from_gdf(intersection_network, id_column="id")

        # North segment: start at intersection (100,100), end at (100,200)
        north_geom = intersection_network.iloc[0].geometry
        start_deg, end_deg = infer_endpoint_degree(north_geom, ctx, tolerance_m=5.0)

        # Start is at 4-way intersection, end is dead end
        assert start_deg == 4
        assert end_deg == 1

    def test_infer_degree_dead_end(self, dead_end_network):
        """Dead end segments should have degree 1 at dead end."""
        from matcher.features.spatial_context import (
            SpatialContextIndex,
            infer_endpoint_degree,
        )

        ctx = SpatialContextIndex()
        ctx.build_from_gdf(dead_end_network, id_column="id")

        # Dead end segment
        dead_geom = dead_end_network.iloc[1].geometry
        start_deg, end_deg = infer_endpoint_degree(dead_geom, ctx, tolerance_m=5.0)

        # Start connects to main road (exactly 2: dead end + main road), end is dead end
        assert start_deg == 2  # Connects to main road endpoint
        assert end_deg == 1  # Dead end

    def test_isolated_segment_degree_one(self, dead_end_network):
        """Isolated segment should have degree 1 at both ends."""
        from matcher.features.spatial_context import (
            SpatialContextIndex,
            infer_endpoint_degree,
        )

        ctx = SpatialContextIndex()
        ctx.build_from_gdf(dead_end_network, id_column="id")

        # Isolated segment (id="dead2")
        isolated_geom = dead_end_network.iloc[2].geometry
        start_deg, end_deg = infer_endpoint_degree(isolated_geom, ctx, tolerance_m=5.0)

        # Both ends are isolated
        assert start_deg == 1
        assert end_deg == 1


class TestTopologyFeatures:
    """Tests for topology feature computation."""

    @pytest.fixture
    def intersection_network(self):
        """4-way intersection at (100, 100)."""
        return gpd.GeoDataFrame(
            {
                "id": ["n", "s", "e", "w"],
                "geometry": [
                    LineString([(100, 100), (100, 200)]),
                    LineString([(100, 0), (100, 100)]),
                    LineString([(100, 100), (200, 100)]),
                    LineString([(0, 100), (100, 100)]),
                ],
            },
            crs="EPSG:32618",
        )

    def test_compute_topology_features(self, intersection_network):
        """Verify topology features are computed correctly."""
        from matcher.features.spatial_context import (
            SpatialContextIndex,
            compute_topology_features,
        )

        ctx = SpatialContextIndex()
        ctx.build_from_gdf(intersection_network, id_column="id")

        # North segment: intersection at start, dead end at end
        north_geom = intersection_network.iloc[0].geometry
        features = compute_topology_features(north_geom, ctx, tolerance_m=5.0)

        assert features["from_degree"] == 4
        assert features["to_degree"] == 1
        assert features["is_dead_end"] is True  # Has degree 1 endpoint
        assert features["is_intersection"] is True  # Has degree > 2 endpoint
        assert features["degree_signature"] == (1, 4)

    def test_degree_match_score_identical(self):
        """Identical degrees should have perfect match score."""
        from matcher.features.spatial_context import compute_degree_match_score

        score = compute_degree_match_score(4, 1, 4, 1)
        assert score == pytest.approx(1.0)

    def test_degree_match_score_swapped(self):
        """Swapped endpoints should still have perfect match score."""
        from matcher.features.spatial_context import compute_degree_match_score

        score = compute_degree_match_score(4, 1, 1, 4)  # Endpoints reversed
        assert score == pytest.approx(1.0)

    def test_degree_match_score_different(self):
        """Different degrees should have lower match score."""
        from matcher.features.spatial_context import compute_degree_match_score

        score = compute_degree_match_score(4, 1, 2, 2)
        assert score < 1.0
        assert score > 0.0

    def test_degree_signature_similarity_identical(self):
        """Identical signatures should have similarity 1.0."""
        from matcher.features.spatial_context import compute_degree_signature_similarity

        sim = compute_degree_signature_similarity((1, 4), (1, 4))
        assert sim == pytest.approx(1.0)

    def test_degree_signature_similarity_different(self):
        """Different signatures should have lower similarity."""
        from matcher.features.spatial_context import compute_degree_signature_similarity

        sim = compute_degree_signature_similarity((1, 4), (2, 2))
        assert sim < 1.0

    def test_degree_signature_similarity_empty(self):
        """Empty signatures should return NaN (missing data)."""
        from matcher.features.spatial_context import compute_degree_signature_similarity

        assert math.isnan(compute_degree_signature_similarity((), (1, 2)))
        assert math.isnan(compute_degree_signature_similarity((1, 2), ()))


class TestPerpendicularOffsetBatch:
    """Tests that batch perpendicular offset matches single-pair results."""

    def test_parallel_lines(self):
        """Batch results match single-pair for parallel lines at known offset."""
        road = LineString([(0, 0), (100, 0)])
        sidewalk = LineString([(0, 3), (100, 3)])

        # Single-pair
        mean_s, iqr_s, p95_s = compute_perpendicular_offset(sidewalk, road)

        # Batch (size 1)
        targets = np.array([sidewalk], dtype=object)
        anchors = np.array([road], dtype=object)
        means_b, iqrs_b, p95s_b = compute_perpendicular_offset_batch(targets, anchors)

        assert means_b[0] == pytest.approx(mean_s, abs=1e-6)
        assert iqrs_b[0] == pytest.approx(iqr_s, abs=1e-6)
        assert p95s_b[0] == pytest.approx(p95_s, abs=1e-6)

    def test_multiple_pairs(self):
        """Batch results match single-pair for multiple diverse geometries."""
        pairs = [
            # Parallel close
            (LineString([(0, 5), (100, 5)]), LineString([(0, 0), (100, 0)])),
            # Parallel far
            (LineString([(0, 20), (100, 20)]), LineString([(0, 0), (100, 0)])),
            # Angled
            (LineString([(0, 0), (100, 50)]), LineString([(0, 0), (100, 0)])),
            # Short segment
            (LineString([(0, 3), (10, 3)]), LineString([(0, 0), (10, 0)])),
        ]

        targets = np.array([p[0] for p in pairs], dtype=object)
        anchors = np.array([p[1] for p in pairs], dtype=object)
        means_b, iqrs_b, p95s_b = compute_perpendicular_offset_batch(targets, anchors)

        for i, (target, anchor) in enumerate(pairs):
            mean_s, iqr_s, p95_s = compute_perpendicular_offset(target, anchor)
            assert means_b[i] == pytest.approx(mean_s, abs=1e-6), f"Pair {i} mean mismatch"
            assert iqrs_b[i] == pytest.approx(iqr_s, abs=1e-6), f"Pair {i} IQR mismatch"
            assert p95s_b[i] == pytest.approx(p95_s, abs=1e-6), f"Pair {i} p95 mismatch"

    def test_empty_geometries(self):
        """Empty geometries return inf values."""
        empty = LineString()
        road = LineString([(0, 0), (100, 0)])

        targets = np.array([empty, road], dtype=object)
        anchors = np.array([road, empty], dtype=object)
        means, iqrs, p95s = compute_perpendicular_offset_batch(targets, anchors)

        assert means[0] == float("inf")
        assert means[1] == float("inf")

    def test_none_geometries(self):
        """None geometries return inf values without raising."""
        road = LineString([(0, 0), (100, 0)])
        sidewalk = LineString([(0, 3), (100, 3)])

        targets = np.array([None, sidewalk], dtype=object)
        anchors = np.array([road, None], dtype=object)
        means, iqrs, p95s = compute_perpendicular_offset_batch(targets, anchors)

        assert means[0] == float("inf")
        assert means[1] == float("inf")

    def test_empty_batch(self):
        """Empty arrays return empty results."""
        means, iqrs, p95s = compute_perpendicular_offset_batch(
            np.array([], dtype=object), np.array([], dtype=object)
        )
        assert len(means) == 0


class TestParallelSiblingDetection:
    """Tests for parallel sibling (split carriageway) detection."""

    def test_extract_route_number(self):
        """Route numbers are correctly extracted from road names."""
        from matcher.features.relational import extract_route_number

        # Interstate format
        assert extract_route_number("I-90") == "90"
        assert extract_route_number("I 90") == "90"
        assert extract_route_number("Interstate 90") == "90"

        # US Highway format
        assert extract_route_number("US-101") == "101"
        assert extract_route_number("US Highway 101") == "101"

        # State route format
        assert extract_route_number("Route 66") == "66"
        assert extract_route_number("State Highway 1") == "1"
        assert extract_route_number("SR-12") == "12"

        # UK format (M/A roads)
        assert extract_route_number("M25") == "25"
        assert extract_route_number("A1") == "1"

        # No route number
        assert extract_route_number("Main Street") is None
        assert extract_route_number("Oak Avenue") is None
        assert extract_route_number(None) is None
        assert extract_route_number("") is None

    def test_names_compatible(self):
        """Name compatibility logic for sibling detection."""
        from matcher.features.relational import names_compatible

        # Both unnamed - inconclusive, must rely on class
        assert names_compatible(None, None) is None
        assert names_compatible("", "") is None

        # One unnamed - inconclusive, must rely on class
        assert names_compatible("I-90", None) is None
        assert names_compatible(None, "Main St") is None

        # Exact match
        assert names_compatible("Main Street", "Main Street") is True
        assert names_compatible("main street", "MAIN STREET") is True

        # Route number match
        assert names_compatible("I-90", "Interstate 90") is True
        assert names_compatible("US-101", "US Highway 101") is True

        # Different names - not compatible
        assert names_compatible("Main Street", "Oak Avenue") is False
        assert names_compatible("I-90", "I-95") is False

    def test_classes_compatible(self):
        """Road class compatibility for sibling detection."""
        from matcher.features.relational import classes_compatible

        # Same class
        assert classes_compatible("motorway", "motorway") is True
        assert classes_compatible("residential", "residential") is True

        # Within 1 tier
        assert classes_compatible("motorway", "trunk") is True
        assert classes_compatible("primary", "secondary") is True

        # More than 1 tier apart
        assert classes_compatible("motorway", "residential") is False
        assert classes_compatible("primary", "service") is False

        # None/unknown - permissive
        assert classes_compatible(None, "motorway") is True
        assert classes_compatible("motorway", None) is True
        assert classes_compatible(None, None) is True

    def test_find_parallel_sibling_dual_carriageway(self):
        """Detect parallel sibling for split highway geometry."""
        from shapely import STRtree

        from matcher.features.relational import find_parallel_sibling

        # Create a split highway: two parallel lines 15m apart (typical dual carriageway)
        eastbound = LineString([(0, 0), (500, 0)])
        westbound = LineString([(0, 15), (500, 15)])  # 15m offset, parallel

        geometries = [eastbound, westbound]
        spatial_index = STRtree(geometries)
        # segment_data is parallel to geometries: [(id, name, class), ...]
        segment_data = [("eb", "I-90", "motorway"), ("wb", "I-90", "motorway")]

        # Eastbound should find westbound as sibling
        has_sibling, dist, parallel_fraction = find_parallel_sibling(
            segment=eastbound,
            segment_id="eb",
            segment_name="I-90",
            segment_class="motorway",
            spatial_index=spatial_index,
            segment_data=segment_data,
        )
        assert has_sibling is True
        assert dist == pytest.approx(15.0, abs=1.0)
        assert parallel_fraction > 0.8  # Fully parallel segments

        # Westbound should find eastbound as sibling
        has_sibling, dist, parallel_fraction = find_parallel_sibling(
            segment=westbound,
            segment_id="wb",
            segment_name="I-90",
            segment_class="motorway",
            spatial_index=spatial_index,
            segment_data=segment_data,
        )
        assert has_sibling is True
        assert dist == pytest.approx(15.0, abs=1.0)
        assert parallel_fraction > 0.8

    def test_find_parallel_sibling_no_sibling(self):
        """No sibling for isolated centerline road."""
        from shapely import STRtree

        from matcher.features.relational import find_parallel_sibling

        # Single centerline road (no parallel twin)
        main_st = LineString([(0, 0), (500, 0)])
        oak_ave = LineString([(0, 100), (500, 100)])  # Far away, different road

        geometries = [main_st, oak_ave]
        spatial_index = STRtree(geometries)
        segment_data = [
            ("main", "Main Street", "residential"),
            ("oak", "Oak Avenue", "residential"),
        ]

        # Main St should NOT find Oak Ave as sibling (too far)
        has_sibling, dist, parallel_fraction = find_parallel_sibling(
            segment=main_st,
            segment_id="main",
            segment_name="Main Street",
            segment_class="residential",
            spatial_index=spatial_index,
            segment_data=segment_data,
        )
        assert has_sibling is False
        assert dist == float("inf")
        assert parallel_fraction == 0.0

    def test_find_parallel_sibling_too_close(self):
        """Siblings too close (<5m) should not be detected."""
        from shapely import STRtree

        from matcher.features.relational import find_parallel_sibling

        # Two parallel lines only 2m apart (too close for dual carriageway)
        line_a = LineString([(0, 0), (500, 0)])
        line_b = LineString([(0, 2), (500, 2)])  # Only 2m offset

        geometries = [line_a, line_b]
        spatial_index = STRtree(geometries)
        segment_data = [("a", "I-90", "motorway"), ("b", "I-90", "motorway")]

        has_sibling, dist, parallel_fraction = find_parallel_sibling(
            segment=line_a,
            segment_id="a",
            segment_name="I-90",
            segment_class="motorway",
            spatial_index=spatial_index,
            segment_data=segment_data,
        )
        assert has_sibling is False  # 2m is below minimum threshold

    def test_find_parallel_sibling_not_parallel(self):
        """Perpendicular roads should not be detected as siblings."""
        from shapely import STRtree

        from matcher.features.relational import find_parallel_sibling

        # One horizontal, one perpendicular (crossing road)
        horizontal = LineString([(0, 0), (500, 0)])
        perpendicular = LineString([(250, -100), (250, 100)])  # Crosses at 90 degrees

        geometries = [horizontal, perpendicular]
        spatial_index = STRtree(geometries)
        segment_data = [("h", "I-90", "motorway"), ("p", "Exit 5", "motorway")]

        has_sibling, dist, parallel_fraction = find_parallel_sibling(
            segment=horizontal,
            segment_id="h",
            segment_name="I-90",
            segment_class="motorway",
            spatial_index=spatial_index,
            segment_data=segment_data,
        )
        assert has_sibling is False  # Not parallel enough

    def test_find_parallel_sibling_named_twin_still_fires(self):
        """A genuine dual carriageway (same name, ~15m offset, same span) fires.

        This is the true-positive the gate must preserve: two carriageways of the
        same named road, running parallel over the same stretch.
        """
        from shapely import STRtree

        from matcher.features.relational import find_parallel_sibling

        eastbound = LineString([(0, 0), (500, 0)])
        westbound = LineString([(0, 15), (500, 15)])  # 15m offset, same 500m span

        geometries = [eastbound, westbound]
        spatial_index = STRtree(geometries)
        segment_data = [
            ("eb", "State Route 9", "primary"),
            ("wb", "State Route 9", "primary"),
        ]

        has_sibling, dist, parallel_fraction = find_parallel_sibling(
            segment=eastbound,
            segment_id="eb",
            segment_name="State Route 9",
            segment_class="primary",
            spatial_index=spatial_index,
            segment_data=segment_data,
        )
        assert has_sibling is True
        assert dist == pytest.approx(15.0, abs=1.0)
        assert parallel_fraction > 0.8

    def test_find_parallel_sibling_unnamed_distinct_roads_no_fire(self):
        """Two distinct UNNAMED parallel roads must NOT be flagged as siblings.

        Both unnamed and the same class, ~20m apart and parallel, but the
        neighbor is much shorter (120m vs 500m). The OLD code fired on this
        (class tolerance + partial parallelism), which is exactly the
        over-firing the audit flagged. A real carriageway twin spans roughly the
        same stretch, so the comparable-extent gate rejects this.
        """
        from shapely import STRtree

        from matcher.features.relational import find_parallel_sibling

        long_road = LineString([(0, 0), (500, 0)])
        # Distinct short road 20m away, only 120m long (24% of the long road).
        short_road = LineString([(0, 20), (120, 20)])

        geometries = [long_road, short_road]
        spatial_index = STRtree(geometries)
        segment_data = [
            ("long", None, "residential"),
            ("short", None, "residential"),
        ]

        has_sibling, _dist, _pf = find_parallel_sibling(
            segment=long_road,
            segment_id="long",
            segment_name=None,
            segment_class="residential",
            spatial_index=spatial_index,
            segment_data=segment_data,
        )
        assert has_sibling is False  # Distinct road: extent not comparable

    def test_find_parallel_sibling_unnamed_vs_named_distinct_no_fire(self):
        """An unnamed query with a named DISTINCT parallel neighbor must not fire.

        The query is unnamed so names_compatible is None ("no opinion"). The two
        roads are a full-span parallel pair 20m apart with comparable extent, but
        their classes differ by one tier (residential vs unclassified). The OLD
        class-tolerance gate (max_tier_diff=1) accepted a one-tier difference and
        FIRED here; the new exact-class requirement on the unnamed path rejects
        it, since without name evidence a class mismatch is not the same road.
        """
        from shapely import STRtree

        from matcher.features.relational import find_parallel_sibling

        query = LineString([(0, 0), (500, 0)])
        neighbor = LineString([(0, 20), (500, 20)])  # parallel, same span, 20m

        geometries = [query, neighbor]
        spatial_index = STRtree(geometries)
        segment_data = [
            ("q", None, "residential"),
            ("n", "Elm Street", "unclassified"),  # one tier from residential
        ]

        has_sibling, _dist, _pf = find_parallel_sibling(
            segment=query,
            segment_id="q",
            segment_name=None,
            segment_class="residential",
            spatial_index=spatial_index,
            segment_data=segment_data,
        )
        assert has_sibling is False

    def test_find_parallel_sibling_unnamed_nan_classes_no_fire(self):
        """Unnamed segments with NaN classes must NOT count as an exact class match.

        pipeline.py builds class lists from GeoDataFrame columns, so missing
        rows yield float NaN — which is truthy (`bool(nan) is True`) and
        stringifies to "nan", so a naive comparison would treat two MISSING
        classes as identical and grant positive same-road evidence from absent
        data. Missing data is not evidence: with no names AND no classes, even
        perfect geometric twins must not fire.
        """
        import numpy as np
        from shapely import STRtree

        from matcher.features.relational import find_parallel_sibling

        # Otherwise-perfect geometric twins: same span, parallel, 15m apart
        road_a = LineString([(0, 0), (500, 0)])
        road_b = LineString([(0, 15), (500, 15)])

        geometries = [road_a, road_b]
        spatial_index = STRtree(geometries)
        segment_data = [("a", None, np.nan), ("b", None, np.nan)]

        has_sibling, _dist, _pf = find_parallel_sibling(
            segment=road_a,
            segment_id="a",
            segment_name=None,
            segment_class=np.nan,
            spatial_index=spatial_index,
            segment_data=segment_data,
        )
        assert has_sibling is False

    def test_find_parallel_sibling_unnamed_valid_identical_classes_fires(self):
        """Unnamed twins with valid identical class strings remain eligible.

        Control for the NaN-class rejection: same geometry as the NaN test but
        with real identical class strings, satisfying all three unnamed-path
        gates (exact class, high parallel fraction, comparable extent).
        """
        from shapely import STRtree

        from matcher.features.relational import find_parallel_sibling

        road_a = LineString([(0, 0), (500, 0)])
        road_b = LineString([(0, 15), (500, 15)])

        geometries = [road_a, road_b]
        spatial_index = STRtree(geometries)
        segment_data = [("a", None, "motorway"), ("b", None, "motorway")]

        has_sibling, dist, parallel_fraction = find_parallel_sibling(
            segment=road_a,
            segment_id="a",
            segment_name=None,
            segment_class="motorway",
            spatial_index=spatial_index,
            segment_data=segment_data,
        )
        assert has_sibling is True
        assert dist == pytest.approx(15.0, abs=1.0)
        assert parallel_fraction > 0.8

    def test_precompute_parallel_siblings(self):
        """Batch precomputation of sibling info for dataset."""
        from matcher.features.relational import precompute_parallel_siblings

        # Split highway geometry
        eastbound = LineString([(0, 0), (500, 0)])
        westbound = LineString([(0, 12), (500, 12)])  # 12m offset
        centerline_road = LineString([(0, 200), (500, 200)])  # Isolated road

        result = precompute_parallel_siblings(
            geometries=[eastbound, westbound, centerline_road],
            segment_ids=["eb", "wb", "center"],
            names=["I-90", "I-90", "Main Street"],
            classes=["motorway", "motorway", "residential"],
        )

        # Eastbound and westbound should have siblings
        assert result["eb"][0] is True
        assert result["eb"][1] == pytest.approx(12.0, abs=1.0)

        assert result["wb"][0] is True
        assert result["wb"][1] == pytest.approx(12.0, abs=1.0)

        # Centerline road should not have sibling
        assert result["center"][0] is False
        assert result["center"][1] == float("inf")

    def test_get_expected_half_width(self):
        """Expected half-width lookup by road class."""
        from matcher.features.relational import get_expected_half_width

        # Known classes
        assert get_expected_half_width("motorway") == 14.0
        assert get_expected_half_width("trunk") == 10.0
        assert get_expected_half_width("residential") == 3.5

        # Unknown class - default
        assert get_expected_half_width("unknown") == 4.0
        assert get_expected_half_width(None) == 4.0

        # Case insensitive
        assert get_expected_half_width("MOTORWAY") == 14.0
        assert get_expected_half_width("Residential") == 3.5

    def test_sibling_detection_performance(self):
        """Sibling detection should complete in <5 seconds for 10k segments.

        Uses a realistic road network pattern with sparse spatial distribution.
        """
        import time

        from matcher.features.relational import precompute_parallel_siblings

        # Generate 10k test segments - realistic sparse distribution
        # (not a dense grid - that's unrealistic for road networks)
        n_segments = 10000
        geometries = []
        segment_ids = []
        names = []
        classes = []

        for i in range(n_segments):
            # Spread segments across a large area with realistic spacing
            # Most roads are 100-500m apart, not 20m like in a grid
            row = i // 50  # 50 segments per "corridor"
            col = i % 50
            x_start = col * 500  # 500m spacing between roads
            y = row * 100  # 100m spacing between corridors

            # Occasionally create dual carriageways (every 10th corridor)
            if row % 10 == 0:
                # Dual carriageway - first segment
                geom = LineString([(x_start, y), (x_start + 400, y)])
                names.append(f"Highway {row // 10}")
                classes.append("motorway")
            elif row % 10 == 1:
                # Second carriageway of the dual - 15m offset from first (row above is at y-100)
                prev_row_y = (row - 1) * 100
                geom = LineString([(x_start, prev_row_y + 15), (x_start + 400, prev_row_y + 15)])
                names.append(f"Highway {row // 10}")
                classes.append("motorway")
            else:
                # Regular road
                geom = LineString([(x_start, y), (x_start + 200, y)])
                names.append(f"Street {i}")
                classes.append("residential")

            geometries.append(geom)
            segment_ids.append(f"seg_{i}")

        start = time.perf_counter()
        result = precompute_parallel_siblings(geometries, segment_ids, names, classes)
        elapsed = time.perf_counter() - start

        assert len(result) == n_segments
        # 5 second threshold - allows for CI variance while catching major regressions
        assert elapsed < 5.0, (
            f"Sibling detection too slow: {elapsed:.2f}s for {n_segments} segments"
        )

        # Verify some siblings were detected (sanity check)
        siblings_found = sum(1 for v in result.values() if v[0])
        assert siblings_found > 0, "Expected to find some siblings in test data"

    def test_find_parallel_sibling_cross_class_with_matching_names(self):
        """Sibling detection should work when names match but classes differ.

        Real-world case: East Mountain Avenue (tertiary in Overture) should match
        a parallel sibling E MOUNTAIN AVE (primary in local data) because names match.
        The 2-tier class difference (primary=2, tertiary=4) should NOT block detection
        when names clearly indicate it's the same road.
        """
        from shapely import STRtree

        from matcher.features.relational import find_parallel_sibling

        # Create a split highway where one side is classified differently
        # This happens when conflating datasets with different classification schemes
        eastbound = LineString([(0, 0), (500, 0)])
        westbound = LineString([(0, 15), (500, 15)])  # 15m offset, parallel

        geometries = [eastbound, westbound]
        spatial_index = STRtree(geometries)
        # Same road name, but different classes (primary vs tertiary = 2 tier diff)
        segment_data = [
            ("eb", "East Mountain Avenue", "tertiary"),
            ("wb", "East Mountain Avenue", "primary"),
        ]

        # Should find sibling because NAMES MATCH even though classes differ by 2 tiers
        has_sibling, dist, parallel_fraction = find_parallel_sibling(
            segment=eastbound,
            segment_id="eb",
            segment_name="East Mountain Avenue",
            segment_class="tertiary",
            spatial_index=spatial_index,
            segment_data=segment_data,
        )
        assert has_sibling is True, (
            "Should detect sibling when names match, even with 2-tier class difference"
        )
        assert dist == pytest.approx(15.0, abs=1.0)
        assert parallel_fraction > 0.8  # Should be highly parallel


class TestLocalParallelAlignment:
    """Tests for local alignment algorithm that handles partial parallelism."""

    def test_local_vs_overall_alignment_curved_segment(self):
        """Local alignment should better handle curved segments."""
        import math

        from matcher.features.relational import compute_parallel_alignment

        # Create two parallel curved segments (arcs)
        # These are locally parallel but their overall direction differs

        # Semicircular arcs - parallel at every point but overall heading differs by 180°
        n_points = 20
        radius_a = 100
        radius_b = 115  # 15m offset
        arc_a_points = []
        arc_b_points = []
        for i in range(n_points):
            angle = math.pi * i / (n_points - 1)  # 0 to pi
            arc_a_points.append((radius_a * math.cos(angle), radius_a * math.sin(angle)))
            arc_b_points.append((radius_b * math.cos(angle), radius_b * math.sin(angle)))

        arc_a = LineString(arc_a_points)
        arc_b = LineString(arc_b_points)

        # Local alignment - samples headings along the curves
        local_alignment, parallel_fraction = compute_parallel_alignment(
            arc_a, arc_b, return_fraction=True, use_local_alignment=True
        )

        # For parallel arcs, local alignment should be higher than overall
        # because the curves are locally parallel at every sample point
        assert local_alignment > 0.8, (
            f"Local alignment {local_alignment} should be high for parallel arcs"
        )
        assert parallel_fraction > 0.7, (
            f"Parallel fraction {parallel_fraction} should be high for parallel arcs"
        )

    def test_split_curve_remerge_scenario(self):
        """Test split carriageway that detours around obstacle then rejoins.

        One carriageway detours around terrain/buildings while the other stays
        straight. The algorithm should detect the parallel portions even when
        overall heading differs.
        """
        from shapely import STRtree

        from matcher.features.relational import find_parallel_sibling

        # Straight carriageway (500m long)
        straight = LineString([(0, 0), (500, 0)])

        # Curved carriageway - starts parallel, curves around obstacle, then rejoins
        # Parallel at start (0-150m), diverges (150-350m), parallel again (350-500m)
        curved_points = [
            (0, 15),  # Start parallel, 15m offset
            (100, 15),  # Still parallel
            (150, 15),  # Begin diverging
            (200, 40),  # Diverged
            (250, 50),  # Max divergence
            (300, 40),  # Reconverging
            (350, 15),  # Back to parallel
            (400, 15),  # Parallel again
            (500, 15),  # End parallel
        ]
        curved = LineString(curved_points)

        geometries = [straight, curved]
        spatial_index = STRtree(geometries)
        segment_data = [("straight", "Highway 1", "trunk"), ("curved", "Highway 1", "trunk")]

        # Should detect sibling despite the middle portion diverging
        has_sibling, dist, parallel_fraction = find_parallel_sibling(
            segment=straight,
            segment_id="straight",
            segment_name="Highway 1",
            segment_class="trunk",
            spatial_index=spatial_index,
            segment_data=segment_data,
        )

        assert has_sibling is True, "Should detect sibling even with curved section"
        # Parallel fraction should reflect that ~60% of the segment is parallel
        # (0-150m + 350-500m = 300m out of 500m)
        assert parallel_fraction >= 0.3, (
            f"Parallel fraction {parallel_fraction} should be >= 0.3 for partial parallel"
        )

    def test_diverging_at_endpoints_scenario(self):
        """Test split carriageway that diverges at both ends (cloverleaf entry/exit).

        This is the exact scenario from the user's case - segments that are
        parallel in the middle but diverge at the ends (frontage roads merging
        onto highway).
        """
        from shapely import STRtree

        from matcher.features.relational import find_parallel_sibling

        # Main highway (straight)
        highway = LineString([(0, 0), (500, 0)])

        # Frontage road - diverges at start, parallel in middle, diverges at end
        frontage_points = [
            (0, 50),  # Start diverged (50m away - merging from side road)
            (75, 30),  # Converging
            (150, 15),  # Now parallel
            (200, 15),  # Parallel
            (300, 15),  # Parallel
            (350, 15),  # Still parallel
            (425, 30),  # Diverging
            (500, 50),  # End diverged (50m away - exit)
        ]
        frontage = LineString(frontage_points)

        geometries = [highway, frontage]
        spatial_index = STRtree(geometries)
        segment_data = [("hwy", "I-90", "motorway"), ("frontage", "I-90", "motorway")]

        # Should detect sibling - middle portion is clearly parallel
        has_sibling, dist, parallel_fraction = find_parallel_sibling(
            segment=highway,
            segment_id="hwy",
            segment_name="I-90",
            segment_class="motorway",
            spatial_index=spatial_index,
            segment_data=segment_data,
        )

        assert has_sibling is True, (
            "Should detect sibling even when endpoints diverge (frontage road case)"
        )
        # Distance should use p25, capturing the parallel portion's offset (~15m)
        # not the mean which would be inflated by the diverging sections
        assert dist < 25, f"Distance {dist} should reflect parallel portion, not diverging ends"
        assert parallel_fraction >= 0.3, f"Parallel fraction {parallel_fraction} should be >= 0.3"

    def test_intermittent_median_split_touch_split(self):
        """Test segments that are parallel, touch briefly, split again.

        This handles roads with intermittent medians that alternate between
        divided and undivided sections.
        """
        from shapely import STRtree

        from matcher.features.relational import find_parallel_sibling

        # One carriageway (straight baseline)
        baseline = LineString([(0, 0), (600, 0)])

        # Other carriageway - parallel, touches, parallel again
        other_points = [
            (0, 15),  # Parallel section 1
            (150, 15),
            (175, 5),  # Converging (median ends)
            (200, 0),  # Touching
            (225, 0),  # Still touching
            (250, 5),  # Diverging (median starts again)
            (275, 15),
            (450, 15),  # Parallel section 2
            (600, 15),
        ]
        other = LineString(other_points)

        geometries = [baseline, other]
        spatial_index = STRtree(geometries)
        segment_data = [("base", "Main St", "primary"), ("other", "Main St", "primary")]

        has_sibling, dist, parallel_fraction = find_parallel_sibling(
            segment=baseline,
            segment_id="base",
            segment_name="Main St",
            segment_class="primary",
            spatial_index=spatial_index,
            segment_data=segment_data,
        )

        # Should detect sibling - significant portions are parallel
        assert has_sibling is True, "Should detect sibling with intermittent median"
        assert parallel_fraction >= 0.3, (
            f"Parallel fraction {parallel_fraction} should capture parallel portions"
        )

    def test_different_curvature_profiles(self):
        """Test one side straight, other serpentine but still parallel at consistent offset.

        This tests robustness to curvature differences - as long as the local
        headings are similar at each sample, they should be considered parallel.
        """
        from shapely import STRtree

        from matcher.features.relational import find_parallel_sibling

        # Straight baseline
        baseline = LineString([(0, 0), (500, 0)])

        # Serpentine parallel - wiggles but maintains ~15m offset
        serpentine_points = [
            (0, 15),
            (50, 17),  # Slight wiggle
            (100, 13),
            (150, 17),
            (200, 13),
            (250, 15),
            (300, 17),
            (350, 13),
            (400, 15),
            (450, 17),
            (500, 15),
        ]
        serpentine = LineString(serpentine_points)

        geometries = [baseline, serpentine]
        spatial_index = STRtree(geometries)
        segment_data = [("base", "Road A", "secondary"), ("serp", "Road A", "secondary")]

        has_sibling, dist, parallel_fraction = find_parallel_sibling(
            segment=baseline,
            segment_id="base",
            segment_name="Road A",
            segment_class="secondary",
            spatial_index=spatial_index,
            segment_data=segment_data,
        )

        # The serpentine wiggles only a few meters - overall direction is still
        # parallel. Local alignment should handle this well.
        assert has_sibling is True, "Should detect sibling despite serpentine wiggles"
        assert parallel_fraction > 0.6, (
            f"Parallel fraction {parallel_fraction} should be high for locally parallel curves"
        )
