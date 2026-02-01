"""Tests for relational feature extraction."""

import geopandas as gpd
import numpy as np
import pytest
from shapely import LineString

from matcher.features.relational import (
    compute_endpoint_proximity,
    compute_parallel_alignment,
    compute_perpendicular_offset,
    compute_perpendicular_offset_batch,
    compute_side_of_street,
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

    @pytest.mark.parametrize(
        "sidewalk,expected_side",
        [
            (LineString([(0, 3), (100, 3)]), "left"),  # North of east-going road
            (LineString([(0, -3), (100, -3)]), "right"),  # South of east-going road
        ],
    )
    def test_side_of_street(self, sidewalk, expected_side):
        road = LineString([(0, 0), (100, 0)])
        side, confidence = compute_side_of_street(sidewalk, road)
        assert side == expected_side
        assert confidence > 0.8

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
        """Empty signatures should return 0."""
        from matcher.features.spatial_context import compute_degree_signature_similarity

        assert compute_degree_signature_similarity((), (1, 2)) == 0.0
        assert compute_degree_signature_similarity((1, 2), ()) == 0.0


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
        has_sibling, dist = find_parallel_sibling(
            segment=eastbound,
            segment_id="eb",
            segment_name="I-90",
            segment_class="motorway",
            spatial_index=spatial_index,
            segment_data=segment_data,
        )
        assert has_sibling is True
        assert dist == pytest.approx(15.0, abs=1.0)

        # Westbound should find eastbound as sibling
        has_sibling, dist = find_parallel_sibling(
            segment=westbound,
            segment_id="wb",
            segment_name="I-90",
            segment_class="motorway",
            spatial_index=spatial_index,
            segment_data=segment_data,
        )
        assert has_sibling is True
        assert dist == pytest.approx(15.0, abs=1.0)

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
        has_sibling, dist = find_parallel_sibling(
            segment=main_st,
            segment_id="main",
            segment_name="Main Street",
            segment_class="residential",
            spatial_index=spatial_index,
            segment_data=segment_data,
        )
        assert has_sibling is False
        assert dist == float("inf")

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

        has_sibling, dist = find_parallel_sibling(
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

        has_sibling, dist = find_parallel_sibling(
            segment=horizontal,
            segment_id="h",
            segment_name="I-90",
            segment_class="motorway",
            spatial_index=spatial_index,
            segment_data=segment_data,
        )
        assert has_sibling is False  # Not parallel enough

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
                geom = LineString(
                    [(x_start, prev_row_y + 15), (x_start + 400, prev_row_y + 15)]
                )
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
