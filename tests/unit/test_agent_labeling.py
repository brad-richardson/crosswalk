"""Tests for agent_labeling module."""

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import LineString

from matcher.agent_labeling import (
    AgentLabelStore,
    SamplingConfig,
    generate_metadata_yaml,
)
from matcher.agent_labeling.context_generator import _round_value
from matcher.agent_labeling.image_renderer import (
    BACKGROUND_COLOR,
    REFERENCE_COLOR,
    ROAD_CONTEXT_COLOR,
    _draw_dashed_linestring,
    _expand_bbox,
    _geo_to_pixel,
    _get_combined_bbox,
    _to_linestring,
    render_candidate_variant,
    render_geometry_only,
    render_subline_geometry_only,
    render_subline_road_context,
)
from matcher.agent_labeling.sampler import SampledCandidate


class TestSamplingConfig:
    """Tests for SamplingConfig."""

    def test_default_config(self):
        config = SamplingConfig()
        assert config.n_candidates == 100
        assert config.seed == 42
        assert config.buffer_distance_m == 50.0
        # Default focuses on uncertain zone (0.1-0.9) where agent labels add most value
        assert "uncertain" in config.confidence_buckets
        assert config.confidence_buckets["uncertain"] == (0.1, 0.9)
        assert config.bucket_proportions["uncertain"] == 1.0

    def test_custom_config(self):
        config = SamplingConfig(
            n_candidates=50,
            seed=123,
            confidence_buckets={"test": (0.0, 1.0)},
            bucket_proportions={"test": 1.0},
        )
        assert config.n_candidates == 50
        assert config.seed == 123


class TestImageRenderer:
    """Tests for image rendering functions."""

    def test_to_linestring_with_linestring(self):
        line = LineString([(0, 0), (1, 1)])
        result = _to_linestring(line)
        assert result == line

    def test_to_linestring_with_none(self):
        result = _to_linestring(None)
        assert result is None

    def test_expand_bbox(self):
        bbox = (0, 0, 10, 10)
        expanded = _expand_bbox(bbox, padding_ratio=0.1)
        assert expanded == (-1, -1, 11, 11)

    def test_get_combined_bbox(self):
        line1 = LineString([(0, 0), (10, 0)])
        line2 = LineString([(5, 5), (15, 5)])
        bbox = _get_combined_bbox(line1, line2, padding_ratio=0.0)
        assert bbox == (0, 0, 15, 5)

    def test_geo_to_pixel(self):
        bbox = (0, 0, 10, 10)
        size = (100, 100)
        # Center point
        px, py = _geo_to_pixel(5, 5, bbox, size)
        assert px == 50
        assert py == 50  # y is inverted

    def test_render_geometry_only(self):
        ref = LineString([(0, 0), (10, 0)])
        target = LineString([(0, 1), (10, 1)])
        img = render_geometry_only(ref, target, size=(100, 100))
        assert isinstance(img, Image.Image)
        assert img.size == (100, 100)


class TestContextGenerator:
    """Tests for context generation."""

    def test_round_value_float(self):
        assert _round_value(3.14159, decimals=2) == 3.14
        assert _round_value(3.14159, decimals=4) == 3.1416

    def test_round_value_none(self):
        assert _round_value(None) is None

    def test_round_value_string(self):
        assert _round_value("test") == "test"

    def test_generate_metadata_yaml(self):
        candidate = SampledCandidate(
            ref_id="ref_123",
            target_id="target_456",
            ref_geometry=LineString([(0, 0), (10, 10)]),
            target_geometry=LineString([(1, 1), (11, 11)]),
            ref_name="Main Street",
            target_name="MAIN ST",
            ref_class="residential",
            target_class="Local Road",
            ml_confidence=0.85,
            ml_decision="match",
            features={
                "hausdorff_distance_m": 5.5,
                "buffer_iou": 0.92,
                "name_levenshtein": 0.8,
            },
            dataset="test_dataset",
            confidence_bucket="high",
        )

        yaml_str = generate_metadata_yaml(candidate, "batch_001")
        assert "ref_123" in yaml_str
        assert "target_456" in yaml_str
        assert "Main Street" in yaml_str
        assert "match" in yaml_str
        assert "0.85" in yaml_str


class TestAgentLabelStore:
    """Tests for AgentLabelStore."""

    def test_add_label(self, tmp_path):
        batch_dir = tmp_path / "test_batch"
        batch_dir.mkdir()

        store = AgentLabelStore(batch_dir, "test_agent")
        store.add_label(
            ref_id="ref_1",
            target_id="target_1",
            label="match",
            confidence=0.95,
            reasoning="Lines align well",
        )

        assert len(store.df) == 1
        assert store.df.iloc[0]["label"] == "match"

    def test_save_and_load(self, tmp_path):
        batch_dir = tmp_path / "test_batch"
        batch_dir.mkdir()

        store = AgentLabelStore(batch_dir, "test_agent")
        store.add_label("ref_1", "target_1", "match", 0.9, "test")
        store.save()

        # Create new store instance and verify data persists
        store2 = AgentLabelStore(batch_dir, "test_agent")
        assert len(store2.df) == 1

    def test_get_labeled_pairs(self, tmp_path):
        batch_dir = tmp_path / "test_batch"
        batch_dir.mkdir()

        store = AgentLabelStore(batch_dir, "test_agent")
        store.add_label("ref_1", "target_1", "match", 0.9, "")
        store.add_label("ref_2", "target_2", "no_match", 0.8, "")

        pairs = store.get_labeled_pairs()
        assert len(pairs) == 2
        assert ("ref_1", "target_1") in pairs

    def test_get_stats(self, tmp_path):
        batch_dir = tmp_path / "test_batch"
        batch_dir.mkdir()

        store = AgentLabelStore(batch_dir, "test_agent")
        store.add_label("ref_1", "target_1", "match", 0.9, "")
        store.add_label("ref_2", "target_2", "no_match", 0.8, "")
        store.add_label("ref_3", "target_3", "unsure", 0.5, "")

        stats = store.get_stats()
        assert stats["total"] == 3
        assert stats["match"] == 1
        assert stats["no_match"] == 1
        assert stats["unsure"] == 1

    def test_list_agents(self, tmp_path):
        batch_dir = tmp_path / "test_batch"
        labels_dir = batch_dir / "labels"

        # Create agent directories with CSVs
        for agent in ["claude", "gpt4"]:
            agent_dir = labels_dir / agent
            agent_dir.mkdir(parents=True)
            (agent_dir / "data.csv").write_text("ref_id,target_id,label\nref_1,target_1,match\n")

        agents = AgentLabelStore.list_agents(batch_dir)
        assert sorted(agents) == ["claude", "gpt4"]

    def test_find_disagreements(self, tmp_path):
        batch_dir = tmp_path / "test_batch"
        labels_dir = batch_dir / "labels"

        # Claude says match, GPT4 says no_match
        claude_dir = labels_dir / "claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / "data.csv").write_text(
            "ref_id,target_id,label,confidence,agent_id\nref_1,target_1,match,0.9,claude\n"
        )

        gpt4_dir = labels_dir / "gpt4"
        gpt4_dir.mkdir(parents=True)
        (gpt4_dir / "data.csv").write_text(
            "ref_id,target_id,label,confidence,agent_id\nref_1,target_1,no_match,0.8,gpt4\n"
        )

        disagreements = AgentLabelStore.find_disagreements(batch_dir)
        assert len(disagreements) == 1
        assert disagreements.iloc[0]["ref_id"] == "ref_1"
        assert disagreements.iloc[0]["agreement_ratio"] == 0.5

    def test_compute_consensus(self, tmp_path):
        batch_dir = tmp_path / "test_batch"
        labels_dir = batch_dir / "labels"

        # 2 agents say match, 1 says no_match
        for agent, label in [("claude", "match"), ("gpt4", "match"), ("gemini", "no_match")]:
            agent_dir = labels_dir / agent
            agent_dir.mkdir(parents=True)
            (agent_dir / "data.csv").write_text(
                f"ref_id,target_id,label,confidence,agent_id\nref_1,target_1,{label},0.9,{agent}\n"
            )

        consensus = AgentLabelStore.compute_consensus(batch_dir, min_agents=2)
        assert len(consensus) == 1
        row = consensus.iloc[0]
        assert row["consensus_label"] == "match"
        assert row["agreement_ratio"] == pytest.approx(2 / 3)
        assert row["num_agents"] == 3


class TestDashedLinestring:
    """Tests for dashed line drawing."""

    def test_dashed_line_produces_image(self):
        """Dashed line on white image should produce non-white pixels."""
        from PIL import ImageDraw

        img = Image.new("RGB", (256, 256), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)
        line = LineString([(0, 0), (10, 0)])
        bbox = (-1, -1, 11, 1)
        _draw_dashed_linestring(draw, line, bbox, (256, 256), REFERENCE_COLOR, 2)

        pixels = np.array(img)
        non_white = np.any(pixels != 255, axis=2)
        assert non_white.sum() > 0, "Expected some dashed pixels drawn"

    def test_dashed_line_has_gaps(self):
        """A dashed line should have alternating colored/white regions."""
        from PIL import ImageDraw

        img = Image.new("RGB", (400, 50), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)
        line = LineString([(0, 0.5), (10, 0.5)])
        bbox = (0, 0, 10, 1)
        _draw_dashed_linestring(draw, line, bbox, (400, 50), REFERENCE_COLOR, 2, (20, 10))

        # Sample the middle row
        pixels = np.array(img)
        mid_row = pixels[25, :, :]
        is_colored = np.any(mid_row != 255, axis=1)

        # Should have at least one transition from colored to white
        transitions = np.diff(is_colored.astype(int))
        assert np.sum(transitions != 0) > 0, "Expected dash/gap transitions"

    def test_dashed_line_empty_geometry(self):
        """Empty geometry should not crash."""
        from PIL import ImageDraw

        img = Image.new("RGB", (100, 100), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)
        _draw_dashed_linestring(draw, None, (0, 0, 1, 1), (100, 100), REFERENCE_COLOR, 2)
        # Should not raise

    def test_dashed_line_empty_linestring(self):
        """Empty LineString should not crash."""
        from PIL import ImageDraw

        img = Image.new("RGB", (100, 100), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)
        line = LineString()
        _draw_dashed_linestring(draw, line, (0, 0, 1, 1), (100, 100), REFERENCE_COLOR, 2)
        # Should not raise


class TestSublineRendering:
    """Tests for subline rendering functions."""

    def test_subline_geometry_only_partial_alignment(self):
        """Partial alignment should render both faded and bright layers."""
        ref = LineString([(0, 0), (10, 0)])
        target = LineString([(0, 1), (10, 1)])

        img = render_subline_geometry_only(
            ref,
            target,
            ref_start_frac=0.0,
            ref_end_frac=0.5,
            target_start_frac=0.2,
            target_end_frac=1.0,
            size=(256, 256),
        )

        assert isinstance(img, Image.Image)
        assert img.size == (256, 256)

        # Should have non-white pixels (geometry drawn)
        pixels = np.array(img)
        non_white = np.any(pixels != 255, axis=2)
        assert non_white.sum() > 0

    def test_subline_geometry_only_full_alignment(self):
        """Full alignment fracs (0.0-1.0) should fall back to standard rendering."""
        ref = LineString([(0, 0), (10, 0)])
        target = LineString([(0, 1), (10, 1)])

        padding = 0.3
        subline_img = render_subline_geometry_only(
            ref,
            target,
            ref_start_frac=0.0,
            ref_end_frac=1.0,
            target_start_frac=0.0,
            target_end_frac=1.0,
            size=(256, 256),
            padding_ratio=padding,
        )
        standard_img = render_geometry_only(
            ref,
            target,
            size=(256, 256),
            padding_ratio=padding,
        )

        # Both should produce identical images (same padding → same bbox)
        subline_arr = np.array(subline_img)
        standard_arr = np.array(standard_img)
        assert np.array_equal(subline_arr, standard_arr)

    def test_subline_road_context_with_roads(self):
        """Road context variant should render gray context road pixels."""
        ref = LineString([(0, 0), (10, 0)])
        target = LineString([(0, 1), (10, 1)])
        ctx = [LineString([(0, -1), (10, -1)])]

        img = render_subline_road_context(
            ref,
            target,
            ref_start_frac=0.0,
            ref_end_frac=0.5,
            target_start_frac=0.2,
            target_end_frac=1.0,
            context_roads=ctx,
            size=(256, 256),
        )

        assert isinstance(img, Image.Image)
        pixels = np.array(img)

        # Check for gray pixels (context road color is (200, 200, 200))
        gray_mask = (
            (pixels[:, :, 0] == ROAD_CONTEXT_COLOR[0])
            & (pixels[:, :, 1] == ROAD_CONTEXT_COLOR[1])
            & (pixels[:, :, 2] == ROAD_CONTEXT_COLOR[2])
        )
        assert gray_mask.sum() > 0, "Expected gray context road pixels"

    def test_subline_road_context_no_roads(self):
        """No context roads should produce same output as geometry-only subline."""
        ref = LineString([(0, 0), (10, 0)])
        target = LineString([(0, 1), (10, 1)])

        img_ctx = render_subline_road_context(
            ref,
            target,
            ref_start_frac=0.0,
            ref_end_frac=0.5,
            target_start_frac=0.2,
            target_end_frac=1.0,
            context_roads=None,
            size=(256, 256),
        )
        img_geo = render_subline_geometry_only(
            ref,
            target,
            ref_start_frac=0.0,
            ref_end_frac=0.5,
            target_start_frac=0.2,
            target_end_frac=1.0,
            size=(256, 256),
        )

        # Without context roads, both should render identically
        assert np.array_equal(np.array(img_ctx), np.array(img_geo))

    def test_subline_degenerate_fracs(self):
        """start==end fracs should gracefully fall back to full segment rendering."""
        ref = LineString([(0, 0), (10, 0)])
        target = LineString([(0, 1), (10, 1)])

        # Degenerate: start == end
        img = render_subline_geometry_only(
            ref,
            target,
            ref_start_frac=0.5,
            ref_end_frac=0.5,
            target_start_frac=0.3,
            target_end_frac=0.3,
            size=(256, 256),
        )

        assert isinstance(img, Image.Image)
        assert img.size == (256, 256)
        # Should still render (falls back to full segment)
        pixels = np.array(img)
        non_white = np.any(pixels != 255, axis=2)
        assert non_white.sum() > 0


class TestRenderCandidateVariantSubline:
    """Tests for render_candidate_variant with subline basemaps."""

    def test_dispatch_subline_geometry_only(self):
        ref = LineString([(0, 0), (10, 0)])
        target = LineString([(0, 1), (10, 1)])

        result, meta = render_candidate_variant(
            ref,
            target,
            basemap="subline_geometry_only",
            alignment_fracs={
                "ref_start_frac": 0.0,
                "ref_end_frac": 0.5,
                "target_start_frac": 0.2,
                "target_end_frac": 1.0,
            },
        )

        assert isinstance(result, Image.Image)
        assert meta["basemap"] == "subline_geometry_only"
        assert meta["format"] == "png"

    def test_dispatch_subline_road_context(self):
        ref = LineString([(0, 0), (10, 0)])
        target = LineString([(0, 1), (10, 1)])
        ctx = [LineString([(0, -1), (10, -1)])]

        result, meta = render_candidate_variant(
            ref,
            target,
            basemap="subline_road_context",
            context_roads=ctx,
            alignment_fracs={
                "ref_start_frac": 0.1,
                "ref_end_frac": 0.9,
                "target_start_frac": 0.0,
                "target_end_frac": 1.0,
            },
        )

        assert isinstance(result, Image.Image)
        assert meta["basemap"] == "subline_road_context"

    def test_alignment_fracs_none_default(self):
        """alignment_fracs=None should not crash, treats as full alignment."""
        ref = LineString([(0, 0), (10, 0)])
        target = LineString([(0, 1), (10, 1)])

        result, meta = render_candidate_variant(
            ref,
            target,
            basemap="subline_geometry_only",
            alignment_fracs=None,
        )

        assert isinstance(result, Image.Image)
        assert meta["basemap"] == "subline_geometry_only"


class TestSampledCandidateAlignment:
    """Tests for alignment fraction fields on SampledCandidate."""

    def _make_candidate(self, **kwargs):
        defaults = dict(
            ref_id="ref_1",
            target_id="target_1",
            ref_geometry=LineString([(0, 0), (10, 0)]),
            target_geometry=LineString([(0, 1), (10, 1)]),
            ref_name=None,
            target_name=None,
            ref_class=None,
            target_class=None,
            ml_confidence=0.5,
            ml_decision="review",
            features={},
            dataset="test",
            confidence_bucket="medium",
        )
        defaults.update(kwargs)
        return SampledCandidate(**defaults)

    def test_default_fracs(self):
        c = self._make_candidate()
        assert c.ref_start_frac == 0.0
        assert c.ref_end_frac == 1.0
        assert c.target_start_frac == 0.0
        assert c.target_end_frac == 1.0

    def test_custom_fracs(self):
        c = self._make_candidate(
            ref_start_frac=0.1,
            ref_end_frac=0.9,
            target_start_frac=0.2,
            target_end_frac=0.8,
        )
        assert c.ref_start_frac == 0.1
        assert c.ref_end_frac == 0.9
        assert c.target_start_frac == 0.2
        assert c.target_end_frac == 0.8
