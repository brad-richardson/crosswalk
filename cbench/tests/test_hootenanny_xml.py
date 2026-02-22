"""Tests for Hootenanny XML match extraction (no geopandas needed)."""

from pathlib import Path

from cbench.adapters.hootenanny import (
    _build_id_map,
    _get_all_matcher_ids,
    extract_matches_from_conflated,
)

# Minimal OSM XML fixtures


def _write_osm(path: Path, content: str) -> Path:
    path.write_text(f'<?xml version="1.0" encoding="UTF-8"?>\n{content}')
    return path


def test_build_id_map_new_format(tmp_path):
    osm = _write_osm(
        tmp_path / "ref.osm",
        """<osm version="0.6">
        <way id="-1" version="1">
            <nd ref="-10"/>
            <tag k="matcher_ref_abc123" v="original-id-1"/>
        </way>
        <way id="-2" version="1">
            <nd ref="-11"/>
            <tag k="matcher_ref_def456" v="original-id-2"/>
        </way>
        </osm>""",
    )
    id_map = _build_id_map(osm, source_tag="ref")
    assert id_map == {"-1": "original-id-1", "-2": "original-id-2"}


def test_get_all_matcher_ids(tmp_path):
    osm = _write_osm(
        tmp_path / "tgt.osm",
        """<osm version="0.6">
        <way id="-1" version="1">
            <tag k="matcher_tgt_aaa" v="id-a"/>
        </way>
        <way id="-2" version="1">
            <tag k="matcher_tgt_bbb" v="id-b"/>
        </way>
        </osm>""",
    )
    ids = _get_all_matcher_ids(osm, source_tag="tgt")
    assert ids == {"id-a", "id-b"}


def test_extract_matches_from_merged_ways(tmp_path):
    """Merged way has both ref and tgt matcher tags -> match pair."""
    ref_osm = _write_osm(
        tmp_path / "ref.osm",
        """<osm version="0.6">
        <way id="-1" version="1">
            <tag k="matcher_ref_r1" v="ref-001"/>
        </way>
        </osm>""",
    )
    tgt_osm = _write_osm(
        tmp_path / "tgt.osm",
        """<osm version="0.6">
        <way id="-2" version="1">
            <tag k="matcher_tgt_t1" v="tgt-001"/>
        </way>
        </osm>""",
    )
    conflated = _write_osm(
        tmp_path / "conflated.osm",
        """<osm version="0.6">
        <way id="-3" version="1">
            <tag k="matcher_ref_r1" v="ref-001"/>
            <tag k="matcher_tgt_t1" v="tgt-001"/>
            <tag k="highway" v="residential"/>
        </way>
        </osm>""",
    )
    matches = extract_matches_from_conflated(conflated, ref_osm, tgt_osm)
    assert len(matches) == 1
    assert matches[0] == ("ref-001", "tgt-001", "merge")


def test_extract_matches_from_review_relation(tmp_path):
    """Review relation linking ref and tgt members -> match pair."""
    ref_osm = _write_osm(
        tmp_path / "ref.osm",
        """<osm version="0.6">
        <way id="-1" version="1">
            <tag k="matcher_ref_r1" v="ref-001"/>
        </way>
        </osm>""",
    )
    tgt_osm = _write_osm(
        tmp_path / "tgt.osm",
        """<osm version="0.6">
        <way id="-2" version="1">
            <tag k="matcher_tgt_t1" v="tgt-001"/>
        </way>
        </osm>""",
    )
    conflated = _write_osm(
        tmp_path / "conflated.osm",
        """<osm version="0.6">
        <way id="-10" version="1">
            <tag k="matcher_ref_r1" v="ref-001"/>
            <tag k="hoot:status" v="1"/>
        </way>
        <way id="-11" version="1">
            <tag k="matcher_tgt_t1" v="tgt-001"/>
            <tag k="hoot:status" v="2"/>
        </way>
        <relation id="-100" version="1">
            <member type="way" ref="-10" role=""/>
            <member type="way" ref="-11" role=""/>
            <tag k="type" v="review"/>
        </relation>
        </osm>""",
    )
    matches = extract_matches_from_conflated(conflated, ref_osm, tgt_osm)
    assert len(matches) == 1
    assert matches[0] == ("ref-001", "tgt-001", "review")


def test_extract_no_matches(tmp_path):
    """No matches when ref and tgt are separate in output."""
    ref_osm = _write_osm(
        tmp_path / "ref.osm",
        """<osm version="0.6">
        <way id="-1" version="1">
            <tag k="matcher_ref_r1" v="ref-001"/>
        </way>
        </osm>""",
    )
    tgt_osm = _write_osm(
        tmp_path / "tgt.osm",
        """<osm version="0.6">
        <way id="-2" version="1">
            <tag k="matcher_tgt_t1" v="tgt-001"/>
        </way>
        </osm>""",
    )
    conflated = _write_osm(
        tmp_path / "conflated.osm",
        """<osm version="0.6">
        <way id="-1" version="1">
            <tag k="matcher_ref_r1" v="ref-001"/>
        </way>
        <way id="-2" version="1">
            <tag k="matcher_tgt_t1" v="tgt-001"/>
        </way>
        </osm>""",
    )
    matches = extract_matches_from_conflated(conflated, ref_osm, tgt_osm)
    assert len(matches) == 0
