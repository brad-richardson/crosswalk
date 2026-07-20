"""Pairwise stitch-review helpers shared by queue generation and the web UI.

Stitch groups persist geometries for their selected member ids, while the exact
identity candidate universe also includes rejected edges incident to those
members.  One endpoint of a rejected edge may therefore live outside the group
and have no geometry in the normal ``ref_geometries`` / ``target_geometries``
maps.  The pairwise reviewer needs both endpoints, so this module enriches a
queue group with the missing raw-data geometries and basic attributes.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from shapely.geometry import mapping

from ..config import CLASS_COLUMN, NAMES_COLUMN
from ..filenames import PROJECT_ROOT, find_overture_segments, find_target_file


def candidate_edge_union(group: dict) -> list[dict]:
    """Return selected + rejected candidate edges, deduplicated stably."""
    seen: set[tuple[str, str]] = set()
    result: list[dict] = []
    for edge in (group.get("edges") or []) + (group.get("rejected_edges") or []):
        key = (str(edge.get("ref_id", "")), str(edge.get("target_id", "")))
        if not all(key) or key in seen:
            continue
        seen.add(key)
        result.append(edge)
    return result


def _round_geojson(geojson: dict, precision: int = 6) -> dict:
    """Round GeoJSON coordinates without depending on the web package."""

    def _round_coords(coords):
        if not coords:
            return coords
        if isinstance(coords[0], (int, float)):
            return [round(value, precision) for value in coords]
        return [_round_coords(value) for value in coords]

    if geojson.get("type") == "GeometryCollection":
        return {
            **geojson,
            "geometries": [
                _round_geojson(geometry, precision) for geometry in geojson.get("geometries", [])
            ],
        }
    if "coordinates" not in geojson:
        return geojson
    return {**geojson, "coordinates": _round_coords(geojson["coordinates"])}


def _known_ids(group: dict, side: str) -> set[str]:
    result: set[str] = set()
    for key in (
        f"{side}_geometries",
        f"context_{side}_geometries",
        f"candidate_{side}_geometries",
    ):
        result.update(str(value) for value in (group.get(key) or {}))
    return result


def missing_candidate_endpoint_ids(groups: Iterable[dict]) -> tuple[set[str], set[str]]:
    """Return candidate endpoint ids whose geometry is absent from the groups."""
    missing_ref: set[str] = set()
    missing_target: set[str] = set()
    for group in groups:
        known_ref = _known_ids(group, "ref")
        known_target = _known_ids(group, "target")
        for edge in candidate_edge_union(group):
            ref_id = str(edge["ref_id"])
            target_id = str(edge["target_id"])
            if ref_id not in known_ref:
                missing_ref.add(ref_id)
            if target_id not in known_target:
                missing_target.add(target_id)
    return missing_ref, missing_target


def _display_name(value) -> str:
    from ..pipeline.runner import _extract_name_string, _is_nan

    return "" if _is_nan(value) else _extract_name_string(value)


def _display_class(value) -> str:
    from ..pipeline.runner import _is_nan

    return "" if _is_nan(value) else str(value)


def _attach_rows(groups: Iterable[dict], rows, side: str) -> int:
    """Attach filtered raw rows to every group that references their ids."""
    by_id = {}
    for _, row in rows.iterrows():
        segment_id = str(row.get("id", ""))
        geometry = row.geometry
        if not segment_id or geometry is None or geometry.is_empty:
            continue
        by_id[segment_id] = {
            "geometry": _round_geojson(mapping(geometry)),
            "name": _display_name(row.get(NAMES_COLUMN)),
            "class": _display_class(row.get(CLASS_COLUMN)),
        }

    attached_ids: set[str] = set()
    id_field = "ref_id" if side == "ref" else "target_id"
    for group in groups:
        needed = {str(edge[id_field]) for edge in candidate_edge_union(group)}
        known = _known_ids(group, side)
        geometry_map = group.setdefault(f"candidate_{side}_geometries", {})
        name_map = group.setdefault(f"candidate_{side}_names", {})
        class_map = group.setdefault(f"candidate_{side}_classes", {})
        for segment_id in sorted(needed - known):
            item = by_id.get(segment_id)
            if item is None:
                continue
            geometry_map[segment_id] = item["geometry"]
            name_map[segment_id] = item["name"]
            class_map[segment_id] = item["class"]
            attached_ids.add(segment_id)
    return len(attached_ids)


def enrich_candidate_endpoints(
    groups: list[dict],
    dataset_id: str,
    *,
    data_dir: Path | None = None,
) -> dict[str, int]:
    """Attach missing exact-candidate endpoint data to ``groups`` in place.

    Reads only the requested ids through Parquet filters.  Missing raw files or
    ids are reported in the returned counters and never make the review queue
    unreadable; the UI can surface a geometry-unavailable warning for a pair.
    """
    import geopandas as gpd

    groups = list(groups)
    missing_ref, missing_target = missing_candidate_endpoint_ids(groups)
    stats = {
        "requested_ref": len(missing_ref),
        "requested_target": len(missing_target),
        "attached_ref": 0,
        "attached_target": 0,
    }
    if not missing_ref and not missing_target:
        return stats

    root = Path(data_dir) if data_dir is not None else PROJECT_ROOT / "data" / "raw"
    ref_path = find_overture_segments(root, dataset_id)
    target_path = find_target_file(root, dataset_id)
    columns = ["id", "geometry", NAMES_COLUMN, CLASS_COLUMN]

    if missing_ref and ref_path:
        rows = gpd.read_parquet(
            ref_path,
            columns=columns,
            filters=[("id", "in", sorted(missing_ref))],
        )
        stats["attached_ref"] = _attach_rows(groups, rows, "ref")
    if missing_target and target_path:
        rows = gpd.read_parquet(
            target_path,
            columns=columns,
            filters=[("id", "in", sorted(missing_target))],
        )
        stats["attached_target"] = _attach_rows(groups, rows, "target")
    return stats
