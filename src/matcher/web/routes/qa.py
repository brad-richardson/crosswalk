"""Integration QA routes for the matcher web UI."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.templating import Jinja2Templates
from shapely.geometry import mapping

from ...filenames import integration_cache_dir
from ..services import (
    list_datasets,
    load_qa_edges,
    record_qa_decision,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/qa")
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

# Color mapping for edge layers
LAYER_COLORS = {
    "reference": "#2196F3",  # blue
    "non_reference": "#4caf50",  # green
    "net_new": "#00bcd4",  # cyan
    "disconnected": "#f44336",  # red
    "filtered": "#ff9800",  # orange
    "bridge": "#9c27b0",  # purple
}


def _edges_to_geojson(edge_data: dict) -> str:
    """Convert edge GeoDataFrames into a single GeoJSON FeatureCollection.

    Each Feature has properties: layer (type name), color, edge_id.

    Reference edges (from `edges` GDF where _source=="reference") use blue.
    Non-reference edges from `edges` GDF use green.
    Other GDFs get their own colors.

    Args:
        edge_data: Dict of name -> GeoDataFrame from load_qa_edges

    Returns:
        JSON string of a GeoJSON FeatureCollection.
    """
    features = []

    # Process the main edges GeoDataFrame
    edges_gdf = edge_data.get("edges")
    if edges_gdf is not None and len(edges_gdf) > 0:
        for _, row in edges_gdf.iterrows():
            source = row.get("_source", row.get("source", ""))
            if source == "reference":
                layer = "reference"
                color = LAYER_COLORS["reference"]
            else:
                layer = "non_reference"
                color = LAYER_COLORS["non_reference"]

            feature = {
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": {
                    "layer": layer,
                    "color": color,
                    "edge_id": int(row.get("edge_id", 0)),
                    "original_id": str(row.get("_original_id", row.get("original_id", ""))),
                    "road_class": str(row.get("road_class", "")),
                    "length_m": float(row.get("length_m", 0)),
                    "_source": str(source),
                },
            }
            features.append(feature)

    # Process other edge files (filenames match integration output.py)
    other_files = {
        "net_new": ("net_new", LAYER_COLORS["net_new"]),
        "disconnected": ("disconnected", LAYER_COLORS["disconnected"]),
        "filtered": ("filtered", LAYER_COLORS["filtered"]),
        "bridges": ("bridge", LAYER_COLORS["bridge"]),
    }

    for file_key, (layer_name, color) in other_files.items():
        gdf = edge_data.get(file_key)
        if gdf is not None and len(gdf) > 0:
            for _, row in gdf.iterrows():
                feature = {
                    "type": "Feature",
                    "geometry": mapping(row.geometry),
                    "properties": {
                        "layer": layer_name,
                        "color": color,
                        "edge_id": int(row.get("edge_id", 0)),
                        "original_id": str(row.get("_original_id", row.get("original_id", ""))),
                        "road_class": str(row.get("road_class", "")),
                        "length_m": float(row.get("length_m", 0)),
                    },
                }
                features.append(feature)

    return json.dumps({"type": "FeatureCollection", "features": features})


@router.get("")
async def qa_page(
    request: Request,
    dataset: str | None = None,
    type: str = "merged",
):
    """Render the QA page.

    Args:
        request: FastAPI request
        dataset: Optional dataset ID to load
        type: Edge type to review ("merged" or "orphan")
    """
    datasets = list_datasets()

    if not dataset:
        context = {
            "mode": "qa",
            "datasets": datasets,
            "dataset": None,
            "edge_type": type,
            "edge_geojson": "{}",
        }
        return templates.TemplateResponse(request, "qa/page.html", context)

    # Load edges
    try:
        edge_data = load_qa_edges(dataset)
        edge_geojson = _edges_to_geojson(edge_data)
    except Exception:
        logger.exception("Failed to load QA edges for dataset %s", dataset)
        edge_geojson = json.dumps({"type": "FeatureCollection", "features": []})

    context = {
        "mode": "qa",
        "datasets": datasets,
        "dataset": dataset,
        "edge_type": type,
        "edge_geojson": edge_geojson,
    }
    return templates.TemplateResponse(request, "qa/page.html", context)


@router.get("/edge/{edge_id}")
async def edge_detail(
    request: Request,
    edge_id: int,
    dataset: str = "",
    type: str = "merged",
):
    """Return edge detail fragment.

    Args:
        request: FastAPI request
        edge_id: Edge ID to look up
        dataset: Dataset ID
        type: Edge type ("merged" or "orphan")
    """
    edge = None

    if dataset:
        try:
            edge_data = load_qa_edges(dataset)
            # Search all GeoDataFrames for the edge
            for gdf_name, gdf in edge_data.items():
                if gdf is not None and len(gdf) > 0 and "edge_id" in gdf.columns:
                    matches = gdf[gdf["edge_id"] == edge_id]
                    if len(matches) > 0:
                        row = matches.iloc[0]
                        source = str(row.get("_source", row.get("source", "")))

                        # Derive layer label consistently
                        if gdf_name == "edges":
                            layer = "reference" if source == "reference" else "non_reference"
                        else:
                            # Use the file key directly (net_new, disconnected, etc.)
                            layer = gdf_name

                        edge = {
                            "edge_id": int(row.get("edge_id", 0)),
                            "original_id": str(
                                row.get(
                                    "_original_id",
                                    row.get("original_id", ""),
                                )
                            ),
                            "road_class": str(row.get("road_class", "")),
                            "length_m": float(row.get("length_m", 0)),
                            "_source": source,
                            "layer": layer,
                        }
                        break
        except Exception:
            logger.exception("Failed to load edge %d for dataset %s", edge_id, dataset)

    context = {
        "edge": edge,
        "dataset": dataset,
        "edge_type": type,
    }
    return templates.TemplateResponse(request, "qa/edge.html", context)


@router.post("/decision")
async def record_decision(
    request: Request,
    edge_id: int = Form(...),
    original_id: str = Form(""),
    dataset: str = Form(...),
    edge_type: str = Form("merged"),
    decision: str = Form(...),
    reason: str = Form(""),
    note: str = Form(""),
):
    """Record a QA accept/reject decision.

    Args:
        request: FastAPI request
        edge_id: Edge ID
        original_id: Original edge ID
        dataset: Dataset ID
        edge_type: "orphan" or "merged"
        decision: "correct" or "incorrect"
        reason: Reason for the decision
        note: Optional reviewer note
    """
    try:
        record_qa_decision(
            edge_id=edge_id,
            original_id=original_id,
            dataset_id=dataset,
            edge_type=edge_type,
            decision=decision,
            reason=reason,
            note=note,
        )
        message = f"Recorded: edge {edge_id} = {decision}"
        success = True
    except Exception:
        logger.exception("Failed to record decision for edge %d", edge_id)
        message = f"Error recording decision for edge {edge_id}"
        success = False

    context = {
        "message": message,
        "success": success,
        "edge_id": edge_id,
        "dataset": dataset,
        "edge_type": edge_type,
    }
    return templates.TemplateResponse(request, "qa/decision_result.html", context)


@router.get("/pipeline/{dataset}")
async def pipeline_status(
    request: Request,
    dataset: str,
):
    """Check pipeline status for a dataset.

    Args:
        request: FastAPI request
        dataset: Dataset ID
    """
    cache_dir = integration_cache_dir(dataset)
    edges_path = cache_dir / "edges.parquet"

    status = "ready" if edges_path.exists() else "not_run"

    context = {
        "dataset": dataset,
        "status": status,
    }
    return templates.TemplateResponse(request, "qa/pipeline.html", context)
