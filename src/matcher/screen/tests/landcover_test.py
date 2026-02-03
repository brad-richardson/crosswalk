"""Landcover screen test.

Detects target segments that pass through restricted landcover types
like wetlands or sports fields.
"""

from ..base import register_test
from ..constants import LANDCOVER_BUFFER_M
from ..context import fetch_overture_landcover
from .polygon_buffer_test import PolygonBufferTest


@register_test
class LandcoverTest(PolygonBufferTest):
    """Test if a candidate segment passes through restricted landcover.

    Checks for:
    - Wetlands (marsh, swamp, bog) - roads shouldn't pass through
    - Sports fields (pitch, track, stadium) - no roads on playing surfaces
    """

    name = "landcover"
    feature_type = "restricted landcover"
    default_buffers = LANDCOVER_BUFFER_M
    fetch_func = staticmethod(fetch_overture_landcover)
