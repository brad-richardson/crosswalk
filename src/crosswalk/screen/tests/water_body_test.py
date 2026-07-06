"""Water body screen test.

Detects target segments that intersect water bodies (lakes, rivers, etc.)
beyond what could reasonably be a bridge.
"""

from ..base import register_test
from ..constants import WATER_BUFFER_M
from ..context import fetch_overture_water
from .polygon_buffer_test import PolygonBufferTest


@register_test
class WaterBodyTest(PolygonBufferTest):
    """Test if a candidate segment intersects water bodies.

    Buffers water bodies based on road type - vehicle roads need more
    clearance than pedestrian paths.
    """

    name = "water_body"
    feature_type = "water body"
    default_buffers = WATER_BUFFER_M
    fetch_func = staticmethod(fetch_overture_water)
