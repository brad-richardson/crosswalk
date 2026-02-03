"""Building footprint screen test.

Detects target segments that pass through or too close to building footprints.
"""

from ..base import register_test
from ..constants import BUILDING_BUFFER_M
from ..context import fetch_overture_buildings
from .polygon_buffer_test import PolygonBufferTest


@register_test
class BuildingTest(PolygonBufferTest):
    """Test if a candidate segment passes through buildings.

    Buffers building footprints based on road type - vehicle roads need
    more clearance than pedestrian paths.
    """

    name = "building"
    feature_type = "building"
    default_buffers = BUILDING_BUFFER_M
    fetch_func = staticmethod(fetch_overture_buildings)
