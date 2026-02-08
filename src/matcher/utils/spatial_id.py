"""Stable spatial suffix for target segment IDs.

Uses H3 hexagonal grid to produce a suffix that:
- Disambiguates segments with the same upstream ID in different locations
- Is stable against minor GPS corrections (within ~1km H3 cell)
- Is deterministic and decodable (spatially meaningful)

The suffix is computed from the midpoint of the line (via interpolation),
which is a pure arithmetic operation independent of GEOS version or
simplification algorithms.
"""

import h3
import shapely

# H3 resolution 8 produces ~1km-wide hexagons (531m edge length).
# Stable against typical GPS corrections (<100m) while still
# disambiguating segments that truly moved (>500m).
H3_RESOLUTION = 8

# At resolution 8, the last 5 hex chars of the H3 index are always "fffff"
# (unused child digits for resolutions 9-15, filled with 7 = binary 111).
# We strip them to keep IDs shorter. To restore a full H3 index, pad with "f"
# to 15 chars: suffix.ljust(15, "f")
H3_TRAILING_F_COUNT = 5


def compute_spatial_suffix(geom, resolution: int = H3_RESOLUTION) -> str:
    """Compute a stable spatial suffix for ID disambiguation.

    Steps:
    1. Compute midpoint along the line via interpolation (not centroid —
       centroid can fall outside the geometry for curved/horseshoe roads)
    2. Get H3 cell at the given resolution
    3. Return the H3 index with trailing "fffff" stripped (10 chars for res 8)

    To restore the full H3 index for spatial lookups:
        full_h3 = suffix.ljust(15, "f")

    No geometry simplification is performed — line_interpolate_point is a
    pure arithmetic operation that produces identical results regardless
    of GEOS version or platform.

    Args:
        geom: Shapely geometry (must be in WGS84/EPSG:4326, LineString)
        resolution: H3 resolution (default 8, ~1km hexagons)

    Returns:
        Trimmed H3 index string (10 chars for res 8), e.g. "882a306603"
    """
    midpoint = shapely.line_interpolate_point(geom, 0.5, normalized=True)
    h3_index = h3.latlng_to_cell(midpoint.y, midpoint.x, resolution)
    return h3_index[:-H3_TRAILING_F_COUNT]
