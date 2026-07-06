"""Tool adapters for mbench."""

from __future__ import annotations

from mbench.adapters.crosswalk import CrosswalkAdapter

REGISTRY: dict[str, type] = {
    "crosswalk": CrosswalkAdapter,
}

# Deprecated engine-id aliases: resolved (with a stderr warning) by
# ``mbench.cli._get_adapter``. The tool was renamed matcher -> crosswalk (2026-07-05).
DEPRECATED_ALIASES: dict[str, str] = {
    "matcher": "crosswalk",
}

# Hootenanny adapter requires geopandas (optional dep)
try:
    from mbench.adapters.hootenanny import HootAdapter

    REGISTRY["hootenanny"] = HootAdapter
except ImportError:
    pass

# Naive geometric baseline also requires geopandas (optional dep)
try:
    from mbench.adapters.naive import NaiveAdapter

    REGISTRY["naive"] = NaiveAdapter
except ImportError:
    pass

# Valhalla Meili map-matching baseline requires geopandas + pyosmium (optional).
try:
    from mbench.adapters.meili import MeiliAdapter

    REGISTRY["meili"] = MeiliAdapter
except ImportError:
    pass

# GraphHopper map-matching baseline requires geopandas + pyosmium for the shared
# Overture->PBF conversion (same optional deps as Meili). The engine itself is a
# JVM library run via `jbang` at match time (checked at run() time, not import).
try:
    from mbench.adapters.graphhopper import GraphHopperAdapter

    REGISTRY["graphhopper"] = GraphHopperAdapter
except ImportError:
    pass
