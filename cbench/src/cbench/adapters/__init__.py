"""Tool adapters for cbench."""

from __future__ import annotations

from cbench.adapters.matcher import MatcherAdapter

REGISTRY: dict[str, type] = {
    "matcher": MatcherAdapter,
}

# Hootenanny adapter requires geopandas (optional dep)
try:
    from cbench.adapters.hootenanny import HootAdapter

    REGISTRY["hootenanny"] = HootAdapter
except ImportError:
    pass

# Naive geometric baseline also requires geopandas (optional dep)
try:
    from cbench.adapters.naive import NaiveAdapter

    REGISTRY["naive"] = NaiveAdapter
except ImportError:
    pass

# Valhalla Meili map-matching baseline requires geopandas + pyosmium (optional).
try:
    from cbench.adapters.meili import MeiliAdapter

    REGISTRY["meili"] = MeiliAdapter
except ImportError:
    pass
