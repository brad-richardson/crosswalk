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
