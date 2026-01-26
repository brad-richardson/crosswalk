"""CLI entry point for the road network conflation pipeline.

This package splits CLI commands into logical modules:
- fetch.py: Data fetching commands (fetch target, reference, overture, osm, etc.)
- pipeline.py: Core pipeline commands (topology, match, eval-bridge)
- ml.py: ML commands (train, eval-model, compute-features, benchmark)
- labeling.py: Labeling commands (label UI, agent batches, backfill)
- integration.py: Integration commands (integrate, qa-integration)
- data.py: Data/utility commands (validate-matching, validate-data, discover-classes, version)
"""

# Import app from shared module
# Import all submodules to register their commands on the app.
# The order doesn't matter - each module decorates `app` or `fetch_app` at import time.
from . import data, fetch, integration, labeling, ml, pipeline  # noqa: F401
from ._app import app

__all__ = ["app"]

if __name__ == "__main__":
    app()
