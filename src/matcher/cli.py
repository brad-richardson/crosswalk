"""CLI entry point for the road network conflation pipeline.

This module re-exports the app from the cli package for backwards compatibility.
All CLI implementation is in the cli/ package.
"""

from matcher.cli import app

if __name__ == "__main__":
    app()
