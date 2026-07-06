"""CLI package for the road network conflation pipeline."""

from dotenv import load_dotenv

load_dotenv()

import typer

from .agent import agent_app
from .analyze import analyze_app
from .blocking_recall import register_blocking_recall_commands
from .classify import class_app
from .data import data_app
from .factory import factory_app
from .main import register_commands

# Create the main app
app = typer.Typer(
    name="crosswalk",
    help="Road network conflation pipeline - link local roads to Overture GERS",
    no_args_is_help=True,
)

# Register top-level commands (match, train, eval, backfill, ui, version)
register_commands(app)
register_blocking_recall_commands(app)

# Add command groups
app.add_typer(data_app, name="data")
app.add_typer(analyze_app, name="analyze")
app.add_typer(class_app, name="class")
app.add_typer(agent_app, name="agent")
app.add_typer(factory_app, name="factory")


def matcher_deprecated() -> None:
    """Deprecated ``matcher`` console-script alias: warn, then forward to ``crosswalk``.

    The project was renamed ``matcher`` -> ``crosswalk`` (2026-07-05). This shim keeps
    the old entry point working while emitting a deprecation warning to stderr;
    it will be removed in a future release.
    """
    import sys

    print(
        "warning: 'matcher' has been renamed to 'crosswalk'. The 'matcher' alias is "
        "deprecated and will be removed in a future release; please use 'crosswalk'.",
        file=sys.stderr,
    )
    app()


__all__ = ["app", "matcher_deprecated"]
