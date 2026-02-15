"""CLI package for the road network conflation pipeline."""

from dotenv import load_dotenv

load_dotenv()

import typer

from .agent import agent_app
from .analyze import analyze_app
from .classify import class_app
from .data import data_app
from .main import register_commands

# Create the main app
app = typer.Typer(
    name="matcher",
    help="Road network conflation pipeline - link local roads to Overture GERS",
    no_args_is_help=True,
)

# Register top-level commands (match, train, eval, backfill, ui, version)
register_commands(app)

# Add command groups
app.add_typer(data_app, name="data")
app.add_typer(analyze_app, name="analyze")
app.add_typer(class_app, name="class")
app.add_typer(agent_app, name="agent")

__all__ = ["app"]
