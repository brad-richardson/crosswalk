"""CLI package for the road network conflation pipeline."""

from dotenv import load_dotenv

load_dotenv()

import typer

from .agent import agent_app
from .classify import class_app
from .data import data_app
from .integrate import integrate_app
from .labels import labels_app
from .main import register_commands
from .ml import ml_app
from .validate import validate_app

# Create the main app
app = typer.Typer(
    name="matcher",
    help="Road network conflation pipeline - link local roads to Overture GERS",
    no_args_is_help=True,
)

# Register top-level commands (match, train, label, match-eval, screen, version)
register_commands(app)

# Add command groups
app.add_typer(data_app, name="data")
app.add_typer(ml_app, name="ml")
app.add_typer(integrate_app, name="integrate")
app.add_typer(class_app, name="class")
app.add_typer(agent_app, name="agent")
app.add_typer(validate_app, name="validate")
app.add_typer(labels_app, name="labels")

__all__ = ["app"]
