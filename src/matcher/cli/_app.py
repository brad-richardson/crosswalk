"""Shared CLI app and console instances.

All CLI submodules import from here to avoid circular imports.
"""

import typer
from rich.console import Console

app = typer.Typer(
    name="matcher",
    help="Road network conflation pipeline - link local roads to Overture GERS",
    no_args_is_help=True,
)
console = Console()

# Create fetch subcommand group
fetch_app = typer.Typer(
    name="fetch",
    help="Fetch road data from various sources",
    no_args_is_help=True,
)
app.add_typer(fetch_app, name="fetch")
