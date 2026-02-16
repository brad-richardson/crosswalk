"""Shared Jinja2 templates instance with cache-busting support."""

import time
from pathlib import Path

from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).parent / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals["cache_bust"] = str(int(time.time()))
