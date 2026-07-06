"""CI validation for per-dataset JSON recipe files (``datasets/*.json``).

Contribution-time gate for the OA-style dataset registry format proposed in
``docs/plans/2026-07-06-dataset-registry-format.md``: every ``datasets/<name>.json``
must validate against ``datasets/schema/dataset.schema.json``, and its ``name``
field must equal the filename stem (the dataset key used across ``labels/``,
``bridges/dataset=<name>/`` and mbench).

Phase 0: the JSON files are inert examples — no consumer reads them yet — but
this test already gives contributors a fast, local, deterministic check
(``uv run pytest tests/unit/test_dataset_registry_schema.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from crosswalk.datasets.schema import get_datasets_dir

SCHEMA_PATH = get_datasets_dir() / "schema" / "dataset.schema.json"
RECIPE_PATHS = sorted(get_datasets_dir().glob("*.json"))


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def test_schema_itself_is_valid_draft_2020_12() -> None:
    schema = _load(SCHEMA_PATH)
    jsonschema.Draft202012Validator.check_schema(schema)


def test_examples_exist() -> None:
    # Phase 0 ships two converted examples; if this fails the sample recipes moved.
    assert RECIPE_PATHS, f"no datasets/*.json recipe files found in {get_datasets_dir()}"


@pytest.mark.parametrize("recipe_path", RECIPE_PATHS, ids=lambda p: p.stem)
def test_recipe_validates_against_schema(recipe_path: Path) -> None:
    schema = _load(SCHEMA_PATH)
    recipe = _load(recipe_path)
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(recipe), key=lambda e: list(e.absolute_path))
    assert not errors, "\n".join(
        f"{recipe_path.name} :: {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in errors
    )


@pytest.mark.parametrize("recipe_path", RECIPE_PATHS, ids=lambda p: p.stem)
def test_recipe_name_matches_filename(recipe_path: Path) -> None:
    recipe = _load(recipe_path)
    assert recipe["name"] == recipe_path.stem, (
        f"{recipe_path.name}: 'name' field ({recipe['name']!r}) must equal the "
        f"filename stem ({recipe_path.stem!r}) — it is the registry key"
    )


@pytest.mark.parametrize("recipe_path", RECIPE_PATHS, ids=lambda p: p.stem)
def test_approved_recipes_carry_attribution(recipe_path: Path) -> None:
    """Mirror of the default-deny publisher rule (factory/licenses.py): approved
    without attribution is an invalid state, caught at PR time instead of at
    publish time."""
    recipe = _load(recipe_path)
    lic = recipe["license"]
    if lic["status"] == "approved":
        assert lic.get("attribution"), f"{recipe_path.name}: approved but no attribution"
        assert lic.get("url"), f"{recipe_path.name}: approved but no license url"
