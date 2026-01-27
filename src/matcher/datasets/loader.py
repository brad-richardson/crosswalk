"""Unified dataset loading utility.

Centralizes post-fetch dataset loading: file discovery, parquet reading,
CRS normalization, geometry filtering, column validation, and session caching.

Example usage:
    from matcher.datasets import DatasetLoader

    loader = DatasetLoader()
    pair = loader.load_pair("us_boston_streets", project=True)
    print(f"Reference: {len(pair.reference)} segments")
    print(f"Target: {len(pair.target)} segments")

    # With session caching for repeated loads:
    with loader.session():
        pair1 = loader.load_pair("us_boston_streets")
        pair2 = loader.load_pair("us_boston_streets")  # served from cache
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import geopandas as gpd
from loguru import logger

from ..filenames import (
    extract_version_from_filename,
    find_osm_segments,
    find_overture_segments,
    find_target_file,
)
from ..utils.crs import ensure_projected_crs
from ..utils.geometry import filter_to_linestrings

# Columns that are required for any loaded GeoDataFrame
_REQUIRED_COLUMNS = {"id", "geometry"}

# Columns that are expected but optional — a warning is logged when missing
_EXPECTED_COLUMNS = {"names", "class", "subclass", "connectors"}


class LoadedPair(NamedTuple):
    """A loaded reference + target dataset pair."""

    reference: gpd.GeoDataFrame
    """Reference GeoDataFrame (e.g. Overture segments)."""

    target: gpd.GeoDataFrame
    """Target GeoDataFrame (e.g. local road data or OSM segments)."""

    dataset_id: str
    """Identifier for the dataset pair."""

    reference_path: Path
    """Path to the reference file on disk."""

    target_path: Path
    """Path to the target file on disk."""


class DatasetLoader:
    """Centralized loader for reference/target dataset pairs.

    Handles file discovery, parquet reading, CRS normalization,
    geometry filtering, column validation, and optional session caching.

    Args:
        data_dir: Directory containing raw data files.
            Defaults to ``data/raw`` relative to the project root.
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        if data_dir is None:
            # Default: project root / data / raw
            # Project root is 3 levels up from this file (datasets/loader.py)
            project_root = Path(__file__).parents[3]
            data_dir = project_root / "data" / "raw"
        self._data_dir = Path(data_dir)
        self._cache: dict[Path, gpd.GeoDataFrame] | None = None

    @property
    def data_dir(self) -> Path:
        """The data directory used for file discovery."""
        return self._data_dir

    # ------------------------------------------------------------------
    # Session caching
    # ------------------------------------------------------------------

    @contextmanager
    def session(self):
        """Context manager that enables in-memory caching of loaded GeoDataFrames.

        Within a session, repeated loads of the same file path return the
        cached GeoDataFrame without re-reading from disk.

        Example::

            with loader.session():
                pair1 = loader.load_pair("us_boston_streets")
                pair2 = loader.load_pair("us_boston_streets")  # cache hit
        """
        self._cache = {}
        try:
            yield self
        finally:
            self._cache = None

    def clear_cache(self) -> None:
        """Clear the session cache (if active)."""
        if self._cache is not None:
            self._cache.clear()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_available(self) -> list[str]:
        """List dataset IDs that have both reference and target files on disk.

        Combines datasets discovered from YAML configs (via
        ``list_dataset_configs()``) with auto-discovered OSM datasets
        (files matching ``*_osm_segments_*.parquet``).

        Returns:
            Sorted list of dataset identifiers.
        """
        available: list[str] = []

        # 1. Datasets from YAML configs (use schema module — same source as CLI/UI)
        from .schema import list_dataset_configs as _list_yaml

        for name in _list_yaml():
            ref = find_overture_segments(self._data_dir, name)
            tgt = find_target_file(self._data_dir, name)
            if ref and tgt:
                available.append(name)

        # 2. Auto-discovered OSM datasets
        if self._data_dir.exists():
            for osm_file in self._data_dir.glob("*_osm_segments*.parquet"):
                version = extract_version_from_filename(osm_file)
                if version is None:
                    continue
                base_name = osm_file.stem.rsplit("_v", 1)[0]
                if not base_name.endswith("_osm_segments"):
                    continue
                region = base_name.replace("_osm_segments", "")
                dataset_id = f"{region}_osm"
                if dataset_id in available:
                    continue
                ref = find_overture_segments(self._data_dir, region)
                if ref:
                    available.append(dataset_id)

        return sorted(set(available))

    def find_reference_path(self, dataset_id: str) -> Path | None:
        """Find the reference (Overture) file path for a dataset.

        For OSM variant datasets (suffix ``_osm``), the base region is used
        to locate the Overture reference.

        Returns:
            Resolved ``Path``, or ``None`` if no file is found.
        """
        base = self._strip_osm_suffix(dataset_id)
        return find_overture_segments(self._data_dir, base)

    def find_target_path(self, dataset_id: str) -> Path | None:
        """Find the target file path for a dataset.

        For OSM variant datasets (suffix ``_osm``), looks for the
        corresponding OSM segments file using the base region.

        Returns:
            Resolved ``Path``, or ``None`` if no file is found.
        """
        if dataset_id.endswith("_osm"):
            base = dataset_id[: -len("_osm")]
            return find_osm_segments(self._data_dir, base)
        return find_target_file(self._data_dir, dataset_id)

    # ------------------------------------------------------------------
    # Core loading
    # ------------------------------------------------------------------

    def load_reference(self, dataset_id: str) -> gpd.GeoDataFrame:
        """Load the reference GeoDataFrame for *dataset_id*.

        Raises:
            FileNotFoundError: If the reference file cannot be found.
        """
        ref_path = self.find_reference_path(dataset_id)
        if ref_path is None:
            base = self._strip_osm_suffix(dataset_id)
            raise FileNotFoundError(
                f"Reference file not found for '{dataset_id}'. Run: matcher fetch overture {base}"
            )
        return self._load_gdf(ref_path)

    def load_target(self, dataset_id: str) -> gpd.GeoDataFrame:
        """Load the target GeoDataFrame for *dataset_id*.

        Raises:
            FileNotFoundError: If the target file cannot be found.
        """
        target_path = self.find_target_path(dataset_id)
        if target_path is None:
            base = self._strip_osm_suffix(dataset_id)
            if dataset_id.endswith("_osm"):
                hint = f"Run: matcher fetch osm {base}"
            else:
                hint = f"Run: matcher fetch target {dataset_id}"
            raise FileNotFoundError(f"Target file not found for '{dataset_id}'. {hint}")
        return self._load_gdf(target_path)

    def load_pair(
        self,
        dataset_id: str,
        *,
        project: bool = False,
    ) -> LoadedPair:
        """Load a reference/target pair for *dataset_id*.

        Args:
            dataset_id: Identifier for the dataset (e.g. ``"us_boston_streets"``
                or ``"us_boston_streets_osm"``).
            project: If ``True``, reproject both GeoDataFrames to a projected
                CRS (UTM) suitable for distance computations.

        Returns:
            A :class:`LoadedPair` with the loaded GeoDataFrames and metadata.

        Raises:
            FileNotFoundError: If reference or target files are missing.
        """
        ref_path, target_path = self._resolve_paths(dataset_id)
        reference = self._load_gdf(ref_path)
        target = self._load_gdf(target_path)

        if project:
            result = ensure_projected_crs(reference, target)
            reference = result.reference
            target = result.target

        return LoadedPair(
            reference=reference,
            target=target,
            dataset_id=dataset_id,
            reference_path=ref_path,
            target_path=target_path,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_paths(self, dataset_id: str) -> tuple[Path, Path]:
        """Resolve reference and target file paths for *dataset_id*.

        Raises:
            FileNotFoundError: With an actionable message if either file is missing.
        """
        ref_path = self.find_reference_path(dataset_id)
        target_path = self.find_target_path(dataset_id)

        base = self._strip_osm_suffix(dataset_id)

        if ref_path is None:
            raise FileNotFoundError(
                f"Reference file not found for '{dataset_id}'. Run: matcher fetch overture {base}"
            )
        if target_path is None:
            if dataset_id.endswith("_osm"):
                hint = f"Run: matcher fetch osm {base}"
            else:
                hint = f"Run: matcher fetch target {dataset_id}"
            raise FileNotFoundError(f"Target file not found for '{dataset_id}'. {hint}")
        return ref_path, target_path

    def _load_gdf(self, path: Path) -> gpd.GeoDataFrame:
        """Load a single GeoDataFrame with CRS normalisation, filtering, and validation.

        Steps:
            1. Return from session cache if available.
            2. Read parquet (or fall back to ``gpd.read_file``).
            3. Default CRS to WGS84 (EPSG:4326) if missing.
            4. Filter to LineString geometries.
            5. Drop null geometries.
            6. Validate required columns (raise on missing ``id``).
            7. Log missing optional columns at debug level.
            8. Validate that the result is non-empty.
            9. Cache in session if active.

        Raises:
            ValueError: If required columns are missing or result is empty.
        """
        resolved = path.resolve()

        # 1. Session cache check
        if self._cache is not None and resolved in self._cache:
            logger.debug(f"Session cache hit: {path.name}")
            return self._cache[resolved]

        # 2. Read file
        if path.suffix == ".parquet":
            gdf = gpd.read_parquet(path)
        else:
            gdf = gpd.read_file(path)

        # 3. Default CRS
        if gdf.crs is None:
            logger.info(f"No CRS found for {path.name}, defaulting to WGS84 (EPSG:4326)")
            gdf = gdf.set_crs("EPSG:4326")

        # 4. Filter to LineStrings (also drops nulls/MultiLineStrings)
        gdf = filter_to_linestrings(gdf, source_name=path.name)

        # 5. Drop remaining null geometries (belt-and-suspenders)
        null_mask = gdf.geometry.isna()
        if null_mask.any():
            gdf = gdf[~null_mask].copy()

        # 6. Validate required columns
        present = set(gdf.columns)
        missing_required = _REQUIRED_COLUMNS - present - {"geometry"}  # geometry is always present
        if missing_required:
            raise ValueError(
                f"{path.name}: missing required columns {missing_required}. "
                "These columns are required for downstream processing."
            )

        # 7. Log missing optional columns
        missing_optional = _EXPECTED_COLUMNS - present
        if missing_optional:
            logger.debug(f"{path.name}: missing optional columns {missing_optional}")

        # 8. Validate non-empty
        if gdf.empty:
            raise ValueError(
                f"No LineString geometries remaining after filtering {path.name}. "
                f"Check the source data."
            )

        # 9. Cache if session active
        if self._cache is not None:
            self._cache[resolved] = gdf

        return gdf

    @staticmethod
    def _strip_osm_suffix(dataset_id: str) -> str:
        """Strip the ``_osm`` suffix to get the base region/dataset name."""
        if dataset_id.endswith("_osm"):
            return dataset_id[: -len("_osm")]
        return dataset_id
