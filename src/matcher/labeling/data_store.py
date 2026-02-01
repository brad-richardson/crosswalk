"""Data persistence - GeoParquet storage for raw pair data (geometries + attributes).

This module provides the DataStore class which stores raw pair data in GeoParquet format.
Unlike the feature-embedded approach, this stores only geometries and attributes,
allowing features to be recomputed as the feature computation logic evolves.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger
from shapely.geometry import LineString

# Default directory for pair data
DEFAULT_DATA_DIR = Path("labels/data")

# Schema for data store - geometries stored as WKB in GeoParquet
DATA_COLUMNS = [
    "gers_id",
    "target_id",
    "ref_geometry",  # WKB geometry column
    "target_geometry",  # WKB geometry column
    "ref_name",
    "target_name",
    "ref_class",
    "target_class",
    "ref_subclass",
    "target_subclass",
    # Linear-referenced attributes (JSON-serialized)
    "ref_names_lr",
    "target_names_lr",
    "ref_subclass_lr",
    "target_subclass_lr",
    "ref_level_lr",
    "target_level_lr",
    "ref_road_flags_lr",
    "target_road_flags_lr",
]


def _serialize_lr_data(lr_data: list | None) -> str | None:
    """Serialize linear-referenced data to JSON string."""
    if lr_data is None:
        return None
    try:
        return json.dumps(lr_data)
    except (TypeError, ValueError):
        return None


def _deserialize_lr_data(raw: str | None) -> list | None:
    """Deserialize linear-referenced data from JSON string."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _hive_partitioning_kwargs() -> dict:
    """Return kwargs for read_parquet to handle Hive partitioning consistently.

    Forces the 'dataset' partition column to be a plain string type instead of
    dictionary-encoded, avoiding type mismatches (int8 vs int32 indices) when
    reading files written at different times or with different pyarrow versions.
    """
    import pyarrow as pa
    import pyarrow.dataset as ds

    return {
        "partitioning": ds.partitioning(
            pa.schema([("dataset", pa.string())]),
            flavor="hive",
        )
    }


@dataclass
class DataStore:
    """Manages raw pair data (geometries + attributes) in GeoParquet.

    Stores a companion GeoParquet file that captures geometries and attributes
    for labeled pairs. This allows features to be recomputed when feature
    computation logic changes, without losing the original geometry data.

    Storage format:
        labels/data/dataset={dataset_id}/data.parquet

    The GeoParquet format stores geometries as WKB, which geopandas reads/writes
    natively. This is more efficient and standardized than WKT storage.
    """

    dataset_id: str
    data_dir: Path = DEFAULT_DATA_DIR
    _gdf: gpd.GeoDataFrame | None = field(default=None, repr=False)

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.partition_path = self.data_dir / f"dataset={self.dataset_id}"
        self.parquet_path = self.partition_path / "data.parquet"
        self._gdf = None

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """Lazy load GeoDataFrame."""
        if self._gdf is None:
            self._gdf = self._load()
        return self._gdf

    def _load(self) -> gpd.GeoDataFrame:
        """Load data from GeoParquet, returning empty GeoDataFrame if file doesn't exist."""
        from shapely import wkb

        if not self.parquet_path.exists():
            return self._empty_geodataframe()

        try:
            gdf = gpd.read_parquet(self.parquet_path, **_hive_partitioning_kwargs())

            # Convert target_geometry from WKB if stored that way
            if "target_geometry_wkb" in gdf.columns:
                gdf["target_geometry"] = gdf["target_geometry_wkb"].apply(
                    lambda b: wkb.loads(b) if b is not None else None
                )
                gdf = gdf.drop(columns=["target_geometry_wkb"])

            # Ensure all expected columns exist
            for col in DATA_COLUMNS:
                if col not in gdf.columns:
                    gdf[col] = None
            return gdf
        except Exception as e:
            logger.warning(f"Failed to load data store from {self.parquet_path}: {e}")
            return self._empty_geodataframe()

    def _empty_geodataframe(self) -> gpd.GeoDataFrame:
        """Create empty GeoDataFrame with correct schema."""
        return gpd.GeoDataFrame(
            {col: pd.Series(dtype="object") for col in DATA_COLUMNS},
            geometry="ref_geometry",
            crs="EPSG:4326",
        )

    def add(
        self,
        gers_id: str,
        target_id: str,
        ref_geometry: LineString,
        target_geometry: LineString,
        ref_name: str | None = None,
        target_name: str | None = None,
        ref_class: str | None = None,
        target_class: str | None = None,
        ref_subclass: str | None = None,
        target_subclass: str | None = None,
        ref_names_lr: list | None = None,
        target_names_lr: list | None = None,
        ref_subclass_lr: list | None = None,
        target_subclass_lr: list | None = None,
        ref_level_lr: list | None = None,
        target_level_lr: list | None = None,
        ref_road_flags_lr: list | None = None,
        target_road_flags_lr: list | None = None,
    ) -> None:
        """Add or update a data record for a labeled pair.

        Deduplicates on (gers_id, target_id) composite key, keeping the latest.

        Args:
            gers_id: Overture reference segment ID
            target_id: Target segment ID
            ref_geometry: Reference geometry (WGS84 LineString)
            target_geometry: Target geometry (WGS84 LineString)
            ref_name: Reference segment name
            target_name: Target segment name
            ref_class: Reference road class
            target_class: Target road class
            ref_subclass: Reference road subclass
            target_subclass: Target road subclass
            ref_names_lr: Reference linear-referenced names data
            target_names_lr: Target linear-referenced names data
            ref_subclass_lr: Reference linear-referenced subclass data
            target_subclass_lr: Target linear-referenced subclass data
            ref_level_lr: Reference linear-referenced level data
            target_level_lr: Target linear-referenced level data
            ref_road_flags_lr: Reference linear-referenced road flags data
            target_road_flags_lr: Target linear-referenced road flags data
        """
        new_row = {
            "gers_id": str(gers_id),
            "target_id": str(target_id),
            "ref_geometry": ref_geometry,
            "target_geometry": target_geometry,
            "ref_name": ref_name,
            "target_name": target_name,
            "ref_class": str(ref_class) if ref_class is not None else None,
            "target_class": str(target_class) if target_class is not None else None,
            "ref_subclass": str(ref_subclass) if ref_subclass is not None else None,
            "target_subclass": str(target_subclass) if target_subclass is not None else None,
            "ref_names_lr": _serialize_lr_data(ref_names_lr),
            "target_names_lr": _serialize_lr_data(target_names_lr),
            "ref_subclass_lr": _serialize_lr_data(ref_subclass_lr),
            "target_subclass_lr": _serialize_lr_data(target_subclass_lr),
            "ref_level_lr": _serialize_lr_data(ref_level_lr),
            "target_level_lr": _serialize_lr_data(target_level_lr),
            "ref_road_flags_lr": _serialize_lr_data(ref_road_flags_lr),
            "target_road_flags_lr": _serialize_lr_data(target_road_flags_lr),
        }

        gdf = self.gdf

        # Remove existing entry for this pair (dedup on composite key)
        mask = (gdf["gers_id"] == str(gers_id)) & (gdf["target_id"] == str(target_id))
        if mask.any():
            gdf = gdf[~mask]

        # Create new row as GeoDataFrame
        new_gdf = gpd.GeoDataFrame([new_row], geometry="ref_geometry", crs="EPSG:4326")
        self._gdf = pd.concat([gdf, new_gdf], ignore_index=True)

    def get_pair(self, gers_id: str, target_id: str) -> dict | None:
        """Get data for a labeled pair.

        Args:
            gers_id: Overture reference segment ID
            target_id: Target segment ID

        Returns:
            Dict with geometries and attributes, or None if not found.
        """
        gdf = self.gdf
        mask = (gdf["gers_id"] == str(gers_id)) & (gdf["target_id"] == str(target_id))
        matches = gdf[mask]

        if len(matches) == 0:
            return None

        row = matches.iloc[-1]  # Latest entry

        return {
            "gers_id": row["gers_id"],
            "target_id": row["target_id"],
            "ref_geometry": row["ref_geometry"],
            "target_geometry": row["target_geometry"],
            "ref_name": row.get("ref_name"),
            "target_name": row.get("target_name"),
            "ref_class": row.get("ref_class"),
            "target_class": row.get("target_class"),
            "ref_subclass": row.get("ref_subclass"),
            "target_subclass": row.get("target_subclass"),
            "ref_names_lr": _deserialize_lr_data(row.get("ref_names_lr")),
            "target_names_lr": _deserialize_lr_data(row.get("target_names_lr")),
            "ref_subclass_lr": _deserialize_lr_data(row.get("ref_subclass_lr")),
            "target_subclass_lr": _deserialize_lr_data(row.get("target_subclass_lr")),
            "ref_level_lr": _deserialize_lr_data(row.get("ref_level_lr")),
            "target_level_lr": _deserialize_lr_data(row.get("target_level_lr")),
            "ref_road_flags_lr": _deserialize_lr_data(row.get("ref_road_flags_lr")),
            "target_road_flags_lr": _deserialize_lr_data(row.get("target_road_flags_lr")),
        }

    def has_pair(self, gers_id: str, target_id: str) -> bool:
        """Check if a data record exists for a labeled pair."""
        gdf = self.gdf
        mask = (gdf["gers_id"] == str(gers_id)) & (gdf["target_id"] == str(target_id))
        return mask.any()

    def update_lr_attributes(
        self,
        gers_id: str,
        target_id: str,
        ref_names_lr: list | None = None,
        target_names_lr: list | None = None,
        ref_subclass_lr: list | None = None,
        target_subclass_lr: list | None = None,
        ref_level_lr: list | None = None,
        target_level_lr: list | None = None,
        ref_road_flags_lr: list | None = None,
        target_road_flags_lr: list | None = None,
    ) -> bool:
        """Update LR attributes for an existing data record.

        Used during backfill to add LR data to existing records without
        replacing the geometries or flat attributes.

        Args:
            gers_id: Overture reference segment ID
            target_id: Target segment ID
            ref_names_lr: Reference linear-referenced names data
            target_names_lr: Target linear-referenced names data
            ref_subclass_lr: Reference linear-referenced subclass data
            target_subclass_lr: Target linear-referenced subclass data
            ref_level_lr: Reference linear-referenced level data
            target_level_lr: Target linear-referenced level data
            ref_road_flags_lr: Reference linear-referenced road flags data
            target_road_flags_lr: Target linear-referenced road flags data

        Returns:
            True if record was found and updated, False if not found
        """
        gdf = self.gdf
        mask = (gdf["gers_id"] == str(gers_id)) & (gdf["target_id"] == str(target_id))

        if not mask.any():
            return False

        # Find the index of the matching row
        idx = gdf[mask].index[-1]

        # Update LR columns if provided
        if ref_names_lr is not None:
            self._gdf.at[idx, "ref_names_lr"] = _serialize_lr_data(ref_names_lr)
        if target_names_lr is not None:
            self._gdf.at[idx, "target_names_lr"] = _serialize_lr_data(target_names_lr)
        if ref_subclass_lr is not None:
            self._gdf.at[idx, "ref_subclass_lr"] = _serialize_lr_data(ref_subclass_lr)
        if target_subclass_lr is not None:
            self._gdf.at[idx, "target_subclass_lr"] = _serialize_lr_data(target_subclass_lr)
        if ref_level_lr is not None:
            self._gdf.at[idx, "ref_level_lr"] = _serialize_lr_data(ref_level_lr)
        if target_level_lr is not None:
            self._gdf.at[idx, "target_level_lr"] = _serialize_lr_data(target_level_lr)
        if ref_road_flags_lr is not None:
            self._gdf.at[idx, "ref_road_flags_lr"] = _serialize_lr_data(ref_road_flags_lr)
        if target_road_flags_lr is not None:
            self._gdf.at[idx, "target_road_flags_lr"] = _serialize_lr_data(target_road_flags_lr)

        return True

    def save(self) -> None:
        """Save data to GeoParquet atomically with backup.

        Uses write-to-temp-then-rename pattern for atomic writes.
        Both geometry columns are saved as WKB in the GeoParquet file.
        """
        from shapely import wkb

        self.partition_path.mkdir(parents=True, exist_ok=True)

        temp_path = self.parquet_path.with_suffix(".parquet.tmp")
        backup_path = self.parquet_path.with_suffix(".parquet.bak")

        # Convert to GeoDataFrame with ref_geometry as active geometry
        # and target_geometry stored as WKB bytes
        gdf = self._gdf.copy()

        # Store target_geometry as WKB (bytes) so pyarrow can handle it
        if "target_geometry" in gdf.columns and len(gdf) > 0:
            gdf["target_geometry_wkb"] = gdf["target_geometry"].apply(
                lambda g: wkb.dumps(g) if g is not None else None
            )
            gdf = gdf.drop(columns=["target_geometry"])

        # Write to temp file first
        gdf.to_parquet(temp_path, compression="zstd")

        # Backup existing file (if present)
        if self.parquet_path.exists():
            self.parquet_path.replace(backup_path)

        # Atomic replace temp to final
        temp_path.replace(self.parquet_path)

    @staticmethod
    def load_all(data_dir: Path = DEFAULT_DATA_DIR) -> gpd.GeoDataFrame:
        """Load all data partitions.

        Uses Hive partitioning to read all dataset partitions
        and adds a 'dataset' column from the partition path.

        Args:
            data_dir: Directory containing Hive-partitioned data parquets

        Returns:
            GeoDataFrame with all pair data and 'dataset' column
        """
        from shapely import wkb

        data_dir = Path(data_dir)
        if not data_dir.exists():
            return gpd.GeoDataFrame(columns=DATA_COLUMNS + ["dataset"])

        gdfs = []

        for partition_dir in data_dir.glob("dataset=*"):
            if not partition_dir.is_dir():
                continue
            dataset_id = partition_dir.name.split("=")[1]
            parquet_path = partition_dir / "data.parquet"
            if parquet_path.exists():
                try:
                    gdf = gpd.read_parquet(parquet_path, **_hive_partitioning_kwargs())

                    # Convert target_geometry from WKB if stored that way
                    if "target_geometry_wkb" in gdf.columns:
                        gdf["target_geometry"] = gdf["target_geometry_wkb"].apply(
                            lambda b: wkb.loads(b) if b is not None else None
                        )
                        gdf = gdf.drop(columns=["target_geometry_wkb"])

                    gdf["dataset"] = dataset_id
                    gdfs.append(gdf)
                except Exception as e:
                    logger.warning(f"Failed to load {parquet_path}: {e}")

        if gdfs:
            result = pd.concat(gdfs, ignore_index=True)
            # Ensure all expected columns exist (fill missing with None)
            for col in DATA_COLUMNS:
                if col not in result.columns:
                    result[col] = None
            return result
        return gpd.GeoDataFrame(columns=DATA_COLUMNS + ["dataset"])
