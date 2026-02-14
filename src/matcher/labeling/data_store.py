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
    "ref_oneway_lr",
    "target_oneway_lr",
    "ref_speed_limit_kph_lr",
    "target_speed_limit_kph_lr",
    # Topology context (captured at labeling time from full network)
    "ref_from_degree",
    "ref_to_degree",
    "ref_is_dead_end",
    "ref_is_intersection",
    "ref_degree_signature",  # JSON-serialized tuple
    "target_from_degree",
    "target_to_degree",
    "target_is_dead_end",
    "target_is_intersection",
    "target_degree_signature",  # JSON-serialized tuple
]


def _convert_numpy(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return [_convert_numpy(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy(x) for x in obj]
    return obj


def _serialize_lr_data(lr_data: list | None) -> str | None:
    """Serialize linear-referenced data to JSON string.

    Handles numpy arrays from Overture parquet reads by converting to
    native Python types before JSON serialization.
    """
    if lr_data is None:
        return None
    try:
        return json.dumps(_convert_numpy(lr_data))
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


def _serialize_degree_sig(sig: tuple | list | None) -> str | None:
    """Serialize degree_signature tuple to JSON string."""
    if sig is None:
        return None
    try:
        return json.dumps(list(sig))
    except (TypeError, ValueError):
        return None


def _deserialize_degree_sig(raw) -> tuple | None:
    """Deserialize degree_signature from JSON string."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        return tuple(json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        return None


def _flatten_topology(prefix: str, topo: dict | None) -> dict:
    """Flatten a topology dict into prefixed columns for storage."""
    if topo is None:
        return {}
    return {
        f"{prefix}_from_degree": topo.get("from_degree"),
        f"{prefix}_to_degree": topo.get("to_degree"),
        f"{prefix}_is_dead_end": topo.get("is_dead_end"),
        f"{prefix}_is_intersection": topo.get("is_intersection"),
        f"{prefix}_degree_signature": _serialize_degree_sig(topo.get("degree_signature")),
    }


def _reconstruct_topology(prefix: str, row) -> dict | None:
    """Reconstruct a topology dict from prefixed columns in a row.

    Returns None if all topology columns are missing (backward compat).
    """
    from_deg = row.get(f"{prefix}_from_degree")
    to_deg = row.get(f"{prefix}_to_degree")
    is_dead = row.get(f"{prefix}_is_dead_end")
    is_inter = row.get(f"{prefix}_is_intersection")
    deg_sig_raw = row.get(f"{prefix}_degree_signature")

    # All missing → no topology stored
    if all(v is None or (isinstance(v, float) and pd.isna(v)) for v in [from_deg, to_deg]):
        return None

    # Convert numeric types back (parquet may store as float)
    def _to_num(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return float("nan")
        return int(v) if float(v) == int(float(v)) else float(v)

    def _to_bool_or_nan(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return float("nan")
        return bool(v)

    return {
        "from_degree": _to_num(from_deg),
        "to_degree": _to_num(to_deg),
        "is_dead_end": _to_bool_or_nan(is_dead),
        "is_intersection": _to_bool_or_nan(is_inter),
        "degree_signature": _deserialize_degree_sig(deg_sig_raw) or (),
    }


@dataclass
class DataStore:
    """Manages raw pair data (geometries + attributes) in Parquet.

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
        """Load data from Parquet, returning empty GeoDataFrame if file doesn't exist."""
        import pyarrow.parquet as pq
        from shapely import wkb

        if not self.parquet_path.exists():
            return self._empty_geodataframe()

        try:
            # Read directly without Hive partitioning (single file, not dataset)
            # This avoids schema merge issues when partition column types differ
            pf = pq.ParquetFile(str(self.parquet_path))
            table = pf.read()
            df = table.to_pandas()

            # Convert WKB geometries to Shapely objects
            if "ref_geometry" in df.columns:
                df["ref_geometry"] = df["ref_geometry"].apply(
                    lambda b: wkb.loads(b) if b is not None else None
                )
            if "target_geometry_wkb" in df.columns:
                df["target_geometry"] = df["target_geometry_wkb"].apply(
                    lambda b: wkb.loads(b) if b is not None else None
                )
                df = df.drop(columns=["target_geometry_wkb"])
            elif "target_geometry" in df.columns:
                df["target_geometry"] = df["target_geometry"].apply(
                    lambda b: wkb.loads(b) if b is not None else None
                )

            # Convert to GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry="ref_geometry", crs="EPSG:4326")

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
        ref_oneway_lr: list | None = None,
        target_oneway_lr: list | None = None,
        ref_speed_limit_kph_lr: list | None = None,
        target_speed_limit_kph_lr: list | None = None,
        ref_topology: dict | None = None,
        target_topology: dict | None = None,
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
            ref_oneway_lr: Reference linear-referenced one-way direction data
            target_oneway_lr: Target linear-referenced one-way direction data
            ref_speed_limit_kph_lr: Reference linear-referenced speed limit (kph)
            target_speed_limit_kph_lr: Target linear-referenced speed limit (kph)
            ref_topology: Reference topology dict from compute_all_topology()
            target_topology: Target topology dict from compute_all_topology()
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
            "ref_oneway_lr": _serialize_lr_data(ref_oneway_lr),
            "target_oneway_lr": _serialize_lr_data(target_oneway_lr),
            "ref_speed_limit_kph_lr": _serialize_lr_data(ref_speed_limit_kph_lr),
            "target_speed_limit_kph_lr": _serialize_lr_data(target_speed_limit_kph_lr),
            **_flatten_topology("ref", ref_topology),
            **_flatten_topology("target", target_topology),
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

        def _str_or_none(val):
            """Convert NaN/non-string values to None for string fields."""
            return val if isinstance(val, str) else None

        result = {
            "gers_id": row["gers_id"],
            "target_id": row["target_id"],
            "ref_geometry": row["ref_geometry"],
            "target_geometry": row["target_geometry"],
            "ref_name": _str_or_none(row.get("ref_name")),
            "target_name": _str_or_none(row.get("target_name")),
            "ref_class": _str_or_none(row.get("ref_class")),
            "target_class": _str_or_none(row.get("target_class")),
            "ref_subclass": _str_or_none(row.get("ref_subclass")),
            "target_subclass": _str_or_none(row.get("target_subclass")),
            "ref_names_lr": _deserialize_lr_data(row.get("ref_names_lr")),
            "target_names_lr": _deserialize_lr_data(row.get("target_names_lr")),
            "ref_subclass_lr": _deserialize_lr_data(row.get("ref_subclass_lr")),
            "target_subclass_lr": _deserialize_lr_data(row.get("target_subclass_lr")),
            "ref_level_lr": _deserialize_lr_data(row.get("ref_level_lr")),
            "target_level_lr": _deserialize_lr_data(row.get("target_level_lr")),
            "ref_road_flags_lr": _deserialize_lr_data(row.get("ref_road_flags_lr")),
            "target_road_flags_lr": _deserialize_lr_data(row.get("target_road_flags_lr")),
        }

        # Reconstruct topology dicts (None if columns missing — backward compat)
        result["ref_topology"] = _reconstruct_topology("ref", row)
        result["target_topology"] = _reconstruct_topology("target", row)

        return result

    def has_pair(self, gers_id: str, target_id: str) -> bool:
        """Check if a data record exists for a labeled pair."""
        gdf = self.gdf
        mask = (gdf["gers_id"] == str(gers_id)) & (gdf["target_id"] == str(target_id))
        return mask.any()

    def delete_pair(self, gers_id: str, target_id: str) -> bool:
        """Delete data for a labeled pair.

        Args:
            gers_id: Overture reference segment ID
            target_id: Target segment ID

        Returns:
            True if found and deleted, False if pair not found.
        """
        gdf = self.gdf
        mask = (gdf["gers_id"] == str(gers_id)) & (gdf["target_id"] == str(target_id))
        if not mask.any():
            return False
        self._gdf = gdf[~mask].reset_index(drop=True)
        self.save()
        return True

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
        ref_oneway_lr: list | None = None,
        target_oneway_lr: list | None = None,
        ref_speed_limit_kph_lr: list | None = None,
        target_speed_limit_kph_lr: list | None = None,
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
            ref_oneway_lr: Reference linear-referenced one-way direction data
            target_oneway_lr: Target linear-referenced one-way direction data
            ref_speed_limit_kph_lr: Reference linear-referenced speed limit (kph)
            target_speed_limit_kph_lr: Target linear-referenced speed limit (kph)

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
        if ref_oneway_lr is not None:
            self._gdf.at[idx, "ref_oneway_lr"] = _serialize_lr_data(ref_oneway_lr)
        if target_oneway_lr is not None:
            self._gdf.at[idx, "target_oneway_lr"] = _serialize_lr_data(target_oneway_lr)
        if ref_speed_limit_kph_lr is not None:
            self._gdf.at[idx, "ref_speed_limit_kph_lr"] = _serialize_lr_data(ref_speed_limit_kph_lr)
        if target_speed_limit_kph_lr is not None:
            self._gdf.at[idx, "target_speed_limit_kph_lr"] = _serialize_lr_data(
                target_speed_limit_kph_lr
            )

        return True

    def update_topology(
        self,
        gers_id: str,
        target_id: str,
        ref_topology: dict | None = None,
        target_topology: dict | None = None,
    ) -> bool:
        """Update topology columns for an existing data record.

        Used during backfill to persist computed topology for future use,
        so that target data files don't need to be present on subsequent runs.

        Returns:
            True if record was found and updated, False if not found
        """
        gdf = self.gdf
        mask = (gdf["gers_id"] == str(gers_id)) & (gdf["target_id"] == str(target_id))

        if not mask.any():
            return False

        idx = gdf[mask].index[-1]

        if ref_topology is not None:
            flat = _flatten_topology("ref", ref_topology)
            for col, val in flat.items():
                self._gdf.at[idx, col] = val

        if target_topology is not None:
            flat = _flatten_topology("target", target_topology)
            for col, val in flat.items():
                self._gdf.at[idx, col] = val

        return True

    def save(self) -> None:
        """Save data to Parquet atomically with backup.

        Uses write-to-temp-then-rename pattern for atomic writes.
        Both geometry columns are serialized as WKB bytes in a plain
        Parquet file (not GeoParquet) since we have two geometry columns.
        """
        from shapely import wkb

        self.partition_path.mkdir(parents=True, exist_ok=True)

        temp_path = self.parquet_path.with_suffix(".parquet.tmp")
        backup_path = self.parquet_path.with_suffix(".parquet.bak")

        # Build a plain DataFrame with geometry columns as WKB bytes.
        # Must use .tolist() to fully escape GeoSeries geometry dtype,
        # otherwise pyarrow can't serialize the column.
        data = {}
        for col in self._gdf.columns:
            if col in ("ref_geometry", "target_geometry") and len(self._gdf) > 0:
                data[col] = [wkb.dumps(g) if g is not None else None for g in self._gdf[col]]
            else:
                data[col] = self._gdf[col].tolist()
        df = pd.DataFrame(data)

        # Write to temp file first
        df.to_parquet(temp_path, compression="zstd", index=False)

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

        dfs = []

        for partition_dir in data_dir.glob("dataset=*"):
            if not partition_dir.is_dir():
                continue
            dataset_id = partition_dir.name.split("=")[1]
            parquet_path = partition_dir / "data.parquet"
            if parquet_path.exists():
                try:
                    df = pd.read_parquet(parquet_path)

                    # Convert WKB geometries to Shapely objects
                    for geom_col in ("ref_geometry", "target_geometry"):
                        if geom_col in df.columns:
                            df[geom_col] = df[geom_col].apply(
                                lambda b: wkb.loads(b) if b is not None else None
                            )

                    # Handle legacy target_geometry_wkb column name
                    if "target_geometry_wkb" in df.columns:
                        df["target_geometry"] = df["target_geometry_wkb"].apply(
                            lambda b: wkb.loads(b) if b is not None else None
                        )
                        df = df.drop(columns=["target_geometry_wkb"])

                    df["dataset"] = dataset_id
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {parquet_path}: {e}")

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            # Ensure all expected columns exist (fill missing with None)
            for col in DATA_COLUMNS:
                if col not in combined.columns:
                    combined[col] = None
            return gpd.GeoDataFrame(combined, geometry="ref_geometry", crs="EPSG:4326")
        return gpd.GeoDataFrame(columns=DATA_COLUMNS + ["dataset"])
