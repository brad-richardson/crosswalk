"""Geometry persistence - companion CSV storing WKT geometries and attributes for labeled pairs."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger
from shapely import wkt
from shapely.geometry import LineString

# Default directory for geometry companion files (parallel to labels/)
DEFAULT_GEOMETRIES_DIR = Path("label_geometries")

# Columns for the geometry companion CSV
# Attributes are stored as JSON dicts for schema flexibility — adding new
# attributes (e.g., speed_limit, surface_type) doesn't require migration.
GEOMETRY_COLUMNS = [
    "gers_id",
    "target_id",
    "ref_geometry_wkt",
    "target_geometry_wkt",
    "ref_attributes",  # JSON dict: {"name": ..., "class": ..., "subclass": ..., ...}
    "target_attributes",  # JSON dict: {"name": ..., "class": ..., "subclass": ..., ...}
]


def _serialize_attributes(
    name: str | None = None,
    road_class: str | None = None,
    subclass: str | None = None,
    names_lr: list | None = None,
    subclass_lr: list | None = None,
    level_lr: list | None = None,
    road_flags_lr: list | None = None,
) -> str:
    """Serialize attribute key-value pairs to JSON, dropping None values.

    Converts numpy/pandas types to plain Python strings for JSON compatibility.
    LR data is stored as-is (already JSON-serializable list of dicts).

    Args:
        name: Flat name string
        road_class: Road class (use 'road_class' to avoid Python keyword)
        subclass: Road subclass
        names_lr: Linear-referenced names data
        subclass_lr: Linear-referenced subclass data
        level_lr: Linear-referenced level data
        road_flags_lr: Linear-referenced road flags data

    Returns:
        JSON string of attributes
    """
    clean = {}

    # Flat attributes (stored as strings)
    if name is not None:
        clean["name"] = str(name)
    if road_class is not None and not (isinstance(road_class, float) and pd.isna(road_class)):
        clean["class"] = str(road_class)
    if subclass is not None and not (isinstance(subclass, float) and pd.isna(subclass)):
        clean["subclass"] = str(subclass)

    # LR data (stored as-is - already JSON-serializable)
    if names_lr is not None:
        clean["names_lr"] = names_lr
    if subclass_lr is not None:
        clean["subclass_lr"] = subclass_lr
    if level_lr is not None:
        clean["level_lr"] = level_lr
    if road_flags_lr is not None:
        clean["road_flags_lr"] = road_flags_lr

    return json.dumps(clean)


def _deserialize_attributes(raw: str | None) -> dict:
    """Deserialize JSON attributes string, returning empty dict on failure."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


@dataclass
class GeometryStore:
    """Manages persisted geometries and attributes for labeled pairs.

    Stores a companion CSV alongside labels/ that captures raw geometries
    (as WKT) and attributes at label-creation time. This allows feature
    recomputation even when segment IDs drift after data re-fetches.

    Storage format:
        label_geometries/dataset={dataset_id}/data.csv

    Schema:
        gers_id, target_id           — composite key (join to labels/)
        ref_geometry_wkt             — WGS84 WKT geometry
        target_geometry_wkt          — WGS84 WKT geometry
        ref_attributes               — JSON dict of ref attributes
        target_attributes            — JSON dict of target attributes

    The JSON attribute columns are schema-flexible: any key-value pairs
    can be stored without requiring CSV schema changes.
    """

    dataset_id: str
    geometries_dir: Path = DEFAULT_GEOMETRIES_DIR
    _df: pd.DataFrame | None = None

    def __post_init__(self):
        self.geometries_dir = Path(self.geometries_dir)
        self.partition_path = self.geometries_dir / f"dataset={self.dataset_id}"
        self.csv_path = self.partition_path / "data.csv"
        self._df = None

    @property
    def df(self) -> pd.DataFrame:
        """Lazy load dataframe."""
        if self._df is None:
            self._df = self._load()
        return self._df

    def _load(self) -> pd.DataFrame:
        """Load geometries from CSV, returning empty DataFrame if file doesn't exist.

        Handles backward compatibility with the old fixed-column schema
        (ref_name, target_name, ref_class, target_class, ref_subclass,
        target_subclass) by migrating to JSON attributes on load.
        """
        if not self.csv_path.exists():
            return self._empty_dataframe()

        try:
            df = pd.read_csv(self.csv_path, dtype=str)

            # Migrate old fixed-column schema to JSON attributes
            if "ref_name" in df.columns and "ref_attributes" not in df.columns:
                logger.info(f"Migrating {self.csv_path} from fixed columns to JSON attributes")
                df = self._migrate_fixed_to_json(df)

            # Ensure all expected columns exist (forward compatibility)
            for col in GEOMETRY_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df
        except Exception as e:
            logger.warning(f"Failed to load geometry store from {self.csv_path}: {e}")
            return self._empty_dataframe()

    @staticmethod
    def _migrate_fixed_to_json(df: pd.DataFrame) -> pd.DataFrame:
        """Migrate old fixed-column schema to JSON attributes."""
        old_ref_cols = ["ref_name", "ref_class", "ref_subclass"]
        old_target_cols = ["target_name", "target_class", "target_subclass"]

        ref_attrs = []
        target_attrs = []
        for _, row in df.iterrows():
            ref_dict = {}
            for col in old_ref_cols:
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    key = col.replace("ref_", "")
                    ref_dict[key] = str(val)
            ref_attrs.append(json.dumps(ref_dict) if ref_dict else "{}")

            target_dict = {}
            for col in old_target_cols:
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    key = col.replace("target_", "")
                    target_dict[key] = str(val)
            target_attrs.append(json.dumps(target_dict) if target_dict else "{}")

        df["ref_attributes"] = ref_attrs
        df["target_attributes"] = target_attrs

        # Drop old columns
        cols_to_drop = [c for c in old_ref_cols + old_target_cols if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        return df

    def _empty_dataframe(self) -> pd.DataFrame:
        """Create empty DataFrame with correct schema."""
        return pd.DataFrame(columns=GEOMETRY_COLUMNS)

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
        """Add or update a geometry record for a labeled pair.

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
            "ref_geometry_wkt": wkt.dumps(ref_geometry),
            "target_geometry_wkt": wkt.dumps(target_geometry),
            "ref_attributes": _serialize_attributes(
                name=ref_name,
                road_class=ref_class,
                subclass=ref_subclass,
                names_lr=ref_names_lr,
                subclass_lr=ref_subclass_lr,
                level_lr=ref_level_lr,
                road_flags_lr=ref_road_flags_lr,
            ),
            "target_attributes": _serialize_attributes(
                name=target_name,
                road_class=target_class,
                subclass=target_subclass,
                names_lr=target_names_lr,
                subclass_lr=target_subclass_lr,
                level_lr=target_level_lr,
                road_flags_lr=target_road_flags_lr,
            ),
        }

        df = self.df

        # Remove existing entry for this pair (dedup on composite key)
        mask = (df["gers_id"] == str(gers_id)) & (df["target_id"] == str(target_id))
        if mask.any():
            df = df[~mask]

        self._df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    def get_pair(self, gers_id: str, target_id: str) -> dict | None:
        """Get persisted geometry and attributes for a labeled pair.

        Args:
            gers_id: Overture reference segment ID
            target_id: Target segment ID

        Returns:
            Dict with parsed LineString geometries and attributes, or None if not found.
            Attributes are unpacked from JSON into top-level keys:
            - Flat attributes: ref_name, target_name, ref_class, target_class, etc.
            - LR attributes: ref_names_lr, target_names_lr, etc.
        """
        df = self.df
        mask = (df["gers_id"] == str(gers_id)) & (df["target_id"] == str(target_id))
        matches = df[mask]

        if len(matches) == 0:
            return None

        row = matches.iloc[-1]  # Latest entry

        try:
            ref_geom = wkt.loads(row["ref_geometry_wkt"])
            target_geom = wkt.loads(row["target_geometry_wkt"])
        except Exception as e:
            logger.warning(f"Failed to parse WKT for {gers_id}/{target_id}: {e}")
            return None

        ref_attrs = _deserialize_attributes(row.get("ref_attributes"))
        target_attrs = _deserialize_attributes(row.get("target_attributes"))

        return {
            "gers_id": row["gers_id"],
            "target_id": row["target_id"],
            "ref_geometry": ref_geom,
            "target_geometry": target_geom,
            # Flat attributes
            "ref_name": ref_attrs.get("name"),
            "target_name": target_attrs.get("name"),
            "ref_class": ref_attrs.get("class"),
            "target_class": target_attrs.get("class"),
            "ref_subclass": ref_attrs.get("subclass"),
            "target_subclass": target_attrs.get("subclass"),
            # LR attributes
            "ref_names_lr": ref_attrs.get("names_lr"),
            "target_names_lr": target_attrs.get("names_lr"),
            "ref_subclass_lr": ref_attrs.get("subclass_lr"),
            "target_subclass_lr": target_attrs.get("subclass_lr"),
            "ref_level_lr": ref_attrs.get("level_lr"),
            "target_level_lr": target_attrs.get("level_lr"),
            "ref_road_flags_lr": ref_attrs.get("road_flags_lr"),
            "target_road_flags_lr": target_attrs.get("road_flags_lr"),
        }

    def has_pair(self, gers_id: str, target_id: str) -> bool:
        """Check if a geometry record exists for a labeled pair."""
        df = self.df
        mask = (df["gers_id"] == str(gers_id)) & (df["target_id"] == str(target_id))
        return mask.any()

    def save(self) -> None:
        """Save geometries to CSV atomically with backup.

        Uses write-to-temp-then-rename pattern matching LabelStore.save().
        """
        self.partition_path.mkdir(parents=True, exist_ok=True)

        temp_path = self.csv_path.with_suffix(".csv.tmp")
        backup_path = self.csv_path.with_suffix(".csv.bak")

        # Write to temp file first
        self._df.to_csv(temp_path, index=False)

        # Backup existing file (if present)
        if self.csv_path.exists():
            self.csv_path.replace(backup_path)

        # Atomic replace temp to final
        temp_path.replace(self.csv_path)

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
        """Update LR attributes for an existing geometry record.

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
        df = self.df
        mask = (df["gers_id"] == str(gers_id)) & (df["target_id"] == str(target_id))

        if not mask.any():
            return False

        # Find the index of the matching row
        idx = df[mask].index[-1]

        # Get existing attributes
        ref_attrs = _deserialize_attributes(df.at[idx, "ref_attributes"])
        target_attrs = _deserialize_attributes(df.at[idx, "target_attributes"])

        # Update with new LR data
        if ref_names_lr is not None:
            ref_attrs["names_lr"] = ref_names_lr
        if ref_subclass_lr is not None:
            ref_attrs["subclass_lr"] = ref_subclass_lr
        if ref_level_lr is not None:
            ref_attrs["level_lr"] = ref_level_lr
        if ref_road_flags_lr is not None:
            ref_attrs["road_flags_lr"] = ref_road_flags_lr

        if target_names_lr is not None:
            target_attrs["names_lr"] = target_names_lr
        if target_subclass_lr is not None:
            target_attrs["subclass_lr"] = target_subclass_lr
        if target_level_lr is not None:
            target_attrs["level_lr"] = target_level_lr
        if target_road_flags_lr is not None:
            target_attrs["road_flags_lr"] = target_road_flags_lr

        # Serialize back to JSON
        self._df.at[idx, "ref_attributes"] = json.dumps(ref_attrs)
        self._df.at[idx, "target_attributes"] = json.dumps(target_attrs)

        return True
