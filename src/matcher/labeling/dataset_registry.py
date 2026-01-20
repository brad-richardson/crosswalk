"""Dataset registry for managing labeled dataset metadata."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DEFAULT_REGISTRY_PATH = Path("datasets.csv")


@dataclass
class Dataset:
    """Metadata for a labeled dataset."""

    dataset_id: str  # Primary key, used in paths (e.g., "boston_streets")
    name: str  # Human-readable name (e.g., "Boston Streets")
    type: str  # "road", "bike", "sidewalk", "transit"
    fetch_url: str = ""  # URL to download source data
    info_url: str = ""  # Documentation URL
    metadata: dict = field(default_factory=dict)  # Extensible JSON properties

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV storage."""
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "type": self.type,
            "fetch_url": self.fetch_url,
            "info_url": self.info_url,
            "metadata": json.dumps(self.metadata) if self.metadata else "{}",
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Dataset":
        """Create from dictionary (CSV row)."""
        metadata = d.get("metadata", "{}")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        return cls(
            dataset_id=d["dataset_id"],
            name=d["name"],
            type=d["type"],
            fetch_url=d.get("fetch_url", ""),
            info_url=d.get("info_url", ""),
            metadata=metadata or {},
        )


class DatasetRegistry:
    """Manages dataset metadata stored in datasets.csv."""

    def __init__(self, path: Path = None):
        self.path = Path(path) if path else DEFAULT_REGISTRY_PATH
        self._df: pd.DataFrame | None = None

    @property
    def df(self) -> pd.DataFrame:
        """Lazy load dataframe."""
        if self._df is None:
            self._df = self._load()
        return self._df

    def _load(self) -> pd.DataFrame:
        """Load registry from CSV."""
        if self.path.exists():
            try:
                return pd.read_csv(self.path)
            except Exception:
                pass
        # Return empty DataFrame with correct columns
        return pd.DataFrame(
            columns=["dataset_id", "name", "type", "fetch_url", "info_url", "metadata"]
        )

    def _save(self) -> None:
        """Save registry to CSV."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.path, index=False, float_format="%.10f")

    def get(self, dataset_id: str) -> Dataset | None:
        """Get dataset by ID."""
        matches = self.df[self.df["dataset_id"] == dataset_id]
        if len(matches) == 0:
            return None
        return Dataset.from_dict(matches.iloc[0].to_dict())

    def list_all(self) -> list[Dataset]:
        """List all datasets."""
        return [Dataset.from_dict(row.to_dict()) for _, row in self.df.iterrows()]

    def add(self, dataset: Dataset) -> None:
        """Add a new dataset to the registry."""
        if self.exists(dataset.dataset_id):
            raise ValueError(f"Dataset {dataset.dataset_id} already exists")
        new_row = pd.DataFrame([dataset.to_dict()])
        self._df = pd.concat([self.df, new_row], ignore_index=True)
        self._save()

    def update(self, dataset: Dataset) -> None:
        """Update an existing dataset."""
        if not self.exists(dataset.dataset_id):
            raise ValueError(f"Dataset {dataset.dataset_id} does not exist")
        mask = self.df["dataset_id"] == dataset.dataset_id
        for key, value in dataset.to_dict().items():
            self._df.loc[mask, key] = value
        self._save()

    def exists(self, dataset_id: str) -> bool:
        """Check if dataset exists."""
        return dataset_id in self.df["dataset_id"].values

    def ensure_exists(self, dataset_id: str, name: str = None, type: str = "road") -> Dataset:
        """Get or create a dataset entry."""
        if self.exists(dataset_id):
            return self.get(dataset_id)
        # Create new dataset with defaults
        dataset = Dataset(
            dataset_id=dataset_id,
            name=name or dataset_id.replace("_", " ").title(),
            type=type,
        )
        self.add(dataset)
        return dataset

    def get_crs(self, dataset_id: str) -> str | None:
        """Get CRS for a dataset from its metadata.

        Args:
            dataset_id: Dataset identifier

        Returns:
            CRS string (e.g., "EPSG:4326") or None if not set
        """
        dataset = self.get(dataset_id)
        if dataset is None:
            return None
        return dataset.metadata.get("crs")

    def set_crs(self, dataset_id: str, crs: str) -> None:
        """Set CRS for a dataset in its metadata.

        Args:
            dataset_id: Dataset identifier
            crs: CRS string (e.g., "EPSG:4326")

        Raises:
            ValueError: If dataset doesn't exist
        """
        dataset = self.get(dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset {dataset_id} does not exist")
        dataset.metadata["crs"] = crs
        self.update(dataset)
