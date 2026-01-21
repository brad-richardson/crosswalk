"""Dataset registry for managing dataset metadata.

Reads from YAML configs in the datasets/ directory at repo root.
Provides backward-compatible interface for code that used the old CSV-based registry.
"""

from dataclasses import dataclass, field
from pathlib import Path

from ..datasets.schema import (
    DatasetConfig,
    get_dataset_config,
    get_datasets_dir,
    list_dataset_configs,
    save_dataset_config,
)


@dataclass
class Dataset:
    """Metadata for a labeled dataset.

    This is a compatibility wrapper around the new DatasetConfig schema.
    """

    dataset_id: str  # Primary key, used in paths (e.g., "boston_streets")
    name: str  # Human-readable name (e.g., "Boston Streets")
    type: str  # "road", "bike", "sidewalk", "transit"
    fetch_url: str = ""  # URL to download source data
    info_url: str = ""  # Documentation URL
    metadata: dict = field(default_factory=dict)  # Extensible JSON properties

    @classmethod
    def from_config(cls, config: DatasetConfig) -> "Dataset":
        """Create from DatasetConfig."""
        metadata = {}
        if config.fetch and config.fetch.crs:
            metadata["crs"] = config.fetch.crs
        if config.fetch and config.fetch.class_column:
            metadata["classification_column"] = config.fetch.class_column

        return cls(
            dataset_id=config.name,
            name=config.display_name or config.name.replace("_", " ").title(),
            type=config.type,
            fetch_url=config.source.url if config.source else "",
            info_url=config.source.portal_url if config.source else "",
            metadata=metadata,
        )


class DatasetRegistry:
    """Manages dataset metadata from YAML configs in datasets/ directory.

    This is a compatibility wrapper that reads from the new YAML-based
    configuration system while maintaining the same interface as the
    old CSV-based registry.
    """

    def __init__(self, path: Path | None = None):
        """Initialize the registry.

        Args:
            path: Ignored (kept for backward compatibility).
                  Always reads from datasets/ directory.
        """
        self._configs_dir = get_datasets_dir()
        self._cache: dict[str, DatasetConfig] = {}

    def _load_config(self, dataset_id: str) -> DatasetConfig | None:
        """Load a dataset config, using cache."""
        if dataset_id not in self._cache:
            config = get_dataset_config(dataset_id)
            if config:
                self._cache[dataset_id] = config
        return self._cache.get(dataset_id)

    def get(self, dataset_id: str) -> Dataset | None:
        """Get dataset by ID."""
        config = self._load_config(dataset_id)
        if config is None:
            return None
        return Dataset.from_config(config)

    def list_all(self) -> list[Dataset]:
        """List all datasets."""
        datasets = []
        for name in list_dataset_configs():
            config = self._load_config(name)
            if config:
                datasets.append(Dataset.from_config(config))
        return datasets

    def add(self, dataset: Dataset) -> None:
        """Add a new dataset to the registry.

        Creates a new YAML config file.
        """
        if self.exists(dataset.dataset_id):
            raise ValueError(f"Dataset {dataset.dataset_id} already exists")

        from ..datasets.schema import FetchConfig, SourceConfig

        config = DatasetConfig(
            name=dataset.dataset_id,
            display_name=dataset.name,
            type=dataset.type,
            source=SourceConfig(
                type="arcgis" if dataset.fetch_url else "unknown",
                url=dataset.fetch_url or None,
                portal_url=dataset.info_url or None,
            ),
            fetch=FetchConfig(
                crs=dataset.metadata.get("crs", "EPSG:4326"),
                class_column=dataset.metadata.get("classification_column"),
            ),
        )

        config_path = self._configs_dir / f"{dataset.dataset_id}.yaml"
        save_dataset_config(config, config_path)
        self._cache[dataset.dataset_id] = config

    def update(self, dataset: Dataset) -> None:
        """Update an existing dataset."""
        if not self.exists(dataset.dataset_id):
            raise ValueError(f"Dataset {dataset.dataset_id} does not exist")

        config = self._load_config(dataset.dataset_id)
        if config is None:
            raise ValueError(f"Dataset {dataset.dataset_id} does not exist")

        # Update config with new values
        config.display_name = dataset.name
        config.type = dataset.type

        if config.source:
            config.source.url = dataset.fetch_url or None
            config.source.portal_url = dataset.info_url or None

        if config.fetch:
            if dataset.metadata.get("crs"):
                config.fetch.crs = dataset.metadata["crs"]
            if dataset.metadata.get("classification_column"):
                config.fetch.class_column = dataset.metadata["classification_column"]

        config_path = self._configs_dir / f"{dataset.dataset_id}.yaml"
        save_dataset_config(config, config_path)
        self._cache[dataset.dataset_id] = config

    def exists(self, dataset_id: str) -> bool:
        """Check if dataset exists."""
        config_path = self._configs_dir / f"{dataset_id}.yaml"
        return config_path.exists()

    def ensure_exists(
        self, dataset_id: str, name: str | None = None, type: str = "road"
    ) -> Dataset:
        """Get or create a dataset entry."""
        if self.exists(dataset_id):
            dataset = self.get(dataset_id)
            if dataset:
                return dataset

        # Create new dataset with defaults
        dataset = Dataset(
            dataset_id=dataset_id,
            name=name or dataset_id.replace("_", " ").title(),
            type=type,
        )
        self.add(dataset)
        return dataset

    def get_crs(self, dataset_id: str) -> str | None:
        """Get CRS for a dataset from its config.

        Args:
            dataset_id: Dataset identifier

        Returns:
            CRS string (e.g., "EPSG:4326") or None if not set
        """
        config = self._load_config(dataset_id)
        if config is None:
            return None
        if config.fetch:
            return config.fetch.crs
        return None

    def set_crs(self, dataset_id: str, crs: str) -> None:
        """Set CRS for a dataset in its config.

        Args:
            dataset_id: Dataset identifier
            crs: CRS string (e.g., "EPSG:4326")

        Raises:
            ValueError: If dataset doesn't exist
        """
        config = self._load_config(dataset_id)
        if config is None:
            raise ValueError(f"Dataset {dataset_id} does not exist")

        from ..datasets.schema import FetchConfig

        if config.fetch is None:
            config.fetch = FetchConfig(crs=crs)
        else:
            config.fetch.crs = crs

        config_path = self._configs_dir / f"{dataset_id}.yaml"
        save_dataset_config(config, config_path)
        self._cache[dataset_id] = config

    def get_config(self, dataset_id: str) -> DatasetConfig | None:
        """Get the full DatasetConfig for a dataset.

        This provides access to the new unified config format.
        """
        return self._load_config(dataset_id)

    def get_bbox(self, dataset_id: str) -> tuple[float, float, float, float] | None:
        """Get bounding box for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Bounding box tuple (xmin, ymin, xmax, ymax) or None
        """
        config = self._load_config(dataset_id)
        if config is None:
            return None
        if config.fetch:
            return config.fetch.bbox
        return None
