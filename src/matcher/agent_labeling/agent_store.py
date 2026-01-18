"""Agent label storage for consensus-based labeling.

Stores agent labels separately from human labels (training data) to enable:
- Tracking labels by agent ID
- Consensus detection across multiple agents
- Disagreement analysis for human review
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

# Column definitions for agent labels
AGENT_LABEL_COLUMNS = [
    "ref_id",
    "target_id",
    "label",  # "match", "no_match", "unsure"
    "confidence",  # Agent's confidence (0.0-1.0)
    "reasoning",  # Agent's reasoning/explanation
    "agent_id",
    "labeled_at",  # ISO timestamp
]


@dataclass
class AgentLabelStore:
    """Store agent labels separately from human labels.

    Each agent's labels are stored in a separate CSV file within the batch,
    enabling per-agent tracking and cross-agent consensus analysis.
    """

    batch_dir: Path
    agent_id: str
    _df: pd.DataFrame | None = None

    def __post_init__(self):
        self.batch_dir = Path(self.batch_dir)
        self.labels_dir = self.batch_dir / "labels" / self.agent_id
        self.csv_path = self.labels_dir / "data.csv"
        self._df = None

    @property
    def df(self) -> pd.DataFrame:
        """Lazy load dataframe."""
        if self._df is None:
            self._df = self._load()
        return self._df

    def _load(self) -> pd.DataFrame:
        """Load labels from CSV."""
        if self.csv_path.exists():
            try:
                return pd.read_csv(self.csv_path)
            except Exception as e:
                logger.warning(f"Failed to load labels from {self.csv_path}: {e}")
        return self._empty_dataframe()

    def _empty_dataframe(self) -> pd.DataFrame:
        """Create empty DataFrame with correct schema."""
        return pd.DataFrame(columns=AGENT_LABEL_COLUMNS)

    def save(self) -> None:
        """Save labels to CSV."""
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(self.csv_path, index=False)
        logger.info(f"Saved {len(self._df)} labels to {self.csv_path}")

    def add_label(
        self,
        ref_id: str,
        target_id: str,
        label: str,
        confidence: float = 1.0,
        reasoning: str = "",
    ) -> None:
        """Add a single label.

        Args:
            ref_id: Reference segment ID
            target_id: Target segment ID
            label: Label value (match, no_match, unsure)
            confidence: Agent's confidence in the label (0.0-1.0)
            reasoning: Agent's reasoning/explanation
        """
        new_row = {
            "ref_id": str(ref_id),
            "target_id": str(target_id),
            "label": label,
            "confidence": confidence,
            "reasoning": reasoning,
            "agent_id": self.agent_id,
            "labeled_at": datetime.now(UTC).isoformat(),
        }

        self._df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

    def add_labels_from_dataframe(self, labels_df: pd.DataFrame) -> int:
        """Add multiple labels from a DataFrame.

        Expected columns: ref_id, target_id, label, confidence (optional), reasoning (optional)

        Args:
            labels_df: DataFrame with labels

        Returns:
            Number of labels added
        """
        required_cols = ["ref_id", "target_id", "label"]
        for col in required_cols:
            if col not in labels_df.columns:
                raise ValueError(f"Missing required column: {col}")

        count = 0
        for _, row in labels_df.iterrows():
            self.add_label(
                ref_id=str(row["ref_id"]),
                target_id=str(row["target_id"]),
                label=str(row["label"]),
                confidence=float(row.get("confidence", 1.0)),
                reasoning=str(row.get("reasoning", "")),
            )
            count += 1

        return count

    def get_labeled_pairs(self) -> set[tuple[str, str]]:
        """Get set of labeled (ref_id, target_id) pairs."""
        df = self.df
        if len(df) == 0:
            return set()
        return set(zip(df["ref_id"], df["target_id"]))

    def get_stats(self) -> dict:
        """Get labeling statistics."""
        df = self.df
        if len(df) == 0:
            return {"total": 0, "match": 0, "no_match": 0, "unsure": 0}

        return {
            "total": len(df),
            "match": (df["label"] == "match").sum(),
            "no_match": (df["label"] == "no_match").sum(),
            "unsure": (df["label"] == "unsure").sum(),
            "mean_confidence": df["confidence"].mean() if "confidence" in df.columns else None,
        }

    @staticmethod
    def load_all_agents(batch_dir: Path) -> pd.DataFrame:
        """Load labels from all agents in a batch.

        Args:
            batch_dir: Batch directory

        Returns:
            DataFrame with all labels and 'agent_id' column
        """
        labels_dir = Path(batch_dir) / "labels"
        if not labels_dir.exists():
            return pd.DataFrame(columns=AGENT_LABEL_COLUMNS)

        dfs = []
        for agent_dir in labels_dir.iterdir():
            if agent_dir.is_dir():
                csv_path = agent_dir / "data.csv"
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        if "agent_id" not in df.columns:
                            df["agent_id"] = agent_dir.name
                        dfs.append(df)
                    except Exception as e:
                        logger.warning(f"Failed to load {csv_path}: {e}")

        if not dfs:
            return pd.DataFrame(columns=AGENT_LABEL_COLUMNS)

        return pd.concat(dfs, ignore_index=True)

    @staticmethod
    def find_disagreements(batch_dir: Path) -> pd.DataFrame:
        """Find candidates where agents disagree.

        Args:
            batch_dir: Batch directory

        Returns:
            DataFrame with disagreement info:
            - ref_id, target_id: Candidate IDs
            - agents: List of agents who labeled
            - labels: Dict of agent_id -> label
            - agreement_ratio: Fraction of agents with majority label
        """
        all_labels = AgentLabelStore.load_all_agents(batch_dir)

        if len(all_labels) == 0:
            return pd.DataFrame(
                columns=["ref_id", "target_id", "agents", "labels", "agreement_ratio"]
            )

        # Group by candidate
        disagreements = []
        for (ref_id, target_id), group in all_labels.groupby(["ref_id", "target_id"]):
            agents = group["agent_id"].tolist()
            labels_dict = dict(zip(group["agent_id"], group["label"]))

            if len(agents) < 2:
                continue  # Need at least 2 agents to have disagreement

            # Calculate agreement
            label_counts = group["label"].value_counts()
            majority_count = label_counts.max()
            agreement_ratio = majority_count / len(agents)

            # Only include if there's disagreement (agreement < 100%)
            if agreement_ratio < 1.0:
                disagreements.append(
                    {
                        "ref_id": ref_id,
                        "target_id": target_id,
                        "agents": agents,
                        "labels": labels_dict,
                        "agreement_ratio": agreement_ratio,
                    }
                )

        return pd.DataFrame(disagreements)

    @staticmethod
    def compute_consensus(batch_dir: Path, min_agents: int = 2) -> pd.DataFrame:
        """Compute consensus labels from multiple agents.

        Args:
            batch_dir: Batch directory
            min_agents: Minimum number of agents required for consensus

        Returns:
            DataFrame with:
            - ref_id, target_id: Candidate IDs
            - consensus_label: Majority label or 'no_consensus'
            - agreement_ratio: Fraction agreeing with consensus
            - num_agents: Number of agents who labeled
            - labels: Dict of agent_id -> label
        """
        all_labels = AgentLabelStore.load_all_agents(batch_dir)

        if len(all_labels) == 0:
            return pd.DataFrame(
                columns=[
                    "ref_id",
                    "target_id",
                    "consensus_label",
                    "agreement_ratio",
                    "num_agents",
                    "labels",
                ]
            )

        results = []
        for (ref_id, target_id), group in all_labels.groupby(["ref_id", "target_id"]):
            agents = group["agent_id"].tolist()
            labels_dict = dict(zip(group["agent_id"], group["label"]))
            num_agents = len(agents)

            if num_agents < min_agents:
                continue

            # Find majority label
            label_counts = group["label"].value_counts()
            majority_label = label_counts.index[0]
            majority_count = label_counts.iloc[0]
            agreement_ratio = majority_count / num_agents

            results.append(
                {
                    "ref_id": ref_id,
                    "target_id": target_id,
                    "consensus_label": majority_label,
                    "agreement_ratio": agreement_ratio,
                    "num_agents": num_agents,
                    "labels": labels_dict,
                }
            )

        return pd.DataFrame(results)

    @staticmethod
    def list_agents(batch_dir: Path) -> list[str]:
        """List all agents who have provided labels for a batch.

        Args:
            batch_dir: Batch directory

        Returns:
            List of agent IDs
        """
        labels_dir = Path(batch_dir) / "labels"
        if not labels_dir.exists():
            return []

        agents = []
        for agent_dir in labels_dir.iterdir():
            if agent_dir.is_dir():
                csv_path = agent_dir / "data.csv"
                if csv_path.exists():
                    agents.append(agent_dir.name)

        return sorted(agents)


def import_labels_csv(
    batch_dir: Path,
    agent_id: str,
    csv_path: Path,
) -> int:
    """Import labels from a CSV file.

    Args:
        batch_dir: Batch directory
        agent_id: Agent identifier
        csv_path: Path to CSV file with labels

    Returns:
        Number of labels imported
    """
    df = pd.read_csv(csv_path)

    store = AgentLabelStore(batch_dir, agent_id)
    count = store.add_labels_from_dataframe(df)
    store.save()

    logger.info(f"Imported {count} labels for agent '{agent_id}'")
    return count
