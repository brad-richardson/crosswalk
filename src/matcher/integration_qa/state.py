"""Session state management for integration QA app."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

CONFIG_PATH = Path.home() / ".matcher_reviewer_config.json"


@dataclass
class QASession:
    """Session state for integration QA."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    reviewer_name: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    current_view: str = "orphans"  # "orphans" or "merged"
    current_index: int = 0
    filter_by_component: Optional[int] = None
    filter_by_source: Optional[str] = None
    filter_by_priority: Optional[str] = None  # "high", "medium", "low"
    show_reviewed: bool = False
    undo_stack: list[dict] = field(default_factory=list)

    def push_undo(self, action: dict) -> None:
        """Push action to undo stack."""
        self.undo_stack.append(action)
        # Keep only last 50 actions
        if len(self.undo_stack) > 50:
            self.undo_stack = self.undo_stack[-50:]

    def pop_undo(self) -> Optional[dict]:
        """Pop last action from undo stack."""
        if self.undo_stack:
            return self.undo_stack.pop()
        return None


def load_reviewer_name() -> str:
    """Load saved reviewer name from config file."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                config = json.load(f)
                return config.get("reviewer_name", "")
        except Exception:
            # Config file may be corrupted or have incompatible format;
            # fall back to default value rather than crash
            pass
    return ""


def save_reviewer_name(name: str) -> None:
    """Save reviewer name to config file."""
    config = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                config = json.load(f)
        except Exception:
            # Config file may be corrupted; start fresh rather than crash
            pass

    config["reviewer_name"] = name
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f)
