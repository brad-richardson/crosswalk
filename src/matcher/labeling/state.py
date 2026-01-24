"""Session state management for the labeling UI."""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import streamlit as st


@dataclass
class LabelingSession:
    """Tracks the current labeling session state."""

    session_id: str
    labeler_name: str
    started_at: datetime
    current_index: int = 0
    decision_filter: str | None = None  # "review", "match", "no_match", or None
    undo_stack: list = field(default_factory=list)


def init_session_state() -> None:
    """Initialize session state on first load."""
    if "session" not in st.session_state:
        st.session_state.session = LabelingSession(
            session_id=str(uuid.uuid4())[:8],
            labeler_name="",
            started_at=datetime.now(UTC),
        )

    if "candidates" not in st.session_state:
        st.session_state.candidates = []

    if "label_store" not in st.session_state:
        st.session_state.label_store = None

    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False

    if "is_loading" not in st.session_state:
        st.session_state.is_loading = False


def get_session() -> LabelingSession:
    """Get the current session."""
    return st.session_state.session


def set_labeler_name(name: str) -> None:
    """Set the labeler name."""
    st.session_state.session.labeler_name = name


def set_decision_filter(filter_value: str | None) -> None:
    """Set the decision filter and reset index."""
    st.session_state.session.decision_filter = filter_value
    st.session_state.session.current_index = 0


def advance_to_next() -> None:
    """Move to the next candidate."""
    st.session_state.session.current_index += 1


def go_to_previous() -> None:
    """Move to the previous candidate."""
    if st.session_state.session.current_index > 0:
        st.session_state.session.current_index -= 1


def push_undo(ref_id: str, target_id: str, label: str) -> None:
    """Push an action to the undo stack."""
    st.session_state.session.undo_stack.append(
        {
            "ref_id": ref_id,
            "target_id": target_id,
            "label": label,
            "index": st.session_state.session.current_index,
        }
    )
    # Keep only last 50 actions
    if len(st.session_state.session.undo_stack) > 50:
        st.session_state.session.undo_stack.pop(0)


def pop_undo() -> dict | None:
    """Pop the last action from the undo stack."""
    if st.session_state.session.undo_stack:
        return st.session_state.session.undo_stack.pop()
    return None


def reset_session() -> None:
    """Reset the session state."""
    st.session_state.session = LabelingSession(
        session_id=str(uuid.uuid4())[:8],
        labeler_name=st.session_state.session.labeler_name,
        started_at=datetime.now(UTC),
    )
    st.session_state.candidates = []
    st.session_state.data_loaded = False
