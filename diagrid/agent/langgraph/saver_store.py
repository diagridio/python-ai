# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A thread's state in a LangGraph checkpoint saver (``MemorySaver`` and the like).

The saver keeps the thread's latest checkpoint as LangGraph writes one: the
channel values as Python objects. They are converted to and from the
workflow's channel state at the edges. The saver must have the synchronous
API (``get_tuple`` and ``put``), as the activities that use it are
synchronous.
"""

from typing import Any, Dict, List, Optional

from .models import ChannelState

#: The checkpoint namespace of a graph's own (not a subgraph's) checkpoints.
_ROOT_NS = ""


def is_saver(obj: Any) -> bool:
    """Whether ``obj`` is a LangGraph checkpoint saver."""
    try:
        from langgraph.checkpoint.base import BaseCheckpointSaver
    except ImportError:  # pragma: no cover - langgraph without checkpoints
        return False
    return isinstance(obj, BaseCheckpointSaver)


def _config(thread_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": _ROOT_NS}}


class SaverThreadStore:
    """A thread's last state in a LangGraph ``BaseCheckpointSaver``."""

    def __init__(self, saver: Any, channels: List[str]) -> None:
        self._saver = saver
        # Only the graph's state channels: a native checkpoint also holds the
        # loop's own trigger channels, which the workflow does not keep.
        self._channels = frozenset(channels)

    def load(self, thread_id: str) -> Optional[Dict[str, Any]]:
        from .workflow import _serialize_input

        found = self._saver.get_tuple(_config(thread_id))
        if found is None:
            return None
        values = {
            name: value
            for name, value in (found.checkpoint.get("channel_values") or {}).items()
            if not self._channels or name in self._channels
        }
        stored = _serialize_input(values)
        return ChannelState(
            values=stored, versions={name: 1 for name in stored}
        ).to_dict()

    def save(self, checkpoint: Any) -> None:
        from langgraph.checkpoint.base import empty_checkpoint

        from .workflow import _reconstruct_state

        state = ChannelState.from_dict(checkpoint.channel_state)
        values = {
            name: value
            for name, value in _reconstruct_state(state).items()
            if name in state.values
        }
        versions = {name: self._saver.get_next_version(None, None) for name in values}
        native = empty_checkpoint()
        native["channel_values"] = values
        native["channel_versions"] = versions
        self._saver.put(
            _config(checkpoint.thread_id),
            native,
            {"source": "loop", "step": checkpoint.steps, "parents": {}},
            versions,
        )
