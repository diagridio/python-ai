# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Which stream modes a streamed run records.

A streamed run carries the modes it was started with in its config
(:data:`STREAM_MODES_KEY`), which every activity is handed. The chunks of
``messages``, ``updates`` and ``custom`` are always written; a ``values``
chunk (the whole output state) and the ``tasks``/``debug`` events (a task's
start event carries the node's whole input) only when the run asked for that
mode. Without them a run streamed with the default modes writes no copy of
the state per activity to the stream log.

A run whose config has no modes (started before the runner recorded them,
or not streamed) records every mode, as before.
"""

import json
from typing import Any, Dict, FrozenSet, Iterable, List, Optional

from diagrid.agent.core.workflow.stream import UPDATES, VALUES

from .task_events import DEBUG, TASKS

#: Workflow config key: the stream modes the run was started with.
STREAM_MODES_KEY = "diagrid_stream_modes"

#: Modes whose chunks a streamed run always writes.
ALWAYS_RECORDED = ("messages", UPDATES, "custom")
#: Modes whose chunks are written only when the run asked for them.
ON_REQUEST = (VALUES, TASKS, DEBUG)


#: The modes of a task's start and result events, which are written together.
_TASK_MODES = frozenset((TASKS, DEBUG))


class UnrecordedStreamMode(ValueError):
    """A stream asked for a mode its run did not record (``422``)."""


def recorded_modes(config: Optional[Dict[str, Any]]) -> Optional[FrozenSet[str]]:
    """The modes a run with ``config`` records, or ``None`` for all of them.

    ``tasks`` and ``debug`` are the same events, written together: a run
    started with either records both.
    """
    requested = (config or {}).get(STREAM_MODES_KEY)
    if not isinstance(requested, list):
        return None
    recorded = frozenset(ALWAYS_RECORDED) | frozenset(requested)
    if recorded & _TASK_MODES:
        recorded |= _TASK_MODES
    return recorded


def records(config: Optional[Dict[str, Any]], mode: str) -> bool:
    """Whether a run with ``config`` writes chunks of ``mode``."""
    modes = recorded_modes(config)
    return modes is None or mode in modes


def records_values(config: Optional[Dict[str, Any]]) -> bool:
    """Whether a run with ``config`` writes ``values`` chunks."""
    return records(config, VALUES)


def records_task_events(config: Optional[Dict[str, Any]]) -> bool:
    """Whether a run with ``config`` writes ``tasks`` and ``debug`` events."""
    return records(config, TASKS)


def with_stream_modes(config: Dict[str, Any], modes: Iterable[str]) -> Dict[str, Any]:
    """A copy of ``config`` that records ``modes`` (in a canonical order)."""
    wanted = set(modes)
    order = [*ALWAYS_RECORDED, *ON_REQUEST]
    return {**config, STREAM_MODES_KEY: [m for m in order if m in wanted]}


def recorded_in_state(state: Any) -> Optional[FrozenSet[str]]:
    """The modes the run of a workflow ``state`` records (from its input)."""
    raw = getattr(state, "serialized_input", None)
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return None
    config = data.get("config") if isinstance(data, dict) else None
    return recorded_modes(config if isinstance(config, dict) else None)


def check_recorded(requested: Iterable[str], state: Any, workflow_id: str) -> List[str]:
    """``requested`` if the run of ``state`` recorded all of them.

    Raises :class:`UnrecordedStreamMode`, naming the modes the run recorded,
    for a mode whose chunks were never written.
    """
    wanted = list(requested)
    recorded = recorded_in_state(state)
    if recorded is None:
        return wanted
    missing = [m for m in wanted if m not in recorded]
    if missing:
        order = [*ALWAYS_RECORDED, *ON_REQUEST]
        raise UnrecordedStreamMode(
            f"Run {workflow_id!r} did not record stream_mode {missing!r}: it "
            f"records {[m for m in order if m in recorded]!r}, the modes it was "
            "started with plus messages, updates and custom. "
            + _refusal_hint(missing, workflow_id)
        )
    return wanted


def _refusal_hint(missing: List[str], workflow_id: str) -> str:
    """Where a client finds what the refused modes would have carried."""
    if VALUES in missing:
        return (
            "Its final state is in the final event and in "
            f"GET /agent/run/{workflow_id}."
        )
    return (
        "Its task events are written only when it is started with tasks or "
        "debug; the updates mode carries each task's writes."
    )
