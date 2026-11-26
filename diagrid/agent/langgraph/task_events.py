# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""The ``tasks`` and ``debug`` stream modes, shaped as LangGraph shapes them.

A task's start event is sent when its node starts, and its result next to its
``updates`` chunk. ``debug`` wraps both with the step, a timestamp and the
event's type. Without a checkpointer that is everything native ``debug`` sends;
there are no LangGraph checkpoints here, so no ``checkpoint`` events.

A failed attempt sends no result: Dapr retries it, ``node_reset`` covers its
partial output, and only the workflow knows which failure is the last.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langgraph.checkpoint.base import EXCLUDED_METADATA_KEYS

TASKS = "tasks"
DEBUG = "debug"
#: ``debug`` event types.
TASK = "task"
TASK_RESULT = "task_result"

# How LangGraph marks a channel written more than once by one task.
_MULTIPLE_WRITES = "$writes"


def task_id(workflow_id: str, step: int, task_key: str) -> str:
    """A task's id: the same for its start, its result and every attempt."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, f"{workflow_id}:{step}:{task_key}"))


def task_metadata(
    metadata: Dict[str, Any], tags: Sequence[str]
) -> Optional[Dict[str, Any]]:
    """What a start event carries as ``metadata``: the user's keys and tags."""
    kept = {k: v for k, v in metadata.items() if k not in EXCLUDED_METADATA_KEYS}
    user_tags = [t for t in tags if not t.startswith("seq:step")]
    if user_tags:
        kept["tags"] = user_tags
    return kept or None


def task_start(
    task: str,
    name: str,
    task_input: Any,
    triggers: Sequence[str],
    metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """A ``tasks`` start event: ``{id, name, input, triggers[, metadata]}``."""
    payload: Dict[str, Any] = {
        "id": task,
        "name": name,
        "input": task_input,
        "triggers": tuple(triggers),
    }
    if metadata:
        payload["metadata"] = metadata
    return payload


def task_result(task: str, name: str, writes: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """A ``tasks`` result event of a task that finished with ``writes``."""
    return {
        "id": task,
        "name": name,
        "error": None,
        "result": _result_writes(writes),
        "interrupts": [],
    }


def _result_writes(writes: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """``{channel: value}``; a channel written more than once lists its writes."""
    result: Dict[str, Any] = {}
    for channel, value in writes:
        if channel not in result:
            result[channel] = value
            continue
        existing = result[channel]
        earlier = (
            existing[_MULTIPLE_WRITES]
            if isinstance(existing, dict) and _MULTIPLE_WRITES in existing
            else [existing]
        )
        result[channel] = {_MULTIPLE_WRITES: [*earlier, value]}
    return result


def debug_event(step: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    """A ``tasks`` event as ``debug`` sends it; ``step`` as ``langgraph_step``."""
    return {
        "step": step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": TASK_RESULT if "result" in payload else TASK,
        "payload": payload,
    }
