# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Pieces of the LangGraph runner's streaming: encoding, events and routing."""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from diagrid.agent.core.workflow import stream as stream_module
from diagrid.agent.core.workflow.stream import (
    CONTROL_MODES,
    NODE_RESET,
    STREAM_DROPPED,
    WORKFLOW_STARTED,
    StreamClosed,
    StreamEvent,
)

from .streaming import (
    EXECUTED_KEY,
    GraphRunError,
    StreamControl,
    stream_unavailable_event,
    to_wire,
)

logger = logging.getLogger(__name__)

FINAL_EVENTS = (
    "workflow_completed",
    "workflow_failed",
    "workflow_terminated",
    "workflow_error",
)


def encode_chunk(event: StreamEvent) -> Tuple[Any, bool]:
    """``(wire form, ok)`` of a chunk; a chunk that cannot be encoded is replaced.

    ``to_wire`` runs arbitrary ``to_dict`` / ``model_dump`` on whatever a node
    wrote to ``custom``, so it can raise. That is a problem with one chunk, not
    with the run: the stream carries on with a placeholder.
    """
    try:
        return to_wire(event.mode, event.data), True
    except Exception as e:
        try:
            text = repr(event.data)[:200]
        except Exception:
            text = f"<{type(event.data).__name__}>"
        logger.warning(
            "Run %s: a %r chunk of node %r could not be encoded (%s: %s); "
            "sending a placeholder",
            event.workflow_id,
            event.mode,
            event.node,
            type(e).__name__,
            e,
        )
        return {"unserializable": text}, False


def warn_reset_unseen(workflow_id: str, data: Dict[str, Any]) -> None:
    """Say that a retried node's earlier chunks stay in a stream that cannot show it.

    A single-mode ``astream`` yields bare chunks, as native ``astream`` does, so
    a node reset has no place in it; with no ``on_control`` either, the
    consumer is never told and its text for that node holds both attempts.
    """
    logger.warning(
        "Run %s: node %r was retried (step %s) and its first attempt's chunks "
        "were already yielded. This astream has no way to tell you: pass "
        "on_control, or a list of stream modes, to receive node_reset and "
        "discard the partial output.",
        workflow_id,
        data.get("node"),
        data.get("step"),
    )


def count_unencodable(final: Dict[str, Any], unencodable: int) -> Dict[str, Any]:
    """``final`` with the chunks lost to encoding added to its ``dropped`` count."""
    if not unencodable:
        return final
    return {**final, "dropped": int(final.get("dropped") or 0) + unencodable}


#: Yielded by ``_stream_events`` (when asked) after an idle stretch.
KEEPALIVE = object()


def started_event(event: Dict[str, Any], workflow_id: str, thread_id: str) -> Any:
    """The stream's first event: the ids a client needs to find or stop its run."""
    return StreamEvent(
        workflow_id=workflow_id,
        step=-1,
        node="",
        attempt_id="",
        seq=0,
        ns=(),
        mode=WORKFLOW_STARTED,
        data={
            "workflow_id": event.get("workflow_id") or workflow_id,
            "thread_id": event.get("thread_id") or thread_id,
        },
    )


def without_executed(event: Dict[str, Any]) -> Dict[str, Any]:
    """``event`` minus the per-step node list, which is for the stream only."""
    if EXECUTED_KEY not in event:
        return event
    return {k: v for k, v in event.items() if k != EXECUTED_KEY}


async def next_item(sink: Any, keepalive: bool) -> Any:
    """The next item of ``sink``, or :data:`KEEPALIVE` after an idle stretch."""
    if not keepalive:
        return await sink.get()
    try:
        return await asyncio.wait_for(sink.get(), stream_module.SSE_KEEPALIVE_SECONDS)
    except asyncio.TimeoutError:
        return KEEPALIVE


def failure_event(
    error: BaseException, error_type: Optional[str] = None
) -> Dict[str, Any]:
    """A ``workflow_failed`` event for ``error``.

    ``error_type`` replaces the exception's own type name; it is how a stream
    failure says it is not the run's (see ``PollError`` and ``StreamError``).
    """
    return {
        "type": "workflow_failed",
        "error": {
            "message": str(error),
            "error_type": error_type or type(error).__name__,
        },
    }


def route_astream_item(
    item: Any,
    modes: List[str],
    is_list: bool,
    on_control: Optional[Callable[[StreamControl], None]],
) -> Tuple[List[Any], Optional[GraphRunError]]:
    """What ``astream`` yields for one item of the event stream, and its error.

    Hands control events to ``on_control``. The error is a
    :class:`GraphRunError` for a terminal ``workflow_failed``: the caller
    yields what came with it (a ``stream_dropped`` of a failed run, say) and
    then raises it.
    """
    out: List[Any] = []
    if isinstance(item, StreamEvent):
        if item.mode in CONTROL_MODES:
            control = StreamControl(item.mode, item.data)
            if on_control is not None:
                on_control(control)
            # workflow_started only informs on_control: it is not something the
            # stream modes of native astream have.
            if is_list and item.mode != WORKFLOW_STARTED:
                out.append((control.kind, control.data))
            elif item.mode == NODE_RESET and not is_list and on_control is None:
                warn_reset_unseen(item.workflow_id, control.data)
        elif item.mode in modes:
            out.append((item.mode, item.data) if is_list else item.data)
        return out, None
    if item.get("dropped"):
        control = StreamControl(STREAM_DROPPED, {"dropped": item["dropped"]})
        if on_control is not None:
            on_control(control)
        if is_list:
            out.append((control.kind, control.data))
    if item.get("type") == "workflow_failed":
        return out, GraphRunError(item)
    return out, None


def closing_events(
    closed: StreamClosed, seen_nodes: Set[Tuple[int, str]], workflow_id: str
) -> List[Any]:
    """The items a stream ends with: ``stream_unavailable`` if due, then the final."""
    out: List[Any] = []
    unavailable = stream_unavailable_event(closed.final, seen_nodes, workflow_id)
    if unavailable is not None:
        logger.warning(
            "Streamed run %s: no chunks for steps %s (nodes %s)",
            workflow_id,
            unavailable.data["missing_steps"],
            [m["node"] for m in unavailable.data["missing_nodes"]],
        )
        out.append(unavailable)
    final = without_executed(closed.final)
    out.append({**final, "dropped": closed.dropped} if closed.dropped else final)
    return out
