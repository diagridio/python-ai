# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Draining a stream's sink, and turning what it yields into SSE items.

Shared by a new run's stream (``POST /agent/run``, ``astream``) and a
reconnect (``GET /agent/run/{id}/stream``, ``astream_resume``).
"""

from dataclasses import replace
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple

from diagrid.agent.core.workflow.stream import (
    BRANCH_DONE,
    CONTROL_MODES,
    KEEPALIVE_EVENT,
    UPDATES,
    VALUES,
    WORKFLOW_STARTED,
    StreamClosed,
    StreamEvent,
)
from diagrid.agent.core.workflow.stream_cursor import END_ID, START_ID

from .models import GraphWorkflowOutput
from .stream_events import (
    KEEPALIVE,
    closing_events,
    count_unencodable,
    encode_chunk,
    next_item,
)
from .stream_source import StreamSource
from .streaming import BRANCHES_KEY, EXECUTED_KEY, FINAL_STATE_KEY


def parse_output(
    workflow_id: str, output_dict: Dict[str, Any], streamed: bool
) -> Dict[str, Any]:
    """The ``workflow_completed`` event of a finished workflow's output.

    Which nodes ran in each step, the routers and the state it ended with are
    for the stream (``stream_unavailable``, the last ``values``), not for a
    caller of ``run_async``: only a streamed run gets them.
    """
    output = GraphWorkflowOutput.from_dict(output_dict)
    return {
        "type": "workflow_completed",
        "workflow_id": workflow_id,
        "output": output.output,
        "steps": output.steps,
        "status": output.status,
        **({"error": output.error} if output.error else {}),
        **(
            {"thread_saved": False, "thread_error": output.thread_error}
            if output.thread_saved is False
            else {}
        ),
        **({EXECUTED_KEY: output.executed} if streamed and output.executed else {}),
        **({BRANCHES_KEY: output.branches} if streamed and output.branches else {}),
        **(
            {FINAL_STATE_KEY: output.channel_state.to_dict()}
            if streamed and output.channel_state is not None
            else {}
        ),
    }


async def drain(
    source: StreamSource,
    *,
    workflow_id: str,
    seen: Set[Tuple[int, str]],
    keepalive: bool,
    output_channels: Optional[List[str]],
) -> AsyncGenerator[Any, None]:
    """Yield the sink's chunks, then the items the stream ends with.

    ``seen`` collects the ``(step, node)`` pairs whose ``updates`` (or router
    ``branch_done`` marker) arrived, for ``stream_unavailable``.
    """
    values_sent: Set[int] = set()
    follower = source.follower
    while True:
        item = await next_item(source.sink, keepalive)
        if isinstance(item, StreamClosed):
            # Unregister before handing over the result: chunks a late
            # activity attempt writes from here on are dropped.
            await source.close()
            if follower is not None:  # the client had these before a reconnect
                values_sent.update(range(follower.values_through + 1))
            for event in closing_events(
                item,
                seen,
                workflow_id,
                output_channels=output_channels,
                values_sent=values_sent,
                unavailable_reason=source.unavailable_reason,
                purged=source.purged,
            ):
                if (
                    follower is not None
                    and isinstance(event, StreamEvent)
                    and event.mode == VALUES
                ):
                    # its own id: a reconnect from it does not send it again
                    event = replace(event, event_id=follower.event_id(event.step))
                yield event
            return
        if isinstance(item, StreamEvent) and item.mode == BRANCH_DONE:
            # a router that ran here: internal, never sent on
            seen.add((item.step, item.node))
            continue
        if isinstance(item, StreamEvent) and item.mode == UPDATES:
            seen.add((item.step, item.node))
        if isinstance(item, StreamEvent) and item.mode == VALUES:
            values_sent.add(item.step)
        yield item


class SseIds:
    """The SSE id of each event of a stream read from the log: its cursor.

    A stream on the in-process transport has no cursor: its events are
    numbered per connection (``()`` leaves that to the framer).
    """

    def __init__(self, replayable: bool) -> None:
        self._replayable = replayable
        self._last = START_ID

    def of(self, item: StreamEvent) -> Tuple[str, ...]:
        if not self._replayable:
            return ()
        if item.mode == WORKFLOW_STARTED:
            return (START_ID,)
        if item.event_id:
            self._last = item.event_id
        # a control event reuses the cursor of the chunk it follows
        return (self._last,)

    def final(self) -> Tuple[str, ...]:
        return (END_ID,) if self._replayable else ()


class SseEncoder:
    """``(event, data[, id])`` for each item of a stream, as ``frame_sse`` takes it."""

    def __init__(self, modes: List[str], replayable: bool) -> None:
        self._modes = modes
        self._ids = SseIds(replayable)
        self.unencodable = 0
        self.workflow_id: Optional[str] = None

    def encode(self, item: Any) -> Optional[Tuple[Any, ...]]:
        """The SSE item for ``item``, or ``None`` for one the client did not ask for."""
        if item is KEEPALIVE:
            return (KEEPALIVE_EVENT, None)
        if isinstance(item, StreamEvent):
            event_id = self._ids.of(item)
            if item.mode not in CONTROL_MODES and item.mode not in self._modes:
                return None
            if item.mode == WORKFLOW_STARTED:
                self.workflow_id = item.data["workflow_id"]
            wire, ok = encode_chunk(item)
            self.unencodable += 0 if ok else 1
            return (item.mode, wire, *event_id)
        final = count_unencodable(item, self.unencodable)
        return (item["type"], final, *self._ids.final())
