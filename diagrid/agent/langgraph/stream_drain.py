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
from diagrid.agent.core.workflow.stream_cursor import (
    END_ID,
    START_ID,
    encode_cursor,
    parse_cursor,
)

from .models import GraphWorkflowOutput
from .stream_events import (
    KEEPALIVE,
    STILL_RUNNING_KEY,
    closing_events,
    count_unencodable,
    encode_chunk,
    next_item,
)
from .stream_attempts import STATUS_ATTEMPTS
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
        **({STATUS_ATTEMPTS: output.attempts} if streamed and output.attempts else {}),
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

    def __init__(self, replayable: bool, resume_from: Optional[str] = None) -> None:
        self._replayable = replayable
        #: Where the client is: the cursor it reconnected with, until a chunk.
        self._last = (
            encode_cursor(parse_cursor(resume_from)) if resume_from else START_ID
        )

    def of(self, item: StreamEvent) -> Tuple[str, ...]:
        if not self._replayable:
            return ()
        if item.mode == WORKFLOW_STARTED:
            return (START_ID,)
        if item.event_id:
            self._last = item.event_id
        # a control event reuses the cursor of the chunk it follows
        return (self._last,)

    def final(self, event: Dict[str, Any]) -> Tuple[str, ...]:
        """The id of the event a stream ends with.

        ``v1.end`` says the client has the run's end. A stream that failed
        while the run goes on has not: its failure repeats the last chunk's
        cursor, so a reconnect with it (what an ``EventSource`` does) gets the
        rest of the run.
        """
        if not self._replayable:
            return ()
        return (self._last,) if event.get(STILL_RUNNING_KEY) else (END_ID,)


class SseEncoder:
    """``(event, data[, id])`` for each item of a stream, as ``frame_sse`` takes it.

    ``resume_from`` is the id a reconnecting client sent.
    """

    def __init__(
        self, modes: List[str], replayable: bool, resume_from: Optional[str] = None
    ) -> None:
        self._modes = modes
        self._ids = SseIds(replayable, resume_from)
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
        return self.encode_final(item)

    def encode_final(self, event: Dict[str, Any]) -> Tuple[Any, ...]:
        """The SSE item of the event a stream ends with."""
        final = count_unencodable(event, self.unencodable)
        return (event["type"], final, *self._ids.final(event))


def encode_final(
    encoder: Optional[SseEncoder], event: Dict[str, Any]
) -> Tuple[Any, ...]:
    """The SSE item of a final event sent when the stream itself raised.

    With an encoder it carries the id the stream got to (see
    :meth:`SseIds.final`); a stream refused before it had one numbers it.
    """
    if encoder is None:
        return (event["type"], event)
    return encoder.encode_final(event)
