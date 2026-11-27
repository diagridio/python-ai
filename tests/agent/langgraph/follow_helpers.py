# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A log follower fed by hand: units written with real writers, read as SSE."""

import asyncio
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from diagrid.agent.core.workflow.stream import InProcessSink, StreamEvent
from diagrid.agent.core.workflow.stream_cursor import StreamCursor
from diagrid.agent.core.workflow.stream_log import (
    LogWriter,
    StateLogStreamConfig,
    StateLogTransport,
)
from diagrid.agent.langgraph.stream_drain import SseEncoder, drain
from diagrid.agent.langgraph.stream_follow import (
    LogFollower,
    decode_wire,
    encode_wire,
)
from diagrid.agent.langgraph.stream_source import StreamSource

from ..core.log_store_fake import FakeLogStore

WF = "wf"
MODES = ["custom", "updates"]
#: Two chunks per batch: a unit's first chunks are in the log before it ends.
CONFIG = StateLogStreamConfig(flush_interval_s=60, flush_max_chunks=2, read_poll_s=0.01)

Frame = Tuple[Any, ...]


def transport(store: FakeLogStore) -> StateLogTransport:
    return StateLogTransport(CONFIG, store=store).with_codec(encode_wire, decode_wire)


def chunk(step: int, node: str, mode: str, data: Any, seq: int) -> StreamEvent:
    return StreamEvent(
        workflow_id=WF,
        step=step,
        node=node,
        attempt_id="",
        seq=seq,
        ns=(),
        mode=mode,
        data=data,
    )


def write(
    log: StateLogTransport, step: int, node: str, items: List[Tuple[str, Any]]
) -> LogWriter:
    """A writer of ``(step, node)`` that put ``items``; finish it to end the unit."""
    writer = log.writer(WF, step, node, streamed=True)
    assert writer is not None
    for seq, (mode, data) in enumerate(items):
        writer.put(chunk(step, node, mode, data, seq))
    return writer


class FollowedStream:
    """A stream read from the log, as SSE ``(event, data, id)`` items."""

    def __init__(
        self,
        log: StateLogTransport,
        status: Callable[[], Dict[str, Any]],
        *,
        cursor: Optional[StreamCursor] = None,
        poll_s: float = 0.01,
    ) -> None:
        sink = InProcessSink(WF, asyncio.get_running_loop(), bounded=False)
        self.follower = LogFollower(
            log.reader(WF),
            sink,
            workflow_id=WF,
            status=status,
            poll_s=poll_s,
            cursor=cursor,
            replay=cursor is not None,
        )
        task = asyncio.create_task(self.follower.run())
        self._source = StreamSource(
            sink, self.follower.finish, follower=self.follower, task=task
        )
        self._encoder = SseEncoder(MODES, True)
        self._items = drain(
            self._source,
            workflow_id=WF,
            seen=set(),
            keepalive=False,
            output_channels=None,
        )
        self._frames = self._encoded()

    async def _encoded(self) -> AsyncIterator[Frame]:
        async for item in self._items:
            out = self._encoder.encode(item)
            if out is not None:
                yield out

    async def take(self, n: int) -> List[Frame]:
        """The next ``n`` frames."""
        return [await self._frames.__anext__() for _ in range(n)]

    async def rest(self) -> List[Frame]:
        """Every frame up to and including the final event."""
        return [frame async for frame in self._frames]

    async def close(self) -> None:
        await self._frames.aclose()
        await self._items.aclose()
        await self._source.close()
