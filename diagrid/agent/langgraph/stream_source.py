# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Where a stream's chunks come from: this process's activities, or the log.

Either way the stream drains one :class:`InProcessSink`. With the in-process
transport the sink is registered and activities in this process write to it;
with the state-store log a :class:`LogFollower` fills it from the log, which
activities on any replica wrote.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Set, Tuple

from diagrid.agent.core.workflow.stream import (
    InProcessSink,
    InProcessTransport,
    StreamTransport,
)
from diagrid.agent.core.workflow.stream_cursor import StreamCursor
from diagrid.agent.core.workflow.stream_log import StateLogTransport

from .activity_stream import parse_status
from .stream_follow import LogFollower

logger = logging.getLogger(__name__)

#: ``stream_unavailable`` reason when the log lacks a unit that ran: its
#: writes failed (or the log was cut short).
WRITE_FAILED = "write_failed"
#: ``stream_unavailable`` reason on a reconnect to a run streamed in process.
NOT_REPLAYABLE = "transport_not_replayable"


@dataclass
class StreamSource:
    """The sink a stream drains and how the run's end reaches it."""

    sink: InProcessSink
    #: Called once with the run's terminal event.
    on_final: Callable[[Dict[str, Any]], None]
    follower: Optional[LogFollower] = None
    task: Optional["asyncio.Task[None]"] = None
    #: Unregisters the sink (in-process transport only).
    release: Optional[Callable[[], None]] = None

    @property
    def unavailable_reason(self) -> Optional[str]:
        return WRITE_FAILED if self.follower is not None else None

    @property
    def purged(self) -> bool:
        return self.follower is not None and self.follower.purged

    async def close(self) -> None:
        if self.release is not None:
            self.release()
        if self.task is not None and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)


def custom_status(client: Any, workflow_id: str) -> Dict[str, Any]:
    """The run's custom status (blocking: call it off the event loop)."""
    state = client.get_workflow_state(instance_id=workflow_id)
    return parse_status(getattr(state, "serialized_custom_status", None))


def open_source(
    transport: StreamTransport,
    workflow_id: str,
    *,
    status: Callable[[], Dict[str, Any]],
    seen: Set[Tuple[int, str]],
    cursor: Optional[StreamCursor] = None,
) -> StreamSource:
    """The source of a stream of ``workflow_id`` on ``transport``.

    Raises ``ValueError`` if an in-process stream of this run is already open.
    """
    if isinstance(transport, StateLogTransport):
        sink = InProcessSink(workflow_id, asyncio.get_running_loop(), bounded=False)
        follower = LogFollower(
            transport.reader(workflow_id),
            sink,
            workflow_id=workflow_id,
            status=status,
            poll_s=transport.config.read_poll_s,
            cursor=cursor,
            decode=transport.decode,
            seen=seen,
        )
        task = asyncio.create_task(follower.run())
        return StreamSource(sink, follower.finish, follower=follower, task=task)
    registry = (
        transport.registry
        if isinstance(transport, InProcessTransport)
        else InProcessTransport().registry
    )
    sink = registry.open(workflow_id)
    return StreamSource(
        sink, sink.close, release=lambda: registry.close(workflow_id, sink)
    )


def resolve_transport(
    given: Optional[StreamTransport],
    registry: Optional[Any],
    store_name: Optional[str],
    own_registry: Any,
) -> StreamTransport:
    """The transport a runner streams through.

    The one given; else the in-process transport over a given registry; else
    the state-store log on the runner's store (any app on Catalyst has one,
    since workflows need it); else in-process. A log transport is bound to
    the LangGraph codec, so chunks keep their message objects.
    """
    from diagrid.agent.core.workflow.stream_log import StateLogStreamConfig

    from .stream_follow import decode_wire, encode_wire

    transport: StreamTransport
    if given is not None:
        transport = given
    elif registry is not None:
        transport = InProcessTransport(registry)
    elif store_name:
        transport = StateLogTransport(StateLogStreamConfig(store_name=store_name))
    else:
        transport = InProcessTransport(own_registry)
    if isinstance(transport, StateLogTransport):
        transport = transport.with_codec(encode_wire, decode_wire)
    return transport
