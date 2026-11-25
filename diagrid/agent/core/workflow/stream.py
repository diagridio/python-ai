# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Transport for streaming chunks out of workflow activities.

Activities run in worker threads; the HTTP connection that wants the chunks
lives on the asyncio loop of the API. A :class:`StreamSink` sits between them.
Only the in-process sink exists today (single replica: activity and API share
a process). :func:`get_stream_sink` is the seam for a later cross-replica sink:
it is the only place an activity looks a sink up.
"""

import asyncio
import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)

# Upper bound on buffered chunks per run. Past it, token and custom chunks are
# dropped (counted, and reported on the final event) so a slow consumer cannot
# grow memory without limit. Updates, node resets and the close marker are never
# dropped: they cannot be re-sent and are bounded by the size of the graph.
MAX_BUFFERED_EVENTS = 10_000

NODE_RESET = "node_reset"
STREAM_UNAVAILABLE = "stream_unavailable"
STREAM_DROPPED = "stream_dropped"
# Events about the stream itself, delivered whatever stream modes were asked for.
CONTROL_MODES = (NODE_RESET, STREAM_UNAVAILABLE)

# Runs remembered by StreamSinkRegistry.first_report; older ones are forgotten.
_MAX_REPORTED_RUNS = 1024
UPDATES = "updates"


@dataclass(frozen=True)
class StreamEvent:
    """One chunk, wrapped with enough identity to order it and tell attempts apart."""

    workflow_id: str
    step: int
    node: str
    attempt_id: str
    seq: int
    ns: Tuple[str, ...]
    mode: str
    data: Any


class StreamSink(Protocol):
    """Where an activity attempt hands its chunks."""

    def put(self, event: StreamEvent) -> None: ...


class StreamClosed:
    """Last item a sink hands its consumer: the run is over.

    ``final`` is the run's terminal event (``workflow_completed`` or
    ``workflow_failed`` with its payload). ``dropped`` is how many chunks the
    sink discarded because the consumer fell behind.
    """

    def __init__(self, final: Optional[Dict[str, Any]] = None) -> None:
        self.final: Dict[str, Any] = final or {}
        self.dropped = 0


class InProcessSink:
    """Per-run buffer fed from activity threads and drained on an event loop.

    Retry handling: activities are at-least-once. The first attempt to emit
    for a ``(step, node)`` owns it. A chunk from a different attempt switches
    ownership and emits a ``node_reset`` event so consumers can drop the
    partial text; the old attempt's chunks are then ignored. After a node's
    ``updates`` chunk was accepted, chunks from any other attempt are dropped.
    """

    def __init__(self, workflow_id: str, loop: asyncio.AbstractEventLoop) -> None:
        self.workflow_id = workflow_id
        self._loop = loop
        self._queue: "asyncio.Queue[Any]" = asyncio.Queue()
        self._lock = threading.Lock()
        self._owner: Dict[Tuple[int, str], str] = {}
        self._finished: Dict[Tuple[int, str], str] = {}
        self._retired: set = set()
        self._closed = False
        self._dropped = 0

    def put(self, event: StreamEvent) -> None:
        key = (event.step, event.node)
        with self._lock:
            if self._closed:
                return
            if self._finished.get(key, event.attempt_id) != event.attempt_id:
                return
            if (key, event.attempt_id) in self._retired:
                return
            owner = self._owner.get(key)
            if owner is not None and owner != event.attempt_id:
                self._retired.add((key, owner))
                self._enqueue(
                    StreamEvent(
                        event.workflow_id,
                        event.step,
                        event.node,
                        event.attempt_id,
                        event.seq,
                        event.ns,
                        NODE_RESET,
                        {"node": event.node, "step": event.step},
                    )
                )
            self._owner[key] = event.attempt_id
            if event.mode == UPDATES:
                self._finished[key] = event.attempt_id
            self._enqueue(event)

    def close(self, final: Optional[Dict[str, Any]] = None) -> None:
        """Mark the run finished; the consumer stops after draining."""
        with self._lock:
            if self._closed:
                return
            self._enqueue(StreamClosed(final))
            self._closed = True

    async def get(self) -> Any:
        return await self._queue.get()

    def _enqueue(self, item: Any) -> None:
        def _put() -> None:
            # Runs on the loop thread in the order chunks were handed over, so
            # the count on the close marker covers every chunk before it.
            if isinstance(item, StreamClosed):
                item.dropped = self._dropped
                if self._dropped:
                    logger.warning(
                        "Stream %s dropped %d chunks: the consumer fell behind",
                        self.workflow_id,
                        self._dropped,
                    )
            elif self._queue.qsize() >= MAX_BUFFERED_EVENTS and not _is_control(item):
                if not self._dropped:
                    logger.warning(
                        "Stream buffer for %s is full; dropping chunks",
                        self.workflow_id,
                    )
                self._dropped += 1
                return
            self._queue.put_nowait(item)

        try:
            self._loop.call_soon_threadsafe(_put)
        except RuntimeError:  # loop already closed
            self._closed = True


def _is_control(item: Any) -> bool:
    """Chunks that must not be dropped: they cannot be re-sent."""
    return isinstance(item, StreamEvent) and item.mode in (UPDATES, NODE_RESET)


class StreamSinkRegistry:
    """The sinks of the streams open in this process, keyed by workflow id.

    A runner owns one (see ``DaprWorkflowGraphRunner(stream_registry=...)``) and
    binds it to the activities it registers, so two runners in one process never
    see each other's runs. :data:`default_registry` is what an activity falls
    back to when nothing was bound to it.
    """

    def __init__(self) -> None:
        self._sinks: Dict[str, InProcessSink] = {}
        self._lock = threading.Lock()
        self._reported: "OrderedDict[str, None]" = OrderedDict()

    def open(self, workflow_id: str) -> InProcessSink:
        """Register a sink for ``workflow_id`` on the running event loop.

        Raises ``ValueError`` if a stream for this id is already open, so a
        reused id cannot orphan the first run's sink.
        """
        sink = InProcessSink(workflow_id, asyncio.get_running_loop())
        with self._lock:
            if workflow_id in self._sinks:
                raise ValueError(f"A stream for {workflow_id!r} is already open")
            self._sinks[workflow_id] = sink
        return sink

    def close(self, workflow_id: str, sink: Optional[InProcessSink] = None) -> None:
        """Unregister and close the sink; later chunks for this run are dropped.

        With ``sink`` given, only that sink is removed, never a newer one that
        took over the id.
        """
        with self._lock:
            current = self._sinks.get(workflow_id)
            if current is None or (sink is not None and current is not sink):
                return
            del self._sinks[workflow_id]
        current.close()

    def first_report(self, workflow_id: str) -> bool:
        """True the first time it is asked about ``workflow_id``.

        Lets an activity log a problem once per run instead of once per node.
        Remembers the most recent runs only, so it cannot grow without limit.
        """
        with self._lock:
            if workflow_id in self._reported:
                return False
            self._reported[workflow_id] = None
            while len(self._reported) > _MAX_REPORTED_RUNS:
                self._reported.popitem(last=False)
            return True

    def get(self, workflow_id: Optional[str]) -> Optional[InProcessSink]:
        """The open sink for ``workflow_id``, or ``None``."""
        if not workflow_id:
            return None
        with self._lock:
            return self._sinks.get(workflow_id)


#: Fallback for code that was not handed a registry.
default_registry = StreamSinkRegistry()


def open_stream(
    workflow_id: str, *, registry: Optional[StreamSinkRegistry] = None
) -> InProcessSink:
    """:meth:`StreamSinkRegistry.open` on ``registry`` (default: the module's)."""
    return (registry or default_registry).open(workflow_id)


def close_stream(
    workflow_id: str,
    sink: Optional[InProcessSink] = None,
    *,
    registry: Optional[StreamSinkRegistry] = None,
) -> None:
    """:meth:`StreamSinkRegistry.close` on ``registry`` (default: the module's)."""
    (registry or default_registry).close(workflow_id, sink)


def get_stream_sink(
    workflow_id: Optional[str], *, registry: Optional[StreamSinkRegistry] = None
) -> Optional[StreamSink]:
    """Sink for the run an activity belongs to, or ``None`` when not streaming.

    Seam for the multi-replica case: a remote sink would be returned here when
    the run's config names another replica as the stream target.
    """
    return (registry or default_registry).get(workflow_id)


def format_sse(event: str, data: Any, event_id: Optional[int] = None) -> str:
    """Frame one Server-Sent Event."""
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    payload = json.dumps(data, default=str)
    lines.extend(f"data: {part}" for part in payload.split("\n"))
    return "\n".join(lines) + "\n\n"
