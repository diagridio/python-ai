# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Transport for streaming chunks out of workflow activities.

Activities run in worker threads; the HTTP connection that wants the chunks
lives on the asyncio loop of the API. A :class:`StreamTransport` hands each
activity execution a writer, and the serving side reads what it wrote. Two
exist: :class:`InProcessTransport` (activity and API share a process, a single
replica) and the state-store event log (``stream_log.StateLogTransport``),
which any replica can read and resume from.
"""

import asyncio
import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)

# Upper bound on buffered chunks per run. Past it, token and custom chunks are
# dropped (counted, and reported on the final event) so a slow consumer cannot
# grow memory without limit. Updates, node resets and the close marker are never
# dropped: they cannot be re-sent and are bounded by the size of the graph
# (the workflow_started event is sent once).
MAX_BUFFERED_EVENTS = 10_000

UPDATES = "updates"
#: The graph's state after a step. Every activity of the next step is handed
#: that state and offers it; the sink keeps the first offer of each step.
VALUES = "values"
NODE_RESET = "node_reset"
STREAM_UNAVAILABLE = "stream_unavailable"
STREAM_DROPPED = "stream_dropped"
#: The run's event log outlived its TTL: what is left cannot be replayed.
STREAM_PURGED = "stream_purged"
#: First event of a stream: the ids a client needs to find or stop its run.
WORKFLOW_STARTED = "workflow_started"
#: The run ended but its thread's state could not be saved (the output stands).
THREAD_NOT_SAVED = "thread_not_saved"
# Events about the stream itself, delivered whatever stream modes were asked for.
CONTROL_MODES = (
    WORKFLOW_STARTED,
    NODE_RESET,
    STREAM_UNAVAILABLE,
    STREAM_PURGED,
    THREAD_NOT_SAVED,
)
#: Internal marker: a conditional edge's function finished streaming here. It
#: tells the stream which routers ran in this process and never reaches a client.
BRANCH_DONE = "branch_done"
# What a full buffer never drops: the control events, node updates and router
# markers, which cannot be re-sent either.
_KEPT_WHEN_FULL = (*CONTROL_MODES, UPDATES, BRANCH_DONE)

#: Seconds an idle SSE stream waits before it sends a keepalive comment, so a
#: proxy or load balancer does not cut it during a long tool call or LLM gap.
SSE_KEEPALIVE_SECONDS = 15.0
#: The ``event`` name a runner yields for a keepalive; framed as an SSE comment.
KEEPALIVE_EVENT = "keepalive"

# Runs remembered by StreamSinkRegistry.first_report; older ones are forgotten.
_MAX_REPORTED_RUNS = 1024


def display_step(step: int) -> int:
    """A workflow step number as a client sees it.

    Workflow steps count from 0. Clients match ``node_reset`` and
    ``stream_unavailable`` against the ``langgraph_step`` of a chunk's metadata,
    which Pregel counts from 1, so every step a client sees is numbered from 1.
    """
    return step + 1


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
    #: The node name the chunk's metadata carries (``langgraph_node``), when
    #: ``node`` is a key of the sink's own (a ``Send`` task or a router).
    display_node: Optional[str] = None
    #: The task's ``langgraph_path`` when it is not the node's own (a ``Send``).
    path: Optional[Tuple[Any, ...]] = None
    #: The conditional edge whose function sent the chunk, so a retry of that
    #: function is not taken for a retry of the node itself.
    branch: Optional[str] = None
    #: The SSE id of the stream after this event, when the transport can
    #: resume from it (the state-store log); ``None`` otherwise.
    event_id: Optional[str] = None

    def reset_data(self) -> Dict[str, Any]:
        """The ``node_reset`` payload for a retry of this event's task.

        It names the node and step as the chunks do, so a client can match them
        (plus the path of a ``Send`` task, as the chunk's metadata has it).
        """
        data: Dict[str, Any] = {
            "node": self.display_node or self.node,
            "step": display_step(self.step),
        }
        if self.path:
            data["path"] = list(self.path)
        if self.branch:
            data["branch"] = self.branch
        return data


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

    A ``values`` chunk is the same whoever sends it, so the first one for a key
    is kept and any later one dropped, with no ``node_reset``.
    """

    def __init__(
        self,
        workflow_id: str,
        loop: asyncio.AbstractEventLoop,
        *,
        bounded: bool = True,
    ) -> None:
        self.workflow_id = workflow_id
        #: ``False`` when the writer waits for room itself (a log reader), so
        #: nothing is ever dropped here.
        self._bounded = bounded
        self._loop = loop
        self._queue: "asyncio.Queue[Any]" = asyncio.Queue()
        self._lock = threading.Lock()
        self._owner: Dict[Tuple[int, str], str] = {}
        self._finished: Dict[Tuple[int, str], str] = {}
        self._retired: set = set()
        self._values: set = set()
        self._closed = False
        self._dropped = 0

    def put(self, event: StreamEvent) -> None:
        key = (event.step, event.node)
        with self._lock:
            if self._closed:
                return
            if event.mode == VALUES:
                if key not in self._values:
                    self._values.add(key)
                    self._enqueue(event)
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
                        event.reset_data(),
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

    def backlog(self) -> int:
        """Items waiting for the consumer."""
        return self._queue.qsize()

    def prime(self, step: int, node: str, attempt_id: str) -> None:
        """Take ``attempt_id`` as the owner of ``(step, node)`` without a chunk.

        A stream resumed from a cursor uses it for an attempt the client saw
        part of: the next attempt's first chunk then sends ``node_reset``.
        """
        with self._lock:
            self._owner.setdefault((step, node), attempt_id)

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
            elif (
                self._bounded
                and self._queue.qsize() >= MAX_BUFFERED_EVENTS
                and not _is_kept_when_full(item)
            ):
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


def is_kept_when_full(mode: str) -> bool:
    """Whether chunks of ``mode`` must survive a full buffer: they cannot be re-sent."""
    return mode in _KEPT_WHEN_FULL


def _is_kept_when_full(item: Any) -> bool:
    return isinstance(item, StreamEvent) and is_kept_when_full(item.mode)


def _remember(runs: "OrderedDict[str, None]", workflow_id: str) -> None:
    """Record ``workflow_id`` as the latest run, forgetting the oldest past the bound."""
    runs[workflow_id] = None
    runs.move_to_end(workflow_id)
    while len(runs) > _MAX_REPORTED_RUNS:
        runs.popitem(last=False)


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
        self._closed: "OrderedDict[str, None]" = OrderedDict()

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
            self._closed.pop(workflow_id, None)
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
            _remember(self._closed, workflow_id)
        current.close()

    def was_closed(self, workflow_id: str) -> bool:
        """True when this process opened a stream for ``workflow_id`` and closed it.

        Tells a run whose consumer left (or whose stream ended) from one whose
        activity runs in a process that never had a sink. Remembers the most
        recent closes only.
        """
        with self._lock:
            return workflow_id in self._closed

    def first_report(self, workflow_id: str) -> bool:
        """True the first time it is asked about ``workflow_id``.

        Lets an activity log a problem once per run instead of once per node.
        Remembers the most recent runs only, so it cannot grow without limit.
        """
        with self._lock:
            if workflow_id in self._reported:
                return False
            _remember(self._reported, workflow_id)
            return True

    def get(self, workflow_id: Optional[str]) -> Optional[InProcessSink]:
        """The open sink for ``workflow_id``, or ``None``."""
        if not workflow_id:
            return None
        with self._lock:
            return self._sinks.get(workflow_id)


#: Fallback for code that was not handed a registry.
default_registry = StreamSinkRegistry()


class ActivityStreamWriter(Protocol):
    """What one activity execution writes its chunks to."""

    def put(self, event: StreamEvent) -> None: ...

    def finish(self, *, done: bool) -> None:
        """The execution ended; ``done`` when it is the task's last attempt."""
        ...


class StreamTransport(Protocol):
    """How a streamed run's chunks get from its activities to its stream."""

    #: Whether a stream can be read again from a cursor (reconnect, replay).
    replayable: bool

    def writer(
        self, workflow_id: str, step: int, unit: str, *, streamed: bool
    ) -> Optional[ActivityStreamWriter]:
        """The writer for one execution of ``unit`` at ``step``, or ``None``."""
        ...


class _SinkWriter:
    """An in-process sink as an activity's writer: nothing to flush."""

    def __init__(self, sink: InProcessSink) -> None:
        self._sink = sink

    def put(self, event: StreamEvent) -> None:
        self._sink.put(event)

    def finish(self, *, done: bool) -> None:
        return None


class InProcessTransport:
    """Chunks go to the sink a stream opened in this process, if there is one.

    Works while the activities run in the process that serves the stream (a
    single replica); a chunk written anywhere else is lost, and the stream
    says so with ``stream_unavailable``. Nothing can be replayed.
    """

    replayable = False

    def __init__(self, registry: Optional[StreamSinkRegistry] = None) -> None:
        self.registry = registry if registry is not None else default_registry

    def writer(
        self, workflow_id: str, step: int, unit: str, *, streamed: bool
    ) -> Optional[ActivityStreamWriter]:
        sink = self.registry.get(workflow_id)
        if sink is None:
            if streamed:
                _warn_sink_missing(self.registry, workflow_id, unit)
            return None
        return _SinkWriter(sink)


def _warn_sink_missing(
    registry: StreamSinkRegistry, workflow_id: str, unit: str
) -> None:
    """Say, once per run, that a streamed run's chunks are going nowhere."""
    if not registry.first_report(workflow_id):
        return
    if registry.was_closed(workflow_id):
        # This process had the stream and it ended (the client left, or the
        # run's stream was closed): not a second replica, nothing to act on.
        logger.debug(
            "Run %s: its stream was closed in this process (its consumer left "
            "or the stream ended); chunks of %r are not delivered",
            workflow_id,
            unit,
        )
        return
    logger.warning(
        "Run %s is streamed but this process has no stream sink for it, so the "
        "chunks of %r (and of any node that runs here) are not delivered. "
        "The activity is running in another process (several replicas) or the "
        "run outlived the process that streams it. Give the app a state store "
        "to stream across replicas; the run itself is unaffected.",
        workflow_id,
        unit,
    )


def get_stream_sink(
    workflow_id: Optional[str], *, registry: Optional[StreamSinkRegistry] = None
) -> Optional[StreamSink]:
    """Sink for the run an activity belongs to, or ``None`` when not streaming.

    Seam for the multi-replica case: a remote sink would be returned here when
    the run's config names another replica as the stream target.
    """
    return (registry or default_registry).get(workflow_id)


def format_sse(event: str, data: Any, event_id: Optional[Any] = None) -> str:
    """Frame one Server-Sent Event (a keepalive is a comment, not an event)."""
    if event == KEEPALIVE_EVENT:
        return ": keepalive\n\n"
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    payload = json.dumps(data, default=str)
    # json.dumps escapes newlines, so the payload is always one line.
    lines.append(f"data: {payload}")
    return "\n".join(lines) + "\n\n"


async def frame_sse(events: AsyncIterator[Tuple[Any, ...]]) -> AsyncIterator[str]:
    """Frame ``(event, data)`` or ``(event, data, id)`` items as SSE.

    The only place a streamed run is turned into SSE frames, so a runner just
    yields pairs. An item that brings its own id (a cursor a client can resume
    from) is sent with it; otherwise events are numbered on this connection.
    Publishing is the runner's job (``on_result``).
    """
    counter = 0
    try:
        async for item in events:
            name, data = item[0], item[1]
            given = item[2] if len(item) > 2 else None
            if name != KEEPALIVE_EVENT:
                counter += 1
            yield format_sse(name, data, given if given is not None else counter)
    finally:
        # A client that disconnects ends the generator at a yield; close the
        # runner's stream now rather than when it is collected.
        aclose = getattr(events, "aclose", None)
        if aclose is not None:
            await aclose()
