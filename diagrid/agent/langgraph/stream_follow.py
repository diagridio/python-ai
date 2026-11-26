# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Reading a run's stream back from the state-store event log.

The follower finds the run's units in its custom status (the final event once
the run ended), reads what is new in each of them, and puts the chunks into
an in-process sink the stream is drained from, so a stream read from the log
is consumed exactly like one fed by activities in this process. It reads at
the consumer's pace: a slow client lags, it does not lose chunks.

Each chunk carries the cursor after it as its SSE id (see ``stream_cursor``).
"""

import asyncio
import logging
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from diagrid.agent.core.workflow.stream import (
    STREAM_PURGED,
    VALUES,
    InProcessSink,
    StreamEvent,
)
from diagrid.agent.core.workflow.stream_cursor import (
    ATTEMPT_PREFIX,
    StreamCursor,
    encode_cursor,
)
from diagrid.agent.core.workflow.stream_log import (
    LogReader,
    UnitChunk,
    UnitWant,
    event_from_wire,
)

from .activity_stream import STATUS_BRANCHES, STATUS_EXECUTED
from .stream_events import encode_chunk, stream_failure_event
from .streaming import BRANCHES_KEY, EXECUTED_KEY, SUPPORTED_MODES

logger = logging.getLogger(__name__)

#: The follower stops reading while this many chunks wait for the consumer.
BACKLOG_HIGH = 256
#: Reads that fail in a row before the stream reports the store unreadable.
READ_ATTEMPTS = 5

Unit = Tuple[int, str]


def encode_wire(event: StreamEvent) -> Any:
    """A chunk's data as the log keeps it: its wire form (see ``to_wire``)."""
    return encode_chunk(event)[0]


def decode_wire(mode: str, data: Any) -> Any:
    """A chunk's data as an in-process consumer gets it, from its wire form."""
    if mode not in SUPPORTED_MODES:
        return data
    from langchain_core._api.beta_decorator import suppress_langchain_beta_warning
    from langchain_core.load import load

    try:
        with suppress_langchain_beta_warning():
            value = load(data)
    except Exception:  # not something load() knows: keep it as sent
        return data
    if mode == "messages" and isinstance(value, list) and len(value) == 2:
        return tuple(value)
    return value


def units_of(status: Dict[str, Any]) -> List[Unit]:
    """The run's units in replay order: by step, nodes first, then routers.

    ``status`` is the custom status or the final event (same keys).
    """
    executed = status.get(STATUS_EXECUTED) or status.get(EXECUTED_KEY) or []
    branches = status.get(STATUS_BRANCHES) or status.get(BRANCHES_KEY) or []
    units: List[Tuple[int, int, int, str]] = []
    for step, keys in enumerate(executed):
        for order, key in enumerate(keys):
            units.append((step, 0, order, str(key)))
    for order, (step, key) in enumerate(branches):
        units.append((int(step), 1, order, str(key)))
    seen: Set[Unit] = set()
    ordered: List[Unit] = []
    for step, _, _, key in sorted(units):
        if (step, key) not in seen:
            seen.add((step, key))
            ordered.append((step, key))
    return ordered


@dataclass
class _UnitState:
    """How far the stream got in one unit."""

    attempt: Optional[str] = None
    from_batch: int = 0
    #: Events of the attempt handed to the sink (or read by an earlier client).
    consumed: int = 0
    #: Events to skip on the first read: the client got them before.
    skip: int = 0
    done: bool = False
    total: int = 0
    #: The attempt an earlier client saw part of, when it was replaced.
    replaced: Optional[str] = None

    @property
    def complete(self) -> bool:
        return self.done and self.consumed >= self.total


class LogFollower:
    """Feeds ``sink`` from the run's log until the run ends, then closes it."""

    def __init__(
        self,
        reader: LogReader,
        sink: InProcessSink,
        *,
        workflow_id: str,
        status: Callable[[], Dict[str, Any]],
        poll_s: float,
        cursor: Optional[StreamCursor] = None,
        decode: Callable[[str, Any], Any] = decode_wire,
        seen: Optional[Set[Unit]] = None,
    ) -> None:
        self._reader = reader
        self._sink = sink
        self._workflow_id = workflow_id
        self._status = status
        self._poll_s = poll_s
        self._cursor = cursor or StreamCursor()
        self._decode = decode
        self._units: Dict[Unit, _UnitState] = {}
        self._order: List[Unit] = []
        #: Units the client had in full or in part before this stream: the
        #: stream does not report them missing.
        self._seen = seen if seen is not None else set()
        self.purged = False
        self._final: "asyncio.Future[Dict[str, Any]]" = (
            asyncio.get_running_loop().create_future()
        )

    def finish(self, final: Dict[str, Any]) -> None:
        """The run ended with ``final``: read what is left, then close the sink."""
        if not self._final.done():
            self._final.set_result(final)

    async def run(self) -> None:
        failures = 0
        while True:
            final = self._final.result() if self._final.done() else None
            try:
                await self._read_once(final)
                failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as e:
                failures += 1
                logger.warning(
                    "Reading the stream log of %s failed (%d/%d): %s",
                    self._workflow_id,
                    failures,
                    READ_ATTEMPTS,
                    e,
                )
                if failures >= READ_ATTEMPTS:
                    self._sink.close(stream_failure_event(e, "StreamReadError", True))
                    return
                await asyncio.sleep(self._poll_s)
                continue
            if final is not None:
                self._sink.close(final)
                return
            try:
                await asyncio.wait_for(asyncio.shield(self._final), self._poll_s)
            except asyncio.TimeoutError:
                pass

    async def _read_once(self, final: Optional[Dict[str, Any]]) -> None:
        if self.purged or self._cursor.at_end:
            return
        status = final if final is not None else await asyncio.to_thread(self._status)
        self._learn(units_of(status or {}))
        wants = [
            UnitWant(
                unit[0],
                unit[1],
                attempt=state.attempt,
                from_batch=state.from_batch,
                skip_events=state.skip,
            )
            for unit in self._order
            if not (state := self._units[unit]).complete
        ]
        if not wants:
            return
        chunks = await asyncio.to_thread(self._reader.read, wants)
        for chunk in chunks:
            if chunk.purged:
                self._purge(chunk)
                return
            await self._deliver(chunk)

    def _learn(self, units: List[Unit]) -> None:
        for unit in units:
            if unit in self._units:
                continue
            state = _UnitState()
            if self._cursor.skips_step(unit[0]):
                state = _UnitState(done=True)
                self._seen.add(unit)
            else:
                position = self._cursor.position(*unit)
                if position is not None:
                    state = _UnitState(attempt=position[0], skip=position[1])
                    state.consumed = position[1]
                    self._seen.add(unit)
            self._units[unit] = state
            self._order.append(unit)
        self._order.sort(key=lambda u: u[0])

    def _purge(self, chunk: UnitChunk) -> None:
        self.purged = True
        logger.warning(
            "The stream log of %s is purged (unit %s of step %s): it outlived its TTL",
            self._workflow_id,
            chunk.unit,
            chunk.step,
        )
        self._sink.put(
            StreamEvent(
                workflow_id=self._workflow_id,
                step=chunk.step,
                node="",
                attempt_id="",
                seq=0,
                ns=(),
                mode=STREAM_PURGED,
                data={"workflow_id": self._workflow_id, "step": chunk.step + 1},
                event_id=self.event_id(),
            )
        )

    async def _deliver(self, chunk: UnitChunk) -> None:
        state = self._units[(chunk.step, chunk.unit)]
        if chunk.attempt is None:
            return
        if chunk.reset:
            # an attempt the client saw part of was replaced: its first chunk
            # in this stream must tell the client to drop that part
            had = state.consumed > 0
            state.replaced = state.attempt if had else None
        primed: Set[Tuple[int, str]] = set()
        state.attempt = chunk.attempt
        state.from_batch = chunk.batches
        state.skip = 0
        state.done = chunk.done
        state.total = chunk.total
        state.consumed = chunk.start
        for wire in chunk.events:
            event = event_from_wire(wire, self._workflow_id, self._decode)
            await self._wait_for_room()
            key = (event.step, event.node)
            if state.replaced and event.mode != VALUES and key not in primed:
                self._sink.prime(event.step, event.node, state.replaced)
                primed.add(key)
            state.consumed += 1
            self._sink.put(replace(event, event_id=self.event_id()))
        state.replaced = None

    async def _wait_for_room(self) -> None:
        while self._sink.backlog() >= BACKLOG_HIGH:
            await asyncio.sleep(0.01)

    def event_id(self) -> str:
        """The SSE id of the stream as it stands: a cursor any replica resumes."""
        open_steps = [u[0] for u in self._order if not self._units[u].complete]
        if open_steps:
            base = min(open_steps)
        elif self._order:
            base = max(u[0] for u in self._order)
        else:
            return encode_cursor(self._cursor)
        positions = tuple(
            (step, unit, (state.attempt or "")[:ATTEMPT_PREFIX], state.consumed)
            for (step, unit) in self._order
            if step >= base and (state := self._units[(step, unit)]).consumed > 0
        )
        return encode_cursor(StreamCursor(base=base, positions=positions))
