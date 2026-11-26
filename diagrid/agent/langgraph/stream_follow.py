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

from .activity_stream import STATUS_BRANCHES, STATUS_EXECUTED, STATUS_EXECUTED_FROM
from .interrupts import INTERRUPT_KEY, as_interrupts
from .stream_attempts import attempts_of, unit_key
from .stream_events import encode_chunk, stream_failure_event
from .streaming import BRANCHES_KEY, EXECUTED_KEY, SUPPORTED_MODES

logger = logging.getLogger(__name__)

#: The follower stops reading while this many chunks wait for the consumer.
BACKLOG_HIGH = 256
#: Reads that fail in a row before the stream reports the store unreadable.
READ_ATTEMPTS = 5
#: Polls between reads of the run's status while a unit is still being written.
STATUS_EVERY = 10

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
    if mode == "updates" and isinstance(value, dict) and INTERRUPT_KEY in value:
        return {INTERRUPT_KEY: as_interrupts(value[INTERRUPT_KEY] or ())}
    return value


def units_of(status: Dict[str, Any]) -> List[Unit]:
    """The run's units in replay order: by step, nodes first, then routers.

    ``status`` is the custom status or the final event (same keys).
    """
    executed = status.get(STATUS_EXECUTED) or status.get(EXECUTED_KEY) or []
    branches = status.get(STATUS_BRANCHES) or status.get(BRANCHES_KEY) or []
    first = int(status.get(STATUS_EXECUTED_FROM) or 0)
    units: List[Tuple[int, int, int, str]] = []
    for step, keys in enumerate(executed, start=first):
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


def _same_attempt(a: str, b: str) -> bool:
    """Whether two attempt ids (or a cursor's prefix of one) name one attempt."""
    return bool(a) and bool(b) and (a.startswith(b) or b.startswith(a))


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
    #: The attempt whose output the run used, once the workflow said so.
    winner: Optional[str] = None

    @property
    def complete(self) -> bool:
        # read another attempt, but the run used the winner: read that one
        # (a step a cursor skips whole has no attempt here: it was read)
        if (
            self.winner
            and self.attempt
            and not _same_attempt(self.attempt, self.winner)
        ):
            return False
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
        from_end: bool = False,
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
        #: Polls since the run's status was last read (see ``_run_status``).
        self._polls_since_status = STATUS_EVERY
        #: The last step whose ``values`` chunk went out (here or before a
        #: reconnect): every unit of a step logs it, and it is sent once.
        self.values_through = (
            -1 if self._cursor.values_through is None else self._cursor.values_through
        )
        self.purged = False
        #: Start after what the log holds when the follower starts (a resumed run).
        self._from_end = from_end
        #: Set once the follower knows where it starts reading.
        self.ready = asyncio.Event()
        self._final: "asyncio.Future[Dict[str, Any]]" = (
            asyncio.get_running_loop().create_future()
        )

    def finish(self, final: Dict[str, Any]) -> None:
        """The run ended with ``final``: read what is left, then close the sink."""
        if not self._final.done():
            self._final.set_result(final)

    async def run(self) -> None:
        failures = 0
        if self._from_end:
            try:
                await self.skip_to_end()
            except Exception as e:
                self.ready.set()
                self._sink.close(stream_failure_event(e, "StreamReadError", True))
                return
        self.ready.set()
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

    async def skip_to_end(self) -> None:
        """Take everything the log holds now as read: a stream of what comes next.

        A resumed run streams the rest of the run from here, so this is read
        before the resume is sent.
        """
        status = await asyncio.to_thread(self._status)
        self._learn(units_of(status or {}))
        wants = [UnitWant(step, unit, from_batch=1 << 30) for step, unit in self._order]
        if not wants:
            return
        for chunk in await asyncio.to_thread(self._reader.read, wants):
            state = self._units[(chunk.step, chunk.unit)]
            state.attempt = chunk.attempt
            state.from_batch = chunk.batches
            state.consumed = state.total = chunk.total
            state.done = chunk.done
            self._seen.add((chunk.step, chunk.unit))

    async def _read_once(self, final: Optional[Dict[str, Any]]) -> None:
        if self._cursor.at_end:
            return
        status = await self._run_status(final)
        if status is not None:
            if final is None and self._behind_window(status):
                return  # the final event lists every step: read in order then
            self._learn(units_of(status))
            winners = attempts_of(status)
            for key, state in self._units.items():
                state.winner = winners.get(unit_key(*key), state.winner)
        wants = [
            UnitWant(
                unit[0],
                unit[1],
                attempt=state.attempt,
                from_batch=state.from_batch,
                skip_events=state.skip,
                winner=state.winner,
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
                continue
            await self._deliver(chunk)

    async def _run_status(
        self, final: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """The run's status, when there may be something new in it.

        New units appear when a step starts, and the run says which attempt
        it used when the next one does; while a known unit is still being
        written the status is read only every :data:`STATUS_EVERY` polls (a
        unit whose writes failed must not hold the stream back for good).
        Dapr returns the status only with the run's input and output, so this
        is the expensive read of a poll.
        """
        if final is not None:
            return final
        self._polls_since_status += 1
        writing = any(not state.complete for state in self._units.values())
        if writing and self._polls_since_status < STATUS_EVERY:
            return None
        self._polls_since_status = 0
        return await asyncio.to_thread(self._status) or {}

    def _behind_window(self, status: Dict[str, Any]) -> bool:
        """Whether the status no longer lists steps this reader still needs."""
        first = int(status.get(STATUS_EXECUTED_FROM) or 0)
        if not first or self._cursor.skips_step(first - 1):
            return False
        return not any(step < first for step, _ in self._units)

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
        """A unit whose batches expired: skip what is gone, read on from there.

        A run that waited on ``interrupt()`` past the batch TTL lost its first
        steps' chunks, but what came after the resume is in the log: the
        stream says ``stream_purged`` once and goes on. The unit stays open, so
        a later attempt of it (the resumed task) is read from its start.
        """
        state = self._units[(chunk.step, chunk.unit)]
        state.attempt = chunk.attempt
        state.from_batch = chunk.batches
        state.consumed = state.total = chunk.total
        state.skip = 0
        state.done = chunk.done
        self._seen.add((chunk.step, chunk.unit))
        if self.purged:
            return
        self.purged = True
        logger.warning(
            "The stream log of %s is purged (unit %s of step %s): it outlived its "
            "TTL; streaming what is left",
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
            if had and state.attempt:
                # the sink may have closed the unit on the old attempt's updates
                for node in {str(w.get("node")) for w in chunk.events}:
                    self._sink.reopen(chunk.step, node, state.attempt)
        primed: Set[Tuple[int, str]] = set()
        state.attempt = chunk.attempt
        state.from_batch = chunk.batches
        state.skip = 0
        state.done = chunk.done
        state.total = chunk.total
        state.consumed = chunk.start
        for wire in chunk.events:
            event = event_from_wire(wire, self._workflow_id, self._decode)
            if event.mode == VALUES:
                if event.step <= self.values_through:
                    state.consumed += 1  # another unit's copy already went out
                    continue
                self.values_through = event.step
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

    def event_id(self, values_through: Optional[int] = None) -> str:
        """The SSE id of the stream as it stands: a cursor any replica resumes.

        ``values_through`` is the step of a ``values`` chunk sent outside the
        log (the run's final state), which the id then covers.
        """
        values = max(
            self.values_through, -1 if values_through is None else values_through
        )
        through = None if values < 0 else values
        open_steps = [u[0] for u in self._order if not self._units[u].complete]
        if open_steps:
            base = min(open_steps)
        elif self._order:
            base = max(u[0] for u in self._order)
        else:
            return encode_cursor(replace(self._cursor, values_through=through))
        positions = tuple(
            (step, unit, (state.attempt or "")[:ATTEMPT_PREFIX], state.consumed)
            for (step, unit) in self._order
            if step >= base and (state := self._units[(step, unit)]).consumed > 0
        )
        return encode_cursor(
            StreamCursor(base=base, positions=positions, values_through=through)
        )
