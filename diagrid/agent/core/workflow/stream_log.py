# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A run's stream as an event log in a Dapr state store.

Every activity execution (a *unit*: one node, ``Send`` task or router at one
step) appends its chunks in batches, and any replica reads them back from any
point. Keys use ``:`` between parts, each part path-escaped; ``||`` is never
used (Dapr reserves it for actor keys):

- ``<prefix>:<wf>:<step>:<unit>:head``: ``{"attempt", "batches", "events",
  "done"}``, rewritten after each batch, kept ``record_ttl_factor`` times
  longer than the batches.
- ``<prefix>:<wf>:<step>:<unit>:<attempt>:<n>``: batch ``n`` of an attempt, a
  list of events.
- ``<prefix>:<wf>:<step>:<unit>:<attempt>:end``: the head of an attempt that
  finished, as it was when it did (kept as long as the head).

A Dapr bulk save is neither atomic nor ordered, so a flush writes its batch
first and the head that counts it in a second save: a head never names a
batch that was not written. A reader that still misses a counted batch (a
replica that lags) looks again after ``purge_recheck_s`` before it calls the
unit purged.

A retried activity is a new attempt: it takes the head over, and a reader that
sees the attempt change starts that unit again (the consumer turns that into
``node_reset``). An attempt that finished (``done``) keeps the head: a late
attempt (Dapr ran the activity again) logs its batches and its end record but
leaves the head alone. The workflow says which attempt's output it used (the
*winner*); a reader told the winner reads it through its end record. A head
whose count names a batch that is gone means the log outlived its TTL: the
reader reports it purged.
"""

import itertools
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple
from urllib.parse import quote

from .stream import MAX_BUFFERED_EVENTS, StreamEvent, is_kept_when_full

logger = logging.getLogger(__name__)

#: ``(key, JSON value, Dapr state metadata)``
LogItem = Tuple[str, Any, Dict[str, str]]


@dataclass(frozen=True)
class StateLogStreamConfig:
    """How the state-store event log is written and read."""

    #: The Dapr state store; the runner fills in its own when this is ``None``.
    store_name: Optional[str] = None
    key_prefix: str = "lgstream"
    #: Seconds a batch is kept.
    event_ttl_s: int = 3600
    #: Heads outlive batches by this factor, so a purge is told from a gap.
    record_ttl_factor: int = 24
    flush_interval_s: float = 0.1
    flush_max_chunks: int = 64
    #: How often a reader looks for new batches.
    read_poll_s: float = 0.2
    #: A batch the head counts but the store does not return is looked for
    #: again after this long before the unit is called purged.
    purge_recheck_s: float = 0.5
    #: Events an activity holds while the store refuses writes; past it,
    #: tokens and custom chunks are dropped, markers and updates never.
    max_buffered: int = MAX_BUFFERED_EVENTS
    #: Tries at the last write of an activity (its ``updates`` and done head),
    #: with a backoff that doubles from ``final_flush_backoff_s``.
    final_flush_attempts: int = 4
    final_flush_backoff_s: float = 0.05


class LogStore(Protocol):
    """The two state calls the log needs."""

    def save_bulk(self, items: Sequence[LogItem]) -> None: ...

    def get_bulk(self, keys: Sequence[str]) -> Dict[str, Any]: ...


class DaprLogStore:
    """:class:`LogStore` on a Dapr state store, through the sidecar."""

    def __init__(self, store_name: str) -> None:
        self.store_name = store_name
        self._client: Any = None
        self._lock = threading.Lock()

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is None:
                from dapr.clients import DaprClient

                self._client = DaprClient()
            return self._client

    def save_bulk(self, items: Sequence[LogItem]) -> None:
        import json

        from dapr.clients.grpc._state import StateItem

        states = [
            StateItem(key=key, value=json.dumps(value), metadata=metadata)
            for key, value, metadata in items
        ]
        self._get_client().save_bulk_state(store_name=self.store_name, states=states)

    def get_bulk(self, keys: Sequence[str]) -> Dict[str, Any]:
        import json

        if not keys:
            return {}
        response = self._get_client().get_bulk_state(
            store_name=self.store_name, keys=list(keys)
        )
        found: Dict[str, Any] = {}
        for item in response.items:
            if getattr(item, "error", ""):
                raise RuntimeError(f"state store read of {item.key!r}: {item.error}")
            if item.data:
                found[item.key] = json.loads(item.data)
        return found


def escape_key_part(part: Any) -> str:
    """One key part, path-escaped: no ``:``, ``#``, ``/`` or ``|`` is left."""
    return quote(str(part), safe="")


def _unit_prefix(prefix: str, workflow_id: str, step: int, unit: str) -> str:
    return ":".join(escape_key_part(p) for p in (prefix, workflow_id, step, unit))


def head_key(prefix: str, workflow_id: str, step: int, unit: str) -> str:
    return f"{_unit_prefix(prefix, workflow_id, step, unit)}:head"


def end_key(prefix: str, workflow_id: str, step: int, unit: str, attempt: str) -> str:
    base = _unit_prefix(prefix, workflow_id, step, unit)
    return f"{base}:{escape_key_part(attempt)}:end"


def batch_key(
    prefix: str, workflow_id: str, step: int, unit: str, attempt: str, n: int
) -> str:
    base = _unit_prefix(prefix, workflow_id, step, unit)
    return f"{base}:{escape_key_part(attempt)}:{n}"


def wire_event(
    event: StreamEvent, encode: Callable[[StreamEvent], Any]
) -> Dict[str, Any]:
    """``event`` as the log stores it; ``encode`` makes its data JSON."""
    return {
        "step": event.step,
        "node": event.node,
        "attempt": event.attempt_id,
        "seq": event.seq,
        "ns": list(event.ns),
        "mode": event.mode,
        "data": encode(event),
        "display_node": event.display_node,
        "path": list(event.path) if event.path else None,
        "branch": event.branch,
    }


def event_from_wire(
    wire: Dict[str, Any], workflow_id: str, decode: Callable[[str, Any], Any]
) -> StreamEvent:
    """The :class:`StreamEvent` a stored event stands for."""
    path = wire.get("path")
    return StreamEvent(
        workflow_id=workflow_id,
        step=int(wire["step"]),
        node=str(wire["node"]),
        attempt_id=str(wire["attempt"]),
        seq=int(wire["seq"]),
        ns=tuple(wire.get("ns") or ()),
        mode=str(wire["mode"]),
        data=decode(str(wire["mode"]), wire.get("data")),
        display_node=wire.get("display_node"),
        path=tuple(path) if path else None,
        branch=wire.get("branch"),
    )


def _ttl(seconds: int) -> Dict[str, str]:
    return {"ttlInSeconds": str(int(seconds))}


class LogWriter:
    """Appends one activity execution's chunks to its unit of the log.

    Flushes every ``flush_interval_s``, every ``flush_max_chunks`` chunks and
    on :meth:`finish`, which writes synchronously, so the unit is complete
    before the activity returns. A failed write is logged and counted, and
    the chunks are tried again with the next flush; the last flush, which
    nothing follows, is tried again with backoff (``final_flush_attempts``).
    A write never fails the node.
    """

    def __init__(
        self,
        store: LogStore,
        config: StateLogStreamConfig,
        workflow_id: str,
        step: int,
        unit: str,
        *,
        encode: Callable[[StreamEvent], Any],
    ) -> None:
        self._store = store
        self._config = config
        self._workflow_id = workflow_id
        self._step = step
        self._unit = unit
        self._encode = encode
        self.attempt = uuid.uuid4().hex
        self._pending: List[Dict[str, Any]] = []
        self._buffer_lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._batches = 0
        self._events = 0
        self._blocked: Optional[bool] = None
        self._finished = False
        self.write_failures = 0
        self.dropped = 0

    def put(self, event: StreamEvent) -> None:
        wire = wire_event(event, self._encode)
        with self._buffer_lock:
            if self._finished:
                return
            if len(
                self._pending
            ) >= self._config.max_buffered and not is_kept_when_full(event.mode):
                self._note_dropped()
                return
            self._pending.append(wire)
            full = len(self._pending) >= self._config.flush_max_chunks
            if not full and self._timer is None:
                self._timer = threading.Timer(
                    self._config.flush_interval_s, self._timed_flush
                )
                self._timer.daemon = True
                self._timer.start()
        if full:
            self._flush(done=False)

    def finish(self, *, done: bool) -> None:
        """Write what is left; ``done`` marks the attempt as the unit's last."""
        with self._buffer_lock:
            self._finished = True
            timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()
        # another attempt may have finished since the first flush: a done
        # head is never overwritten, so look at the head again
        self._blocked = None
        delay = self._config.final_flush_backoff_s
        for attempt in range(max(1, self._config.final_flush_attempts)):
            if attempt:
                time.sleep(delay)
                delay *= 2
            if self._flush(done=done, final=True):
                return
        logger.warning(
            "Stream log of %s (%s): the last write failed %d times; the unit's "
            "last chunks are lost and readers report it as not written",
            self._workflow_id,
            self._unit,
            self._config.final_flush_attempts,
        )

    def _note_dropped(self) -> None:
        if not self.dropped:
            logger.warning(
                "Stream log of %s (%s) is full while the store refuses writes; "
                "dropping token and custom chunks",
                self._workflow_id,
                self._unit,
            )
        self.dropped += 1

    def _timed_flush(self) -> None:
        with self._buffer_lock:
            self._timer = None
        self._flush(done=False)

    def _flush(self, *, done: bool, final: bool = False) -> bool:
        """Write what is pending; whether nothing is left to write."""
        with self._flush_lock:
            with self._buffer_lock:
                batch = list(self._pending)
            if not batch and not final:
                return True
            if not batch and not done:
                return True
            batches = self._batches + (1 if batch else 0)
            head = {
                "attempt": self.attempt,
                "batches": batches,
                "events": self._events + len(batch),
                "done": done,
            }
            record_ttl = _ttl(self._config.event_ttl_s * self._config.record_ttl_factor)
            prefix = (
                self._config.key_prefix,
                self._workflow_id,
                self._step,
                self._unit,
            )
            items: List[LogItem] = []
            if not self._is_blocked():
                items.append((head_key(*prefix), head, record_ttl))
            if done:
                items.append((end_key(*prefix, self.attempt), head, record_ttl))
            try:
                # the batch lands before any head counts it; a batch written
                # again (its head failed) is the same batch with more events
                if batch:
                    key = batch_key(*prefix, self.attempt, self._batches)
                    self._store.save_bulk(
                        [(key, batch, _ttl(self._config.event_ttl_s))]
                    )
                if items:
                    self._store.save_bulk(items)
            except Exception as e:  # never fail the node over its stream
                self.write_failures += 1
                logger.warning(
                    "Stream log write for %s (%s) failed (%s: %s); retrying with "
                    "the next batch",
                    self._workflow_id,
                    self._unit,
                    type(e).__name__,
                    e,
                )
                return False
            self._batches = batches
            self._events += len(batch)
            with self._buffer_lock:
                del self._pending[: len(batch)]
            return True

    def _is_blocked(self) -> bool:
        """Whether an attempt that already finished owns the unit.

        Read at the first flush and again at the last one.
        """
        if self._blocked is None:
            key = head_key(
                self._config.key_prefix, self._workflow_id, self._step, self._unit
            )
            try:
                head = self._store.get_bulk([key]).get(key)
            except Exception as e:
                logger.warning(
                    "Stream log head of %s (%s) could not be read (%s); writing anyway",
                    self._workflow_id,
                    self._unit,
                    e,
                )
                head = None
            self._blocked = bool(
                isinstance(head, dict)
                and head.get("done")
                and head.get("attempt") != self.attempt
            )
        return self._blocked


@dataclass(frozen=True)
class UnitWant:
    """What a reader asks of one unit: what it has of it already."""

    step: int
    unit: str
    #: The attempt it read so far (a prefix is enough); ``None``: none.
    attempt: Optional[str] = None
    #: The first batch it has not read, of ``attempt``.
    from_batch: int = 0
    #: Events of the batches from ``from_batch`` on that it already has.
    skip_events: int = 0
    #: The attempt whose output the workflow used, once it is known: it is
    #: read (through its end record) whatever attempt holds the head.
    winner: Optional[str] = None


@dataclass(frozen=True)
class UnitChunk:
    """What a read found in one unit."""

    step: int
    unit: str
    #: The unit's attempt now; ``None`` when it has no head yet.
    attempt: Optional[str] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    #: Index, within the attempt, of the first event in ``events``.
    start: int = 0
    #: Batches of the attempt there are (all read once these are).
    batches: int = 0
    #: Events of the attempt there are.
    total: int = 0
    done: bool = False
    #: The attempt is not the one the reader asked about: start again.
    reset: bool = False
    #: A batch the head counts is gone (TTL): the log cannot be replayed.
    purged: bool = False


class LogReader:
    """Reads units of one run's log: heads in one round trip, batches in another.

    A store that fails raises: a reader must say so, never return an empty
    stream as if nothing had happened.
    """

    def __init__(
        self, store: LogStore, config: StateLogStreamConfig, workflow_id: str
    ) -> None:
        self._store = store
        self._config = config
        self._workflow_id = workflow_id

    def read(self, wants: Sequence[UnitWant]) -> List[UnitChunk]:
        prefix = self._config.key_prefix
        head_keys = [head_key(prefix, self._workflow_id, w.step, w.unit) for w in wants]
        end_keys = {
            i: end_key(prefix, self._workflow_id, w.step, w.unit, w.winner)
            for i, w in enumerate(wants)
            if w.winner
        }
        wanted_heads = [*head_keys, *end_keys.values()]
        heads = self._store.get_bulk(wanted_heads) if wanted_heads else {}
        plans: List[Tuple[UnitWant, Dict[str, Any], bool, List[str]]] = []
        for i, (want, key) in enumerate(zip(wants, head_keys)):
            head = heads.get(key)
            won = heads.get(end_keys[i]) if i in end_keys else None
            if isinstance(won, dict) and not _same_attempt(
                str((head or {}).get("attempt")), str(want.winner)
            ):
                head = won  # the head is another attempt's: read the winner
            if not isinstance(head, dict):
                plans.append((want, {}, False, []))
                continue
            attempt = str(head.get("attempt"))
            reset = bool(want.attempt) and not attempt.startswith(str(want.attempt))
            first = 0 if reset else want.from_batch
            keys = [
                batch_key(prefix, self._workflow_id, want.step, want.unit, attempt, n)
                for n in range(first, int(head.get("batches") or 0))
            ]
            plans.append((want, head, reset, keys))
        wanted = list(itertools.chain.from_iterable(p[3] for p in plans))
        batches = self._store.get_bulk(wanted) if wanted else {}
        missing = [key for key in wanted if key not in batches]
        if missing:
            # a head is written after its batches, so a counted batch that is
            # not there is purged, or on a replica that has not caught up yet
            time.sleep(self._config.purge_recheck_s)
            batches = {**batches, **self._store.get_bulk(missing)}
        return [
            self._chunk(want, head, reset, keys, batches)
            for want, head, reset, keys in plans
        ]

    def _chunk(
        self,
        want: UnitWant,
        head: Dict[str, Any],
        reset: bool,
        keys: List[str],
        batches: Dict[str, Any],
    ) -> UnitChunk:
        if not head:
            return UnitChunk(want.step, want.unit)
        attempt = str(head["attempt"])
        total = int(head.get("events") or 0)
        count = int(head.get("batches") or 0)
        done = bool(head.get("done"))
        if any(key not in batches for key in keys):
            return UnitChunk(
                want.step,
                want.unit,
                attempt,
                batches=count,
                total=total,
                done=done,
                reset=reset,
                purged=True,
            )
        events = [e for key in keys for e in batches[key]]
        skip = 0 if reset else min(want.skip_events, len(events))
        # the events of batches before the first one read are not in ``events``
        before = 0 if reset else max(0, total - len(events))
        return UnitChunk(
            want.step,
            want.unit,
            attempt,
            events=events[skip:],
            start=before + skip,
            batches=count,
            total=total,
            done=done,
            reset=reset,
        )


def _json_data(event: StreamEvent) -> Any:
    return event.data


def _as_is(mode: str, data: Any) -> Any:
    return data


class StateLogTransport:
    """Streams through the state-store event log: any replica reads, any time.

    ``store`` defaults to the Dapr state store ``config.store_name``. ``encode``
    turns a chunk's data into JSON for the log and ``decode`` turns it back; a
    runner binds its own with :meth:`with_codec`.
    """

    replayable = True

    def __init__(
        self,
        config: Optional[StateLogStreamConfig] = None,
        *,
        store: Optional[LogStore] = None,
        encode: Callable[[StreamEvent], Any] = _json_data,
        decode: Callable[[str, Any], Any] = _as_is,
    ) -> None:
        self.config = config or StateLogStreamConfig()
        if store is None:
            if not self.config.store_name:
                raise ValueError("StateLogTransport needs a store or a store_name")
            store = DaprLogStore(self.config.store_name)
        self.store = store
        self.encode = encode
        self.decode = decode

    @property
    def store_name(self) -> Optional[str]:
        return self.config.store_name or getattr(self.store, "store_name", None)

    def with_codec(
        self,
        encode: Callable[[StreamEvent], Any],
        decode: Callable[[str, Any], Any],
    ) -> "StateLogTransport":
        """A transport on the same store that encodes and decodes this way."""
        return StateLogTransport(
            self.config, store=self.store, encode=encode, decode=decode
        )

    def writer(
        self, workflow_id: str, step: int, unit: str, *, streamed: bool
    ) -> Optional[LogWriter]:
        if not streamed:
            return None
        return LogWriter(
            self.store, self.config, workflow_id, step, unit, encode=self.encode
        )

    def reader(self, workflow_id: str) -> LogReader:
        return LogReader(self.store, self.config, workflow_id)


def _same_attempt(a: str, b: str) -> bool:
    """Whether two attempt ids (or a cursor's prefix of one) name one attempt."""
    return bool(a) and bool(b) and (a.startswith(b) or b.startswith(a))
