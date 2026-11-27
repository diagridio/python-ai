# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""The SSE id of a stream read from the state-store event log.

The id is where the client is in the run's log, in a form any replica (or a
restarted process) can resume from: ``v1.<base>.<positions>``. Steps are
barriers, so every unit of a step below ``base`` was read in full; for the
units of ``base`` and later that were read in part, ``positions`` holds the
attempt (its first 8 characters) and how many of its events were read, as
base64url JSON. The id grows with one step's parallelism, not with the run.
A fourth part, ``.<step>``, is the last step whose ``values`` chunk the
client has: every unit of a step logs that chunk, and it is sent once.

``v1.start`` replays from the beginning and ``v1.end`` means the final event
was received. An id that does not parse replays from the start too.
"""

import base64
import binascii
import json
from dataclasses import dataclass
from typing import Optional, Tuple

VERSION = "v1"
START_ID = f"{VERSION}.start"
END_ID = f"{VERSION}.end"
#: How much of an attempt id a cursor keeps: enough to tell attempts apart.
ATTEMPT_PREFIX = 8

#: ``(step, unit, attempt prefix, events read)``
Position = Tuple[int, str, str, int]


@dataclass(frozen=True)
class StreamCursor:
    """How far a client got in a run's event log."""

    #: Units of steps below this were read in full; ``None``: nothing was.
    base: Optional[int] = None
    positions: Tuple[Position, ...] = ()
    #: The client has the final event.
    at_end: bool = False
    #: The last step whose ``values`` chunk the client has; ``None``: none.
    values_through: Optional[int] = None

    def skips_step(self, step: int) -> bool:
        """Whether every unit of ``step`` was already read."""
        return self.base is not None and step < self.base

    def position(self, step: int, unit: str) -> Optional[Tuple[str, int]]:
        """``(attempt prefix, events read)`` of a unit read in part, else ``None``."""
        for p_step, p_unit, attempt, events in self.positions:
            if p_step == step and p_unit == unit:
                return attempt, events
        return None


def encode_cursor(cursor: StreamCursor) -> str:
    """The SSE id for ``cursor``."""
    if cursor.at_end:
        return END_ID
    if cursor.base is None and not cursor.positions and cursor.values_through is None:
        return START_ID
    payload = json.dumps(
        [list(p) for p in cursor.positions], separators=(",", ":")
    ).encode()
    packed = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    text = f"{VERSION}.{cursor.base if cursor.base is not None else -2}.{packed}"
    if cursor.values_through is not None:
        text += f".{cursor.values_through}"
    return text


def parse_cursor(text: Optional[str]) -> StreamCursor:
    """The cursor an SSE id names; anything that does not parse is the start."""
    if text == END_ID:
        return StreamCursor(at_end=True)
    try:
        return _parse(text or "")
    except (ValueError, TypeError, binascii.Error):
        return StreamCursor()


def _parse(text: str) -> StreamCursor:
    parts = text.split(".")
    if len(parts) not in (3, 4):
        raise ValueError(text)
    version, base, packed = parts[:3]
    values_through = int(parts[3]) if len(parts) == 4 else None
    if version != VERSION or (values_through is not None and values_through < 0):
        raise ValueError(text)
    padded = packed + "=" * (-len(packed) % 4)
    raw = json.loads(base64.urlsafe_b64decode(padded.encode()))
    if not isinstance(raw, list):
        raise ValueError(text)
    positions = tuple(_position(p) for p in raw)
    step = int(base)
    return StreamCursor(
        base=None if step == -2 else step,
        positions=positions,
        values_through=values_through,
    )


def _position(item: object) -> Position:
    if not isinstance(item, list) or len(item) != 4:
        raise ValueError(item)
    step, unit, attempt, events = item
    if not isinstance(unit, str) or not isinstance(attempt, str):
        raise ValueError(item)
    if int(events) < 0:
        raise ValueError(item)
    return int(step), unit, attempt, int(events)
