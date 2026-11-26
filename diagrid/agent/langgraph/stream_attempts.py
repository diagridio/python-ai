# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Which attempt of each unit the run used, so its stream shows that one.

Dapr runs an activity at least once: a completion that got lost is run again,
and two attempts may even run at once. Each attempt logs its own chunks; the
activity's output names its attempt, the workflow keeps the attempt of the
output it used per unit (``"<step>:<unit>"``), and publishes them with the
units in its custom status and its output. A stream reader follows that
attempt (``UnitWant.winner``), whatever attempt holds the unit's head.
"""

from typing import Any, Dict, Iterable, Optional

#: Custom status (and stream-only output) key: the attempt each unit used.
STATUS_ATTEMPTS = "attempts"


def unit_key(step: int, unit: str) -> str:
    return f"{step}:{unit}"


def with_attempt(output: Dict[str, Any], writer: Any) -> Dict[str, Any]:
    """An activity's output naming the attempt its chunks were logged under."""
    attempt = getattr(writer, "attempt", None)
    if not isinstance(attempt, str) or not attempt:
        return output
    return {**output, "stream_attempt": attempt}


def record_attempts(
    attempts: Dict[str, str], step: int, units: Iterable[str], outputs: Iterable[Any]
) -> None:
    """Keep the attempt of each unit's output (workflow code: no I/O)."""
    for unit, output in zip(units, outputs):
        attempt = getattr(output, "stream_attempt", None)
        if attempt:
            attempts[unit_key(step, unit)] = attempt


def attempts_of(status: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """The attempts a custom status or a final event names, by unit key."""
    found = (status or {}).get(STATUS_ATTEMPTS)
    if not isinstance(found, dict):
        return {}
    return {str(k): str(v) for k, v in found.items() if v}
