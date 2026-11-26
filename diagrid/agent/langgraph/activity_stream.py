# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Where an activity of a streamed run writes, and what the workflow publishes.

An activity asks the runner's transport for a writer once per execution; the
node, its routers and its ``values`` chunk all write to it. The workflow
publishes the units each step runs in its custom status (no I/O, so it is safe
in workflow code), which is how a reader on any replica finds them.
"""

import json
from typing import Any, Dict, Optional, Union

from diagrid.agent.core.workflow.stream import (
    ActivityStreamWriter,
    InProcessTransport,
    StreamSinkRegistry,
    StreamTransport,
)

from .streaming import STREAM_RUN_KEY

#: Custom status keys: the units of each step, and the routers run apart.
STATUS_EXECUTED = "executed"
STATUS_BRANCHES = "branches"


def transport_of(
    stream: Union[StreamTransport, StreamSinkRegistry, None],
) -> StreamTransport:
    """``stream`` as a transport: a registry is the in-process transport over it."""
    if stream is None:
        return InProcessTransport()
    if isinstance(stream, StreamSinkRegistry):
        return InProcessTransport(stream)
    return stream


def activity_writer(
    ctx: Any,
    transport: StreamTransport,
    step: int,
    unit: str,
    config: Dict[str, Any],
) -> Optional[ActivityStreamWriter]:
    """The writer of this execution of ``unit``, or ``None`` when not streaming."""
    workflow_id = getattr(ctx, "workflow_id", None)
    if not workflow_id:
        return None
    return transport.writer(
        str(workflow_id), step, unit, streamed=bool(config.get(STREAM_RUN_KEY))
    )


def run_status(run: Any) -> Dict[str, Any]:
    """The custom status of a run: what a stream reader needs to find its
    units, and the interrupt the run waits on."""
    from .interrupts import STATUS_INTERRUPT, STATUS_INTERRUPT_SEQ
    from .stream_attempts import STATUS_ATTEMPTS

    status: Dict[str, Any] = {}
    if (run.config or {}).get(STREAM_RUN_KEY):
        status[STATUS_EXECUTED] = run.executed
        status[STATUS_BRANCHES] = run.branches
        if run.attempts:
            status[STATUS_ATTEMPTS] = run.attempts
    if run.interrupt_seq:
        status[STATUS_INTERRUPT_SEQ] = run.interrupt_seq
    if run.interrupt:
        status[STATUS_INTERRUPT] = run.interrupt
    return status


def publish_run_status(run: Any) -> None:
    """Set the run's custom status, if it has one to publish (workflow code)."""
    status = run_status(run)
    setter = getattr(run.ctx, "set_custom_status", None)
    if status and setter is not None:
        setter(json.dumps(status))


def parse_status(raw: Any) -> Dict[str, Any]:
    """A run's custom status as a dict (empty when it has none or it is not ours)."""
    if not raw:
        return {}
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
