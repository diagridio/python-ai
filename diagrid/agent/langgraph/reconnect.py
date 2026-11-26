# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Reconnecting to a run's stream from wherever the client got to.

``GET /agent/run/{workflow_id}/stream`` (with ``Last-Event-ID``, or the
``last_event_id`` query parameter for a client that cannot set the header)
and ``runner.astream_resume`` share :func:`resume_events`. On the state-store
transport any replica replays the run's log after the cursor, then follows
the run live and ends with the final event and its result. On the in-process
transport nothing can be replayed: the stream says so, then still ends with
the result.
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, AsyncIterator, Dict, Optional, Set, Tuple

from diagrid.agent.core.workflow import stream as stream_module
from diagrid.agent.core.workflow.stream import (
    STREAM_UNAVAILABLE,
    StreamEvent,
    frame_sse,
)
from diagrid.agent.core.workflow.stream_cursor import parse_cursor

from .stream_drain import SseEncoder, drain
from .stream_events import (
    FINAL_EVENTS,
    KEEPALIVE,
    started_event,
    stream_failure_event,
    without_executed,
)
from .stream_source import NOT_REPLAYABLE, custom_status, open_source
from .streaming import normalize_final, normalize_stream_mode

logger = logging.getLogger(__name__)

NOT_REPLAYABLE_MESSAGE = (
    "This runner streams in process, so a stream cannot be resumed or read on "
    "another replica: reconnecting needs a state store (any Dapr state store "
    "the app has; on Catalyst the workflow store is enough). The run's result "
    "follows when it ends."
)


def _thread_of(state: Any, workflow_id: str) -> str:
    """The thread a run belongs to, from its workflow input."""
    raw = getattr(state, "serialized_input", None)
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        data = None
    thread = data.get("thread_id") if isinstance(data, dict) else None
    return str(thread or workflow_id)


def not_replayable_event(workflow_id: str) -> StreamEvent:
    return StreamEvent(
        workflow_id=workflow_id,
        step=-1,
        node="",
        attempt_id="",
        seq=0,
        ns=(),
        mode=STREAM_UNAVAILABLE,
        data={
            "workflow_id": workflow_id,
            "reason": NOT_REPLAYABLE,
            "message": NOT_REPLAYABLE_MESSAGE,
            "missing_steps": [],
            "missing_nodes": [],
        },
    )


async def follow_final(
    runner: Any, workflow_id: str, thread_id: str, poll_interval: float
) -> Dict[str, Any]:
    """Wait for the run to end; its normalized terminal event (stream-only keys kept)."""
    final: Dict[str, Any] = {}
    try:
        async for event in runner._follow_run(
            workflow_id, thread_id, poll_interval=poll_interval, streamed=True
        ):
            if event["type"] in FINAL_EVENTS:
                final = event
    except Exception as e:
        logger.exception("Following run %s failed", workflow_id)
        final = stream_failure_event(e, "PollError", True)
    return normalize_final(final, workflow_id=workflow_id, thread_id=thread_id)


async def _with_keepalive(
    task: "asyncio.Task[Dict[str, Any]]", keepalive: bool
) -> AsyncGenerator[Any, None]:
    """Keepalives while ``task`` runs (when asked), then its result."""
    while True:
        if not keepalive:
            yield await task
            return
        done, _ = await asyncio.wait(
            {task}, timeout=stream_module.SSE_KEEPALIVE_SECONDS
        )
        if done:
            yield task.result()
            return
        yield KEEPALIVE


async def resume_events(
    runner: Any,
    workflow_id: str,
    last_event_id: Optional[str],
    *,
    poll_interval: float,
    keepalive: bool,
) -> AsyncGenerator[Any, None]:
    """The items of a reconnected stream, as ``_stream_events`` yields them.

    Raises ``LookupError`` for a run the workflow engine does not know.
    """
    client = runner._workflow_client
    state = await asyncio.to_thread(client.get_workflow_state, instance_id=workflow_id)
    if state is None:
        raise LookupError(f"Workflow {workflow_id!r} not found")
    thread_id = _thread_of(state, workflow_id)
    cursor = parse_cursor(last_event_id)
    transport = runner.stream_transport
    if not transport.replayable or cursor.at_end:
        task = asyncio.create_task(
            follow_final(runner, workflow_id, thread_id, poll_interval)
        )
        try:
            if not cursor.at_end:
                yield started_event(workflow_id, thread_id)
                yield not_replayable_event(workflow_id)
            async for item in _with_keepalive(task, keepalive):
                yield without_executed(item) if isinstance(item, dict) else item
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return
    seen: Set[Tuple[int, str]] = set()
    source = open_source(
        transport,
        workflow_id,
        status=lambda: custom_status(client, workflow_id),
        seen=seen,
        cursor=cursor,
    )

    async def pump() -> None:
        source.on_final(
            await follow_final(runner, workflow_id, thread_id, poll_interval)
        )

    pump_task = asyncio.create_task(pump())
    try:
        yield started_event(workflow_id, thread_id)
        async for item in drain(
            source,
            workflow_id=workflow_id,
            seen=seen,
            keepalive=keepalive,
            output_channels=runner.graph_config.output_channels,
        ):
            yield item
    finally:
        await source.close()
        pump_task.cancel()
        await asyncio.gather(pump_task, return_exceptions=True)


async def resume_sse(
    runner: Any, workflow_id: str, last_event_id: Optional[str], modes: Any
) -> AsyncIterator[Tuple[Any, ...]]:
    """The SSE items of a reconnect; a failure of the stream ends it in-band."""
    encoder = SseEncoder(modes, runner.stream_transport.replayable)
    try:
        async for item in resume_events(
            runner, workflow_id, last_event_id, poll_interval=0.5, keepalive=True
        ):
            out = encoder.encode(item)
            if out is not None:
                yield out
    except Exception as e:
        logger.exception("Reconnected stream of %s failed", workflow_id)
        failed = stream_failure_event(e, "StreamError", True)
        yield (
            "workflow_failed",
            normalize_final(failed, workflow_id=workflow_id, thread_id=workflow_id),
        )


def _stream_modes(query: Any, default: Any) -> Any:
    raw = [m for value in query.getlist("stream_mode") for m in value.split(",") if m]
    modes, _ = normalize_stream_mode(raw or default)
    return modes


def add_reconnect_route(app: Any, runner: Any) -> None:
    """Register ``GET /agent/run/{workflow_id}/stream`` on ``app``."""
    from fastapi import HTTPException, Request
    from fastapi.responses import StreamingResponse

    @app.get("/agent/run/{workflow_id}/stream")
    async def stream_run(workflow_id: str, request: Request) -> Any:
        try:
            modes = _stream_modes(request.query_params, runner._DEFAULT_STREAM_MODES)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        # Off the loop: the state lookup is a blocking gRPC call.
        state = await asyncio.to_thread(
            runner._workflow_client.get_workflow_state, instance_id=workflow_id
        )
        if state is None:
            raise HTTPException(status_code=404, detail="Workflow not found")
        last_event_id = request.headers.get(
            "last-event-id"
        ) or request.query_params.get("last_event_id")
        return StreamingResponse(
            frame_sse(resume_sse(runner, workflow_id, last_event_id, modes)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
