# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Resuming a run paused by ``interrupt()``: ``Command(resume=...)``.

``POST /agent/run/{workflow_id}/resume`` (``{"resume": value}``, with
``"stream": true`` for SSE) and ``astream(Command(resume=...),
workflow_id=...)`` raise the workflow's ``resume`` event and follow the same
run to its end, or to its next interrupt. A stream of a resumed run carries
what happens after the resume only.

Two callers may resume the same interrupt at once: both raise the event, the
run takes the first and ignores the other. Each resume carries an id, the run
publishes the id it took, and the caller whose resume it ignored gets
:class:`ResumeConflict` (``409``) instead of a result its value had no part in.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, AsyncGenerator, AsyncIterator, Dict, Set, Tuple

from diagrid.agent.core.workflow.stream import frame_sse

from .activity_stream import parse_status
from .interrupts import RESUME_EVENT, pending_interrupt, resume_taken
from .reconnect import _thread_of, follow_final
from .stream_drain import SseEncoder, drain
from .stream_events import started_event, stream_failure_event, without_executed
from .stream_source import custom_status, open_source
from .streaming import normalize_final, normalize_stream_mode

logger = logging.getLogger(__name__)


class NotInterrupted(ValueError):
    """The run is not waiting for a resume (it ended, or never paused)."""


class ResumeConflict(NotInterrupted):
    """Another resume of the same interrupt got to the run first."""


#: Seconds a caller waits for the run to take its resume before following it
#: anyway (a sidecar that is slow to deliver the event).
TAKE_TIMEOUT_S = 30.0


async def _pending(runner: Any, workflow_id: str) -> Tuple[int, str]:
    """``(interrupt seq, thread id)`` of a run waiting for a resume.

    Raises ``LookupError`` for an unknown run, :class:`NotInterrupted` for a
    run that is not waiting.
    """
    client = runner._workflow_client
    state = await asyncio.to_thread(client.get_workflow_state, instance_id=workflow_id)
    if state is None:
        raise LookupError(f"Workflow {workflow_id!r} not found")
    pending = pending_interrupt(
        parse_status(getattr(state, "serialized_custom_status", None))
    )
    if pending is None:
        raise NotInterrupted(f"Workflow {workflow_id!r} is not waiting for a resume")
    return int(pending.get("seq") or 0), _thread_of(state, workflow_id)


async def _send_resume(runner: Any, workflow_id: str, seq: int, value: Any) -> None:
    """Raise the resume and wait until the run took it (or another one)."""
    resume_id = uuid.uuid4().hex
    await asyncio.to_thread(
        runner._workflow_client.raise_workflow_event,
        instance_id=workflow_id,
        event_name=RESUME_EVENT,
        data={"seq": seq, "resume": value, "id": resume_id},
    )
    deadline = time.monotonic() + TAKE_TIMEOUT_S
    while time.monotonic() < deadline:
        taken = resume_taken(
            await asyncio.to_thread(
                custom_status, runner._workflow_client, workflow_id
            ),
            seq,
            resume_id,
        )
        if taken is False:
            raise ResumeConflict(
                f"Interrupt {seq} of workflow {workflow_id!r} was already resumed "
                "by another request"
            )
        if taken:
            return
        await asyncio.sleep(0.05)
    logger.warning(
        "Run %s did not take resume %s within %.0fs; following it anyway",
        workflow_id,
        resume_id,
        TAKE_TIMEOUT_S,
    )


async def resume_run_events(
    runner: Any,
    workflow_id: str,
    value: Any,
    *,
    poll_interval: float,
    keepalive: bool,
) -> AsyncGenerator[Any, None]:
    """Resume the run and yield its stream from there, as ``_stream_events`` does."""
    seq, thread_id = await _pending(runner, workflow_id)
    client = runner._workflow_client
    seen: Set[Tuple[int, str]] = set()
    source = open_source(
        runner.stream_transport,
        workflow_id,
        status=lambda: custom_status(client, workflow_id),
        seen=seen,
        # what the log holds now was streamed before the interrupt
        from_end=True,
    )
    pump_task = None
    try:
        if source.follower is not None:
            await source.follower.ready.wait()
        await _send_resume(runner, workflow_id, seq, value)

        async def pump() -> None:
            source.on_final(
                await follow_final(
                    runner, workflow_id, thread_id, poll_interval, after_seq=seq
                )
            )

        pump_task = asyncio.create_task(pump())
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
        if pump_task is not None:
            pump_task.cancel()
            await asyncio.gather(pump_task, return_exceptions=True)


async def resume_run_result(
    runner: Any, workflow_id: str, value: Any, *, poll_interval: float
) -> Dict[str, Any]:
    """Resume the run and wait for it: the body a plain ``POST /agent/run`` returns."""
    seq, thread_id = await _pending(runner, workflow_id)
    await _send_resume(runner, workflow_id, seq, value)
    final = await follow_final(
        runner, workflow_id, thread_id, poll_interval, after_seq=seq
    )
    return {
        "instance_id": workflow_id,
        "thread_id": thread_id,
        **without_executed(final),
    }


async def _resume_sse(
    runner: Any, workflow_id: str, value: Any, modes: Any
) -> AsyncGenerator[Tuple[Any, ...], None]:
    encoder = SseEncoder(modes, runner.stream_transport.replayable)
    try:
        async for item in resume_run_events(
            runner, workflow_id, value, poll_interval=0.5, keepalive=True
        ):
            out = encoder.encode(item)
            if out is not None:
                yield out
    except ResumeConflict:
        raise  # before the first event: the route answers 409
    except Exception as e:
        logger.exception("Resumed run %s failed to stream", workflow_id)
        failed = stream_failure_event(e, "StreamError", True)
        yield (
            "workflow_failed",
            normalize_final(failed, workflow_id=workflow_id, thread_id=workflow_id),
        )


async def _after(first: Any, rest: AsyncIterator[Any]) -> AsyncIterator[Any]:
    yield first
    async for item in rest:
        yield item


def add_resume_route(app: Any, runner: Any) -> None:
    """Register ``POST /agent/run/{workflow_id}/resume`` on ``app``."""
    from fastapi import HTTPException, Request
    from fastapi.responses import StreamingResponse

    @app.post("/agent/run/{workflow_id}/resume")
    async def resume_run(workflow_id: str, request: Request) -> Any:
        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(status_code=422, detail="the body must be JSON")
        if not isinstance(body, dict) or "resume" not in body:
            raise HTTPException(
                status_code=422, detail='the body must be {"resume": <value>}'
            )
        try:
            modes, _ = normalize_stream_mode(
                body.get("stream_mode") or runner._DEFAULT_STREAM_MODES
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        try:
            await _pending(runner, workflow_id)
        except LookupError:
            raise HTTPException(status_code=404, detail="Workflow not found")
        except NotInterrupted as e:
            raise HTTPException(status_code=409, detail=str(e))
        if body.get("stream") is True:
            events = _resume_sse(runner, workflow_id, body["resume"], modes)
            try:
                # the first event comes once the run took this resume
                first = await events.__anext__()
            except (NotInterrupted, LookupError) as e:
                await events.aclose()
                code = 404 if isinstance(e, LookupError) else 409
                raise HTTPException(status_code=code, detail=str(e))
            return StreamingResponse(
                frame_sse(_after(first, events)),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            return await resume_run_result(
                runner, workflow_id, body["resume"], poll_interval=0.5
            )
        except NotInterrupted as e:
            raise HTTPException(status_code=409, detail=str(e))
