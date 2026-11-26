# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""What ``runner.invoke`` returns: the run's output, its interrupts, or an error.

``invoke(input)`` schedules a run and waits for it here; ``invoke(
Command(resume=...), workflow_id=...)`` resumes a run paused by
``interrupt()`` and waits for it the same way, as LangGraph's ``invoke`` with
a ``Command`` continues the thread.
"""

import json
import logging
import time
import uuid
from typing import Any, Callable, Optional

from dapr.ext.workflow import WorkflowStatus

from .activity_stream import parse_status
from .interrupt_ids import partial_map_error
from .interrupts import INTERRUPT_KEY, RESUME_EVENT, as_interrupts
from .interrupts import interrupted_event, pending_interrupt, resume_taken
from .models import GraphWorkflowOutput
from .streaming import FAILED_STATUSES, normalize_final, run_error

logger = logging.getLogger(__name__)

#: The workflow output key of a graph whose whole state is one value.
_ROOT = "__root__"


def _completed_output(state: Any, runner: Any, workflow_id: str, thread_id: str) -> Any:
    """The output of a completed run; a run that failed in the graph raises."""
    if not state.serialized_output:
        return {}
    raw = state.serialized_output
    output = GraphWorkflowOutput.from_dict(
        json.loads(raw) if isinstance(raw, str) else raw
    )
    if output.status in FAILED_STATUSES:
        # Not a result: a run that hit its step limit or failed in a node raises.
        raise run_error(
            normalize_final(
                {
                    "type": "workflow_completed",
                    "status": output.status,
                    "error": output.error,
                    "output": output.output,
                },
                workflow_id=workflow_id,
                thread_id=thread_id,
            )
        )
    if output.thread_saved is False:
        logger.warning(
            "Run %s completed, but the state of thread %r was not saved: %s",
            workflow_id,
            thread_id,
            output.thread_error,
        )
    if runner._is_root_state() and _ROOT in output.output:
        return output.output[_ROOT]
    return output.output


def wait_invoke_result(
    runner: Any,
    workflow_id: str,
    thread_id: str,
    *,
    poll_interval: float,
    timeout: Optional[float],
    after_seq: int = 0,
    on_completed: Optional[Callable[[], None]] = None,
) -> Any:
    """Poll the run until it ends or pauses on an interrupt newer than ``after_seq``.

    A pause returns ``{"__interrupt__": [Interrupt, ...]}`` as LangGraph's
    ``invoke`` does; a run that failed raises. ``on_completed`` is called once
    the workflow completed (in the graph's error or not).
    """
    client = runner._workflow_client
    start = time.time()
    while True:
        time.sleep(poll_interval)
        if timeout and (time.time() - start) > timeout:
            raise TimeoutError(f"Workflow {workflow_id} timed out after {timeout}s")
        state = client.get_workflow_state(instance_id=workflow_id)
        if state is None:
            raise RuntimeError(f"Workflow {workflow_id} state not found")
        paused = interrupted_event(
            parse_status(state.serialized_custom_status), workflow_id, after_seq
        )
        if state.runtime_status == WorkflowStatus.RUNNING and paused:
            return {INTERRUPT_KEY: list(as_interrupts(paused["interrupts"]))}
        if state.runtime_status == WorkflowStatus.COMPLETED:
            if on_completed is not None:
                on_completed()
            return _completed_output(state, runner, workflow_id, thread_id)
        if state.runtime_status == WorkflowStatus.FAILED:
            details = state.failure_details
            raise RuntimeError(
                getattr(details, "message", str(details))
                if details
                else "Workflow failed"
            )
        if state.runtime_status == WorkflowStatus.TERMINATED:
            raise RuntimeError(f"Workflow {workflow_id} was terminated")


def _wait_taken(
    client: Any, workflow_id: str, seq: int, resume_id: str, poll_interval: float
) -> None:
    """Wait until the run took this resume; another one taken raises."""
    from .resume_run import TAKE_TIMEOUT_S, ResumeConflict

    deadline = time.monotonic() + TAKE_TIMEOUT_S
    while time.monotonic() < deadline:
        state = client.get_workflow_state(instance_id=workflow_id)
        status = parse_status(getattr(state, "serialized_custom_status", None))
        taken = resume_taken(status, seq, resume_id)
        if taken is False:
            raise ResumeConflict(
                f"Interrupt {seq} of workflow {workflow_id!r} was already resumed "
                "by another request"
            )
        if taken:
            return
        time.sleep(min(poll_interval, 0.05))


def invoke_resume(
    runner: Any,
    command: Any,
    workflow_id: Optional[str],
    *,
    poll_interval: float,
    timeout: Optional[float],
) -> Any:
    """Resume the run ``workflow_id`` with ``command.resume`` and wait for it.

    Raises ``ValueError`` for a ``Command`` that is not a resume, without the
    run's ``workflow_id``, or for a run that is not waiting for a resume, and
    ``LookupError`` for an unknown run.
    """
    from .reconnect import _thread_of

    if getattr(command, "resume", None) is None or not workflow_id:
        raise ValueError(
            "invoke(Command(resume=...)) resumes a run paused by interrupt(): "
            "pass the run's workflow_id (other Commands are not supported as input)"
        )
    client = runner._workflow_client
    state = client.get_workflow_state(instance_id=workflow_id)
    if state is None:
        raise LookupError(f"Workflow {workflow_id!r} not found")
    pending = pending_interrupt(parse_status(state.serialized_custom_status))
    if state.runtime_status != WorkflowStatus.RUNNING or pending is None:
        raise ValueError(f"Workflow {workflow_id!r} is not waiting for a resume")
    error = partial_map_error(pending["interrupts"], command.resume)
    if error is not None:
        raise ValueError(error)
    seq = int(pending.get("seq") or 0)
    resume_id = uuid.uuid4().hex
    client.raise_workflow_event(
        instance_id=workflow_id,
        event_name=RESUME_EVENT,
        data={"seq": seq, "resume": command.resume, "id": resume_id},
    )
    _wait_taken(client, workflow_id, seq, resume_id, poll_interval)
    return wait_invoke_result(
        runner,
        workflow_id,
        _thread_of(state, workflow_id),
        poll_interval=poll_interval,
        timeout=timeout,
        after_seq=seq,
    )
