# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""``interrupt()`` and ``Command(resume=...)`` on the Dapr workflow runner.

LangGraph pauses a task that calls ``interrupt(value)``: the run ends with the
interrupt, the checkpointer keeps the thread, and a later run on the same
thread with ``Command(resume=...)`` re-runs the task, where ``interrupt``
now returns the resume value. Here the workflow is the checkpoint:

- The node activity runs the node with LangGraph's interrupt scratchpad. When
  it raises ``GraphInterrupt`` the activity returns the interrupts (no writes,
  not an error, so Dapr does not retry it) and streams them as LangGraph's
  ``{"__interrupt__": ...}`` updates chunk.
- The workflow keeps the step's other results, publishes the interrupts in its
  custom status, and waits for the ``resume`` external event: durably, in the
  workflow history, on whatever replica, for as long as it takes.
- A resume raises that event. The workflow re-runs the interrupted tasks of
  the same step with the resume values (``interrupt`` returns them in order),
  then goes on as if the step had just finished. A task that interrupts again
  waits for the next resume.
"""

import itertools
import json
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple

from .models import ExecuteNodeOutput

#: The external event a resume raises on the workflow.
RESUME_EVENT = "resume"
#: Custom status: the pending interrupt, and the number of the last one.
STATUS_INTERRUPT = "interrupt"
STATUS_INTERRUPT_SEQ = "interrupt_seq"
#: The updates-chunk key LangGraph streams interrupts under.
INTERRUPT_KEY = "__interrupt__"
#: The final event of a stream whose run is waiting for a resume.
WORKFLOW_INTERRUPTED = "workflow_interrupted"


def _plain(value: Any) -> Any:
    """``value`` as JSON (an interrupt's payload crosses the workflow and HTTP)."""
    return json.loads(json.dumps(value, default=str))


def _no_null_resume(consume: bool = False) -> Any:
    """No resume value without a task: every resume here names its tasks."""
    return None


def interrupt_config(
    *,
    namespace: str,
    step: int,
    stop: Optional[int],
    resume: Optional[List[Any]],
    send: Callable[[Sequence[Tuple[str, Any]]], None],
) -> Dict[str, Any]:
    """The ``configurable`` keys ``langgraph.types.interrupt`` reads.

    ``namespace`` names the task, so an interrupt's id is the same on every
    run of it; ``resume`` holds the values the task's interrupts get, in order.
    """
    from langgraph._internal._constants import (
        CONFIG_KEY_CHECKPOINT_NS,
        CONFIG_KEY_SCRATCHPAD,
        CONFIG_KEY_SEND,
    )
    from langgraph._internal._scratchpad import PregelScratchpad

    calls, interrupts, subgraphs = (
        itertools.count(),
        itertools.count(),
        itertools.count(),
    )
    scratchpad = PregelScratchpad(
        step=step,
        stop=stop or 0,
        call_counter=lambda: next(calls),
        interrupt_counter=lambda: next(interrupts),
        get_null_resume=_no_null_resume,
        resume=list(resume or []),
        subgraph_counter=lambda: next(subgraphs),
    )
    return {
        CONFIG_KEY_SCRATCHPAD: scratchpad,
        CONFIG_KEY_CHECKPOINT_NS: namespace,
        CONFIG_KEY_SEND: send,
    }


def without_resume_writes(writes: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    """``writes`` minus the bookkeeping ``interrupt()`` records for Pregel."""
    from langgraph._internal._constants import RESUME

    return [w for w in writes if w[0] != RESUME]


def interrupt_payloads(error: BaseException) -> List[Dict[str, Any]]:
    """The interrupts a ``GraphInterrupt`` carries, as JSON."""
    found = error.args[0] if error.args else ()
    out = []
    for item in found or ():
        payload: Dict[str, Any] = {
            "id": str(getattr(item, "id", "")),
            "value": _plain(getattr(item, "value", None)),
        }
        schema = getattr(item, "response_schema", None)
        if schema is not None:
            payload["response_schema"] = _plain(schema)
        out.append(payload)
    return out


def as_interrupts(payloads: Sequence[Dict[str, Any]]) -> Tuple[Any, ...]:
    """LangGraph ``Interrupt`` objects for ``payloads`` (what native astream yields)."""
    from langgraph.types import Interrupt

    out = []
    for p in payloads:
        if isinstance(p, Interrupt):
            out.append(p)
        elif isinstance(p, dict):
            out.append(Interrupt(value=p.get("value"), id=str(p.get("id", ""))))
    return tuple(out)


def resume_values(pending: Sequence[Dict[str, Any]], resume: Any) -> Dict[str, Any]:
    """The resume value of each pending interrupt id.

    A dict whose keys are all pending interrupt ids maps them one by one (as
    LangGraph's ``Command(resume={id: value})``); anything else resumes every
    pending interrupt with the same value.
    """
    ids = [str(p["id"]) for p in pending]
    if isinstance(resume, dict) and resume and set(resume) <= set(ids):
        return {i: resume[i] for i in ids if i in resume}
    return {i: resume for i in ids}


def settle_interrupts(
    run: Any,
    outputs: List[ExecuteNodeOutput],
    step: int,
    rerun: Callable[
        [Dict[int, List[Any]]], Generator[Any, Any, Dict[int, ExecuteNodeOutput]]
    ],
    publish: Callable[[Any], None],
) -> Generator[Any, Any, List[ExecuteNodeOutput]]:
    """Workflow step: wait out every interrupt of a step's tasks (deterministic).

    ``rerun`` runs the tasks at the given positions again with their resume
    values and returns their new outputs; ``publish`` sets the custom status.
    """
    resumes: Dict[int, List[Any]] = {}
    while True:
        pending = [
            {**p, "node": o.node_name, "task": i}
            for i, o in enumerate(outputs)
            for p in (o.interrupts or [])
        ]
        if not pending:
            return outputs
        run.interrupt_seq += 1
        run.interrupt = {"seq": run.interrupt_seq, "step": step, "interrupts": pending}
        publish(run)
        values = yield from _wait_resume(run.ctx, run.interrupt_seq, pending)
        run.interrupt = None
        publish(run)
        for item in pending:
            if item["id"] in values:
                resumes.setdefault(item["task"], []).append(values[item["id"]])
        again = yield from rerun(
            {task: resumes.get(task, []) for task in {p["task"] for p in pending}}
        )
        outputs = [again.get(i, output) for i, output in enumerate(outputs)]


def _wait_resume(
    ctx: Any, seq: int, pending: List[Dict[str, Any]]
) -> Generator[Any, Any, Dict[str, Any]]:
    """Wait for the resume of interrupt ``seq``; others (stale) are ignored."""
    while True:
        event = yield ctx.wait_for_external_event(RESUME_EVENT)
        data = event if isinstance(event, dict) else {"resume": event}
        if data.get("seq") in (None, seq):
            return resume_values(pending, data.get("resume"))


def pending_interrupt(status: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The interrupt a run waits on, from its custom status, or ``None``."""
    pending = status.get(STATUS_INTERRUPT)
    return pending if isinstance(pending, dict) and pending.get("interrupts") else None


def interrupted_event(
    status: Dict[str, Any], workflow_id: str, after_seq: int = 0
) -> Optional[Dict[str, Any]]:
    """The ``workflow_interrupted`` event for a run waiting on a newer interrupt."""
    pending = pending_interrupt(status)
    if pending is None or int(pending.get("seq") or 0) <= after_seq:
        return None
    return {
        "type": WORKFLOW_INTERRUPTED,
        "workflow_id": workflow_id,
        "status": "interrupted",
        "interrupt_seq": int(pending.get("seq") or 0),
        "interrupts": [
            {k: v for k, v in item.items() if k != "task"}
            for item in pending["interrupts"]
        ],
    }
