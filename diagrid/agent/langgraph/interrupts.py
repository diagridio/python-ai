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

An ``interrupt()`` below the node pauses the task too: in a subgraph the node
runs (its ``__interrupt__`` result is the task's interrupt, and on the re-run
the subgraph gets its resume value through the task's scratchpad) and in the
node's router (it shares the task's scratchpad).

The scratchpad is a LangGraph internal, supported for langgraph
:data:`SUPPORTED_LANGGRAPH`.
"""

import itertools
import json
import threading
from types import SimpleNamespace
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple

from .models import ExecuteNodeOutput

#: The langgraph versions whose internals ``interrupt()`` support is built on.
SUPPORTED_LANGGRAPH = ">=1.2.14,<2"

#: The external event a resume raises on the workflow.
RESUME_EVENT = "resume"
#: Custom status: the pending interrupt, and the number of the last one.
STATUS_INTERRUPT = "interrupt"
STATUS_INTERRUPT_SEQ = "interrupt_seq"
#: Custom status: the resume the run took last, ``{"seq", "id"}``.
STATUS_RESUMED = "resumed"
#: The updates-chunk key LangGraph streams interrupts under.
INTERRUPT_KEY = "__interrupt__"
#: The final event of a stream whose run is waiting for a resume.
WORKFLOW_INTERRUPTED = "workflow_interrupted"


def _plain(value: Any) -> Any:
    """``value`` as JSON (an interrupt's payload crosses the workflow and HTTP)."""
    return json.loads(json.dumps(value, default=str))


def _internals() -> Any:
    """The LangGraph internals ``interrupt()`` support reads, or a clear error."""
    try:
        from langgraph._internal._constants import (
            CONFIG_KEY_CHECKPOINT_NS,
            CONFIG_KEY_SCRATCHPAD,
            CONFIG_KEY_SEND,
            RESUME,
        )
        from langgraph._internal._scratchpad import PregelScratchpad
    except ImportError as e:
        raise ImportError(
            "interrupt() on the Dapr workflow runner needs langgraph "
            f"{SUPPORTED_LANGGRAPH}: this langgraph does not have the internals "
            f"it uses ({e})"
        ) from e
    return SimpleNamespace(
        CHECKPOINT_NS=CONFIG_KEY_CHECKPOINT_NS,
        SCRATCHPAD=CONFIG_KEY_SCRATCHPAD,
        SEND=CONFIG_KEY_SEND,
        RESUME=RESUME,
        PregelScratchpad=PregelScratchpad,
    )


class TaskResumes:
    """The resume values of one task's interrupts, handed out in the order asked.

    The task's own ``interrupt()`` calls and those of a subgraph it runs (which
    ask the task's scratchpad for a resume value when they have none) share
    one order: the order the interrupts happened in on the run that paused.
    ``nested_missed`` is set when a subgraph asked and none was left: it
    paused, and its interrupt must not be lost.
    """

    def __init__(self, values: Optional[Sequence[Any]] = None) -> None:
        self._values = tuple(values or ())
        self._next = 0
        self._lock = threading.Lock()
        self.nested_missed = False

    @property
    def values(self) -> List[Any]:
        return list(self._values)

    def next_index(self) -> int:
        """The position of the task's next own ``interrupt()`` call."""
        with self._lock:
            index = self._next
            self._next += 1
            return index

    def null_resume(self, consume: bool = False) -> Any:
        """The next resume value, for an interrupt that has none of its own."""
        with self._lock:
            if self._next < len(self._values):
                value = self._values[self._next]
                if consume:
                    self._next += 1
                return value
            if consume:
                self.nested_missed = True
            return None


def interrupt_config(
    *,
    namespace: str,
    step: int,
    stop: Optional[int],
    resume: Optional[List[Any]],
    send: Callable[[Sequence[Tuple[str, Any]]], None],
    resumes: Optional[TaskResumes] = None,
) -> Dict[str, Any]:
    """The ``configurable`` keys ``langgraph.types.interrupt`` reads.

    ``namespace`` names the task, so an interrupt's id is the same on every
    run of it; ``resume`` holds the values the task's interrupts get, in order
    (``resumes`` instead shares them with another scratchpad of the task).
    """
    lg = _internals()
    task = resumes if resumes is not None else TaskResumes(resume)
    calls, subgraphs = itertools.count(), itertools.count()
    scratchpad = lg.PregelScratchpad(
        step=step,
        stop=stop or 0,
        call_counter=lambda: next(calls),
        interrupt_counter=task.next_index,
        get_null_resume=task.null_resume,
        resume=task.values,
        subgraph_counter=lambda: next(subgraphs),
    )
    return {
        lg.SCRATCHPAD: scratchpad,
        lg.CHECKPOINT_NS: namespace,
        lg.SEND: send,
    }


def without_resume_writes(writes: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    """``writes`` minus the bookkeeping ``interrupt()`` records for Pregel."""
    resume = _internals().RESUME
    return [w for w in writes if w[0] != resume]


#: A router run in its own activity (not its node's) cannot pause the run.
ROUTER_INTERRUPT_ERROR = (
    "interrupt() in a router pauses the run only when the router runs in its "
    "node's activity, as it does in runs started on this version; this one ran "
    "on its own (a run started on an older version, or a router that raised "
    "in its node's activity)"
)


def router_interrupt_conf(cond_input: Any) -> Dict[str, Any]:
    """The scratchpad of a router run in its own activity: it can only report."""
    return interrupt_config(
        namespace=f"{cond_input.source_node}:router:{cond_input.condition_name}",
        step=cond_input.step or 0,
        stop=cond_input.recursion_limit,
        resume=None,
        send=lambda writes: None,
    )


class DroppedInterrupt(RuntimeError):
    """A subgraph the node ran paused, and the node returned without its interrupt."""


def nested_interrupts(result: Any, resumes: TaskResumes, node_name: str) -> Any:
    """``result`` of a node that returned, unless a subgraph in it paused.

    A compiled subgraph run as the node (or called by it, its result returned)
    ends with ``{"__interrupt__": [...]}`` instead of raising: that is the
    task's interrupt, raised here as LangGraph's parent task would. A subgraph
    that paused while the node dropped its result raises
    :class:`DroppedInterrupt`: the run must not go on as if it had not.
    """
    from langgraph.errors import GraphInterrupt

    found = result.get(INTERRUPT_KEY) if isinstance(result, dict) else None
    if found:
        raise GraphInterrupt(tuple(found))
    if resumes.nested_missed:
        raise DroppedInterrupt(
            f"A subgraph called by node '{node_name}' paused on interrupt(), "
            "but the node did not return the subgraph's result, so the pause "
            "would be lost. Return the subgraph's result from the node (or add "
            "the compiled subgraph as the node itself)."
        )
    return result


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
        values, resume_id = yield from _wait_resume(run.ctx, run.interrupt_seq, pending)
        run.interrupt = None
        # which resume the run took: a caller whose resume it ignored is told
        run.resumed = {"seq": run.interrupt_seq, "id": resume_id}
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
) -> Generator[Any, Any, Tuple[Dict[str, Any], Optional[str]]]:
    """Wait for the resume of interrupt ``seq``; others (stale) are ignored.

    Returns the resume values and the id the resume was raised with.
    """
    while True:
        event = yield ctx.wait_for_external_event(RESUME_EVENT)
        data = event if isinstance(event, dict) else {"resume": event}
        if data.get("seq") in (None, seq):
            raised = data.get("id")
            return (
                resume_values(pending, data.get("resume")),
                None if raised is None else str(raised),
            )


def resume_taken(status: Dict[str, Any], seq: int, resume_id: str) -> Optional[bool]:
    """Whether the run took the resume ``resume_id`` of interrupt ``seq``.

    ``None`` while it is still waiting for one. A run started on a version
    that does not say which resume it took is taken to have taken this one.
    """
    resumed = status.get(STATUS_RESUMED)
    if isinstance(resumed, dict) and int(resumed.get("seq") or 0) >= seq:
        if int(resumed.get("seq") or 0) > seq:
            return True  # the run is past it: nothing to tell
        return resumed.get("id") in (None, resume_id)
    waiting = pending_interrupt(status)
    if waiting is not None and int(waiting.get("seq") or 0) == seq:
        return None
    return True


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
