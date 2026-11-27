# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Stable ids for the interrupts of a subgraph a task runs, and resumes by id.

LangGraph names an interrupt by its task's checkpoint namespace. A subgraph
the node runs starts afresh on every run of the task (its checkpoint is not
kept), so the namespaces of its tasks, and the ids of their interrupts, change
from one run to the next. Two of its nodes that pause in parallel ask the
task's scratchpad for their resume values in whatever order they get there.

So a nested interrupt is reported under an id made of the task's namespace and
the path of node names down to it (stable on every run), and the workflow
hands its resume value to the task tagged with that id: each interrupt gets
its own value, and one left unanswered keeps its id. Interrupts whose paths
collide (a subgraph that sends two tasks to one node) keep LangGraph's ids and
take their values in order; an id map that answers only some of them is
refused.
"""

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: The key of a resume value the workflow tags with the interrupt it is for.
RESUME_FOR = "__diagrid_resume_for__"
#: An interrupt payload's key holding the path of a nested interrupt.
PATH_KEY = "path"
#: LangGraph's separators in a checkpoint namespace: between levels, and
#: between a node's name and its task id (neither is allowed in a node name).
_NS_SEP = "|"
_NS_END = ":"


def nested_ids(namespace: str) -> Optional[Tuple[str, str, str]]:
    """``(LangGraph's id, stable id, path)`` of the interrupt asking now.

    ``None`` when the one asking is the task's own (its namespace is the
    task's), or when it cannot be told (no runnable config).
    """
    from langgraph._internal._constants import CONFIG_KEY_CHECKPOINT_NS
    from langgraph.config import get_config
    from langgraph.types import Interrupt

    try:
        ns = get_config().get("configurable", {}).get(CONFIG_KEY_CHECKPOINT_NS)
    except RuntimeError:  # not inside a runnable
        return None
    if not ns or ns == namespace:
        return None
    path = _NS_SEP.join(part.rsplit(_NS_END, 1)[0] for part in ns.split(_NS_SEP))
    stable = hashlib.sha256(f"{namespace}{_NS_SEP}{path}".encode()).hexdigest()[:32]
    return Interrupt.from_ns(value=None, ns=ns).id, stable, path


def tagged(interrupt_id: str, value: Any) -> Dict[str, Any]:
    """``value`` tagged with the nested interrupt it answers (see :func:`split`)."""
    return {RESUME_FOR: interrupt_id, "value": value}


def split(values: Sequence[Any]) -> Tuple[List[Any], Dict[str, List[Any]]]:
    """A task's resume values: ``(in order, by nested interrupt id)``.

    A nested node that asks again keeps its id (as LangGraph's interrupts of
    one task share theirs), so each id has its answers in the order given.
    """
    ordered: List[Any] = []
    by_id: Dict[str, List[Any]] = {}
    for value in values:
        if isinstance(value, dict) and set(value) == {RESUME_FOR, "value"}:
            by_id.setdefault(str(value[RESUME_FOR]), []).append(value["value"])
        else:
            ordered.append(value)
    return ordered, by_id


def partial_map_error(pending: Sequence[Dict[str, Any]], resume: Any) -> Optional[str]:
    """Why ``resume`` cannot resume ``pending``, or ``None`` when it can.

    An id map that leaves out an interrupt of a task with several pending,
    where one of them has no stable id, cannot say which value is whose.
    """
    ids = {str(p.get("id")) for p in pending}
    if not (isinstance(resume, dict) and resume and set(resume) <= ids):
        return None
    by_task: Dict[Any, List[Dict[str, Any]]] = {}
    for item in pending:
        by_task.setdefault(item.get("task"), []).append(item)
    for items in by_task.values():
        answered = [str(i.get("id")) in resume for i in items]
        unstable = any(not i.get(PATH_KEY) for i in items)
        if len(items) > 1 and unstable and not all(answered):
            node = items[0].get("node")
            return (
                f"node {node!r} waits on {len(items)} interrupts of tasks a "
                "subgraph sent to one node, which keep no id across runs: "
                "answer every interrupt of the node in one resume (an id map "
                "naming them all, or one value for all)"
            )
    return None
