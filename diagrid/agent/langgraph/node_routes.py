# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A node's conditional edges, evaluated inside the node's activity.

LangGraph attaches a router to its source node as one of the node's writers, so
it runs inside the node's task, right after the node returns, and its chunks
come before the node's ``updates`` chunk. The node activity does the same when
the workflow hands it the node's routes (``ExecuteNodeInput.routes``).

A router reads the state the step started from plus this task's own writes. A
router that raises is not retried here: the activity reports its routes as not
done and the workflow evaluates them in the separate condition activity, so the
node itself does not run again.
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from diagrid.agent.core.workflow.stream import StreamSink

from .models import (
    ChannelState,
    EvaluateConditionInput,
    ExecuteNodeInput,
    NodeWrite,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteOutcome:
    """What a node's conditional edges decided, as the activity reports it.

    ``done`` is false when a router raised: the workflow then evaluates the
    node's routes itself. ``error`` fails the run (a ``path_map`` miss, a
    reducer that raised on this task's writes).
    """

    done: bool
    next_nodes: List[str] = field(default_factory=list)
    sends: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def output_fields(self) -> Dict[str, Any]:
        """The ``ExecuteNodeOutput`` fields this outcome sets."""
        return {
            "routes_done": self.done,
            "next_nodes": list(self.next_nodes),
            "route_sends": list(self.sends),
            "route_error": self.error,
        }


def _router_state(start: ChannelState, writes: List[NodeWrite]) -> ChannelState:
    """The state the step started from with this task's ``writes`` applied."""
    from .workflow import _apply_write

    state = copy.deepcopy(start)
    for write in writes:
        _apply_write(state, write.channel, write.value)
    return state


def _route_input(
    node_input: ExecuteNodeInput, route: Dict[str, Any], state: ChannelState
) -> EvaluateConditionInput:
    """What the condition activity would be given for ``route`` of this task."""
    send = node_input.send
    return EvaluateConditionInput(
        source_node=node_input.node_name,
        condition_name=route["condition"],
        channel_state=state,
        config=node_input.config,
        thread_id=node_input.thread_id,
        path_map=route.get("path_map"),
        step=node_input.step,
        send_slot=None if send is None else send["slot"],
        recursion_limit=node_input.recursion_limit,
    )


def route_in_node(
    ctx: Any,
    node_input: ExecuteNodeInput,
    writes: List[NodeWrite],
    sink: Optional[StreamSink],
    interrupt_conf: Optional[Dict[str, Any]] = None,
) -> RouteOutcome:
    """Evaluate the node's conditional edges on the state with ``writes`` applied.

    The routers stream as they do in the condition activity: as the source
    node, keyed by edge and task, so a later fallback attempt of one reads as
    that router's retry, not the node's. ``interrupt_conf`` is the task's
    ``interrupt()`` scratchpad: a router that interrupts pauses the task.
    """
    from langgraph.errors import GraphInterrupt

    from .workflow import ReducerError, evaluate_condition

    try:
        state = _router_state(node_input.channel_state, writes)
    except ReducerError as e:
        # Raising would make Dapr retry the node; the run fails with it instead.
        return RouteOutcome(done=True, error=str(e))
    next_nodes: List[str] = []
    sends: List[Dict[str, Any]] = []
    for route in node_input.routes or []:
        cond_input = _route_input(node_input, route, state)
        try:
            output = evaluate_condition(ctx, cond_input, sink, interrupt_conf)
        except GraphInterrupt:
            raise
        except Exception:
            logger.warning(
                "Router %r of node %r raised inside the node's activity; the "
                "workflow evaluates the node's routes in their own activity",
                cond_input.condition_name,
                node_input.node_name,
                exc_info=True,
            )
            return RouteOutcome(done=False)
        if output.error:
            return RouteOutcome(done=True, error=output.error)
        next_nodes.extend(output.next_nodes)
        sends.extend(output.sends)
    return RouteOutcome(done=True, next_nodes=next_nodes, sends=sends)
