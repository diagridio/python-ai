# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""``stream_mode="values"``: the graph's output state after each step.

LangGraph sends the output channels after a step that wrote one of them (and
after the input, when the input wrote one). Here the state after step s is the
input of every activity of step s+1, so the first of them to start sends it,
keyed by the step it opens; the sink keeps the first. The state after the last
step is in the run's final event.
"""

import uuid
from typing import Any, Dict, Iterable, List, Optional, Set

from diagrid.agent.core.workflow.stream import VALUES, StreamEvent

from .models import ChannelState

#: The sink key every ``values`` chunk of one step shares.
VALUES_KEY = "__values__"

# The one channel of a graph whose whole state is a single value.
_ROOT = "__root__"

# The statuses whose final state LangGraph still sends: a run that completed,
# and one cut by its step limit (native sends the last step's values, then
# raises). A run that failed in a node or a router never applied that step.
_FINAL_VALUE_STATUSES = ("completed", "max_steps_reached")


def wants_values(updated_channels: Iterable[str], output_channels: List[str]) -> bool:
    """Whether a step that wrote ``updated_channels`` has a ``values`` chunk."""
    return not set(output_channels).isdisjoint(updated_channels)


def output_values(state: Dict[str, Any], output_channels: List[str]) -> Any:
    """The chunk itself: the output channels that hold a value, as LangGraph reads them."""
    if output_channels == [_ROOT]:
        return state.get(_ROOT)
    return {name: state[name] for name in output_channels if name in state}


def values_event(workflow_id: str, step: int, chunk: Any) -> StreamEvent:
    """The ``values`` chunk of the state step ``step`` starts from (0: the input)."""
    return StreamEvent(
        workflow_id=workflow_id,
        step=step,
        node=VALUES_KEY,
        attempt_id=uuid.uuid4().hex,
        seq=0,
        ns=(),
        mode=VALUES,
        data=chunk,
    )


def state_values(
    workflow_id: str,
    step: int,
    channel_state: ChannelState,
    output_channels: List[str],
) -> Optional[StreamEvent]:
    """The ``values`` chunk of ``channel_state``, or ``None`` if it has none."""
    if not wants_values(channel_state.updated_channels, output_channels):
        return None
    from .workflow import _reconstruct_state

    state = _reconstruct_state(channel_state)
    return values_event(workflow_id, step, output_values(state, output_channels))


def final_values(
    final: Dict[str, Any],
    output_channels: List[str],
    sent_steps: Set[int],
    workflow_id: str,
) -> Optional[StreamEvent]:
    """The ``values`` chunk of the state a run ended with, if it is due.

    ``final`` is the run's terminal event, with its stream-only channel state;
    ``sent_steps`` are the steps whose ``values`` chunk was already sent.
    """
    from .streaming import FINAL_STATE_KEY

    steps = final.get("steps")
    stored = final.get(FINAL_STATE_KEY)
    if final.get("status") not in _FINAL_VALUE_STATUSES or not isinstance(steps, int):
        return None
    if not isinstance(stored, dict) or steps in sent_steps:
        return None
    return state_values(
        workflow_id, steps, ChannelState.from_dict(stored), output_channels
    )
