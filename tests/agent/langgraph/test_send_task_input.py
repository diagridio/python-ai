# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""What the workflow hands the activity of a ``Send`` task.

A ``Send`` task is numbered once, over the whole step, and that number (its
slot) is all the activity needs: the path of its chunks and the key of its
attempts both follow from it.
"""

import json
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from diagrid.agent.langgraph import DaprWorkflowGraphRunner

from .golden_harness import _Out
from .streaming_helpers import _IdCtx
from .test_runner import _run, _RunnerTestCase


class _RecordingCtx(_IdCtx):
    """Keeps what each node activity was scheduled with."""

    def __init__(self, runner: Any, workflow_id: str) -> None:
        super().__init__(runner, workflow_id)
        self.node_inputs: List[Dict[str, Any]] = []

    def call_activity(self, activity: Any, *, input: Any, retry_policy: Any = None):
        self.node_inputs.append(input)
        return super().call_activity(activity, input=input, retry_policy=retry_policy)


def _graph() -> Any:
    def fan(state: _Out) -> Any:
        return Command(goto=[Send("work", {"i": i}) for i in (1, 2)])

    graph = StateGraph(_Out)
    graph.add_node("fan", fan)
    graph.add_node("work", lambda arg: {"out": [arg["i"]]})
    graph.add_edge(START, "fan")
    graph.add_edge("work", END)
    return graph.compile()


class TestSendTaskInput(_RunnerTestCase):
    def test_a_send_task_carries_its_argument_and_one_slot(self):
        runner = DaprWorkflowGraphRunner(graph=_graph(), name="send-input")
        ctx = _RecordingCtx(runner, "wf-send-input")
        out = _run(runner, {"out": []}, ctx)
        self.assertEqual(out.status, "completed", out.error)
        sends = [i["send"] for i in ctx.node_inputs if i.get("send")]
        self.assertEqual([sorted(s) for s in sends], [["arg", "slot"]] * 2)
        self.assertEqual([s["slot"] for s in sends], [0, 1])
        json.dumps(sends)
