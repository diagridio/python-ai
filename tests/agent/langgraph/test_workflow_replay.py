# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""``agent_workflow`` is deterministic: a replay schedules what the first run did.

Dapr replays the orchestrator from its first line for every decision, feeding the
recorded result of each activity. Anything the code invents (a random id) then
differs between the run that scheduled an activity and the replay that consumes
its result.
"""

import json
from typing import Any, Dict, List, Tuple
from unittest import mock

from langgraph.graph import END, START, MessagesState, StateGraph

from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.workflow import (
    agent_workflow,
    set_default_graph_config,
)

from .test_runner import _InlineWorkflowContext, _RunnerTestCase

Recorded = List[Tuple[Any, Any]]


class _RecordingCtx(_InlineWorkflowContext):
    """The first execution: runs each activity and records its input and result."""

    def __init__(self, runner: DaprWorkflowGraphRunner):
        super().__init__(runner)
        self.recorded: Recorded = []

    def call_activity(self, activity: Any, *, input: Any, retry_policy: Any = None):
        result = super().call_activity(activity, input=input, retry_policy=retry_policy)
        self.recorded.append((json.loads(json.dumps(input)), result))
        return result


class _ReplayCtx(_InlineWorkflowContext):
    """A replay: an activity must be scheduled with the recorded input."""

    is_replaying = True

    def __init__(self, runner: DaprWorkflowGraphRunner, recorded: Recorded):
        super().__init__(runner)
        self._recorded = list(recorded)
        self.mismatches: List[Tuple[Any, Any]] = []

    def call_activity(self, activity: Any, *, input: Any, retry_policy: Any = None):
        want_input, result = self._recorded.pop(0)
        got = json.loads(json.dumps(input))
        if got != want_input:
            self.mismatches.append((want_input, got))
        return result


def _drive(ctx: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
    with mock.patch("diagrid.agent.langgraph.workflow.when_all", side_effect=list):
        workflow = agent_workflow(ctx, dict(input_data))
        result = None
        try:
            while True:
                result = workflow.send(result)
        except StopIteration as stop:
            return stop.value  # type: ignore[no-any-return]


class TestReplay(_RunnerTestCase):
    def _runner(self) -> DaprWorkflowGraphRunner:
        def echo(state: MessagesState) -> dict:
            # returns the messages it was handed: add_messages dedupes by id
            return {"messages": state["messages"]}

        graph = StateGraph(MessagesState)
        graph.add_node("echo", echo)
        graph.add_edge(START, "echo")
        graph.add_edge("echo", END)
        runner = DaprWorkflowGraphRunner(graph=graph.compile(), name="replay")
        set_default_graph_config(runner.graph_config, max_steps=10)
        return runner

    def test_a_replay_schedules_the_same_activity_inputs(self):
        runner = self._runner()
        first = _RecordingCtx(runner)
        out = _drive(first, {"task": "hello"})
        replay = _ReplayCtx(runner, first.recorded)
        replayed = _drive(replay, {"task": "hello"})
        self.assertEqual(replay.mismatches, [])
        self.assertEqual(replayed, out)

    def test_the_message_is_not_duplicated_by_a_replay(self):
        runner = self._runner()
        first = _RecordingCtx(runner)
        _drive(first, {"task": "hello"})
        replayed = _drive(_ReplayCtx(runner, first.recorded), {"task": "hello"})
        contents = [m["content"] for m in replayed["output"]["messages"]]
        self.assertEqual(contents, ["hello"])

    def test_two_instances_of_one_input_get_ids_of_their_own(self):
        runner = self._runner()

        def ids(instance_id: str) -> List[str]:
            ctx = _RecordingCtx(runner)
            ctx.instance_id = instance_id
            out = _drive(ctx, {"task": "hello"})
            return [m["id"] for m in out["output"]["messages"]]

        self.assertNotEqual(ids("wf-one"), ids("wf-two"))
        self.assertEqual(ids("wf-one"), ids("wf-one"))
