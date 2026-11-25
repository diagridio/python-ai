# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Ids of messages a node writes are made in the activity, so a replay keeps them.

``add_messages`` gives an id-less message a random one. Run in the orchestrator it
would invent a new id on every replay, and a later ``RemoveMessage(id=...)`` (the
trim and summarise pattern) would name an id the replay never made.
"""

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.workflow import set_default_graph_config

from .test_runner import _RunnerTestCase
from .test_workflow_replay import _drive, _RecordingCtx, _ReplayCtx


def _runner(*nodes: Any) -> DaprWorkflowGraphRunner:
    graph = StateGraph(MessagesState)
    names = []
    for i, fn in enumerate(nodes):
        graph.add_node(f"n{i}", fn)
        names.append(f"n{i}")
    graph.add_edge(START, names[0])
    for a, b in zip(names, names[1:]):
        graph.add_edge(a, b)
    graph.add_edge(names[-1], END)
    runner = DaprWorkflowGraphRunner(graph=graph.compile(), name="ids")
    set_default_graph_config(runner.graph_config, max_steps=10)
    return runner


class TestWrittenMessageIds(_RunnerTestCase):
    def _record_and_replay(self, runner: DaprWorkflowGraphRunner):
        first = _RecordingCtx(runner)
        out = _drive(first, {"task": "hello"})
        replay = _ReplayCtx(runner, first.recorded)
        replayed = _drive(replay, {"task": "hello"})
        return out, replayed, replay

    def test_a_remove_message_of_an_id_less_message_survives_a_replay(self):
        def write(state: MessagesState) -> dict:
            return {"messages": [AIMessage(content="x")]}

        def trim(state: MessagesState) -> dict:
            return {"messages": [RemoveMessage(id=state["messages"][-1].id)]}

        out, replayed, replay = self._record_and_replay(_runner(write, trim))
        self.assertEqual(out["status"], "completed")
        self.assertEqual([m["content"] for m in out["output"]["messages"]], ["hello"])
        self.assertEqual(replay.mismatches, [])
        self.assertEqual(replayed, out)

    def test_a_replay_gives_a_tool_message_the_id_the_node_saw(self):
        def write(state: MessagesState) -> dict:
            return {"messages": [ToolMessage(content="r", tool_call_id="c1")]}

        def trim(state: MessagesState) -> dict:
            return {"messages": [RemoveMessage(id=state["messages"][-1].id)]}

        out, replayed, replay = self._record_and_replay(_runner(write, trim))
        self.assertEqual(replay.mismatches, [])
        self.assertEqual(replayed, out)

    def test_every_written_message_has_an_id(self):
        def write(state: MessagesState) -> dict:
            return {"messages": [AIMessage(content="x"), ("assistant", "y")]}

        out, _, _ = self._record_and_replay(_runner(write))
        ids: List[Any] = [m.get("id") for m in out["output"]["messages"]]
        self.assertTrue(all(isinstance(i, str) and i for i in ids), ids)
        self.assertEqual(len(set(ids)), len(ids))


class TestReducerFailure(_RunnerTestCase):
    def test_a_reducer_that_raises_fails_the_run_instead_of_overwriting(self):
        from typing import Annotated

        from typing_extensions import TypedDict

        def strict(old: list, new: list) -> list:
            if "bad" in new:
                raise ValueError("no bad values")
            return old + new

        class State(TypedDict):
            log: Annotated[list, strict]

        graph = StateGraph(State)
        graph.add_node("a", lambda s: {"log": ["ok"]})
        graph.add_node("b", lambda s: {"log": ["bad"]})
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        runner = DaprWorkflowGraphRunner(graph=graph.compile(), name="strict")
        set_default_graph_config(runner.graph_config, max_steps=10)
        out = _drive(_RecordingCtx(runner), {"log": []})
        self.assertEqual(out["status"], "error")
        self.assertIn("log", out["error"])
        self.assertIn("no bad values", out["error"])
        self.assertEqual(out["output"], {"log": ["ok"]})
