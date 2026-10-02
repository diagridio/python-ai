# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for DaprWorkflowGraphRunner, running real graphs end to end in-process.

``_run`` drives ``agent_workflow`` with a stand-in workflow context that executes
each activity inline and JSON round-trips its input and output, as Dapr does, so
whole graphs run through the durable runtime without a sidecar.
"""

import json
import unittest
from typing import Any, Dict, List, Literal, Optional, TypedDict
from unittest import mock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime

from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.models import (
    ChannelState,
    GraphWorkflowInput,
    GraphWorkflowOutput,
)
from diagrid.agent.langgraph.workflow import (
    agent_workflow,
    clear_registries,
    evaluate_condition_activity,
    execute_node_activity,
    get_registered_node,
)


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def _clear_dapr_registration() -> None:
    """Remove Dapr's registration markers so the workflow can be re-registered."""
    for fn in (agent_workflow, execute_node_activity, evaluate_condition_activity):
        fn.__dict__.pop("_workflow_registered", None)
        fn.__dict__.pop("_activity_registered", None)
        fn.__dict__.pop("_dapr_alternate_name", None)
    clear_registries()


class _ActivityCtx:
    workflow_id = "wf-test"
    task_id = 1


class _InlineWorkflowContext:
    """Runs activities inline, crossing JSON both ways like the Dapr engine."""

    is_replaying = False
    instance_id = "wf-test"

    def call_activity(self, activity: Any, *, input: Any, retry_policy: Any = None):
        result = activity(_ActivityCtx(), json.loads(json.dumps(input)))
        return json.loads(json.dumps(result))


def _run(runner: DaprWorkflowGraphRunner, graph_input: Dict[str, Any]):
    """Run the graph through ``agent_workflow`` and return its output."""
    workflow_input = GraphWorkflowInput(
        graph_config=runner.graph_config,
        channel_state=ChannelState(
            values=runner._serialize_input(graph_input),
            versions={k: 1 for k in graph_input},
            updated_channels=list(graph_input),
        ),
        step=0,
        max_steps=20,
        thread_id="thread-test",
    )
    with mock.patch("diagrid.agent.langgraph.workflow.when_all", side_effect=list):
        workflow = agent_workflow(
            _InlineWorkflowContext(), json.loads(json.dumps(workflow_input.to_dict()))
        )
        result = None
        try:
            while True:
                result = workflow.send(result)
        except StopIteration as stop:
            return GraphWorkflowOutput.from_dict(stop.value)


def _call_add_then_answer(state: MessagesState) -> dict:
    """Model stand-in: request ``add(2, 3)``, then answer once it has the result."""
    if any(isinstance(m, ToolMessage) for m in state["messages"]):
        return {"messages": [AIMessage(content="done")]}
    call = {"name": "add", "args": {"a": 2, "b": 3}, "id": "call-1"}
    return {"messages": [AIMessage(content="", tool_calls=[call])]}


class _ScriptedChatModel(BaseChatModel):
    """Chat model that calls ``add(2, 3)`` once, then answers."""

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedChatModel":
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        state: MessagesState = {"messages": messages}
        message = _call_add_then_answer(state)["messages"][0]
        return ChatResult(generations=[ChatGeneration(message=message)])


class _Value(TypedDict, total=False):
    value: int
    path: str


def _value_graph(router: Any, path_map: Any = None) -> Any:
    """START -> classify -> (router) -> high | low -> END."""
    graph = StateGraph(_Value)
    graph.add_node("classify", lambda state: {"path": ""})
    graph.add_node("high", lambda state: {"path": "high"})
    graph.add_node("low", lambda state: {"path": "low"})
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", router, path_map)
    graph.add_edge("high", END)
    graph.add_edge("low", END)
    return graph.compile()


def _by_size(state: _Value) -> str:
    return "big" if state["value"] > 100 else "small"


def _by_size_typed(state: _Value) -> Literal["high", "low"]:
    return "high" if state["value"] > 100 else "low"


class _RunnerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        _clear_dapr_registration()

    def tearDown(self) -> None:
        _clear_dapr_registration()


class TestNodeRegistration(_RunnerTestCase):
    """Nodes are registered as their Runnable wrappers, not the raw functions."""

    def test_function_node_registered_as_runnable(self):
        graph = StateGraph(MessagesState)
        graph.add_node("agent", _call_add_then_answer)
        graph.add_edge(START, "agent")
        DaprWorkflowGraphRunner(graph=graph.compile(), name="reg-function")

        self.assertTrue(hasattr(get_registered_node("agent"), "invoke"))

    def test_tool_node_registered_as_itself(self):
        tool_node = ToolNode([add])
        graph = StateGraph(MessagesState)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "tools")
        DaprWorkflowGraphRunner(graph=graph.compile(), name="reg-toolnode")

        self.assertIs(get_registered_node("tools"), tool_node)

    def test_async_node_registered(self):
        async def node(state: _Value) -> dict:
            return {"value": 1}

        graph = StateGraph(_Value)
        graph.add_node("node", node)
        graph.add_edge(START, "node")
        DaprWorkflowGraphRunner(graph=graph.compile(), name="reg-async")

        self.assertIsNotNone(get_registered_node("node"))


class TestConditionalEdgePathMap(_RunnerTestCase):
    """The conditional edge's path_map is carried into the graph config."""

    def _condition_edge(self, compiled: Any):
        runner = DaprWorkflowGraphRunner(graph=compiled, name="path-map-config")
        (edge,) = [e for e in runner.graph_config.edges if e.condition]
        return edge

    def test_dict_path_map_recorded(self):
        edge = self._condition_edge(
            _value_graph(_by_size, {"big": "high", "small": "low"})
        )
        self.assertEqual(edge.path_map, {"big": "high", "small": "low"})

    def test_list_path_map_recorded(self):
        edge = self._condition_edge(_value_graph(_by_size_typed, ["high", "low"]))
        self.assertEqual(edge.path_map, {"high": "high", "low": "low"})

    def test_path_map_inferred_from_literal_return(self):
        edge = self._condition_edge(_value_graph(_by_size_typed))
        self.assertEqual(edge.path_map, {"high": "high", "low": "low"})

    def test_no_path_map(self):
        edge = self._condition_edge(_value_graph(_by_size))
        self.assertIsNone(edge.path_map)


class TestRunGraph(_RunnerTestCase):
    """Whole graphs run through the durable runtime."""

    def test_tool_node_graph(self):
        graph = StateGraph(MessagesState)
        graph.add_node("agent", _call_add_then_answer)
        graph.add_node("tools", ToolNode([add]))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")
        runner = DaprWorkflowGraphRunner(graph=graph.compile(), name="run-toolnode")

        out = _run(runner, {"messages": [{"role": "user", "content": "2 + 3?"}]})

        self.assertEqual(out.status, "completed", out.error)
        messages = out.output["messages"]
        tool_results = [m for m in messages if m["type"] == "tool"]
        self.assertEqual([m["content"] for m in tool_results], ["5"])
        self.assertEqual(messages[-1]["content"], "done")

    def test_node_with_runtime_parameter(self):
        def capture(state: _Value, runtime: Runtime) -> dict:
            info = runtime.execution_info
            return {"path": info.thread_id if info else "no execution info"}

        graph = StateGraph(_Value)
        graph.add_node("capture", capture)
        graph.add_edge(START, "capture")
        graph.add_edge("capture", END)
        runner = DaprWorkflowGraphRunner(graph=graph.compile(), name="run-runtime")

        out = _run(runner, {"value": 1})

        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.output["path"], "thread-test")

    def test_async_node(self):
        async def double(state: _Value) -> dict:
            return {"value": state["value"] * 2}

        graph = StateGraph(_Value)
        graph.add_node("double", double)
        graph.add_edge(START, "double")
        graph.add_edge("double", END)
        runner = DaprWorkflowGraphRunner(graph=graph.compile(), name="run-async")

        out = _run(runner, {"value": 21})

        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.output["value"], 42)

    def test_async_condition(self):
        async def route(state: _Value) -> str:
            return "high" if state["value"] > 100 else "low"

        runner = DaprWorkflowGraphRunner(
            graph=_value_graph(route), name="run-async-condition"
        )

        out = _run(runner, {"value": 150})

        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.output["path"], "high")

    def test_path_map_labels_route_to_nodes(self):
        runner = DaprWorkflowGraphRunner(
            graph=_value_graph(_by_size, {"big": "high", "small": "low"}),
            name="run-path-map",
        )

        self.assertEqual(_run(runner, {"value": 150}).output["path"], "high")
        self.assertEqual(_run(runner, {"value": 5}).output["path"], "low")

    def test_label_missing_from_path_map_is_an_error(self):
        runner = DaprWorkflowGraphRunner(
            graph=_value_graph(_by_size, {"big": "high"}), name="run-bad-label"
        )

        out = _run(runner, {"value": 5})

        self.assertEqual(out.status, "error")
        self.assertIn("returned 'small', which is not in its path_map", out.error)

    def test_create_agent_graph(self):
        from langchain.agents import create_agent

        from diagrid.agent.deepagents import DaprWorkflowDeepAgentRunner

        runner = DaprWorkflowDeepAgentRunner(
            agent=create_agent(_ScriptedChatModel(), [add]), name="run-create-agent"
        )

        out = _run(runner, {"messages": [{"role": "user", "content": "2 + 3?"}]})

        self.assertEqual(out.status, "completed", out.error)
        messages = out.output["messages"]
        self.assertEqual([m["content"] for m in messages if m["type"] == "tool"], ["5"])
        self.assertEqual(messages[-1]["content"], "done")


if __name__ == "__main__":
    unittest.main()
