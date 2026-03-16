# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for DaprWorkflowDeepAgentRunner."""

import unittest
from unittest import mock
from typing import Any

from diagrid.agent.langgraph.workflow import (
    clear_registries,
    get_registered_node,
    get_channel_reducer,
)


class FakeNodeSpec:
    """Node spec with a .bound attribute (RunnableCallable wrapper)."""

    def __init__(self, bound: Any) -> None:
        self.bound = bound


class FakeRunnableNodeSpec:
    """Node spec with a .runnable attribute."""

    def __init__(self, runnable: Any) -> None:
        self.runnable = runnable


class FakeChannel:
    """Channel with a .reducer attribute."""

    def __init__(self, reducer: Any = None, operator: Any = None) -> None:
        if reducer is not None:
            self.reducer = reducer
        if operator is not None:
            self.operator = operator


class FakeGraph:
    """Minimal fake CompiledStateGraph."""

    def __init__(
        self,
        nodes: dict | None = None,
        channels: dict | None = None,
    ) -> None:
        self.nodes = nodes or {}
        self.channels = channels or {}


@mock.patch("diagrid.agent.core.workflow.runner.WorkflowRuntime")
@mock.patch("diagrid.agent.core.workflow.runner.DaprWorkflowClient")
class TestDeepAgentRunnerInit(unittest.TestCase):
    """Tests for constructor and property."""

    def test_agent_maps_to_graph(self, mock_client, mock_runtime):
        from diagrid.agent.deepagents.runner import DaprWorkflowDeepAgentRunner

        graph = FakeGraph()
        runner = DaprWorkflowDeepAgentRunner(agent=graph, name="test")
        self.assertIs(runner._graph, graph)

    def test_agent_property(self, mock_client, mock_runtime):
        from diagrid.agent.deepagents.runner import DaprWorkflowDeepAgentRunner

        graph = FakeGraph()
        runner = DaprWorkflowDeepAgentRunner(agent=graph, name="test")
        self.assertIs(runner.agent, graph)

    def test_default_max_steps(self, mock_client, mock_runtime):
        from diagrid.agent.deepagents.runner import DaprWorkflowDeepAgentRunner

        graph = FakeGraph()
        runner = DaprWorkflowDeepAgentRunner(agent=graph, name="test")
        self.assertEqual(runner._max_steps, 100)

    def test_custom_max_steps(self, mock_client, mock_runtime):
        from diagrid.agent.deepagents.runner import DaprWorkflowDeepAgentRunner

        graph = FakeGraph()
        runner = DaprWorkflowDeepAgentRunner(agent=graph, name="test", max_steps=50)
        self.assertEqual(runner._max_steps, 50)

    def test_optional_params_passed_through(self, mock_client, mock_runtime):
        from diagrid.agent.deepagents.runner import DaprWorkflowDeepAgentRunner

        graph = FakeGraph()
        runner = DaprWorkflowDeepAgentRunner(
            agent=graph,
            name="test",
            host="myhost",
            port="3500",
            role="planner",
            goal="plan things",
        )
        self.assertEqual(runner._name, "test")


class TestRegisterGraphComponents(unittest.TestCase):
    """Tests for _register_graph_components override."""

    def setUp(self) -> None:
        clear_registries()

    def tearDown(self) -> None:
        clear_registries()

    @mock.patch("diagrid.agent.core.workflow.runner.WorkflowRuntime")
    @mock.patch("diagrid.agent.core.workflow.runner.DaprWorkflowClient")
    def _make_runner(self, graph, mock_client, mock_runtime):
        from diagrid.agent.deepagents.runner import DaprWorkflowDeepAgentRunner

        return DaprWorkflowDeepAgentRunner(agent=graph, name="test")

    def test_registers_bound_wrapper(self):
        """Should register node_spec.bound, not bound.func."""
        bound_obj = mock.MagicMock(name="RunnableCallable")
        node_spec = FakeNodeSpec(bound=bound_obj)
        graph = FakeGraph(nodes={"agent": node_spec})

        runner = self._make_runner(graph)
        runner._register_graph_components()

        registered = get_registered_node("agent")
        self.assertIs(registered, bound_obj)

    def test_registers_runnable_fallback(self):
        """Should fall back to node_spec.runnable if no .bound."""
        runnable_obj = mock.MagicMock(name="Runnable")
        node_spec = FakeRunnableNodeSpec(runnable=runnable_obj)
        graph = FakeGraph(nodes={"tools": node_spec})

        runner = self._make_runner(graph)
        runner._register_graph_components()

        registered = get_registered_node("tools")
        self.assertIs(registered, runnable_obj)

    def test_registers_callable_fallback(self):
        """Should fall back to the node_spec itself if callable."""

        def my_func(state: dict) -> dict:
            return state

        graph = FakeGraph(nodes={"simple": my_func})

        runner = self._make_runner(graph)
        runner._register_graph_components()

        registered = get_registered_node("simple")
        self.assertIs(registered, my_func)

    def test_skips_start_end_nodes(self):
        """Should skip __start__, __end__, START, END sentinel nodes."""
        func = mock.MagicMock()
        graph = FakeGraph(
            nodes={
                "__start__": func,
                "__end__": func,
                "agent": FakeNodeSpec(bound=func),
            }
        )

        runner = self._make_runner(graph)
        runner._register_graph_components()

        self.assertIsNone(get_registered_node("__start__"))
        self.assertIsNone(get_registered_node("__end__"))
        self.assertIs(get_registered_node("agent"), func)

    def test_warns_on_non_extractable_node(self):
        """Should warn if node_spec has no bound/runnable and isn't callable."""
        node_spec = object()  # not callable, no bound/runnable
        graph = FakeGraph(nodes={"bad_node": node_spec})

        runner = self._make_runner(graph)
        with self.assertLogs(
            "diagrid.agent.deepagents.runner", level="WARNING"
        ) as logs:
            runner._register_graph_components()

        self.assertTrue(any("bad_node" in msg for msg in logs.output))

    def test_registers_channel_reducer(self):
        """Should register channel with .reducer attribute."""

        def add_reducer(a: list, b: list) -> list:
            return a + b

        channel = FakeChannel(reducer=add_reducer)
        graph = FakeGraph(
            nodes={},
            channels={"messages": channel},
        )

        runner = self._make_runner(graph)
        runner._register_graph_components()

        registered = get_channel_reducer("messages")
        self.assertIs(registered, add_reducer)

    def test_registers_channel_operator(self):
        """Should register channel using .operator when .reducer is absent."""

        def op(a: list, b: list) -> list:
            return a + b

        channel = FakeChannel(operator=op)
        graph = FakeGraph(
            nodes={},
            channels={"messages": channel},
        )

        runner = self._make_runner(graph)
        runner._register_graph_components()

        registered = get_channel_reducer("messages")
        self.assertIs(registered, op)

    def test_skips_channel_without_reducer(self):
        """Channels with no reducer or operator should be skipped."""
        channel = object()  # no reducer, no operator
        graph = FakeGraph(
            nodes={},
            channels={"plain": channel},
        )

        runner = self._make_runner(graph)
        runner._register_graph_components()

        self.assertIsNone(get_channel_reducer("plain"))

    def test_clears_registries_on_call(self):
        """Should call clear_registries at the start."""
        from diagrid.agent.langgraph.workflow import register_node as reg

        reg("leftover", lambda x: x)
        self.assertIsNotNone(get_registered_node("leftover"))

        graph = FakeGraph(nodes={})
        runner = self._make_runner(graph)
        runner._register_graph_components()

        self.assertIsNone(get_registered_node("leftover"))

    @mock.patch("diagrid.agent.deepagents.runner.set_serializer")
    def test_sets_json_plus_serializer(self, mock_set_serializer):
        """Should set JsonPlusSerializer when available."""
        graph = FakeGraph(nodes={})
        runner = self._make_runner(graph)
        mock_set_serializer.reset_mock()
        runner._register_graph_components()

        mock_set_serializer.assert_called_once()

    @mock.patch(
        "diagrid.agent.deepagents.runner.set_serializer",
        side_effect=ImportError("no module"),
    )
    def test_handles_serializer_import_error(self, mock_set_serializer):
        """Should warn but not raise if JsonPlusSerializer is unavailable."""
        graph = FakeGraph(nodes={})
        runner = self._make_runner(graph)
        # Should not raise
        runner._register_graph_components()


if __name__ == "__main__":
    unittest.main()
