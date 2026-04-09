# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Integration tests for DaprWorkflowDeepAgentRunner.

These tests exercise the runner against real Dapr and OpenAI services.
No mocks are used.

Run with a Dapr sidecar:
    dapr run --app-id deepagent-test \
        --resources-path ./diagrid/agent/deepagents/examples/components -- \
        uv run pytest tests/agent/deepagents/test_runner.py -m integration -v

Prerequisites:
    - Dapr initialized with Redis running
    - OPENAI_API_KEY environment variable set
"""

import asyncio
import time
import unittest

import pytest

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool

from diagrid.agent.deepagents import DaprWorkflowDeepAgentRunner
from diagrid.agent.langgraph.workflow import (
    agent_workflow,
    execute_node_activity,
    evaluate_condition_activity,
    clear_registries,
    get_registered_node,
    get_channel_reducer,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together.
    Args: a — first number, b — second number."""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together.
    Args: a — first number, b — second number."""
    return a * b


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_dapr_registration() -> None:
    """Remove Dapr's registration markers so functions can be re-registered."""
    for fn in (agent_workflow, execute_node_activity, evaluate_condition_activity):
        fn.__dict__.pop("_workflow_registered", None)
        fn.__dict__.pop("_activity_registered", None)
        fn.__dict__.pop("_dapr_alternate_name", None)
    clear_registries()


_model = ChatOpenAI(model="gpt-4o-mini")


def _make_graph_with_tools():
    return create_agent(_model, [add_numbers, multiply_numbers])


def _make_simple_graph():
    return create_agent(_model, [])


# ---------------------------------------------------------------------------
# Constructor and property tests
# ---------------------------------------------------------------------------


class TestDeepAgentRunnerInit(unittest.TestCase):
    """Tests for constructor and properties using real graphs."""

    def setUp(self) -> None:
        _clear_dapr_registration()
        self.graph = _make_graph_with_tools()

    def tearDown(self) -> None:
        _clear_dapr_registration()

    def test_agent_maps_to_graph(self):
        runner = DaprWorkflowDeepAgentRunner(agent=self.graph, name="test-init")
        self.assertIs(runner._graph, self.graph)

    def test_agent_property(self):
        runner = DaprWorkflowDeepAgentRunner(agent=self.graph, name="test-prop")
        self.assertIs(runner.agent, self.graph)

    def test_default_max_steps(self):
        runner = DaprWorkflowDeepAgentRunner(agent=self.graph, name="test-steps")
        self.assertEqual(runner._max_steps, 100)

    def test_custom_max_steps(self):
        runner = DaprWorkflowDeepAgentRunner(
            agent=self.graph, name="test-custom", max_steps=50
        )
        self.assertEqual(runner._max_steps, 50)

    def test_name_stored(self):
        runner = DaprWorkflowDeepAgentRunner(
            agent=self.graph, name="my-agent", role="planner", goal="plan"
        )
        self.assertEqual(runner._name, "my-agent")


# ---------------------------------------------------------------------------
# Graph registration tests
# ---------------------------------------------------------------------------


class TestRegisterGraphComponents(unittest.TestCase):
    """Tests for _register_graph_components using real graphs."""

    def setUp(self) -> None:
        _clear_dapr_registration()

    def tearDown(self) -> None:
        _clear_dapr_registration()

    def test_registers_model_node(self):
        graph = _make_graph_with_tools()
        runner = DaprWorkflowDeepAgentRunner(agent=graph, name="test-reg-model")
        runner._register_graph_components()
        self.assertIsNotNone(get_registered_node("model"))

    def test_registers_tools_node(self):
        graph = _make_graph_with_tools()
        runner = DaprWorkflowDeepAgentRunner(agent=graph, name="test-reg-tools")
        runner._register_graph_components()
        self.assertIsNotNone(get_registered_node("tools"))

    def test_skips_start_end(self):
        graph = _make_graph_with_tools()
        runner = DaprWorkflowDeepAgentRunner(agent=graph, name="test-reg-skip")
        runner._register_graph_components()
        self.assertIsNone(get_registered_node("__start__"))
        self.assertIsNone(get_registered_node("__end__"))

    def test_all_real_nodes_registered(self):
        graph = _make_graph_with_tools()
        runner = DaprWorkflowDeepAgentRunner(agent=graph, name="test-reg-all")
        runner._register_graph_components()
        for name in graph.nodes:
            if name in ("__start__", "__end__"):
                continue
            self.assertIsNotNone(
                get_registered_node(name), f"Node {name} not registered"
            )

    def test_registers_messages_reducer(self):
        graph = _make_graph_with_tools()
        runner = DaprWorkflowDeepAgentRunner(agent=graph, name="test-reg-reducer")
        runner._register_graph_components()
        self.assertIsNotNone(get_channel_reducer("messages"))

    def test_clears_prior_registrations(self):
        from diagrid.agent.langgraph.workflow import register_node

        register_node("leftover", lambda x: x)
        self.assertIsNotNone(get_registered_node("leftover"))

        graph = _make_graph_with_tools()
        runner = DaprWorkflowDeepAgentRunner(agent=graph, name="test-reg-clear")
        runner._register_graph_components()
        self.assertIsNone(get_registered_node("leftover"))

    def test_bound_wrapper_preserved(self):
        """Nodes should be registered as RunnableCallable wrappers."""
        graph = _make_graph_with_tools()
        runner = DaprWorkflowDeepAgentRunner(agent=graph, name="test-reg-bound")
        runner._register_graph_components()

        model_node = get_registered_node("model")
        self.assertIsNotNone(model_node)
        self.assertTrue(
            hasattr(model_node, "invoke"),
            "Registered model node should have .invoke (RunnableCallable)",
        )


# ---------------------------------------------------------------------------
# Lifecycle tests (require Dapr sidecar)
# ---------------------------------------------------------------------------


class TestLifecycle(unittest.TestCase):
    """Lifecycle tests — require a running Dapr sidecar."""

    def setUp(self) -> None:
        _clear_dapr_registration()

    def tearDown(self) -> None:
        _clear_dapr_registration()

    def test_start_and_shutdown(self):
        graph = _make_graph_with_tools()
        runner = DaprWorkflowDeepAgentRunner(agent=graph, name="test-lifecycle")
        runner.start()
        try:
            self.assertTrue(runner._started)
            self.assertIsNotNone(runner._workflow_client)
        finally:
            runner.shutdown()
        self.assertFalse(runner._started)


# ---------------------------------------------------------------------------
# End-to-end workflow tests (require Dapr sidecar + OpenAI)
# ---------------------------------------------------------------------------


class TestWorkflowExecution(unittest.TestCase):
    """End-to-end workflow tests with real LLM and Dapr."""

    def setUp(self) -> None:
        _clear_dapr_registration()

    def tearDown(self) -> None:
        _clear_dapr_registration()

    def test_simple_conversation(self):
        """Run a workflow with a simple LLM conversation (no tools)."""
        graph = _make_simple_graph()
        runner = DaprWorkflowDeepAgentRunner(
            agent=graph, name="test-simple-wf", max_steps=10
        )
        runner.start()
        time.sleep(1)

        completed = False

        async def _run():
            nonlocal completed
            async for event in runner.run_async(
                input={
                    "messages": [{"role": "user", "content": "Say hello in one word."}]
                },
                thread_id=f"test-simple-{int(time.time())}",
            ):
                if event["type"] == "workflow_completed":
                    output = event.get("output", {})
                    messages = output.get("messages", [])
                    assert len(messages) > 0
                    completed = True
                    break
                elif event["type"] == "workflow_failed":
                    raise AssertionError(f"Workflow failed: {event.get('error')}")

        try:
            asyncio.run(_run())
        finally:
            runner.shutdown()

        self.assertTrue(completed, "Workflow did not complete")

    def test_tool_calling_workflow(self):
        """Run a workflow that requires the LLM to call tools."""
        graph = _make_graph_with_tools()
        runner = DaprWorkflowDeepAgentRunner(
            agent=graph, name="test-tool-wf", max_steps=20
        )
        runner.start()
        time.sleep(1)

        completed = False
        final_content = ""

        async def _run():
            nonlocal completed, final_content
            async for event in runner.run_async(
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is 7 + 5? Use the add_numbers tool.",
                        }
                    ]
                },
                thread_id=f"test-tools-{int(time.time())}",
            ):
                if event["type"] == "workflow_completed":
                    output = event.get("output", {})
                    messages = output.get("messages", [])
                    assert len(messages) >= 3  # user + tool call + response
                    last = messages[-1]
                    final_content = (
                        last.get("content", "")
                        if isinstance(last, dict)
                        else getattr(last, "content", str(last))
                    )
                    completed = True
                    break
                elif event["type"] == "workflow_failed":
                    raise AssertionError(f"Workflow failed: {event.get('error')}")

        try:
            asyncio.run(_run())
        finally:
            runner.shutdown()

        self.assertTrue(completed, "Workflow did not complete")
        self.assertIn("12", str(final_content), f"Expected 12 in: {final_content}")


if __name__ == "__main__":
    unittest.main()
