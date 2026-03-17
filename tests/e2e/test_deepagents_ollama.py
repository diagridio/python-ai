"""E2E tests for Deep Agents with Ollama backend via Dapr Workflows."""

import threading
import uuid
from typing import List, TypedDict

import pytest

from tests.e2e.conftest import clear_dapr_registration, import_deepagents


# ---------------------------------------------------------------------------
# Test A: Workflow lifecycle (from simple_agent.py)
# ---------------------------------------------------------------------------


@pytest.mark.ollama
@pytest.mark.integration
def test_deepagents_workflow_lifecycle(
    ollama_model: str,
) -> None:
    """Test that a Deep Agent completes a full workflow cycle.

    Creates a Deep Agent with a deterministic tool, runs it through the
    Dapr Workflow runner, and asserts on workflow lifecycle.
    """
    from langchain_core.tools import tool

    create_deep_agent = import_deepagents()

    from diagrid.agent.deepagents import DaprWorkflowDeepAgentRunner

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a specified city."""
        weather_data = {
            "Tokyo": "Sunny, 22C",
            "London": "Cloudy, 15C",
            "New York": "Partly cloudy, 18C",
            "Paris": "Rainy, 12C",
        }
        return weather_data.get(city, f"Weather data not available for {city}")

    agent = create_deep_agent(
        model=f"openai:{ollama_model}",
        tools=[get_weather],
        system_prompt=(
            "You are a helpful weather assistant. "
            "When asked about weather, use the get_weather tool."
        ),
        name="e2e-deep-agent",
    )

    runner = DaprWorkflowDeepAgentRunner(
        agent=agent,
        name="e2e-deepagents-test",
        max_steps=50,
    )
    try:
        runner.start()
        result = runner.invoke(
            input={
                "messages": [
                    {"role": "user", "content": "What is the weather in Tokyo?"}
                ]
            },
            thread_id=f"e2e-deep-{uuid.uuid4().hex[:8]}",
            timeout=300,
        )

        assert result is not None, "invoke returned None — workflow never completed"
        assert "messages" in result, "expected 'messages' key in result"
    finally:
        runner.shutdown()
        clear_dapr_registration()


# ---------------------------------------------------------------------------
# Test B: Multi-tool workflow (from simple_agent.py 2-tool pattern)
# ---------------------------------------------------------------------------


@pytest.mark.ollama
@pytest.mark.integration
def test_deepagents_multi_tool(
    ollama_model: str,
) -> None:
    """Test that a Deep Agent with multiple tools completes a workflow.

    Validates multi-tool registration works end-to-end.
    """
    from langchain_core.tools import tool

    clear_dapr_registration()
    create_deep_agent = import_deepagents()

    from diagrid.agent.deepagents import DaprWorkflowDeepAgentRunner

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a specified city."""
        weather_data = {
            "Tokyo": "Sunny, 22C",
            "London": "Cloudy, 15C",
            "New York": "Partly cloudy, 18C",
            "Paris": "Rainy, 12C",
        }
        return weather_data.get(city, f"Weather data not available for {city}")

    @tool
    def search_web(query: str) -> str:
        """Search the web for information on a given topic."""
        return (
            f"Search results for '{query}': Found 10 relevant articles about {query}."
        )

    agent = create_deep_agent(
        model=f"openai:{ollama_model}",
        tools=[get_weather, search_web],
        system_prompt=(
            "You are an expert research assistant. Use the available "
            "tools when needed to answer user questions accurately."
        ),
        name="e2e-deep-agent-multi",
    )

    runner = DaprWorkflowDeepAgentRunner(
        agent=agent,
        name="e2e-deepagents-multi-tool",
        max_steps=50,
    )
    try:
        runner.start()
        result = runner.invoke(
            input={
                "messages": [
                    {"role": "user", "content": "What is the weather in London?"}
                ]
            },
            thread_id=f"e2e-deep-mt-{uuid.uuid4().hex[:8]}",
            timeout=300,
        )

        assert result is not None, "invoke returned None — workflow never completed"
        assert "messages" in result, "expected 'messages' key in result"
    finally:
        runner.shutdown()
        clear_dapr_registration()


# ---------------------------------------------------------------------------
# Test C: Deterministic retry (node fails transiently, Dapr retries)
# ---------------------------------------------------------------------------

# Module-level counters for retry test, protected by a lock.
_retry_lock = threading.Lock()
_retry_counts: dict[str, int] = {"node1": 0, "node2": 0, "node3": 0}


def _reset_retry_counts() -> None:
    with _retry_lock:
        for key in _retry_counts:
            _retry_counts[key] = 0


class _RetryState(TypedDict):
    messages: List[str]
    step: int


def _da_node_one(state: _RetryState) -> dict:
    """First node — always succeeds."""
    with _retry_lock:
        _retry_counts["node1"] += 1
    return {
        "messages": state["messages"] + ["Node 1 completed"],
        "step": state["step"] + 1,
    }


def _da_node_two(state: _RetryState) -> dict:
    """Second node — fails on first 2 attempts, succeeds on 3rd."""
    with _retry_lock:
        _retry_counts["node2"] += 1
        current = _retry_counts["node2"]
    if current <= 2:
        raise ConnectionError(f"Simulated failure (attempt {current})")
    return {
        "messages": state["messages"] + ["Node 2 completed after retries"],
        "step": state["step"] + 1,
    }


def _da_node_three(state: _RetryState) -> dict:
    """Third node — always succeeds."""
    with _retry_lock:
        _retry_counts["node3"] += 1
    return {
        "messages": state["messages"] + ["Node 3 completed"],
        "step": state["step"] + 1,
    }


@pytest.mark.integration
def test_deepagents_retry() -> None:
    """Test that Dapr retries a failing node activity in a Deep Agents runner.

    Builds a plain 3-node LangGraph and passes it to DaprWorkflowDeepAgentRunner.
    Node 2 raises ConnectionError on attempts 1-2, succeeds on attempt 3.
    Dapr's RetryPolicy handles the retry automatically.

    No LLM needed — purely deterministic.
    """
    from langgraph.graph import END, START, StateGraph

    from diagrid.agent.deepagents import DaprWorkflowDeepAgentRunner

    clear_dapr_registration()
    _reset_retry_counts()

    graph = StateGraph(_RetryState)
    graph.add_node("node_one", _da_node_one)
    graph.add_node("node_two", _da_node_two)
    graph.add_node("node_three", _da_node_three)
    graph.add_edge(START, "node_one")
    graph.add_edge("node_one", "node_two")
    graph.add_edge("node_two", "node_three")
    graph.add_edge("node_three", END)
    compiled = graph.compile()

    runner = DaprWorkflowDeepAgentRunner(
        agent=compiled,
        name="e2e-deepagents-retry-test",
        max_steps=50,
    )
    try:
        runner.start()
        result = runner.invoke(
            input={"messages": ["Start"], "step": 0},
            thread_id=f"e2e-deep-retry-{uuid.uuid4().hex[:8]}",
            timeout=60,
        )

        assert result is not None, "invoke returned None"
        assert result["step"] == 3, f"expected step=3, got {result['step']}"

        assert _retry_counts["node1"] == 1, (
            f"node_one should execute once, got {_retry_counts['node1']}"
        )
        assert _retry_counts["node2"] == 3, (
            f"node_two should execute 3 times (2 failures + 1 success), "
            f"got {_retry_counts['node2']}"
        )
        assert _retry_counts["node3"] == 1, (
            f"node_three should execute once, got {_retry_counts['node3']}"
        )
    finally:
        runner.shutdown()
        clear_dapr_registration()
