"""E2E tests for OpenAI Agents SDK agent with Ollama backend via Dapr Workflows."""

import threading
import uuid

import pytest

from tests.e2e.conftest import clear_agent_registration

_OPENAI_WF = "diagrid.agent.openai_agents.workflow"


@pytest.mark.ollama
@pytest.mark.integration
def test_openai_agents_workflow_lifecycle(ollama_model: str) -> None:
    """Test that an OpenAI Agents SDK agent completes a full workflow cycle.

    Creates an agent with a deterministic tool, runs it through the
    Dapr Workflow runner, and asserts on workflow lifecycle (not LLM content).
    """
    from agents import Agent, function_tool

    from diagrid.agent.openai_agents import DaprWorkflowAgentRunner

    @function_tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"The weather in {city} is sunny and 22 degrees Celsius"

    agent = Agent(
        name="weather_agent",
        instructions=(
            "You are a helpful weather assistant. "
            "When asked about weather, use the get_weather tool."
        ),
        model=ollama_model,
        tools=[get_weather],
    )

    runner = DaprWorkflowAgentRunner(
        agent=agent,
        name="e2e-openai-agents-test",
    )
    try:
        runner.start()
        result = runner.run_sync(
            user_message="What is the weather in Tokyo?",
            session_id=f"e2e-openai-{uuid.uuid4().hex[:8]}",
            timeout=120,
        )

        assert result is not None, "run_sync returned None — workflow never completed"
        assert result.status == "completed", f"unexpected status: {result.status}"
        assert result.iterations >= 1, (
            f"expected >=1 iteration, got {result.iterations}"
        )
        assert result.final_response is not None, "final_response was None"
    finally:
        runner.shutdown()


# ---------------------------------------------------------------------------
# Test B: Multi-tool workflow (from simple_agent.py 3-tool pattern)
# ---------------------------------------------------------------------------


@pytest.mark.ollama
@pytest.mark.integration
def test_openai_agents_multi_tool(ollama_model: str) -> None:
    """Test that an OpenAI Agents agent with 3 tools completes a workflow.

    Validates multi-tool registration works end-to-end, matching
    the simple_agent.py example pattern.
    """
    clear_agent_registration(_OPENAI_WF)

    from agents import Agent, function_tool

    from diagrid.agent.openai_agents import DaprWorkflowAgentRunner

    @function_tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"The weather in {city} is sunny and 22 degrees Celsius"

    @function_tool
    def search_web(query: str) -> str:
        """Search the web for information."""
        return f"Search results for '{query}': Found 10 relevant documents."

    @function_tool
    def get_datetime() -> str:
        """Get the current date and time."""
        return "2026-03-17T12:00:00Z"

    agent = Agent(
        name="research_assistant",
        instructions=(
            "You are an expert research assistant with access to weather, "
            "web search, and datetime tools."
        ),
        model=ollama_model,
        tools=[get_weather, search_web, get_datetime],
    )

    runner = DaprWorkflowAgentRunner(
        agent=agent,
        name="e2e-openai-agents-multi-tool",
        max_iterations=10,
    )
    try:
        runner.start()
        result = runner.run_sync(
            user_message="What is the weather in Tokyo?",
            session_id=f"e2e-openai-mt-{uuid.uuid4().hex[:8]}",
            timeout=120,
        )

        assert result is not None, "run_sync returned None"
        assert result.status == "completed", f"unexpected status: {result.status}"
        assert result.iterations >= 1
        assert result.final_response is not None
    finally:
        runner.shutdown()
        clear_agent_registration(_OPENAI_WF)


# ---------------------------------------------------------------------------
# Test C: LLM-driven retry (tool fails transiently, LLM retries)
# ---------------------------------------------------------------------------

# Module-level counters for retry test, protected by a lock.
_retry_lock = threading.Lock()
_retry_counts: dict[str, int] = {"tool1": 0, "tool2": 0, "tool3": 0}


def _reset_retry_counts() -> None:
    with _retry_lock:
        for key in _retry_counts:
            _retry_counts[key] = 0


@pytest.mark.ollama
@pytest.mark.integration
@pytest.mark.xfail(
    reason="Small models may not reliably retry failed tools",
    strict=False,
)
def test_openai_agents_retry(ollama_model: str) -> None:
    """Test that an OpenAI Agents workflow recovers when a tool fails transiently.

    Creates an agent with 3 tools. Tool 2 fails on the first 2 invocations,
    then succeeds. The LLM should see the error and retry the tool call.
    """
    clear_agent_registration(_OPENAI_WF)

    from agents import Agent, function_tool

    from diagrid.agent.openai_agents import DaprWorkflowAgentRunner

    _reset_retry_counts()

    @function_tool
    def tool_step_one(input_data: str) -> str:
        """Initialize data for the workflow. First step."""
        with _retry_lock:
            _retry_counts["tool1"] += 1
        return f"Step 1 done: initialized with '{input_data}'"

    @function_tool
    def tool_step_two(data: str) -> str:
        """Process data from step one. Second step."""
        with _retry_lock:
            _retry_counts["tool2"] += 1
            current = _retry_counts["tool2"]
        if current <= 2:
            raise ConnectionError(f"Simulated transient failure (attempt {current})")
        return "Step 2 done: data processed successfully"

    @function_tool
    def tool_step_three(data: str) -> str:
        """Finalize and return results. Third and final step."""
        with _retry_lock:
            _retry_counts["tool3"] += 1
        return "Step 3 done: all steps completed"

    agent = Agent(
        name="retry_agent",
        instructions=(
            "You are a sequential task processor. You MUST call all three tools "
            "in order:\n"
            "1. First call tool_step_one with 'test_data'\n"
            "2. Then call tool_step_two with the result\n"
            "3. Finally call tool_step_three with the result\n\n"
            "If a tool returns an error, try calling it again with the same input. "
            "Do NOT skip any steps."
        ),
        model=ollama_model,
        tools=[tool_step_one, tool_step_two, tool_step_three],
    )

    runner = DaprWorkflowAgentRunner(
        agent=agent,
        name="e2e-openai-agents-retry",
        max_iterations=10,
    )
    try:
        runner.start()
        result = runner.run_sync(
            user_message=(
                "Process 'test_data' through all three steps in order: "
                "tool_step_one, then tool_step_two, then tool_step_three. "
                "If any tool fails, retry it."
            ),
            session_id=f"e2e-openai-retry-{uuid.uuid4().hex[:8]}",
            timeout=180,
        )

        assert result is not None, "run_sync returned None"
        assert result.status == "completed", f"unexpected status: {result.status}"
        assert _retry_counts["tool1"] >= 1, (
            f"tool_step_one not called (count={_retry_counts['tool1']})"
        )
        assert _retry_counts["tool2"] > 1, (
            f"tool_step_two was not retried (count={_retry_counts['tool2']}), "
            "expected >1 calls due to transient failures"
        )
        assert _retry_counts["tool3"] >= 1, (
            f"tool_step_three not called (count={_retry_counts['tool3']}), "
            "workflow did not reach the final step"
        )
    finally:
        runner.shutdown()
        clear_agent_registration(_OPENAI_WF)
