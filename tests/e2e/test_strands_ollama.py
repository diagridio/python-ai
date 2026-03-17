"""E2E tests for Strands agent with Ollama backend via Dapr Workflows."""

import threading
import uuid

import pytest


# ---------------------------------------------------------------------------
# Test A: Workflow lifecycle (from simple_agent.py)
# ---------------------------------------------------------------------------


@pytest.mark.ollama
@pytest.mark.integration
def test_strands_workflow_lifecycle(
    ollama_model: str,
    ollama_endpoint: str,
) -> None:
    """Test that a Strands agent completes a full workflow cycle.

    Creates an agent with a deterministic tool, runs it through the
    Dapr Workflow runner, and asserts on workflow lifecycle.
    """
    from strands import Agent, tool
    from strands.models.openai import OpenAIModel

    from diagrid.agent.strands import DaprWorkflowAgentRunner

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city.

        Args:
            city: The city name

        Returns:
            Weather information
        """
        return f"Weather in {city}: Sunny, 72F (22C), humidity 45%"

    agent = Agent(
        model=OpenAIModel(
            model_id=ollama_model,
            client_args={"base_url": ollama_endpoint, "api_key": "ollama"},
        ),
        tools=[get_weather],
        system_prompt=(
            "You are a helpful weather assistant. "
            "When asked about weather, use the get_weather tool."
        ),
    )

    runner = DaprWorkflowAgentRunner(
        agent=agent,
        name="e2e-strands-test",
    )
    try:
        runner.start()
        result = runner.run_sync(
            task="What is the weather in Tokyo?",
            session_id=f"e2e-strands-{uuid.uuid4().hex[:8]}",
            timeout=120,
        )

        assert result is not None, "run_sync returned None — workflow never completed"
        assert result.result is not None, "result was None"
    finally:
        runner.shutdown()


# ---------------------------------------------------------------------------
# Test B: Multi-tool workflow (from simple_agent.py 3-tool pattern)
# ---------------------------------------------------------------------------


@pytest.mark.ollama
@pytest.mark.integration
def test_strands_multi_tool(
    ollama_model: str,
    ollama_endpoint: str,
) -> None:
    """Test that a Strands agent with 3 tools completes a workflow.

    Validates multi-tool registration works end-to-end, matching
    the simple_agent.py example pattern.
    """
    from strands import Agent, tool
    from strands.models.openai import OpenAIModel

    from diagrid.agent.strands import DaprWorkflowAgentRunner

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city.

        Args:
            city: The city name

        Returns:
            Weather information
        """
        return f"Weather in {city}: Sunny, 72F (22C), humidity 45%"

    @tool
    def search_web(query: str) -> str:
        """Search the web for information.

        Args:
            query: The search query

        Returns:
            Search results
        """
        return (
            f"Search results for '{query}': Found 10 relevant documents about {query}."
        )

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression.

        Args:
            expression: The math expression to evaluate

        Returns:
            The result of the calculation
        """
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        try:
            result = eval(expression)  # noqa: S307
            return str(float(result))
        except Exception as e:
            return f"Error: {e}"

    agent = Agent(
        model=OpenAIModel(
            model_id=ollama_model,
            client_args={"base_url": ollama_endpoint, "api_key": "ollama"},
        ),
        tools=[get_weather, search_web, calculate],
        system_prompt=(
            "You are a helpful assistant with access to web search, "
            "calculations, and weather information. Use your tools when "
            "appropriate to answer user questions accurately."
        ),
    )

    runner = DaprWorkflowAgentRunner(
        agent=agent,
        name="e2e-strands-multi-tool",
    )
    try:
        runner.start()
        result = runner.run_sync(
            task="What is the weather in San Francisco?",
            session_id=f"e2e-strands-mt-{uuid.uuid4().hex[:8]}",
            timeout=120,
        )

        assert result is not None, "run_sync returned None — workflow never completed"
        assert result.result is not None, "result was None"
    finally:
        runner.shutdown()


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
def test_strands_retry(
    ollama_model: str,
    ollama_endpoint: str,
) -> None:
    """Test that a Strands workflow recovers when a tool fails transiently.

    Creates an agent with 3 tools. Tool 2 fails on the first 2 invocations,
    then succeeds. The LLM should see the error and retry the tool call.
    """
    from strands import Agent, tool
    from strands.models.openai import OpenAIModel

    from diagrid.agent.strands import DaprWorkflowAgentRunner

    _reset_retry_counts()

    @tool
    def tool_step_one(input_data: str) -> str:
        """Initialize data for the workflow. First step.

        Args:
            input_data: The data to initialize

        Returns:
            Confirmation of initialization
        """
        with _retry_lock:
            _retry_counts["tool1"] += 1
        return f"Step 1 done: initialized with '{input_data}'"

    @tool
    def tool_step_two(data: str) -> str:
        """Process data from step one. Second step.

        Args:
            data: The data to process

        Returns:
            Confirmation of processing
        """
        with _retry_lock:
            _retry_counts["tool2"] += 1
            current = _retry_counts["tool2"]
        if current <= 2:
            raise ConnectionError(f"Simulated transient failure (attempt {current})")
        return "Step 2 done: data processed successfully"

    @tool
    def tool_step_three(data: str) -> str:
        """Finalize and return results. Third and final step.

        Args:
            data: The data to finalize

        Returns:
            Confirmation of completion
        """
        with _retry_lock:
            _retry_counts["tool3"] += 1
        return "Step 3 done: all steps completed"

    agent = Agent(
        model=OpenAIModel(
            model_id=ollama_model,
            client_args={"base_url": ollama_endpoint, "api_key": "ollama"},
        ),
        tools=[tool_step_one, tool_step_two, tool_step_three],
        system_prompt=(
            "You are a sequential task processor. You MUST call all three tools "
            "in order:\n"
            "1. First call tool_step_one with 'test_data'\n"
            "2. Then call tool_step_two with the result\n"
            "3. Finally call tool_step_three with the result\n\n"
            "If a tool returns an error, try calling it again with the same input. "
            "Do NOT skip any steps."
        ),
    )

    runner = DaprWorkflowAgentRunner(
        agent=agent,
        name="e2e-strands-retry",
    )
    try:
        runner.start()
        result = runner.run_sync(
            task=(
                "Process 'test_data' through all three steps in order: "
                "tool_step_one, then tool_step_two, then tool_step_three. "
                "If any tool fails, retry it."
            ),
            session_id=f"e2e-strands-retry-{uuid.uuid4().hex[:8]}",
            timeout=180,
        )

        assert result is not None, "run_sync returned None"
        assert result.result is not None, "result was None"
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
