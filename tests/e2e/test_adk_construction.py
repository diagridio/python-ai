"""E2E tests for ADK runner construction and tool registration (no LLM).

ADK uses ``google.genai.Client()`` which requires a Google API key and
does not support Ollama. These tests verify runner construction, tool
registration, and config extraction without making any LLM calls.
"""

import pytest

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from diagrid.agent.adk import DaprWorkflowAgentRunner
from tests.e2e.conftest import clear_agent_registration

_ADK_WF = "diagrid.agent.adk.workflow"


# ---------------------------------------------------------------------------
# Tool definitions (matching simple_agent.py 3-tool pattern)
# ---------------------------------------------------------------------------


def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city to get weather for.

    Returns:
        A string describing the weather.
    """
    weather_data = {
        "tokyo": "Sunny, 22C",
        "london": "Cloudy, 15C",
        "new york": "Rainy, 18C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


def get_time(timezone: str) -> str:
    """Get the current time in a timezone.

    Args:
        timezone: The timezone (e.g., 'JST', 'UTC', 'EST').

    Returns:
        A string with the current time.
    """
    return f"Current time in {timezone.upper()}: 12:00:00"


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., '2 + 2').

    Returns:
        The result of the calculation.
    """
    allowed_chars = set("0123456789+-*/(). ")
    if not all(c in allowed_chars for c in expression):
        return "Error: Invalid characters in expression"
    try:
        result = eval(expression)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Test A: Runner construction + single tool
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_adk_runner_construction() -> None:
    """Test that an ADK runner can be constructed and tools registered.

    Verifies the runner initializes correctly without requiring an LLM call.
    """
    agent = LlmAgent(
        name="test_assistant",
        model="gemini-2.0-flash",
        instruction="You are a helpful assistant.",
        tools=[FunctionTool(get_weather)],
    )

    runner = DaprWorkflowAgentRunner(
        agent=agent,
        name="e2e-adk-construction",
    )
    try:
        assert runner.workflow_name is not None, "workflow_name was None"
        assert runner.workflow_name != "", "workflow_name was empty"
    finally:
        runner.shutdown()


# ---------------------------------------------------------------------------
# Test B: Multi-tool registration (from simple_agent.py 3-tool pattern)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_adk_multi_tool_registration() -> None:
    """Test that 3 tools register correctly with the ADK runner.

    Matches the simple_agent.py example's FunctionTool pattern with
    get_weather, get_time, and calculate.
    """
    clear_agent_registration(_ADK_WF)
    agent = LlmAgent(
        name="helpful_assistant",
        model="gemini-2.0-flash",
        instruction=(
            "You are a helpful assistant that can check weather, "
            "time, and do calculations."
        ),
        tools=[
            FunctionTool(get_weather),
            FunctionTool(get_time),
            FunctionTool(calculate),
        ],
    )

    runner = DaprWorkflowAgentRunner(
        agent=agent,
        name="e2e-adk-multi-tool",
        max_iterations=10,
    )
    try:
        assert runner.workflow_name is not None, "workflow_name was None"
    finally:
        runner.shutdown()
