# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""
Simple Google ADK Agent with Dapr Workflow Example

This example demonstrates how to run a Google ADK agent with durable execution
using Dapr Workflows. Each tool execution becomes a separate Dapr activity.

Prerequisites:
    - Redis running on localhost:6379
    - Google API key set in environment: export GOOGLE_API_KEY=your-key
    - Install the extension: pip install diagrid google-adk

Run with:
    cd diagrid/agent/adk/examples
    dapr run --app-id adk-agent --resources-path ./components -- python3 simple_agent.py
"""

import asyncio

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from diagrid.agent.adk import DaprWorkflowAgentRunner


# Define some simple tools
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city to get weather for.

    Returns:
        A string describing the weather.
    """
    # Mock weather data
    weather_data = {
        "tokyo": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "new york": "Rainy, 18°C",
        "paris": "Partly cloudy, 20°C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


def get_time(timezone: str) -> str:
    """Get the current time in a timezone.

    Args:
        timezone: The timezone (e.g., 'JST', 'UTC', 'EST').

    Returns:
        A string with the current time.
    """
    from datetime import datetime, timedelta

    # Mock timezone offsets
    offsets = {
        "jst": 9,
        "utc": 0,
        "est": -5,
        "pst": -8,
        "cet": 1,
    }
    offset = offsets.get(timezone.lower(), 0)
    time = datetime.utcnow() + timedelta(hours=offset)
    return f"Current time in {timezone.upper()}: {time.strftime('%H:%M:%S')}"


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., '2 + 2').

    Returns:
        The result of the calculation.
    """
    try:
        # Only allow safe mathematical operations
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


async def main():
    print("=" * 60, flush=True)
    print("Google ADK Agent with Dapr Workflow - Durable Execution", flush=True)
    print("=" * 60, flush=True)

    # Create the ADK agent with tools
    agent = LlmAgent(
        name="helpful_assistant",
        model="gemini-2.0-flash",
        instruction="You are a helpful assistant that can check weather, time, and do calculations. Be concise in your responses.",
        tools=[
            FunctionTool(get_weather),
            FunctionTool(get_time),
            FunctionTool(calculate),
        ],
    )

    print(f"\nAgent: {agent.name}", flush=True)
    print("Model: gemini-2.0-flash", flush=True)
    print("Tools: get_weather, get_time, calculate", flush=True)
    print("-" * 60, flush=True)

    # Create the Dapr Workflow runner
    runner = DaprWorkflowAgentRunner(
        agent=agent,
        max_iterations=10,
    )

    # Start the workflow runtime
    print("\nStarting Dapr Workflow runtime...", flush=True)
    runner.start()
    print("Runtime started!", flush=True)

    try:
        # Example conversation
        user_message = "What's the weather in Tokyo and what time is it there (JST)?"
        print(f"\nUser: {user_message}", flush=True)
        print("-" * 60, flush=True)

        # Run the agent
        print("\nRunning agent (each tool call is a durable activity)...\n", flush=True)

        async for event in runner.run_async(
            user_message=user_message,
            session_id="example-session-001",
            user_id="example-user",
            app_name="adk-example",
        ):
            event_type = event["type"]

            if event_type == "workflow_started":
                print(f"[Workflow] Started: {event.get('workflow_id')}", flush=True)

            elif event_type == "workflow_status_changed":
                print(f"[Workflow] Status: {event.get('status')}", flush=True)

            elif event_type == "workflow_completed":
                print("\n[Workflow] Completed!", flush=True)
                print(f"  Iterations: {event.get('iterations')}", flush=True)
                print(f"  Status: {event.get('status')}", flush=True)
                if event.get("final_response"):
                    print(f"\nAssistant: {event.get('final_response')}", flush=True)

            elif event_type == "workflow_failed":
                error = event.get("error")
                print("\n[Workflow] Failed!", flush=True)
                if isinstance(error, dict):
                    print(f"  Error Type: {error.get('error_type')}", flush=True)
                    print(f"  Message: {error.get('message')}", flush=True)
                    if error.get("stack_trace"):
                        print(f"  Stack Trace:\n{error.get('stack_trace')}", flush=True)
                else:
                    print(f"  Error: {error}", flush=True)

            elif event_type == "workflow_error":
                print(f"\n[Workflow] Error: {event.get('error')}", flush=True)

        print("\n" + "=" * 60, flush=True)
        print("Workflow execution complete!", flush=True)
        print("=" * 60, flush=True)

    finally:
        # Shutdown the runtime
        print("\nShutting down workflow runtime...", flush=True)
        runner.shutdown()
        print("Done!", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
