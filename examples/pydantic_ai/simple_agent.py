#!/usr/bin/env python3

# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""
Example: Simple Pydantic AI Agent with Dapr Workflows

This example demonstrates how to run a Pydantic AI agent with durable tool
execution using Dapr Workflows. Each tool call is executed as a workflow
activity, providing fault tolerance and durability.

Prerequisites:
    1. Dapr installed and initialized: dapr init
    2. Required packages: pip install diagrid pydantic-ai
    3. OPENAI_API_KEY environment variable set

Run with Dapr:
    dapr run --app-id pydantic-ai-agent --resources-path ./components -- python3 simple_agent.py
"""

import asyncio

from pydantic_ai import Agent
from diagrid.agent.pydantic_ai import DaprWorkflowAgentRunner


# Define tools as plain Python functions
def get_weather(city: str) -> str:
    """Get the current weather for a specified city."""
    weather_data = {
        "Tokyo": "Sunny, 22°C",
        "London": "Cloudy, 15°C",
        "New York": "Partly cloudy, 18°C",
        "Paris": "Rainy, 12°C",
    }
    return weather_data.get(city, f"Weather data not available for {city}")


def search_web(query: str) -> str:
    """Search the web for information on a given topic."""
    return f"Search results for '{query}': Found 10 relevant articles about {query}."


def get_datetime() -> str:
    """Get the current date and time."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def main():
    # Create a Pydantic AI agent with tools
    agent = Agent(
        "gpt-4o-mini",
        system_prompt="""You are an expert research assistant with access to various
        information sources. You excel at finding and synthesizing information
        to provide comprehensive answers to user queries.""",
        tools=[get_weather, search_web, get_datetime],
    )

    # Create the Dapr Workflow runner
    runner = DaprWorkflowAgentRunner(
        agent=agent,
        name="simple-agent",
        max_iterations=10,
    )

    try:
        # Start the workflow runtime
        print("Starting Dapr Workflow runtime...")
        runner.start()
        print("Runtime started successfully!")

        # Run the agent with a session ID
        session_id = "demo-session-001"
        print(f"\nExecuting agent with session: {session_id}")
        print("=" * 60)

        # Process events as they come
        async for event in runner.run_async(
            user_message="""Find out the current weather in Tokyo and search for
            recent news about AI developments. Provide a brief summary.""",
            session_id=session_id,
        ):
            event_type = event["type"]

            if event_type == "workflow_started":
                print(f"\nWorkflow started: {event.get('workflow_id')}")

            elif event_type == "workflow_status_changed":
                print(f"Status: {event.get('status')}")

            elif event_type == "workflow_completed":
                print("\n" + "=" * 60)
                print("AGENT COMPLETED")
                print("=" * 60)
                print(f"Iterations: {event.get('iterations')}")
                print(f"Status: {event.get('status')}")
                print("\nFinal Response:")
                print("-" * 40)
                print(event.get("final_response", "No response"))

            elif event_type == "workflow_failed":
                print(f"\nWorkflow FAILED: {event.get('error')}")

            elif event_type == "workflow_error":
                print(f"\nWorkflow ERROR: {event.get('error')}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Shutdown the runtime
        print("\nShutting down Dapr Workflow runtime...")
        runner.shutdown()
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
