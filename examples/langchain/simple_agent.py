#!/usr/bin/env python3

# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""
Example: Simple LangChain Agent with Dapr Workflows

This example demonstrates how to run a LangChain model + tools agent with
durable execution using Dapr Workflows. Each chat-model call and each tool
execution becomes a separate Dapr Workflow activity, providing fault
tolerance and durability.

Prerequisites:
    1. Dapr installed and initialized: dapr init
    2. Required packages: uv sync --all-packages --extra langchain
    3. OPENAI_API_KEY environment variable set

Run with Dapr:
    dapr run --app-id langchain-agent --resources-path ./resources -- python3 simple_agent.py
"""

import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from diagrid.agent.langchain import DaprWorkflowAgentRunner


@tool
def search_web(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query
    """
    return f"Search results for '{query}': Found 10 relevant documents about {query}."


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: The math expression to evaluate
    """
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)  # noqa: S307
        return str(float(result))
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name
    """
    return f"Weather in {city}: Sunny, 72F (22C), humidity 45%"


async def main():
    """Run the LangChain agent as a Dapr workflow."""

    # Create the Dapr Workflow runner directly around plain langchain_core
    # building blocks - a chat model and a list of tools.
    runner = DaprWorkflowAgentRunner(
        model=ChatOpenAI(model="gpt-4o-mini"),
        tools=[search_web, calculate, get_weather],
        name="simple-agent",
        system_prompt=(
            "You are a helpful assistant with access to web search, "
            "calculations, and weather information. Use your tools when "
            "appropriate to answer user questions accurately."
        ),
    )

    try:
        print("Starting Dapr Workflow runtime...")
        runner.start()
        print("Runtime started successfully!")

        session_id = "demo-session-001"
        print(f"\nExecuting agent with session: {session_id}")
        print("=" * 60)

        async for event in runner.run_async(
            task="What's 2 + 2, and what's the weather in San Francisco?",
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
                print(f"\nResponse: {event.get('result', 'No response')}")
                print(f"\nIterations: {event.get('iterations')}")

            elif event_type == "workflow_failed":
                print(f"\nWorkflow FAILED: {event.get('error')}")

            elif event_type == "workflow_error":
                print(f"\nWorkflow ERROR: {event.get('error')}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        print("\nShutting down Dapr Workflow runtime...")
        runner.shutdown()
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
