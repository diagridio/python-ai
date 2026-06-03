#!/usr/bin/env python3

# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""
Example: LangChain Deep Agent with Dapr Workflows

This example demonstrates how to run a LangChain Deep Agent with durable
execution using Dapr Workflows. Each node in the Deep Agent's graph runs
as a workflow activity, providing fault tolerance and durability.

Deep Agents compile to standard LangGraph CompiledStateGraphs, so the
existing Dapr/LangGraph integration handles execution transparently.

Prerequisites:
    1. Dapr installed and initialized: dapr init
    2. Redis running: docker run -d --name redis -p 6379:6379 redis:latest
    3. Required packages: pip install diagrid deepagents langchain-openai
    4. OPENAI_API_KEY environment variable set

Run with Dapr:
    dapr run --app-id deep-agent --resources-path ./components -- python3 simple_agent.py
"""

import asyncio

from deepagents import create_deep_agent
from langchain_core.tools import tool

from diagrid.agent.deepagents import DaprWorkflowDeepAgentRunner


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a specified city."""
    weather_data = {
        "Tokyo": "Sunny, 22°C",
        "London": "Cloudy, 15°C",
        "New York": "Partly cloudy, 18°C",
        "Paris": "Rainy, 12°C",
    }
    return weather_data.get(city, f"Weather data not available for {city}")


@tool
def search_web(query: str) -> str:
    """Search the web for information on a given topic."""
    return f"Search results for '{query}': Found 10 relevant articles about {query}."


async def main():
    # Create a Deep Agent — returns a compiled LangGraph
    agent = create_deep_agent(
        model="openai:gpt-4o-mini",
        tools=[get_weather, search_web],
        system_prompt="""You are an expert research assistant. Use the available
        tools when needed to answer user questions accurately.""",
        name="research-assistant",
    )

    # Wrap it with the Dapr Workflow runner for durable execution
    runner = DaprWorkflowDeepAgentRunner(
        agent=agent,
        name="deep-agent",
        max_steps=50,
    )

    try:
        print("Starting Dapr Workflow runtime...")
        runner.start()
        print("Runtime started successfully!")

        thread_id = "demo-thread-001"
        print(f"\nExecuting Deep Agent with thread: {thread_id}")
        print("=" * 60)

        async for event in runner.run_async(
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": "What's the weather in Tokyo? Also search for recent AI news.",
                    }
                ]
            },
            thread_id=thread_id,
        ):
            event_type = event["type"]

            if event_type == "workflow_started":
                print(f"\nWorkflow started: {event.get('workflow_id')}")

            elif event_type == "workflow_status_changed":
                print(f"Status: {event.get('status')}")

            elif event_type == "workflow_completed":
                print("\n" + "=" * 60)
                print("DEEP AGENT COMPLETED")
                print("=" * 60)
                output = event.get("output", {})
                messages = output.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    # Handle both dict and object forms
                    content = (
                        last_msg.get("content", "")
                        if isinstance(last_msg, dict)
                        else getattr(last_msg, "content", str(last_msg))
                    )
                    print(f"\nFinal Response:\n{content}")
                print(f"\nSteps: {event.get('steps', 'N/A')}")

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
